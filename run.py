"""
run.py
======
Main entry point for StressGait-MM experiments.

Usage examples
--------------
# Quick sanity-check (1 run, 5 epochs):
    python run.py --mode quick

# Single-seed first pass (6 configs × seed 42):
    python run.py --mode full --single_seed 42

# Full 3-seed grid (18 configs):
    python run.py --mode full

# Ablations (single-seed):
    python run.py --mode full --run_ablations

# Extended protocols (P3, P4, P5):
    python run.py --mode full --run_extended

# Generate all figures + tables from finished runs:
    python run.py --mode full --analysis_only

Configuration
-------------
Edit ``config.py`` (``Config`` dataclass) to set your data paths and
hyperparameters before running.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

from config import Config
from data import (
    SkeletonDataManager,
    SubjectIDMap,
    VibrationCache,
    build_subject_map,
)
from experiments.ablations import run_ablations
from experiments.grid import METHODS_FOR_TABLE, build_configs, run_grid
from analysis.figures import (
    make_phase_attn_figure,
    make_speed_figure,
    visualize_data_pipeline,
)
from analysis.tables import METHODS_FOR_TABLE, make_table1, threshold_sensitivity_sweep
from evaluation.protocols import (
    loso_folds_p5,
    run_with_splits,
    split_cross_view_p3,
    split_cvxc_p4,
)
from training.pretrain import get_or_create_ssl_state

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="StressGait-MM experiment runner")
    p.add_argument("--mode", default="quick",
                   choices=["quick", "lite", "full"],
                   help="Experiment mode (default: quick)")
    p.add_argument("--single_seed", type=int, default=None,
                   help="Run only this seed (useful for first-pass with full mode)")
    p.add_argument("--run_ablations", action="store_true",
                   help="Run ablation experiments after the main grid")
    p.add_argument("--run_extended", action="store_true",
                   help="Run extended evaluation protocols (P3, P4, P5)")
    p.add_argument("--analysis_only", action="store_true",
                   help="Skip training; regenerate tables and figures only")
    p.add_argument("--time_budget_hr", type=float, default=8.5,
                   help="Session time budget in hours (default: 8.5)")
    p.add_argument("--output_root", default=None,
                   help="Override Config.output_root")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.output_root:
        cfg = Config(**{**vars(cfg), "output_root": args.output_root})
    work = Path(cfg.output_root)
    for d in ("artifacts", "runs", "figures", "tables"):
        (work / d).mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # ------------------------------------------------------------------
    # Phase 0 — subject map
    # ------------------------------------------------------------------
    print("\n=== Phase 0: subject map ===")
    subject_map = build_subject_map(cfg, verbose=True)
    map_path = work / "artifacts" / "subject_map.json"
    with open(map_path, "w") as f:
        json.dump(subject_map, f, indent=2, sort_keys=True)
    print(f"[Phase 0] saved → {map_path}")

    # ------------------------------------------------------------------
    # Phase 1 — load data + vibration cache
    # ------------------------------------------------------------------
    print("\n=== Phase 1: load skeleton + vibration ===")
    manager = SkeletonDataManager(cfg)
    manager.load()
    all_samples = list(manager.samples)
    subject_ids = SubjectIDMap([s[2] for s in all_samples])
    vib_cache = VibrationCache(cfg, subject_map=subject_map)

    # ------------------------------------------------------------------
    # Phase 2 — SSL pre-training (cached after first run)
    # ------------------------------------------------------------------
    print("\n=== Phase 2: SSL pre-training ===")
    pretrained_gcn = get_or_create_ssl_state(cfg, all_samples, epochs=15)

    if args.analysis_only:
        print("\n[analysis_only] skipping training — regenerating outputs …")
    else:
        # ------------------------------------------------------------------
        # Phase 3 — main experiment grid
        # ------------------------------------------------------------------
        print(f"\n=== Phase 3: experiment grid ({args.mode}) ===")
        run_grid(
            cfg,
            mode=args.mode,
            vib_cache=vib_cache,
            subject_ids=subject_ids,
            pretrained_gcn=pretrained_gcn,
            time_budget_hr=args.time_budget_hr,
            single_seed=args.single_seed,
        )

        # ------------------------------------------------------------------
        # Phase 4 — ablations (optional)
        # ------------------------------------------------------------------
        if args.run_ablations:
            print("\n=== Phase 4: ablation experiments ===")
            run_ablations(
                cfg, vib_cache, subject_ids, pretrained_gcn,
                seeds=(42,) if args.single_seed == 42 else (42, 1337, 2024),
                time_budget_hr=args.time_budget_hr,
            )

        # ------------------------------------------------------------------
        # Phase 5 — extended protocols P3 / P4 / P5 (optional)
        # ------------------------------------------------------------------
        if args.run_extended:
            _run_extended_protocols(cfg, all_samples, vib_cache, subject_ids,
                                    pretrained_gcn, args.time_budget_hr)

    # ------------------------------------------------------------------
    # Analysis — tables and figures
    # ------------------------------------------------------------------
    print("\n=== Analysis: tables and figures ===")
    methods = METHODS_FOR_TABLE[args.mode]
    make_table1(work / "runs", work / "tables", methods)
    threshold_sensitivity_sweep(cfg, thresholds=(0.3, 0.5, 0.7),
                                out_root=work / "tables")
    make_phase_attn_figure(work / "runs", cfg, work / "figures", subject_map)
    make_speed_figure(work / "runs", work / "figures", methods)
    visualize_data_pipeline(cfg, out_root=work / "figures")
    print("\nDone.")


def _run_extended_protocols(cfg, all_samples, vib_cache, subject_ids,
                             pretrained_gcn, time_budget_hr):
    import time
    from dataclasses import replace as _dc_replace
    session_start = time.time()
    tables_dir = Path(cfg.output_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ext_rows = []
    all_views = list(cfg.views)

    def _time_ok(margin=1.0):
        return (time.time() - session_start) / 3600 < (time_budget_hr - margin)

    def _base(**over):
        d = vars(cfg); d.update(over)
        d["vib_sensors"] = tuple(d["vib_sensors"])
        d.update({"fusion": "gpc", "modalities": "both", "epochs": 15, "warmup_epochs": 3})
        return Config(**d)

    # P3 — cross-view
    print("\n=== P3: Cross-View Transfer ===")
    for v in all_views:
        if not _time_ok(): break
        tr, dv, te = split_cross_view_p3(all_samples, train_view=v, seed=42)
        r = run_with_splits(_base(seed=42), tr, dv, te, vib_cache, subject_ids,
                            pretrained_gcn, "P3", f"p3_trainview-{v}_s42",
                            extra_meta={"train_view": v})
        ext_rows.append({"protocol": "P3", "train_view": v, **(r.get("meta") or {})})

    # P4 — cross-view × cross-condition
    print("\n=== P4: Cross-View x Cross-Condition ===")
    for i in range(len(all_views)):
        for j in range(len(all_views)):
            if i == j: continue
            test_vs = [v for k, v in enumerate(all_views) if k not in (i, j)]
            if not test_vs or not _time_ok(): continue
            nv, sv, tv = all_views[i], all_views[j], test_vs[0]
            tr, dv, te = split_cvxc_p4(all_samples, nv, sv, tv, seed=42)
            r = run_with_splits(_base(seed=42), tr, dv, te, vib_cache, subject_ids,
                                pretrained_gcn, "P4", f"p4_N-{nv}_S-{sv}_T-{tv}",
                                extra_meta={"normal_view": nv, "stress_view": sv, "test_view": tv})
            ext_rows.append({"protocol": "P4", "normal_view": nv, "stress_view": sv,
                             "test_view": tv, **(r.get("meta") or {})})

    # P5 — LOSO
    print("\n=== P5: Leave-One-Subject-Out ===")
    import numpy as np
    folds = loso_folds_p5(all_samples, max_folds=6, seed=42)
    loso_aucs = []
    for tr, dv, te, held in folds:
        if not _time_ok(): break
        r = run_with_splits(_base(seed=42), tr, dv, te, vib_cache, subject_ids,
                            pretrained_gcn, "P5", f"p5_loso_hold-{held}",
                            extra_meta={"held_subject": held})
        if np.isfinite(r.get("test_auc", float("nan"))):
            loso_aucs.append(r["test_auc"])
        ext_rows.append({"protocol": "P5", "held_subject": held, **(r.get("meta") or {})})
    if loso_aucs:
        print(f"\n[P5] LOSO AUC: {np.mean(loso_aucs):.4f} ± {np.std(loso_aucs):.4f} "
              f"over {len(loso_aucs)} folds")

    if ext_rows:
        import pandas as pd
        pd.DataFrame(ext_rows).to_csv(tables_dir / "extended_protocols.csv", index=False)
        print(f"\n[extended] → {tables_dir / 'extended_protocols.csv'}")


if __name__ == "__main__":
    main()

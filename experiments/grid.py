"""
experiments/grid.py
===================
Resumable experiment grid runner.

``build_configs()`` generates the full list of :class:`~config.Config`
objects for a given ``EXPERIMENT_MODE``.  ``run_experiment()`` trains
and evaluates one configuration, persisting results to disk; re-runs
are instant thanks to the ``skip_if_done`` cache.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import Config
from data.dataset import SubjectIDMap
from data.skeleton_manager import SkeletonDataManager, split_subject_3way
from data.vibration_cache import VibrationCache
from evaluation.metrics import subject_level_aucs
from training.trainer import train_one

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def build_configs(base: Config, mode: str) -> List[Config]:
    """Return the list of experiment configurations for ``mode``.

    Parameters
    ----------
    base:
        Template configuration; grid entries override selected fields.
    mode:
        ``'quick'`` (1 run, 5 epochs) |
        ``'lite'``  (1 method × 2 seeds) |
        ``'full'``  (6 methods × 3 seeds = 18 runs).
    """
    def _c(**over):
        d = asdict(base)
        d.update(over)
        d["vib_sensors"] = tuple(d["vib_sensors"])
        return Config(**d)

    if mode == "quick":
        return [_c(fusion="gpc", modalities="both", seed=42, epochs=5,
                   run_tag="quick_gpc_s42")]

    if mode == "lite":
        rows = []
        for seed in (42, 1337):
            rows.append(_c(fusion="gpc", modalities="both", seed=seed,
                           epochs=20, run_tag=f"lite_gpc_s{seed}"))
        return rows

    if mode == "full":
        combos = [
            ("concat", "skel",  "skel"),
            ("concat", "vib",   "vib"),
            ("late",   "both",  "late"),
            ("concat", "both",  "concat"),
            ("gated",  "both",  "gated"),
            ("gpc",    "both",  "gpc"),
        ]
        rows = []
        for fus, mod, name in combos:
            for seed in (42, 1337, 2024):
                rows.append(_c(fusion=fus, modalities=mod, seed=seed,
                               epochs=30, run_tag=f"t1_{name}_s{seed}"))
        return rows

    raise ValueError(f"Unknown experiment mode: {mode!r}")


def _preds_to_df(preds: dict, extra: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "sample_id": preds["sample_id"],
        "subject":   preds["subject"],
        "view":      preds["view"],
        "cond":      preds["cond"],
        "speed":     preds["speed"],
        "y_true":    preds["y_true"].astype(np.int8),
        "y_pred":    preds["y_pred"].astype(np.float32),
    })
    for k, v in extra.items():
        df[k] = v
    return df


def run_experiment(
    cfg: Config,
    vib_cache: Optional[VibrationCache],
    subject_ids: SubjectIDMap,
    pretrained_gcn: Optional[dict] = None,
    skip_if_done: bool = True,
) -> dict:
    """Train and evaluate one experiment configuration.

    Re-runs are skipped automatically when ``skip_if_done=True`` and a
    valid ``preds_test.parquet`` already exists for this config hash.

    Returns
    -------
    dict with keys:
        ``'test_auc'``, ``'test_auc_subjcond'``, ``'skipped'``, ``'run_dir'``.
    """
    run_dir = Path(cfg.output_root) / "runs" / (cfg.run_tag or cfg.hash())
    run_dir.mkdir(parents=True, exist_ok=True)
    tp = run_dir / "preds_test.parquet"
    mp = run_dir / "meta.json"

    if skip_if_done and tp.exists() and mp.exists():
        try:
            df = pd.read_parquet(tp)
            if len(df) > 0:
                w_auc = (float(roc_auc_score(df["y_true"], df["y_pred"]))
                         if df["y_true"].nunique() > 1 else float("nan"))
                sc = (df.groupby(["subject", "cond"])
                        .agg(yt=("y_true", "first"), yp=("y_pred", "mean"))
                        .reset_index())
                sc_auc = (float(roc_auc_score(sc["yt"], sc["yp"]))
                          if sc["yt"].nunique() > 1 else float("nan"))
                return {"test_auc": w_auc, "test_auc_subjcond": sc_auc,
                        "skipped": True, "run_dir": str(run_dir)}
        except Exception:
            pass

    manager = SkeletonDataManager(cfg)
    manager.load()
    samples = list(manager.samples)
    tr, dv, te = split_subject_3way(samples, cfg.val_split, cfg.test_split, cfg.seed)
    print(f"  [{cfg.run_tag}] split train={len(tr)} dev={len(dv)} test={len(te)}")

    result = train_one(cfg, tr, dv, te, vib_cache, subject_ids, pretrained_gcn)
    extra = {
        "fusion": cfg.fusion, "modalities": cfg.modalities, "seed": cfg.seed,
        "protocol": cfg.protocol, "use_ssl": cfg.use_ssl_pretrain,
        "use_phase_loss": cfg.use_phase_loss, "use_sisc": cfg.use_sisc,
        "n_sensors": len(cfg.vib_sensors), "cfg_hash": cfg.hash(),
    }
    dev_df  = _preds_to_df(result["dev_preds"],  extra | {"split": "dev"})
    test_df = _preds_to_df(result["test_preds"], extra | {"split": "test"})
    dev_df.to_parquet(run_dir / "preds_dev.parquet", index=False)
    test_df.to_parquet(tp, index=False)

    meta = {
        "config": asdict(cfg),
        "best_dev_auc": float(result["best_dev_auc"]),
        "best_epoch": int(result["best_epoch"]),
        "nan_batch_rate": float(result["nan_batch_rate"]),
        "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
    }
    tm = subject_level_aucs(result["test_preds"])
    meta.update({
        "test_auc_window":   tm["window_auc"],
        "test_auc_subject":  tm["subject_auc"],
        "test_auc_subjcond": tm["subjcond_auc"],
        "n_test_subjects":   tm["n_subjects"],
    })
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    w_auc  = tm["window_auc"]
    sc_auc = tm["subjcond_auc"]
    print(
        f"  [{cfg.run_tag}] DONE  "
        f"best_dev={result['best_dev_auc']:.4f}  "
        f"test_auc_window={w_auc:.4f}  "
        f"test_auc_subjcond={sc_auc:.4f}  "
        f"NaN={result['nan_batch_rate']:.2%}"
    )
    return {"test_auc": w_auc, "test_auc_subjcond": sc_auc,
            "skipped": False, "run_dir": str(run_dir)}


def run_grid(
    base_cfg: Config,
    mode: str,
    vib_cache: Optional[VibrationCache],
    subject_ids: SubjectIDMap,
    pretrained_gcn: Optional[dict],
    time_budget_hr: float = 8.5,
    single_seed: Optional[int] = None,
) -> List[dict]:
    """Run all configurations in the grid, respecting a session time budget.

    Parameters
    ----------
    single_seed:
        When set, only run configs with this seed (useful for completing
        a first pass before adding error bars).
    time_budget_hr:
        Stop if the estimated next run would exceed this wall-clock limit.
    """
    configs = build_configs(base_cfg, mode)
    if single_seed is not None:
        configs = [c for c in configs if c.seed == single_seed]
    print(f"\n=== Grid ({mode}) — {len(configs)} configs ===")

    session_start = time.time()
    run_times: List[float] = []
    summary: List[dict] = []

    for i, cfg in enumerate(configs, 1):
        elapsed_hr = (time.time() - session_start) / 3600
        proj = (sum(run_times) / len(run_times) * 1.2 / 3600) if run_times else 0.0
        if elapsed_hr + proj > time_budget_hr - 0.5:
            print(
                f"\n[runner] ({i}/{len(configs)}) budget guard — "
                f"elapsed={elapsed_hr:.2f}h. Re-run to continue."
            )
            break
        print(
            f"\n[runner] ({i}/{len(configs)}) {cfg.run_tag}  "
            f"(elapsed={elapsed_hr:.2f}h, proj_next={proj:.2f}h)"
        )
        t0 = time.time()
        try:
            r = run_experiment(cfg, vib_cache, subject_ids, pretrained_gcn)
            dt = time.time() - t0
            if not r["skipped"]:
                run_times.append(dt)
            summary.append({
                "run_tag": cfg.run_tag,
                "test_auc": r["test_auc"],
                "test_auc_subjcond": r.get("test_auc_subjcond", float("nan")),
                "skipped": r["skipped"],
                "wall_time_s": dt,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.append({"run_tag": cfg.run_tag, "error": str(e)})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_grid_summary(summary)
    return summary


def _print_grid_summary(summary: List[dict]) -> None:
    print("\n=== Grid summary ===")
    print(f"  {'run':<25} {'AUC_w':>8} {'AUC_sc':>8} {'skip':>6}")
    for s in summary:
        if "error" in s:
            print(f"  {s['run_tag']:<25}  ERROR: {s['error']}")
            continue
        w  = s.get("test_auc", float("nan"))
        sc = s.get("test_auc_subjcond", float("nan"))
        w_str  = f"{w:.4f}"  if np.isfinite(w)  else "   n/a"
        sc_str = f"{sc:.4f}" if np.isfinite(sc) else "   n/a"
        print(f"  {s['run_tag']:<25} {w_str:>8} {sc_str:>8} {str(s['skipped']):>6}")
    n_complete = sum(
        1 for s in summary
        if "error" not in s and np.isfinite(s.get("test_auc", float("nan")))
    )
    print(f"\n[gate] grid complete: {n_complete == len(summary)}  "
          f"({n_complete}/{len(summary)})")

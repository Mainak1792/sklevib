"""
experiments/ablations.py
========================
Ablation study runner.

Each ablation flips one component switch in ``Config`` while keeping all
other settings at the best multi-modal configuration (GPC fusion, both
modalities).  Runs are resumable via the skip-if-done cache.

Ablations defined here
-----------------------
* ``no_ssl``   — disable SSL pre-training (``use_ssl_pretrain=False``)
* ``no_brsa``  — disable Body-Region Spatial Attention (``use_brsa=False``)
* ``no_tsa``   — disable temporal self-attention + phase loss
                 (``use_tsa_skel=False, use_phase_loss=False``)
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config
from data.dataset import SubjectIDMap
from data.vibration_cache import VibrationCache
from experiments.grid import run_experiment

# (run_tag_template, config_overrides, use_pretrained_gcn)
ABLATION_SPECS: List[Tuple[str, Dict, bool]] = [
    ("abl_nossl_s{seed}",  {"use_ssl_pretrain": False},              False),
    ("abl_nobrsa_s{seed}", {"use_brsa": False},                      True),
    ("abl_notsa_s{seed}",  {"use_tsa_skel": False,
                             "use_phase_loss": False},               True),
]


def run_ablations(
    base_cfg: Config,
    vib_cache: Optional[VibrationCache],
    subject_ids: SubjectIDMap,
    pretrained_gcn: Optional[dict],
    seeds: Tuple[int, ...] = (42,),
    epochs: int = 30,
    time_budget_hr: float = 8.5,
) -> List[dict]:
    """Run ablation experiments and return a summary list.

    Parameters
    ----------
    seeds:
        Seeds to evaluate.  Use ``(42,)`` for a single-seed first pass;
        expand to ``(42, 1337, 2024)`` for error bars.
    """
    session_start = time.time()
    summary = []

    for seed in seeds:
        for tag_tmpl, overrides, use_pre in ABLATION_SPECS:
            elapsed_hr = (time.time() - session_start) / 3600
            if elapsed_hr > time_budget_hr - 1.0:
                print(f"[ablations] budget guard — stopping at {elapsed_hr:.2f}h. "
                      "Re-run to continue.")
                return summary

            tag = tag_tmpl.format(seed=seed)
            d = asdict(base_cfg)
            d.update(overrides)
            d["vib_sensors"] = tuple(d["vib_sensors"])
            d.update({
                "fusion": "gpc",
                "modalities": "both",
                "seed": seed,
                "epochs": epochs,
                "run_tag": tag,
            })
            cfg_abl = Config(**d)
            pre_gcn = pretrained_gcn if use_pre else None
            print(f"\n[ablations] training {tag}  (overrides: {overrides})")
            try:
                r = run_experiment(cfg_abl, vib_cache, subject_ids, pre_gcn)
                summary.append({
                    "run_tag": tag,
                    "seed": seed,
                    "overrides": str(overrides),
                    **{k: r.get(k) for k in ("test_auc", "test_auc_subjcond", "skipped")},
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                summary.append({"run_tag": tag, "error": str(e)})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== Ablation summary ===")
    print(f"  {'run':<28} {'AUC_w':>8} {'AUC_sc':>8} {'skip':>6}")
    for s in summary:
        if "error" in s:
            print(f"  {s['run_tag']:<28}  ERROR: {s['error']}")
            continue
        w  = s.get("test_auc", float("nan"))
        sc = s.get("test_auc_subjcond", float("nan"))
        w_str  = f"{w:.4f}"  if np.isfinite(w)  else "   n/a"
        sc_str = f"{sc:.4f}" if np.isfinite(sc) else "   n/a"
        print(f"  {s['run_tag']:<28} {w_str:>8} {sc_str:>8} {str(s.get('skipped','')):>6}")
    return summary

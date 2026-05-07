"""
evaluation/protocols.py
=======================
Extended evaluation protocol splitters and runners (P3 – P5).

P3 — Cross-view transfer:
    Train on one camera view; test on the remaining views.

P4 — Cross-view × cross-condition:
    Train normal windows from one view and stress windows from another;
    test on a third view.  Diagnostic for view–label shortcuts.

P5 — Leave-one-subject-out (capped):
    Iterate over held-out subjects; budget-aware via ``max_folds``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import Config
from data.dataset import SubjectIDMap
from data.vibration_cache import VibrationCache
from evaluation.metrics import subject_level_aucs
from training.trainer import train_one


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def split_cross_view_p3(
    samples: List,
    train_view: str,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[List, List, List]:
    """P3 split: train on ``train_view``, test on all other views."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted({s[2] for s in samples}))
    rng.shuffle(subs)
    n_val = max(1, int(round(val_frac * len(subs))))
    val_subs = set(subs[:n_val])
    other_views = {s[4] for s in samples} - {train_view}
    tr = [s for s in samples if s[4] == train_view and s[2] not in val_subs]
    dv = [s for s in samples if s[4] == train_view and s[2] in val_subs]
    te = [s for s in samples if s[4] in other_views]
    return tr, dv, te


def split_cvxc_p4(
    samples: List,
    normal_view: str,
    stress_view: str,
    test_view: str,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[List, List, List]:
    """P4 split: normal from one view, stress from another, test on third."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted({s[2] for s in samples}))
    rng.shuffle(subs)
    n_val = max(1, int(round(val_frac * len(subs))))
    val_subs = set(subs[:n_val])

    def _train_ok(s):
        return s[2] not in val_subs and (
            (s[1] == 0 and s[4] == normal_view)
            or (s[1] == 1 and s[4] == stress_view)
        )

    def _dev_ok(s):
        return s[2] in val_subs and (
            (s[1] == 0 and s[4] == normal_view)
            or (s[1] == 1 and s[4] == stress_view)
        )

    tr = [s for s in samples if _train_ok(s)]
    dv = [s for s in samples if _dev_ok(s)]
    te = [s for s in samples if s[4] == test_view]
    return tr, dv, te


def loso_folds_p5(
    samples: List,
    max_folds: int = 8,
    seed: int = 0,
) -> List[Tuple[List, List, List, str]]:
    """P5: leave-one-subject-out folds capped at ``max_folds``."""
    subs = sorted({s[2] for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    test_subs = subs[: min(max_folds, len(subs))]
    folds = []
    for sb in test_subs:
        tr_all = [s for s in samples if s[2] != sb]
        te = [s for s in samples if s[2] == sb]
        remaining = sorted({s[2] for s in tr_all})
        if not remaining:
            continue
        dv_sub = remaining[hash(sb) % len(remaining)]
        tr = [s for s in tr_all if s[2] != dv_sub]
        dv = [s for s in tr_all if s[2] == dv_sub]
        folds.append((tr, dv, te, sb))
    return folds


# ---------------------------------------------------------------------------
# Generic protocol runner
# ---------------------------------------------------------------------------

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


def run_with_splits(
    cfg: Config,
    tr: List,
    dv: List,
    te: List,
    vib_cache: Optional[VibrationCache],
    subject_ids: SubjectIDMap,
    pretrained_gcn: Optional[dict],
    protocol_tag: str,
    run_tag: str,
    extra_meta: Optional[dict] = None,
    skip_if_done: bool = True,
) -> dict:
    """Train and evaluate with arbitrary pre-built splits.

    Persists parquets and meta.json in the same layout as
    :func:`~experiments.grid.run_experiment` so downstream table builders
    work unchanged.
    """
    from dataclasses import replace as _dc_replace
    cfg = _dc_replace(cfg, run_tag=run_tag, protocol=protocol_tag)
    run_dir = Path(cfg.output_root) / "runs" / cfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    tp = run_dir / "preds_test.parquet"
    mp = run_dir / "meta.json"

    if skip_if_done and tp.exists() and mp.exists():
        try:
            df = pd.read_parquet(tp)
            if len(df) > 0:
                w_auc = (
                    float(roc_auc_score(df["y_true"], df["y_pred"]))
                    if df["y_true"].nunique() > 1
                    else float("nan")
                )
                return {"test_auc": w_auc, "skipped": True, "run_dir": str(run_dir)}
        except Exception:
            pass

    if not tr or not dv or not te:
        print(f"  [{run_tag}] SKIPPED: empty split "
              f"(tr={len(tr)}, dv={len(dv)}, te={len(te)})")
        return {"test_auc": float("nan"), "skipped": True,
                "run_dir": str(run_dir), "reason": "empty_split"}

    print(f"  [{run_tag}] split train={len(tr)} dev={len(dv)} test={len(te)}")
    result = train_one(cfg, tr, dv, te, vib_cache, subject_ids, pretrained_gcn)

    extra = {
        "fusion": cfg.fusion, "modalities": cfg.modalities, "seed": cfg.seed,
        "protocol": protocol_tag, "use_ssl": cfg.use_ssl_pretrain,
        "n_sensors": len(cfg.vib_sensors), "cfg_hash": cfg.hash(),
    }
    if extra_meta:
        extra.update(extra_meta)

    dev_df  = _preds_to_df(result["dev_preds"],  extra | {"split": "dev"})
    test_df = _preds_to_df(result["test_preds"], extra | {"split": "test"})
    dev_df.to_parquet(run_dir / "preds_dev.parquet", index=False)
    test_df.to_parquet(tp, index=False)

    meta: dict = {
        "config": asdict(cfg),
        "protocol": protocol_tag,
        "best_dev_auc": float(result["best_dev_auc"]),
        "best_epoch": int(result["best_epoch"]),
        "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
    }
    if extra_meta:
        meta.update(extra_meta)
    tm = subject_level_aucs(result["test_preds"])
    meta.update({
        "test_auc_window":   tm["window_auc"],
        "test_auc_subject":  tm["subject_auc"],
        "test_auc_subjcond": tm["subjcond_auc"],
    })
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  [{run_tag}] DONE  test_auc_w={tm['window_auc']:.4f}")
    return {
        "test_auc": tm["window_auc"],
        "skipped": False,
        "run_dir": str(run_dir),
        "meta": meta,
    }

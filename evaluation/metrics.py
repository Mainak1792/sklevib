"""
evaluation/metrics.py
=====================
Evaluation functions: per-sample prediction collection, subject-level AUC
aggregation, and statistical tests (paired bootstrap, McNemar).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from sklearn.metrics import roc_auc_score
from torch.amp import autocast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, ds_for_meta) -> Dict[str, np.ndarray]:
    """Run model on ``loader`` and collect predictions with metadata.

    Returns
    -------
    dict with keys:
        ``sample_id``, ``y_pred``, ``y_true``, ``subject``,
        ``view``, ``cond``, ``speed``.
    """
    model.eval()
    sids, preds, labels = [], [], []
    for skel, vib, hv, label, _, sid in loader:
        skel = skel.to(DEVICE)
        vib = vib.to(DEVICE)
        hv = hv.to(DEVICE)
        with autocast("cuda", enabled=USE_AMP):
            logits, _, _ = model(skel, vib, hv)
        p = torch.sigmoid(logits.squeeze(-1)).float().cpu().numpy()
        if np.any(np.isnan(p)):
            p = np.nan_to_num(p, nan=0.5)
        preds.append(p)
        labels.append(label.numpy())
        sids.append(sid.numpy())

    sids = np.concatenate(sids)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "sample_id": sids,
        "y_pred": preds,
        "y_true": labels,
        "subject": np.array([ds_for_meta.data[i][2] for i in sids]),
        "view":    np.array([ds_for_meta.data[i][4] for i in sids]),
        "cond":    np.array([ds_for_meta.data[i][5] for i in sids]),
        "speed":   np.array([ds_for_meta.data[i][3] for i in sids], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Subject-level AUC aggregation
# ---------------------------------------------------------------------------

def subject_level_aucs(preds: Dict) -> Dict[str, float]:
    """Compute window, subject-mean, and (subject, condition)-mean AUCs.

    The ``(subject, condition)``-level AUC is the primary metric for
    paired within-subject stress detection: it averages all window
    predictions within a (subject, condition) cell before scoring.

    Returns
    -------
    dict with keys:
        ``window_auc``, ``subject_auc``, ``subjcond_auc``,
        ``n_subjects``, ``n_subjcond_groups``.
    """
    yt = preds["y_true"]
    yp = preds["y_pred"]
    subj = preds["subject"]
    cond = preds["cond"]
    out = {}

    out["window_auc"] = (
        float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    )

    df = pd.DataFrame({"yt": yt, "yp": yp, "subj": subj, "cond": cond})
    s = df.groupby("subj").agg(
        yt_nunique=("yt", "nunique"),
        yt_mean=("yt", "mean"),
        yp_mean=("yp", "mean"),
    )
    if s["yt_nunique"].max() == 1:
        sub_yt = s["yt_mean"].round().astype(int).to_numpy()
        sub_yp = s["yp_mean"].to_numpy()
        out["subject_auc"] = (
            float(roc_auc_score(sub_yt, sub_yp))
            if len(np.unique(sub_yt)) > 1
            else float("nan")
        )
    else:
        out["subject_auc"] = float("nan")
    out["n_subjects"] = int(s.shape[0])

    sc = (
        df.groupby(["subj", "cond"])
        .agg(yt=("yt", "first"), yp=("yp", "mean"))
        .reset_index()
    )
    out["subjcond_auc"] = (
        float(roc_auc_score(sc["yt"], sc["yp"]))
        if sc["yt"].nunique() > 1
        else float("nan")
    )
    out["n_subjcond_groups"] = int(sc.shape[0])
    return out


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_bootstrap_auc_diff(
    yt: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap confidence interval for AUC(A) − AUC(B).

    Returns
    -------
    mean, ci_lo, ci_hi:
        Point estimate and 95 % interval.
    """
    rng = np.random.default_rng(seed)
    N = len(yt)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        y = yt[idx]
        if len(np.unique(y)) < 2:
            diffs[i] = 0.0
            continue
        diffs[i] = roc_auc_score(y, ya[idx]) - roc_auc_score(y, yb[idx])
    return (
        float(diffs.mean()),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def mcnemar_p(
    yt: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    thr: float = 0.5,
) -> float:
    """Two-sided McNemar test p-value comparing binary predictions A and B."""
    yt = np.asarray(yt).astype(int)
    ca = (np.asarray(ya) > thr).astype(int) == yt
    cb = (np.asarray(yb) > thr).astype(int) == yt
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))

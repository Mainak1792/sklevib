"""
analysis/tables.py
==================
Generates paper-ready tables from persisted run results.

Table 1 — Comparison of fusion strategies
    Two-panel: (subject, condition)-level AUC [headline] and window-level
    AUC [secondary].  Includes paired bootstrap confidence intervals and
    McNemar p-values vs. the chosen baseline.

Threshold sensitivity table (Supplementary)
    Runs the data-curation pipeline at several activity thresholds and
    reports the effect on retained window counts and subject balance.
"""

from __future__ import annotations

import io
import contextlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import Config
from data.skeleton_manager import SkeletonDataManager
from evaluation.metrics import paired_bootstrap_auc_diff, mcnemar_p

# -----------------------------------------------------------------------
# Method definitions for each experiment mode
# -----------------------------------------------------------------------

METHODS_FOR_TABLE = {
    "quick": [
        ("GPC-Fusion (ours)", "gpc",    "both"),
    ],
    "lite": [
        ("Skeleton-only",     "concat", "skel"),
        ("Gated fusion",      "gated",  "both"),
        ("GPC-Fusion (ours)", "gpc",    "both"),
    ],
    "full": [
        ("Skeleton-only",     "concat", "skel"),
        ("Vibration-only",    "concat", "vib"),
        ("Late fusion",       "late",   "both"),
        ("Concat fusion",     "concat", "both"),
        ("Gated fusion",      "gated",  "both"),
        ("GPC-Fusion (ours)", "gpc",    "both"),
    ],
}


# -----------------------------------------------------------------------
# Table 1 builder
# -----------------------------------------------------------------------

def _subjcond_preds(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["seed", "subject", "cond"])
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
             n_windows=("y_pred", "size"))
        .reset_index()
    )


def make_table1(
    runs_root: str | Path,
    out_root: str | Path,
    methods: List[Tuple[str, str, str]],
    baseline_name: str = "Gated fusion",
) -> pd.DataFrame:
    """Build Table 1 and write ``table1.csv`` to ``out_root``.

    Parameters
    ----------
    methods:
        List of ``(display_name, fusion, modalities)`` tuples in the
        desired row order.
    baseline_name:
        Name of the method used as baseline for statistical tests.
    """
    dfs = [pd.read_parquet(p)
           for p in Path(runs_root).glob("*/preds_test.parquet")]
    if not dfs:
        print("[table1] no runs found.")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df_sc = _subjcond_preds(df)
    meta_cols = ["seed", "fusion", "modalities"]
    df_sc = df_sc.merge(df[meta_cols].drop_duplicates(subset=meta_cols),
                        on="seed", how="left")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    def _sub_for(fus, mod, sc=False):
        src = df_sc if sc else df
        return src[(src["fusion"] == fus) & (src["modalities"] == mod)]

    def _pool_window(sub):
        keys = sorted(sub["seed"].unique())
        yt, yp, sid = [], [], []
        for s in keys:
            g = sub[sub["seed"] == s].sort_values("sample_id")
            yt.append(g["y_true"].to_numpy())
            yp.append(g["y_pred"].to_numpy())
            sid.append(np.array([f"s{s}_{int(i)}" for i in g["sample_id"]]))
        return np.concatenate(yt), np.concatenate(yp), np.concatenate(sid)

    def _pool_sc(sub_sc):
        keys = sorted(sub_sc["seed"].unique())
        yt, yp, sid = [], [], []
        for s in keys:
            g = sub_sc[sub_sc["seed"] == s].sort_values(["subject", "cond"])
            yt.append(g["y_true"].to_numpy())
            yp.append(g["y_pred"].to_numpy())
            sid.append(np.array([f"s{s}_{r.subject}_{r.cond}"
                                  for r in g.itertuples()]))
        return np.concatenate(yt), np.concatenate(yp), np.concatenate(sid)

    def _aucs_by_seed(sub, on_sc=False):
        out = []
        for s, g in sub.groupby("seed"):
            g = g.sort_values("sample_id" if not on_sc else ["subject", "cond"])
            yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
            if len(np.unique(yt)) > 1:
                out.append(roc_auc_score(yt, yp))
        return out

    # Find baseline
    baseline_w = baseline_sc = None
    for name, fus, mod in methods:
        if name == baseline_name:
            w_sub  = _sub_for(fus, mod, sc=False)
            sc_sub = _sub_for(fus, mod, sc=True)
            if len(w_sub):  baseline_w  = _pool_window(w_sub)
            if len(sc_sub): baseline_sc = _pool_sc(sc_sub)
            break

    rows = []
    for name, fus, mod in methods:
        w_sub  = _sub_for(fus, mod, sc=False)
        sc_sub = _sub_for(fus, mod, sc=True)
        if len(w_sub) == 0:
            continue
        aucs_w  = _aucs_by_seed(w_sub,  on_sc=False)
        aucs_sc = _aucs_by_seed(sc_sub, on_sc=True)
        row: dict = {
            "method": name,
            "n_seeds": len(aucs_w),
            "auc_window_mean":   float(np.mean(aucs_w))  if aucs_w  else np.nan,
            "auc_window_std":    float(np.std(aucs_w))   if aucs_w  else np.nan,
            "auc_subjcond_mean": float(np.mean(aucs_sc)) if aucs_sc else np.nan,
            "auc_subjcond_std":  float(np.std(aucs_sc))  if aucs_sc else np.nan,
        }
        if baseline_w is not None and name != baseline_name:
            yt, yp, sid = _pool_window(w_sub)
            common = np.intersect1d(sid, baseline_w[2])
            if len(common) > 10:
                a = pd.Series(yp, index=sid).loc[common].to_numpy()
                b = pd.Series(baseline_w[1], index=baseline_w[2]).loc[common].to_numpy()
                yc = pd.Series(yt, index=sid).loc[common].to_numpy()
                d, lo, hi = paired_bootstrap_auc_diff(yc, a, b)
                row.update({"w_delta_auc": d, "w_ci_lo": lo, "w_ci_hi": hi,
                            "w_p_mcnemar": mcnemar_p(yc, a, b)})
        if baseline_sc is not None and name != baseline_name:
            yt, yp, sid = _pool_sc(sc_sub)
            common = np.intersect1d(sid, baseline_sc[2])
            if len(common) > 10:
                a = pd.Series(yp, index=sid).loc[common].to_numpy()
                b = pd.Series(baseline_sc[1], index=baseline_sc[2]).loc[common].to_numpy()
                yc = pd.Series(yt, index=sid).loc[common].to_numpy()
                d, lo, hi = paired_bootstrap_auc_diff(yc, a, b)
                row.update({"sc_delta_auc": d, "sc_ci_lo": lo, "sc_ci_hi": hi,
                            "sc_p_mcnemar": mcnemar_p(yc, a, b)})
        rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(out_root / "table1.csv", index=False)

    def _panel(prefix, title):
        print(f"\n{title}  (baseline: {baseline_name})")
        print("-" * 100)
        print(f"{'Method':<22} {'AUC μ±σ':<18} {'ΔAUC (95% CI)':<30} {'McNemar p':<12}")
        for r in rows:
            m = r.get(f"{prefix}_mean"); s = r.get(f"{prefix}_std")
            ausd = f"{m:.4f}±{s:.4f}" if (m is not None and np.isfinite(m)) else "—"
            d_key = "w_delta_auc" if "window" in prefix else "sc_delta_auc"
            if d_key in r:
                lo_k = d_key.replace("delta_auc", "ci_lo")
                hi_k = d_key.replace("delta_auc", "ci_hi")
                pk   = d_key.replace("delta_auc", "p_mcnemar")
                ds = f"{r[d_key]:+.4f} [{r[lo_k]:+.4f},{r[hi_k]:+.4f}]"
                ps = f"{r[pk]:.4g}"
            else:
                ds, ps = "(baseline)", "—"
            print(f"{r['method']:<22} {ausd:<18} {ds:<30} {ps:<12}")

    _panel("auc_subjcond",
           "Table 1a — (Subject, Condition) AUC  [headline]")
    _panel("auc_window",
           "Table 1b — Window-level AUC          [secondary]")
    print(f"\n  CSV → {out_root / 'table1.csv'}")
    return tab


# -----------------------------------------------------------------------
# Threshold sensitivity table (Supplementary)
# -----------------------------------------------------------------------

def threshold_sensitivity_sweep(
    cfg0: Config,
    thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
    out_root: str | Path | None = None,
) -> pd.DataFrame:
    """Re-run the data-curation pipeline at several activity thresholds.

    No model training is performed; only the window-curation funnel is
    re-evaluated.  Output: ``table_threshold_sensitivity.csv``.
    """
    from dataclasses import replace as _dc_replace
    out_root = Path(out_root) if out_root else Path(cfg0.output_root) / "tables"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    base = cfg0

    for thr in thresholds:
        cfg = _dc_replace(base, min_hip_speed_torso_per_sec=float(thr))
        mgr = SkeletonDataManager(cfg)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mgr.load()
        d = mgr.diagnostics
        labs = [s[1] for s in mgr.samples]
        n_norm = sum(1 for l in labs if l == 0)
        n_strs = sum(1 for l in labs if l == 1)
        sp = d.get("kept_speed_stats") or {}
        rows.append(dict(
            threshold             = float(thr),
            candidate_windows     = d["windows_candidate"],
            dropped_low_activity  = d["dropped_low_activity"],
            windows_after_qc      = d["windows_after_qc"],
            windows_after_balance = d["windows_after_balance"],
            retention_pct         = round(
                100 * d["windows_after_balance"] / max(1, d["windows_candidate"]), 2
            ),
            subjects_kept         = d["subjects_after_balance"],
            subjects_dropped      = d["subjects_dropped_imbalance"],
            n_normal              = n_norm,
            n_stress              = n_strs,
            stress_normal_ratio   = round(n_strs / max(1, n_norm), 3),
            speed_mean_torso_s    = round(float(sp.get("mean", float("nan"))), 3),
            speed_p10_torso_s     = round(float(sp.get("p10", float("nan"))), 3),
        ))

    df = pd.DataFrame(rows)
    csv_path = out_root / "table_threshold_sensitivity.csv"
    df.to_csv(csv_path, index=False)

    print("\nSupplementary Table — Activity-threshold sensitivity")
    print("=" * 100)
    for r in rows:
        print(
            f"thr={r['threshold']:.2f}  windows={r['windows_after_balance']}  "
            f"retain={r['retention_pct']:.1f}%  subjects={r['subjects_kept']}  "
            f"n/s={r['stress_normal_ratio']:.3f}  speed_mean={r['speed_mean_torso_s']:.3f}"
        )
    print(f"\n  CSV → {csv_path}")
    return df

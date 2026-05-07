"""
analysis/figures.py
===================
Generates all paper figures from persisted run results.

Figures produced
----------------
fig_phase_attention     — Per-phase GPC attention weights by class.
fig_speed_invariance    — Speed-correlation diagnostics (Table 3).
fig_speed_histogram     — Window speed distribution before/after filtering.
fig_curation_funnel     — Data retention bar chart.
fig_pipeline_examples   — Hip trajectory + speed timeline per clip.
fig_skeleton_thumbnails — Stick-figure snapshots of kept windows.
"""

from __future__ import annotations

import glob
import io
import contextlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.amp import autocast
from torch.utils.data import DataLoader

from config import Config
from data.dataset import MMDataset, SubjectIDMap
from data.skeleton_manager import SkeletonDataManager, split_subject_3way
from data.skeleton_qc import (
    clip_torso_length,
    frame_validity_mask,
    segment_unidirectional_passes,
    walking_axis,
)
from data.vibration_cache import VibrationCache
from models.stress_gait import MultiModalStressGait

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

_VIZ_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
]


def _worker_init(wid):
    import random as _r
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s); _r.seed(s)


# ---------------------------------------------------------------------------
# Phase-attention figure
# ---------------------------------------------------------------------------

def make_phase_attn_figure(
    runs_root: str | Path,
    cfg_base: Config,
    out_root: str | Path,
    subject_map: dict,
) -> None:
    gpc_ckpts = sorted(glob.glob(str(runs_root) + "/*gpc*/best.pt"))
    if not gpc_ckpts:
        print("[phase-attn] no GPC checkpoint found."); return
    ckpt_path = gpc_ckpts[0]
    print(f"[phase-attn] using {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = Config(**ck["config"])
    model = MultiModalStressGait(cfg).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    mgr = SkeletonDataManager(cfg); mgr.load()
    _, dv, _ = split_subject_3way(mgr.samples, cfg.val_split, cfg.test_split, cfg.seed)
    vc = VibrationCache(cfg, subject_map=subject_map)
    sids = SubjectIDMap(sorted(set(s[2] for s in mgr.samples)))
    ds = MMDataset(dv, cfg, vc, sids, is_train=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0,
                        pin_memory=USE_AMP, worker_init_fn=_worker_init)

    As, Av, Y, HV = [], [], [], []
    with torch.no_grad():
        for skel, vib, hv, label, _, _ in loader:
            skel = skel.to(DEVICE); vib = vib.to(DEVICE); hv = hv.to(DEVICE)
            _, _, aux = model(skel, vib, hv)
            if "gpc_alpha_s" not in aux:
                print("[phase-attn] model has no GPC auxiliary outputs."); return
            As.append(aux["gpc_alpha_s"].cpu().numpy())
            Av.append(aux["gpc_alpha_v"].cpu().numpy())
            Y.append(label.numpy()); HV.append(hv.cpu().numpy())

    As = np.concatenate(As); Av = np.concatenate(Av)
    Y = np.concatenate(Y); HV = np.concatenate(HV)
    K = cfg.phase_K
    phi = np.linspace(0, 1, K, endpoint=False) + 0.5 / K
    As_s, As_n = As[Y == 1].mean(0), As[Y == 0].mean(0)
    mask_v = HV > 0.5
    Av_s = Av[(Y == 1) & mask_v].mean(0) if (Y == 1 & mask_v).any() else np.zeros(K)
    Av_n = Av[(Y == 0) & mask_v].mean(0) if (Y == 0 & mask_v).any() else np.zeros(K)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(phi, As_s, "-o", color="C3", label="stress")
    axes[0].plot(phi, As_n, "-o", color="C0", label="normal")
    axes[0].set_title("Skeleton α(φ)"); axes[0].set_xlabel("Gait phase φ")
    axes[0].legend(fontsize=8)
    axes[1].plot(phi, Av_s, "-o", color="C3"); axes[1].plot(phi, Av_n, "-o", color="C0")
    axes[1].set_title("Vibration α(φ)"); axes[1].set_xlabel("Gait phase φ")
    axes[2].plot(phi, As_s - As_n, "-o", color="purple", label="skel (stress−normal)")
    axes[2].plot(phi, Av_s - Av_n, "-s", color="teal",   label="vib (stress−normal)")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("Class difference"); axes[2].legend(fontsize=8)
    try:
        r_diff, _ = pearsonr(As_s - As_n, Av_s - Av_n)
    except Exception:
        r_diff = float("nan")
    fig.suptitle(f"Phase-attention by class  |  skel↔vib diff Pearson r = {r_diff:.3f}")
    fig.tight_layout()
    out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_phase_attention.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "fig_phase_attention.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase-attn] → {out / 'fig_phase_attention.pdf'}")


# ---------------------------------------------------------------------------
# Speed invariance figure + Table 3
# ---------------------------------------------------------------------------

def make_speed_figure(
    runs_root: str | Path,
    out_root: str | Path,
    methods: List[Tuple[str, str, str]],
) -> None:
    dfs = [pd.read_parquet(p) for p in Path(runs_root).glob("*/preds_test.parquet")]
    if not dfs:
        return
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)

    def _partial_r(x, y, z):
        zc = np.vstack([z, np.ones_like(z)]).T
        bx, _, _, _ = np.linalg.lstsq(zc, x, rcond=None)
        by, _, _, _ = np.linalg.lstsq(zc, y, rcond=None)
        return float(pearsonr(x - zc @ bx, y - zc @ by)[0])

    rows = []; labels = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, fus, mod) in enumerate(methods):
        sub = df[(df["fusion"] == fus) & (df["modalities"] == mod)]
        if len(sub) == 0: continue
        yt = sub["y_true"].to_numpy().astype(float)
        yp = sub["y_pred"].to_numpy(); sp = sub["speed"].to_numpy()
        if len(np.unique(yt)) < 2:
            rows.append({"method": name, "note": "single-class"}); continue
        r_sp = pearsonr(sp, yp)[0]; r_pl = pearsonr(yp, yt)[0]
        r_part = _partial_r(yp, yt, sp)
        preserv = abs(r_part) / max(1e-6, abs(r_pl))
        rows.append({"method": name, "r_speed_pred": r_sp, "r_pred_label": r_pl,
                     "partial_r": r_part, "r_preservation": preserv,
                     "test_auc": roc_auc_score(yt, yp)})
        axes[0].bar(i, abs(r_sp)); labels.append(name)
        axes[1].bar(i, preserv)

    for ax, title in zip(axes, [
        "|r(speed, prediction)| — lower is better",
        "|partial r(pred,label | speed)| / |r(pred,label)| — higher = speed-invariant",
    ]):
        ax.set_title(title); ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    axes[0].axhline(0.1, color="k", ls=":", lw=0.5)
    axes[1].axhline(1.0, color="k", ls=":", lw=0.5); axes[1].set_ylim(0, 1.2)
    fig.tight_layout()
    out = Path(out_root); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_speed_invariance.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "fig_speed_invariance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "table3_speed.csv", index=False)
    print(f"[speed fig] → {out / 'fig_speed_invariance.pdf'}")


# ---------------------------------------------------------------------------
# Data-inspection visualisations
# ---------------------------------------------------------------------------

def visualize_data_pipeline(
    cfg: Config,
    out_root: Optional[str | Path] = None,
    n_example_clips: int = 4,
) -> None:
    """Generate four diagnostic figures from the data-curation pipeline."""
    import pandas as pd
    out = Path(out_root) if out_root else Path(cfg.output_root) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    mgr = SkeletonDataManager(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        mgr.load()
    diag = mgr.diagnostics
    print(f"[viz] {len(mgr.samples)} kept windows, "
          f"{diag['subjects_after_balance']} subjects")

    # -- 1. Speed histogram (before vs after filter) --
    from dataclasses import replace as _dc_replace
    speeds_kept = np.array([s[3] for s in mgr.samples], dtype=np.float32)
    try:
        cfg_noact = _dc_replace(cfg, use_activity_filter=False)
    except Exception:
        cfg_noact = cfg
    mgr_raw = SkeletonDataManager(cfg_noact)
    with contextlib.redirect_stdout(io.StringIO()):
        mgr_raw.load()
    speeds_raw = np.array([s[3] for s in mgr_raw.samples], dtype=np.float32)
    if len(speeds_raw):
        xmax = float(np.percentile(speeds_raw, 99.5))
        bins = np.linspace(0, max(xmax, cfg.min_hip_speed_torso_per_sec * 2), 60)
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), sharey=False)
        axes[0].hist(speeds_raw, bins=bins, color="#888", alpha=0.75, edgecolor="white")
        axes[0].axvline(cfg.min_hip_speed_torso_per_sec, color="crimson",
                        ls="--", lw=2, label=f"thr={cfg.min_hip_speed_torso_per_sec:.2f}")
        axes[0].set_xlabel("mean hip speed (torso/s)")
        axes[0].set_title(f"BEFORE filter  (n={len(speeds_raw)})"); axes[0].legend()
        if len(speeds_kept):
            axes[1].hist(speeds_kept, bins=bins, color="#4472C4", alpha=0.85, edgecolor="white")
            axes[1].axvline(cfg.min_hip_speed_torso_per_sec, color="crimson", ls="--", lw=2)
            axes[1].set_title(f"AFTER filter  (n={len(speeds_kept)})")
        fig.tight_layout()
        fig.savefig(out / "fig_speed_histogram.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(out / "fig_speed_histogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] → {out / 'fig_speed_histogram.pdf'}")

    # -- 2. Curation funnel --
    c_ = diag["windows_candidate"]
    stages = [
        ("candidate windows",       c_),
        ("  after valid-frames",    c_ - diag["dropped_invalid_frames"]),
        ("  after activity filter", c_ - diag["dropped_invalid_frames"] - diag["dropped_low_activity"]),
        ("  after direction check", diag["windows_after_qc"]),
        ("  after subject balance", diag["windows_after_balance"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.barh(range(len(stages)), [max(0, s[1]) for s in stages],
                   color=["#4472C4"] * 4 + ["#2E5F8F"])
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in stages]); ax.invert_yaxis()
    top = max(stages[0][1], 1)
    for i, (bar, (_, n)) in enumerate(zip(bars, stages)):
        ax.text(max(0, n) + top * 0.01, i, f"{n:,} ({100*max(0,n)/top:.1f}%)",
                va="center", fontsize=9)
    ax.set_xlim(0, top * 1.22)
    ax.set_title("Data-curation retention funnel"); fig.tight_layout()
    fig.savefig(out / "fig_curation_funnel.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out / "fig_curation_funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] → {out / 'fig_curation_funnel.pdf'}")

    # -- 3. Skeleton thumbnails --
    by_cond: dict = defaultdict(list)
    for s in mgr.samples:
        by_cond[s[5]].append(s)
    sel = []
    rng = np.random.default_rng(11)
    for cond in sorted(by_cond):
        pool = by_cond[cond]
        idxs = rng.choice(len(pool), size=min(2, len(pool)), replace=False)
        for i in idxs:
            sel.append(pool[int(i)])
    if sel:
        n_rows, n_cols = len(sel), 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows))
        if n_rows == 1: axes = axes[None, :]
        for r, (seg, lab, subj, sp, view, cond) in enumerate(sel):
            for cc, fi in enumerate([int(seg.shape[0] * f) for f in (0.1, 0.5, 0.9)]):
                ax = axes[r, cc]; xy = seg[fi]
                for a, b in _VIZ_EDGES:
                    if a < xy.shape[0] and b < xy.shape[0]:
                        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                                "-", color="#333", lw=1.5)
                ax.scatter(xy[:, 0], xy[:, 1], s=6, color="crimson", alpha=0.6)
                ax.invert_yaxis(); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(-3, 3); ax.set_ylim(3, -3)
                if cc == 0: ax.set_ylabel(f"{cond}\n{subj[:14]}", fontsize=8)
                ax.set_title(f"fr {fi}  sp={sp:.2f}", fontsize=7)
        fig.suptitle("Sample kept windows (hip-centred, torso-normalised)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_skeleton_thumbnails.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(out / "fig_skeleton_thumbnails.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] → {out / 'fig_skeleton_thumbnails.pdf'}")

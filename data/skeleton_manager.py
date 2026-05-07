"""
data/skeleton_manager.py
========================
Loads skeleton windows with multi-stage quality control and returns a
flat list of ``(skel_norm, label, subject, speed, view, cond)`` tuples.

QC pipeline (four stages)
-------------------------
1. **Clip-level** — too short or no detectable torso → skip.
2. **Pass-level** — segment into unidirectional walks; discard turns.
3. **Window-level** — frame-validity fraction, activity speed threshold,
   direction-reversal safety check.
4. **Subject-level** — drop subjects missing windows in either normal
   *or* stress condition (paired within-subject contrast).

A JSON diagnostics file is written to ``<output_root>/artifacts/``
after every ``load()`` call.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import Config
from .skeleton_qc import (
    clip_torso_length,
    frame_validity_mask,
    num_direction_reversals,
    segment_unidirectional_passes,
    walking_axis,
    window_speed_tps,
)

# Type alias for a single sample
Sample = Tuple[np.ndarray, int, str, float, str, str]


class SkeletonDataManager:
    """Load and quality-filter skeleton windows.

    Attributes
    ----------
    samples:
        List of ``(skel_norm, label, subject, speed_tps, view, cond)``
        where ``skel_norm`` is ``(seq_len, 33, 2)`` float32, hip-centred
        and torso-normalised.
    diagnostics:
        Dict of QC counters written to ``artifacts/filter_diagnostics.json``.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.samples: List[Sample] = []
        self.diagnostics: Dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> "SkeletonDataManager":
        """Run full QC pipeline. Populates ``self.samples`` in-place."""
        c = self.cfg
        root = Path(c.skeleton_root)

        # Build label map and disk-alias reverse map
        label_map = {cn: 0 for cn in c.conditions_normal}
        label_map.update({cn: 1 for cn in c.conditions_stress})
        if c.use_covariates:
            label_map.update({cn: 0 for cn in c.conditions_covariate})

        disk_to_internal = {v: k for k, v in c.folder_aliases.items()}
        for disk, internal in disk_to_internal.items():
            if internal in label_map and disk not in label_map:
                label_map[disk] = label_map[internal]

        stress_conds = set(c.conditions_stress)
        diag = dict(
            clips_scanned=0, clips_too_short=0, clips_unusable=0,
            passes_total=0, passes_too_short=0,
            windows_candidate=0,
            dropped_invalid_frames=0,
            dropped_low_activity=0,
            dropped_multi_direction=0,
            windows_after_qc=0,
            subjects_before_balance=0,
            subjects_dropped_imbalance=0,
            subjects_after_balance=0,
            windows_after_balance=0,
        )
        by_view: Dict[str, int] = defaultdict(int)
        by_cond: Dict[str, int] = defaultdict(int)
        cand: List[Sample] = []

        for view in c.views:
            vdir = root / view
            if not vdir.is_dir():
                continue
            for cdir in sorted(vdir.iterdir()):
                if not cdir.is_dir():
                    continue
                cname_disk = cdir.name.lower()
                if cname_disk not in label_map:
                    continue
                label = label_map[cname_disk]
                cname = disk_to_internal.get(cname_disk, cname_disk)

                for sdir in sorted(cdir.iterdir()):
                    if not sdir.is_dir():
                        continue
                    sp = sdir / "skeleton.npy"
                    if not sp.exists():
                        continue
                    sk = np.load(str(sp)).astype(np.float32)
                    if sk.shape[0] < c.min_frames:
                        diag["clips_too_short"] += 1
                        continue
                    diag["clips_scanned"] += 1

                    # Read per-clip metadata (fps, resolution)
                    fps, W, H = 20.0, 640, 360
                    mp = sdir / "metadata.json"
                    if mp.exists():
                        try:
                            with open(mp) as f:
                                _meta = json.load(f)
                            fps = float(_meta.get("fps", 20.0))
                            _res = _meta.get("resolution", [640, 360])
                            W, H = int(_res[0]), int(_res[1])
                        except Exception:
                            pass

                    # Un-normalise from [0,1] to pixels
                    sk = sk.copy()
                    sk[:, :, 0] *= W
                    sk[:, :, 1] *= H

                    subj = sdir.name
                    valid_mask = frame_validity_mask(sk)
                    torso = clip_torso_length(sk, valid_mask)
                    if torso <= 0 or int(valid_mask.sum()) < c.min_frames:
                        diag["clips_unusable"] += 1
                        continue
                    axis = walking_axis(sk, valid_mask)

                    if c.use_unidirectional_segmentation:
                        passes = segment_unidirectional_passes(
                            sk, valid_mask, axis,
                            c.turn_vel_smooth_frames,
                            c.turn_min_pass_frames,
                        )
                    else:
                        passes = [(0, int(sk.shape[0]))]
                    diag["passes_total"] += len(passes)

                    for ps, pe in passes:
                        if pe - ps < c.min_frames:
                            diag["passes_too_short"] += 1
                            continue
                        last_start = max(ps, pe - c.seq_len)
                        starts = list(range(ps, last_start + 1, c.window_stride))
                        if not starts:
                            starts = [ps]
                        for start in starts:
                            end = min(start + c.seq_len, pe)
                            if end - start < c.min_frames:
                                continue
                            seg_raw = sk[start:end]
                            diag["windows_candidate"] += 1

                            vfrac = float(valid_mask[start:end].mean())
                            if vfrac < c.min_window_valid_frac:
                                diag["dropped_invalid_frames"] += 1
                                continue

                            speed_tps = window_speed_tps(seg_raw, fps, torso)
                            if c.use_activity_filter and \
                                    speed_tps < c.min_hip_speed_torso_per_sec:
                                diag["dropped_low_activity"] += 1
                                continue

                            if c.use_unidirectional_segmentation:
                                nrev = num_direction_reversals(
                                    seg_raw, axis,
                                    c.turn_vel_smooth_frames,
                                    valid_mask=valid_mask[start:end],
                                )
                                if nrev > c.max_direction_changes_per_window:
                                    diag["dropped_multi_direction"] += 1
                                    continue

                            seg = self._resample(seg_raw[:, :, :2], c.seq_len)
                            seg = self._normalise(seg)
                            cand.append((seg, label, subj, speed_tps, view, cname))
                            diag["windows_after_qc"] += 1
                            by_view[view] += 1
                            by_cond[cname] += 1

        # Subject-level balancing
        if c.require_both_conditions_per_subject:
            per_subj: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {"normal": 0, "stress": 0}
            )
            for _, _, sj, _, _, cn in cand:
                per_subj[sj]["stress" if cn in stress_conds else "normal"] += 1
            diag["subjects_before_balance"] = len(per_subj)
            mn = c.min_windows_per_condition_per_subject
            kept_subj = {
                sj for sj, cnt in per_subj.items()
                if cnt["normal"] >= mn and cnt["stress"] >= mn
            }
            diag["subjects_dropped_imbalance"] = len(per_subj) - len(kept_subj)
            diag["subjects_dropped_list"] = sorted(set(per_subj) - kept_subj)[:50]

            by_view_b: Dict[str, int] = defaultdict(int)
            by_cond_b: Dict[str, int] = defaultdict(int)
            cand = [s for s in cand if s[2] in kept_subj]
            for _, _, _, _, vw, cn in cand:
                by_view_b[vw] += 1
                by_cond_b[cn] += 1
            by_view, by_cond = by_view_b, by_cond_b
            diag["subjects_after_balance"] = len(kept_subj)
        else:
            diag["subjects_before_balance"] = len(set(s[2] for s in cand))
            diag["subjects_after_balance"] = diag["subjects_before_balance"]
            diag["subjects_dropped_list"] = []

        diag["windows_after_balance"] = len(cand)

        if cand:
            arr = np.array([x[3] for x in cand], dtype=np.float32)
            diag["kept_speed_stats"] = dict(
                mean=float(arr.mean()), std=float(arr.std()),
                p10=float(np.percentile(arr, 10)),
                p50=float(np.percentile(arr, 50)),
                p90=float(np.percentile(arr, 90)),
                min=float(arr.min()), max=float(arr.max()),
            )
        else:
            diag["kept_speed_stats"] = {}

        self.samples = cand
        self.diagnostics = {
            **diag,
            "by_view": dict(by_view),
            "by_cond": dict(by_cond),
            "threshold_config": dict(
                min_hip_speed_torso_per_sec=c.min_hip_speed_torso_per_sec,
                min_window_valid_frac=c.min_window_valid_frac,
                turn_min_pass_frames=c.turn_min_pass_frames,
                max_direction_changes_per_window=c.max_direction_changes_per_window,
                min_windows_per_condition_per_subject=c.min_windows_per_condition_per_subject,
            ),
        }
        self._save_diagnostics()
        self._print_funnel(diag, by_view, by_cond, c)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_diagnostics(self) -> None:
        out = Path(self.cfg.output_root) / "artifacts" / "filter_diagnostics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.diagnostics, f, indent=2, default=str)

    @staticmethod
    def _resample(sk: np.ndarray, target: int) -> np.ndarray:
        T = sk.shape[0]
        if T == target:
            return sk
        return sk[np.linspace(0, T - 1, target, dtype=int)]

    @staticmethod
    def _normalise(xy: np.ndarray) -> np.ndarray:
        """Hip-centre and torso-normalise a ``(T, 33, 2)`` window."""
        hip = (xy[:, 23] + xy[:, 24]) / 2
        centred = xy - hip[:, None, :]
        ls = xy[:, 11]; rs = xy[:, 12]
        torso = float(np.mean(np.linalg.norm(((ls + rs) / 2) - hip, axis=-1))) + 1e-8
        return (centred / torso).astype(np.float32)

    def _print_funnel(self, diag, by_view, by_cond, c):
        print("\n========== Data-curation funnel ==========")
        print(f"clips scanned              : {diag['clips_scanned']}")
        print(f"  too short                : {diag['clips_too_short']}")
        print(f"  unusable (no torso)      : {diag['clips_unusable']}")
        print(f"unidirectional passes kept : {diag['passes_total']}  "
              f"(too short: {diag['passes_too_short']})")
        print(f"candidate windows          : {diag['windows_candidate']}")
        print(f"  dropped: invalid frames  : {diag['dropped_invalid_frames']}  "
              f"(min valid frac={c.min_window_valid_frac})")
        print(f"  dropped: low activity    : {diag['dropped_low_activity']}  "
              f"(threshold={c.min_hip_speed_torso_per_sec} torso/s)")
        print(f"  dropped: multi-direction : {diag['dropped_multi_direction']}  "
              f"(max reversals={c.max_direction_changes_per_window})")
        print(f"windows after QC           : {diag['windows_after_qc']}")
        print(f"subjects before balance    : {diag['subjects_before_balance']}")
        print(f"subjects dropped (no both) : {diag['subjects_dropped_imbalance']}")
        print(f"subjects after balance     : {diag['subjects_after_balance']}")
        print(f"windows after balance      : {diag['windows_after_balance']}")
        if diag["kept_speed_stats"]:
            s = diag["kept_speed_stats"]
            print(f"kept-window speed (torso/s): mean={s['mean']:.2f}  "
                  f"p10={s['p10']:.2f}  p50={s['p50']:.2f}  p90={s['p90']:.2f}")
        for v, n in by_view.items():
            print(f"  view {v:<10} {n:>6}")
        for c_, n in by_cond.items():
            print(f"  cond {c_:<10} {n:>6}")
        labs = [s[1] for s in self.samples]
        print(f"  final normal={sum(1 for l in labs if l == 0)}  "
              f"final stress={sum(1 for l in labs if l == 1)}  "
              f"final subjects={len(set(s[2] for s in self.samples))}")
        print("==========================================\n")


# ---------------------------------------------------------------------------
# Subject-disjoint data splits
# ---------------------------------------------------------------------------

def split_subject_3way(
    samples: List[Sample],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Subject-disjoint train / validation / test split.

    Returns
    -------
    tr, va, te:
        Three lists of samples with no subject overlap.
    """
    subs = np.array(sorted(set(s[2] for s in samples)))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test_s = set(subs[:n_test])
    val_s = set(subs[n_test: n_test + n_val])
    tr, va, te = [], [], []
    for s in samples:
        if s[2] in test_s:
            te.append(s)
        elif s[2] in val_s:
            va.append(s)
        else:
            tr.append(s)
    return tr, va, te

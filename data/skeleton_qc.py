"""
data/skeleton_qc.py
===================
Per-clip and per-window quality-control helpers for MediaPipe skeleton data.

All functions operate on **raw** (un-centred) skeleton arrays of shape
``(T, 33, C)`` where ``C >= 2`` (x, y in pixels; optional channel 2 is
MediaPipe visibility in [0, 1]).

Design notes
------------
* Coordinates must be un-normalised to pixels *before* any QC call.
  MediaPipe stores landmarks in [0, 1]; scale x by frame width and y by
  frame height to obtain a consistent metric.
* Speed is expressed in *torso-lengths/second* so it is invariant to
  camera distance and subject body size.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Frame validity
# ---------------------------------------------------------------------------

def frame_validity_mask(skel_raw: np.ndarray, vis_thr: float = 0.3) -> np.ndarray:
    """Boolean mask: True for frames where all torso joints are confidently visible.

    Uses MediaPipe visibility score (channel 2) when available; falls back to
    a NaN / origin heuristic otherwise.

    Parameters
    ----------
    skel_raw:
        Shape ``(T, 33, C)``.  Channel 2, if present, is visibility ∈ [0, 1].
    vis_thr:
        Minimum visibility for joints 11, 12, 23, 24 (shoulders + hips).
    """
    if skel_raw.shape[-1] < 3:
        key = skel_raw[:, [11, 12, 23, 24], :2]
        has_nan = np.any(~np.isfinite(key), axis=(1, 2))
        has_origin = np.any(np.all(np.abs(key) < 1e-6, axis=2), axis=1)
        return ~(has_nan | has_origin)

    vis = skel_raw[:, [11, 12, 23, 24], 2]
    finite = np.all(np.isfinite(vis), axis=1)
    return finite & np.all(vis > vis_thr, axis=1)


# ---------------------------------------------------------------------------
# Clip-level statistics
# ---------------------------------------------------------------------------

def clip_torso_length(skel_raw: np.ndarray, valid_mask: np.ndarray) -> float:
    """Median distance (pixels) between hip mid-point and shoulder mid-point
    over valid frames.  Returns 0.0 if fewer than 3 valid frames."""
    hip = (skel_raw[:, 23, :2] + skel_raw[:, 24, :2]) / 2
    shoulder = (skel_raw[:, 11, :2] + skel_raw[:, 12, :2]) / 2
    d = np.linalg.norm(shoulder - hip, axis=-1)
    good = valid_mask & np.isfinite(d) & (d > 1e-6)
    return float(np.median(d[good])) if good.sum() >= 3 else 0.0


def walking_axis(skel_raw: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Principal axis of hip motion via PCA (shape ``(2,)``).

    Camera-agnostic: works for any camera view because it uses the largest
    eigenvector of the hip covariance matrix.
    """
    hip = (skel_raw[:, 23, :2] + skel_raw[:, 24, :2]) / 2
    h = hip[valid_mask]
    if len(h) < 3:
        return np.array([1.0, 0.0], dtype=np.float32)
    c = h - h.mean(0, keepdims=True)
    cov = c.T @ c
    _, v = np.linalg.eigh(cov)
    axis = v[:, -1]
    if float(np.dot(c[-1] - c[0], axis)) < 0:
        axis = -axis
    n = float(np.linalg.norm(axis))
    return (axis / n).astype(np.float32) if n > 0 else np.array([1.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Private: moving average
# ---------------------------------------------------------------------------

def _moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if k <= 1 or len(x) < 2:
        return x.copy()
    k = int(min(k, len(x)))
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(xp, kernel, mode="valid")
    return sm[: len(x)].astype(np.float32)


# ---------------------------------------------------------------------------
# Unidirectional segmentation
# ---------------------------------------------------------------------------

def segment_unidirectional_passes(
    skel_raw: np.ndarray,
    valid_mask: np.ndarray,
    axis: np.ndarray,
    smooth_k: int,
    min_pass_frames: int,
) -> list[tuple[int, int]]:
    """Return ``[(start, end)]`` frame-ranges of monotone hip motion along ``axis``.

    Turning points and occlusions both terminate a pass cleanly.

    Parameters
    ----------
    skel_raw:
        Shape ``(T, 33, C)`` in pixel coordinates.
    valid_mask:
        Boolean, shape ``(T,)``.
    axis:
        Unit vector from :func:`walking_axis`.
    smooth_k:
        Gaussian smoothing kernel half-width in frames.
    min_pass_frames:
        Passes shorter than this are discarded.
    """
    hip = (skel_raw[:, 23, :2] + skel_raw[:, 24, :2]) / 2
    # Forward-fill hip across invalid frames
    hip_ff = hip.copy()
    last_ok = None
    for t in range(len(hip_ff)):
        if valid_mask[t]:
            last_ok = hip_ff[t].copy()
        elif last_ok is not None:
            hip_ff[t] = last_ok

    proj = _moving_average_1d(hip_ff @ axis, smooth_k)
    vel = np.diff(proj)

    pair_valid = (valid_mask[:-1] & valid_mask[1:]).astype(bool)
    vel_clean = vel[pair_valid] if pair_valid.any() else vel
    eps = max(1e-6, 0.25 * float(np.std(vel_clean)))
    sgn = np.sign(vel) * (np.abs(vel) > eps)
    sgn = sgn * pair_valid.astype(sgn.dtype)

    passes, i, N = [], 0, len(sgn)
    while i < N:
        if sgn[i] == 0:
            i += 1
            continue
        cur, j = sgn[i], i
        while j < N and sgn[j] == cur:
            j += 1
        start_f, end_f = i, j + 1
        if end_f - start_f >= min_pass_frames:
            passes.append((int(start_f), int(end_f)))
        i = j
    return passes


def num_direction_reversals(
    skel_raw: np.ndarray,
    axis: np.ndarray,
    smooth_k: int,
    valid_mask: np.ndarray | None = None,
) -> int:
    """Count sign-reversals of smoothed hip velocity along ``axis``."""
    hip = (skel_raw[:, 23, :2] + skel_raw[:, 24, :2]) / 2
    if valid_mask is not None:
        hip = hip.copy()
        last = None
        for t in range(len(hip)):
            if valid_mask[t]:
                last = hip[t].copy()
            elif last is not None:
                hip[t] = last
    proj = _moving_average_1d(hip @ axis, smooth_k)
    vel = np.diff(proj)
    if valid_mask is not None:
        pair_valid = (valid_mask[:-1] & valid_mask[1:]).astype(bool)
        vel_clean = vel[pair_valid] if pair_valid.any() else vel
    else:
        vel_clean = vel
    eps = max(1e-6, 0.25 * float(np.std(vel_clean)))
    sgn = np.sign(vel) * (np.abs(vel) > eps)
    nz = sgn[sgn != 0]
    if len(nz) < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


# ---------------------------------------------------------------------------
# Window-level speed
# ---------------------------------------------------------------------------

def window_speed_tps(skel_raw: np.ndarray, fps: float, torso: float) -> float:
    """Mean hip speed in torso-lengths/sec over a window segment."""
    if torso <= 0:
        return 0.0
    hip = (skel_raw[:, 23, :2] + skel_raw[:, 24, :2]) / 2
    step = np.linalg.norm(np.diff(hip, axis=0), axis=-1)
    if len(step) == 0:
        return 0.0
    return float(np.mean(step)) * float(fps) / float(torso)

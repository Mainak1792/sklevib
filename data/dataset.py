"""
data/dataset.py
===============
PyTorch Dataset and a deterministic SubjectIDMap for use across all splits.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config
from .vibration_cache import VibrationCache


class SubjectIDMap:
    """Deterministic integer IDs for subjects, consistent across runs.

    IDs are assigned in alphabetical order of the subject's folder name,
    so they are reproducible as long as the subject list does not change.
    """

    def __init__(self, names: List[str]) -> None:
        uniq = sorted(set(names))
        self._n2i: Dict[str, int] = {n: i for i, n in enumerate(uniq)}

    def id_of(self, name: str) -> int:
        return self._n2i[name]

    def __len__(self) -> int:
        return len(self._n2i)


# Left-right symmetric MediaPipe joint pairs (for horizontal-flip augmentation)
_LR_SYMMETRIC = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
    (1, 4), (2, 5), (3, 6), (7, 8),
]


class MMDataset(Dataset):
    """Multi-modal skeleton + vibration dataset.

    Each item returns a 6-tuple::

        skel  : (seq_len, 33, 2) float32 tensor
        vib   : (S, 3*n_sensors, H, W) float32 tensor
        has_vib: scalar float32  (1.0 if vib strips available, else 0.0)
        label : scalar float32
        sid   : scalar int64  (subject integer ID)
        idx   : scalar int64  (index into self.data, for post-hoc lookup)

    Parameters
    ----------
    data:
        List of ``(skel_norm, label, subject, speed, view, cond)`` from
        :class:`~data.skeleton_manager.SkeletonDataManager`.
    cfg:
        Experiment config.
    vib_cache:
        Pre-loaded vibration cache, or ``None`` for skeleton-only runs.
    sids:
        :class:`SubjectIDMap` built from the **full** sample list so IDs
        are stable across train / val / test splits.
    is_train:
        Whether to apply data augmentation.
    """

    def __init__(
        self,
        data,
        cfg: Config,
        vib_cache: Optional[VibrationCache],
        sids: SubjectIDMap,
        is_train: bool = True,
    ) -> None:
        self.data = data
        self.cfg = cfg
        self.vib = vib_cache
        self.sids = sids
        self.is_train = is_train
        self.S = cfg.vib_n_strips_train if is_train else cfg.vib_n_strips_eval

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _aug_skel(self, s: np.ndarray) -> np.ndarray:
        if random.random() < 0.3:
            s = s + np.random.randn(*s.shape).astype(np.float32) * 0.01
        if random.random() < 0.3:
            s = np.roll(s, random.randint(-3, 3), axis=0)
        if random.random() < 0.5:
            s = s.copy()
            s[:, :, 0] *= -1
            for a, b in _LR_SYMMETRIC:
                tmp = s[:, a, :].copy()
                s[:, a, :] = s[:, b, :]
                s[:, b, :] = tmp
        return s

    def _aug_vib(self, v: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            v = v[:, :, ::-1, :].copy()
        if random.random() < 0.3:
            v = np.clip(v * (1.0 + (random.random() - 0.5) * 0.2), 0, 1)
        return v

    # ------------------------------------------------------------------
    # Vibration strip sampling
    # ------------------------------------------------------------------

    def _sample_strips(self, strips: Optional[list]) -> np.ndarray:
        S = self.S
        C = 3 * len(self.cfg.vib_sensors)
        H, W = self.cfg.vib_img_size
        if not strips:
            return np.zeros((S, H, W, C), dtype=np.float32)
        n = len(strips)
        if self.is_train:
            idx = np.random.choice(n, size=S, replace=(n < S))
        else:
            if n == 1:
                idx = [0] * S
            elif n >= S:
                idx = np.linspace(0, n - 1, S, dtype=int)
            else:
                idx = list(range(n)) + [n - 1] * (S - n)
        return np.stack([strips[i] for i in idx], axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        skel, label, subj, _speed, _view, cond = self.data[idx][:6]
        skel = skel.copy()
        if self.is_train:
            skel = self._aug_skel(skel)

        strips: Optional[list] = None
        if self.cfg.modalities in ("vib", "both") and self.vib is not None:
            strips = self.vib.get(subj, cond)

        arr = self._sample_strips(strips)
        has_vib = 1.0 if strips else 0.0
        if strips and self.is_train:
            arr = self._aug_vib(arr)

        return (
            torch.from_numpy(skel).float(),
            torch.from_numpy(arr).permute(0, 3, 1, 2).float(),
            torch.tensor(has_vib, dtype=torch.float32),
            torch.tensor(float(label), dtype=torch.float32),
            torch.tensor(self.sids.id_of(subj), dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )

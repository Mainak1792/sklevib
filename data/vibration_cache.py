"""
data/vibration_cache.py
=======================
Loads and caches CWT strip images for the vibration modality.

Each bucket is keyed by ``(vib_folder_name, condition, sensor)`` and
stores up to ``vib_max_strips`` images stacked channel-wise over sensors:
``(H, W, 3 * n_sensors)`` float32 in [0, 1].
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config


def _load_cwt_imgs(
    folder: str,
    size: Tuple[int, int] = (224, 224),
    n: int = 10,
) -> Optional[List[np.ndarray]]:
    """Load up to ``n`` evenly-spaced CWT images from ``folder``."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(
        f for f in glob.glob(os.path.join(folder, "*.*"))
        if f.lower().endswith(exts)
    )
    if not files:
        return None
    chosen = (
        files if len(files) <= n
        else [files[int(i * len(files) / n)] for i in range(n)]
    )
    imgs = []
    for fp in chosen:
        img = cv2.imread(fp)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        r = img.max() - img.min()
        if r > 1e-8:
            img = (img - img.min()) / r
        imgs.append(np.clip(img, 0, 1).astype(np.float32))
    return imgs if imgs else None


class VibrationCache:
    """Pre-loads CWT strip images for every ``(subject, condition, sensor)`` bucket.

    Parameters
    ----------
    cfg:
        Experiment config.
    subject_map:
        ``{skel_subject_name: vib_folder_name}`` from :func:`build_subject_map`.
    """

    def __init__(self, cfg: Config, subject_map: Dict[str, str]) -> None:
        self.cfg = cfg
        self.subject_map = dict(subject_map)
        self.store: Dict[Tuple[str, str, str], List[np.ndarray]] = {}

        root = Path(cfg.vib_root)
        if not root.exists():
            print(f"[VibCache] WARNING: {root} does not exist — "
                  "vibration data will be unavailable.")
            return

        all_cond = set(cfg.conditions_normal) | set(cfg.conditions_stress)
        if cfg.use_covariates:
            all_cond |= set(cfg.conditions_covariate)

        for sensor in cfg.vib_sensors:
            sdir = root / sensor
            if not sdir.is_dir():
                continue
            for cond in sorted(all_cond):
                folder_name = cfg.folder_aliases.get(cond, cond)
                cdir = sdir / folder_name
                if not cdir.is_dir():
                    continue
                subj_dirs = [d for d in sorted(cdir.iterdir()) if d.is_dir()]
                args = [(str(d), d.name) for d in subj_dirs]

                def _load(arg):
                    folder, name = arg
                    return name, _load_cwt_imgs(folder, cfg.vib_img_size, cfg.vib_cache_n)

                with ThreadPoolExecutor(max_workers=8) as ex:
                    for name, imgs in ex.map(_load, args):
                        if imgs:
                            self.store[(name, cond.lower(), sensor)] = imgs[: cfg.vib_max_strips]

        print(
            f"[VibCache] {len(self.store)} buckets across "
            f"{len(cfg.vib_sensors)} sensor(s)"
        )

    def get(
        self, skel_subject: str, cond: str
    ) -> Optional[List[np.ndarray]]:
        """Return a list of ``(H, W, 3*n_sensors)`` strip images or ``None``.

        All sensors are stacked channel-wise.  Returns ``None`` if any
        sensor bucket is missing for this subject / condition pair.
        """
        if not self.store:
            return None
        vib_name = self.subject_map.get(skel_subject)
        if not vib_name:
            return None
        per_sensor = []
        for sensor in self.cfg.vib_sensors:
            s = self.store.get((vib_name, cond.lower(), sensor))
            if s is None:
                return None
            per_sensor.append(s)
        n = min(len(s) for s in per_sensor)
        if n == 0:
            return None
        return [
            np.concatenate(
                [per_sensor[j][i] for j in range(len(per_sensor))], axis=-1
            ).astype(np.float32)
            for i in range(n)
        ]

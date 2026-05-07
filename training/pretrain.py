"""
training/pretrain.py
====================
Masked joint prediction pre-training for the GCN backbone (SSL).

A fraction of joints is zeroed out and the model is trained to reconstruct
the masked positions.  The resulting encoder captures kinematic priors that
improve convergence and generalisation of the downstream supervised task.

The pre-trained state dict is cached to disk; subsequent runs load from
cache so the overhead is paid only once per dataset.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from config import Config
from models.graph import SpatialGCN, build_adjacency

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


class MaskedJointPredictor(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.gcn = SpatialGCN(
            cfg.joint_dim,
            cfg.graph_hidden,
            cfg.num_gcn_layers,
            build_adjacency(cfg.num_joints),
            cfg.dropout,
        )
        self.pred = nn.Sequential(
            nn.Linear(cfg.graph_hidden, cfg.graph_hidden),
            nn.ReLU(),
            nn.Linear(cfg.graph_hidden, cfg.joint_dim),
        )
        self.mask_ratio = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape
        mask = torch.rand(B, T, V, device=x.device) < self.mask_ratio
        t = x.clone()
        xm = x.clone()
        xm[mask] = 0.0
        f = self.gcn(xm)
        p = self.pred(f)
        return F.mse_loss(p[mask], t[mask])


class _SkeletonOnlyDataset(Dataset):
    def __init__(self, samples: List) -> None:
        self.sk = [s[0] for s in samples]

    def __len__(self) -> int:
        return len(self.sk)

    def __getitem__(self, i: int) -> torch.Tensor:
        s = self.sk[i].copy()
        if random.random() < 0.5:
            s += np.random.randn(*s.shape).astype(np.float32) * 0.015
        if random.random() < 0.5:
            s[:, :, 0] *= -1
        return torch.tensor(s, dtype=torch.float32)


def _worker_init_fn(wid: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def pretrain_gcn(
    cfg: Config,
    samples: List,
    epochs: int = 15,
    verbose: bool = True,
) -> dict:
    """Pre-train the GCN backbone via masked joint prediction.

    Parameters
    ----------
    cfg:
        Experiment config (seeds, architecture, batch size).
    samples:
        Full sample list from :class:`~data.SkeletonDataManager`.
    epochs:
        Number of pre-training epochs.
    verbose:
        Print per-epoch loss.

    Returns
    -------
    state_dict:
        CPU-side GCN ``state_dict`` ready for :meth:`MultiModalStressGait.load_pretrained_gcn`.
    """
    ds = _SkeletonOnlyDataset(samples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=USE_AMP,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    model = MaskedJointPredictor(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler("cuda") if USE_AMP else None

    if verbose:
        print(f"[SSL pretrain] {len(ds)} samples × {epochs} epochs — device: {DEVICE}")

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with autocast("cuda"):
                    loss = model(b)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = model(b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach())
        sch.step()
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  ep {ep:>2}/{epochs}  loss={tot / len(loader):.5f}")

    state = {k: v.cpu() for k, v in model.gcn.state_dict().items()}
    del model, opt
    if USE_AMP:
        torch.cuda.empty_cache()
    gc.collect()
    return state


def get_or_create_ssl_state(cfg: Config, samples: List, epochs: int = 15) -> dict:
    """Load cached SSL weights or run pre-training if no cache exists."""
    cache = Path(cfg.output_root) / "artifacts" / "ssl_state.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        state = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[SSL pretrain] loaded cached weights from {cache}")
        return state
    print("[SSL pretrain] no cache found — running pre-training …")
    state = pretrain_gcn(cfg, samples, epochs=epochs, verbose=True)
    torch.save(state, cache)
    print(f"[SSL pretrain] saved → {cache}")
    return state

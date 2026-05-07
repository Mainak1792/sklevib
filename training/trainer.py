"""
training/trainer.py
===================
Supervised training loop for :class:`~models.MultiModalStressGait`.

Features
--------
* Mixed-precision training (AMP) when a CUDA device is detected.
* Warmup + cosine annealing learning-rate schedule.
* Separate learning-rate group for pre-trained sub-networks (GCN + ViT
  backbone) vs. randomly-initialised heads.
* ViT backbone is frozen for the first ``vib_backbone_freeze_epochs``
  epochs to stabilise the skeleton branch.
* Early stopping (patience = 7 epochs) on (subject, condition)-level AUC.
* Checkpoint saved only on AUC improvement; best model restored before
  test evaluation.
"""

from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config
from data.dataset import MMDataset, SubjectIDMap
from data.vibration_cache import VibrationCache
from evaluation.metrics import evaluate, subject_level_aucs
from losses.losses import LossBundle
from models.stress_gait import MultiModalStressGait

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def _worker_init(wid: int) -> None:
    import random
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def _build_param_groups(
    model: MultiModalStressGait,
    cfg: Config,
    pretrained_gcn_ids: set,
    lr: float,
) -> List[dict]:
    slow_ids = set(pretrained_gcn_ids)
    if hasattr(model, "vib_encoder"):
        slow_ids |= {id(p) for p in model.vib_encoder.backbone.parameters()}
    slow, fast = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (slow if id(p) in slow_ids else fast).append(p)
    return [{"params": slow, "lr": lr * 0.1}, {"params": fast, "lr": lr}]


def train_one(
    cfg: Config,
    tr: List,
    dv: List,
    te: List,
    vib_cache: Optional[VibrationCache],
    sids: SubjectIDMap,
    pretrained_gcn: Optional[dict] = None,
) -> Dict:
    """Train a single model configuration and return predictions.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    tr, dv, te:
        Train / validation / test sample lists from
        :func:`~data.split_subject_3way`.
    vib_cache:
        Pre-loaded vibration cache.
    sids:
        Subject ID map (built from the full sample list).
    pretrained_gcn:
        Optional GCN ``state_dict`` from SSL pre-training.

    Returns
    -------
    result:
        ``{'best_dev_auc', 'best_epoch', 'nan_batch_rate',
           'dev_preds', 'test_preds'}``
    """
    import random
    import os

    # Seeding
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    tr_ds = MMDataset(tr, cfg, vib_cache, sids, is_train=True)
    dv_ds = MMDataset(dv, cfg, vib_cache, sids, is_train=False)
    te_ds = MMDataset(te, cfg, vib_cache, sids, is_train=False)
    tl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=0, pin_memory=USE_AMP, drop_last=True,
                    worker_init_fn=_worker_init)
    dl = DataLoader(dv_ds, batch_size=cfg.batch_size, num_workers=0,
                    pin_memory=USE_AMP, worker_init_fn=_worker_init)
    el = DataLoader(te_ds, batch_size=cfg.batch_size, num_workers=0,
                    pin_memory=USE_AMP, worker_init_fn=_worker_init)

    model = MultiModalStressGait(cfg).to(DEVICE)
    if pretrained_gcn is not None and cfg.use_ssl_pretrain:
        model.load_pretrained_gcn(pretrained_gcn)
    if cfg.modalities in ("vib", "both") and cfg.vib_backbone_freeze_epochs > 0:
        model.vib_encoder.freeze_backbone()

    loss_fn = LossBundle(cfg).to(DEVICE)
    pretrained_gcn_ids = (
        {id(p) for p in model.gcn.parameters()} if pretrained_gcn else set()
    )

    def _build_opt():
        return torch.optim.AdamW(
            _build_param_groups(model, cfg, pretrained_gcn_ids, cfg.lr),
            weight_decay=cfg.weight_decay,
        )

    opt = _build_opt()
    ws = cfg.warmup_epochs * max(1, len(tl))
    ts = cfg.epochs * max(1, len(tl))
    wu = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=max(1, ws))
    co = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, ts - ws), eta_min=1e-6)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [wu, co], milestones=[ws])
    scaler = GradScaler("cuda") if USE_AMP else None

    run_dir = Path(cfg.output_root) / "runs" / (cfg.run_tag or cfg.hash())
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pt"
    best_auc = -1.0
    best_ep = 0
    patience = 0
    nan_b = 0

    for ep in range(1, cfg.epochs + 1):
        # Unfreeze ViT backbone
        if (
            ep == cfg.vib_backbone_freeze_epochs + 1
            and cfg.vib_backbone_freeze_epochs > 0
            and cfg.modalities in ("vib", "both")
        ):
            model.vib_encoder.unfreeze_backbone()
            opt = _build_opt()

        model.train()
        tl_loss = 0.0
        nb = 0
        gnorm_sum = 0.0
        gnorm_n = 0

        for skel, vib, hv, label, sid_, _ in tl:
            skel = skel.to(DEVICE, non_blocking=True)
            vib = vib.to(DEVICE, non_blocking=True)
            hv = hv.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            sid_ = sid_.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=USE_AMP):
                logits, proj, aux = model(skel, vib, hv)
                logits = logits.squeeze(-1)
                loss, _parts = loss_fn(logits, label, proj, sid_, aux)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_b += 1
                opt.zero_grad(set_to_none=True)
                continue

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
                opt.step()

            gnorm_sum += gnorm
            gnorm_n += 1
            sch.step()
            tl_loss += float(loss.detach())
            nb += 1

        dev = evaluate(model, dl, dv_ds)
        dev_m = subject_level_aucs(dev)
        dev_auc = dev_m["window_auc"] if np.isfinite(dev_m["window_auc"]) else 0.0
        dev_sc = dev_m["subjcond_auc"]
        avg_gnorm = gnorm_sum / max(1, gnorm_n)
        tag = cfg.run_tag or cfg.hash()
        sc_str = f"{dev_sc:.4f}" if np.isfinite(dev_sc) else "  n/a"
        print(
            f"  [{tag}] ep {ep:>2}/{cfg.epochs}"
            f"  loss={tl_loss / max(1, nb):.4f}"
            f"  dev_auc_w={dev_auc:.4f}"
            f"  dev_auc_sc={sc_str}"
            f"  gnorm={avg_gnorm:.2f}"
        )

        select_metric = dev_sc if np.isfinite(dev_sc) else dev_auc
        if select_metric > best_auc:
            best_auc, best_ep, patience = select_metric, ep, 0
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "dev_auc_window": dev_auc,
                    "dev_auc_subjcond": dev_sc,
                    "config": asdict(cfg),
                },
                ckpt,
            )
        else:
            patience += 1
            if patience >= 7:
                break

    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd["model_state_dict"])
    dev_final = evaluate(model, dl, dv_ds)
    test_final = evaluate(model, el, te_ds)
    del model, opt
    if USE_AMP:
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "best_dev_auc": best_auc,
        "best_epoch": best_ep,
        "nan_batch_rate": nan_b / max(1, cfg.epochs * len(tl)),
        "dev_preds": dev_final,
        "test_preds": test_final,
    }

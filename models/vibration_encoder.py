"""
models/vibration_encoder.py
============================
Vibration-modality encoder: ResNet-18 backbone + GeM pooling +
temporal self-attention (TSA) across CWT strips.

Multi-sensor input is handled by stacking channels at the first conv layer
and redistributing the ImageNet pre-trained weights proportionally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from config import Config
from .graph import TSA


class GeM(nn.Module):
    """Generalised mean pooling (power p is learnable)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1)),
            )
            .pow(1.0 / self.p)
            .flatten(1)
        )


class VibEncoder(nn.Module):
    """Encode a sequence of ``S`` CWT strip images into a fixed-size embedding.

    Architecture: ResNet-18 → GeM → linear projection → TSA over strips.

    Parameters
    ----------
    cfg:
        Experiment config.
    pretrained:
        Whether to initialise the ResNet backbone from ImageNet weights.
    """

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.n_sensor = len(cfg.vib_sensors)

        try:
            w = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            r18 = tvm.resnet18(weights=w)
        except Exception:
            r18 = tvm.resnet18(weights=None)

        if self.n_sensor > 1:
            first = r18.conv1
            nc = nn.Conv2d(
                3 * self.n_sensor, first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                bias=False,
            )
            with torch.no_grad():
                nc.weight.copy_(
                    first.weight.repeat(1, self.n_sensor, 1, 1) / self.n_sensor
                )
            r18.conv1 = nc

        self.backbone = nn.Sequential(*list(r18.children())[:-2])
        self.pool = GeM()
        self.proj = nn.Sequential(
            nn.Linear(512, cfg.vib_embed_dim),
            nn.LayerNorm(cfg.vib_embed_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        max_len = max(cfg.vib_max_strips, cfg.vib_n_strips_eval, 8)
        self.tsa = TSA(cfg.vib_embed_dim, max_len=max_len, dropout=cfg.dropout)

    def forward(self, vib: torch.Tensor, has_vib: torch.Tensor):
        """
        Parameters
        ----------
        vib:
            Shape ``(B, S, C, H, W)``.
        has_vib:
            Shape ``(B,)`` float32; 1.0 where vibration data is available.

        Returns
        -------
        seq:
            ``(B, S, embed_dim)`` — per-strip embeddings (zeroed for missing).
        summary:
            ``(B, embed_dim)`` — TSA summary (zeroed for missing).
        attn:
            ``(B, S)`` — TSA attention weights.
        """
        B, S, C, H, W = vib.shape
        pooled = self.pool(self.backbone(vib.view(B * S, C, H, W)))
        seq = self.proj(pooled).view(B, S, -1)
        summary, attn = self.tsa(seq)
        m = has_vib.view(B, 1).to(summary.dtype)
        return seq * m.unsqueeze(-1), summary * m, attn

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

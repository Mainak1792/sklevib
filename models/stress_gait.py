"""
models/stress_gait.py
=====================
Main multi-modal stress detection model.

Supports all fusion strategies (gpc | gated | concat | late) and
modality combinations (skel | vib | both) controlled via ``Config``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .graph import build_adjacency, build_region_mask, SpatialGCN, BRSA, TSA
from .vibration_encoder import VibEncoder
from .gpc_fusion import GPCFusion


class MultiModalStressGait(nn.Module):
    """Multi-modal skeleton + vibration model for binary stress detection.

    Parameters
    ----------
    cfg:
        Controls architecture switches (modalities, fusion, use_brsa, etc.)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.fusion in ("gpc", "gated", "concat", "late")
        assert cfg.modalities in ("skel", "vib", "both")
        self.cfg = cfg

        adj = build_adjacency(cfg.num_joints)
        rmask = build_region_mask(cfg.num_joints)

        # ---- Skeleton branch ----
        self.gcn = SpatialGCN(
            cfg.joint_dim, cfg.graph_hidden, cfg.num_gcn_layers, adj, cfg.dropout
        )
        self.temporal_pool = nn.Sequential(
            nn.Linear(cfg.num_joints * cfg.graph_hidden, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        if cfg.use_brsa:
            self.brsa = BRSA(cfg.graph_hidden, rmask)
            self.brsa_proj = nn.Linear(cfg.graph_hidden, cfg.hidden_dim)
        if cfg.use_tsa_skel:
            self.skel_tsa = TSA(cfg.hidden_dim, cfg.seq_len, cfg.dropout)
        self.skel_gate = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.Sigmoid()
        )

        # ---- Vibration branch ----
        if cfg.modalities in ("vib", "both"):
            self.vib_encoder = VibEncoder(cfg, pretrained=True)

        # ---- Fusion ----
        if cfg.modalities == "both":
            if cfg.fusion == "gpc":
                self.gpc = GPCFusion(cfg.hidden_dim, K=cfg.phase_K)
            elif cfg.fusion == "gated":
                self.mm_gate = nn.Sequential(
                    nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.Sigmoid()
                )
            elif cfg.fusion == "concat":
                self.mm_proj = nn.Sequential(
                    nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                )
            # "late" requires no extra parameters

        self.mm_norm = nn.LayerNorm(cfg.hidden_dim)
        self.classifier = nn.Linear(cfg.hidden_dim, 1)
        self.projector = nn.Linear(cfg.hidden_dim, 64)

    # ------------------------------------------------------------------
    # Skeleton encoder
    # ------------------------------------------------------------------

    def _encode_skel(self, skel: torch.Tensor):
        B, T, V, C = skel.shape
        jf = self.gcn(skel)                                        # (B,T,V,hidden)
        ff = self.temporal_pool(jf.reshape(B, T, -1))              # (B,T,hidden_dim)

        rf = ra = None
        if self.cfg.use_brsa:
            rf, ra = self.brsa(jf)
            rf = self.brsa_proj(rf)

        fa = None
        if self.cfg.use_tsa_skel:
            tf, fa, fs = self.skel_tsa(ff, return_seq=True)
        else:
            tf = ff.mean(dim=1)
            fs = ff

        if rf is not None:
            g = self.skel_gate(torch.cat([rf, tf], dim=-1))
            summary = g * rf + (1 - g) * tf
        else:
            summary = tf

        return summary, fs, ra, fa

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, skel: torch.Tensor, vib: torch.Tensor, has_vib: torch.Tensor):
        """
        Parameters
        ----------
        skel:
            ``(B, T, 33, 2)`` normalised skeleton windows.
        vib:
            ``(B, S, C, H, W)`` CWT strip images (may be zero-filled).
        has_vib:
            ``(B,)`` float32 availability mask.

        Returns
        -------
        logits:
            ``(B, 1)`` un-normalised binary logits.
        proj:
            ``(B, 64)`` L2-normalised projector embeddings.
        aux:
            Dict of auxiliary tensors (attention weights, phase signals).
        """
        aux = {}
        B = skel.shape[0]
        skel_sum, skel_seq, ra, fa = self._encode_skel(skel)
        aux["region_attn"] = ra
        aux["frame_attn"] = fa

        # Modality dropout during training
        if self.training and self.cfg.modality_dropout_p > 0 and self.cfg.modalities == "both":
            drop = (torch.rand(B, device=skel.device) < self.cfg.modality_dropout_p).float()
            has_vib = has_vib * (1.0 - drop)

        # Skeleton-only
        if self.cfg.modalities == "skel":
            fused = self.mm_norm(skel_sum)
            return (
                self.classifier(fused),
                F.normalize(self.projector(fused), dim=-1),
                aux,
            )

        vib_seq, vib_sum, va = self.vib_encoder(vib, has_vib)
        aux["vib_attn"] = va

        # Vibration-only
        if self.cfg.modalities == "vib":
            fused = self.mm_norm(vib_sum)
            return (
                self.classifier(fused),
                F.normalize(self.projector(fused), dim=-1),
                aux,
            )

        # Multi-modal fusion
        if self.cfg.fusion == "gpc":
            fused, gpc_aux = self.gpc(skel_seq, vib_seq, skel, has_vib)
            aux.update({f"gpc_{k}": v for k, v in gpc_aux.items()})

        elif self.cfg.fusion == "gated":
            g = self.mm_gate(torch.cat([skel_sum, vib_sum], dim=-1))
            m = has_vib.view(-1, 1).to(g.dtype)
            eff = g * m + (1.0 - m)
            fused = eff * skel_sum + (1 - eff) * vib_sum

        elif self.cfg.fusion == "concat":
            m = has_vib.view(-1, 1).to(vib_sum.dtype)
            fused = self.mm_proj(torch.cat([skel_sum, vib_sum * m], dim=-1))

        else:  # late
            fused = 0.5 * (skel_sum + vib_sum)

        fused = self.mm_norm(fused)
        return (
            self.classifier(fused),
            F.normalize(self.projector(fused), dim=-1),
            aux,
        )

    def load_pretrained_gcn(self, state_dict: dict) -> None:
        """Load GCN weights from SSL pre-training."""
        self.gcn.load_state_dict(state_dict)

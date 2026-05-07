"""
losses/losses.py
================
Loss functions used during training.

FocalLoss:
    Class-balanced focal loss for binary classification.
SupConLoss:
    Supervised contrastive loss (within-batch positives).
SISCLoss:
    Subject-invariant supervised contrastive loss with a rolling memory
    bank; positives must be same-class but different-subject, preventing
    the network from trivially encoding subject identity.
LossBundle:
    Convenience wrapper that computes and returns all active loss terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.gpc_fusion import phase_consistency_loss


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        self.a = alpha
        self.g = gamma

    def forward(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        p = torch.sigmoid(logits)
        pt = tgt * p + (1 - tgt) * (1 - p)
        at = tgt * self.a + (1 - tgt) * (1 - self.a)
        return (at * (1 - pt) ** self.g * bce).mean()


# ---------------------------------------------------------------------------
# Supervised contrastive loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.t = temperature

    def forward(self, f: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        B = f.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=f.device)
        sim = f @ f.T / self.t
        mask = (l.unsqueeze(0) == l.unsqueeze(1)).float()
        eye = torch.eye(B, device=f.device)
        mask *= 1 - eye
        sim = sim - sim.max(1, keepdim=True).values.detach()
        exp = torch.exp(sim) * (1 - eye)
        lp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-8)
        pos = mask.sum(1)
        valid = pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=f.device)
        return -((mask * lp).sum(1) / (pos + 1e-8))[valid].mean()


# ---------------------------------------------------------------------------
# Subject-invariant supervised contrastive loss
# ---------------------------------------------------------------------------

class SISCLoss(nn.Module):
    """Subject-Invariant Supervised Contrastive loss.

    Uses a rolling memory bank so that, within the positive pairs, both
    embeddings must come from *different subjects*.  This prevents the
    network from encoding subject-specific appearance rather than stress.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        bank_size: int = 256,
        dim: int = 64,
    ) -> None:
        super().__init__()
        self.t = temperature
        self.bs = bank_size
        self.register_buffer("fb", F.normalize(torch.randn(bank_size, dim), dim=-1))
        self.register_buffer("lb", torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer("sb", torch.zeros(bank_size, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _push(self, f: torch.Tensor, l: torch.Tensor, s: torch.Tensor) -> None:
        B = f.shape[0]
        p = int(self.ptr)
        if p + B > self.bs:
            B = self.bs - p
        self.fb[p: p + B] = f[:B].detach()
        self.lb[p: p + B] = l[:B].detach()
        self.sb[p: p + B] = s[:B].detach()
        self.ptr[0] = (p + B) % self.bs

    def forward(
        self,
        f: torch.Tensor,
        l: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        B = f.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=f.device)
        af = torch.cat([f, self.fb], 0)
        al = torch.cat([l, self.lb], 0)
        asj = torch.cat([s, self.sb], 0)
        N = af.shape[0]

        sim = torch.clamp(f @ af.T / self.t, -50, 50)
        sml = (l.unsqueeze(1) == al.unsqueeze(0)).float()
        dfs = (s.unsqueeze(1) != asj.unsqueeze(0)).float()

        sm = torch.zeros(B, N, device=f.device)
        sm[:, :B] = torch.eye(B, device=f.device)
        ns = 1 - sm

        pm = sml * dfs * ns
        nm = (1 - sml) * ns

        sim = sim - sim.max(1, keepdim=True).values.detach()
        dn = (torch.exp(sim) * (pm + nm)).sum(1, keepdim=True) + 1e-8
        lp = sim - torch.log(dn)

        pos = pm.sum(1)
        valid = pos > 0
        self._push(f, l, s)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=f.device)
        return -((pm * lp).sum(1) / (pos + 1e-8))[valid].mean()


# ---------------------------------------------------------------------------
# Loss bundle
# ---------------------------------------------------------------------------

class LossBundle:
    """Compute and aggregate all training losses.

    Parameters
    ----------
    cfg:
        Active config; controls which terms are enabled and their weights.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.focal = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
        self.supcon = SupConLoss(cfg.supcon_temp)
        self.sisc = SISCLoss(cfg.supcon_temp, 256, 64) if cfg.use_sisc else None

    def to(self, device) -> "LossBundle":
        if self.sisc is not None:
            self.sisc = self.sisc.to(device)
        return self

    def __call__(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        proj: torch.Tensor,
        sid: torch.Tensor,
        aux: dict,
    ):
        c = self.cfg
        parts = {}
        parts["focal"] = self.focal(logits, label)
        parts["supcon"] = c.supcon_weight * self.supcon(proj, label.long())
        if self.sisc is not None:
            parts["sisc"] = c.sisc_weight * self.sisc(proj, label.long(), sid)
        if (
            c.modalities == "both"
            and c.fusion == "gpc"
            and c.use_phase_loss
            and "gpc_alpha_s" in aux
            and "gpc_alpha_v" in aux
        ):
            has_vib = (aux.get("vib_attn", torch.zeros(1)).abs().sum(-1) > 0).float()
            parts["phase"] = c.phase_loss_weight * phase_consistency_loss(
                aux["gpc_alpha_s"], aux["gpc_alpha_v"], has_vib
            )
        total = sum(parts.values())
        return total, {k: float(v.detach()) for k, v in parts.items()}

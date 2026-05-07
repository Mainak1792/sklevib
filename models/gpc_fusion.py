"""
models/gpc_fusion.py
====================
Gait-Phase-Coupled (GPC) Fusion — the core novel contribution.

Architecture overview
---------------------
1. **DifferentiableGaitPhaseEncoder**: estimates continuous gait phase
   φ(t) ∈ [0, 1) from ankle kinematics using a differentiable phase-plane
   approach. No peak detection; speed-invariant by construction.

2. **Soft phase binning**: features from both modalities are scattered into
   K learnable phase bins using a triangle kernel (``phase_scatter_mean``).

3. **PhaseCoupledCrossAttention**: cross-modality gating conditioned on the
   per-phase representations, with graceful degradation when vibration data
   is absent (``has_vib`` mask).

4. **GPCFusion**: wires the above components. The vibration strips are mapped
   to a pseudo-phase via a small learnable warp network so they align with
   the skeletal gait cycle.

5. **phase_consistency_loss**: JS-divergence between skeleton and vibration
   phase-attention distributions, used as an auxiliary training signal.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable gait-phase encoder
# ---------------------------------------------------------------------------

class DifferentiableGaitPhaseEncoder(nn.Module):
    """Continuous gait phase φ(t) ∈ [0, 1) from ankle position / velocity.

    The phase plane of each ankle's vertical trajectory is computed via
    finite differences and lightly smoothed. Both ankles are combined into
    a single circular mean to yield a single phase signal.

    Parameters
    ----------
    kernel:
        Gaussian smoothing half-width (frames).
    """

    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self, kernel: int = 9) -> None:
        super().__init__()
        self.smooth = nn.Conv1d(
            2, 2, kernel_size=kernel, padding=kernel // 2, groups=2, bias=False
        )
        with torch.no_grad():
            x = torch.arange(kernel, dtype=torch.float) - kernel // 2
            g = torch.exp(-(x ** 2) / (2 * (kernel / 4) ** 2))
            g = g / g.sum()
            self.smooth.weight.copy_(
                g.view(1, 1, -1).expand(2, 1, -1).contiguous()
            )

    def forward(self, skel_xy: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        skel_xy:
            Shape ``(B, T, 33, 2)`` — hip-centred, torso-normalised skeleton.

        Returns
        -------
        phase:
            Shape ``(B, T)`` with values in ``[0, 1)``.
        """
        B, T, V, C = skel_xy.shape
        y_l = skel_xy[:, :, self.LEFT_ANKLE, 1]
        y_r = skel_xy[:, :, self.RIGHT_ANKLE, 1]

        vy_l = torch.zeros_like(y_l)
        vy_r = torch.zeros_like(y_r)
        vy_l[:, 1:-1] = (y_l[:, 2:] - y_l[:, :-2]) * 0.5
        vy_r[:, 1:-1] = (y_r[:, 2:] - y_r[:, :-2]) * 0.5
        vy_l[:, 0] = y_l[:, 1] - y_l[:, 0]
        vy_l[:, -1] = y_l[:, -1] - y_l[:, -2]
        vy_r[:, 0] = y_r[:, 1] - y_r[:, 0]
        vy_r[:, -1] = y_r[:, -1] - y_r[:, -2]

        v = torch.stack([vy_l, vy_r], dim=1)
        v = self.smooth(v)
        vy_l, vy_r = v[:, 0], v[:, 1]

        yc_l = y_l - y_l.mean(dim=-1, keepdim=True)
        yc_r = y_r - y_r.mean(dim=-1, keepdim=True)
        theta_l = torch.atan2(vy_l, yc_l)
        theta_r = torch.atan2(vy_r, yc_r) + math.pi

        z = 0.5 * (
            torch.complex(torch.cos(theta_l), torch.sin(theta_l))
            + torch.complex(torch.cos(theta_r), torch.sin(theta_r))
        )
        angle = torch.atan2(z.imag, z.real)
        phase = (angle / (2.0 * math.pi)) + 0.5
        return phase.clamp(0.0, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# Phase scatter helpers
# ---------------------------------------------------------------------------

def soft_phase_bins(
    phase: torch.Tensor, K: int, bandwidth: float = 1.0
) -> torch.Tensor:
    """Triangle-kernel soft assignment of frames to ``K`` gait-phase bins.

    Parameters
    ----------
    phase:
        Shape ``(B, T)`` ∈ [0, 1).
    K:
        Number of phase bins.

    Returns
    -------
    weights:
        Shape ``(B, T, K)`` row-normalised.
    """
    centres = (torch.arange(K, device=phase.device, dtype=phase.dtype) + 0.5) / K
    d = (phase.unsqueeze(-1) - centres.view(1, 1, K)).abs()
    d = torch.minimum(d, 1.0 - d)
    r = bandwidth / K
    w = (1.0 - d / r).clamp(min=0.0)
    return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def phase_scatter_mean(
    x: torch.Tensor, phase: torch.Tensor, K: int, bandwidth: float = 1.0
) -> torch.Tensor:
    """Scatter-mean features into K phase bins.

    Parameters
    ----------
    x:
        Shape ``(B, T, C)``.
    phase:
        Shape ``(B, T)``.

    Returns
    -------
    binned:
        Shape ``(B, K, C)``.
    """
    W = soft_phase_bins(phase, K, bandwidth)
    num = torch.einsum("btk,btc->bkc", W, x)
    den = W.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
    return num / den


# ---------------------------------------------------------------------------
# Phase-coupled cross-attention
# ---------------------------------------------------------------------------

class PhaseCoupledCrossAttention(nn.Module):
    """Cross-modal gate conditioned on per-phase representations.

    When vibration is unavailable (``has_vib`` = 0), the gate degrades
    gracefully to identity on the skeleton branch.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn_s = nn.Linear(dim, 1)
        self.attn_v = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        skel_phase: torch.Tensor,
        vib_phase: torch.Tensor,
        has_vib: torch.Tensor,
    ):
        B, K, C = skel_phase.shape
        alpha_s = F.softmax(self.attn_s(skel_phase).squeeze(-1), dim=-1)
        alpha_v = F.softmax(self.attn_v(vib_phase).squeeze(-1), dim=-1)
        g = self.gate(torch.cat([skel_phase, vib_phase], dim=-1))
        m = has_vib.view(B, 1, 1).to(g.dtype)
        g = g * m + (1.0 - m)
        fused_phase = g * skel_phase + (1.0 - g) * vib_phase
        fused_phase = self.norm(fused_phase)
        fused = torch.einsum("bk,bkc->bc", alpha_s, fused_phase)
        return fused, alpha_s, alpha_v


# ---------------------------------------------------------------------------
# GPC Fusion module
# ---------------------------------------------------------------------------

class GPCFusion(nn.Module):
    """Full Gait-Phase-Coupled fusion block.

    Parameters
    ----------
    dim:
        Feature dimension shared by both modalities.
    K:
        Number of gait-phase bins.
    bandwidth:
        Triangle-kernel bandwidth (in units of 1/K).
    """

    def __init__(self, dim: int, K: int = 16, bandwidth: float = 1.0) -> None:
        super().__init__()
        self.K = K
        self.bw = bandwidth
        self.phase_encoder = DifferentiableGaitPhaseEncoder()
        self.attn = PhaseCoupledCrossAttention(dim)
        self.vib_phase_warp = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def _vib_strip_phase(self, B: int, S: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0, 1, S, device=device).view(S, 1)
        phi = self.vib_phase_warp(t).squeeze(-1)
        phi = torch.cumsum(F.softmax(phi, dim=0), dim=0)
        phi = phi / phi[-1].clamp(min=1e-6)
        return phi.unsqueeze(0).expand(B, S).contiguous().clamp(0.0, 1.0 - 1e-6)

    def forward(
        self,
        skel_seq: torch.Tensor,
        vib_seq: torch.Tensor,
        skel_raw: torch.Tensor,
        has_vib: torch.Tensor,
    ):
        """
        Parameters
        ----------
        skel_seq:
            ``(B, T, dim)`` — temporal skeleton features.
        vib_seq:
            ``(B, S, dim)`` — strip-level vibration features.
        skel_raw:
            ``(B, T, 33, 2)`` — raw skeleton for phase estimation.
        has_vib:
            ``(B,)`` availability mask.

        Returns
        -------
        fused:
            ``(B, dim)`` fused embedding.
        aux:
            Dict with ``alpha_s``, ``alpha_v``, ``phase_skel``, ``phase_vib``.
        """
        B, T, C = skel_seq.shape
        S = vib_seq.shape[1]

        phase_skel = self.phase_encoder(skel_raw)
        phase_vib = self._vib_strip_phase(B, S, skel_seq.device)

        skel_phase = phase_scatter_mean(skel_seq, phase_skel, self.K, self.bw)
        vib_phase = phase_scatter_mean(vib_seq, phase_vib, self.K, self.bw)

        fused, a_s, a_v = self.attn(skel_phase, vib_phase, has_vib)
        return fused, {
            "alpha_s": a_s,
            "alpha_v": a_v,
            "phase_skel": phase_skel,
            "phase_vib": phase_vib,
        }


# ---------------------------------------------------------------------------
# Phase consistency loss
# ---------------------------------------------------------------------------

def phase_consistency_loss(
    a_s: torch.Tensor,
    a_v: torch.Tensor,
    has_vib: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """JS-divergence between per-phase attention distributions.

    Only computed for samples where ``has_vib > 0.5``.
    """
    p = a_s.clamp(min=eps)
    q = a_v.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(-1)
    kl_qm = (q * (q.log() - m.log())).sum(-1)
    js = 0.5 * (kl_pm + kl_qm)
    valid = has_vib > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=a_s.device)
    return js[valid].mean()

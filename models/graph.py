"""
models/graph.py
===============
Skeleton-branch backbone modules.

Components
----------
SpatialGCN:
    Stack of graph convolution layers over the MediaPipe 33-joint skeleton.
BRSA (Body-Region Spatial Attention):
    Aggregates joint features into anatomical regions and scores their
    relevance for the current clip.
TSA (Temporal Self-Attention):
    Transformer-style encoder that summarises a sequence of frame-level
    feature vectors into a single embedding.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe 33-joint kinematic skeleton edges
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]

BODY_REGIONS = {
    "head":      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "shoulders": [11, 12],
    "arms":      [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    "torso":     [11, 12, 23, 24],
    "legs":      [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
}


def build_adjacency(V: int = 33) -> torch.Tensor:
    """Symmetric normalised adjacency matrix for the 33-joint skeleton."""
    A = torch.zeros(V, V)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += torch.eye(V)
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / (D + 1e-8).sqrt())
    return D_inv_sqrt @ A @ D_inv_sqrt


def build_region_mask(V: int = 33) -> torch.Tensor:
    """Row-normalised ``(R, V)`` mask for ``R`` body regions."""
    M = torch.zeros(len(BODY_REGIONS), V)
    for i, (_, ids) in enumerate(BODY_REGIONS.items()):
        for j in ids:
            M[i, j] = 1.0
    return M / (M.sum(dim=1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------------------
# Graph convolution layer
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adj", adj.clone())
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C)
        B, T, V, C = x.shape
        x_flat = x.reshape(B * T, V, C)
        xg = torch.matmul(self.adj, x_flat).reshape(B * T * V, C)
        h = self.fc(xg).reshape(B * T, V, -1)
        h = self.bn(h.permute(0, 2, 1)).permute(0, 2, 1)
        return F.relu(h).reshape(B, T, V, -1)


# ---------------------------------------------------------------------------
# Spatial GCN
# ---------------------------------------------------------------------------

class SpatialGCN(nn.Module):
    """Stack of ``n_layers`` graph convolutions."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        n_layers: int,
        adj: torch.Tensor,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dims = [in_channels] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            GraphConv(dims[i], dims[i + 1], adj) for i in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.drop(layer(x))
        return x


# ---------------------------------------------------------------------------
# Body-Region Spatial Attention (BRSA)
# ---------------------------------------------------------------------------

class BRSA(nn.Module):
    """Soft-attention pooling over anatomical body regions.

    Aggregates joint features ``(B, T, V, C)`` → ``(B, C)`` using a
    learnable per-region importance score.
    """

    def __init__(self, dim: int, region_mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("region_mask", region_mask.clone())
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, V, C)
        rf = torch.einsum("btvc,rv->btrc", x, self.region_mask)  # (B,T,R,C)
        rt = rf.mean(dim=1)                                       # (B,R,C)
        w = F.softmax(self.attn(rt).squeeze(-1), dim=-1)          # (B,R)
        return torch.einsum("br,brc->bc", w, rt), w               # (B,C), (B,R)


# ---------------------------------------------------------------------------
# Temporal Self-Attention (TSA)
# ---------------------------------------------------------------------------

class TSA(nn.Module):
    """Single-head temporal self-attention encoder with sinusoidal PE."""

    def __init__(self, dim: int, max_len: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        if dim > 1:
            pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, return_seq: bool = False):
        B, L, C = x.shape
        x = x + self.pe[:, :L, :]
        q, k, v = self.q(x), self.k(x), self.v(x)
        a = F.softmax(q @ k.transpose(1, 2) / self.scale, dim=-1)
        h = self.norm(x + a @ v)
        h = h + self.ffn(h)
        w = F.softmax(self.scorer(h).squeeze(-1), dim=-1)         # (B,L)
        summary = torch.einsum("bl,blc->bc", w, h)                # (B,C)
        return (summary, w, h) if return_seq else (summary, w)

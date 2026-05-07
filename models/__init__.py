"""models package — backbone networks, novel GPC fusion, and main model."""
from .graph import build_adjacency, build_region_mask, SpatialGCN, BRSA, TSA
from .vibration_encoder import VibEncoder
from .gpc_fusion import (
    DifferentiableGaitPhaseEncoder,
    GPCFusion,
    phase_consistency_loss,
)
from .stress_gait import MultiModalStressGait

__all__ = [
    "build_adjacency",
    "build_region_mask",
    "SpatialGCN",
    "BRSA",
    "TSA",
    "VibEncoder",
    "DifferentiableGaitPhaseEncoder",
    "GPCFusion",
    "phase_consistency_loss",
    "MultiModalStressGait",
]

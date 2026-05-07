"""
config.py
=========
Central configuration dataclass for StressGait-MM experiments.
Edit ONLY this file to set data paths and hyperparameters;
all other modules import ``Config`` and respect it.
"""

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data paths  — set these to match your local dataset layout
    # ------------------------------------------------------------------
    skeleton_root: str = "data/raw/skeletons"
    """Root directory: <skeleton_root>/<view>/<condition>/<subject>/skeleton.npy"""

    vib_root: str = "data/raw/vibration"
    """Root directory: <vib_root>/<sensor>/<condition>/<subject>/*.png"""

    subject_xlsx_path: Optional[str] = "data/raw/subject_metadata.xlsx"
    """Optional Excel file mapping subject names to numeric IDs.
    Set to None to skip; subjects will then be keyed by folder name only."""

    subject_xlsx_name_col: str = "Person_Name"
    """Column in the Excel file containing subject display names."""

    subject_xlsx_prefix: str = "person_"
    """Prefix prepended to the 1-based row index when building person IDs."""

    output_root: str = "outputs"
    """All run artefacts (checkpoints, parquets, figures, tables) land here."""

    # ------------------------------------------------------------------
    # Dataset semantics
    # ------------------------------------------------------------------
    views: Tuple[str, ...] = ("farside", "middle", "nearside")
    conditions_stress: Tuple[str, ...] = ("oral",)
    conditions_normal: Tuple[str, ...] = ("normal",)
    conditions_covariate: Tuple[str, ...] = ("bag",)
    use_covariates: bool = False

    folder_aliases: Dict[str, str] = field(
        default_factory=lambda: {"oral": "cog"}
    )
    """Maps internal condition name → on-disk folder name when they differ.
    Applied symmetrically on both modalities."""

    # ------------------------------------------------------------------
    # Vibration modality
    # ------------------------------------------------------------------
    vib_sensors: Tuple[str, ...] = ("channel_1", "channel_2")
    vib_img_size: Tuple[int, int] = (224, 224)
    vib_cache_n: int = 10
    """Max CWT strips to cache per (subject, condition, sensor) bucket."""
    vib_max_strips: int = 8
    vib_n_strips_train: int = 4
    vib_n_strips_eval: int = 4

    # ------------------------------------------------------------------
    # Skeleton modality
    # ------------------------------------------------------------------
    num_joints: int = 33
    joint_dim: int = 2
    seq_len: int = 64
    """Window length in frames after temporal resampling."""
    window_stride: int = 32
    """Stride between consecutive windows (in raw frames)."""
    min_frames: int = 30
    """Minimum raw clip/window length; shorter clips are discarded."""

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    hidden_dim: int = 96
    vib_embed_dim: int = 96
    num_gcn_layers: int = 2
    graph_hidden: int = 32
    dropout: float = 0.4

    # Ablation switches (overridden by the experiment grid)
    modalities: str = "both"   # 'skel' | 'vib' | 'both'
    fusion: str = "gpc"        # 'gpc' | 'gated' | 'concat' | 'late'
    use_brsa: bool = True
    use_tsa_skel: bool = True
    use_ssl_pretrain: bool = True
    use_phase_loss: bool = True
    phase_K: int = 16

    # ------------------------------------------------------------------
    # Activity & quality filtering
    # ------------------------------------------------------------------
    use_activity_filter: bool = True
    min_hip_speed_torso_per_sec: float = 0.3
    """Minimum mean hip speed in torso-lengths/sec to consider a window
    as 'actively walking'.  Calibrated at fps=20 on the target corpus."""
    min_window_valid_frac: float = 0.9
    """Fraction of frames with detectable torso joints required per window."""

    # ------------------------------------------------------------------
    # Unidirectional walking segmentation
    # ------------------------------------------------------------------
    use_unidirectional_segmentation: bool = True
    turn_vel_smooth_frames: int = 15
    turn_min_pass_frames: int = 40
    max_direction_changes_per_window: int = 1

    # ------------------------------------------------------------------
    # Subject-level balancing
    # ------------------------------------------------------------------
    require_both_conditions_per_subject: bool = True
    min_windows_per_condition_per_subject: int = 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 20
    warmup_epochs: int = 5
    vib_backbone_freeze_epochs: int = 3
    grad_clip: float = 1.0

    # ------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    supcon_weight: float = 0.5
    supcon_temp: float = 0.05
    sisc_weight: float = 0.3
    use_sisc: bool = True
    phase_loss_weight: float = 0.2
    modality_dropout_p: float = 0.15

    # ------------------------------------------------------------------
    # Experiment bookkeeping
    # ------------------------------------------------------------------
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    protocol: str = "P1"
    run_tag: str = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def hash(self) -> str:
        d = asdict(self)
        for k in ("output_root", "run_tag"):
            d.pop(k, None)
        return hashlib.sha1(
            json.dumps(d, sort_keys=True, default=str).encode()
        ).hexdigest()[:10]

    def summary(self) -> str:
        return (
            f"{self.protocol}|{self.fusion}/{self.modalities}"
            f"|seed={self.seed}|{self.hash()}"
        )

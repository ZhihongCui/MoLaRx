"""
Configuration for VL Pipeline: LFM2-VL-1.6B Image-based OOD Generalization
============================================================================

Same SCM framework as text pipeline, but with image input:
  City → Visual Features → Hidden States → Decision
  Distance, Velocity → Decision (causal path)

Language backbone = LFM2-1.2B (same as text experiment, 16 layers).
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set

# ──────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────
VL_MODEL_ID = "LiquidAI/LFM2-VL-1.6B"
NUSCENES_DATAROOT = "data/nuscenes"
NUSCENES_VERSION = "v1.0-trainval"

OUTPUT_DIR = Path("lfm2_vl_pipeline/outputs")
DATA_DIR = OUTPUT_DIR / "data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ──────────────────────────────────────────────────────────────
#  LFM2 Language Backbone Architecture (same as text experiment)
# ──────────────────────────────────────────────────────────────
NUM_LAYERS = 16
HIDDEN_DIM = 2048
GQA_LAYER_INDICES: Set[int] = {2, 5, 8, 10, 12, 14}

LAYER_TYPES = {}
for i in range(NUM_LAYERS):
    LAYER_TYPES[i] = "GQA" if i in GQA_LAYER_INDICES else "LIV"

LIV_LAYER_INDICES = [i for i in range(NUM_LAYERS) if i not in GQA_LAYER_INDICES]
GQA_LAYER_LIST = sorted(GQA_LAYER_INDICES)

# ──────────────────────────────────────────────────────────────
#  City-based OOD Split
#  REVERSED: train on Boston, test OOD on Singapore
#  (ablation to verify domain gap is real and bidirectional)
# ──────────────────────────────────────────────────────────────
ID_CITY = "boston"
OOD_CITY = "singapore"

ID_LOCATIONS = [
    "boston-seaport",
]
OOD_LOCATIONS = [
    "singapore-onenorth",
    "singapore-queenstown",
    "singapore-hollandvillage",
]

# ──────────────────────────────────────────────────────────────
#  Camera
# ──────────────────────────────────────────────────────────────
CAMERA_CHANNEL = "CAM_FRONT"

# ──────────────────────────────────────────────────────────────
#  Vulnerability Map (same as text pipeline)
# ──────────────────────────────────────────────────────────────
VULNERABILITY_MAP = {
    "human.pedestrian.adult": "high",
    "human.pedestrian.child": "high",
    "human.pedestrian.construction_worker": "high",
    "human.pedestrian.personal_mobility": "high",
    "human.pedestrian.police_officer": "high",
    "human.pedestrian.stroller": "high",
    "human.pedestrian.wheelchair": "high",
    "animal": "high",
    "vehicle.bicycle": "medium",
    "vehicle.motorcycle": "medium",
    "vehicle.car": "low",
    "vehicle.truck": "low",
    "vehicle.bus.bendy": "low",
    "vehicle.bus.rigid": "low",
    "vehicle.trailer": "low",
    "vehicle.construction": "low",
    "vehicle.emergency.ambulance": "low",
    "vehicle.emergency.police": "low",
    "movable_object.barrier": "none",
    "movable_object.trafficcone": "none",
    "movable_object.pushable_pullable": "none",
    "movable_object.debris": "none",
    "static_object.bicycle_rack": "none",
}

WEATHER_KEYWORDS = {
    "rain": ["rain", "rainy", "after rain", "wet"],
    "night": ["night", "dark", "darkness", "evening", "dusk"],
    "fog": ["fog", "foggy", "haze", "hazy"],
    "clear": ["clear", "sunny", "day", "daytime"],
}
ALL_WEATHER_LABELS = ["clear", "rain", "night", "fog"]

# ──────────────────────────────────────────────────────────────
#  Decision Labels
# ──────────────────────────────────────────────────────────────
DECISION_LABELS = {0: "accelerate", 1: "maintain", 2: "decelerate"}
DECISION_KEYWORDS = {
    0: ["accelerate", "speed up", "increase speed"],
    1: ["maintain", "keep speed", "current speed", "steady speed"],
    2: ["decelerate", "slow down", "reduce speed", "brake", "braking"],
}

# ──────────────────────────────────────────────────────────────
#  Hyperparameters
# ──────────────────────────────────────────────────────────────

@dataclass
class VLConfig:
    """Hyperparameters for the VL pipeline."""

    # Model
    model_id: str = VL_MODEL_ID
    max_seq_len: int = 1024  # longer than text pipeline (image tokens)

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training (Phase 0)
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data split
    # Note: test_id size = (1 - train_ratio - val_ratio) * len(id_records)
    # Aim for ~3000+ ID test samples so OOD gap estimate is statistically reliable.
    # With ~13700 Singapore keyframes: 70/7/23 → train≈9590, val≈959, test_id≈3151
    train_ratio: float = 0.70
    val_ratio: float = 0.07
    test_ratio: float = 0.23   # remainder goes to test_id
    seed: int = 42

    # OOD test set size cap: sample at most this many Boston frames so that
    # |test_ood| ≈ |test_id|, making the ID-OOD gap comparison fair.
    # Set to -1 to use all Boston samples (old behaviour).
    test_ood_max_samples: int = 3500  # Boston ID test ≈ 3500; cap OOD to match

    # Ego acceleration threshold
    auto_threshold: bool = True
    accel_threshold: float = 0.5  # fallback if auto_threshold=False

    # Phase 1: CIS
    cis_threshold: float = 1.0
    cis_epsilon: float = 0.1

    # Phase 2: Domain Alignment
    lambda_da: float = 0.5
    mmd_bandwidths: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])

    # Phase 2 strategies
    strategies: List[str] = field(default_factory=lambda: [
        "cis_selective_lora",   # LoRA only on phi_low layers + L_DA
        "full_lora_da",         # LoRA on all layers + L_DA (ablation)
        "random_selective_lora", # LoRA on random layers + L_DA (ablation)
        "full_lora_no_da",      # LoRA on all layers, no L_DA (ablation)
    ])

    # Output
    output_dir: str = str(OUTPUT_DIR)
    logging_steps: int = 10
    save_strategy: str = "epoch"

    # Evaluation
    max_new_tokens: int = 100

"""
Configuration for Qwen2-VL Pipeline (Cross-architecture comparison)
===================================================================

Same task and data as LFM2-VL pipeline, but using Qwen2-VL-2B-Instruct
as the backbone. This serves as a control: Qwen has no LTC-derived
causal layers — all 28 layers are standard transformer attention.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set

# ──────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
NUSCENES_DATAROOT = "data/nuscenes"

# Reuse the same data as LFM2 pipeline
LFM2_DATA_DIR = Path("lfm2_vl_pipeline/outputs/data")

OUTPUT_DIR = Path("qwen_vl_pipeline/outputs")
DATA_DIR = LFM2_DATA_DIR  # Same data splits
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ──────────────────────────────────────────────────────────────
#  Qwen2-VL Architecture
# ──────────────────────────────────────────────────────────────
NUM_LAYERS = 28
HIDDEN_DIM = 1536

# All layers are standard attention — no LIV/GQA distinction
LAYER_TYPES = {i: "ATTN" for i in range(NUM_LAYERS)}

# ──────────────────────────────────────────────────────────────
#  City-based OOD Split (same as LFM2)
# ──────────────────────────────────────────────────────────────
ID_CITY = "boston"
OOD_CITY = "singapore"

ALL_WEATHER_LABELS = ["clear", "rain", "night", "fog"]

# ──────────────────────────────────────────────────────────────
#  Hyperparameters
# ──────────────────────────────────────────────────────────────

@dataclass
class QwenVLConfig:
    """Hyperparameters for the Qwen VL pipeline."""

    # Model
    model_id: str = QWEN_MODEL_ID

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training (Phase 0)
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data
    seed: int = 42

    # Evaluation
    max_new_tokens: int = 100

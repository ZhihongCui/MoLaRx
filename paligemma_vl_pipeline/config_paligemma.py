"""Config for PaliGemma2-3B pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PALI_MODEL_ID = "google/paligemma2-3b-mix-224"
NUM_LAYERS = 26          # Gemma2 decoder layers
HIDDEN_DIM = 2304
IMAGE_TOKEN_ID = 257152  # PaliGemma <image> token

NUS_DATA_DIR = Path("lfm2_vl_pipeline/outputs/data")
AV2_DATA_DIR = Path("argoverse2_pipeline/outputs/data")

OUTPUT_DIR     = Path("paligemma_vl_pipeline/outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

TARGET_IMAGE_SIZE = (224, 224)

INSTRUCTION = ("Predict future trajectory as 6 waypoints (x,y) "
               "in ego-frame metres at t+0.5s to t+3.0s.")
TEMPORAL_PREFIX = ("Three consecutive driving camera frames "
                   "(t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")


@dataclass
class PaliGemmaConfig:
    model_id: str = PALI_MODEL_ID
    max_seq_len: int = 1200

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True

    train_ratio: float = 0.70
    val_ratio: float = 0.07
    test_ratio: float = 0.23
    test_ood_max_samples: int = 3500
    seed: int = 42

    cis_epsilon: float = 0.1
    num_paired_samples: int = 100

    lambda_mmd: float = 0.5
    mmd_bandwidths: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    cis_threshold_img: float = 0.30
    cis_threshold_txt: float = 2.00

    output_dir: str = str(OUTPUT_DIR)
    logging_steps: int = 10
    save_strategy: str = "epoch"
    max_new_tokens: int = 120

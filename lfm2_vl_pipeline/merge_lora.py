"""
Merge LoRA checkpoint into base model for Phase 1/2 use.
=========================================================

After Phase 0 fine-tuning, the LoRA adapter is saved as a checkpoint.
This script merges the LoRA weights into the base model and saves the
merged model to `phase0_vl_merged/` for use by Phase 1 and Phase 2.

Usage:
  python lfm2_vl_pipeline/merge_lora.py [--checkpoint PATH]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig,
    VL_MODEL_ID,
    CHECKPOINT_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint directory")
    args = parser.parse_args()

    cfg = VLConfig()

    # Find checkpoint
    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
    else:
        lora_dir = CHECKPOINT_DIR / "phase0_vl_lora"
        ckpts = sorted(lora_dir.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[1]))
        if not ckpts:
            print("[Merge] No checkpoints found!")
            sys.exit(1)
        ckpt_dir = ckpts[-1]  # Latest checkpoint

    print(f"\n{'=' * 60}")
    print(f"  Merging LoRA Checkpoint")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Base model: {VL_MODEL_ID}")

    # Load base model
    print(f"\n[Merge] Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        VL_MODEL_ID,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(VL_MODEL_ID)

    # Load LoRA adapter
    print(f"[Merge] Loading LoRA adapter from {ckpt_dir}...")
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir))

    # Merge
    print(f"[Merge] Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # Save
    merged_dir = CHECKPOINT_DIR / "phase0_vl_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Merge] Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))

    print(f"\n  Merged model saved to: {merged_dir}")
    print(f"  You can now run Phase 1 and Phase 2.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
Visual Grounding (Challenge 1) — Temporal Frame Pairs + Curriculum Text Masking
================================================================================

Problem: The Phase 0 model ignores images entirely (image ablation: 0.3% ADE change).
Goal: Force the VLM to actually use visual information for trajectory prediction.

Approach:
  1. Temporal Frame Pairs: Input 3 consecutive frames (t-0.5s, t, t+0.5s)
     so the model has motion information only available from images.
  2. Curriculum Text Masking: Progressively remove text features across epochs
     to force the model to rely on visual cues.
       - Epoch 1: Full text + 3 images (learn to process multi-image)
       - Epoch 2: Mask lead_velocity and front_distance (keep ego_speed)
       - Epoch 3: Mask all numerical features (only keep task instruction)

Base model: Phase 0 merged checkpoint (already fine-tuned on single-image).

Usage:
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/visual_grounding_finetune.py
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


# ════════════════════════════════════════════════════════════════
#  Text Masking Utilities
# ════════════════════════════════════════════════════════════════

def mask_text_features(question: str, mask_level: int) -> str:
    """
    Progressively mask text features in the question.

    mask_level:
      0 = no masking (full text)
      1 = mask lead_velocity and front_distance
      2 = mask all numerical features (ego_speed, front_distance, lead_velocity)
    """
    if mask_level == 0:
        return question

    if mask_level == 1:
        # Remove front_distance and lead_velocity, keep ego_speed
        question = re.sub(r',?\s*front_distance=[\d.]+ m', '', question)
        question = re.sub(r',?\s*lead_velocity=[\-\d.]+ m/s', '', question)
        # Clean up any leading comma/space
        question = re.sub(r'^\s*,\s*', '', question)
        question = re.sub(r'\.\s*,', '.', question)
        return question

    if mask_level >= 2:
        # Remove all numerical features, keep only the instruction
        instruction = "Predict future trajectory as 6 waypoints (x,y) in ego-frame metres at t+0.5s to t+3.0s."
        return instruction

    return question


# ════════════════════════════════════════════════════════════════
#  Temporal Frame Lookup
# ════════════════════════════════════════════════════════════════

def build_temporal_map(nusc, dataroot):
    """
    Build a mapping: sample_token -> (prev_image_path, curr_image_path, next_image_path).
    Pre-compute to avoid repeated nuScenes lookups during training.
    """
    temporal_map = {}
    for sample in nusc.sample:
        token = sample['token']
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        curr_path = os.path.join(dataroot, cam_data['filename'])

        prev_path = curr_path  # fallback: duplicate current
        if sample['prev']:
            try:
                prev_sample = nusc.get('sample', sample['prev'])
                prev_cam = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])
                prev_path = os.path.join(dataroot, prev_cam['filename'])
            except Exception:
                pass

        next_path = curr_path  # fallback: duplicate current
        if sample['next']:
            try:
                next_sample = nusc.get('sample', sample['next'])
                next_cam = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
                next_path = os.path.join(dataroot, next_cam['filename'])
            except Exception:
                pass

        temporal_map[token] = (prev_path, curr_path, next_path)

    return temporal_map


# ════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════

class TemporalVLDataset(Dataset):
    """Dataset with 3-frame temporal input and curriculum text masking."""

    def __init__(self, jsonl_path: str, processor, temporal_map: dict, mask_level: int = 0):
        self.processor = processor
        self.temporal_map = temporal_map
        self.mask_level = mask_level
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                # Only keep records with valid sample_token
                if rec.get("sample_token") and rec["sample_token"] in temporal_map:
                    self.records.append(rec)

        print(f"  Loaded {len(self.records)} records (mask_level={mask_level})")

    def set_mask_level(self, level):
        """Update masking level for curriculum learning."""
        self.mask_level = level

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        token = rec["sample_token"]
        prev_path, curr_path, next_path = self.temporal_map[token]

        # Load 3 temporal frames (resize to 512x512 to reduce token count)
        # Full res (1600x900) → 6945 tokens for 3 images; 512x512 → 786 tokens
        TARGET_SIZE = (512, 512)
        images = []
        for path in [prev_path, curr_path, next_path]:
            try:
                img = Image.open(path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)))

        # Apply text masking
        masked_question = mask_text_features(rec["question"], self.mask_level)

        # Add temporal context prefix
        temporal_prefix = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                          "Use visual motion cues across frames. ")

        # Build conversation with 3 images
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # t-1
                    {"type": "image"},  # t
                    {"type": "image"},  # t+1
                    {"type": "text", "text": temporal_prefix + masked_question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": rec["answer"]},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text,
            "images": images,  # 3 images
        }


def collate_fn_factory(processor):
    """Create collate function for multi-image batches."""

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["images"] for ex in examples]

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        labels = batch["input_ids"].clone()
        if hasattr(batch, "attention_mask"):
            labels[batch["attention_mask"] == 0] = -100

        # Mask prompt tokens (everything before assistant response)
        for i in range(labels.size(0)):
            text_i = texts[i]
            answer_pos = _find_answer_start(text_i)
            if answer_pos > 0:
                prompt_tokens = processor.tokenizer(
                    text_i[:answer_pos],
                    add_special_tokens=True,
                    truncation=True,
                )
                prompt_len = len(prompt_tokens["input_ids"])
                labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def _find_answer_start(text: str) -> int:
    """Find character position where assistant answer starts."""
    markers = ["<|assistant|>", "assistant\n", "### Answer\n"]
    for marker in markers:
        pos = text.rfind(marker)
        if pos >= 0:
            return pos + len(marker)
    return int(len(text) * 0.7)


# ════════════════════════════════════════════════════════════════
#  Curriculum Training Loop
# ════════════════════════════════════════════════════════════════

def train_epoch(model, processor, train_dataset, val_dataset, cfg, epoch, output_dir):
    """Train one epoch with current mask level."""

    training_args = SFTConfig(
        output_dir=str(output_dir / f"epoch_{epoch}"),
        num_train_epochs=1,
        per_device_train_batch_size=4,  # 3 resized images = ~786 tokens, fits in memory
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="adamw_torch_fused",
        learning_rate=cfg.learning_rate * (0.5 ** epoch),  # LR decay per epoch
        weight_decay=cfg.weight_decay,
        warmup_ratio=0.05,
        bf16=cfg.bf16,
        save_strategy="no",  # save manually
        eval_strategy="no",  # eval manually after
        logging_steps=cfg.logging_steps,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=cfg.seed + epoch,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_factory(processor),
        processing_class=processor,
    )

    trainer.train()
    return trainer


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model dir (default: phase0 merged)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of curriculum epochs")
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Cap training samples (-1 = all)")
    args = parser.parse_args()

    cfg = VLConfig()

    print(f"\n{'=' * 70}")
    print(f"  VISUAL GROUNDING: Temporal Frames + Curriculum Text Masking")
    print(f"{'=' * 70}")

    # ── Load base model (Phase 0 checkpoint) ──
    base_dir = args.base_model or str(CHECKPOINT_DIR / "phase0_vl_merged")
    print(f"\n[VG] Loading base model from {base_dir}")
    processor = AutoProcessor.from_pretrained(base_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Apply LoRA (fresh, on top of Phase 0 merged) ──
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Load nuScenes for temporal frame lookup ──
    print("\n[VG] Loading nuScenes for temporal frame mapping...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    temporal_map = build_temporal_map(nusc, 'data/nuscenes')
    print(f"  Built temporal map for {len(temporal_map)} samples")

    # Free nuScenes from memory after building the map
    del nusc

    # ── Output directory ──
    vg_output_dir = CHECKPOINT_DIR / "visual_grounding"
    vg_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Curriculum Training ──
    # mask_schedule: epoch -> mask_level
    # 0 = full text, 1 = partial mask, 2 = full mask
    mask_schedule = {0: 0, 1: 1, 2: 2}

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    for epoch in range(args.num_epochs):
        mask_level = mask_schedule.get(epoch, 2)
        lr = cfg.learning_rate * (0.5 ** epoch)

        print(f"\n{'=' * 60}")
        print(f"  Epoch {epoch + 1}/{args.num_epochs} | mask_level={mask_level} | lr={lr:.2e}")
        print(f"  Masking: {['Full text + 3 images', 'Partial mask (keep ego_speed)', 'Full mask (instruction only)'][mask_level]}")
        print(f"{'=' * 60}")

        # Create dataset with current mask level
        train_dataset = TemporalVLDataset(
            str(train_path), processor, temporal_map, mask_level=mask_level
        )
        val_dataset = TemporalVLDataset(
            str(val_path), processor, temporal_map, mask_level=mask_level
        )

        if args.max_train_samples > 0 and len(train_dataset.records) > args.max_train_samples:
            rng = random.Random(cfg.seed + epoch)
            indices = rng.sample(range(len(train_dataset.records)), args.max_train_samples)
            train_dataset.records = [train_dataset.records[i] for i in sorted(indices)]
            print(f"  Subsampled to {len(train_dataset.records)} training samples")

        trainer = train_epoch(
            model, processor, train_dataset, val_dataset, cfg, epoch, vg_output_dir
        )

        print(f"  Epoch {epoch + 1} complete.")

    # ── Save final model ──
    print("\n[VG] Saving final LoRA model...")
    final_lora_dir = vg_output_dir / "vg_lora_final"
    trainer.save_model(str(final_lora_dir))
    processor.save_pretrained(str(final_lora_dir))

    print("[VG] Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = vg_output_dir / "vg_merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"[VG] Saved merged model to {merged_dir}")

    # ── Save training config ──
    config_record = {
        "base_model": base_dir,
        "num_epochs": args.num_epochs,
        "mask_schedule": mask_schedule,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "initial_lr": cfg.learning_rate,
        "max_train_samples": args.max_train_samples,
    }
    with open(vg_output_dir / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Visual Grounding Training Complete")
    print(f"  Next: Run evaluation with image ablation to verify model uses images")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

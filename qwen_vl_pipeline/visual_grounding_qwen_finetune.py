"""
Visual Grounding (TVG) — Qwen2-VL-2B: Image-Only Training (No Text Features)
=============================================================================

Adapted from LFM2's visual_grounding_v2_finetune.py for Qwen2-VL architecture.
Purpose: Remove text shortcuts and add 3 temporal frames so the model
genuinely relies on visual input. This is a prerequisite for meaningful
CIS diagnosis (without TVG, CIS_img ≈ 0 because the model ignores images).

Usage:
  cd /mnt/vdb/LFM_project
  CUDA_VISIBLE_DEVICES=1 python qwen_vl_pipeline/visual_grounding_qwen_finetune.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from nuscenes.nuscenes import NuScenes

from config_qwen import (
    QwenVLConfig,
    QWEN_MODEL_ID,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    NUM_LAYERS,
    HIDDEN_DIM,
)


# ════════════════════════════════════════════════════════════════
#  Temporal Frame Lookup
# ════════════════════════════════════════════════════════════════

def build_temporal_map(nusc, dataroot):
    """sample_token -> (prev_image_path, curr_image_path, next_image_path)."""
    temporal_map = {}
    for sample in nusc.sample:
        token = sample['token']
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        curr_path = os.path.join(dataroot, cam_data['filename'])

        prev_path = curr_path
        if sample['prev']:
            try:
                prev_sample = nusc.get('sample', sample['prev'])
                prev_cam = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])
                prev_path = os.path.join(dataroot, prev_cam['filename'])
            except Exception:
                pass

        next_path = curr_path
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
#  Dataset — Image-Only (no text features), 3 temporal frames
# ════════════════════════════════════════════════════════════════

INSTRUCTION = "Predict future trajectory as 6 waypoints (x,y) in ego-frame metres at t+0.5s to t+3.0s."
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")
TARGET_SIZE = (256, 256)  # Smaller than LFM2's 512 to keep Qwen2-VL vision encoder tractable


class ImageOnlyQwenDataset(Dataset):
    """3-frame temporal input with NO text numerical features for Qwen2-VL."""

    def __init__(self, jsonl_path: str, processor, temporal_map: dict):
        self.processor = processor
        self.temporal_map = temporal_map
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("sample_token") and rec["sample_token"] in temporal_map:
                    self.records.append(rec)
        print(f"  Loaded {len(self.records)} records (image-only, no text features)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        token = rec["sample_token"]
        prev_path, curr_path, next_path = self.temporal_map[token]

        images = []
        for path in [prev_path, curr_path, next_path]:
            try:
                img = Image.open(path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)))

        # NO text features — only task instruction + temporal prefix
        question = TEMPORAL_PREFIX + INSTRUCTION

        # Qwen2-VL chat format: each image is a separate content entry with PIL object
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": rec["answer"],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False,
        )

        return {"text": text, "images": images}


def collate_fn_factory(processor):
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["images"] for ex in examples]

        batch = processor(
            text=texts, images=images,
            return_tensors="pt", padding=True, truncation=True,
        )

        labels = batch["input_ids"].clone()
        if hasattr(batch, "attention_mask"):
            labels[batch["attention_mask"] == 0] = -100

        for i in range(labels.size(0)):
            text_i = texts[i]
            answer_pos = find_answer_start(text_i)
            if answer_pos > 0:
                prompt_tokens = processor.tokenizer(
                    text_i[:answer_pos], add_special_tokens=True, truncation=True,
                )
                prompt_len = len(prompt_tokens["input_ids"])
                labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def find_answer_start(text: str) -> int:
    """Find character position where assistant answer starts."""
    markers = ["<|im_start|>assistant\n", "assistant\n", "<|assistant|>"]
    for marker in markers:
        pos = text.rfind(marker)
        if pos >= 0:
            return pos + len(marker)
    return int(len(text) * 0.7)


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    cfg = QwenVLConfig()

    print(f"\n{'=' * 70}")
    print(f"  VISUAL GROUNDING (TVG): Qwen2-VL Image-Only Training")
    print(f"  {args.num_epochs} epochs, batch={args.batch_size}×{args.grad_accum}={args.batch_size * args.grad_accum}")
    print(f"{'=' * 70}")

    # ── Load base model (Phase 0 merged checkpoint) ──
    base_dir = args.base_model or str(CHECKPOINT_DIR / "phase0_qwen_merged")
    print(f"\n[TVG-Qwen] Loading base model from {base_dir}")
    processor = AutoProcessor.from_pretrained(base_dir)
    # Limit vision tokens per image for 3-frame input tractability
    processor.image_processor.min_pixels = 256 * 256      # ~256 tokens per image
    processor.image_processor.max_pixels = 256 * 256       # cap at same
    print(f"  Image pixels: min={processor.image_processor.min_pixels}, max={processor.image_processor.max_pixels}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params / 1e6:.1f}M")
    print(f"  Architecture: {NUM_LAYERS} transformer attention layers, hidden_dim={HIDDEN_DIM}")

    # ── Apply LoRA ──
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
    print("\n[TVG-Qwen] Loading nuScenes for temporal frame mapping...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    temporal_map = build_temporal_map(nusc, 'data/nuscenes')
    print(f"  Built temporal map for {len(temporal_map)} samples")
    del nusc

    # ── Output directory ──
    tvg_dir = CHECKPOINT_DIR / "tvg_qwen"
    tvg_dir.mkdir(parents=True, exist_ok=True)

    # ── Load datasets ──
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    train_dataset = ImageOnlyQwenDataset(str(train_path), processor, temporal_map)
    val_dataset = ImageOnlyQwenDataset(str(val_path), processor, temporal_map)

    if args.max_train_samples > 0 and len(train_dataset.records) > args.max_train_samples:
        rng = random.Random(cfg.seed)
        indices = rng.sample(range(len(train_dataset.records)), args.max_train_samples)
        train_dataset.records = [train_dataset.records[i] for i in sorted(indices)]
        print(f"  Subsampled to {len(train_dataset.records)} training samples")

    # ── Training ──
    training_args = SFTConfig(
        output_dir=str(tvg_dir / "training"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="adamw_torch_fused",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=0.05,
        bf16=cfg.bf16,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="no",
        logging_steps=10,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=cfg.seed,
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

    print(f"\n[TVG-Qwen] Starting training — {len(train_dataset)} samples, {args.num_epochs} epochs")
    trainer.train()

    # ── Save final model ──
    print("\n[TVG-Qwen] Saving final LoRA model...")
    final_lora_dir = tvg_dir / "tvg_lora_final"
    trainer.save_model(str(final_lora_dir))
    processor.save_pretrained(str(final_lora_dir))

    print("[TVG-Qwen] Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = tvg_dir / "tvg_merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"[TVG-Qwen] Saved merged model to {merged_dir}")

    # ── Save config ──
    config_record = {
        "approach": "TVG image-only (no text numerical features)",
        "architecture": "Qwen2-VL-2B-Instruct",
        "base_model": base_dir,
        "num_epochs": args.num_epochs,
        "text_features": "NONE — instruction only",
        "temporal_frames": 3,
        "image_size": list(TARGET_SIZE),
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "learning_rate": cfg.learning_rate,
        "max_train_samples": args.max_train_samples,
    }
    with open(tvg_dir / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  TVG Training Complete — Qwen2-VL Image-Only")
    print(f"  Next: Run CIS diagnosis on {merged_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

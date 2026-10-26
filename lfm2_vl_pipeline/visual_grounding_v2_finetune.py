"""
Visual Grounding v2 — Image-Only Training (No Text Features)
=============================================================

Previous VG attempt (curriculum masking) FAILED — model still ignores images.
Root cause: text features (ego_speed, front_distance, lead_velocity) perfectly
predict trajectory, so the model has no incentive to learn from images.

Stronger approach: Remove ALL text numerical features from epoch 1.
The model gets ONLY the task instruction + 3 temporal frames.
This forces the model to extract motion/scene info from images.

Usage:
  cd /mnt/vdb/LFM_project
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/visual_grounding_v2_finetune.py
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
#  Dataset — Image-Only (no text features)
# ════════════════════════════════════════════════════════════════

INSTRUCTION = "Predict future trajectory as 6 waypoints (x,y) in ego-frame metres at t+0.5s to t+3.0s."
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")
TARGET_SIZE = (512, 512)


class ImageOnlyVLDataset(Dataset):
    """3-frame temporal input with NO text numerical features."""

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

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": question},
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
            answer_pos = _find_answer_start(text_i)
            if answer_pos > 0:
                prompt_tokens = processor.tokenizer(
                    text_i[:answer_pos], add_special_tokens=True, truncation=True,
                )
                prompt_len = len(prompt_tokens["input_ids"])
                labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def _find_answer_start(text: str) -> int:
    markers = ["<|assistant|>", "assistant\n", "### Answer\n"]
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    cfg = VLConfig()

    print(f"\n{'=' * 70}")
    print(f"  VISUAL GROUNDING v2: Image-Only Training (NO text features)")
    print(f"  {args.num_epochs} epochs, batch={args.batch_size}×{args.grad_accum}={args.batch_size * args.grad_accum}")
    print(f"{'=' * 70}")

    # ── Load base model (Phase 0 checkpoint) ──
    base_dir = args.base_model or str(CHECKPOINT_DIR / "phase0_vl_merged")
    print(f"\n[VG2] Loading base model from {base_dir}")
    processor = AutoProcessor.from_pretrained(base_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

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
    print("\n[VG2] Loading nuScenes for temporal frame mapping...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    temporal_map = build_temporal_map(nusc, 'data/nuscenes')
    print(f"  Built temporal map for {len(temporal_map)} samples")
    del nusc

    # ── Output directory ──
    vg2_dir = CHECKPOINT_DIR / "visual_grounding_v2"
    vg2_dir.mkdir(parents=True, exist_ok=True)

    # ── Load datasets ──
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    train_dataset = ImageOnlyVLDataset(str(train_path), processor, temporal_map)
    val_dataset = ImageOnlyVLDataset(str(val_path), processor, temporal_map)

    if args.max_train_samples > 0 and len(train_dataset.records) > args.max_train_samples:
        rng = random.Random(cfg.seed)
        indices = rng.sample(range(len(train_dataset.records)), args.max_train_samples)
        train_dataset.records = [train_dataset.records[i] for i in sorted(indices)]
        print(f"  Subsampled to {len(train_dataset.records)} training samples")

    # ── Training ──
    training_args = SFTConfig(
        output_dir=str(vg2_dir / "training"),
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
        logging_steps=cfg.logging_steps,
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

    print(f"\n[VG2] Starting training — {len(train_dataset)} samples, {args.num_epochs} epochs")
    trainer.train()

    # ── Save final model ──
    print("\n[VG2] Saving final LoRA model...")
    final_lora_dir = vg2_dir / "vg2_lora_final"
    trainer.save_model(str(final_lora_dir))
    processor.save_pretrained(str(final_lora_dir))

    print("[VG2] Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = vg2_dir / "vg2_merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"[VG2] Saved merged model to {merged_dir}")

    # ── Save config ──
    config_record = {
        "approach": "image-only (no text numerical features from epoch 1)",
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
    with open(vg2_dir / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  VG v2 Training Complete — Image-Only")
    print(f"  Next: Run visual_grounding_evaluate.py with --model_dir {merged_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

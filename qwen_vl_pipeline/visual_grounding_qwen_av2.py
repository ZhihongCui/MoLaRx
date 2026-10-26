"""
Visual Grounding (TVG) — Qwen2-VL-2B on **Argoverse 2** (PIT → MIA).
=====================================================================

Adapted from visual_grounding_qwen_finetune.py. Differences:
- Reads Argoverse 2 JSONL (image_paths already 3-frame list; no NuScenes devkit)
- Output directory: qwen_vl_pipeline/outputs/checkpoints/tvg_qwen_av2
- Source city = Pittsburgh (implicit in AV2 train split)

Usage:
  cd /mnt/vdb/LFM_project
  CUDA_VISIBLE_DEVICES=1 python qwen_vl_pipeline/visual_grounding_qwen_av2.py \
      --num_epochs 1
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_qwen import (
    QwenVLConfig,
    QWEN_MODEL_ID,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    NUM_LAYERS,
    HIDDEN_DIM,
)


# ════════════════════════════════════════════════════════════════
#  AV2 data paths (matching lfm2 AV2 pipeline)
# ════════════════════════════════════════════════════════════════

AV2_DATA_DIR = Path("argoverse2_pipeline/outputs/data")

INSTRUCTION = ("Predict future trajectory as 6 waypoints (x,y) "
               "in ego-frame metres at t+0.5s to t+3.0s.")
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")
TARGET_SIZE = (256, 256)


# ════════════════════════════════════════════════════════════════
#  Dataset (AV2: image_paths is already a 3-list in each record)
# ════════════════════════════════════════════════════════════════

class ImageOnlyQwenAV2Dataset(Dataset):
    """3-frame temporal input, NO text numerical features, for Qwen2-VL on AV2."""

    def __init__(self, jsonl_path: str, processor):
        self.processor = processor
        self.records = []
        skip_stats = Counter()
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                ips = rec.get("image_paths")
                if not (isinstance(ips, list) and len(ips) == 3):
                    skip_stats["bad_image_paths"] += 1
                    continue
                if not rec.get("answer"):
                    skip_stats["no_answer"] += 1
                    continue
                self.records.append(rec)
        print(f"  Loaded {len(self.records)} AV2 records "
              f"(skipped: {dict(skip_stats)})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prev_path, curr_path, next_path = rec["image_paths"]

        images = []
        for path in [prev_path, curr_path, next_path]:
            try:
                img = Image.open(path).convert("RGB").resize(TARGET_SIZE,
                                                              Image.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)))

        question = TEMPORAL_PREFIX + INSTRUCTION

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
    parser.add_argument("--base_model", type=str, default=None,
                        help="Default: Phase 0 merged ckpt (phase0_qwen_merged)")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    cfg = QwenVLConfig()

    print(f"\n{'=' * 70}")
    print(f"  VISUAL GROUNDING (TVG): Qwen2-VL on Argoverse 2 (PIT -> MIA)")
    print(f"  {args.num_epochs} epochs, batch={args.batch_size}x"
          f"{args.grad_accum}={args.batch_size * args.grad_accum}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    # ── Load base model ─────────────────────────────────────────
    base_dir = args.base_model or str(CHECKPOINT_DIR / "phase0_qwen_merged")
    print(f"\n[TVG-Qwen-AV2] Loading base model from {base_dir}")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(base_dir)
    processor.image_processor.min_pixels = 256 * 256
    processor.image_processor.max_pixels = 256 * 256
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params / 1e6:.1f}M")
    print(f"  Architecture: {NUM_LAYERS} transformer layers, hidden_dim={HIDDEN_DIM}")
    sys.stdout.flush()

    # ── LoRA ────────────────────────────────────────────────────
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

    # ── Output directory ────────────────────────────────────────
    tvg_dir = CHECKPOINT_DIR / "tvg_qwen_av2"
    tvg_dir.mkdir(parents=True, exist_ok=True)

    # ── Load AV2 datasets ───────────────────────────────────────
    train_path = AV2_DATA_DIR / "train.jsonl"
    val_path   = AV2_DATA_DIR / "val.jsonl"
    if not train_path.exists():
        print(f"[FATAL] Missing {train_path}")
        sys.exit(2)

    print(f"\n[TVG-Qwen-AV2] Loading AV2 data from {AV2_DATA_DIR} ...")
    sys.stdout.flush()
    train_dataset = ImageOnlyQwenAV2Dataset(str(train_path), processor)
    val_dataset   = ImageOnlyQwenAV2Dataset(str(val_path),   processor)

    if len(train_dataset.records) == 0:
        print(f"[FATAL] 0 training records loaded — check image_paths format.")
        sys.exit(2)

    if args.max_train_samples > 0 and len(train_dataset.records) > args.max_train_samples:
        rng = random.Random(cfg.seed)
        idx = rng.sample(range(len(train_dataset.records)), args.max_train_samples)
        train_dataset.records = [train_dataset.records[i] for i in sorted(idx)]
        print(f"  Subsampled train to {len(train_dataset.records)}")

    # ── Training ────────────────────────────────────────────────
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

    print(f"\n[TVG-Qwen-AV2] Starting training — {len(train_dataset)} samples, "
          f"{args.num_epochs} epochs")
    sys.stdout.flush()
    trainer.train()

    # ── Save ────────────────────────────────────────────────────
    print("\n[TVG-Qwen-AV2] Saving final LoRA ...")
    final_lora_dir = tvg_dir / "tvg_lora_final"
    trainer.save_model(str(final_lora_dir))
    processor.save_pretrained(str(final_lora_dir))

    print("[TVG-Qwen-AV2] Merging LoRA ...")
    merged_model = model.merge_and_unload()
    merged_dir = tvg_dir / "tvg_merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"[TVG-Qwen-AV2] Saved merged model to {merged_dir}")
    sys.stdout.flush()

    config_record = {
        "approach":           "TVG image-only (AV2 PIT->MIA)",
        "architecture":       "Qwen2-VL-2B-Instruct",
        "base_model":         base_dir,
        "dataset":            "argoverse2",
        "num_epochs":         args.num_epochs,
        "text_features":      "NONE - instruction only",
        "temporal_frames":    3,
        "image_size":         list(TARGET_SIZE),
        "lora_r":             cfg.lora_r,
        "lora_alpha":         cfg.lora_alpha,
        "batch_size":         args.batch_size,
        "grad_accum":         args.grad_accum,
        "effective_batch":    args.batch_size * args.grad_accum,
        "learning_rate":      cfg.learning_rate,
        "max_train_samples":  args.max_train_samples,
    }
    with open(tvg_dir / "training_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  TVG-Qwen-AV2 Training Complete")
    print(f"  Next: Run CIS on {merged_dir} (import partition from nuS)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

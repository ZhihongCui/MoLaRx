"""
Phase 0 (VL) — LoRA Fine-tune LFM2-VL-1.6B on Singapore Driving Images
=========================================================================

Fine-tunes the VLM on Singapore CAM_FRONT images with driving decision QA.
Uses LoRA on all linear layers of the language backbone (Phase 0 baseline).

Usage:
  python lfm2_vl_pipeline/phase0_vl_finetune.py
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from config_vl import (
    VLConfig,
    VL_MODEL_ID,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


# ════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════

class DrivingVLDataset(Dataset):
    """Dataset that returns conversations in the VLM chat format."""

    def __init__(self, jsonl_path: str, processor):
        self.processor = processor
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load image
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            # Fallback: return a blank image if path not found
            print(f"  [WARNING] Cannot load {rec['image_path']}: {e}")
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # Build conversation in VLM chat format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": rec["answer"]},
                ],
            },
        ]

        # Apply chat template to get formatted text
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text,
            "images": [image],
        }


def collate_fn_factory(processor):
    """Create a collate function that uses the processor."""

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["images"] for ex in examples]

        # Process through VL processor (handles both image + text tokenization)
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Create labels: same as input_ids but mask non-answer tokens
        labels = batch["input_ids"].clone()
        # Mask padding tokens
        if hasattr(batch, "attention_mask"):
            labels[batch["attention_mask"] == 0] = -100

        # Mask all tokens before the assistant response
        # The processor encodes the full conversation; we need to find
        # where the assistant response starts and mask everything before it
        pad_token_id = processor.tokenizer.pad_token_id
        for i in range(labels.size(0)):
            # Find the assistant marker in the tokenized text
            # For LFM2-VL, the chat template uses specific markers
            input_ids_i = batch["input_ids"][i]
            # Simple heuristic: find the last occurrence of the user-end marker
            # and mask everything up to and including it
            # This is template-dependent; we'll use a conservative approach
            text_i = texts[i]
            # Find answer start position by tokenizing just the prompt part
            answer_marker = rec_answer_start(text_i)
            if answer_marker > 0:
                prompt_tokens = processor.tokenizer(
                    text_i[:answer_marker],
                    add_special_tokens=True,
                    truncation=True,
                )
                prompt_len = len(prompt_tokens["input_ids"])
                labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def rec_answer_start(text: str) -> int:
    """Find the character position where the assistant answer starts."""
    # Look for assistant turn marker in the chat template
    markers = ["<|assistant|>", "assistant\n", "### Answer\n"]
    for marker in markers:
        pos = text.rfind(marker)
        if pos >= 0:
            return pos + len(marker)
    # Fallback: use 70% of text as boundary
    return int(len(text) * 0.7)


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    cfg = VLConfig()

    print(f"\n{'=' * 70}")
    print(f"  PHASE 0 (VL): LoRA Fine-tune LFM2-VL-1.6B")
    print(f"{'=' * 70}")

    # ── Load model and processor ──
    print(f"\n[Phase 0] Loading {VL_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(VL_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        VL_MODEL_ID,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Print model structure for debugging layer names
    print("\n[Phase 0] Model structure (first 5 named modules):")
    for i, (name, _) in enumerate(model.named_modules()):
        if i < 20:
            print(f"  {name}")
    print("  ...")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Phase 0] Total parameters: {n_params / 1e6:.1f}M")

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

    # ── Load data ──
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    if not train_path.exists():
        print(f"[Phase 0] ERROR: {train_path} not found. Run prepare_vl_data.py first.")
        sys.exit(1)

    print(f"\n[Phase 0] Loading training data from {train_path}")
    train_dataset = DrivingVLDataset(str(train_path), processor)
    val_dataset = DrivingVLDataset(str(val_path), processor)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # ── Training config ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHECKPOINT_DIR / "phase0_vl_lora"

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="adamw_torch_fused",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        save_strategy=cfg.save_strategy,
        eval_strategy="epoch",
        logging_steps=cfg.logging_steps,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=cfg.seed,
        report_to="none",
    )

    # ── Train ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_factory(processor),
        processing_class=processor,
    )

    print(f"\n[Phase 0] Starting training...")
    trainer.train()

    # ── Save ──
    final_dir = CHECKPOINT_DIR / "phase0_vl_final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\n[Phase 0] Saved model + processor to {final_dir}")

    # Also save the merged model for easier loading in Phase 1
    print(f"[Phase 0] Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = CHECKPOINT_DIR / "phase0_vl_merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"[Phase 0] Saved merged model to {merged_dir}")

    print(f"\n{'=' * 70}")
    print(f"  Phase 0 (VL) Complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

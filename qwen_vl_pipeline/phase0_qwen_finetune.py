"""
Phase 0 (Qwen-VL) — LoRA Fine-tune Qwen2-VL-2B-Instruct on Driving Data
=========================================================================

Same task and data as LFM2-VL Phase 0, but using Qwen2-VL as backbone.
This provides a cross-architecture comparison: standard transformer (Qwen)
vs LTC-derived hybrid (LFM2).

Usage:
  CUDA_VISIBLE_DEVICES=1 python qwen_vl_pipeline/phase0_qwen_finetune.py
"""

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from config_qwen import (
    QwenVLConfig,
    QWEN_MODEL_ID,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


class DrivingVLDataset(Dataset):
    """Dataset that returns conversations in VLM chat format."""

    def __init__(self, jsonl_path: str, processor):
        self.processor = processor
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  [WARNING] Cannot load {rec['image_path']}: {e}")
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # Qwen2-VL chat format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": rec["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": rec["answer"],
            },
        ]

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

        # Mask prompt tokens (only train on assistant response)
        for i in range(labels.size(0)):
            text_i = texts[i]
            answer_pos = find_answer_start(text_i)
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


def find_answer_start(text: str) -> int:
    """Find character position where assistant answer starts."""
    markers = ["<|im_start|>assistant\n", "assistant\n", "<|assistant|>"]
    for marker in markers:
        pos = text.rfind(marker)
        if pos >= 0:
            return pos + len(marker)
    return int(len(text) * 0.7)


def main():
    cfg = QwenVLConfig()

    print(f"\n{'=' * 70}")
    print(f"  PHASE 0 (Qwen-VL): LoRA Fine-tune Qwen2-VL-2B-Instruct")
    print(f"{'=' * 70}")

    # ── Load model and processor ──
    print(f"\n[Phase 0] Loading {QWEN_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Print model structure
    print("\n[Phase 0] Model structure (first 20 named modules):")
    for i, (name, _) in enumerate(model.named_modules()):
        if i < 20:
            print(f"  {name}")
    print("  ...")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Phase 0] Total parameters: {n_params / 1e6:.1f}M")
    print(f"  Architecture: {NUM_LAYERS} transformer attention layers")
    print(f"  Hidden dim: {HIDDEN_DIM}")

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

    # ── Load data (same splits as LFM2) ──
    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    if not train_path.exists():
        print(f"[Phase 0] ERROR: {train_path} not found.")
        print(f"  Make sure LFM2 pipeline data exists at {DATA_DIR}")
        sys.exit(1)

    print(f"\n[Phase 0] Loading training data from {train_path}")
    train_dataset = DrivingVLDataset(str(train_path), processor)
    val_dataset = DrivingVLDataset(str(val_path), processor)
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # ── Training config ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHECKPOINT_DIR / "phase0_qwen_lora"

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
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
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
    final_dir = CHECKPOINT_DIR / "phase0_qwen_final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\n[Phase 0] Saved model + processor to {final_dir}")

    # Merge LoRA
    print(f"[Phase 0] Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = CHECKPOINT_DIR / "phase0_qwen_merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"[Phase 0] Saved merged model to {merged_dir}")

    print(f"\n{'=' * 70}")
    print(f"  Phase 0 (Qwen-VL) Complete")
    print(f"{'=' * 70}")


from config_qwen import NUM_LAYERS, HIDDEN_DIM

if __name__ == "__main__":
    main()

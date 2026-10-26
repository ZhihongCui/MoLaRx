"""Phase 0: TVG fine-tune for PaliGemma2-3B.

Usage:
  cd /mnt/vdb/LFM_project
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python paligemma_vl_pipeline/phase0_paligemma_tvg.py \
      --dataset nuscenes --num_epochs 1
"""
import argparse, json, os, random, sys, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_paligemma import (
    PaliGemmaConfig, PALI_MODEL_ID, CHECKPOINT_DIR,
    NUS_DATA_DIR, AV2_DATA_DIR, NUM_LAYERS,
)
from paligemma_common import prepare_input_3frame, build_temporal_map_for_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "av2"])
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = PaliGemmaConfig()
    if args.num_epochs is not None: cfg.num_epochs = args.num_epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.grad_accum is not None: cfg.gradient_accumulation_steps = args.grad_accum

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)

    data_dir = NUS_DATA_DIR if args.dataset == "nuscenes" else AV2_DATA_DIR
    ckpt_dir = CHECKPOINT_DIR / f"tvg_{args.dataset}"

    print(f"\n{'='*70}")
    print(f"  PaliGemma2-3B Phase 0 (TVG) on {args.dataset}")
    print(f"  data:   {data_dir}")
    print(f"  output: {ckpt_dir}")
    print(f"  epochs: {cfg.num_epochs}, grad_accum: {cfg.gradient_accumulation_steps}")
    print(f"  dry_run: {args.dry_run}")
    print(f"{'='*70}")
    sys.stdout.flush()

    print(f"\n[Phase 0] Building temporal map for {args.dataset} ...")
    t0 = time.time()
    temporal_map = build_temporal_map_for_dataset(args.dataset)
    if temporal_map is not None:
        print(f"  temporal_map: {len(temporal_map):,} samples ({time.time()-t0:.1f}s)")
    else:
        print(f"  temporal_map: None (AV2 uses image_paths)")
    sys.stdout.flush()

    print(f"\n[Phase 0] Loading {PALI_MODEL_ID} ...")
    sys.stdout.flush()
    processor = PaliGemmaProcessor.from_pretrained(PALI_MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PALI_MODEL_ID,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    device = next(model.parameters()).device
    print(f"  Device: {device}, base mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    sys.stdout.flush()

    peft_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        layers_to_transform=list(range(NUM_LAYERS)),
        layers_pattern="layers",
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")
    sys.stdout.flush()

    records = [json.loads(l) for l in open(data_dir / "train.jsonl")]
    if args.max_samples > 0 and len(records) > args.max_samples:
        records = random.sample(records, args.max_samples)
    if args.dry_run: records = records[:5]
    print(f"[Phase 0] Training samples: {len(records)}")
    sys.stdout.flush()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    model.train()
    history = []
    skip_stats = Counter()
    MAX_PRINTED = 3

    def _skip(reason, exc=None, i=None):
        skip_stats[reason] += 1
        if skip_stats[reason] <= MAX_PRINTED:
            print(f"  [SKIP][{reason}] sample {i}{': '+repr(exc) if exc else ''}")
            if exc is not None and skip_stats[reason] == 1:
                traceback.print_exc()
            sys.stdout.flush()

    for epoch in range(cfg.num_epochs):
        print(f"\n  Epoch {epoch+1}/{cfg.num_epochs}")
        sys.stdout.flush()
        epoch_loss = 0.0; n_steps = 0
        random.shuffle(records); optimizer.zero_grad()
        pbar = tqdm(range(len(records)), desc=f"Epoch {epoch+1}")
        for i in pbar:
            rec = records[i]
            try:
                inputs = prepare_input_3frame(rec, processor, device,
                                              temporal_map=temporal_map, include_labels=True)
            except Exception as e:
                _skip("prepare_exc", e, i); continue
            if inputs is None:
                _skip("prepare_none", None, i); continue
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs)
                loss = out.loss
            except torch.cuda.OutOfMemoryError as e:
                _skip("cuda_oom", e, i); torch.cuda.empty_cache(); continue
            except Exception as e:
                _skip("forward_exc", e, i); continue
            if loss is None:
                _skip("loss_none", None, i); continue

            (loss / cfg.gradient_accumulation_steps).backward()
            epoch_loss += loss.item(); n_steps += 1

            if n_steps % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            if n_steps % cfg.logging_steps == 0:
                pbar.set_postfix({"loss": f"{epoch_loss/n_steps:.4f}",
                                  "skip": sum(skip_stats.values())})
            del out, loss
            torch.cuda.empty_cache()

        if n_steps % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()

        print(f"  === Epoch {epoch+1} summary ===")
        print(f"    successful steps: {n_steps}/{len(records)}")
        print(f"    skip breakdown: {dict(skip_stats)}")
        if n_steps == 0:
            print(f"  [FATAL] 0 successful steps. Inspect [SKIP] above.")
            sys.exit(2)
        avg_loss = epoch_loss / n_steps
        history.append({"epoch": epoch+1, "loss": avg_loss,
                        "steps": n_steps, "skips": dict(skip_stats)})
        print(f"  → avg loss = {avg_loss:.4f}")
        sys.stdout.flush()

        if not args.dry_run:
            ep_dir = ckpt_dir / f"epoch_{epoch+1}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ep_dir))
            processor.save_pretrained(str(ep_dir))

    if args.dry_run:
        print("\n[Phase 0] Dry-run complete — skipping merge/save.")
        return

    print("\n[Phase 0] Merging LoRA ...")
    sys.stdout.flush()
    merged = model.merge_and_unload()
    final = ckpt_dir / "tvg_merged"
    final.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"[Phase 0] Saved merged to {final}")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    del model, merged
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
Phase 2 (VL) — CIS-Guided Selective Text-Channel Domain Alignment
==================================================================

Core innovation (updated based on Phase 1 v2 findings):
  - Phase 1 showed CIS_txt >> CIS_img: text numerical shift is the dominant
    gap source; image channel is already domain-invariant (LTC property).
  - Phase 2 selectively applies LoRA + MMD ONLY to high-CIS_txt layers,
    and computes MMD exclusively on TEXT token positions (not image tokens).

L_total = L_task(Boston) + lambda_DA * L_DA_txt(Boston_txt, Singapore_txt)

Strategies (ablation study):
  cis_selective_lora    — LoRA on phi_txt_repair layers + text-only L_DA (MAIN)
  full_lora_da          — LoRA on all layers + last-token L_DA (ablation: no CIS)
  random_selective_lora — LoRA on random layers + text-only L_DA (ablation: CIS vs random)
  full_lora_no_da       — LoRA on all layers, no L_DA (ablation: no alignment)
  txt_da_no_lora        — No LoRA, text-only L_DA only (ablation: alignment without LoRA)

Usage:
  cd /mnt/vdb/LFM_project
  python lfm2_vl_pipeline/phase2_vl_finetune.py --strategy cis_selective_lora
  python lfm2_vl_pipeline/phase2_vl_finetune.py --strategy all
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig,
    VL_MODEL_ID,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    NUM_LAYERS,
    GQA_LAYER_INDICES,
    LAYER_TYPES,
)
from utils.mmd_loss import MMDLoss

# ── Token IDs (same as Phase 1 v2) ───────────────────────────────
IMAGE_TOKEN_ID = 396   # <image>
IMAGE_START_ID = 498   # <|image_start|>
IMAGE_END_ID   = 499   # <|image_end|>

# ── Paths ─────────────────────────────────────────────────────────
PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl_v2"   # Updated to v2
PHASE2_OUTPUT = OUTPUT_DIR / "phase2_vl"

# ── CIS threshold for layer selection ────────────────────────────
CIS_TXT_THETA = 0.5   # layers with CIS_txt > 0.5 → phi_txt_repair


# ════════════════════════════════════════════════════════════════
#  Token Mask Utility (reused from Phase 1 v2)
# ════════════════════════════════════════════════════════════════

def get_text_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Return boolean mask for text token positions.
    Image tokens = positions between <|image_start|> and <|image_end|>.
    Text tokens = everything else.
    """
    seq_len = input_ids.shape[0]
    img_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)

    start_pos = (input_ids == IMAGE_START_ID).nonzero(as_tuple=True)[0]
    end_pos   = (input_ids == IMAGE_END_ID).nonzero(as_tuple=True)[0]

    if len(start_pos) > 0 and len(end_pos) > 0:
        img_mask[start_pos[0]:end_pos[-1] + 1] = True
    else:
        img_mask = (input_ids == IMAGE_TOKEN_ID)

    return ~img_mask   # text = NOT image


# ════════════════════════════════════════════════════════════════
#  Layer Partition Loading
# ════════════════════════════════════════════════════════════════

def load_partition(theta: float = CIS_TXT_THETA):
    """
    Load CIS layer partition from Phase 1 v2.
    Uses phi_txt_repair (high CIS_txt) as LoRA + MMD target layers.

    Falls back to re-computing from phase1_summary.json if
    layer_partition.json was generated with different theta.
    """
    summary_path = PHASE1_OUTPUT / "phase1_summary.json"
    partition_path = PHASE1_OUTPUT / "layer_partition.json"

    # Prefer re-computing from summary with correct theta
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        phi_repair = []
        phi_freeze  = []
        for l_str, info in sorted(summary.items(), key=lambda x: int(x[0])):
            l = int(l_str)
            if info.get("cis_txt", 0) > theta:
                phi_repair.append(l)
            else:
                phi_freeze.append(l)
        print(f"[Phase 2] Partition from phase1_summary.json (theta={theta}):")
        print(f"          phi_txt_repair = {phi_repair}")
        print(f"          phi_freeze     = {phi_freeze}")
        return {
            "theta": theta,
            "phi_txt_repair": phi_repair,
            "phi_freeze": phi_freeze,
        }

    # Fallback: try partition file
    if partition_path.exists():
        with open(partition_path) as f:
            p = json.load(f)
        # Handle both old and new key names
        if "phi_txt_repair" in p:
            return p
        elif "phi_low" in p:
            print("[Phase 2] WARNING: Using old partition format (phi_low/phi_high)")
            return {
                "theta": p.get("theta", theta),
                "phi_txt_repair": p.get("phi_low", []),
                "phi_freeze": p.get("phi_high", []),
            }

    # Last resort default
    print("[Phase 2] WARNING: No partition file found, using default layers 8-16")
    return {
        "theta": theta,
        "phi_txt_repair": [8, 9, 10, 11, 12, 13, 14, 15, 16],
        "phi_freeze": list(range(8)),
    }


# ════════════════════════════════════════════════════════════════
#  Batch Preparation (fixed for LFM2-VL processor)
# ════════════════════════════════════════════════════════════════

def prepare_single(rec, processor, device, include_labels=True):
    """
    Prepare a single record as model inputs.
    Uses the correct LFM2-VL call: image embedded in content dict.
    """
    try:
        img = Image.open(rec["image_path"]).convert("RGB")
    except Exception:
        img = Image.new("RGB", (640, 480), color=(128, 128, 128))

    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rec["question"]},
            ],
        },
    ]
    if include_labels:
        conv.append({
            "role": "assistant",
            "content": [{"type": "text", "text": rec["answer"]}],
        })

    inputs = processor.apply_chat_template(
        conv,
        add_generation_prompt=not include_labels,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    if include_labels:
        labels = inputs["input_ids"].clone()
        # Mask padding and prompt tokens — only train on assistant reply
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels

    return inputs


# ════════════════════════════════════════════════════════════════
#  LoRA Model Creation
# ════════════════════════════════════════════════════════════════

def create_lora_model(model, cfg, strategy, partition):
    """
    Apply LoRA adapters to strategy-specific layers.

    Strategy → layers:
      cis_selective_lora    → phi_txt_repair (high CIS_txt layers)
      full_lora_da          → all layers
      random_selective_lora → random N layers (N = |phi_txt_repair|)
      full_lora_no_da       → all layers
      txt_da_no_lora        → no LoRA (skip)
    """
    if strategy == "txt_da_no_lora":
        print("[Phase 2] txt_da_no_lora: skipping LoRA")
        return model, []

    phi_repair = partition["phi_txt_repair"]

    if strategy == "cis_selective_lora":
        # LoRA only on high-CIS_txt layers (the text domain gap carriers)
        # Convert layer output indices to backbone layer indices (subtract 1)
        layers_to_transform = [l - 1 for l in phi_repair if l > 0]
        print(f"[Phase 2] CIS Selective LoRA on layers: {layers_to_transform}")

    elif strategy == "full_lora_da":
        layers_to_transform = list(range(NUM_LAYERS))
        print(f"[Phase 2] Full LoRA + DA: all {NUM_LAYERS} layers")

    elif strategy == "random_selective_lora":
        n = len([l - 1 for l in phi_repair if l > 0])
        random.seed(cfg.seed + 99)
        layers_to_transform = sorted(random.sample(range(NUM_LAYERS), n))
        print(f"[Phase 2] Random Selective LoRA ({n} layers): {layers_to_transform}")

    elif strategy == "full_lora_no_da":
        layers_to_transform = list(range(NUM_LAYERS))
        print(f"[Phase 2] Full LoRA (no DA): all {NUM_LAYERS} layers")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Note: layers_to_transform requires target_modules to be a list, not "all-linear"
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",  # attention
            "w1", "w2", "w3",                           # feed-forward
            "in_proj",                                   # LIV conv
        ],
        layers_to_transform=layers_to_transform,
        layers_pattern="layers",
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, layers_to_transform


# ════════════════════════════════════════════════════════════════
#  Text-Channel Feature Extraction
# ════════════════════════════════════════════════════════════════

def extract_text_features(hidden_states, input_ids, target_layers,
                           use_text_mask=True):
    """
    Extract mean-pooled text-channel hidden states per target layer.

    Returns dict {layer_idx: tensor [D]} for a single sample.
    Caller stacks across samples for MMD computation.
    """
    feats = {}
    for l_out in target_layers:
        if l_out >= len(hidden_states):
            continue
        h = hidden_states[l_out][0]  # [seq, D] for batch_size=1

        if use_text_mask:
            txt_mask = get_text_mask(input_ids[0].cpu())
            if txt_mask.sum() == 0:
                continue
            rep = h[txt_mask.to(h.device)].mean(0).float()
        else:
            rep = h[-1].float()  # last token fallback

        feats[l_out] = F.normalize(rep, dim=-1)

    return feats


def compute_mmd_from_features(id_feats_list, ood_feats_list,
                               target_layers, mmd_loss_fn, device):
    """
    Compute mean MMD across target layers given pre-extracted feature lists.

    id_feats_list:  list of dicts {layer: [D]}  (one dict per ID sample)
    ood_feats_list: list of dicts {layer: [D]}  (one dict per OOD sample)
    """
    loss_da = torch.tensor(0.0, device=device)
    n_valid = 0

    for l_out in target_layers:
        id_vecs  = [f[l_out] for f in id_feats_list  if l_out in f]
        ood_vecs = [f[l_out] for f in ood_feats_list if l_out in f]

        if len(id_vecs) < 2 or len(ood_vecs) < 2:
            continue

        h_id  = torch.stack(id_vecs).to(device)   # [N_id,  D]
        h_ood = torch.stack(ood_vecs).to(device)   # [N_ood, D]

        loss_da = loss_da + mmd_loss_fn(h_id, h_ood)
        n_valid += 1

    if n_valid > 0:
        loss_da = loss_da / n_valid

    return loss_da


# ════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════

def run_strategy(strategy, cfg):
    """Run Phase 2 fine-tuning with a specific strategy."""
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2 (VL): {strategy.upper()}")
    print(f"  Text-Channel Selective Domain Alignment")
    print(f"{'=' * 70}")

    use_da     = strategy != "full_lora_no_da"
    use_txt_mask = strategy in ("cis_selective_lora", "random_selective_lora",
                                 "txt_da_no_lora")

    # ── Load partition ───────────────────────────────────────────
    partition = load_partition(theta=CIS_TXT_THETA)
    phi_repair = partition["phi_txt_repair"]
    # hidden_states[l] = output of backbone layer l-1, so:
    # target_layer_outputs = phi_txt_repair (already in hidden_state indices)
    target_layer_outputs = phi_repair

    # ── Load base model (Phase 0 merged) ────────────────────────
    model_dir = str(CHECKPOINT_DIR / "phase0_vl_merged")
    print(f"\n[Phase 2] Loading model from {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"  Device: {device}")

    # ── Apply LoRA ───────────────────────────────────────────────
    model, lora_layers = create_lora_model(model, cfg, strategy, partition)

    # ── Gradient checkpointing (saves ~40% GPU memory) ───────────
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # ── Load data ────────────────────────────────────────────────
    # ID data (Boston with current reversed config) → L_task
    id_records = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            id_records.append(json.loads(line))
    print(f"[Phase 2] ID train set (Boston): {len(id_records)} records")

    # OOD data (Singapore) → L_DA reference (no labels needed)
    ood_records = []
    if use_da:
        with open(DATA_DIR / "test_ood.jsonl") as f:
            for line in f:
                ood_records.append(json.loads(line))
        print(f"[Phase 2] OOD set (Singapore, for L_DA): {len(ood_records)} records")

    # ── Optimizer & Loss ─────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    mmd_loss_fn = MMDLoss(bandwidths=cfg.mmd_bandwidths, adaptive=True)

    # ── Training ──────────────────────────────────────────────────
    history = []
    model.train()

    for epoch in range(cfg.num_epochs):
        print(f"\n  Epoch {epoch + 1}/{cfg.num_epochs}")
        epoch_loss = 0.0
        epoch_task = 0.0
        epoch_da   = 0.0
        n_steps    = 0

        random.shuffle(id_records)
        if use_da:
            random.shuffle(ood_records)
            ood_ptr = 0

        accum_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        # DA_INTERVAL: compute L_DA every N gradient steps to save memory
        DA_INTERVAL = 4
        id_feats_buffer  = []  # accumulated ID text features for MMD
        ood_feats_buffer = []  # accumulated OOD text features for MMD

        pbar = tqdm(range(0, len(id_records), 1),   # batch_size=1 per sample
                    desc=f"  Epoch {epoch + 1}", total=len(id_records))

        for step_idx in pbar:
            rec = id_records[step_idx]

            # ── Single forward pass: task loss + ID hidden states ──
            try:
                inputs = prepare_single(rec, processor, device, include_labels=True)
            except Exception:
                continue

            need_hs = use_da and (n_steps % DA_INTERVAL == 0 or
                                   len(id_feats_buffer) < 4)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs,
                                output_hidden_states=need_hs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            loss_task = out.loss if out.loss is not None else torch.tensor(0.0, device=device)

            # Collect ID text features (while we have hidden states)
            if need_hs and out.hidden_states is not None:
                feats = extract_text_features(
                    out.hidden_states, inputs["input_ids"],
                    target_layer_outputs, use_txt_mask
                )
                if feats:
                    id_feats_buffer.append({k: v.detach() for k, v in feats.items()})

            del out
            torch.cuda.empty_cache()

            # ── Collect OOD features (no_grad — OOD is fixed reference) ──
            loss_da = torch.tensor(0.0, device=device)
            if use_da and need_hs and ood_records:
                ood_rec = ood_records[ood_ptr % len(ood_records)]
                ood_ptr += 1
                try:
                    ood_inputs = prepare_single(ood_rec, processor, device,
                                                include_labels=False)
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        ood_out = model(**ood_inputs, output_hidden_states=True)
                    if ood_out.hidden_states is not None:
                        ood_feats = extract_text_features(
                            ood_out.hidden_states, ood_inputs["input_ids"],
                            target_layer_outputs, use_txt_mask
                        )
                        if ood_feats:
                            ood_feats_buffer.append(ood_feats)
                    del ood_out
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # ── Compute MMD when buffer is ready ──
            if (use_da and len(id_feats_buffer) >= 4
                    and len(ood_feats_buffer) >= 4
                    and n_steps % DA_INTERVAL == 0):
                loss_da = compute_mmd_from_features(
                    id_feats_buffer, ood_feats_buffer,
                    target_layer_outputs, mmd_loss_fn, device
                )
                id_feats_buffer.clear()
                ood_feats_buffer.clear()

            # ── Total loss + backward ──
            loss = loss_task + cfg.lambda_da * loss_da
            scaled = loss / cfg.gradient_accumulation_steps
            scaled.backward()

            epoch_loss += loss.item()
            epoch_task += loss_task.item()
            epoch_da   += loss_da.item()
            n_steps    += 1

            if n_steps % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if n_steps % cfg.logging_steps == 0:
                pbar.set_postfix({
                    "loss": f"{epoch_loss/n_steps:.4f}",
                    "task": f"{epoch_task/n_steps:.4f}",
                    "da":   f"{epoch_da/n_steps:.4f}",
                })

        # Flush remaining gradients
        if n_steps % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        metrics = {
            "epoch":     epoch + 1,
            "loss":      epoch_loss / max(n_steps, 1),
            "loss_task": epoch_task / max(n_steps, 1),
            "loss_da":   epoch_da   / max(n_steps, 1),
        }
        history.append(metrics)
        print(f"  → loss={metrics['loss']:.4f}  "
              f"L_task={metrics['loss_task']:.4f}  "
              f"L_DA={metrics['loss_da']:.4f}")

        # Save checkpoint
        ckpt_dir = PHASE2_OUTPUT / "checkpoints" / strategy / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        processor.save_pretrained(str(ckpt_dir))
        print(f"  Checkpoint saved: {ckpt_dir}")

    # ── Merge LoRA and save ───────────────────────────────────────
    if strategy != "txt_da_no_lora":
        print(f"\n[Phase 2] Merging LoRA weights...")
        merged = model.merge_and_unload()
    else:
        merged = model

    final_dir = PHASE2_OUTPUT / "checkpoints" / f"{strategy}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[Phase 2] Saved merged model: {final_dir}")

    # Save history
    PHASE2_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_OUTPUT / f"history_{strategy}.json", "w") as f:
        json.dump(history, f, indent=2)

    del model, merged
    torch.cuda.empty_cache()

    return str(final_dir)


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

STRATEGIES = [
    "cis_selective_lora",    # MAIN method
    "full_lora_da",          # Ablation: no CIS selectivity
    "random_selective_lora", # Ablation: random vs CIS
    "full_lora_no_da",       # Ablation: no domain alignment
    "txt_da_no_lora",        # Ablation: alignment without LoRA
]


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Selective Text-Channel DA")
    parser.add_argument("--strategy", type=str, default="cis_selective_lora",
                        choices=STRATEGIES + ["all"])
    parser.add_argument("--lambda_da",   type=float, default=None)
    parser.add_argument("--num_epochs",  type=int,   default=None)
    parser.add_argument("--theta",       type=float, default=None,
                        help="CIS_txt threshold for layer selection (default=0.5)")
    args = parser.parse_args()

    cfg = VLConfig()
    if args.lambda_da  is not None: cfg.lambda_da  = args.lambda_da
    if args.num_epochs is not None: cfg.num_epochs = args.num_epochs

    global CIS_TXT_THETA
    if args.theta is not None:
        CIS_TXT_THETA = args.theta

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2 (VL) — Text-Channel Selective Domain Alignment")
    print(f"  CIS_txt threshold: {CIS_TXT_THETA}")
    print(f"  lambda_DA: {cfg.lambda_da}")
    print(f"  num_epochs: {cfg.num_epochs}")
    print(f"{'=' * 70}")

    if args.strategy == "all":
        results = {}
        for s in STRATEGIES:
            path = run_strategy(s, cfg)
            results[s] = path
        print(f"\n{'=' * 70}")
        print(f"  ALL STRATEGIES COMPLETE")
        for s, p in results.items():
            print(f"  {s:>25}: {p}")
    else:
        run_strategy(args.strategy, cfg)


if __name__ == "__main__":
    main()

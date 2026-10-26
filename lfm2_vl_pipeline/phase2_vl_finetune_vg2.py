"""
Phase 2 (VL) — CIS-Guided Three-Way Selective Repair for VG v2
================================================================

Adapted for the VG v2 model (3-frame, image-only) with three-way partition:

  phi_align  (L0-1):  High CIS_img → MMD on IMAGE features (align visual domains)
  phi_tune   (L2-9):  High CIS_txt → LoRA retraining (fix text-channel bias)
  phi_freeze (L10-15): Low both    → Freeze (preserve task knowledge)

L_total = L_task + lambda_mmd * SUM_{l in phi_align} MMD(h_img_l^ID, h_img_l^OOD)

Strategies (ablation):
  cis_selective  — Three-way: MMD on phi_align + LoRA on phi_tune + freeze (MAIN)
  full_lora_da   — LoRA on all layers + global MMD (ablation: no CIS selectivity)
  full_lora_only — LoRA on all layers, no MMD (ablation: no alignment)
  mmd_only       — MMD on all layers, no LoRA (ablation: alignment without retraining)

Usage:
  cd /mnt/vdb/LFM_project
  python lfm2_vl_pipeline/phase2_vl_finetune_vg2.py --strategy cis_selective
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
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    NUM_LAYERS,
    LAYER_TYPES,
)
from visual_grounding_v2_finetune import build_temporal_map
from utils.mmd_loss import MMDLoss

# ── Token IDs ────────────────────────────────────────────────────
IMAGE_TOKEN_ID = 396   # <image>
IMAGE_START_ID = 498   # <|image_start|>
IMAGE_END_ID   = 499   # <|image_end|>

# ── Paths ─────────────────────────────────────────────────────────
PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl_vg2"
PHASE2_OUTPUT = OUTPUT_DIR / "phase2_vl_vg2"

# ── VG v2 input format ──────────────────────────────────────────
INSTRUCTION = "Predict future trajectory as 6 waypoints (x,y) in ego-frame metres at t+0.5s to t+3.0s."
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")
TARGET_SIZE = (512, 512)


# ════════════════════════════════════════════════════════════════
#  Token Masks
# ════════════════════════════════════════════════════════════════

def get_image_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Boolean mask for image token positions."""
    seq_len = input_ids.shape[0]
    img_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    start_pos = (input_ids == IMAGE_START_ID).nonzero(as_tuple=True)[0]
    end_pos   = (input_ids == IMAGE_END_ID).nonzero(as_tuple=True)[0]
    if len(start_pos) > 0 and len(end_pos) > 0:
        img_mask[start_pos[0]:end_pos[-1] + 1] = True
    else:
        img_mask = (input_ids == IMAGE_TOKEN_ID)
    return img_mask


def get_text_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return ~get_image_mask(input_ids)


# ════════════════════════════════════════════════════════════════
#  Load Three-Way Partition
# ════════════════════════════════════════════════════════════════

def load_partition():
    """Load layer partition from Phase 1 VG v2 CIS diagnosis."""
    partition_path = PHASE1_OUTPUT / "layer_partition.json"
    if partition_path.exists():
        with open(partition_path) as f:
            p = json.load(f)
        print(f"[Phase 2] Loaded partition from {partition_path}")
        print(f"  phi_align  (MMD):  {p['phi_align']}")
        print(f"  phi_tune   (LoRA): {p['phi_tune']}")
        print(f"  phi_freeze:        {p['phi_freeze']}")
        return p

    # Fallback if partition doesn't exist
    print("[Phase 2] WARNING: No partition found, using default")
    return {
        "phi_align": [0, 1],
        "phi_tune": [2, 3, 4, 5, 6, 7, 8, 9],
        "phi_freeze": [10, 11, 12, 13, 14, 15],
    }


# ════════════════════════════════════════════════════════════════
#  3-Frame Input Preparation (matches VG v2 training)
# ════════════════════════════════════════════════════════════════

def prepare_input_3frame(rec, processor, temporal_map, device, include_labels=True):
    """Prepare 3-frame input matching VG v2 format."""
    token = rec.get("sample_token")
    if not token or token not in temporal_map:
        return None

    prev_path, curr_path, next_path = temporal_map[token]
    images = []
    for p in [prev_path, curr_path, next_path]:
        try:
            images.append(Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS))
        except Exception:
            images.append(Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)))

    question = TEMPORAL_PREFIX + INSTRUCTION

    conv = [{"role": "user", "content": [
        {"type": "image", "image": images[0]},
        {"type": "image", "image": images[1]},
        {"type": "image", "image": images[2]},
        {"type": "text", "text": question},
    ]}]

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
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    if include_labels:
        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels

    return inputs


# ════════════════════════════════════════════════════════════════
#  LoRA Model Creation
# ════════════════════════════════════════════════════════════════

def create_lora_model(model, cfg, strategy, partition):
    """Apply LoRA adapters to strategy-specific layers."""
    if strategy == "mmd_only":
        print("[Phase 2] mmd_only: skipping LoRA")
        return model, []

    if strategy == "cis_selective":
        # LoRA only on phi_tune (high CIS_txt) layers
        layers_to_transform = partition["phi_tune"]
        print(f"[Phase 2] CIS Selective LoRA on phi_tune layers: {layers_to_transform}")
    elif strategy in ("full_lora_da", "full_lora_only"):
        layers_to_transform = list(range(NUM_LAYERS))
        print(f"[Phase 2] Full LoRA: all {NUM_LAYERS} layers")
    elif strategy == "random_partition":
        # Same sizes as CIS partition but randomly assigned
        layers_to_transform = partition["rand_phi_tune"]
        print(f"[Phase 2] Random Partition LoRA on rand_phi_tune: {layers_to_transform}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "w1", "w2", "w3",
            "in_proj",
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
#  Feature Extraction for MMD
# ════════════════════════════════════════════════════════════════

def extract_image_features(hidden_states, input_ids, target_layers):
    """
    Extract mean-pooled IMAGE-channel hidden states at target layers.
    For phi_align: we align image representations across domains.
    """
    feats = {}
    for l in target_layers:
        if l >= len(hidden_states):
            continue
        h = hidden_states[l][0]  # [seq, D]
        img_mask = get_image_mask(input_ids[0])
        if img_mask.sum() == 0:
            continue
        rep = h[img_mask.to(h.device)].mean(0).float()
        feats[l] = F.normalize(rep, dim=-1)
    return feats


def compute_mmd_from_features(id_feats_list, ood_feats_list,
                               target_layers, mmd_loss_fn, device):
    """Compute mean MMD across target layers."""
    loss_da = torch.tensor(0.0, device=device)
    n_valid = 0
    for l in target_layers:
        id_vecs  = [f[l] for f in id_feats_list  if l in f]
        ood_vecs = [f[l] for f in ood_feats_list if l in f]
        if len(id_vecs) < 2 or len(ood_vecs) < 2:
            continue
        h_id  = torch.stack(id_vecs).to(device)
        h_ood = torch.stack(ood_vecs).to(device)
        loss_da = loss_da + mmd_loss_fn(h_id, h_ood)
        n_valid += 1
    if n_valid > 0:
        loss_da = loss_da / n_valid
    return loss_da


# ════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════

def run_strategy(strategy, cfg, temporal_map):
    """Run Phase 2 fine-tuning with a specific strategy."""
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2 (VG v2): {strategy.upper()}")
    print(f"  Three-Way CIS-Guided Selective Repair")
    print(f"{'=' * 70}")

    use_mmd = strategy in ("cis_selective", "full_lora_da", "mmd_only", "random_partition")

    # ── Load partition ───────────────────────────────────────────
    partition = load_partition()

    # For random_partition: generate random 3-way split with same sizes
    if strategy == "random_partition":
        cis_partition = partition
        n_align = len(cis_partition["phi_align"])   # 2
        n_tune  = len(cis_partition["phi_tune"])    # 8
        all_layers = list(range(NUM_LAYERS))
        rng = random.Random(42)  # fixed seed for reproducibility
        rng.shuffle(all_layers)
        partition = {
            "rand_phi_align": sorted(all_layers[:n_align]),
            "rand_phi_tune":  sorted(all_layers[n_align:n_align + n_tune]),
            "rand_phi_freeze": sorted(all_layers[n_align + n_tune:]),
        }
        print(f"[Phase 2] Random partition:")
        print(f"  rand_phi_align  (MMD):  {partition['rand_phi_align']}")
        print(f"  rand_phi_tune   (LoRA): {partition['rand_phi_tune']}")
        print(f"  rand_phi_freeze:        {partition['rand_phi_freeze']}")

    # MMD target layers
    if strategy == "cis_selective":
        mmd_layers = partition["phi_align"]  # MMD on image features at high CIS_img layers
    elif strategy in ("full_lora_da", "mmd_only"):
        mmd_layers = list(range(NUM_LAYERS))  # global MMD (ablation)
    elif strategy == "random_partition":
        mmd_layers = partition["rand_phi_align"]
    else:
        mmd_layers = []

    # ── Load VG v2 base model ────────────────────────────────────
    model_dir = str(CHECKPOINT_DIR / "visual_grounding_v2" / "vg2_merged")
    print(f"\n[Phase 2] Loading VG v2 model from {model_dir}")

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

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # ── Load data ────────────────────────────────────────────────
    id_records = [json.loads(l) for l in open(DATA_DIR / "train.jsonl")]
    # Filter to records with temporal mapping
    id_records = [r for r in id_records if r.get("sample_token") in temporal_map]
    print(f"[Phase 2] ID train set (Boston): {len(id_records)} records (with temporal)")

    ood_records = []
    if use_mmd:
        ood_records = [json.loads(l) for l in open(DATA_DIR / "test_ood.jsonl")]
        ood_records = [r for r in ood_records if r.get("sample_token") in temporal_map]
        print(f"[Phase 2] OOD set (Singapore, for MMD): {len(ood_records)} records (with temporal)")

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

    DA_INTERVAL = 4
    num_epochs = cfg.num_epochs

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_task = 0.0
        epoch_mmd  = 0.0
        n_steps    = 0

        random.shuffle(id_records)
        if use_mmd:
            random.shuffle(ood_records)
            ood_ptr = 0

        id_feats_buffer  = []
        ood_feats_buffer = []
        optimizer.zero_grad()

        pbar = tqdm(range(len(id_records)), desc=f"  Epoch {epoch + 1}")

        for step_idx in pbar:
            rec = id_records[step_idx]

            try:
                inputs = prepare_input_3frame(rec, processor, temporal_map, device,
                                               include_labels=True)
            except Exception:
                continue
            if inputs is None:
                continue

            need_hs = use_mmd and (n_steps % DA_INTERVAL == 0 or len(id_feats_buffer) < 4)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs, output_hidden_states=need_hs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            loss_task = out.loss if out.loss is not None else torch.tensor(0.0, device=device)

            # Collect ID image features for MMD at phi_align layers
            if need_hs and out.hidden_states is not None:
                feats = extract_image_features(
                    out.hidden_states, inputs["input_ids"], mmd_layers
                )
                if feats:
                    id_feats_buffer.append({k: v.detach() for k, v in feats.items()})

            del out
            torch.cuda.empty_cache()

            # ── Collect OOD image features ──
            loss_mmd = torch.tensor(0.0, device=device)
            if use_mmd and need_hs and ood_records:
                ood_rec = ood_records[ood_ptr % len(ood_records)]
                ood_ptr += 1
                try:
                    ood_inputs = prepare_input_3frame(
                        ood_rec, processor, temporal_map, device, include_labels=False
                    )
                    if ood_inputs is not None:
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            ood_out = model(**ood_inputs, output_hidden_states=True)
                        if ood_out.hidden_states is not None:
                            ood_feats = extract_image_features(
                                ood_out.hidden_states, ood_inputs["input_ids"], mmd_layers
                            )
                            if ood_feats:
                                ood_feats_buffer.append(ood_feats)
                        del ood_out
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # ── Compute MMD when buffer is ready ──
            if (use_mmd and len(id_feats_buffer) >= 4
                    and len(ood_feats_buffer) >= 4
                    and n_steps % DA_INTERVAL == 0):
                loss_mmd = compute_mmd_from_features(
                    id_feats_buffer, ood_feats_buffer,
                    mmd_layers, mmd_loss_fn, device
                )
                id_feats_buffer.clear()
                ood_feats_buffer.clear()

            # ── Total loss ──
            loss = loss_task + cfg.lambda_da * loss_mmd
            scaled = loss / cfg.gradient_accumulation_steps
            scaled.backward()

            epoch_loss += loss.item()
            epoch_task += loss_task.item()
            epoch_mmd  += loss_mmd.item()
            n_steps    += 1

            if n_steps % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if n_steps % cfg.logging_steps == 0:
                pbar.set_postfix({
                    "loss": f"{epoch_loss/n_steps:.4f}",
                    "task": f"{epoch_task/n_steps:.4f}",
                    "mmd":  f"{epoch_mmd/n_steps:.4f}",
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
            "loss_mmd":  epoch_mmd  / max(n_steps, 1),
        }
        history.append(metrics)
        print(f"  -> loss={metrics['loss']:.4f}  "
              f"L_task={metrics['loss_task']:.4f}  "
              f"L_MMD={metrics['loss_mmd']:.4f}")

        # Save checkpoint
        ckpt_dir = PHASE2_OUTPUT / "checkpoints" / strategy / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        processor.save_pretrained(str(ckpt_dir))
        print(f"  Checkpoint saved: {ckpt_dir}")

    # ── Merge LoRA and save ──────────────────────────────────────
    if strategy != "mmd_only":
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
    "cis_selective",     # MAIN: MMD on phi_align + LoRA on phi_tune + freeze
    "full_lora_da",      # Ablation: LoRA all + global MMD
    "full_lora_only",    # Ablation: LoRA all, no MMD
    "mmd_only",          # Ablation: MMD all, no LoRA
    "random_partition",  # Ablation: same partition sizes but random layer assignment
]


def main():
    parser = argparse.ArgumentParser(description="Phase 2 VG v2: Three-Way Selective Repair")
    parser.add_argument("--strategy", type=str, default="cis_selective",
                        choices=STRATEGIES + ["all"])
    parser.add_argument("--lambda_da",  type=float, default=None)
    parser.add_argument("--num_epochs", type=int,   default=None)
    args = parser.parse_args()

    cfg = VLConfig()
    if args.lambda_da  is not None: cfg.lambda_da  = args.lambda_da
    if args.num_epochs is not None: cfg.num_epochs = args.num_epochs

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2 (VG v2) — Three-Way CIS-Guided Selective Repair")
    print(f"  lambda_MMD: {cfg.lambda_da}")
    print(f"  num_epochs: {cfg.num_epochs}")
    print(f"{'=' * 70}")

    # Load nuScenes temporal map (shared across strategies)
    print("Loading nuScenes for temporal frames...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    temporal_map = build_temporal_map(nusc, 'data/nuscenes')
    del nusc
    print(f"Temporal map: {len(temporal_map)} samples")

    if args.strategy == "all":
        results = {}
        for s in STRATEGIES:
            path = run_strategy(s, cfg, temporal_map)
            results[s] = path
        print(f"\n{'=' * 70}")
        print(f"  ALL STRATEGIES COMPLETE")
        for s, p in results.items():
            print(f"  {s:>20}: {p}")
    else:
        run_strategy(args.strategy, cfg, temporal_map)


if __name__ == "__main__":
    main()

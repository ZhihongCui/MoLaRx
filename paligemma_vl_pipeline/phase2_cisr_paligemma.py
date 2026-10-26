"""Phase 2 — CIS-Guided Selective Repair (CISR) on PaliGemma2-3B.

Adapts to architecture:
  - Causal VLMs: 3-zone (image/text/invariant) → MMD + LoRA + freeze
  - Prefix-LM (PaliGemma typical): 2-zone (image/invariant)
    → CISR degenerates to MMD on image + freeze; no LoRA since phi_tune={}

Usage:
  cd /mnt/vdb/LFM_project
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python paligemma_vl_pipeline/phase2_cisr_paligemma.py \
      --dataset nuscenes --strategy cis_selective
"""
import argparse, json, os, random, sys, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "paligemma_vl_pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "lfm2_vl_pipeline"))

from config_paligemma import (
    PaliGemmaConfig, CHECKPOINT_DIR, OUTPUT_DIR,
    NUS_DATA_DIR, AV2_DATA_DIR, NUM_LAYERS, IMAGE_TOKEN_ID,
)
from paligemma_common import (
    prepare_input_3frame, get_image_mask,
    build_temporal_map_for_dataset,
)
from utils.mmd_loss import MMDLoss


STRATEGIES = ["cis_selective", "full_lora_da", "full_lora_only", "mmd_only"]


# ════════════════════════════════════════════════════════════════
#  Partition loader
# ════════════════════════════════════════════════════════════════

def load_partition(dataset):
    path = OUTPUT_DIR / f"phase1_cis_{dataset}" / "paligemma_cis_results.json"
    if path.exists():
        with open(path) as f:
            results = json.load(f)
        p = results.get("partition", {})
        if "phi_align" in p:
            return {k: p[k] for k in ("phi_align", "phi_tune", "phi_freeze")}
    # Fallback
    t1, t2 = NUM_LAYERS // 6, 3 * NUM_LAYERS // 4
    print(f"[Phase 2] WARNING: no CIS results; fallback partition")
    return {"phi_align": list(range(t1)),
            "phi_tune": list(range(t1, t2)),
            "phi_freeze": list(range(t2, NUM_LAYERS))}


# ════════════════════════════════════════════════════════════════
#  LoRA (adaptive to empty phi_tune)
# ════════════════════════════════════════════════════════════════

def create_lora_model(model, cfg, strategy, partition):
    if strategy == "mmd_only":
        print("[Phase 2] mmd_only: no LoRA"); return model, []
    if strategy == "cis_selective":
        layers = partition["phi_tune"]
        if not layers:
            print("[Phase 2] cis_selective: phi_tune is empty (prefix-LM style). "
                  "Skipping LoRA; CISR reduces to MMD + freeze.")
            return model, []
        print(f"[Phase 2] cis_selective LoRA on phi_tune = {layers}")
    elif strategy in ("full_lora_da", "full_lora_only"):
        layers = list(range(NUM_LAYERS))
        print(f"[Phase 2] Full LoRA on all {NUM_LAYERS} layers")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    peft_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        layers_to_transform=layers, layers_pattern="layers",
        task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, layers


def extract_image_features(hidden_states, input_ids, phi_align):
    feats = {}
    for l in phi_align:
        if l >= len(hidden_states): continue
        h = hidden_states[l][0]
        mask = get_image_mask(input_ids)
        if mask.sum() == 0: continue
        rep = h[mask.to(h.device)].mean(0).float()
        feats[l] = F.normalize(rep, dim=-1)
    return feats


def compute_mmd_loss(id_feats_list, ood_feats_list, phi_align, mmd_loss_fn, device):
    loss = torch.tensor(0.0, device=device)
    n = 0
    for l in phi_align:
        id_v  = [f[l] for f in id_feats_list  if l in f]
        ood_v = [f[l] for f in ood_feats_list if l in f]
        if len(id_v) < 2 or len(ood_v) < 2: continue
        h_id, h_ood = torch.stack(id_v).to(device), torch.stack(ood_v).to(device)
        loss = loss + mmd_loss_fn(h_id, h_ood)
        n += 1
    return loss / max(n, 1)


def run_strategy(strategy, dataset, cfg, max_samples=-1):
    print(f"\n{'='*70}")
    print(f"  PaliGemma2 CISR — {strategy} on {dataset}")
    print(f"{'='*70}")
    sys.stdout.flush()

    use_mmd = strategy in ("cis_selective", "full_lora_da", "mmd_only")
    partition = load_partition(dataset)
    phi_align = partition["phi_align"]
    print(f"  phi_align ={phi_align}")
    print(f"  phi_tune  ={partition['phi_tune']}")
    print(f"  phi_freeze={partition['phi_freeze']}")
    sys.stdout.flush()

    data_dir = NUS_DATA_DIR if dataset == "nuscenes" else AV2_DATA_DIR
    tvg_ckpt = CHECKPOINT_DIR / f"tvg_{dataset}" / "tvg_merged"
    out_dir  = OUTPUT_DIR / f"phase2_cisr_{dataset}"

    print(f"\n[Phase 2] Building temporal_map for {dataset} ...")
    t0 = time.time()
    temporal_map = build_temporal_map_for_dataset(dataset)
    if temporal_map is not None:
        print(f"  {len(temporal_map):,} entries ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    print(f"\n[Phase 2] Loading TVG model: {tvg_ckpt}")
    sys.stdout.flush()
    processor = PaliGemmaProcessor.from_pretrained(str(tvg_ckpt))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(tvg_ckpt), torch_dtype=torch.bfloat16, device_map="auto")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    device = next(model.parameters()).device
    print(f"  base mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    sys.stdout.flush()

    model, _ = create_lora_model(model, cfg, strategy, partition)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    id_records = [json.loads(l) for l in open(data_dir / "train.jsonl")]
    if max_samples > 0 and len(id_records) > max_samples:
        id_records = random.sample(id_records, max_samples)
    print(f"[Phase 2] Source labeled: {len(id_records)}")
    ood_records = []
    if use_mmd:
        ood_records = [json.loads(l) for l in open(data_dir / "test_ood.jsonl")]
        print(f"[Phase 2] Target unlabeled: {len(ood_records)}")
    sys.stdout.flush()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable, lr=cfg.learning_rate,
                                            weight_decay=cfg.weight_decay)
            print("[Phase 2] Using 8-bit AdamW (bnb)")
        except Exception as _e:
            print(f"[Phase 2] bnb AdamW8bit unavailable ({_e}); fp32 AdamW fallback")
            optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate,
                                           weight_decay=cfg.weight_decay)
    else:
        optimizer = None
        print("[Phase 2] No trainable parameters — inference-only (MMD-only or empty phi_tune)")
    mmd_loss_fn = MMDLoss(bandwidths=[0.1, 1.0, 10.0], adaptive=True)

    DA_INTERVAL = 4
    id_buf, ood_buf = [], []
    history = []
    skip_stats = Counter()
    MAX_PRINTED = 3

    def _skip(r, e=None, ctx=""):
        skip_stats[r] += 1
        if skip_stats[r] <= MAX_PRINTED:
            print(f"  [SKIP][{r}] {ctx}{': '+repr(e) if e else ''}")
            if e is not None and skip_stats[r] == 1: traceback.print_exc()
            sys.stdout.flush()

    model.train()
    for epoch in range(cfg.num_epochs):
        print(f"\n  Epoch {epoch+1}/{cfg.num_epochs}")
        sys.stdout.flush()
        epoch_loss = epoch_task = epoch_mmd = 0.0
        n_steps = 0
        random.shuffle(id_records)
        if use_mmd: random.shuffle(ood_records)
        ood_ptr = 0
        if optimizer: optimizer.zero_grad()

        pbar = tqdm(range(len(id_records)), desc=f"Epoch {epoch+1}")
        for i in pbar:
            rec = id_records[i]
            try:
                inputs = prepare_input_3frame(rec, processor, device,
                                                temporal_map=temporal_map, include_labels=True)
            except Exception as e:
                _skip("prepare_exc", e, f"src {i}"); continue
            if inputs is None:
                _skip("prepare_none", None, f"src {i}"); continue

            need_hs = use_mmd and (n_steps % DA_INTERVAL == 0 or len(id_buf) < 4)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs, output_hidden_states=need_hs)
            except torch.cuda.OutOfMemoryError as e:
                _skip("oom_src", e, f"src {i}"); torch.cuda.empty_cache(); continue
            except Exception as e:
                _skip("forward_src_exc", e, f"src {i}"); continue
            loss_task = out.loss if out.loss is not None else torch.tensor(0.0, device=device)
            if need_hs and getattr(out, "hidden_states", None) is not None:
                feats = extract_image_features(out.hidden_states, inputs["input_ids"], phi_align)
                if feats: id_buf.append({k: v.detach() for k, v in feats.items()})
            del out
            torch.cuda.empty_cache()

            loss_mmd = torch.tensor(0.0, device=device)
            if use_mmd and need_hs and ood_records:
                try:
                    or_ = ood_records[ood_ptr % len(ood_records)]; ood_ptr += 1
                    oi = prepare_input_3frame(or_, processor, device,
                                                temporal_map=temporal_map, include_labels=False)
                    if oi is not None:
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            oo = model(**oi, output_hidden_states=True)
                        if getattr(oo, "hidden_states", None) is not None:
                            feats = extract_image_features(oo.hidden_states, oi["input_ids"], phi_align)
                            if feats: ood_buf.append(feats)
                        del oo
                        torch.cuda.empty_cache()
                except Exception as e:
                    _skip("ood_forward_exc", e, f"src {i}")
            if (use_mmd and len(id_buf) >= 4 and len(ood_buf) >= 4
                    and n_steps % DA_INTERVAL == 0):
                try:
                    loss_mmd = compute_mmd_loss(id_buf, ood_buf, phi_align, mmd_loss_fn, device)
                    id_buf.clear(); ood_buf.clear()
                except Exception as e:
                    _skip("mmd_exc", e, f"src {i}")

            loss = loss_task + 0.5 * loss_mmd
            if optimizer is not None:
                try:
                    scaled = loss / cfg.gradient_accumulation_steps
                    scaled.backward()
                except Exception as e:
                    _skip("backward_exc", e, f"src {i}"); continue

            epoch_loss += loss.item(); epoch_task += loss_task.item(); epoch_mmd += float(loss_mmd)
            n_steps += 1
            if optimizer and n_steps % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()
            if n_steps % 10 == 0:
                pbar.set_postfix({"loss": f"{epoch_loss/n_steps:.4f}",
                                  "task": f"{epoch_task/n_steps:.4f}",
                                  "mmd":  f"{epoch_mmd/n_steps:.4f}",
                                  "skip": sum(skip_stats.values())})

        if optimizer and n_steps % cfg.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()

        print(f"  Epoch {epoch+1}: steps={n_steps}, skips={dict(skip_stats)}")
        if n_steps == 0:
            print(f"  [FATAL] 0 steps."); sys.exit(2)
        metrics = {"epoch": epoch+1, "loss": epoch_loss/n_steps,
                   "loss_task": epoch_task/n_steps, "loss_mmd": epoch_mmd/n_steps,
                   "skips": dict(skip_stats)}
        history.append(metrics)
        print(f"  → {metrics}")
        sys.stdout.flush()
        ep_dir = out_dir / "checkpoints" / strategy / f"epoch_{epoch+1}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ep_dir))
        processor.save_pretrained(str(ep_dir))

    if strategy != "mmd_only" and hasattr(model, "merge_and_unload"):
        print("\n[Phase 2] Merging LoRA ...")
        merged = model.merge_and_unload()
    else:
        merged = model
    final = out_dir / "checkpoints" / f"{strategy}_final"
    final.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"history_{strategy}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Phase 2] Saved to {final}")
    del model, merged
    torch.cuda.empty_cache()
    return str(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nuscenes",
                        choices=["nuscenes", "av2"])
    parser.add_argument("--strategy", type=str, default="cis_selective",
                        choices=STRATEGIES + ["all"])
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = PaliGemmaConfig()
    if args.num_epochs is not None: cfg.num_epochs = args.num_epochs

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)

    if args.strategy == "all":
        for s in STRATEGIES:
            run_strategy(s, args.dataset, cfg, max_samples=args.max_samples)
    else:
        run_strategy(args.strategy, args.dataset, cfg, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

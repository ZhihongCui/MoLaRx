"""Phase 2 — CIS-Guided Selective Repair (CISR) on Qwen2-VL-2B.

Applies:
  - MMD on image tokens for layers in phi_align
  - LoRA retraining on linear projections for layers in phi_tune
  - zero gradient for layers in phi_freeze

Usage:
  cd /mnt/vdb/LFM_project
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python qwen_vl_pipeline/phase2_cisr_qwen.py \
      --dataset nuscenes --strategy cis_selective
"""
import argparse, json, os, random, sys, time, traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "qwen_vl_pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "lfm2_vl_pipeline"))

from config_qwen import (
    QwenVLConfig, QWEN_MODEL_ID,
    DATA_DIR as NUS_DATA_DIR,
    CHECKPOINT_DIR, OUTPUT_DIR, NUM_LAYERS,
)
from utils.mmd_loss import MMDLoss

AV2_DATA_DIR = Path("argoverse2_pipeline/outputs/data")
IMAGE_TOKEN_ID = 151655   # Qwen2-VL <|image_pad|>
TARGET_SIZE = (256, 256)
INSTRUCTION = ("Predict future trajectory as 6 waypoints (x,y) "
               "in ego-frame metres at t+0.5s to t+3.0s.")
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")

STRATEGIES = ["cis_selective", "full_lora_da", "full_lora_only", "mmd_only"]


# ════════════════════════════════════════════════════════════════
#  Partition loader (from Qwen's phase1_summary.json)
# ════════════════════════════════════════════════════════════════

def load_partition(dataset):
    """Load partition from Qwen Phase 1 output. Tries several known paths."""
    sub = "phase1_qwen_tvg" if dataset == "nuscenes" else "phase1_qwen_tvg_av2"
    path = OUTPUT_DIR / sub / "layer_partition.json"
    if path.exists():
        with open(path) as f:
            p = json.load(f)
        if "phi_align" in p:
            return {k: p[k] for k in ("phi_align", "phi_tune", "phi_freeze")}
    # Fallback: reconstruct from phase1_summary.json using config thresholds
    summary_path = OUTPUT_DIR / sub / "phase1_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            d = json.load(f)
        cfg = QwenVLConfig()
        tau_img, tau_txt = 0.30, 2.00
        align, tune, freeze = [], [], []
        layers = sorted([k for k in d if k.startswith("layer_")],
                        key=lambda x: int(x.split("_")[1]))
        for l in layers:
            li = int(l.split("_")[1]) - 1
            img = d[l].get("cis_img_mean", 0.0)
            txt = d[l].get("cis_txt_mean", 0.0)
            if img > tau_img and img >= txt:
                align.append(li)
            elif txt > tau_txt:
                tune.append(li)
            else:
                freeze.append(li)
        return {"phi_align": align, "phi_tune": tune, "phi_freeze": freeze}
    # Last-resort fallback
    t1, t2 = NUM_LAYERS // 6, 3 * NUM_LAYERS // 4
    print(f"[Phase 2] WARNING: no CIS results; fallback partition")
    return {"phi_align": list(range(t1)),
            "phi_tune": list(range(t1, t2)),
            "phi_freeze": list(range(t2, NUM_LAYERS))}


# ════════════════════════════════════════════════════════════════
#  Data loading (adapts to nuscenes or av2)
# ════════════════════════════════════════════════════════════════

def resolve_three_frames(rec, temporal_map=None):
    if temporal_map is not None:
        token = rec.get("sample_token")
        if token and token in temporal_map:
            return temporal_map[token]
    ips = rec.get("image_paths")
    if isinstance(ips, (list, tuple)) and len(ips) == 3:
        return tuple(ips)
    ip = rec.get("image_path")
    if ip:
        return (ip, ip, ip)
    return None


def build_nuscenes_temporal_map(dataroot="data/nuscenes"):
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)
    tm = {}
    for sample in nusc.sample:
        tok = sample["token"]
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        curr = os.path.join(dataroot, cam["filename"])
        prev_p = curr
        if sample["prev"]:
            try:
                pc = nusc.get("sample_data", nusc.get("sample", sample["prev"])["data"]["CAM_FRONT"])
                prev_p = os.path.join(dataroot, pc["filename"])
            except Exception: pass
        next_p = curr
        if sample["next"]:
            try:
                nc = nusc.get("sample_data", nusc.get("sample", sample["next"])["data"]["CAM_FRONT"])
                next_p = os.path.join(dataroot, nc["filename"])
            except Exception: pass
        tm[tok] = (prev_p, curr, next_p)
    return tm


def prepare_input_3frame(rec, processor, device, temporal_map=None, include_labels=True):
    three = resolve_three_frames(rec, temporal_map)
    if three is None: return None
    images = []
    for p in three:
        try:
            img = Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", TARGET_SIZE, color=(128, 128, 128))
        images.append(img)
    question = TEMPORAL_PREFIX + INSTRUCTION
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "image", "image": images[2]},
            {"type": "text", "text": question},
        ],
    }]
    if include_labels:
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": rec["answer"]}]})
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=not include_labels,
        return_tensors="pt", return_dict=True, tokenize=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if include_labels:
        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        labels[0, inputs["input_ids"][0] == IMAGE_TOKEN_ID] = -100
        inputs["labels"] = labels
    return inputs


def get_image_mask(input_ids):
    if input_ids.dim() > 1: input_ids = input_ids[0]
    return (input_ids == IMAGE_TOKEN_ID)


# ════════════════════════════════════════════════════════════════
#  LoRA / MMD
# ════════════════════════════════════════════════════════════════

def create_lora_model(model, cfg, strategy, partition):
    if strategy == "mmd_only":
        print("[Phase 2] mmd_only: no LoRA"); return model, []
    if strategy == "cis_selective":
        layers = partition["phi_tune"]
        if not layers:
            print("[Phase 2] cis_selective but phi_tune is empty (architecture has no text zone). "
                  "No LoRA will be applied (CISR degenerates to MMD + freeze).")
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


# ════════════════════════════════════════════════════════════════
#  Main training loop
# ════════════════════════════════════════════════════════════════

def run_strategy(strategy, dataset, cfg, max_samples=-1):
    print(f"\n{'='*70}")
    print(f"  Qwen2-VL CISR — {strategy} on {dataset}")
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
    tvg_sub  = "tvg_qwen" if dataset == "nuscenes" else "tvg_qwen_av2"
    tvg_ckpt = CHECKPOINT_DIR / tvg_sub / "tvg_merged"
    out_dir  = OUTPUT_DIR / f"phase2_cisr_{dataset}"

    # Build temporal_map for nuScenes (AV2 has image_paths directly)
    temporal_map = None
    if dataset == "nuscenes":
        print(f"\n[Phase 2] Building nuScenes temporal_map ...")
        t0 = time.time()
        temporal_map = build_nuscenes_temporal_map()
        print(f"  {len(temporal_map):,} entries ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    print(f"\n[Phase 2] Loading TVG model: {tvg_ckpt}")
    sys.stdout.flush()
    processor = AutoProcessor.from_pretrained(str(tvg_ckpt))
    processor.image_processor.min_pixels = 256 * 256
    processor.image_processor.max_pixels = 256 * 256
    model = Qwen2VLForConditionalGeneration.from_pretrained(
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
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate,
                                       weight_decay=cfg.weight_decay)
    else:
        optimizer = None
        print("[Phase 2] No trainable parameters (mmd_only or empty phi_tune) — inference-only mode")
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
            print(f"  [FATAL] 0 steps. Inspect skips."); sys.exit(2)
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

    cfg = QwenVLConfig()
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

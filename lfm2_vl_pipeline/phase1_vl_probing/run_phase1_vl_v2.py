"""
Phase 1 (VL) v2 — Modality-Separated DSS-CIS Layer Analysis
=============================================================

Separates image tokens from text tokens and computes:
  - DSS_img(l): Domain Shift Score for image-channel hidden states
  - DSS_txt(l): Domain Shift Score for text-channel hidden states
  - CIS_img(l): Causal Importance Score for image channel
  - CIS_txt(l): Causal Importance Score for text channel

Image tokens: positions between <|image_start|> and <|image_end|>
              (~2300 tokens per sample, pooled to mean vector)
Text tokens:  all other valid (non-image, non-padding) positions
              (~70 tokens per sample, pooled to mean vector)

Regression targets (causal variables):
  - ego_speed: m/s (Boston mean=4.62, Singapore mean=5.71)
  - front_distance: m (Boston mean=6.95, Singapore mean=9.56)
  - lead_velocity: m/s

High DSS + High CIS → layer encodes causal info AND shows domain shift → repair target
Low  CIS_img        → image channel does NOT causally encode domain-relevant info → freeze
High CIS_txt        → text channel encodes causal info with domain shift → Phase 2 target

Usage:
  cd /mnt/vdb/LFM_project
  python lfm2_vl_pipeline/phase1_vl_probing/run_phase1_vl_v2.py
"""

import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_vl import (
    VLConfig,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    NUM_LAYERS,
    HIDDEN_DIM,
    LAYER_TYPES,
)

# ── Token IDs for LFM2-VL ─────────────────────────────────────────
IMAGE_TOKEN_ID    = 396   # <image>
IMAGE_START_ID    = 498   # <|image_start|>
IMAGE_END_ID      = 499   # <|image_end|>

PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl_v2"
MAX_SAMPLES   = 1500   # per city; total ≈ 3000 samples


# ════════════════════════════════════════════════════════════════
#  Step 1: Extract Modality-Separated Hidden States
# ════════════════════════════════════════════════════════════════

def get_modality_masks(input_ids: torch.Tensor):
    """
    Identify image vs text token positions from input_ids.

    Returns:
        img_mask  : bool [seq_len] — True for image token positions
        txt_mask  : bool [seq_len] — True for text token positions
                    (excludes image region AND special framing tokens)
    """
    seq_len = input_ids.shape[0]
    img_mask = torch.zeros(seq_len, dtype=torch.bool)
    txt_mask = torch.zeros(seq_len, dtype=torch.bool)

    # Find image region boundaries
    start_positions = (input_ids == IMAGE_START_ID).nonzero(as_tuple=True)[0]
    end_positions   = (input_ids == IMAGE_END_ID).nonzero(as_tuple=True)[0]

    if len(start_positions) > 0 and len(end_positions) > 0:
        img_start = start_positions[0].item()
        img_end   = end_positions[-1].item()
        # Image region: start to end inclusive (includes framing tokens)
        img_mask[img_start:img_end + 1] = True
    else:
        # Fallback: use IMAGE_TOKEN_ID positions
        img_mask = (input_ids == IMAGE_TOKEN_ID)

    # Text region: everything outside image mask
    txt_mask = ~img_mask

    return img_mask, txt_mask


@torch.no_grad()
def extract_hidden_states_by_modality(model, processor, records, cfg,
                                       max_samples=MAX_SAMPLES):
    """
    Extract mean-pooled hidden states separately for image and text tokens.

    Returns:
        hidden_img : {layer_idx: tensor [N, D]}  — image-channel states
        hidden_txt : {layer_idx: tensor [N, D]}  — text-channel states
        metadata   : list of dicts (city, ego_speed, front_distance, lead_velocity)
    """
    model.eval()

    if len(records) > max_samples:
        random.seed(cfg.seed)
        records = random.sample(records, max_samples)

    # Pre-allocate per-layer lists
    all_img = {l: [] for l in range(NUM_LAYERS + 1)}
    all_txt = {l: [] for l in range(NUM_LAYERS + 1)}
    metadata = []

    n_img_zero = 0  # samples with no image tokens found
    n_txt_zero = 0  # samples with no text tokens found

    for rec in tqdm(records, desc="  Extracting hidden states"):
        # Load image
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # Build conversation (image embedded in content)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": rec["question"]},
                ],
            },
        ]

        try:
            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
        except Exception as e:
            print(f"  [WARN] Tokenization failed: {e}")
            continue

        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        # Forward pass with hidden states
        try:
            outputs = model(**inputs, output_hidden_states=True)
        except Exception as e:
            print(f"  [WARN] Forward pass failed: {e}")
            continue

        hidden_states = outputs.hidden_states  # tuple: (L+1) × [1, seq, D]

        # Get modality masks
        input_ids = inputs["input_ids"][0].cpu()  # [seq_len]
        img_mask, txt_mask = get_modality_masks(input_ids)

        # Also exclude padding from text mask
        if "attention_mask" in inputs:
            valid_mask = inputs["attention_mask"][0].cpu().bool()
            txt_mask = txt_mask & valid_mask

        n_img_tokens = img_mask.sum().item()
        n_txt_tokens = txt_mask.sum().item()

        if n_img_tokens == 0:
            n_img_zero += 1
        if n_txt_tokens == 0:
            n_txt_zero += 1

        # Extract and pool per layer
        for l in range(min(len(hidden_states), NUM_LAYERS + 1)):
            h = hidden_states[l][0].cpu().float()  # [seq_len, D]

            if n_img_tokens > 0:
                h_img = h[img_mask].mean(dim=0)  # [D]
                all_img[l].append(h_img)
            else:
                all_img[l].append(torch.zeros(HIDDEN_DIM))

            if n_txt_tokens > 0:
                h_txt = h[txt_mask].mean(dim=0)  # [D]
                all_txt[l].append(h_txt)
            else:
                all_txt[l].append(torch.zeros(HIDDEN_DIM))

        metadata.append({
            "city":           rec.get("city", "unknown"),
            "ego_speed":      float(rec.get("ego_speed", 0.0)),
            "front_distance": float(rec.get("front_distance", 10.0)),
            "lead_velocity":  float(rec.get("lead_velocity", 0.0)),
        })

        # Free GPU memory
        del outputs, hidden_states, inputs

    if n_img_zero > 0:
        print(f"  [WARN] {n_img_zero} samples had no image tokens detected")
    if n_txt_zero > 0:
        print(f"  [WARN] {n_txt_zero} samples had no text tokens detected")

    # Stack
    hidden_img = {}
    hidden_txt = {}
    for l in range(NUM_LAYERS + 1):
        if all_img[l]:
            hidden_img[l] = torch.stack(all_img[l])  # [N, D]
        if all_txt[l]:
            hidden_txt[l] = torch.stack(all_txt[l])  # [N, D]

    return hidden_img, hidden_txt, metadata


# ════════════════════════════════════════════════════════════════
#  Step 2: Compute DSS per Modality
# ════════════════════════════════════════════════════════════════

def compute_dss_for_modality(hidden_dict, metadata, label=""):
    """
    Compute Domain Shift Score per layer for a given modality.

    DSS(l) = ||mean(h_l|Boston) - mean(h_l|Singapore)|| / avg_norm

    Returns dict: {layer: {"dss": ..., "dss_norm": ..., "layer_type": ...}}
    """
    boston_idx    = [i for i, m in enumerate(metadata) if m["city"] == "boston"]
    singapore_idx = [i for i, m in enumerate(metadata) if m["city"] == "singapore"]

    print(f"\n  DSS [{label}]: {len(boston_idx)} Boston, {len(singapore_idx)} Singapore")

    if not boston_idx or not singapore_idx:
        print(f"  [ERROR] Need both cities for DSS. Got only: "
              f"{'Boston' if boston_idx else 'Singapore'}")
        return {}

    bs_idx_t = torch.tensor(boston_idx)
    sg_idx_t = torch.tensor(singapore_idx)

    dss_results = {}
    for l in sorted(hidden_dict.keys()):
        if l == 0:
            continue  # skip embedding layer residual
        h = hidden_dict[l]  # [N, D]
        mean_bs = h[bs_idx_t].mean(dim=0)
        mean_sg = h[sg_idx_t].mean(dim=0)

        l2_dist  = torch.norm(mean_bs - mean_sg).item()
        avg_norm = 0.5 * (torch.norm(mean_bs).item() + torch.norm(mean_sg).item())
        dss      = l2_dist / (avg_norm + 1e-10)

        dss_results[l] = {
            "dss":         round(dss, 6),
            "l2_dist":     round(l2_dist, 6),
            "avg_norm":    round(avg_norm, 6),
            "layer_type":  LAYER_TYPES.get(l - 1, "UNK"),  # hidden[l] is output of layer l-1
        }

    # Normalize DSS to [0, 1]
    vals = [v["dss"] for v in dss_results.values()]
    dss_max = max(vals) if vals else 1.0
    for l in dss_results:
        dss_results[l]["dss_norm"] = round(dss_results[l]["dss"] / (dss_max + 1e-10), 6)

    return dss_results


# ════════════════════════════════════════════════════════════════
#  Step 3: Regression Probes + CIS per Modality
# ════════════════════════════════════════════════════════════════

def train_probe(X, y, seed=42):
    """Train Ridge regression probe; return R² on held-out set (clamped ≥ 0)."""
    X_np = X.numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

    valid = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np)
    X_np, y_np = X_np[valid], y_np[valid]

    if len(X_np) < 20:
        return 0.0

    X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2,
                                                random_state=seed)
    probe = Ridge(alpha=1.0)
    probe.fit(X_tr, y_tr)
    return max(r2_score(y_te, probe.predict(X_te)), 0.0)


def compute_cis_for_modality(hidden_dict, metadata, dss_results, cfg, label=""):
    """
    Compute CIS per layer for a given modality.

    CIS(l) = R²_causal(h_l) / (DSS_norm(l) + ε)

    R²_causal = mean R² predicting [ego_speed, front_distance] from h_l.

    Interpretation:
      High CIS → this layer's channel CAUSALLY encodes domain-relevant features
                 AND shows domain shift → primary repair target
      Low CIS  → safe to freeze (not causal, or no domain shift)
    """
    ego_speeds     = torch.tensor([m["ego_speed"]      for m in metadata], dtype=torch.float32)
    front_dists    = torch.tensor([m["front_distance"] for m in metadata], dtype=torch.float32)
    lead_vels      = torch.tensor([m["lead_velocity"]  for m in metadata], dtype=torch.float32)

    cis_results = {}
    for l in sorted(hidden_dict.keys()):
        if l == 0:
            continue

        h = hidden_dict[l]  # [N, D]

        r2_speed = train_probe(h, ego_speeds,   seed=cfg.seed)
        r2_dist  = train_probe(h, front_dists,  seed=cfg.seed + 1)
        r2_vel   = train_probe(h, lead_vels,    seed=cfg.seed + 2)

        r2_causal = (r2_speed + r2_dist + r2_vel) / 3.0
        dss_norm  = dss_results.get(l, {}).get("dss_norm", 0.0)
        cis       = r2_causal / (dss_norm + cfg.cis_epsilon)

        layer_type = LAYER_TYPES.get(l - 1, "UNK")
        cis_results[l] = {
            "r2_ego_speed":     round(r2_speed,  6),
            "r2_front_dist":    round(r2_dist,   6),
            "r2_lead_vel":      round(r2_vel,    6),
            "r2_causal":        round(r2_causal, 6),
            "dss_norm":         round(dss_norm,  6),
            "cis":              round(cis,        6),
            "layer_type":       layer_type,
        }

    print(f"\n  CIS [{label}] per layer:")
    print(f"  {'Layer':>5} {'Type':>4} {'R²_speed':>9} {'R²_dist':>8} "
          f"{'R²_vel':>7} {'R²_caus':>8} {'DSS_n':>6} {'CIS':>8}")
    print(f"  {'─' * 65}")
    for l in sorted(cis_results.keys()):
        c = cis_results[l]
        print(f"  {l:>5} {c['layer_type']:>4} {c['r2_ego_speed']:>9.4f} "
              f"{c['r2_front_dist']:>8.4f} {c['r2_lead_vel']:>7.4f} "
              f"{c['r2_causal']:>8.4f} {c['dss_norm']:>6.4f} {c['cis']:>8.4f}")

    return cis_results


# ════════════════════════════════════════════════════════════════
#  Print summary: img vs txt comparison
# ════════════════════════════════════════════════════════════════

def print_comparison_table(cis_img, cis_txt):
    """Print side-by-side CIS comparison for image vs text channels."""
    print(f"\n{'═' * 80}")
    print(f"  MODALITY COMPARISON — CIS_img vs CIS_txt per Layer")
    print(f"  Hypothesis: CIS_img ≈ 0 (LTC filters image domain shift)")
    print(f"              CIS_txt > 0 (text numerical shift persists)")
    print(f"{'═' * 80}")
    print(f"  {'Layer':>5} {'Type':>4} {'CIS_img':>9} {'CIS_txt':>9} "
          f"{'txt/img':>8} {'Dominant':>12}")
    print(f"  {'─' * 55}")

    all_layers = sorted(set(list(cis_img.keys()) + list(cis_txt.keys())))
    for l in all_layers:
        ci = cis_img.get(l, {}).get("cis", 0.0)
        ct = cis_txt.get(l, {}).get("cis", 0.0)
        ltype = cis_img.get(l, cis_txt.get(l, {})).get("layer_type", "?")
        ratio = ct / (ci + 1e-6)
        dominant = "TEXT ✓" if ct > ci else "IMAGE"
        print(f"  {l:>5} {ltype:>4} {ci:>9.4f} {ct:>9.4f} {ratio:>8.2f}x {dominant:>12}")

    # Summary
    img_vals = [cis_img[l]["cis"] for l in cis_img]
    txt_vals = [cis_txt[l]["cis"] for l in cis_txt]
    if img_vals and txt_vals:
        print(f"\n  ★ Mean CIS_img = {np.mean(img_vals):.4f}  "
              f"Mean CIS_txt = {np.mean(txt_vals):.4f}")
        print(f"  ★ Max  CIS_img = {np.max(img_vals):.4f}  "
              f"Max  CIS_txt = {np.max(txt_vals):.4f}")
        overall_ratio = np.mean(txt_vals) / (np.mean(img_vals) + 1e-6)
        if overall_ratio > 2.0:
            print(f"  ✅ Text channel dominates by {overall_ratio:.1f}x — supports LTC visual invariance hypothesis")
        elif overall_ratio < 0.5:
            print(f"  ⚠️  Image channel dominates by {1/overall_ratio:.1f}x — unexpected result")
        else:
            print(f"  ℹ️  Both channels contribute similarly (ratio = {overall_ratio:.2f}x)")

    print(f"{'═' * 80}")


# ════════════════════════════════════════════════════════════════
#  Layer Partition
# ════════════════════════════════════════════════════════════════

def partition_layers(cis_txt, theta):
    """
    Partition layers based on CIS_txt (text channel CIS).

    phi_txt_repair : layers with high text CIS → target for Phase 2 MMD alignment
    phi_freeze     : layers with low text CIS  → keep frozen
    """
    phi_repair = []
    phi_freeze  = []
    for l in sorted(cis_txt.keys()):
        if cis_txt[l]["cis"] > theta:
            phi_repair.append(l)
        else:
            phi_freeze.append(l)

    return {
        "theta":       theta,
        "phi_txt_repair": phi_repair,
        "phi_freeze":   phi_freeze,
    }


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    cfg = VLConfig()

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1 (VL) v2: Modality-Separated DSS-CIS Analysis")
    print(f"  Image tokens: <|image_start|>...<|image_end|>")
    print(f"  Text tokens:  all other valid positions")
    print(f"{'=' * 70}")

    PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────
    model_dir = str(CHECKPOINT_DIR / "phase0_vl_merged")
    if not os.path.isdir(model_dir):
        print(f"[Phase 1] Model not found: {model_dir}")
        sys.exit(1)

    print(f"\n[Phase 1] Loading model from {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    model.eval()

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"  Model on device: {next(model.parameters()).device}")
    print(f"  Image token ID:  {IMAGE_TOKEN_ID}  (<image>)")
    print(f"  Image start ID:  {IMAGE_START_ID}  (<|image_start|>)")
    print(f"  Image end ID:    {IMAGE_END_ID}  (<|image_end|>)")

    # ── Load data (need BOTH cities for DSS) ─────────────────────
    print(f"\n[Phase 1] Loading data...")
    all_records = []
    city_counts = {}
    for split in ["train", "val", "test_id", "test_ood"]:
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            records = [json.loads(line) for line in f]
        for r in records:
            city = r.get("city", "unknown")
            city_counts[city] = city_counts.get(city, 0) + 1
        all_records.extend(records)
        print(f"  Loaded {len(records):>6} from {split}")

    print(f"  Total: {len(all_records)} records")
    for city, cnt in city_counts.items():
        print(f"    {city}: {cnt}")

    # Balance cities
    by_city = {}
    for r in all_records:
        city = r.get("city", "unknown")
        by_city.setdefault(city, []).append(r)

    balanced = []
    for city, recs in by_city.items():
        sampled = recs if len(recs) <= MAX_SAMPLES else random.sample(recs, MAX_SAMPLES)
        balanced.extend(sampled)
        print(f"  Using {len(sampled)} samples from {city}")

    random.seed(cfg.seed)
    random.shuffle(balanced)
    print(f"  Balanced total: {len(balanced)} records")

    # ── Step 1: Extract hidden states ────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Step 1: Extract Modality-Separated Hidden States")
    print(f"{'─' * 60}")

    cache_img  = PHASE1_OUTPUT / "hidden_img.pt"
    cache_txt  = PHASE1_OUTPUT / "hidden_txt.pt"
    cache_meta = PHASE1_OUTPUT / "metadata.json"

    if cache_img.exists() and cache_txt.exists() and cache_meta.exists():
        print("  Loading from cache...")
        hidden_img = torch.load(cache_img, weights_only=True)
        hidden_txt = torch.load(cache_txt, weights_only=True)
        with open(cache_meta) as f:
            metadata = json.load(f)
        print(f"  Cache loaded: {len(metadata)} samples, {len(hidden_img)} layers")
    else:
        t0 = time.time()
        hidden_img, hidden_txt, metadata = extract_hidden_states_by_modality(
            model, processor, balanced, cfg, max_samples=MAX_SAMPLES * 2
        )
        elapsed = time.time() - t0
        print(f"\n  Extraction time: {elapsed/60:.1f} min")
        print(f"  Samples extracted: {len(metadata)}")
        print(f"  Layers: {len(hidden_img)}")
        if 1 in hidden_img:
            print(f"  Hidden dim: {hidden_img[1].shape[1]}")

        # Cache to disk
        torch.save(hidden_img, cache_img)
        torch.save(hidden_txt, cache_txt)
        with open(cache_meta, "w") as f:
            json.dump(metadata, f)
        print(f"  Cached to {PHASE1_OUTPUT}")

    city_dist = {}
    for m in metadata:
        city_dist[m["city"]] = city_dist.get(m["city"], 0) + 1
    print(f"  City distribution: {city_dist}")

    # ── Step 2: DSS per modality ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Step 2: Domain Shift Score (DSS) — per modality")
    print(f"{'─' * 60}")

    dss_img = compute_dss_for_modality(hidden_img, metadata, label="IMAGE")
    dss_txt = compute_dss_for_modality(hidden_txt, metadata, label="TEXT")

    # Print DSS comparison table
    print(f"\n  {'Layer':>5} {'Type':>4} {'DSS_img':>9} {'DSS_img_n':>10} "
          f"{'DSS_txt':>9} {'DSS_txt_n':>10}")
    print(f"  {'─' * 55}")
    all_layers = sorted(set(list(dss_img.keys()) + list(dss_txt.keys())))
    for l in all_layers:
        di = dss_img.get(l, {})
        dt = dss_txt.get(l, {})
        ltype = di.get("layer_type", dt.get("layer_type", "?"))
        print(f"  {l:>5} {ltype:>4} "
              f"{di.get('dss', 0):>9.4f} {di.get('dss_norm', 0):>10.4f} "
              f"{dt.get('dss', 0):>9.4f} {dt.get('dss_norm', 0):>10.4f}")

    # Save DSS
    with open(PHASE1_OUTPUT / "dss_img.json", "w") as f:
        json.dump({str(k): v for k, v in dss_img.items()}, f, indent=2)
    with open(PHASE1_OUTPUT / "dss_txt.json", "w") as f:
        json.dump({str(k): v for k, v in dss_txt.items()}, f, indent=2)
    print(f"\n  DSS saved.")

    # ── Step 3: CIS per modality ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Step 3: Causal Importance Score (CIS) — per modality")
    print(f"{'─' * 60}")

    cis_img = compute_cis_for_modality(hidden_img, metadata, dss_img, cfg, label="IMAGE")
    cis_txt = compute_cis_for_modality(hidden_txt, metadata, dss_txt, cfg, label="TEXT")

    # Save CIS
    with open(PHASE1_OUTPUT / "cis_img.json", "w") as f:
        json.dump({str(k): v for k, v in cis_img.items()}, f, indent=2)
    with open(PHASE1_OUTPUT / "cis_txt.json", "w") as f:
        json.dump({str(k): v for k, v in cis_txt.items()}, f, indent=2)
    print(f"  CIS saved.")

    # ── Comparison Table ──────────────────────────────────────────
    print_comparison_table(cis_img, cis_txt)

    # ── Layer Partition ───────────────────────────────────────────
    partition = partition_layers(cis_txt, cfg.cis_threshold)
    with open(PHASE1_OUTPUT / "layer_partition.json", "w") as f:
        json.dump(partition, f, indent=2)

    print(f"\n  Layer Partition (theta={cfg.cis_threshold}):")
    print(f"    phi_txt_repair (Phase 2 target): {partition['phi_txt_repair']}")
    print(f"    phi_freeze (keep frozen):         {partition['phi_freeze']}")

    # ── Combined JSON ─────────────────────────────────────────────
    combined = {}
    for l in all_layers:
        combined[str(l)] = {
            "layer_type": LAYER_TYPES.get(l - 1, "UNK"),
            "dss_img":    dss_img.get(l, {}).get("dss", 0),
            "dss_img_n":  dss_img.get(l, {}).get("dss_norm", 0),
            "dss_txt":    dss_txt.get(l, {}).get("dss", 0),
            "dss_txt_n":  dss_txt.get(l, {}).get("dss_norm", 0),
            "cis_img":    cis_img.get(l, {}).get("cis", 0),
            "cis_txt":    cis_txt.get(l, {}).get("cis", 0),
            "r2_img":     cis_img.get(l, {}).get("r2_causal", 0),
            "r2_txt":     cis_txt.get(l, {}).get("r2_causal", 0),
        }
    with open(PHASE1_OUTPUT / "phase1_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    # ── Clean up ──────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  Phase 1 (VL) v2 Complete")
    print(f"  Output: {PHASE1_OUTPUT}")
    print(f"  Key files:")
    print(f"    phase1_summary.json  — full per-layer comparison")
    print(f"    layer_partition.json — phi_txt_repair + phi_freeze")
    print(f"    cis_img.json / cis_txt.json")
    print(f"    dss_img.json / dss_txt.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

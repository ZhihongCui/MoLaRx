"""
Phase 1 (VL) — DSS-CIS Layer Analysis on VLM Hidden States
=============================================================

Orchestrates:
  Step 1: Extract hidden states from language backbone of fine-tuned VLM
  Step 2: Compute DSS (Domain Shift Score) per layer
  Step 3: Train regression probes + compute CIS (Causal Importance Score)

Output: layer_partition.json with phi_high (freeze) and phi_low (update)

Usage:
  python lfm2_vl_pipeline/phase1_vl_probing/run_phase1_vl.py
"""

import json
import os
import sys
import time
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
    GQA_LAYER_INDICES,
    LAYER_TYPES,
)


PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl"


# ════════════════════════════════════════════════════════════════
#  Step 1: Extract Hidden States
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_hidden_states(model, processor, records, cfg, max_samples=2000):
    """
    Extract last-token hidden states from VLM language backbone.

    Returns:
        hidden_dict: {layer: tensor [N, D]}
        metadata: list of dicts with city, decision, distance, velocity
    """
    model.eval()

    # Subsample if too many
    if len(records) > max_samples:
        import random
        random.seed(cfg.seed)
        records = random.sample(records, max_samples)

    all_hidden = {l: [] for l in range(NUM_LAYERS + 1)}  # 0..16 (input + 16 layers)
    metadata = []

    for rec in tqdm(records, desc="  Extracting hidden states"):
        # Load image
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))

        # Build prompt (user turn only)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec["question"]},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            images=[image],
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (L+1) tensors

        # Get last non-padding token
        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is not None:
            last_idx = attn_mask.sum(dim=1) - 1  # [1]
        else:
            last_idx = torch.tensor([hidden_states[0].size(1) - 1])

        for l in range(min(len(hidden_states), NUM_LAYERS + 1)):
            h = hidden_states[l][0, last_idx[0], :]  # [D]
            all_hidden[l].append(h.cpu().float())

        metadata.append({
            "city": rec.get("city", "unknown"),
            "decision": rec.get("decision", -1),
            "distance": rec.get("distance", 999.0),
            "velocity": rec.get("velocity", 0.0),
        })

    # Stack
    hidden_dict = {}
    for l in range(NUM_LAYERS + 1):
        if all_hidden[l]:
            hidden_dict[l] = torch.stack(all_hidden[l])  # [N, D]

    return hidden_dict, metadata


def compute_residuals(hidden_dict, n_layers):
    """Compute layer residuals: Delta_l = h_l - h_{l-1}."""
    residuals = {}
    for l in range(n_layers):
        h_l = hidden_dict[l + 1]     # layer l output
        h_prev = hidden_dict[l]       # layer l-1 output (or embedding for l=0)
        residuals[l] = h_l - h_prev   # [N, D]
    return residuals


# ════════════════════════════════════════════════════════════════
#  Step 2: Compute DSS
# ════════════════════════════════════════════════════════════════

def compute_dss(residuals, metadata, n_layers):
    """
    Compute Domain Shift Score per layer.

    DSS(l) = ||mean(Delta_l|Singapore) - mean(Delta_l|Boston)|| / avg_norm
    """
    sg_idx = [i for i, m in enumerate(metadata) if m["city"] == "singapore"]
    bs_idx = [i for i, m in enumerate(metadata) if m["city"] == "boston"]

    print(f"\n  DSS: {len(sg_idx)} Singapore, {len(bs_idx)} Boston samples")

    if len(sg_idx) == 0 or len(bs_idx) == 0:
        print("  ERROR: Need both Singapore and Boston samples for DSS!")
        return {}

    sg_idx_t = torch.tensor(sg_idx)
    bs_idx_t = torch.tensor(bs_idx)

    dss_results = {}
    for l in range(n_layers):
        delta = residuals[l]  # [N, D]
        mean_sg = delta[sg_idx_t].mean(dim=0)
        mean_bs = delta[bs_idx_t].mean(dim=0)

        l2_dist = torch.norm(mean_sg - mean_bs).item()
        avg_norm = 0.5 * (torch.norm(mean_sg).item() + torch.norm(mean_bs).item())
        dss = l2_dist / (avg_norm + 1e-10)

        layer_type = LAYER_TYPES.get(l, "UNK")
        dss_results[l] = {
            "dss": round(dss, 6),
            "l2_dist": round(l2_dist, 6),
            "avg_norm": round(avg_norm, 6),
            "layer_type": layer_type,
        }

    # Normalize
    dss_vals = [v["dss"] for v in dss_results.values()]
    dss_max = max(dss_vals) if dss_vals else 1.0
    for l in dss_results:
        dss_results[l]["dss_norm"] = round(dss_results[l]["dss"] / (dss_max + 1e-10), 6)

    return dss_results


# ════════════════════════════════════════════════════════════════
#  Step 3: Regression Probes + CIS
# ════════════════════════════════════════════════════════════════

def train_regression_probe(X, y, seed=42):
    """Train a Ridge regression probe, return R^2 on held-out set."""
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y

    # Remove NaN/Inf
    valid = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np)
    X_np, y_np = X_np[valid], y_np[valid]

    if len(X_np) < 20:
        return 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=seed
    )

    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    return max(r2, 0.0)  # Clamp negative R^2 to 0


def compute_cis(residuals, metadata, dss_results, n_layers, cfg):
    """
    Compute Causal Importance Score per layer.

    CIS(l) = R^2_causal(Delta_l) / (DSS_norm(l) + epsilon)

    R^2_causal = average R^2 from regression probes predicting
                 distance and velocity from Delta_l.
    """
    # Prepare causal targets
    distances = torch.tensor([m["distance"] for m in metadata], dtype=torch.float32)
    velocities = torch.tensor([m["velocity"] for m in metadata], dtype=torch.float32)

    cis_results = {}
    for l in range(n_layers):
        delta = residuals[l]  # [N, D]

        # R^2 for distance
        r2_dist = train_regression_probe(delta, distances, seed=cfg.seed)
        # R^2 for velocity
        r2_vel = train_regression_probe(delta, velocities, seed=cfg.seed + 1)

        r2_causal = (r2_dist + r2_vel) / 2.0
        dss_norm = dss_results[l]["dss_norm"] if l in dss_results else 0.0
        cis = r2_causal / (dss_norm + cfg.cis_epsilon)

        layer_type = LAYER_TYPES.get(l, "UNK")
        cis_results[l] = {
            "r2_distance": round(r2_dist, 6),
            "r2_velocity": round(r2_vel, 6),
            "r2_causal": round(r2_causal, 6),
            "dss_norm": round(dss_norm, 6),
            "cis": round(cis, 6),
            "layer_type": layer_type,
        }

    return cis_results


def partition_layers(cis_results, theta):
    """Partition layers into phi_high (freeze) and phi_low (update)."""
    phi_high = []
    phi_low = []

    for l, info in sorted(cis_results.items()):
        if info["cis"] > theta:
            phi_high.append(l)
        else:
            phi_low.append(l)

    return {
        "theta": theta,
        "phi_high": phi_high,
        "phi_low": phi_low,
    }


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    cfg = VLConfig()

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1 (VL): DSS-CIS Layer Analysis")
    print(f"{'=' * 70}")

    PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    model_dir = str(CHECKPOINT_DIR / "phase0_vl_merged")
    if not os.path.isdir(model_dir):
        print(f"[Phase 1] Model not found: {model_dir}")
        print("  Run phase0_vl_finetune.py first.")
        sys.exit(1)

    print(f"\n[Phase 1] Loading model from {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Load data (need both cities for DSS) ──
    all_records = []
    for split in ["train", "val", "test_id", "test_ood"]:
        path = DATA_DIR / f"{split}.jsonl"
        if path.exists():
            with open(path) as f:
                records = [json.loads(l) for l in f]
            all_records.extend(records)
            print(f"  Loaded {len(records)} from {split}")

    print(f"  Total: {len(all_records)} records")

    # ── Step 1: Extract hidden states ──
    print(f"\n{'─' * 60}")
    print(f"  Step 1: Extract Hidden States")
    print(f"{'─' * 60}")
    t0 = time.time()

    cache_path = PHASE1_OUTPUT / "hidden_states_cache.pt"
    meta_path = PHASE1_OUTPUT / "metadata_cache.json"

    if cache_path.exists() and meta_path.exists():
        print("  Loading from cache...")
        hidden_dict = torch.load(cache_path, weights_only=True)
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        hidden_dict, metadata = extract_hidden_states(
            model, processor, all_records, cfg, max_samples=2000
        )
        torch.save(hidden_dict, cache_path)
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        print(f"  Cached to {cache_path}")

    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Samples: {len(metadata)}")
    print(f"  Layers: {len(hidden_dict)}")
    if 1 in hidden_dict:
        print(f"  Hidden dim: {hidden_dict[1].shape[1]}")

    # ── Compute residuals ──
    residuals = compute_residuals(hidden_dict, NUM_LAYERS)

    # ── Step 2: DSS ──
    print(f"\n{'─' * 60}")
    print(f"  Step 2: Domain Shift Score (DSS)")
    print(f"{'─' * 60}")

    dss_results = compute_dss(residuals, metadata, NUM_LAYERS)

    # Print DSS table
    print(f"\n  {'Layer':>5} {'Type':>4} {'DSS':>8} {'DSS_norm':>9}")
    print(f"  {'─' * 30}")
    for l in range(NUM_LAYERS):
        if l in dss_results:
            d = dss_results[l]
            print(f"  {l:>5} {d['layer_type']:>4} {d['dss']:>8.4f} {d['dss_norm']:>9.4f}")

    # Save DSS
    with open(PHASE1_OUTPUT / "dss_results.json", "w") as f:
        json.dump({str(k): v for k, v in dss_results.items()}, f, indent=2)

    # ── Step 3: CIS ──
    print(f"\n{'─' * 60}")
    print(f"  Step 3: Causal Importance Score (CIS)")
    print(f"{'─' * 60}")

    cis_results = compute_cis(residuals, metadata, dss_results, NUM_LAYERS, cfg)

    # Print CIS table
    print(f"\n  {'Layer':>5} {'Type':>4} {'R2_dist':>8} {'R2_vel':>7} "
          f"{'R2_caus':>8} {'DSS_n':>6} {'CIS':>8}")
    print(f"  {'─' * 55}")
    for l in range(NUM_LAYERS):
        if l in cis_results:
            c = cis_results[l]
            print(f"  {l:>5} {c['layer_type']:>4} {c['r2_distance']:>8.4f} "
                  f"{c['r2_velocity']:>7.4f} {c['r2_causal']:>8.4f} "
                  f"{c['dss_norm']:>6.4f} {c['cis']:>8.4f}")

    # Save CIS
    with open(PHASE1_OUTPUT / "cis_results.json", "w") as f:
        json.dump({str(k): v for k, v in cis_results.items()}, f, indent=2)

    # ── Layer partition ──
    partition = partition_layers(cis_results, cfg.cis_threshold)
    with open(PHASE1_OUTPUT / "layer_partition.json", "w") as f:
        json.dump(partition, f, indent=2)

    print(f"\n  Layer Partition (theta={cfg.cis_threshold}):")
    print(f"    phi_high (freeze): {partition['phi_high']}")
    print(f"    phi_low  (update): {partition['phi_low']}")

    # ── Clean up GPU memory ──
    del model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  Phase 1 (VL) Complete")
    print(f"  Output: {PHASE1_OUTPUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

"""
Phase 1 (VL) — Domain Classifier Probing
==========================================

For each layer l (0..15), extract hidden states from Singapore and Boston
test samples, then train a linear classifier to predict domain (SG vs BS).

  Domain accuracy high → layer encodes domain-specific info → UPDATE (apply L_DA)
  Domain accuracy ~50% → layer is domain-invariant → FREEZE (leave alone)

This replaces the counterfactual activation patching approach, which fails
when the model relies heavily on text features (injecting cross-domain
hidden states mostly causes Harm due to text signal disruption).

Domain probing directly measures how much domain information each layer
encodes, regardless of whether that information changes decisions.

Usage:
  python lfm2_vl_pipeline/phase1_vl_domain_probing.py [--max_samples N]
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig,
    VL_MODEL_ID,
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
#  Data Loading
# ════════════════════════════════════════════════════════════════

def load_records(jsonl_path, max_samples=None):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)
    return records


# ════════════════════════════════════════════════════════════════
#  Hidden State Extraction
# ════════════════════════════════════════════════════════════════

def extract_hidden_states(model, processor, records, device, batch_size=1):
    """
    Extract last-token hidden states at every layer for a list of records.
    Processes one sample at a time to avoid VLM batching issues.
    Returns: dict[layer_idx] -> np.array of shape [N, hidden_dim]
    """
    all_hidden = defaultdict(list)

    for i in tqdm(range(len(records)), desc="Extracting"):
        rec = records[i]

        img_path = rec["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join("/mnt/vdb/LFM_project", img_path)
        img = Image.open(img_path).convert("RGB")

        # Build chat message (question only, no answer)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rec["question"]},
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Tokenize single sample
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Forward with hidden states
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True)

        # Extract last-token hidden state per layer
        for layer_idx in range(NUM_LAYERS):
            hs = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, D]
            last_hidden = hs[0, -1, :]  # [D] — last token
            all_hidden[layer_idx].append(last_hidden.float().cpu().numpy())

        # Free memory
        del outputs, inputs
        torch.cuda.empty_cache()

    # Stack into arrays
    result = {}
    for layer_idx in range(NUM_LAYERS):
        result[layer_idx] = np.stack(all_hidden[layer_idx], axis=0)

    return result


# ════════════════════════════════════════════════════════════════
#  Domain Probing
# ════════════════════════════════════════════════════════════════

def domain_probing(sg_hidden, bs_hidden, n_cv=5):
    """
    Train a linear classifier to distinguish SG vs BS hidden states.

    Args:
        sg_hidden: dict[layer] -> np.array [N_sg, D]
        bs_hidden: dict[layer] -> np.array [N_bs, D]
        n_cv: number of cross-validation folds

    Returns:
        dict[layer] -> {accuracy, std, n_sg, n_bs}
    """
    results = {}

    for layer_idx in range(NUM_LAYERS):
        h_sg = sg_hidden[layer_idx]  # [N_sg, D]
        h_bs = bs_hidden[layer_idx]  # [N_bs, D]

        # Balance classes: subsample the larger set
        n_min = min(len(h_sg), len(h_bs))
        if len(h_sg) > n_min:
            idx = np.random.RandomState(42).choice(len(h_sg), n_min, replace=False)
            h_sg_bal = h_sg[idx]
        else:
            h_sg_bal = h_sg
        if len(h_bs) > n_min:
            idx = np.random.RandomState(42).choice(len(h_bs), n_min, replace=False)
            h_bs_bal = h_bs[idx]
        else:
            h_bs_bal = h_bs

        # Create dataset
        X = np.concatenate([h_sg_bal, h_bs_bal], axis=0)
        y = np.array([0]*len(h_sg_bal) + [1]*len(h_bs_bal))

        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train logistic regression with cross-validation
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=n_cv, scoring='accuracy')

        results[layer_idx] = {
            "domain_accuracy": float(np.mean(scores)),
            "domain_std": float(np.std(scores)),
            "n_sg": len(h_sg_bal),
            "n_bs": len(h_bs_bal),
            "layer_type": LAYER_TYPES[layer_idx],
        }

        layer_type = LAYER_TYPES[layer_idx]
        print(f"  Layer {layer_idx:>2} ({layer_type}): "
              f"domain_acc={np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return results


# ════════════════════════════════════════════════════════════════
#  Layer Partition
# ════════════════════════════════════════════════════════════════

def partition_layers(probing_results, method="median"):
    """
    Partition layers into phi_high (FREEZE) and phi_low (UPDATE).

    High domain accuracy → encodes domain info → UPDATE (needs alignment)
    Low domain accuracy  → domain invariant   → FREEZE (leave alone)
    """
    accuracies = {l: r["domain_accuracy"] for l, r in probing_results.items()}

    values = list(accuracies.values())
    threshold = float(np.median(values))

    # Layers with HIGH domain accuracy need updating (they encode domain info)
    phi_low = sorted([l for l, a in accuracies.items() if a >= threshold])
    # Layers with LOW domain accuracy are domain-invariant (freeze them)
    phi_high = sorted([l for l, a in accuracies.items() if a < threshold])

    return phi_high, phi_low, threshold


# ════════════════════════════════════════════════════════════════
#  Visualization
# ════════════════════════════════════════════════════════════════

def plot_results(probing_results, phi_high, phi_low, threshold, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = sorted(probing_results.keys())
    accs = [probing_results[l]["domain_accuracy"] for l in layers]
    stds = [probing_results[l]["domain_std"] for l in layers]
    types = [probing_results[l]["layer_type"] for l in layers]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    colors = []
    for l in layers:
        if l in phi_low:
            colors.append('#e74c3c')  # red = UPDATE
        else:
            colors.append('#2ecc71')  # green = FREEZE

    bars = ax.bar(range(len(layers)), accs, color=colors,
                   edgecolor='white', linewidth=0.5, yerr=stds, capsize=3)

    # Threshold line
    ax.axhline(y=threshold, color='blue', linestyle='--', linewidth=1.5,
               label=f'Median threshold={threshold:.4f}')

    # Chance level
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1,
               label='Chance level (50%)')

    # Labels
    for i, l in enumerate(layers):
        label = "U" if l in phi_low else "F"
        color = '#c0392b' if l in phi_low else '#27ae60'
        ax.text(i, accs[i] + stds[i] + 0.005, label,
                ha='center', va='bottom', fontweight='bold', fontsize=9, color=color)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Domain Classification Accuracy', fontsize=12)
    ax.set_title('Domain Classifier Probing: How Much Domain Info Does Each Layer Encode?', fontsize=13)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'L{l}\n({types[i]})' for i, l in enumerate(layers)], fontsize=9)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.45, max(accs) + 0.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {output_path}")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples per domain (SG and BS)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg = VLConfig()

    model_dir = args.model_dir or str(CHECKPOINT_DIR / "phase0_vl_merged")

    print("=" * 70)
    print("  PHASE 1 (VL): Domain Classifier Probing")
    print("  (Linear probe: can layer l distinguish SG from BS?)")
    print("=" * 70)
    print(f"\n  Model: {model_dir}")
    print(f"  Max samples per domain: {args.max_samples}")

    # ── Load model ──
    print("  Loading model...")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"  Device: {device}")

    # Print layer info
    print(f"  Language backbone: {NUM_LAYERS} layers")
    for i in range(NUM_LAYERS):
        lt = LAYER_TYPES[i]
        print(f"    Layer {i:>2} ({lt})")

    # ── Load data ──
    sg_records = load_records(str(DATA_DIR / "test_id.jsonl"), args.max_samples)
    bs_records = load_records(str(DATA_DIR / "test_ood.jsonl"), args.max_samples)
    print(f"\n  Singapore (ID) samples: {len(sg_records)}")
    print(f"  Boston (OOD) samples:   {len(bs_records)}")

    # ── Step 1: Extract hidden states ──
    print(f"\n{'─'*60}")
    print(f"  Step 1: Extract Hidden States")
    print(f"{'─'*60}")

    t0 = time.time()

    print(f"\n  Extracting Singapore hidden states...")
    sg_hidden = extract_hidden_states(model, processor, sg_records, device, args.batch_size)

    print(f"\n  Extracting Boston hidden states...")
    bs_hidden = extract_hidden_states(model, processor, bs_records, device, args.batch_size)

    t_extract = time.time() - t0
    print(f"\n  Extraction time: {t_extract:.1f}s ({t_extract/60:.1f} min)")
    print(f"  SG shape: {sg_hidden[0].shape}")
    print(f"  BS shape: {bs_hidden[0].shape}")

    # ── Step 2: Domain probing ──
    print(f"\n{'─'*60}")
    print(f"  Step 2: Domain Classifier Probing (5-fold CV)")
    print(f"{'─'*60}\n")

    probing_results = domain_probing(sg_hidden, bs_hidden, n_cv=5)

    # ── Step 3: Layer partition ──
    print(f"\n{'─'*60}")
    print(f"  Step 3: Layer Partition")
    print(f"{'─'*60}")

    phi_high, phi_low, threshold = partition_layers(probing_results)

    print(f"  Method: median")
    print(f"  Threshold (median domain accuracy): {threshold:.4f}")
    print(f"  phi_high (FREEZE - domain invariant):   {phi_high}")
    print(f"  phi_low  (UPDATE - encodes domain info): {phi_low}")

    phi_high_types = defaultdict(int)
    phi_low_types = defaultdict(int)
    for l in phi_high:
        phi_high_types[LAYER_TYPES[l]] += 1
    for l in phi_low:
        phi_low_types[LAYER_TYPES[l]] += 1
    print(f"  phi_high types: {dict(phi_high_types)}")
    print(f"  phi_low  types: {dict(phi_low_types)}")

    # ── Save results ──
    os.makedirs(str(PHASE1_OUTPUT), exist_ok=True)

    # Save probing results
    save_data = {}
    for l, r in probing_results.items():
        save_data[str(l)] = r

    results_path = str(PHASE1_OUTPUT / "domain_probing_results.json")
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Probing results saved: {results_path}")

    # Save layer partition
    partition_data = {
        "method": "domain_probing_median",
        "threshold": threshold,
        "phi_high": phi_high,
        "phi_low": phi_low,
        "probing_results": save_data,
    }
    partition_path = str(PHASE1_OUTPUT / "layer_partition.json")
    with open(partition_path, 'w') as f:
        json.dump(partition_data, f, indent=2)
    print(f"  Layer partition saved: {partition_path}")

    # Plot
    fig_path = str(PHASE1_OUTPUT / "domain_probing_analysis.png")
    plot_results(probing_results, phi_high, phi_low, threshold, fig_path)

    print(f"\n{'='*70}")
    print(f"  Phase 1 (VL) Domain Probing Complete")
    print(f"  Output directory: {PHASE1_OUTPUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

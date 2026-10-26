"""
Phase 1 (VL) — Image-Token Counterfactual Layer Analysis
==========================================================

For each layer l, replace ONLY the image token positions of the hidden state
from the Singapore forward pass with those from the Boston forward pass.
Text token positions are kept unchanged.

This isolates the image domain effect from text effects.

  For each (SG, BS) pair and each layer l:
    1. Run SG normally → get clean decision Y_SG
    2. At layer l, replace hidden state at image positions with BS's
    3. Continue forward → get counterfactual decision Y_cf(l)
    4. If Y_cf improves → Benefit; if worsens → Harm

  CIS_img(l) = Benefit(l) / (Harm(l) + epsilon)

  High CIS → layer's image processing carries beneficial cross-domain info → FREEZE
  Low CIS  → layer's image processing carries harmful domain bias → UPDATE

Usage:
  python phase1_vl_image_counterfactual.py [--max_pairs 200]
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig, VL_MODEL_ID, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_LAYERS, HIDDEN_DIM, GQA_LAYER_INDICES, LAYER_TYPES,
    LONG_DECISION_LABELS, LAT_DECISION_LABELS,
    LONG_DECISION_KEYWORDS, LAT_DECISION_KEYWORDS,
)

PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl"
IMAGE_TOKEN_ID = 396  # <image> token id in LFM2-VL


# ════════════════════════════════════════════════════════════════
#  Decision Parsing
# ════════════════════════════════════════════════════════════════

def parse_longitudinal(text: str) -> int:
    t = text.strip().lower()
    first_word = t.split()[0] if t else ""
    if first_word.startswith("decelerat"):
        return 2
    if first_word.startswith("accelerat"):
        return 0
    if first_word.startswith("maintain"):
        return 1
    for d_id, keywords in LONG_DECISION_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return d_id
    return -1


def parse_lateral(text: str) -> int:
    t = text.strip().lower()
    if "turn left" in t or "go left" in t:
        return 0
    if "turn right" in t or "go right" in t:
        return 2
    if "go straight" in t or "straight" in t:
        return 1
    for d_id, keywords in LAT_DECISION_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return d_id
    return -1


def parse_decision(text: str):
    parts = text.strip().split(",")
    if len(parts) >= 2:
        return parse_longitudinal(parts[0]), parse_lateral(parts[1])
    return parse_longitudinal(text), parse_lateral(text)


# ════════════════════════════════════════════════════════════════
#  Find image token positions in input_ids
# ════════════════════════════════════════════════════════════════

def get_image_token_mask(input_ids):
    """Return boolean mask: True for image token positions."""
    return input_ids == IMAGE_TOKEN_ID


# ════════════════════════════════════════════════════════════════
#  Hook-based Image-Token Counterfactual
# ════════════════════════════════════════════════════════════════

class ImageTokenReplacer:
    """
    Hook that replaces image token positions at a specific layer
    with hidden states from another forward pass.
    """
    def __init__(self, model, target_layer: int, replacement_states: torch.Tensor,
                 image_mask: torch.Tensor):
        """
        Args:
            model: the LFM2-VL model
            target_layer: which layer to intervene on
            replacement_states: hidden states from BS at target layer [1, seq_len, hidden_dim]
            image_mask: boolean mask [seq_len] indicating image token positions
        """
        self.target_layer = target_layer
        self.replacement_states = replacement_states
        self.image_mask = image_mask
        self.hook_handle = None
        self._install_hook(model)

    def _install_hook(self, model):
        """Install a forward hook on the target layer."""
        layers = model.model.language_model.layers
        if self.target_layer < len(layers):
            layer = layers[self.target_layer]
            self.hook_handle = layer.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, args):
        """Replace image token positions in the hidden state input."""
        hidden_states = args[0]  # [batch, seq_len, hidden_dim]

        # Only replace image token positions
        mask = self.image_mask.to(hidden_states.device)  # [seq_len]
        replacement = self.replacement_states.to(hidden_states.device)  # [1, seq_len, hidden_dim]

        # Create modified hidden states
        new_hidden = hidden_states.clone()
        new_hidden[:, mask, :] = replacement[:, mask, :]

        return (new_hidden,) + args[1:]

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()


class HiddenStateCapture:
    """Capture hidden states at all layers during a forward pass."""
    def __init__(self, model, num_layers):
        self.hidden_states = {}
        self.hooks = []
        layers = model.model.language_model.layers
        for l in range(min(num_layers, len(layers))):
            hook = layers[l].register_forward_pre_hook(
                lambda mod, args, layer_idx=l: self._capture(layer_idx, args)
            )
            self.hooks.append(hook)

    def _capture(self, layer_idx, args):
        self.hidden_states[layer_idx] = args[0].detach().clone()

    def remove_all(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ════════════════════════════════════════════════════════════════
#  Generate decision from model
# ════════════════════════════════════════════════════════════════

def generate_decision(model, inputs, processor, max_new_tokens=50):
    """Run generate and parse the decision."""
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return parse_decision(text), text.strip()


# ════════════════════════════════════════════════════════════════
#  Prepare inputs for a record
# ════════════════════════════════════════════════════════════════

def prepare_inputs(rec, processor, device):
    """Prepare model inputs from a data record."""
    img = Image.open(rec["image_path"]).convert("RGB")
    conversation = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": rec["question"]},
    ]}]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, tokenize=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


# ════════════════════════════════════════════════════════════════
#  Kinematic pairing
# ════════════════════════════════════════════════════════════════

def pair_samples(sg_records, bs_records, max_pairs=200):
    """Pair SG and BS samples by kinematic similarity (ego_speed + lead_velocity)."""
    import re

    def extract_kin(question):
        kin = []
        # ego_speed
        m = re.search(r"driving at ([\d.]+) m/s", question)
        kin.append(float(m.group(1)) if m else 0.0)
        # lead_velocity
        m = re.search(r"moving at ([-\d.]+) m/s", question)
        kin.append(float(m.group(1)) if m else 0.0)
        return kin

    sg_kin = np.array([extract_kin(r["question"]) for r in sg_records])
    bs_kin = np.array([extract_kin(r["question"]) for r in bs_records])

    # Normalize
    all_kin = np.vstack([sg_kin, bs_kin])
    mins = all_kin.min(axis=0)
    maxs = all_kin.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    sg_norm = (sg_kin - mins) / ranges
    bs_norm = (bs_kin - mins) / ranges

    # Sample SG indices
    n_sg = min(max_pairs, len(sg_records))
    sg_indices = random.sample(range(len(sg_records)), n_sg)

    # Find nearest BS for each SG
    pairs = []
    distances = []
    for si in sg_indices:
        dists = np.linalg.norm(bs_norm - sg_norm[si], axis=1)
        bi = np.argmin(dists)
        pairs.append((si, bi))
        distances.append(dists[bi])

    print(f"  Paired {len(pairs)} samples")
    print(f"  Kinematic distance: mean={np.mean(distances):.3f}, "
          f"median={np.median(distances):.3f}, max={np.max(distances):.3f}")

    return pairs


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(CHECKPOINT_DIR / "phase0_vl_merged"))
    parser.add_argument("--max_pairs", type=int, default=200)
    args = parser.parse_args()

    cfg = VLConfig()
    os.makedirs(PHASE1_OUTPUT, exist_ok=True)

    print("=" * 70)
    print("  PHASE 1 (VL): Image-Token Counterfactual Layer Analysis")
    print("  (Replace ONLY image token positions at each layer)")
    print("=" * 70)
    print(f"\n  Model: {args.model_dir}")
    print(f"  Max pairs: {args.max_pairs}")

    # Load model
    print("  Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"  Device: {device}")

    # Print layer info
    num_layers = len(model.model.language_model.layers)
    print(f"  Language backbone: {num_layers} layers")
    for l in range(num_layers):
        ltype = LAYER_TYPES.get(l, "UNK")
        print(f"    Layer {l:2d} ({ltype})")

    # Load data
    sg_records = [json.loads(line) for line in open(str(DATA_DIR / "test_id.jsonl"))]
    bs_records = [json.loads(line) for line in open(str(DATA_DIR / "test_ood.jsonl"))]
    print(f"  Singapore (ID) test: {len(sg_records)} samples")
    print(f"  Boston (OOD) test:   {len(bs_records)} samples")

    # ── Step 1: Kinematic Pairing ──
    print(f"\n{'─' * 60}")
    print(f"  Step 1: Kinematic Pairing (SG ↔ BS)")
    print(f"{'─' * 60}")
    pairs = pair_samples(sg_records, bs_records, args.max_pairs)

    # ── Step 2: Image-Token Counterfactual Analysis ──
    print(f"\n{'─' * 60}")
    print(f"  Step 2: Image-Token Counterfactual Analysis")
    print(f"{'─' * 60}")
    print(f"  Running on {len(pairs)} pairs...")
    print(f"  (Each pair: 1 clean SG + capture BS hidden states + {num_layers} counterfactual = {num_layers + 2} passes)")

    # Results per layer
    results = {l: {"benefit": 0, "harm": 0, "neutral": 0,
                    "benefit_long": 0, "harm_long": 0,
                    "benefit_lat": 0, "harm_lat": 0}
               for l in range(num_layers)}

    clean_correct_long = 0
    clean_correct_lat = 0
    total_pairs = 0

    t0 = time.time()

    for pair_idx, (sg_idx, bs_idx) in enumerate(pairs):
        sg_rec = sg_records[sg_idx]
        bs_rec = bs_records[bs_idx]

        gt_long = sg_rec["decision_long"]
        gt_lat = sg_rec["decision_lat"]

        # Prepare inputs
        sg_inputs = prepare_inputs(sg_rec, processor, device)
        bs_inputs = prepare_inputs(bs_rec, processor, device)

        # Get image token mask from SG input
        image_mask = get_image_token_mask(sg_inputs["input_ids"][0])  # [seq_len]

        # ── Clean SG decision ──
        (pred_long_clean, pred_lat_clean), _ = generate_decision(model, sg_inputs, processor)

        if pred_long_clean == gt_long:
            clean_correct_long += 1
        if pred_lat_clean == gt_lat:
            clean_correct_lat += 1
        total_pairs += 1

        # ── Capture BS hidden states at all layers ──
        capturer = HiddenStateCapture(model, num_layers)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = model.generate(
                    **bs_inputs, max_new_tokens=5, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
        bs_hidden = dict(capturer.hidden_states)
        capturer.remove_all()

        # ── Counterfactual for each layer ──
        for l in range(num_layers):
            if l not in bs_hidden:
                continue

            # Install hook to replace image tokens at layer l
            replacer = ImageTokenReplacer(model, l, bs_hidden[l], image_mask)

            (pred_long_cf, pred_lat_cf), _ = generate_decision(model, sg_inputs, processor)

            replacer.remove()

            # Classify change
            long_changed = pred_long_cf != pred_long_clean
            lat_changed = pred_lat_cf != pred_lat_clean

            # Longitudinal benefit/harm
            if long_changed:
                if pred_long_clean != gt_long and pred_long_cf == gt_long:
                    results[l]["benefit_long"] += 1
                elif pred_long_clean == gt_long and pred_long_cf != gt_long:
                    results[l]["harm_long"] += 1

            # Lateral benefit/harm
            if lat_changed:
                if pred_lat_clean != gt_lat and pred_lat_cf == gt_lat:
                    results[l]["benefit_lat"] += 1
                elif pred_lat_clean == gt_lat and pred_lat_cf != gt_lat:
                    results[l]["harm_lat"] += 1

            # Combined benefit/harm
            benefit = results[l]["benefit_long"] + results[l]["benefit_lat"]
            harm = results[l]["harm_long"] + results[l]["harm_lat"]
            neutral = (results[l].get("neutral", 0) +
                       (1 if not long_changed and not lat_changed else 0))
            results[l]["benefit"] = benefit
            results[l]["harm"] = harm
            results[l]["neutral"] = neutral

        # Clean up BS hidden states to free memory
        del bs_hidden
        torch.cuda.empty_cache()

        # Progress
        elapsed = time.time() - t0
        if (pair_idx + 1) % 10 == 0 or pair_idx == 0:
            eta = elapsed / (pair_idx + 1) * (len(pairs) - pair_idx - 1)
            print(f"    [{pair_idx+1}/{len(pairs)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s  |  "
                  f"clean_long_acc={clean_correct_long/total_pairs:.3f}, "
                  f"clean_lat_acc={clean_correct_lat/total_pairs:.3f}")

    total_time = time.time() - t0
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Clean accuracy (on paired subset): "
          f"long={clean_correct_long/total_pairs:.3f}, "
          f"lat={clean_correct_lat/total_pairs:.3f}")

    # ── Compute CIS ──
    epsilon = cfg.cis_epsilon
    print(f"\n  Layer Type  Benefit   Harm  Changes  B_rate  H_rate   CIS_img")
    print(f"  {'─' * 60}")

    cis_results = {}
    for l in range(num_layers):
        r = results[l]
        b = r["benefit"]
        h = r["harm"]
        changes = b + h
        total_changes = changes + r.get("neutral", 0)
        cis = b / (h + epsilon)
        b_rate = b / total_pairs
        h_rate = h / total_pairs
        change_rate = changes / total_pairs if total_pairs > 0 else 0

        ltype = LAYER_TYPES.get(l, "UNK")
        print(f"  {l:5d}  {ltype:<3}  {b:6d}  {h:6d}  {changes:6d}  "
              f"{b_rate:.4f}  {h_rate:.4f}  {cis:7.4f}")

        cis_results[str(l)] = {
            "benefit": b, "harm": h,
            "total_changes": changes,
            "cis_img": cis,
            "benefit_rate": b_rate, "harm_rate": h_rate,
            "change_rate": change_rate,
            "benefit_long": r["benefit_long"], "harm_long": r["harm_long"],
            "benefit_lat": r["benefit_lat"], "harm_lat": r["harm_lat"],
            "layer_type": ltype,
        }

    # Print longitudinal/lateral detail
    print(f"\n  Longitudinal detail:")
    print(f"  Layer  B_long  H_long |  B_lat  H_lat")
    print(f"  {'─' * 40}")
    for l in range(num_layers):
        r = results[l]
        print(f"  {l:5d}  {r['benefit_long']:6d}  {r['harm_long']:6d} |  "
              f"{r['benefit_lat']:6d}  {r['harm_lat']:6d}")

    # Save CIS results
    cis_path = PHASE1_OUTPUT / "cis_image_counterfactual_results.json"
    with open(cis_path, "w") as f:
        json.dump(cis_results, f, indent=2)
    print(f"\n  CIS results saved: {cis_path}")

    # ── Step 3: Layer Partition ──
    print(f"\n{'─' * 60}")
    print(f"  Step 3: Layer Partition")
    print(f"{'─' * 60}")

    cis_values = [cis_results[str(l)]["cis_img"] for l in range(num_layers)]
    theta = float(np.median(cis_values))
    print(f"  Method: median")
    print(f"  Threshold (median CIS): {theta:.4f}")

    phi_high = [l for l in range(num_layers) if cis_values[l] >= theta]
    phi_low = [l for l in range(num_layers) if cis_values[l] < theta]
    print(f"  phi_high (FREEZE - beneficial image domain info): {phi_high}")
    print(f"  phi_low  (UPDATE - harmful image domain bias):    {phi_low}")

    from collections import Counter
    phi_high_types = Counter(LAYER_TYPES[l] for l in phi_high)
    phi_low_types = Counter(LAYER_TYPES[l] for l in phi_low)
    print(f"  phi_high types: {dict(phi_high_types)}")
    print(f"  phi_low  types: {dict(phi_low_types)}")

    # Save partition
    partition = {
        "method": "median",
        "theta": theta,
        "phi_high": phi_high,
        "phi_low": phi_low,
        "cis_per_layer": {str(l): cis_values[l] for l in range(num_layers)},
    }
    part_path = PHASE1_OUTPUT / "layer_partition.json"
    with open(part_path, "w") as f:
        json.dump(partition, f, indent=2)
    print(f"\n  Partition saved: {part_path}")

    # ── Visualization ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        layers = list(range(num_layers))
        cis_vals = [cis_results[str(l)]["cis_img"] for l in layers]
        benefits = [cis_results[str(l)]["benefit"] for l in layers]
        harms = [cis_results[str(l)]["harm"] for l in layers]
        changes = [cis_results[str(l)]["total_changes"] for l in layers]
        types = [LAYER_TYPES[l] for l in layers]
        colors = ["#1f77b4" if t == "GQA" else "#ff7f0e" for t in types]
        labels = [f"L{l}\n({t})" for l, t in zip(layers, types)]

        # Panel 1: CIS
        bars = axes[0].bar(layers, cis_vals, color=colors)
        axes[0].axhline(y=theta, color='red', linestyle='--', label=f'Median θ={theta:.3f}')
        axes[0].set_title("Image-Token Counterfactual CIS per Layer")
        axes[0].set_ylabel("CIS_img")
        axes[0].legend()
        for l, bar in zip(layers, bars):
            label = "F" if l in phi_high else "U"
            color = "green" if l in phi_high else "red"
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        label, ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

        # Panel 2: Benefit vs Harm
        x = np.arange(num_layers)
        axes[1].bar(x - 0.2, benefits, 0.4, label="Benefit (wrong→right)", color="green", alpha=0.7)
        axes[1].bar(x + 0.2, harms, 0.4, label="Harm (right→wrong)", color="red", alpha=0.7)
        axes[1].set_title("Image-Token Benefit vs Harm per Layer")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        # Panel 3: Total changes
        axes[2].bar(layers, changes, color=colors)
        axes[2].set_title("Total Image-Token Decision Changes per Layer")
        axes[2].set_ylabel("Decision Changes")
        axes[2].set_xticks(layers)
        axes[2].set_xticklabels(labels)

        plt.tight_layout()
        fig_path = PHASE1_OUTPUT / "cis_image_counterfactual_analysis.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except Exception as e:
        print(f"  Warning: could not save figure: {e}")

    print(f"\n{'=' * 70}")
    print(f"  Phase 1 (VL) Image-Token Counterfactual Analysis Complete")
    print(f"  Output directory: {PHASE1_OUTPUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

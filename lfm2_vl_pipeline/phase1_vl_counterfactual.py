"""
Phase 1 (VL) — Triple Diagnosis for Trajectory Planning (v11)
===============================================================

Three diagnostic tools applied per layer:
  1. Image-token counterfactual: replace image tokens (SG→BS), measure trajectory shift
  2. Text-token counterfactual: replace text tokens (SG→BS), measure trajectory shift
  3. Domain probing: linear classifier to distinguish SG vs BS from hidden states

Output: layer_partition.json with phi_align, phi_tune, phi_freeze

CIS is now measured as trajectory displacement (continuous, in meters) instead
of accuracy drop (binary). This gives much stronger signal.

Usage:
  python lfm2_vl_pipeline/phase1_vl_counterfactual.py [--max_pairs N]
"""

import argparse
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import (
    VLConfig, VL_MODEL_ID, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_LAYERS, HIDDEN_DIM, GQA_LAYER_INDICES, LAYER_TYPES,
)

PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl"
IMAGE_TOKEN_ID = 396  # <image> token id in LFM2-VL


# ════════════════════════════════════════════════════════════════
#  Trajectory Parsing (reused from evaluate)
# ════════════════════════════════════════════════════════════════

def parse_trajectory(text):
    text = text.strip()
    pattern = r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?'
    matches = re.findall(pattern, text)
    if len(matches) < 1:
        return None
    waypoints = []
    for xstr, ystr in matches:
        try:
            waypoints.append((float(xstr), float(ystr)))
        except ValueError:
            continue
    return waypoints if len(waypoints) > 0 else None


def trajectory_displacement(traj1, traj2):
    """Mean L2 displacement between two trajectories."""
    if traj1 is None or traj2 is None:
        return None
    n = min(len(traj1), len(traj2))
    if n == 0:
        return None
    dists = []
    for i in range(n):
        dx = traj1[i][0] - traj2[i][0]
        dy = traj1[i][1] - traj2[i][1]
        dists.append(np.sqrt(dx**2 + dy**2))
    return float(np.mean(dists))


# ════════════════════════════════════════════════════════════════
#  Model Utilities
# ════════════════════════════════════════════════════════════════

def get_language_layers(model):
    layers = model.model.language_model.layers
    assert len(layers) == NUM_LAYERS
    return layers


def prepare_input(processor, record, device):
    try:
        image = Image.open(record["image_path"]).convert("RGB")
    except Exception:
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": record["question"]},
        ],
    }]

    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, tokenize=True,
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


@torch.no_grad()
def generate_trajectory(model, processor, inputs, cfg):
    outputs = model.generate(
        **inputs, max_new_tokens=cfg.max_new_tokens,
        do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated = processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    traj = parse_trajectory(generated)
    return traj, generated


def get_image_text_masks(input_ids):
    """Get boolean masks for image and text token positions."""
    img_mask = (input_ids[0] == IMAGE_TOKEN_ID)
    txt_mask = ~img_mask
    return img_mask, txt_mask


# ════════════════════════════════════════════════════════════════
#  Hidden State Hooks
# ════════════════════════════════════════════════════════════════

class HiddenStateCollector:
    def __init__(self, layers):
        self.hidden_states = {}
        self.hooks = []
        for l in range(len(layers)):
            hook = layers[l].register_forward_hook(self._make_hook(l))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hidden_states[layer_idx] = output[0].detach().clone()
            else:
                self.hidden_states[layer_idx] = output.detach().clone()
        return hook_fn

    def clear(self):
        self.hidden_states = {}

    def remove(self):
        for h in self.hooks:
            h.remove()


class TokenLevelReplacer:
    """
    Replace either image or text tokens at a specific layer.
    mode='image': replace image token positions with replacement's image tokens
    mode='text': replace text token positions with replacement's text tokens
    """
    def __init__(self, layers, target_layer, replacement_hidden,
                 src_mask, rep_mask, mode='image'):
        self.target_layer = target_layer
        self.replacement = replacement_hidden
        self.src_mask = src_mask  # mask for source (SG) input
        self.rep_mask = rep_mask  # mask for replacement (BS) input
        self.mode = mode
        self.fired = False
        self.hook = layers[target_layer].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if self.fired:
            return
        self.fired = True

        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        new_h = h.clone()
        rep = self.replacement

        if self.mode == 'image':
            # Replace image token positions
            src_img = self.src_mask
            rep_img = self.rep_mask
            n_replace = min(src_img.sum().item(), rep_img.sum().item())
            if n_replace > 0:
                src_indices = torch.where(src_img)[0][:n_replace]
                rep_indices = torch.where(rep_img)[0][:n_replace]
                new_h[0, src_indices, :] = rep[0, rep_indices, :]
        else:
            # Replace text token positions
            src_txt = self.src_mask
            rep_txt = self.rep_mask
            n_replace = min(src_txt.sum().item(), rep_txt.sum().item())
            if n_replace > 0:
                src_indices = torch.where(src_txt)[0][:n_replace]
                rep_indices = torch.where(rep_txt)[0][:n_replace]
                new_h[0, src_indices, :] = rep[0, rep_indices, :]

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    def remove(self):
        if self.hook:
            self.hook.remove()


# ════════════════════════════════════════════════════════════════
#  Pairing
# ════════════════════════════════════════════════════════════════

def pair_samples(sg_records, bs_records, max_pairs=200, seed=42):
    def extract_kin(rec):
        return np.array([
            rec.get("ego_speed", 0.0),
            rec.get("distance", 50.0),
            rec.get("velocity", 0.0),
        ])

    sg_kin = np.array([extract_kin(r) for r in sg_records])
    bs_kin = np.array([extract_kin(r) for r in bs_records])

    combined = np.vstack([sg_kin, bs_kin])
    mu = combined.mean(axis=0)
    std = combined.std(axis=0) + 1e-8
    sg_norm = (sg_kin - mu) / std
    bs_norm = (bs_kin - mu) / std

    dists = cdist(sg_norm, bs_norm, metric="euclidean")

    pairs = []
    used_bs = set()
    sg_order = np.argsort(dists.min(axis=1))

    for sg_idx in sg_order:
        if len(pairs) >= max_pairs:
            break
        sorted_bs = np.argsort(dists[sg_idx])
        for bs_idx in sorted_bs:
            if bs_idx not in used_bs:
                pairs.append((int(sg_idx), int(bs_idx)))
                used_bs.add(bs_idx)
                break

    pair_dists = [dists[s, b] for s, b in pairs]
    print(f"  Paired {len(pairs)} samples (kin dist: mean={np.mean(pair_dists):.3f})")
    return pairs


# ════════════════════════════════════════════════════════════════
#  Triple Diagnosis
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_counterfactual_diagnosis(model, processor, sg_records, bs_records,
                                  pairs, layers, cfg, max_pairs=100):
    model.eval()
    device = next(model.parameters()).device

    cis_img = defaultdict(list)  # trajectory displacement per layer (image CF)
    cis_txt = defaultdict(list)  # trajectory displacement per layer (text CF)
    hidden_states_for_probing = defaultdict(list)  # for domain probing

    n_pairs = min(len(pairs), max_pairs)
    parse_ok = 0
    parse_fail = 0

    print(f"\n  Running triple diagnosis on {n_pairs} pairs...")
    t0 = time.time()

    for pair_idx, (sg_idx, bs_idx) in enumerate(pairs[:n_pairs]):
        sg_rec = sg_records[sg_idx]
        bs_rec = bs_records[bs_idx]

        # ── 1. Clean SG trajectory ──
        sg_inputs = prepare_input(processor, sg_rec, device)
        sg_img_mask, sg_txt_mask = get_image_text_masks(sg_inputs["input_ids"])

        clean_traj, clean_text = generate_trajectory(model, processor, sg_inputs, cfg)
        if clean_traj is None:
            parse_fail += 1
            continue
        parse_ok += 1

        # ── 2. Collect hidden states for both ──
        sg_inputs2 = prepare_input(processor, sg_rec, device)
        collector_sg = HiddenStateCollector(layers)
        _ = model(**sg_inputs2)
        sg_hidden = {l: collector_sg.hidden_states[l] for l in collector_sg.hidden_states}
        collector_sg.remove()

        bs_inputs = prepare_input(processor, bs_rec, device)
        bs_img_mask, bs_txt_mask = get_image_text_masks(bs_inputs["input_ids"])
        collector_bs = HiddenStateCollector(layers)
        _ = model(**bs_inputs)
        bs_hidden = {l: collector_bs.hidden_states[l] for l in collector_bs.hidden_states}
        collector_bs.remove()

        # Store mean-pooled hidden states for domain probing
        for l in range(NUM_LAYERS):
            if l in sg_hidden and l in bs_hidden:
                sg_mean = sg_hidden[l][0].mean(dim=0).cpu().float().numpy()
                bs_mean = bs_hidden[l][0].mean(dim=0).cpu().float().numpy()
                hidden_states_for_probing[l].append(("sg", sg_mean))
                hidden_states_for_probing[l].append(("bs", bs_mean))

        # ── 3. Image-token counterfactual per layer ──
        for l in range(NUM_LAYERS):
            if l not in bs_hidden:
                continue

            sg_cf_inputs = prepare_input(processor, sg_rec, device)
            replacer = TokenLevelReplacer(
                layers, l, bs_hidden[l],
                sg_img_mask, bs_img_mask, mode='image'
            )
            try:
                cf_traj, _ = generate_trajectory(model, processor, sg_cf_inputs, cfg)
            finally:
                replacer.remove()

            disp = trajectory_displacement(clean_traj, cf_traj)
            if disp is not None:
                cis_img[l].append(disp)

        # ── 4. Text-token counterfactual per layer ──
        for l in range(NUM_LAYERS):
            if l not in bs_hidden:
                continue

            sg_cf_inputs = prepare_input(processor, sg_rec, device)
            replacer = TokenLevelReplacer(
                layers, l, bs_hidden[l],
                sg_txt_mask, bs_txt_mask, mode='text'
            )
            try:
                cf_traj, _ = generate_trajectory(model, processor, sg_cf_inputs, cfg)
            finally:
                replacer.remove()

            disp = trajectory_displacement(clean_traj, cf_traj)
            if disp is not None:
                cis_txt[l].append(disp)

        # Progress
        if (pair_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (pair_idx + 1) * (n_pairs - pair_idx - 1)
            print(f"    [{pair_idx+1}/{n_pairs}] {elapsed:.0f}s, ETA {eta:.0f}s, "
                  f"parse_ok={parse_ok}, fail={parse_fail}")

    print(f"\n  Done in {time.time()-t0:.0f}s. Parse OK: {parse_ok}, Fail: {parse_fail}")

    # ── Aggregate CIS ──
    results = {}
    for l in range(NUM_LAYERS):
        results[l] = {
            "cis_img_mean": round(float(np.mean(cis_img[l])), 4) if cis_img[l] else 0.0,
            "cis_img_std": round(float(np.std(cis_img[l])), 4) if cis_img[l] else 0.0,
            "cis_txt_mean": round(float(np.mean(cis_txt[l])), 4) if cis_txt[l] else 0.0,
            "cis_txt_std": round(float(np.std(cis_txt[l])), 4) if cis_txt[l] else 0.0,
            "n_samples": len(cis_img[l]),
            "layer_type": LAYER_TYPES.get(l, "UNK"),
        }

    # ── Domain Probing ──
    print(f"\n  Running domain probing...")
    for l in range(NUM_LAYERS):
        if l not in hidden_states_for_probing or len(hidden_states_for_probing[l]) < 20:
            results[l]["dpa"] = 0.5
            continue

        data = hidden_states_for_probing[l]
        X = np.array([d[1] for d in data])
        y = np.array([0 if d[0] == "sg" else 1 for d in data])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(y)//2), scoring='accuracy')
        dpa = float(np.mean(scores))
        results[l]["dpa"] = round(dpa, 4)

    return results


# ════════════════════════════════════════════════════════════════
#  Layer Partition (Triple)
# ════════════════════════════════════════════════════════════════

def partition_layers(results):
    """
    Three-way partition based on CIS_img, CIS_txt, and DPA.

    phi_align:  CIS_img high AND DPA high → image domain interferes
    phi_tune:   CIS_txt high AND DPA low  → text-critical, domain-safe
    phi_freeze: everything else
    """
    cis_img_vals = [results[l]["cis_img_mean"] for l in range(NUM_LAYERS)]
    cis_txt_vals = [results[l]["cis_txt_mean"] for l in range(NUM_LAYERS)]
    dpa_vals = [results[l]["dpa"] for l in range(NUM_LAYERS)]

    # Thresholds: median
    img_thresh = float(np.median(cis_img_vals))
    txt_thresh = float(np.median(cis_txt_vals))
    dpa_thresh = 0.7  # DPA > 70% = significant domain info

    phi_align = []
    phi_tune = []
    phi_freeze = []

    for l in range(NUM_LAYERS):
        cis_img = results[l]["cis_img_mean"]
        cis_txt = results[l]["cis_txt_mean"]
        dpa = results[l]["dpa"]

        if cis_img >= img_thresh and dpa >= dpa_thresh:
            phi_align.append(l)
        elif cis_txt >= txt_thresh and dpa < dpa_thresh:
            phi_tune.append(l)
        else:
            phi_freeze.append(l)

    # Ensure phi_tune is not empty (need at least 1 layer for LoRA)
    if not phi_tune and phi_freeze:
        # Move the layer with highest CIS_txt from freeze to tune
        best_l = max(phi_freeze, key=lambda l: results[l]["cis_txt_mean"])
        phi_freeze.remove(best_l)
        phi_tune.append(best_l)

    return {
        "phi_align": sorted(phi_align),
        "phi_tune": sorted(phi_tune),
        "phi_freeze": sorted(phi_freeze),
        "thresholds": {
            "cis_img": round(img_thresh, 4),
            "cis_txt": round(txt_thresh, 4),
            "dpa": dpa_thresh,
        },
    }


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--max_pairs", type=int, default=100)
    args = parser.parse_args()

    cfg = VLConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1 (VL) v11: Triple Diagnosis (Trajectory)")
    print(f"{'=' * 70}")

    PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir or str(CHECKPOINT_DIR / "phase0_vl_merged")
    print(f"  Model: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    model.eval()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    layers = get_language_layers(model)

    sg_records = [json.loads(l) for l in open(DATA_DIR / "test_id.jsonl")]
    bs_records = [json.loads(l) for l in open(DATA_DIR / "test_ood.jsonl")]
    print(f"  SG: {len(sg_records)}, BS: {len(bs_records)}")

    pairs = pair_samples(sg_records, bs_records, max_pairs=args.max_pairs)

    results = run_counterfactual_diagnosis(
        model, processor, sg_records, bs_records, pairs, layers, cfg,
        max_pairs=args.max_pairs,
    )

    # Print results
    print(f"\n  {'Layer':>5} {'Type':>4} {'CIS_img':>8} {'CIS_txt':>8} {'DPA':>6}")
    print(f"  {'='*40}")
    for l in range(NUM_LAYERS):
        r = results[l]
        print(f"  {l:>5} {r['layer_type']:>4} {r['cis_img_mean']:>8.3f} "
              f"{r['cis_txt_mean']:>8.3f} {r['dpa']:>6.3f}")

    with open(PHASE1_OUTPUT / "triple_diagnosis_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Partition
    partition = partition_layers(results)
    with open(PHASE1_OUTPUT / "layer_partition.json", "w") as f:
        json.dump(partition, f, indent=2)

    print(f"\n  Layer Partition:")
    print(f"    phi_align  (MMD):  {partition['phi_align']}")
    print(f"    phi_tune   (LoRA): {partition['phi_tune']}")
    print(f"    phi_freeze:        {partition['phi_freeze']}")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ls = list(range(NUM_LAYERS))
        types = [results[l]["layer_type"] for l in ls]
        colors = ["#4472C4" if t == "GQA" else "#ED7D31" for t in types]

        axes[0].bar(ls, [results[l]["cis_img_mean"] for l in ls], color=colors)
        axes[0].set_ylabel("CIS_img (m)")
        axes[0].set_title("Image-Token Counterfactual: Trajectory Displacement")
        axes[0].axhline(y=partition["thresholds"]["cis_img"], color="red", linestyle="--", label="threshold")
        axes[0].legend()

        axes[1].bar(ls, [results[l]["cis_txt_mean"] for l in ls], color=colors)
        axes[1].set_ylabel("CIS_txt (m)")
        axes[1].set_title("Text-Token Counterfactual: Trajectory Displacement")
        axes[1].axhline(y=partition["thresholds"]["cis_txt"], color="red", linestyle="--")

        axes[2].bar(ls, [results[l]["dpa"] for l in ls], color=colors)
        axes[2].set_ylabel("DPA")
        axes[2].set_title("Domain Probing Accuracy")
        axes[2].axhline(y=0.5, color="gray", linestyle=":")
        axes[2].axhline(y=partition["thresholds"]["dpa"], color="red", linestyle="--")
        axes[2].set_xlabel("Layer")
        axes[2].set_xticks(ls)
        axes[2].set_xticklabels([f"L{l}\n({types[l]})" for l in ls], fontsize=8)

        plt.tight_layout()
        fig_path = PHASE1_OUTPUT / "triple_diagnosis.png"
        plt.savefig(str(fig_path), dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  Plot failed: {e}")

    del model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"  Phase 1 v11 Triple Diagnosis Complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

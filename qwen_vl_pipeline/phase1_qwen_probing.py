"""
Phase 1 (Qwen) — Layer-wise Causal Probing for Cross-Architecture Comparison
==============================================================================

Same diagnostic as LFM2 Phase 1 but on Qwen2-VL-2B:
  1. Image-token counterfactual: replace image tokens (ID→OOD), measure trajectory shift
  2. Text-token counterfactual: replace text tokens (ID→OOD), measure trajectory shift
  3. Domain probing: linear classifier to distinguish ID vs OOD from hidden states

Qwen2-VL has 28 standard transformer layers (no LTC/LIV distinction).
Comparing CIS profiles between LFM2 (LTC hybrid) and Qwen (pure transformer)
reveals whether LTC's causal inductive bias creates different domain encoding patterns.

Usage:
  CUDA_VISIBLE_DEVICES=1 python qwen_vl_pipeline/phase1_qwen_probing.py [--max_pairs 200]
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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_qwen import (
    QwenVLConfig, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_LAYERS, HIDDEN_DIM, NUSCENES_DATAROOT,
)

PHASE1_OUTPUT = OUTPUT_DIR / "phase1_qwen"
IMAGE_TOKEN_ID = 151655  # <|image_pad|> in Qwen2-VL

# TVG constants
INSTRUCTION = "Predict future trajectory as 6 waypoints (x,y) in ego-frame metres at t+0.5s to t+3.0s."
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")
TARGET_SIZE = (256, 256)


# ════════════════════════════════════════════════════════════════
#  Trajectory Parsing
# ════════════════════════════════════════════════════════════════

def parse_trajectory(text):
    pattern = r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?'
    matches = re.findall(pattern, text.strip())
    if not matches:
        return None
    waypoints = []
    for xstr, ystr in matches:
        try:
            waypoints.append((float(xstr), float(ystr)))
        except ValueError:
            continue
    return waypoints if waypoints else None


def trajectory_displacement(traj1, traj2):
    if traj1 is None or traj2 is None:
        return None
    n = min(len(traj1), len(traj2))
    if n == 0:
        return None
    dists = [np.sqrt((traj1[i][0]-traj2[i][0])**2 + (traj1[i][1]-traj2[i][1])**2)
             for i in range(n)]
    return float(np.mean(dists))


# ════════════════════════════════════════════════════════════════
#  Model Utilities
# ════════════════════════════════════════════════════════════════

def get_language_layers(model):
    """Get language model layers for Qwen2-VL."""
    layers = model.model.language_model.layers
    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"
    return layers


def build_temporal_map(nusc, dataroot):
    """sample_token -> (prev_image_path, curr_image_path, next_image_path)."""
    temporal_map = {}
    for sample in nusc.sample:
        token = sample['token']
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        curr_path = os.path.join(dataroot, cam_data['filename'])
        prev_path = curr_path
        if sample['prev']:
            try:
                prev_sample = nusc.get('sample', sample['prev'])
                prev_cam = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])
                prev_path = os.path.join(dataroot, prev_cam['filename'])
            except Exception:
                pass
        next_path = curr_path
        if sample['next']:
            try:
                next_sample = nusc.get('sample', sample['next'])
                next_cam = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
                next_path = os.path.join(dataroot, next_cam['filename'])
            except Exception:
                pass
        temporal_map[token] = (prev_path, curr_path, next_path)
    return temporal_map


def prepare_input(processor, record, device, tvg_mode=False, temporal_map=None):
    if tvg_mode and temporal_map and record.get("sample_token") in temporal_map:
        # 3-frame TVG input — no text features, just images + instruction
        token = record["sample_token"]
        prev_path, curr_path, next_path = temporal_map[token]
        images = []
        for path in [prev_path, curr_path, next_path]:
            try:
                img = Image.open(path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)))
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
    else:
        # Single-image Phase 0 input
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
    return parse_trajectory(generated), generated


def get_image_text_masks(input_ids):
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
    def __init__(self, layers, target_layer, replacement_hidden,
                 src_mask, rep_mask, mode='image'):
        self.target_layer = target_layer
        self.replacement = replacement_hidden
        self.src_mask = src_mask
        self.rep_mask = rep_mask
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
            src_indices = torch.where(self.src_mask)[0]
            rep_indices = torch.where(self.rep_mask)[0]
        else:
            src_indices = torch.where(self.src_mask)[0]
            rep_indices = torch.where(self.rep_mask)[0]

        n_replace = min(len(src_indices), len(rep_indices))
        if n_replace > 0:
            new_h[0, src_indices[:n_replace], :] = rep[0, rep_indices[:n_replace], :]

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h

    def remove(self):
        if self.hook:
            self.hook.remove()


# ════════════════════════════════════════════════════════════════
#  Pairing (same as LFM2)
# ════════════════════════════════════════════════════════════════

def pair_samples(id_records, ood_records, max_pairs=200, seed=42):
    def extract_kin(rec):
        return np.array([
            rec.get("ego_speed", 0.0),
            rec.get("front_distance", 50.0),
            rec.get("lead_velocity", 0.0),
        ])

    id_kin = np.array([extract_kin(r) for r in id_records])
    ood_kin = np.array([extract_kin(r) for r in ood_records])

    combined = np.vstack([id_kin, ood_kin])
    mu = combined.mean(axis=0)
    std = combined.std(axis=0) + 1e-8
    id_norm = (id_kin - mu) / std
    ood_norm = (ood_kin - mu) / std

    dists = cdist(id_norm, ood_norm, metric="euclidean")

    pairs = []
    used_ood = set()
    id_order = np.argsort(dists.min(axis=1))

    for id_idx in id_order:
        if len(pairs) >= max_pairs:
            break
        sorted_ood = np.argsort(dists[id_idx])
        for ood_idx in sorted_ood:
            if ood_idx not in used_ood:
                pairs.append((int(id_idx), int(ood_idx)))
                used_ood.add(ood_idx)
                break

    pair_dists = [dists[s, b] for s, b in pairs]
    print(f"  Paired {len(pairs)} samples (kin dist: mean={np.mean(pair_dists):.3f})")
    return pairs


# ════════════════════════════════════════════════════════════════
#  Triple Diagnosis
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_counterfactual_diagnosis(model, processor, id_records, ood_records,
                                  pairs, layers, cfg, max_pairs=100,
                                  tvg_mode=False, temporal_map=None):
    model.eval()
    device = next(model.parameters()).device

    cis_img = defaultdict(list)
    cis_txt = defaultdict(list)
    hidden_states_for_probing = defaultdict(list)

    n_pairs = min(len(pairs), max_pairs)
    parse_ok = 0
    parse_fail = 0

    print(f"\n  Running triple diagnosis on {n_pairs} pairs (tvg_mode={tvg_mode})...")
    t0 = time.time()

    for pair_idx, (id_idx, ood_idx) in enumerate(pairs[:n_pairs]):
        id_rec = id_records[id_idx]
        ood_rec = ood_records[ood_idx]

        if (pair_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (pair_idx + 1) * (n_pairs - pair_idx - 1)
            print(f"    Pair {pair_idx+1}/{n_pairs} (ok={parse_ok}, fail={parse_fail}, "
                  f"ETA={eta/60:.1f}min)")

        # 1. Clean ID trajectory
        id_inputs = prepare_input(processor, id_rec, device, tvg_mode, temporal_map)
        id_img_mask, id_txt_mask = get_image_text_masks(id_inputs["input_ids"])

        clean_traj, _ = generate_trajectory(model, processor, id_inputs, cfg)
        if clean_traj is None:
            parse_fail += 1
            continue
        parse_ok += 1

        # 2. Collect hidden states
        id_inputs2 = prepare_input(processor, id_rec, device, tvg_mode, temporal_map)
        collector_id = HiddenStateCollector(layers)
        _ = model(**id_inputs2)
        id_hidden = {l: collector_id.hidden_states[l] for l in collector_id.hidden_states}
        collector_id.remove()

        ood_inputs = prepare_input(processor, ood_rec, device, tvg_mode, temporal_map)
        ood_img_mask, ood_txt_mask = get_image_text_masks(ood_inputs["input_ids"])
        collector_ood = HiddenStateCollector(layers)
        _ = model(**ood_inputs)
        ood_hidden = {l: collector_ood.hidden_states[l] for l in collector_ood.hidden_states}
        collector_ood.remove()

        # Domain probing hidden states
        for l in range(NUM_LAYERS):
            if l in id_hidden and l in ood_hidden:
                id_mean = id_hidden[l][0].mean(dim=0).cpu().float().numpy()
                ood_mean = ood_hidden[l][0].mean(dim=0).cpu().float().numpy()
                hidden_states_for_probing[l].append(("id", id_mean))
                hidden_states_for_probing[l].append(("ood", ood_mean))

        # 3. Image counterfactual per layer
        for l in range(NUM_LAYERS):
            if l not in ood_hidden:
                continue

            # Image CF: replace ID image tokens with OOD image tokens
            replacer = TokenLevelReplacer(
                layers, l, ood_hidden[l],
                id_img_mask, ood_img_mask, mode='image'
            )
            cf_inputs = prepare_input(processor, id_rec, device, tvg_mode, temporal_map)
            cf_traj, _ = generate_trajectory(model, processor, cf_inputs, cfg)
            replacer.remove()

            disp = trajectory_displacement(clean_traj, cf_traj)
            if disp is not None:
                cis_img[l].append(disp)

        # 4. Text counterfactual per layer
        for l in range(NUM_LAYERS):
            if l not in ood_hidden:
                continue

            replacer = TokenLevelReplacer(
                layers, l, ood_hidden[l],
                id_txt_mask, ood_txt_mask, mode='text'
            )
            cf_inputs = prepare_input(processor, id_rec, device, tvg_mode, temporal_map)
            cf_traj, _ = generate_trajectory(model, processor, cf_inputs, cfg)
            replacer.remove()

            disp = trajectory_displacement(clean_traj, cf_traj)
            if disp is not None:
                cis_txt[l].append(disp)

        # Free memory
        del id_hidden, ood_hidden
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f}min (ok={parse_ok}, fail={parse_fail})")

    # 5. Domain probing
    print("\n  Running domain probing...")
    dpa_scores = {}
    for l in range(NUM_LAYERS):
        samples = hidden_states_for_probing.get(l, [])
        if len(samples) < 20:
            continue
        labels = [1 if s[0] == "ood" else 0 for s in samples]
        features = np.array([s[1] for s in samples])

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')
        dpa_scores[l] = float(np.mean(scores))

    return cis_img, cis_txt, dpa_scores


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def partition_layers(cis_img_means, cis_txt_means, num_layers):
    """Partition layers into three zones based on CIS profiles (same logic as LFM2)."""
    img_median = np.median(cis_img_means)
    txt_median = np.median(cis_txt_means)

    phi_align = []   # high CIS_img → visual alignment (MMD)
    phi_tune = []    # high CIS_txt → textual tuning (LoRA)
    phi_freeze = []  # low both → freeze

    for l in range(num_layers):
        high_img = cis_img_means[l] > img_median
        high_txt = cis_txt_means[l] > txt_median
        if high_img and not high_txt:
            phi_align.append(l)
        elif high_txt:
            phi_tune.append(l)
        else:
            phi_freeze.append(l)

    return {"phi_align": phi_align, "phi_tune": phi_tune, "phi_freeze": phi_freeze}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_pairs", type=int, default=200)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "av2"])
    parser.add_argument("--tvg", action="store_true",
                        help="Use 3-frame TVG input (for TVG-trained models)")
    args = parser.parse_args()

    cfg = QwenVLConfig()

    # ── Dataset dispatch (nuscenes/av2) ─────────────────────
    from pathlib import Path as _P
    if args.dataset == "av2":
        data_dir = _P("argoverse2_pipeline/outputs/data")
        default_tvg_ckpt = _P(CHECKPOINT_DIR) / "tvg_qwen_av2" / "tvg_merged"
        phase1_output_subdir = "phase1_qwen_tvg_av2" if args.tvg else "phase1_qwen_av2"
    else:
        data_dir = DATA_DIR
        default_tvg_ckpt = _P(CHECKPOINT_DIR) / "tvg_qwen" / "tvg_merged"
        phase1_output_subdir = "phase1_qwen_tvg" if args.tvg else "phase1_qwen"

    tvg_mode = args.tvg
    if tvg_mode:
        output_dir = OUTPUT_DIR / phase1_output_subdir
    else:
        output_dir = PHASE1_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1 (Qwen): Layer-wise Causal Probing {'(TVG mode)' if tvg_mode else ''}")
    print(f"  28 standard transformer layers (no LTC)")
    print(f"{'=' * 70}")

    # Load model
    if tvg_mode:
        model_dir = args.model_dir or str(default_tvg_ckpt)
    else:
        model_dir = args.model_dir or str(CHECKPOINT_DIR / "phase0_qwen_merged")
    print(f"\n[Phase 1] Loading model from {model_dir}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    if tvg_mode:
        processor.image_processor.min_pixels = 256 * 256
        processor.image_processor.max_pixels = 256 * 256
    model.eval()

    layers = get_language_layers(model)
    print(f"  Language model layers: {len(layers)}")

    # Load temporal map if TVG mode (dataset-dependent)
    temporal_map = None
    if tvg_mode:
        if args.dataset == "av2":
            # AV2 JSONL already carries image_paths as a 3-list; build a
            # synthetic temporal_map keyed by sample_token for uniform interface.
            import json as _json
            temporal_map = {}
            for split in ["train.jsonl", "val.jsonl", "test_id.jsonl", "test_ood.jsonl"]:
                p = data_dir / split
                if not p.exists():
                    continue
                with open(p) as _f:
                    for _line in _f:
                        _rec = _json.loads(_line)
                        _tok = _rec.get("sample_token")
                        _ips = _rec.get("image_paths")
                        if _tok and isinstance(_ips, list) and len(_ips) == 3:
                            temporal_map[_tok] = tuple(_ips)
            print(f"  Built AV2 synthetic temporal_map from image_paths: {len(temporal_map)} samples")
        else:
            print("[Phase 1] Loading nuScenes for temporal frame mapping...")
            nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DATAROOT, verbose=False)
            temporal_map = build_temporal_map(nusc, NUSCENES_DATAROOT)
            print(f"  Built temporal map for {len(temporal_map)} samples")
            del nusc

    # Load test data
    print("[Phase 1] Loading test data...")
    with open(data_dir / "test_id.jsonl") as f:
        id_records = [json.loads(l) for l in f]
    with open(data_dir / "test_ood.jsonl") as f:
        ood_records = [json.loads(l) for l in f]

    # Filter to records with temporal map if TVG mode
    if tvg_mode and temporal_map:
        id_records = [r for r in id_records if r.get("sample_token") in temporal_map]
        ood_records = [r for r in ood_records if r.get("sample_token") in temporal_map]

    print(f"  ID: {len(id_records)}, OOD: {len(ood_records)}")

    # Pair samples
    pairs = pair_samples(id_records, ood_records, max_pairs=args.max_pairs)

    # Run diagnosis
    cis_img, cis_txt, dpa_scores = run_counterfactual_diagnosis(
        model, processor, id_records, ood_records,
        pairs, layers, cfg, max_pairs=args.max_pairs,
        tvg_mode=tvg_mode, temporal_map=temporal_map,
    )

    # Compile results
    summary = {}
    cis_img_means = []
    cis_txt_means = []
    for l in range(NUM_LAYERS):
        img_mean = float(np.mean(cis_img[l])) if cis_img[l] else 0.0
        txt_mean = float(np.mean(cis_txt[l])) if cis_txt[l] else 0.0
        cis_img_means.append(img_mean)
        cis_txt_means.append(txt_mean)
        layer_info = {
            "layer": l + 1,
            "layer_type": "ATTN",
            "cis_img_mean": img_mean,
            "cis_img_std": float(np.std(cis_img[l])) if cis_img[l] else 0.0,
            "cis_img_n": len(cis_img[l]),
            "cis_txt_mean": txt_mean,
            "cis_txt_std": float(np.std(cis_txt[l])) if cis_txt[l] else 0.0,
            "cis_txt_n": len(cis_txt[l]),
            "dpa": dpa_scores.get(l, None),
        }
        summary[f"layer_{l+1}"] = layer_info

    # Save summary
    out_path = output_dir / "phase1_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Phase 1] Results saved to {out_path}")

    # Compute and save partition
    partition = partition_layers(cis_img_means, cis_txt_means, NUM_LAYERS)
    part_path = output_dir / "layer_partition.json"
    with open(part_path, "w") as f:
        json.dump(partition, f, indent=2)
    print(f"[Phase 1] Partition saved to {part_path}")
    print(f"  phi_align  (MMD):  {partition['phi_align']}")
    print(f"  phi_tune   (LoRA): {partition['phi_tune']}")
    print(f"  phi_freeze:        {partition['phi_freeze']}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  Layer | CIS_img (mean) | CIS_txt (mean) | DPA    | Zone")
    print(f"  {'─' * 65}")
    for l in range(NUM_LAYERS):
        info = summary[f"layer_{l+1}"]
        dpa_str = f"{info['dpa']:.3f}" if info['dpa'] else "N/A"
        if l in partition['phi_align']:
            zone = "ALIGN"
        elif l in partition['phi_tune']:
            zone = "TUNE"
        else:
            zone = "FREEZE"
        print(f"  {l+1:3d}   | {info['cis_img_mean']:12.4f}   | {info['cis_txt_mean']:12.4f}   | {dpa_str:>6} | {zone}")

    print(f"\n{'=' * 70}")
    print(f"  Phase 1 (Qwen) Causal Probing Complete {'(TVG)' if tvg_mode else ''}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

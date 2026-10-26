"""
Temporal Frames Verification Experiment
========================================

Quick test: Does providing 3 consecutive frames (t-1, t, t+1) instead of
a single frame change the model's trajectory prediction?

This tests whether the Phase 0 fine-tuned LFM2-VL model can leverage
temporal visual information (motion cues across frames).

We compare:
  1. Single-frame inference (standard)
  2. Three-frame inference (t-0.5s, t, t+0.5s)

If ADE is similar → model doesn't use temporal visual info (consistent with
image ablation finding that model relies on text features).

Usage:
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/temporal_frames_eval.py [--n_samples 200]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from transformers import AutoModelForImageTextToText, AutoProcessor

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import VLConfig, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR


def parse_trajectory(text):
    """Parse (x,y) waypoints from model output."""
    import re
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


def compute_ade(pred, gt):
    """Compute ADE between pred and gt trajectories."""
    n = min(len(pred), len(gt))
    if n == 0:
        return None
    disps = [np.sqrt((pred[i][0]-gt[i][0])**2 + (pred[i][1]-gt[i][1])**2) for i in range(n)]
    return float(np.mean(disps))


def get_temporal_frames(nusc, sample_token, dataroot):
    """Get (prev_image_path, curr_image_path, next_image_path) for a sample.
    Returns None for prev/next if they don't exist (first/last in scene)."""
    sample = nusc.get('sample', sample_token)

    curr_cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    curr_path = os.path.join(dataroot, curr_cam['filename'])

    prev_path = None
    if sample['prev']:
        prev_sample = nusc.get('sample', sample['prev'])
        prev_cam = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])
        prev_path = os.path.join(dataroot, prev_cam['filename'])

    next_path = None
    if sample['next']:
        next_sample = nusc.get('sample', sample['next'])
        next_cam = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
        next_path = os.path.join(dataroot, next_cam['filename'])

    return prev_path, curr_path, next_path


def run_single_frame(model, processor, image_path, question, cfg):
    """Standard single-frame inference."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))

    conversation = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]

    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, tokenize=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    outputs = model.generate(
        **inputs, max_new_tokens=cfg.max_new_tokens,
        do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def run_temporal_frames(model, processor, prev_path, curr_path, next_path, question, cfg):
    """Three-frame inference: provide t-1, t, t+1 images."""
    images = []
    for p in [prev_path, curr_path, next_path]:
        if p and os.path.exists(p):
            images.append(Image.open(p).convert("RGB"))
        else:
            # Use current frame as fallback for missing prev/next
            images.append(Image.open(curr_path).convert("RGB"))

    # Build multi-image conversation
    # Prefix with temporal context instruction
    temporal_prefix = ("You are given 3 consecutive driving camera frames "
                       "(t-0.5s, t, t+0.5s). Use the motion information across frames. ")

    content = [
        {"type": "image", "image": images[0]},  # t-1
        {"type": "image", "image": images[1]},  # t
        {"type": "image", "image": images[2]},  # t+1
        {"type": "text", "text": temporal_prefix + question},
    ]

    conversation = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, tokenize=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    outputs = model.generate(
        **inputs, max_new_tokens=cfg.max_new_tokens,
        do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to evaluate (subset for speed)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (default: phase0 merged checkpoint)")
    parser.add_argument("--split", type=str, default="both",
                        choices=["id", "ood", "both"])
    args = parser.parse_args()

    cfg = VLConfig()

    # Load model
    model_dir = args.model_dir or str(CHECKPOINT_DIR / "phase0_vl_merged")
    print(f"Loading model from {model_dir}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()

    # Load nuScenes for temporal frame lookup
    print("Loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)

    # Load test data
    splits = {}
    if args.split in ("id", "both"):
        with open(DATA_DIR / "test_id.jsonl") as f:
            splits["ID"] = [json.loads(l) for l in f]
    if args.split in ("ood", "both"):
        with open(DATA_DIR / "test_ood.jsonl") as f:
            splits["OOD"] = [json.loads(l) for l in f]

    results = {}

    for split_name, records in splits.items():
        # Subsample
        rng = np.random.RandomState(42)
        if len(records) > args.n_samples:
            indices = rng.choice(len(records), args.n_samples, replace=False)
            records = [records[i] for i in sorted(indices)]

        print(f"\n{'='*60}")
        print(f"  {split_name}: evaluating {len(records)} samples")
        print(f"{'='*60}")

        single_ades = []
        temporal_ades = []
        parse_fail_single = 0
        parse_fail_temporal = 0
        sample_comparisons = []

        for rec in tqdm(records, desc=f"  {split_name}"):
            gt_traj = parse_trajectory(rec.get("answer", ""))
            if gt_traj is None:
                continue

            token = rec.get("sample_token")
            if not token:
                continue

            # Get temporal frames
            prev_path, curr_path, next_path = get_temporal_frames(
                nusc, token, 'data/nuscenes'
            )

            # Single frame inference
            out_single = run_single_frame(model, processor, rec["image_path"], rec["question"], cfg)
            pred_single = parse_trajectory(out_single)

            if pred_single is None:
                parse_fail_single += 1
            else:
                ade_s = compute_ade(pred_single, gt_traj)
                if ade_s is not None:
                    single_ades.append(ade_s)

            # Temporal (3-frame) inference
            out_temporal = run_temporal_frames(
                model, processor, prev_path, curr_path, next_path, rec["question"], cfg
            )
            pred_temporal = parse_trajectory(out_temporal)

            if pred_temporal is None:
                parse_fail_temporal += 1
            else:
                ade_t = compute_ade(pred_temporal, gt_traj)
                if ade_t is not None:
                    temporal_ades.append(ade_t)

            # Save comparison for first few
            if len(sample_comparisons) < 10 and pred_single and pred_temporal:
                sample_comparisons.append({
                    "question": rec["question"][:80],
                    "gt": str(gt_traj[:3]) + "...",
                    "single_pred": str(pred_single[:3]) + "...",
                    "temporal_pred": str(pred_temporal[:3]) + "...",
                    "single_ade": compute_ade(pred_single, gt_traj),
                    "temporal_ade": compute_ade(pred_temporal, gt_traj),
                })

        split_result = {
            "n_evaluated": len(records),
            "single_frame": {
                "n_valid": len(single_ades),
                "parse_failures": parse_fail_single,
                "ade_mean": float(np.mean(single_ades)) if single_ades else None,
                "ade_std": float(np.std(single_ades)) if single_ades else None,
                "ade_median": float(np.median(single_ades)) if single_ades else None,
            },
            "temporal_3frame": {
                "n_valid": len(temporal_ades),
                "parse_failures": parse_fail_temporal,
                "ade_mean": float(np.mean(temporal_ades)) if temporal_ades else None,
                "ade_std": float(np.std(temporal_ades)) if temporal_ades else None,
                "ade_median": float(np.median(temporal_ades)) if temporal_ades else None,
            },
            "sample_comparisons": sample_comparisons,
        }

        # Compute delta
        if single_ades and temporal_ades:
            split_result["ade_delta"] = split_result["temporal_3frame"]["ade_mean"] - split_result["single_frame"]["ade_mean"]
            split_result["ade_delta_pct"] = (split_result["ade_delta"] / split_result["single_frame"]["ade_mean"]) * 100

        results[split_name.lower()] = split_result

        print(f"\n  {split_name} Results:")
        print(f"    Single-frame ADE:  {split_result['single_frame']['ade_mean']:.4f} "
              f"(parse_fail={parse_fail_single})")
        print(f"    Temporal ADE:      {split_result['temporal_3frame']['ade_mean']:.4f} "
              f"(parse_fail={parse_fail_temporal})")
        if "ade_delta" in split_result:
            print(f"    Delta: {split_result['ade_delta']:+.4f}m ({split_result['ade_delta_pct']:+.2f}%)")

        # Print sample comparisons
        print(f"\n  Sample comparisons:")
        for sc in sample_comparisons[:5]:
            print(f"    GT={sc['gt']}  Single={sc['single_pred']}  Temporal={sc['temporal_pred']}")
            print(f"      ADE: single={sc['single_ade']:.3f}  temporal={sc['temporal_ade']:.3f}")

    # Save results
    out_path = OUTPUT_DIR / "temporal_frames_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

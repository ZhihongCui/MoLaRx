"""
Image Ablation Experiment
=========================
Runs the same trajectory evaluation as phase0_vl_evaluate.py but replaces
all real images with blank gray images. If ADE barely changes compared to
the normal evaluation, the model is ignoring images. If ADE jumps
significantly, images are being used and are domain-robust.

Usage:
  CUDA_VISIBLE_DEVICES=1 python lfm2_vl_pipeline/image_ablation_eval.py
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from config_vl import (
    VLConfig,
    DATA_DIR,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
)


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


def compute_ade_fde(pred_traj, gt_traj):
    n = min(len(pred_traj), len(gt_traj))
    if n == 0:
        return None, None
    displacements = []
    for i in range(n):
        dx = pred_traj[i][0] - gt_traj[i][0]
        dy = pred_traj[i][1] - gt_traj[i][1]
        displacements.append(np.sqrt(dx**2 + dy**2))
    return float(np.mean(displacements)), float(displacements[-1])


@torch.no_grad()
def evaluate_split(model, processor, records, cfg, split_name="test", use_blank_image=True):
    model.eval()
    ade_list, fde_list = [], []
    parse_failures, total = 0, 0

    # Create a blank gray image (same size as typical nuScenes camera)
    blank_image = Image.new("RGB", (1600, 900), color=(128, 128, 128))

    for rec in tqdm(records, desc=f"  {split_name}"):
        if use_blank_image:
            image = blank_image
        else:
            try:
                image = Image.open(rec["image_path"]).convert("RGB")
            except Exception:
                image = blank_image

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated = processor.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True,
        )

        total += 1
        pred_traj = parse_trajectory(generated)
        gt_traj = rec.get("trajectory", None)
        if gt_traj is None:
            gt_traj = parse_trajectory(rec.get("answer", ""))

        if pred_traj is None:
            parse_failures += 1
            continue

        if gt_traj is None or len(gt_traj) == 0:
            continue

        ade, fde = compute_ade_fde(pred_traj, gt_traj)
        if ade is not None:
            ade_list.append(ade)
            fde_list.append(fde)

    results = {
        "total": total,
        "parse_failures": parse_failures,
        "parse_rate": 1.0 - parse_failures / max(total, 1),
        "n_evaluated": len(ade_list),
        "ade_mean": float(np.mean(ade_list)) if ade_list else None,
        "ade_median": float(np.median(ade_list)) if ade_list else None,
        "fde_mean": float(np.mean(fde_list)) if fde_list else None,
        "fde_median": float(np.median(fde_list)) if fde_list else None,
    }
    return results


def load_records(split_file):
    records = []
    with open(split_file) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = VLConfig()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = str(CHECKPOINT_DIR / "phase0_vl_merged")

    print("=" * 60)
    print("  IMAGE ABLATION EXPERIMENT")
    print(f"  Model: {model_dir}")
    print("  All images replaced with blank gray (1600x900)")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load test splits
    id_file = DATA_DIR / "test_id.jsonl"
    ood_file = DATA_DIR / "test_ood.jsonl"

    id_records = load_records(id_file)
    ood_records = load_records(ood_file)

    if args.max_samples:
        id_records = id_records[:args.max_samples]
        ood_records = ood_records[:args.max_samples]

    print(f"\n  ID samples: {len(id_records)}")
    print(f"  OOD samples: {len(ood_records)}")

    # Run with blank images
    print("\n--- Evaluation with BLANK images ---")
    id_blank = evaluate_split(model, processor, id_records, cfg,
                              split_name="ID (blank img)", use_blank_image=True)
    ood_blank = evaluate_split(model, processor, ood_records, cfg,
                               split_name="OOD (blank img)", use_blank_image=True)

    print("\n" + "=" * 60)
    print("  RESULTS: BLANK IMAGES")
    print("=" * 60)
    print(f"  ID:  ADE={id_blank['ade_mean']:.3f}m, FDE={id_blank['fde_mean']:.3f}m")
    print(f"  OOD: ADE={ood_blank['ade_mean']:.3f}m, FDE={ood_blank['fde_mean']:.3f}m")
    ade_gap = ood_blank['ade_mean'] - id_blank['ade_mean']
    print(f"  ADE Gap: {ade_gap:+.3f}m")
    print()
    print("  Compare with normal eval (Phase 0 baseline):")
    print("    ID:  ADE=0.910m  |  OOD: ADE=1.232m  |  Gap: 0.323m")
    print()
    print("  If blank ADE ≈ normal ADE → model ignores images")
    print("  If blank ADE >> normal ADE → model uses images (good!)")
    print("=" * 60)

    # Save results
    results_path = OUTPUT_DIR / "image_ablation_eval.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "blank_image": {"id": id_blank, "ood": ood_blank, "ade_gap": ade_gap},
            "reference_normal": {
                "id_ade": 0.910, "ood_ade": 1.232, "ade_gap": 0.323,
            },
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

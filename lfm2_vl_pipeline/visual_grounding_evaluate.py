"""
Visual Grounding Evaluation
============================

Evaluate the VG model in 3 modes:
  1. 3-frame temporal (standard for VG model)
  2. 3-frame temporal with blank images (image ablation)
  3. 3-frame temporal with masked text (text ablation)

Compare to measure whether VG training made the model actually use images.

Usage:
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/visual_grounding_evaluate.py
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
from nuscenes.nuscenes import NuScenes
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import VLConfig, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR
from visual_grounding_finetune import mask_text_features, build_temporal_map


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


def compute_ade_fde(pred, gt):
    n = min(len(pred), len(gt))
    if n == 0:
        return None, None
    disps = [np.sqrt((pred[i][0]-gt[i][0])**2 + (pred[i][1]-gt[i][1])**2) for i in range(n)]
    return float(np.mean(disps)), float(disps[-1])


@torch.no_grad()
def evaluate_mode(model, processor, records, temporal_map, cfg, mode="temporal",
                  split_name="test", n_samples=None):
    """
    Evaluate in different modes:
      - 'temporal': 3-frame input with full text
      - 'temporal_blank': 3-frame input with blank (gray) images
      - 'temporal_notext': 3-frame input with fully masked text
    """
    model.eval()

    if n_samples and len(records) > n_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(records), n_samples, replace=False)
        records = [records[i] for i in sorted(indices)]

    ade_list, fde_list = [], []
    parse_failures = 0

    for rec in tqdm(records, desc=f"  {split_name} ({mode})"):
        token = rec.get("sample_token")
        if not token or token not in temporal_map:
            continue

        prev_path, curr_path, next_path = temporal_map[token]
        gt_traj = parse_trajectory(rec.get("answer", ""))
        if gt_traj is None:
            continue

        # Load images based on mode (resize to 512x512 to match training)
        TARGET_SIZE = (512, 512)
        if mode == "temporal_blank":
            images = [Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)) for _ in range(3)]
        else:
            images = []
            for p in [prev_path, curr_path, next_path]:
                try:
                    images.append(Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS))
                except Exception:
                    images.append(Image.new("RGB", TARGET_SIZE, color=(128, 128, 128)))

        # Prepare text
        if mode == "temporal_notext":
            question = mask_text_features(rec["question"], mask_level=2)
        else:
            question = rec["question"]

        temporal_prefix = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                          "Use visual motion cues across frames. ")

        conversation = [{"role": "user", "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "image", "image": images[2]},
            {"type": "text", "text": temporal_prefix + question},
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
        generated = processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        pred_traj = parse_trajectory(generated)
        if pred_traj is None:
            parse_failures += 1
            continue

        ade, fde = compute_ade_fde(pred_traj, gt_traj)
        if ade is not None:
            ade_list.append(ade)
            fde_list.append(fde)

    result = {
        "mode": mode,
        "n_evaluated": len(records),
        "n_valid": len(ade_list),
        "parse_failures": parse_failures,
        "ade_mean": float(np.mean(ade_list)) if ade_list else None,
        "ade_std": float(np.std(ade_list)) if ade_list else None,
        "fde_mean": float(np.mean(fde_list)) if fde_list else None,
    }

    print(f"    {mode}: ADE={result['ade_mean']:.4f}  FDE={result['fde_mean']:.4f}  "
          f"(parse_fail={parse_failures})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--split", type=str, default="both", choices=["id", "ood", "both"])
    args = parser.parse_args()

    cfg = VLConfig()

    model_dir = args.model_dir or str(CHECKPOINT_DIR / "visual_grounding" / "vg_merged")
    print(f"Loading model from {model_dir}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()

    print("Loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    temporal_map = build_temporal_map(nusc, 'data/nuscenes')
    del nusc

    splits = {}
    if args.split in ("id", "both"):
        with open(DATA_DIR / "test_id.jsonl") as f:
            splits["ID"] = [json.loads(l) for l in f]
    if args.split in ("ood", "both"):
        with open(DATA_DIR / "test_ood.jsonl") as f:
            splits["OOD"] = [json.loads(l) for l in f]

    results = {}
    modes = ["temporal", "temporal_blank", "temporal_notext"]

    for split_name, records in splits.items():
        print(f"\n{'='*60}")
        print(f"  {split_name} Split ({len(records)} total, evaluating {args.n_samples})")
        print(f"{'='*60}")

        split_results = {}
        for mode in modes:
            split_results[mode] = evaluate_mode(
                model, processor, records, temporal_map, cfg,
                mode=mode, split_name=split_name, n_samples=args.n_samples,
            )

        # Compute ablation deltas
        if split_results["temporal"]["ade_mean"] and split_results["temporal_blank"]["ade_mean"]:
            img_delta = split_results["temporal_blank"]["ade_mean"] - split_results["temporal"]["ade_mean"]
            img_delta_pct = (img_delta / split_results["temporal"]["ade_mean"]) * 100
            split_results["image_ablation_delta"] = img_delta
            split_results["image_ablation_delta_pct"] = img_delta_pct
            print(f"  Image ablation: ADE +{img_delta:.4f}m ({img_delta_pct:+.2f}%)")

        if split_results["temporal"]["ade_mean"] and split_results["temporal_notext"]["ade_mean"]:
            txt_delta = split_results["temporal_notext"]["ade_mean"] - split_results["temporal"]["ade_mean"]
            txt_delta_pct = (txt_delta / split_results["temporal"]["ade_mean"]) * 100
            split_results["text_ablation_delta"] = txt_delta
            split_results["text_ablation_delta_pct"] = txt_delta_pct
            print(f"  Text ablation:  ADE +{txt_delta:.4f}m ({txt_delta_pct:+.2f}%)")

        results[split_name.lower()] = split_results

    out_path = OUTPUT_DIR / "visual_grounding_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

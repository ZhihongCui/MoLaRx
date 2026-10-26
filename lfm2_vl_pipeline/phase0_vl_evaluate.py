"""
Phase 0 (VL) — Evaluate Trajectory Planning on ID/OOD (v11)
=============================================================

Metrics:
  - ADE (Average Displacement Error): mean L2 over all waypoints
  - FDE (Final Displacement Error): L2 at last waypoint
  - Derived classification accuracy (from trajectory shape)
  - OOD Gap = metric_OOD - metric_ID

Usage:
  python lfm2_vl_pipeline/phase0_vl_evaluate.py
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
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
    ALL_WEATHER_LABELS,
)


# ════════════════════════════════════════════════════════════════
#  Trajectory Parsing
# ════════════════════════════════════════════════════════════════

def parse_trajectory(text):
    """
    Parse trajectory from model output text.
    Expected format: "(x1,y1) (x2,y2) ... (x6,y6)"
    Returns list of (x, y) tuples, or None if unparseable.
    """
    text = text.strip()
    # Find all (x,y) patterns
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
    if len(waypoints) == 0:
        return None
    return waypoints


def compute_ade_fde(pred_traj, gt_traj):
    """
    Compute ADE and FDE between predicted and ground truth trajectories.
    Handles different lengths by using min(len(pred), len(gt)) waypoints.
    """
    n = min(len(pred_traj), len(gt_traj))
    if n == 0:
        return None, None

    displacements = []
    for i in range(n):
        dx = pred_traj[i][0] - gt_traj[i][0]
        dy = pred_traj[i][1] - gt_traj[i][1]
        displacements.append(np.sqrt(dx**2 + dy**2))

    ade = np.mean(displacements)
    fde = displacements[-1]
    return float(ade), float(fde)


def derive_classification(traj):
    """
    Derive longitudinal and lateral classification from trajectory.
    Longitudinal: based on y-displacement pattern (accelerating/constant/decelerating)
    Lateral: based on final x-displacement
    Returns (long_class, lat_class) or (-1, -1) if can't derive.
    """
    if traj is None or len(traj) < 2:
        return -1, -1

    # Lateral: based on final x position
    final_x = traj[-1][0]
    if final_x > 0.5:
        lat = 0  # left
    elif final_x < -0.5:
        lat = 2  # right
    else:
        lat = 1  # straight

    # Longitudinal: based on y-spacing pattern
    # Compare speed in first half vs second half
    mid = len(traj) // 2
    if mid == 0:
        mid = 1

    first_half_dy = traj[mid-1][1] - (traj[0][1] if mid > 0 else 0)
    second_half_dy = traj[-1][1] - traj[mid-1][1]

    if len(traj) >= 4:
        # More reliable: compare spacing
        if second_half_dy > first_half_dy * 1.15:
            lon = 0  # accelerate
        elif second_half_dy < first_half_dy * 0.85:
            lon = 2  # decelerate
        else:
            lon = 1  # maintain
    else:
        lon = 1  # default maintain for short trajectories

    return lon, lat


# ════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_split(model, processor, records, cfg, split_name="test", debug=False):
    """Evaluate trajectory planning on a split."""
    model.eval()

    ade_list = []
    fde_list = []
    parse_failures = 0
    total = 0

    # Derived classification metrics
    long_correct = 0
    lat_correct = 0
    class_total = 0

    per_weather = {}
    sample_outputs = []  # Store a few for debugging

    for rec in tqdm(records, desc=f"  {split_name}"):
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (640, 480), color=(128, 128, 128))

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

        # Parse predicted trajectory
        pred_traj = parse_trajectory(generated)
        gt_traj = rec.get("trajectory", None)
        if gt_traj is None:
            # Try parsing from answer string
            gt_traj = parse_trajectory(rec.get("answer", ""))

        if pred_traj is None:
            parse_failures += 1
            if debug and parse_failures <= 5:
                print(f"  PARSE FAIL: {generated[:100]}")
            continue

        if gt_traj is None:
            continue

        # Compute ADE/FDE
        ade, fde = compute_ade_fde(pred_traj, gt_traj)
        if ade is not None:
            ade_list.append(ade)
            fde_list.append(fde)

        # Derived classification
        pred_lon, pred_lat = derive_classification(pred_traj)
        gt_lon, gt_lat = derive_classification(gt_traj)

        if pred_lon != -1 and gt_lon != -1:
            class_total += 1
            if pred_lon == gt_lon:
                long_correct += 1
            if pred_lat == gt_lat:
                lat_correct += 1

        # Per-weather
        weather = rec.get("weather", "unknown")
        if weather not in per_weather:
            per_weather[weather] = {"ade": [], "fde": [], "t": 0}
        per_weather[weather]["t"] += 1
        if ade is not None:
            per_weather[weather]["ade"].append(ade)
            per_weather[weather]["fde"].append(fde)

        # Store samples for debug
        if len(sample_outputs) < 5:
            sample_outputs.append({
                "gt": gt_traj[:3] if gt_traj else None,
                "pred": pred_traj[:3] if pred_traj else None,
                "ade": round(ade, 3) if ade else None,
                "raw": generated[:80],
            })

    # Aggregate
    results = {
        "total": total,
        "parse_failures": parse_failures,
        "parse_rate": round(1 - parse_failures / max(total, 1), 4),
        "n_evaluated": len(ade_list),
    }

    if len(ade_list) > 0:
        results["ade_mean"] = round(float(np.mean(ade_list)), 4)
        results["ade_median"] = round(float(np.median(ade_list)), 4)
        results["ade_std"] = round(float(np.std(ade_list)), 4)
        results["fde_mean"] = round(float(np.mean(fde_list)), 4)
        results["fde_median"] = round(float(np.median(fde_list)), 4)
        results["fde_std"] = round(float(np.std(fde_list)), 4)
    else:
        results["ade_mean"] = None
        results["fde_mean"] = None

    # Derived classification
    if class_total > 0:
        results["derived_long_accuracy"] = round(long_correct / class_total, 4)
        results["derived_lat_accuracy"] = round(lat_correct / class_total, 4)
        results["derived_class_total"] = class_total

    # Per-weather
    results["per_weather"] = {}
    for w, s in per_weather.items():
        pw = {"total": s["t"]}
        if len(s["ade"]) > 0:
            pw["ade_mean"] = round(float(np.mean(s["ade"])), 4)
            pw["fde_mean"] = round(float(np.mean(s["fde"])), 4)
        results["per_weather"][w] = pw

    results["sample_outputs"] = sample_outputs

    return results


def print_results(name, results):
    if results is None:
        print(f"  {name}: SKIPPED")
        return

    print(f"\n  {name}:")
    print(f"    Total: {results['total']}, Parse failures: {results['parse_failures']} "
          f"(parse rate: {results['parse_rate']:.1%})")
    print(f"    Evaluated: {results['n_evaluated']}")

    if results.get("ade_mean") is not None:
        print(f"    ADE: {results['ade_mean']:.3f}m (median={results['ade_median']:.3f}, std={results['ade_std']:.3f})")
        print(f"    FDE: {results['fde_mean']:.3f}m (median={results['fde_median']:.3f}, std={results['fde_std']:.3f})")
    else:
        print(f"    ADE/FDE: N/A (no valid predictions)")

    if "derived_long_accuracy" in results:
        print(f"    Derived classification:")
        print(f"      Longitudinal: {results['derived_long_accuracy']:.4f}")
        print(f"      Lateral:      {results['derived_lat_accuracy']:.4f}")

    print(f"    Per-weather:")
    for w in ALL_WEATHER_LABELS:
        if w in results.get("per_weather", {}):
            s = results["per_weather"][w]
            ade_str = f"ADE={s['ade_mean']:.3f}m" if "ade_mean" in s else "N/A"
            print(f"      {w:>6}: {ade_str} (n={s['total']})")

    if results.get("sample_outputs"):
        print(f"    Sample outputs:")
        for s in results["sample_outputs"][:3]:
            print(f"      GT={s['gt']}... Pred={s['pred']}... ADE={s['ade']}")
            print(f"        Raw: {s['raw']}")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick testing")
    args = parser.parse_args()

    cfg = VLConfig()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = str(CHECKPOINT_DIR / "phase0_vl_merged")

    print(f"\n{'=' * 70}")
    print(f"  PHASE 0 (VL) v11: Trajectory Planning Evaluation")
    print(f"{'=' * 70}")
    print(f"  Model: {model_dir}")

    if not os.path.isdir(model_dir):
        print(f"[Eval] Model not found: {model_dir}")
        sys.exit(1)

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    model.eval()

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"[Eval] Model loaded. Device: {model.device}")

    test_id = [json.loads(l) for l in open(DATA_DIR / "test_id.jsonl")]
    test_ood = [json.loads(l) for l in open(DATA_DIR / "test_ood.jsonl")]

    if args.max_samples:
        test_id = test_id[:args.max_samples]
        test_ood = test_ood[:args.max_samples]

    print(f"  Test ID: {len(test_id)} samples")
    print(f"  Test OOD: {len(test_ood)} samples")

    id_results = evaluate_split(model, processor, test_id, cfg,
                                split_name="ID (Singapore)", debug=args.debug)
    ood_results = evaluate_split(model, processor, test_ood, cfg,
                                 split_name="OOD (Boston)", debug=args.debug)

    print_results("ID Test (Singapore)", id_results)
    print_results("OOD Test (Boston)", ood_results)

    # Gaps
    print(f"\n  {'='*50}")
    print(f"  OOD Gaps:")
    if id_results.get("ade_mean") is not None and ood_results.get("ade_mean") is not None:
        ade_gap = ood_results["ade_mean"] - id_results["ade_mean"]
        fde_gap = ood_results["fde_mean"] - id_results["fde_mean"]
        print(f"    ADE Gap: {ade_gap:+.3f}m (OOD - ID)")
        print(f"    FDE Gap: {fde_gap:+.3f}m (OOD - ID)")
        print(f"    (Positive = OOD worse, expected)")
    else:
        ade_gap = None
        fde_gap = None
        print(f"    Cannot compute gaps (missing ADE/FDE)")

    if "derived_long_accuracy" in id_results and "derived_long_accuracy" in ood_results:
        long_gap = id_results["derived_long_accuracy"] - ood_results["derived_long_accuracy"]
        lat_gap = id_results["derived_lat_accuracy"] - ood_results["derived_lat_accuracy"]
        print(f"    Derived Long Acc Gap: {long_gap:+.4f}")
        print(f"    Derived Lat Acc Gap:  {lat_gap:+.4f}")
    print(f"  {'='*50}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "phase0_vl_eval.json"
    # Remove sample_outputs for clean JSON
    id_save = {k: v for k, v in id_results.items() if k != "sample_outputs"}
    ood_save = {k: v for k, v in ood_results.items() if k != "sample_outputs"}
    with open(results_path, "w") as f:
        json.dump({
            "id": id_save, "ood": ood_save,
            "ade_gap": ade_gap, "fde_gap": fde_gap,
        }, f, indent=2)
    print(f"\n[Eval] Results saved to {results_path}")

    print(f"\n{'=' * 70}")
    print(f"  Phase 0 (VL) v11 Trajectory Evaluation Complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

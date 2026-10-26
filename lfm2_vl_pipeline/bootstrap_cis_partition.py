"""
Bootstrap Confidence Intervals for CIS Partition Stability
===========================================================
Validates that the three-way layer partition (phi_align, phi_tune, phi_freeze)
is stable across bootstrap resamples of the CIS sample pairs.

Two modes:
  1. --from_raw: Load raw per-sample CIS values (if available) and resample
  2. --rerun: Re-run CIS with different random pair subsets (slower, more rigorous)

Usage:
  cd /mnt/vdb/LFM_project
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/bootstrap_cis_partition.py --from_raw
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/bootstrap_cis_partition.py --rerun --n_bootstrap 50
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import VLConfig, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, NUM_LAYERS, LAYER_TYPES


PHASE1_OUTPUT = OUTPUT_DIR / "phase1_vl_vg2"
BOOTSTRAP_OUTPUT = OUTPUT_DIR / "bootstrap_cis"


def partition_from_cis(cis_img_means, cis_txt_means, num_layers=NUM_LAYERS):
    """Reproduce the partition logic from the original CIS code."""
    img_thresh = float(np.median(cis_img_means))
    txt_thresh = float(np.median(cis_txt_means))

    phi_align, phi_tune, phi_freeze = [], [], []
    for l in range(num_layers):
        if cis_img_means[l] >= img_thresh and cis_txt_means[l] < txt_thresh:
            phi_align.append(l)
        elif cis_txt_means[l] >= txt_thresh:
            phi_tune.append(l)
        else:
            phi_freeze.append(l)

    if not phi_tune and phi_freeze:
        best_l = max(phi_freeze, key=lambda l: cis_txt_means[l])
        phi_freeze.remove(best_l)
        phi_tune.append(best_l)

    return tuple(sorted(phi_align)), tuple(sorted(phi_tune)), tuple(sorted(phi_freeze))


def bootstrap_from_raw(raw_data_path, n_bootstrap=1000, seed=42):
    """Bootstrap resample from saved raw per-sample CIS values."""
    with open(raw_data_path) as f:
        raw = json.load(f)

    rng = np.random.RandomState(seed)
    n_samples = len(raw["0"]["cis_img_raw"])

    print(f"  Raw data: {n_samples} samples per layer, {NUM_LAYERS} layers")
    print(f"  Running {n_bootstrap} bootstrap resamples...")

    partition_counts = Counter()
    layer_assignments = {l: {"align": 0, "tune": 0, "freeze": 0} for l in range(NUM_LAYERS)}
    all_cis_img_boots = {l: [] for l in range(NUM_LAYERS)}
    all_cis_txt_boots = {l: [] for l in range(NUM_LAYERS)}

    for b in tqdm(range(n_bootstrap), desc="  Bootstrap"):
        # Resample indices with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)

        cis_img_means = []
        cis_txt_means = []
        for l in range(NUM_LAYERS):
            img_vals = np.array(raw[str(l)]["cis_img_raw"])
            txt_vals = np.array(raw[str(l)]["cis_txt_raw"])
            cis_img_means.append(float(np.mean(img_vals[idx])))
            cis_txt_means.append(float(np.mean(txt_vals[idx])))
            all_cis_img_boots[l].append(cis_img_means[-1])
            all_cis_txt_boots[l].append(cis_txt_means[-1])

        align, tune, freeze = partition_from_cis(cis_img_means, cis_txt_means)
        partition_counts[(align, tune, freeze)] += 1

        for l in align:
            layer_assignments[l]["align"] += 1
        for l in tune:
            layer_assignments[l]["tune"] += 1
        for l in freeze:
            layer_assignments[l]["freeze"] += 1

    return partition_counts, layer_assignments, all_cis_img_boots, all_cis_txt_boots


def run_rerun_bootstrap(n_bootstrap=50, max_pairs=100, seed=42):
    """Re-run CIS with different random pair subsets."""
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from nuscenes.nuscenes import NuScenes
    from phase1_vl_counterfactual_vg2 import (
        prepare_input_3frame, generate_trajectory_fn, get_language_layers,
        get_image_text_masks, HiddenStateCollector, TokenLevelReplacer,
        trajectory_displacement, pair_samples, INSTRUCTION, TEMPORAL_PREFIX,
        TARGET_SIZE, IMAGE_TOKEN_ID,
    )
    from visual_grounding_v2_finetune import build_temporal_map

    cfg = VLConfig()
    model_dir = str(CHECKPOINT_DIR / "visual_grounding_v2" / "vg2_merged")

    print(f"  Loading model from {model_dir}...")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
    )
    model.eval()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    layers = get_language_layers(model)
    device = next(model.parameters()).device

    print("  Loading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    temporal_map = build_temporal_map(nusc, 'data/nuscenes')
    del nusc

    sg_records = [json.loads(l) for l in open(DATA_DIR / "test_ood.jsonl")]
    bs_records = [json.loads(l) for l in open(DATA_DIR / "test_id.jsonl")]
    sg_records = [r for r in sg_records if r.get("sample_token") in temporal_map]
    bs_records = [r for r in bs_records if r.get("sample_token") in temporal_map]

    rng = np.random.RandomState(seed)
    partition_counts = Counter()
    layer_assignments = {l: {"align": 0, "tune": 0, "freeze": 0} for l in range(NUM_LAYERS)}

    for b in range(n_bootstrap):
        print(f"\n  === Bootstrap {b+1}/{n_bootstrap} ===")
        # Random subset of pairs with different seed
        sub_seed = seed + b
        random.seed(sub_seed)
        np.random.seed(sub_seed)

        pairs = pair_samples(sg_records, bs_records, max_pairs=max_pairs, seed=sub_seed)

        from phase1_vl_counterfactual_vg2 import run_counterfactual_diagnosis
        results = run_counterfactual_diagnosis(
            model, processor, sg_records, bs_records, pairs, layers,
            temporal_map, cfg, max_pairs=max_pairs,
        )

        cis_img_means = [results[l]["cis_img_mean"] for l in range(NUM_LAYERS)]
        cis_txt_means = [results[l]["cis_txt_mean"] for l in range(NUM_LAYERS)]

        align, tune, freeze = partition_from_cis(cis_img_means, cis_txt_means)
        partition_counts[(align, tune, freeze)] += 1
        for l in align:
            layer_assignments[l]["align"] += 1
        for l in tune:
            layer_assignments[l]["tune"] += 1
        for l in freeze:
            layer_assignments[l]["freeze"] += 1

        print(f"  Partition: align={align}, tune={tune}, freeze={freeze}")

    return partition_counts, layer_assignments, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_raw", action="store_true",
                        help="Bootstrap from saved raw CIS values (fast)")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run CIS with different pair subsets (slow)")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    BOOTSTRAP_OUTPUT.mkdir(parents=True, exist_ok=True)

    raw_path = PHASE1_OUTPUT / "triple_diagnosis_raw.json"

    if args.from_raw:
        if not raw_path.exists():
            print(f"  Raw data not found at {raw_path}")
            print(f"  Please re-run CIS with --save_raw flag first.")
            print(f"  Falling back to parametric bootstrap from summary stats...")

            # Parametric bootstrap from saved mean/std
            with open(PHASE1_OUTPUT / "triple_diagnosis_results.json") as f:
                summary = json.load(f)

            rng = np.random.RandomState(args.seed)
            n_samples = summary["0"]["n_samples"]
            n_bootstrap = args.n_bootstrap

            print(f"  Parametric bootstrap: {n_samples} samples, {n_bootstrap} resamples")

            partition_counts = Counter()
            layer_assignments = {l: {"align": 0, "tune": 0, "freeze": 0} for l in range(NUM_LAYERS)}
            all_cis_img_boots = {l: [] for l in range(NUM_LAYERS)}
            all_cis_txt_boots = {l: [] for l in range(NUM_LAYERS)}

            for b in tqdm(range(n_bootstrap), desc="  Parametric Bootstrap"):
                cis_img_means = []
                cis_txt_means = []
                for l in range(NUM_LAYERS):
                    s = summary[str(l)]
                    # Generate synthetic samples from Normal(mean, std), then take mean of resample
                    # Bootstrap SE of mean ≈ std / sqrt(n)
                    img_boot = rng.normal(s["cis_img_mean"], s["cis_img_std"] / np.sqrt(n_samples))
                    txt_boot = rng.normal(s["cis_txt_mean"], s["cis_txt_std"] / np.sqrt(n_samples))
                    cis_img_means.append(max(0, img_boot))  # CIS >= 0
                    cis_txt_means.append(max(0, txt_boot))
                    all_cis_img_boots[l].append(cis_img_means[-1])
                    all_cis_txt_boots[l].append(cis_txt_means[-1])

                align, tune, freeze = partition_from_cis(cis_img_means, cis_txt_means)
                partition_counts[(align, tune, freeze)] += 1
                for l in align:
                    layer_assignments[l]["align"] += 1
                for l in tune:
                    layer_assignments[l]["tune"] += 1
                for l in freeze:
                    layer_assignments[l]["freeze"] += 1

        else:
            partition_counts, layer_assignments, all_cis_img_boots, all_cis_txt_boots = \
                bootstrap_from_raw(raw_path, args.n_bootstrap, args.seed)

    elif args.rerun:
        partition_counts, layer_assignments, all_cis_img_boots, all_cis_txt_boots = \
            run_rerun_bootstrap(args.n_bootstrap, seed=args.seed)
    else:
        print("Please specify --from_raw or --rerun")
        return

    # ── Report ──
    n_bootstrap = args.n_bootstrap
    print(f"\n{'='*70}")
    print(f"  BOOTSTRAP CIS PARTITION STABILITY ({n_bootstrap} resamples)")
    print(f"{'='*70}")

    # Most common partition
    print(f"\n  Top 5 partitions:")
    for (align, tune, freeze), count in partition_counts.most_common(5):
        pct = count / n_bootstrap * 100
        print(f"    {pct:5.1f}%  align={list(align)}, tune={list(tune)}, freeze={list(freeze)}")

    # Per-layer assignment probabilities
    print(f"\n  {'Layer':>5} {'Type':>4} {'P(align)':>9} {'P(tune)':>8} {'P(freeze)':>10} {'Dominant':>9}")
    print(f"  {'='*50}")
    for l in range(NUM_LAYERS):
        a = layer_assignments[l]
        p_align = a["align"] / n_bootstrap
        p_tune = a["tune"] / n_bootstrap
        p_freeze = a["freeze"] / n_bootstrap
        dominant = max(a, key=a.get)
        print(f"  {l:>5} {LAYER_TYPES.get(l,'?'):>4} {p_align:>9.3f} {p_tune:>8.3f} "
              f"{p_freeze:>10.3f} {dominant:>9}")

    # CIS 95% confidence intervals
    if all_cis_img_boots:
        print(f"\n  CIS 95% Confidence Intervals:")
        print(f"  {'Layer':>5} {'CIS_img mean':>12} {'[95% CI]':>20} {'CIS_txt mean':>12} {'[95% CI]':>20}")
        print(f"  {'='*75}")
        for l in range(NUM_LAYERS):
            img_vals = sorted(all_cis_img_boots[l])
            txt_vals = sorted(all_cis_txt_boots[l])
            img_lo = img_vals[int(0.025 * len(img_vals))]
            img_hi = img_vals[int(0.975 * len(img_vals))]
            txt_lo = txt_vals[int(0.025 * len(txt_vals))]
            txt_hi = txt_vals[int(0.975 * len(txt_vals))]
            img_mean = np.mean(img_vals)
            txt_mean = np.mean(txt_vals)
            print(f"  {l:>5} {img_mean:>12.4f} [{img_lo:>8.4f}, {img_hi:>8.4f}] "
                  f"{txt_mean:>12.4f} [{txt_lo:>8.4f}, {txt_hi:>8.4f}]")

    # Stability metric: % of bootstraps that agree with the original partition
    original_partition = ((0, 1), (2, 3, 4, 5, 6, 7, 8, 9), (10, 11, 12, 13, 14, 15))
    stability = partition_counts.get(original_partition, 0) / n_bootstrap * 100
    print(f"\n  Original partition stability: {stability:.1f}%")

    # Even if exact match is low, check if each layer's dominant assignment matches
    layer_stability = 0
    for l in range(NUM_LAYERS):
        a = layer_assignments[l]
        dominant = max(a, key=a.get)
        if l in [0, 1] and dominant == "align":
            layer_stability += 1
        elif l in range(2, 10) and dominant == "tune":
            layer_stability += 1
        elif l in range(10, 16) and dominant == "freeze":
            layer_stability += 1
    layer_stability_pct = layer_stability / NUM_LAYERS * 100
    print(f"  Per-layer dominant agreement: {layer_stability}/{NUM_LAYERS} ({layer_stability_pct:.0f}%)")

    # Save results
    results = {
        "n_bootstrap": n_bootstrap,
        "partition_stability_exact": stability,
        "layer_stability_pct": layer_stability_pct,
        "layer_assignments": {str(l): layer_assignments[l] for l in range(NUM_LAYERS)},
        "top_partitions": [
            {"align": list(a), "tune": list(t), "freeze": list(f), "count": c, "pct": round(c/n_bootstrap*100, 2)}
            for (a, t, f), c in partition_counts.most_common(10)
        ],
    }
    if all_cis_img_boots:
        results["cis_ci_95"] = {}
        for l in range(NUM_LAYERS):
            img_vals = sorted(all_cis_img_boots[l])
            txt_vals = sorted(all_cis_txt_boots[l])
            results["cis_ci_95"][str(l)] = {
                "cis_img": [round(img_vals[int(0.025*len(img_vals))], 4),
                            round(float(np.mean(img_vals)), 4),
                            round(img_vals[int(0.975*len(img_vals))], 4)],
                "cis_txt": [round(txt_vals[int(0.025*len(txt_vals))], 4),
                            round(float(np.mean(txt_vals)), 4),
                            round(txt_vals[int(0.975*len(txt_vals))], 4)],
            }

    out_path = BOOTSTRAP_OUTPUT / "bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()

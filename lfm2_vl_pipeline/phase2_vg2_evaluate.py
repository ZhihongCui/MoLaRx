"""
Phase 2 (VG v2) — Trajectory Evaluation
=========================================
Evaluates Phase 2 models on ID (Boston) and OOD (Singapore) test sets.
Computes ADE, FDE, and image ablation (blank images) for each model.

Models evaluated:
  1. TVG baseline (vg2_merged)
  2. CISR (cis_selective_final) — our method
  3. Full LoRA + MMD (full_lora_da_final) — ablation

Usage:
  cd /mnt/vdb/LFM_project
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/phase2_vg2_evaluate.py
  CUDA_VISIBLE_DEVICES=0 python lfm2_vl_pipeline/phase2_vg2_evaluate.py --max_samples 200
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import VLConfig, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR

# ── VG v2 format ──
INSTRUCTION = "Predict future trajectory as 6 waypoints (x,y) in ego-frame metres at t+0.5s to t+3.0s."
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) are shown. "
                   "Use visual motion cues across frames. ")
TARGET_SIZE = (512, 512)

# ── Model paths ──
MODELS = {
    "TVG_baseline": str(CHECKPOINT_DIR / "visual_grounding_v2" / "vg2_merged"),
    "CISR": str(OUTPUT_DIR / "phase2_vl_vg2" / "checkpoints" / "cis_selective_final"),
    "Full_LoRA_MMD": str(OUTPUT_DIR / "phase2_vl_vg2" / "checkpoints" / "full_lora_da_final"),
    "Full_LoRA_only": str(OUTPUT_DIR / "phase2_vl_vg2" / "checkpoints" / "full_lora_only_final"),
    "MMD_only": str(OUTPUT_DIR / "phase2_vl_vg2" / "checkpoints" / "mmd_only_final"),
    "Random_Partition": str(OUTPUT_DIR / "phase2_vl_vg2" / "checkpoints" / "random_partition_final"),
}


def parse_trajectory(text):
    """Parse (x,y) waypoints from model output."""
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
    """Compute ADE and FDE."""
    n = min(len(pred), len(gt))
    if n == 0:
        return None, None
    disps = [np.sqrt((pred[i][0]-gt[i][0])**2 + (pred[i][1]-gt[i][1])**2) for i in range(n)]
    return float(np.mean(disps)), float(disps[-1])


def get_temporal_frames(nusc, sample_token, dataroot):
    """Get (prev, curr, next) image paths for temporal input."""
    sample = nusc.get('sample', sample_token)
    curr_cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    curr_path = os.path.join(dataroot, curr_cam['filename'])

    prev_path = curr_path  # fallback
    if sample['prev']:
        prev_sample = nusc.get('sample', sample['prev'])
        prev_cam = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])
        prev_path = os.path.join(dataroot, prev_cam['filename'])

    next_path = curr_path  # fallback
    if sample['next']:
        next_sample = nusc.get('sample', sample['next'])
        next_cam = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])
        next_path = os.path.join(dataroot, next_cam['filename'])

    return prev_path, curr_path, next_path


@torch.no_grad()
def evaluate_model_on_split(model, processor, records, nusc, cfg,
                            split_name="test", use_blank=False):
    """Evaluate a model on a split. If use_blank=True, replace images with gray."""
    model.eval()
    ade_list, fde_list = [], []
    parse_failures = 0
    blank_image = Image.new("RGB", TARGET_SIZE, color=(128, 128, 128))

    for rec in tqdm(records, desc=f"  {split_name}{'(blank)' if use_blank else ''}"):
        token = rec.get("sample_token")
        if not token:
            continue

        try:
            prev_path, curr_path, next_path = get_temporal_frames(nusc, token, 'data/nuscenes')
        except Exception:
            continue

        if use_blank:
            images = [blank_image, blank_image, blank_image]
        else:
            images = []
            for p in [prev_path, curr_path, next_path]:
                try:
                    images.append(Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS))
                except Exception:
                    images.append(blank_image)

        question = TEMPORAL_PREFIX + INSTRUCTION
        conv = [{"role": "user", "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "image", "image": images[2]},
            {"type": "text", "text": question},
        ]}]

        inputs = processor.apply_chat_template(
            conv, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, tokenize=True,
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens,
            do_sample=False, pad_token_id=processor.tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        pred_text = processor.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        pred = parse_trajectory(pred_text)
        gt = parse_trajectory(rec.get("answer", ""))

        if pred is None or gt is None or len(pred) < 3 or len(gt) < 3:
            parse_failures += 1
            continue

        ade, fde = compute_ade_fde(pred, gt)
        if ade is not None:
            ade_list.append(ade)
            fde_list.append(fde)

    return {
        "n_total": len(records),
        "n_evaluated": len(ade_list),
        "parse_failures": parse_failures,
        "ade_mean": round(float(np.mean(ade_list)), 4) if ade_list else None,
        "ade_median": round(float(np.median(ade_list)), 4) if ade_list else None,
        "ade_std": round(float(np.std(ade_list)), 4) if ade_list else None,
        "fde_mean": round(float(np.mean(fde_list)), 4) if fde_list else None,
        "fde_median": round(float(np.median(fde_list)), 4) if fde_list else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split (for quick testing)")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip image ablation (saves time)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODELS.keys()),
                        help="Which models to evaluate (default: all)")
    args = parser.parse_args()

    cfg = VLConfig()
    models_to_eval = args.models or list(MODELS.keys())

    print("=" * 70)
    print("  PHASE 2 (VG v2) — TRAJECTORY EVALUATION")
    print("=" * 70)

    # Load nuScenes
    print("\nLoading nuScenes...")
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)

    # Load test data
    id_records = [json.loads(l) for l in open(DATA_DIR / "test_id.jsonl")]
    ood_records = [json.loads(l) for l in open(DATA_DIR / "test_ood.jsonl")]

    if args.max_samples:
        rng = np.random.RandomState(42)
        if len(id_records) > args.max_samples:
            idx = rng.choice(len(id_records), args.max_samples, replace=False)
            id_records = [id_records[i] for i in sorted(idx)]
        if len(ood_records) > args.max_samples:
            idx = rng.choice(len(ood_records), args.max_samples, replace=False)
            ood_records = [ood_records[i] for i in sorted(idx)]

    print(f"  ID samples: {len(id_records)}")
    print(f"  OOD samples: {len(ood_records)}")

    all_results = {}

    for model_name in models_to_eval:
        model_dir = MODELS[model_name]
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}")
        print(f"  Path: {model_dir}")
        print(f"{'='*70}")

        if not os.path.isdir(model_dir):
            print(f"  SKIPPED — model not found at {model_dir}")
            all_results[model_name] = {"error": "not found"}
            continue

        # Load model
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        print(f"  Loaded in {time.time()-t0:.1f}s, device={model.device}")

        # Normal evaluation
        id_res = evaluate_model_on_split(model, processor, id_records, nusc, cfg, "ID")
        ood_res = evaluate_model_on_split(model, processor, ood_records, nusc, cfg, "OOD")

        model_results = {"id": id_res, "ood": ood_res}

        # Image ablation
        if not args.skip_ablation:
            id_blank = evaluate_model_on_split(model, processor, id_records, nusc, cfg,
                                               "ID", use_blank=True)
            ood_blank = evaluate_model_on_split(model, processor, ood_records, nusc, cfg,
                                                "OOD", use_blank=True)
            model_results["id_blank"] = id_blank
            model_results["ood_blank"] = ood_blank

        all_results[model_name] = model_results

        # Print results
        print(f"\n  {model_name} Results:")
        print(f"    ID:  ADE={id_res['ade_mean']:.3f}m  FDE={id_res['fde_mean']:.3f}m  "
              f"(n={id_res['n_evaluated']})")
        print(f"    OOD: ADE={ood_res['ade_mean']:.3f}m  FDE={ood_res['fde_mean']:.3f}m  "
              f"(n={ood_res['n_evaluated']})")
        if id_res['ade_mean'] and ood_res['ade_mean']:
            gap = (ood_res['ade_mean'] - id_res['ade_mean']) / id_res['ade_mean'] * 100
            print(f"    OOD Gap: {gap:+.1f}%")

        if not args.skip_ablation and "id_blank" in model_results:
            ib = model_results["id_blank"]
            print(f"    ID (blank img):  ADE={ib['ade_mean']:.3f}m")
            if id_res['ade_mean'] and ib['ade_mean']:
                img_dep = (ib['ade_mean'] - id_res['ade_mean']) / id_res['ade_mean'] * 100
                print(f"    Image dependence: {img_dep:+.1f}%")

        # Free GPU
        del model, processor
        torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'='*78}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*78}")
    header = f"  {'Model':>18} {'ID ADE':>8} {'OOD ADE':>9} {'Gap%':>7} {'FDE_ID':>8} {'FDE_OOD':>9}"
    if not args.skip_ablation:
        header += f" {'ImgDep%':>8}"
    print(header)
    print(f"  {'─'*72}")

    for model_name in models_to_eval:
        r = all_results.get(model_name, {})
        if "error" in r:
            print(f"  {model_name:>18}   — (not found)")
            continue
        id_ade = r["id"]["ade_mean"]
        ood_ade = r["ood"]["ade_mean"]
        id_fde = r["id"]["fde_mean"]
        ood_fde = r["ood"]["fde_mean"]
        gap = (ood_ade - id_ade) / id_ade * 100 if id_ade else 0

        line = f"  {model_name:>18} {id_ade:>8.3f} {ood_ade:>9.3f} {gap:>+7.1f} {id_fde:>8.3f} {ood_fde:>9.3f}"

        if not args.skip_ablation and "id_blank" in r:
            blank_ade = r["id_blank"]["ade_mean"]
            img_dep = (blank_ade - id_ade) / id_ade * 100 if id_ade and blank_ade else 0
            line += f" {img_dep:>+8.1f}"

        print(line)

    print(f"{'='*78}")

    # Save results
    out_dir = OUTPUT_DIR / "phase2_vl_vg2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_vg2_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

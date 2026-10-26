"""
Phase 0 (VL) v11: Trajectory Planning Evaluation
==================================================
Evaluates ADE/FDE on ID (Singapore) and OOD (Boston).
Also derives classification accuracy from predicted trajectories.
"""
import json, os, sys, re, torch, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_vl import VLConfig, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR

def parse_coords(text):
    """Parse coordinate string like '(-0.02,4.16) (-0.03,8.32) ...' into list of (x,y)."""
    pairs = re.findall(r'\(([^)]+)\)', text)
    coords = []
    for p in pairs:
        parts = p.split(',')
        if len(parts) == 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return coords

def compute_ade_fde(pred_coords, gt_coords):
    """Compute ADE and FDE between predicted and ground truth trajectories."""
    n = min(len(pred_coords), len(gt_coords))
    if n == 0:
        return None, None
    displacements = []
    for i in range(n):
        dx = pred_coords[i][0] - gt_coords[i][0]
        dy = pred_coords[i][1] - gt_coords[i][1]
        displacements.append(np.sqrt(dx**2 + dy**2))
    ade = np.mean(displacements)
    fde = displacements[-1]
    return float(ade), float(fde)

def traj_to_classification(coords):
    """Derive longitudinal and lateral classification from trajectory."""
    if len(coords) < 2:
        return -1, -1
    # Longitudinal: compare speed in first half vs second half
    mid = len(coords) // 2
    y_vals = [c[1] for c in coords]
    dy_first = y_vals[mid] - y_vals[0] if mid > 0 else 0
    dy_second = y_vals[-1] - y_vals[mid] if len(y_vals) > mid else 0
    speed_change = dy_second - dy_first
    if speed_change > 0.5:
        long_d = 0  # accelerate
    elif speed_change < -0.5:
        long_d = 2  # decelerate
    else:
        long_d = 1  # maintain
    # Lateral: final x displacement
    x_final = coords[-1][0]
    if x_final < -0.5:
        lat_d = 0  # left
    elif x_final > 0.5:
        lat_d = 2  # right
    else:
        lat_d = 1  # straight
    return long_d, lat_d

@torch.no_grad()
def evaluate_split(model, processor, records, cfg, split_name="test"):
    model.eval()
    ade_list, fde_list = [], []
    parse_success, parse_fail = 0, 0
    # For derived classification
    long_correct, long_total = 0, 0
    lat_correct, lat_total = 0, 0

    for rec in tqdm(records, desc=f"  {split_name}"):
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except:
            image = Image.new("RGB", (640, 480), (128, 128, 128))
        conv = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": rec["question"]}]}]
        inputs = processor.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt", return_dict=True, tokenize=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        pred_text = processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

        pred_coords = parse_coords(pred_text)
        gt_coords = parse_coords(rec["answer"])

        if len(pred_coords) >= 3 and len(gt_coords) >= 3:
            ade, fde = compute_ade_fde(pred_coords, gt_coords)
            if ade is not None:
                ade_list.append(ade)
                fde_list.append(fde)
                parse_success += 1
        else:
            parse_fail += 1

        # Derived classification
        if len(pred_coords) >= 2 and len(gt_coords) >= 2:
            pred_long, pred_lat = traj_to_classification(pred_coords)
            gt_long, gt_lat = traj_to_classification(gt_coords)
            if pred_long >= 0 and gt_long >= 0:
                long_total += 1
                if pred_long == gt_long:
                    long_correct += 1
            if pred_lat >= 0 and gt_lat >= 0:
                lat_total += 1
                if pred_lat == gt_lat:
                    lat_correct += 1

    results = {
        "ade_mean": round(float(np.mean(ade_list)), 4) if ade_list else None,
        "ade_median": round(float(np.median(ade_list)), 4) if ade_list else None,
        "ade_std": round(float(np.std(ade_list)), 4) if ade_list else None,
        "fde_mean": round(float(np.mean(fde_list)), 4) if fde_list else None,
        "fde_median": round(float(np.median(fde_list)), 4) if fde_list else None,
        "parse_success": parse_success,
        "parse_fail": parse_fail,
        "parse_rate": round(parse_success / max(parse_success + parse_fail, 1), 4),
        "total": len(records),
        "derived_long_acc": round(long_correct / max(long_total, 1), 4),
        "derived_lat_acc": round(lat_correct / max(lat_total, 1), 4),
    }
    return results

def main():
    cfg = VLConfig()
    model_dir = str(CHECKPOINT_DIR / "phase0_vl_merged")
    print(f"\n{'='*70}")
    print(f"  PHASE 0 (VL) v11: Trajectory Evaluation (ADE/FDE)")
    print(f"{'='*70}")
    print(f"  Model: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"  Device: {model.device}")

    test_id = [json.loads(l) for l in open(DATA_DIR / "test_id.jsonl")]
    test_ood = [json.loads(l) for l in open(DATA_DIR / "test_ood.jsonl")]
    print(f"  Test ID: {len(test_id)} samples")
    print(f"  Test OOD: {len(test_ood)} samples")

    id_res = evaluate_split(model, processor, test_id, cfg, "ID (Singapore)")
    ood_res = evaluate_split(model, processor, test_ood, cfg, "OOD (Boston)")

    print(f"\n{'='*50}")
    print(f"  ID (Singapore):")
    print(f"    ADE: {id_res['ade_mean']:.2f}m (median={id_res['ade_median']:.2f}, std={id_res['ade_std']:.2f})")
    print(f"    FDE: {id_res['fde_mean']:.2f}m")
    print(f"    Parse rate: {id_res['parse_rate']*100:.1f}%")
    print(f"    Derived long acc: {id_res['derived_long_acc']*100:.1f}%")
    print(f"    Derived lat acc: {id_res['derived_lat_acc']*100:.1f}%")

    print(f"\n  OOD (Boston):")
    print(f"    ADE: {ood_res['ade_mean']:.2f}m (median={ood_res['ade_median']:.2f}, std={ood_res['ade_std']:.2f})")
    print(f"    FDE: {ood_res['fde_mean']:.2f}m")
    print(f"    Parse rate: {ood_res['parse_rate']*100:.1f}%")
    print(f"    Derived long acc: {ood_res['derived_long_acc']*100:.1f}%")
    print(f"    Derived lat acc: {ood_res['derived_lat_acc']*100:.1f}%")

    ade_gap = ood_res["ade_mean"] - id_res["ade_mean"] if id_res["ade_mean"] and ood_res["ade_mean"] else None
    fde_gap = ood_res["fde_mean"] - id_res["fde_mean"] if id_res["fde_mean"] and ood_res["fde_mean"] else None

    print(f"\n  OOD Gaps:")
    if ade_gap is not None:
        print(f"    ADE Gap: {ade_gap:+.2f}m (positive = OOD worse)")
        print(f"    FDE Gap: {fde_gap:+.2f}m")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "phase0_v11_traj_eval.json", "w") as f:
        json.dump({"id": id_res, "ood": ood_res, "ade_gap": ade_gap, "fde_gap": fde_gap}, f, indent=2)
    print(f"\n  Saved to {OUTPUT_DIR / 'phase0_v11_traj_eval.json'}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

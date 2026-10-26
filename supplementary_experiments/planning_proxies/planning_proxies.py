"""
Planning-oriented proxy metrics on the cross-city nuScenes splits
(paper App. C.5): lateral error at 3 s and predicted-trajectory
jerk, computed from per-sample predictions of the forward-direction models.

Evaluates the same first-300 records per split as the paper protocol
(Table 6: 300 eval samples/split), saving predictions for offline metrics.

Usage (cwd = .):
  CUDA_VISIBLE_DEVICES=1 python planning_proxies.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from nuscenes.nuscenes import NuScenes

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "lfm2_vl_pipeline"))
from phase2_vg2_evaluate import (INSTRUCTION, TEMPORAL_PREFIX, TARGET_SIZE,
                                 MODELS, parse_trajectory, get_temporal_frames)

DATA = ROOT / "lfm2_vl_pipeline/outputs/data"
OUT_PRED = ROOT / "lfm2_vl_pipeline/outputs/planning_proxies_preds.json"
OUT_JSON = ROOT / "lfm2_vl_pipeline/outputs/planning_proxies.json"
N = 300
DT = 0.5
EVAL_MODELS = ["TVG_baseline", "CISR", "Full_LoRA_only", "Full_LoRA_MMD"]


def to_wp6(x):
    """Coerce parse output (list/np/None) to a (6,2) float array or None."""
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        return None
    if a.shape[0] < 6:
        a = np.vstack([a, np.repeat(a[-1:], 6 - a.shape[0], axis=0)])
    return a[:6]


def parse_answer(ans):
    return to_wp6(parse_trajectory(ans))


def jerk_mag(wp):
    """Mean |jerk| (m/s^3) from 6 waypoints at 0.5 s spacing (incl. origin as t=0)."""
    pts = np.vstack([[0.0, 0.0], wp])              # t = 0 .. 3.0, 7 points
    v = np.diff(pts, axis=0) / DT                  # 6
    a = np.diff(v, axis=0) / DT                    # 5
    j = np.diff(a, axis=0) / DT                    # 4
    return float(np.linalg.norm(j, axis=1).mean())


def main():
    splits = {}
    for name in ["test_id", "test_ood"]:
        recs = [json.loads(l) for l in open(DATA / f"{name}.jsonl")][:N]
        splits[name] = recs
    print({k: len(v) for k, v in splits.items()})

    print("Loading nuScenes DB (temporal frames)...")
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes", verbose=False)

    question = TEMPORAL_PREFIX + INSTRUCTION
    blank = Image.new("RGB", TARGET_SIZE, color=(128, 128, 128))
    all_preds, results = {}, {}

    for mname in EVAL_MODELS:
        mpath = Path(MODELS[mname])
        if not (ROOT / mpath / "config.json").exists() and not (mpath / "config.json").exists():
            print("!! missing ckpt", mname, mpath)
            continue
        print(f"\n== {mname} ==")
        processor = AutoProcessor.from_pretrained(str(mpath))
        model = AutoModelForImageTextToText.from_pretrained(
            str(mpath), dtype=torch.bfloat16).to("cuda").eval()

        all_preds[mname] = {}
        results[mname] = {}
        for sname, recs in splits.items():
            preds = {}
            lat3, ade, jerks, gt_jerks, fails = [], [], [], [], 0
            with torch.no_grad():
                for rec in tqdm(recs, desc=f"{mname}:{sname}"):
                    tok = rec.get("sample_token")
                    try:
                        p_prev, p_cur, p_next = get_temporal_frames(nusc, tok, "data/nuscenes")
                        images = [Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                                  for p in (p_prev, p_cur, p_next)]
                    except Exception:
                        images = [blank, blank, blank]
                    conv = [{"role": "user", "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[1]},
                        {"type": "image", "image": images[2]},
                        {"type": "text", "text": question}]}]
                    inputs = processor.apply_chat_template(
                        conv, add_generation_prompt=True,
                        return_tensors="pt", return_dict=True, tokenize=True)
                    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                              for k, v in inputs.items()}
                    out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                         pad_token_id=processor.tokenizer.pad_token_id)
                    text = processor.tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred = to_wp6(parse_trajectory(text))
                    gt = parse_answer(rec["answer"])
                    if pred is None or gt is None:
                        fails += 1
                        continue
                    preds[tok] = pred.tolist()
                    d = np.linalg.norm(pred - gt, axis=1)
                    ade.append(float(d.mean()))
                    lat3.append(abs(float(pred[5][0]) - float(gt[5][0])))
                    jerks.append(jerk_mag(pred))
                    gt_jerks.append(jerk_mag(gt))
            all_preds[mname][sname] = preds
            results[mname][sname] = {
                "ADE": round(float(np.mean(ade)), 4),
                "lateral_err_at_3s_m": round(float(np.mean(lat3)), 4),
                "pred_jerk_m_s3": round(float(np.mean(jerks)), 4),
                "gt_jerk_m_s3": round(float(np.mean(gt_jerks)), 4),
                "n": len(ade), "parse_failures": fails,
            }
            print(sname, json.dumps(results[mname][sname]))
        del model
        torch.cuda.empty_cache()

    json.dump(all_preds, open(OUT_PRED, "w"))
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\nsaved ->", OUT_JSON)


if __name__ == "__main__":
    main()

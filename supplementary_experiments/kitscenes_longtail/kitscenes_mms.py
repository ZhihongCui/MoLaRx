"""
KITScenes MMS-inspired closest-mode accuracy (paper App. C.5).

Stage --prepare-neg: augment probe_val.jsonl with the dataset's annotated
failure-mode trajectories (wrong_speed, neglect_instruction), resampled to
the same 6-waypoint/0.5 s grid in the same reference frame as the expert.

Stage --eval: re-evaluate the four checkpoints, SAVING per-scenario
predictions, then compute
  - ADE/FDE vs expert (sanity, should match probe_results.json)
  - closest-mode accuracy: fraction of scenarios where
        ADE(pred, expert) < min_k ADE(pred, negative_k)
  - mean mode margin: min_k ADE(pred, neg_k) - ADE(pred, expert)

Usage (cwd = .):
  python kitscenes_mms.py --prepare-neg
  CUDA_VISIBLE_DEVICES=1 python kitscenes_mms.py --eval
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import kitscenes_probe as kp  # reuse timeline/transform/parse helpers

ROOT = Path(".")
KIT = ROOT / "data/kitscenes"
MMS_JSONL = KIT / "probe_val_mms.jsonl"
PREDS_JSON = KIT / "mms_predictions.json"
OUT_JSON = KIT / "mms_results.json"

NEG_KEYS = ["wrong_speed", "neglect_instruction"]


def valid_traj(t):
    return t is not None and len(t) >= 15 and not any(p[0] <= -99 for p in t[:3])


def resample_neg(tr, neg, ref_idx=kp.REF_IDX):
    """Same transform as the expert: interpolate at t_ref + horizons, ref frame at t_ref."""
    past = np.asarray(tr["past"], dtype=float)
    fut = np.asarray(neg, dtype=float)
    t_past = np.arange(-(len(past) - 1), 1) * kp.DT
    t_fut = np.arange(1, len(fut) + 1) * kp.DT
    times = np.concatenate([t_past, t_fut])
    pts = np.vstack([past, fut])

    t_ref = -(20 - ref_idx) * kp.DT
    origin = past[ref_idx]
    d = past[ref_idx + 1] - past[ref_idx - 1]
    if np.linalg.norm(d) < 0.3:
        d = past[20] - past[max(0, ref_idx - 4)]
    if np.linalg.norm(d) < 1e-3:
        d = np.array([1.0, 0.0])
    th = math.atan2(d[1], d[0])
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, s], [-s, c]])
    wps = []
    for h in kp.HORIZONS:
        p = kp.interp_xy(times, pts, t_ref + h)
        local = R @ (p - origin)
        wps.append((round(float(local[1]), 2), round(float(local[0]), 2)))
    return wps


def prepare_neg():
    import pyarrow.parquet as pq
    base = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(kp.PROBE_JSONL)}
    pf = pq.ParquetFile(kp.PARQUET)
    n_neg = 0
    with open(MMS_JSONL, "w") as fout:
        for batch in pf.iter_batches(batch_size=8, columns=["scenario_id", "trajectory"]):
            for row in batch.to_pylist():
                sid = row["scenario_id"]
                if sid not in base:
                    continue
                rec = base[sid]
                tr = row["trajectory"]
                negs = {}
                for k in NEG_KEYS:
                    if valid_traj(tr.get(k)):
                        negs[k] = resample_neg(tr, tr[k])
                rec["negatives"] = negs
                if negs:
                    n_neg += 1
                fout.write(json.dumps(rec) + "\n")
    print(f"wrote {MMS_JSONL}; scenarios with >=1 valid negative: {n_neg}/{len(base)}")


def evaluate():
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from tqdm import tqdm

    records = [json.loads(l) for l in open(MMS_JSONL)]
    question = kp.TEMPORAL_PREFIX + kp.INSTRUCTION
    all_preds, results = {}, {}

    for name, ckpt in kp.MODELS.items():
        ckpt_path = ROOT / ckpt
        if not (ckpt_path / "config.json").exists():
            print("!! missing", name)
            continue
        print(f"\n== {name} ==")
        processor = AutoProcessor.from_pretrained(str(ckpt_path))
        model = AutoModelForImageTextToText.from_pretrained(
            str(ckpt_path), dtype=torch.bfloat16).to("cuda").eval()

        preds = {}
        ade_all, fde_all, correct, margins, n_mode = [], [], 0, [], 0
        with torch.no_grad():
            for rec in tqdm(records):
                images = [Image.open(ROOT / p).convert("RGB").resize(
                    kp.TARGET_SIZE, Image.LANCZOS) for p in rec["frames"]]
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
                pred = kp.parse_trajectory(text)
                if pred is None:
                    continue
                preds[rec["scenario_id"]] = pred.tolist()
                gt = np.asarray(rec["answer_waypoints"])
                d = np.linalg.norm(pred - gt, axis=1)
                ade_all.append(float(d.mean()))
                fde_all.append(float(d[-1]))
                if rec.get("negatives"):
                    ade_exp = float(d.mean())
                    ade_negs = [float(np.linalg.norm(pred - np.asarray(w), axis=1).mean())
                                for w in rec["negatives"].values()]
                    n_mode += 1
                    margins.append(min(ade_negs) - ade_exp)
                    if ade_exp < min(ade_negs):
                        correct += 1

        results[name] = {
            "ADE": round(float(np.mean(ade_all)), 4),
            "FDE": round(float(np.mean(fde_all)), 4),
            "closest_mode_acc": round(correct / n_mode, 4) if n_mode else None,
            "mean_mode_margin_m": round(float(np.mean(margins)), 4) if margins else None,
            "n_mode_scenarios": n_mode,
        }
        all_preds[name] = preds
        print(json.dumps(results[name], indent=2))
        del model
        torch.cuda.empty_cache()

    json.dump(all_preds, open(PREDS_JSON, "w"))
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print("\nsaved ->", OUT_JSON)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-neg", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.prepare_neg:
        prepare_neg()
    if a.eval:
        evaluate()

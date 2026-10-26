"""
KITScenes-LongTail zero-shot probe (paper App. C.5).

Stage 1 (--prepare): convert HF val parquet (100 long-tail scenarios, 5 Hz)
into the VG2 probe format:
  - 3 front-camera frames at indices 15/18/20  (t = -1.0 / -0.4 / 0.0 s),
    i.e. the model's (t-0.5s, t, t+0.5s) triplet with <=0.1 s jitter,
    reference ("now") = idx 18.
  - GT: expert_like trajectory linearly interpolated at t_ref + {0.5..3.0 s},
    transformed into the ego frame at t_ref, mapped to the pipeline's
    (dx=left+, dy=forward+) convention.

Stage 2 (--eval): zero-shot evaluation of forward-direction checkpoints
(vg2_merged / cis_selective / full_lora_only / full_lora_da) with the exact
VG2 prompt; reports ADE/FDE overall and per scenario_type.

Usage (cwd = .):
  python kitscenes_probe.py --prepare
  CUDA_VISIBLE_DEVICES=1 python kitscenes_probe.py --eval
"""

import argparse
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")
KIT = ROOT / "data/kitscenes"
PARQUET = KIT / "data/val-00000-of-00001.parquet"
FRAMES_DIR = KIT / "frames"
PROBE_JSONL = KIT / "probe_val.jsonl"
OUT_JSON = KIT / "probe_results.json"

DT = 0.2                # 5 Hz
FRAME_IDX = (15, 18, 20)
REF_IDX = 18            # "now" for the model
HORIZONS = [0.5 * k for k in range(1, 7)]   # +0.5 .. +3.0 s

INSTRUCTION = ("Predict future trajectory as 6 waypoints (x,y) in ego-frame "
               "metres at t+0.5s to t+3.0s.")
TEMPORAL_PREFIX = ("Three consecutive driving camera frames (t-0.5s, t, t+0.5s) "
                   "are shown. Use visual motion cues across frames. ")
TARGET_SIZE = (512, 512)

MODELS = {
    "Source_only": "lfm2_vl_pipeline/outputs/checkpoints/visual_grounding_v2/vg2_merged",
    "MoLaRx": "lfm2_vl_pipeline/outputs/phase2_vl_vg2/checkpoints/cis_selective_final",
    "Full_LoRA": "lfm2_vl_pipeline/outputs/phase2_vl_vg2/checkpoints/full_lora_only_final",
    "Full_LoRA_MMD": "lfm2_vl_pipeline/outputs/phase2_vl_vg2/checkpoints/full_lora_da_final",
}


# ──────────────────────────────────────────────────────────────
# Stage 1: prepare
# ──────────────────────────────────────────────────────────────

def build_timeline(tr):
    """Return (times, points) over past+future on the 0.2 s grid, t=0 at past[-1]."""
    past = np.asarray(tr["past"], dtype=float)            # (21, 2), t = -4.0..0.0
    fut = np.asarray(tr["expert_like"], dtype=float)      # (25, 2), t = +0.2..+5.0
    t_past = np.arange(-(len(past) - 1), 1) * DT
    t_fut = np.arange(1, len(fut) + 1) * DT
    return np.concatenate([t_past, t_fut]), np.vstack([past, fut])


def interp_xy(times, pts, t):
    x = np.interp(t, times, pts[:, 0])
    y = np.interp(t, times, pts[:, 1])
    return np.array([x, y])


def prepare():
    import pyarrow.parquet as pq
    from PIL import Image

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(PARQUET)
    n_written = 0
    sanity = []

    with open(PROBE_JSONL, "w") as fout:
        for batch in pf.iter_batches(batch_size=4, columns=[
                "scenario_id", "scenario_type", "driving_instruction",
                "frames_camera_front", "trajectory"]):
            for row in batch.to_pylist():
                sid = row["scenario_id"]
                frames = row["frames_camera_front"]
                tr = row["trajectory"]
                if len(frames) < 21 or len(tr["past"]) < 21 or len(tr["expert_like"]) < 15:
                    print(f"skip {sid}: frames={len(frames)} past={len(tr['past'])}")
                    continue

                times, pts = build_timeline(tr)
                t_ref = -(20 - REF_IDX) * DT                     # -0.4 s
                origin = np.asarray(tr["past"][REF_IDX], dtype=float)

                # heading from centred diff (fallback: wider window at low speed)
                d = np.asarray(tr["past"][REF_IDX + 1]) - np.asarray(tr["past"][REF_IDX - 1])
                if np.linalg.norm(d) < 0.3:
                    d = np.asarray(tr["past"][20]) - np.asarray(tr["past"][max(0, REF_IDX - 4)])
                if np.linalg.norm(d) < 1e-3:
                    d = np.array([1.0, 0.0])                     # stationary: assume x-forward
                theta = math.atan2(d[1], d[0])
                c, s = math.cos(theta), math.sin(theta)
                # rows: forward, left (KITScenes frame rotated to ref heading)
                R = np.array([[c, s], [-s, c]])

                wps = []
                for h in HORIZONS:
                    p = interp_xy(times, pts, t_ref + h)
                    local = R @ (p - origin)                     # (fwd, left)
                    wps.append((round(float(local[1]), 2),       # dx = left+
                                round(float(local[0]), 2)))      # dy = forward+

                speed = float(np.linalg.norm(d) / (2 * DT))
                paths = []
                for k, fi in enumerate(FRAME_IDX):
                    img = Image.open(io.BytesIO(frames[fi]["bytes"])).convert("RGB")
                    p = FRAMES_DIR / f"{sid}_f{fi}.jpg"
                    img.save(p, quality=92)
                    paths.append(str(p.relative_to(ROOT)))

                rec = {
                    "scenario_id": sid,
                    "scenario_type": row["scenario_type"],
                    "driving_instruction": row["driving_instruction"],
                    "frames": paths,
                    "answer_waypoints": wps,
                    "ego_speed_est": round(speed, 2),
                }
                fout.write(json.dumps(rec) + "\n")
                n_written += 1

                instr = (row["driving_instruction"] or "").lower()
                if "left" in instr or "right" in instr:
                    sanity.append((sid, instr, round(float(np.mean([w[0] for w in wps])), 2)))

    print(f"wrote {n_written} records -> {PROBE_JSONL}")
    print("sanity (turn scenarios; dx=left+ so 'left' should trend positive):")
    for sid, instr, mdx in sanity:
        print(f"  {sid}  {instr:<22} mean_dx={mdx:+.2f}")


# ──────────────────────────────────────────────────────────────
# Stage 2: eval
# ──────────────────────────────────────────────────────────────

def parse_trajectory(text):
    pattern = r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?'
    matches = re.findall(pattern, text.strip())
    if not matches:
        return None
    wp = [(float(a), float(b)) for a, b in matches[:6]]
    while len(wp) < 6:
        wp.append(wp[-1])
    return np.asarray(wp)


def evaluate():
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from tqdm import tqdm

    records = [json.loads(l) for l in open(PROBE_JSONL)]
    print(f"{len(records)} probe records")
    results = {}

    for name, ckpt in MODELS.items():
        ckpt_path = ROOT / ckpt
        if not (ckpt_path / "config.json").exists():
            print(f"!! missing checkpoint {name}: {ckpt_path}")
            continue
        print(f"\n== {name} ({ckpt}) ==")
        processor = AutoProcessor.from_pretrained(str(ckpt_path))
        model = AutoModelForImageTextToText.from_pretrained(
            str(ckpt_path), dtype=torch.bfloat16).to("cuda").eval()

        ade_all, fde_all = [], []
        by_type = defaultdict(lambda: {"ade": [], "fde": []})
        fails = 0
        question = TEMPORAL_PREFIX + INSTRUCTION

        with torch.no_grad():
            for rec in tqdm(records):
                images = [Image.open(ROOT / p).convert("RGB").resize(
                    TARGET_SIZE, Image.LANCZOS) for p in rec["frames"]]
                conv = [{"role": "user", "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": question},
                ]}]
                inputs = processor.apply_chat_template(
                    conv, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True, tokenize=True)
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
                out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
                text = processor.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = parse_trajectory(text)
                if pred is None:
                    fails += 1
                    continue
                gt = np.asarray(rec["answer_waypoints"])
                d = np.linalg.norm(pred - gt, axis=1)
                ade_all.append(float(d.mean()))
                fde_all.append(float(d[-1]))
                bt = by_type[rec["scenario_type"]]
                bt["ade"].append(float(d.mean()))
                bt["fde"].append(float(d[-1]))

        summary = {
            "ADE": round(float(np.mean(ade_all)), 4),
            "FDE": round(float(np.mean(fde_all)), 4),
            "n": len(ade_all), "parse_failures": fails,
            "by_scenario_type": {
                k: {"ADE": round(float(np.mean(v["ade"])), 4),
                    "FDE": round(float(np.mean(v["fde"])), 4),
                    "n": len(v["ade"])}
                for k, v in sorted(by_type.items())},
        }
        results[name] = summary
        print(json.dumps(summary, indent=2))
        del model
        torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    if args.eval:
        evaluate()
    if not (args.prepare or args.eval):
        ap.error("pass --prepare and/or --eval")

"""Evaluate AV2 reverse-direction (MIA->PIT) checkpoints on the reversed splits.

AV2 records carry their own 3-frame image_paths and VG2 question, so no
temporal map is needed. Reports ADE/FDE on the first 300 records of
test_id (Miami) and test_ood (Pittsburgh) per model.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "lfm2_vl_pipeline"))
from phase2_vg2_evaluate import parse_trajectory, TARGET_SIZE

DATA = ROOT / "argoverse2_pipeline_rev/outputs/data"
PH2 = ROOT / "argoverse2_pipeline_rev/outputs/phase2_cisr"
OUT = ROOT / "argoverse2_pipeline_rev/outputs/av2_rev_eval.json"
N = 300

MODELS = {
    "Source_only": ROOT / "lfm2_vl_pipeline/outputs/checkpoints/visual_grounding_v2/vg2_merged",
    "MoLaRx": PH2 / "checkpoints" / "cis_selective_final",
    "Full_LoRA": PH2 / "checkpoints" / "full_lora_only_final",
}


def to_wp6(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        return None
    if a.shape[0] < 6:
        a = np.vstack([a, np.repeat(a[-1:], 6 - a.shape[0], axis=0)])
    return a[:6]


def main():
    splits = {n: [json.loads(l) for l in open(DATA / f"{n}.jsonl")][:N]
              for n in ["test_id", "test_ood"]}
    print({k: len(v) for k, v in splits.items()})
    results = {}
    for name, mdir in MODELS.items():
        if not (mdir / "config.json").exists():
            print("!! missing", name, mdir)
            continue
        print(f"\n== {name} ==")
        processor = AutoProcessor.from_pretrained(str(mdir))
        model = AutoModelForImageTextToText.from_pretrained(
            str(mdir), dtype=torch.bfloat16).to("cuda").eval()
        results[name] = {}
        for sname, recs in splits.items():
            ade, fde, fails = [], [], 0
            with torch.no_grad():
                for rec in tqdm(recs, desc=f"{name}:{sname}"):
                    try:
                        images = [Image.open(p).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
                                  for p in rec["image_paths"]]
                    except Exception:
                        fails += 1
                        continue
                    conv = [{"role": "user", "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[1]},
                        {"type": "image", "image": images[2]},
                        {"type": "text", "text": rec["question"]}]}]
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
                    gt = to_wp6(parse_trajectory(rec["answer"]))
                    if pred is None or gt is None:
                        fails += 1
                        continue
                    d = np.linalg.norm(pred - gt, axis=1)
                    ade.append(float(d.mean()))
                    fde.append(float(d[-1]))
            results[name][sname] = {
                "ADE": round(float(np.mean(ade)), 4), "FDE": round(float(np.mean(fde)), 4),
                "n": len(ade), "fails": fails}
            print(sname, results[name][sname])
        if "test_id" in results[name] and "test_ood" in results[name]:
            results[name]["gap_ADE"] = round(
                results[name]["test_ood"]["ADE"] - results[name]["test_id"]["ADE"], 4)
        del model
        torch.cuda.empty_cache()
    json.dump(results, open(OUT, "w"), indent=2)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()

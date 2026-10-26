"""Phase 2 PaliGemma2 trajectory eval — ADE/FDE/Gap/ImgDep for one (ckpt, dataset) cell."""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_paligemma import (IMAGE_TOKEN_ID, TARGET_IMAGE_SIZE, TEMPORAL_PREFIX, INSTRUCTION)
from paligemma_common import resolve_three_frames, build_temporal_map_for_dataset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

NUS_DIR = Path("lfm2_vl_pipeline/outputs/data")
AV2_DIR = Path("argoverse2_pipeline/outputs/data")
TRAJ_RE = re.compile(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?")

def parse_traj(t):
    ms = TRAJ_RE.findall(t.strip())
    if not ms: return None
    try: return [(float(x), float(y)) for x, y in ms]
    except ValueError: return None

def ade_fde(p, g):
    n = min(len(p), len(g))
    if n == 0: return None, None
    d = [np.hypot(p[i][0]-g[i][0], p[i][1]-g[i][1]) for i in range(n)]
    return float(np.mean(d)), float(d[-1])

@torch.no_grad()
def eval_split(model, proc, records, tmap, use_blank, tag, max_new_tokens=128):
    model.eval()
    ades, fdes, fails = [], [], 0
    blank = Image.new("RGB", TARGET_IMAGE_SIZE, color=(128,128,128))
    for rec in tqdm(records, desc=tag):
        three = resolve_three_frames(rec, tmap)
        if three is None: fails += 1; continue
        if use_blank: imgs = [blank, blank, blank]
        else:
            imgs = []
            for p in three:
                try: imgs.append(Image.open(p).convert("RGB").resize(TARGET_IMAGE_SIZE, Image.LANCZOS))
                except Exception: imgs.append(blank)
        prefix = "<image><image><image>" + TEMPORAL_PREFIX + INSTRUCTION + "\n"
        try:
            inputs = proc(text=prefix, images=imgs, return_tensors="pt", padding="longest", truncation=True)
            inputs = {k:v.to(model.device) if hasattr(v,"to") else v for k,v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            plen = inputs["input_ids"].shape[1]
            text = proc.tokenizer.decode(out[0][plen:], skip_special_tokens=True)
        except Exception as e:
            fails += 1; continue
        pred = parse_traj(text); gt = parse_traj(rec.get("answer",""))
        if pred is None or gt is None or len(pred)<3 or len(gt)<3: fails += 1; continue
        a, f = ade_fde(pred, gt)
        if a is not None: ades.append(a); fdes.append(f)
    return {"n_total":len(records), "n_eval":len(ades), "n_fail":fails,
            "ade_mean":float(np.mean(ades)) if ades else None,
            "fde_mean":float(np.mean(fdes)) if fdes else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True, choices=["nuscenes","av2"])
    ap.add_argument("--max_samples", type=int, default=300); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_dir = NUS_DIR if args.dataset=="nuscenes" else AV2_DIR
    id_recs = [json.loads(l) for l in open(data_dir/"test_id.jsonl")]
    ood_recs = [json.loads(l) for l in open(data_dir/"test_ood.jsonl")]
    rng = np.random.RandomState(42)
    if len(id_recs)>args.max_samples:
        idx = rng.choice(len(id_recs), args.max_samples, replace=False); id_recs=[id_recs[i] for i in sorted(idx)]
    if len(ood_recs)>args.max_samples:
        idx = rng.choice(len(ood_recs), args.max_samples, replace=False); ood_recs=[ood_recs[i] for i in sorted(idx)]
    tmap = build_temporal_map_for_dataset(args.dataset)
    print(f"Loading PaliGemma from {args.ckpt} ..."); t0=time.time()
    proc = AutoProcessor.from_pretrained(args.ckpt)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"Loaded in {time.time()-t0:.1f}s")
    res = {"ckpt":args.ckpt, "dataset":args.dataset,
           "id":eval_split(mdl,proc,id_recs,tmap,False,"ID"),
           "ood":eval_split(mdl,proc,ood_recs,tmap,False,"OOD"),
           "id_blank":eval_split(mdl,proc,id_recs,tmap,True,"ID-blank"),
           "ood_blank":eval_split(mdl,proc,ood_recs,tmap,True,"OOD-blank")}
    a_id=res["id"]["ade_mean"]; a_ood=res["ood"]["ade_mean"]; a_idb=res["id_blank"]["ade_mean"]
    res["summary"] = {"ADE_id":a_id,"ADE_ood":a_ood,"FDE_id":res["id"]["fde_mean"],"FDE_ood":res["ood"]["fde_mean"],
        "Gap_rel_pct":100*(a_ood-a_id)/a_id if a_id else None,
        "Gap_abs_m":a_ood-a_id if (a_id and a_ood) else None,
        "ImgDep_pct":100*(a_idb-a_id)/a_id if (a_id and a_idb) else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as f: json.dump(res, f, indent=2)
    gr = res["summary"]["Gap_rel_pct"]; idp = res["summary"]["ImgDep_pct"]
    print(f"\n==SUMMARY== {args.dataset}: ADE_id={a_id:.3f}m ADE_ood={a_ood:.3f}m Gap={gr:+.1f}% ImgDep={idp:+.1f}%")
    print(f"Saved to {args.out}")

if __name__=="__main__": main()

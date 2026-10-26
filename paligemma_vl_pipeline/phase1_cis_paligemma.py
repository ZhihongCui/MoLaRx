"""Phase 1 — Causal Importance Score (CIS) diagnosis for PaliGemma2-3B.

Counterfactual token patching per layer and modality.

Usage:
  cd /mnt/vdb/LFM_project
  CUDA_VISIBLE_DEVICES=0 python paligemma_vl_pipeline/phase1_cis_paligemma.py \
      --dataset nuscenes --K 100
"""
import argparse, json, os, random, re, sys, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_paligemma import (
    PaliGemmaConfig, CHECKPOINT_DIR, OUTPUT_DIR,
    NUS_DATA_DIR, AV2_DATA_DIR, NUM_LAYERS,
)
from paligemma_common import (
    prepare_input_3frame, get_image_mask, get_text_mask,
    build_temporal_map_for_dataset,
)

_WAYPOINT_RE = re.compile(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?")

def parse_trajectory(text):
    out = []
    for xs, ys in _WAYPOINT_RE.findall(text):
        try:
            out.append((float(xs), float(ys)))
        except ValueError:
            continue
    return out if out else None


def traj_displacement(pred, gt):
    if pred is None or gt is None: return None
    n = min(len(pred), len(gt))
    if n == 0: return None
    return float(np.sqrt(sum(
        (pred[i][0]-gt[i][0])**2 + (pred[i][1]-gt[i][1])**2 for i in range(n)
    ) / n))


def kinematic_key(rec):
    return np.array([
        float(rec.get("ego_speed", 0.0)),
        float(rec.get("front_distance", 30.0)),
        float(rec.get("lead_velocity", 0.0)),
    ])


def build_paired_samples(id_records, ood_records, K, seed=42):
    rng = np.random.default_rng(seed)
    if len(ood_records) > K:
        idx = rng.choice(len(ood_records), K, replace=False)
        ood_records = [ood_records[i] for i in idx]
    id_keys = np.stack([kinematic_key(r) for r in id_records])
    pairs = []
    for ood_rec in ood_records:
        k = kinematic_key(ood_rec)
        d = np.linalg.norm(id_keys - k, axis=1)
        pairs.append((id_records[int(np.argmin(d))], ood_rec))
    return pairs


def _get_language_layers(model):
    """Navigate to PaliGemma2's Gemma2 decoder layers."""
    # Structure: model.model.language_model (Gemma2Model) -> .layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "layers"): return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"): return lm.model.layers
    raise RuntimeError("Could not locate PaliGemma language model layers")


@torch.no_grad()
def collect_hidden_states(model, inputs):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states, out.logits


@torch.no_grad()
def generate_trajectory(model, processor, inputs, max_new_tokens=100):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    in_len = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(gen[0][in_len:], skip_special_tokens=True)
    return parse_trajectory(text)


@torch.no_grad()
def patched_forward_and_generate(model, processor, ood_inputs,
                                   layer_l, patched_hidden, max_new_tokens=100):
    layers = _get_language_layers(model)
    target_module = layers[layer_l - 1] if layer_l > 0 else None
    state = {"called": False}

    def hook(module, inputs, output):
        if state["called"]:
            return output
        state["called"] = True
        if isinstance(output, tuple):
            return (patched_hidden.to(output[0].dtype),) + output[1:]
        return patched_hidden.to(output.dtype)

    handle = target_module.register_forward_hook(hook) if target_module is not None else None
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(
                **ood_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
    finally:
        if handle is not None: handle.remove()
    in_len = ood_inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(gen[0][in_len:], skip_special_tokens=True)
    return parse_trajectory(text)


def patch_hidden(h_ood, h_id, modality, input_ids_ood):
    mask = get_image_mask(input_ids_ood) if modality == "img" else get_text_mask(input_ids_ood)
    mask = mask.to(h_ood.device)
    patched = h_ood.clone()
    n = min(h_ood.shape[1], h_id.shape[1])
    pos = mask[:n].nonzero(as_tuple=True)[0]
    if pos.numel() > 0:
        patched[0, pos, :] = h_id[0, pos, :]
    return patched


def run_cis(model, processor, pairs, device, num_layers,
            temporal_map=None, max_new_tokens=100):
    per_layer = {l: {"cis_img": [], "cis_txt": [], "dpa_hits": 0}
                  for l in range(num_layers)}
    skip_stats = Counter()
    MAX_PRINTED = 3

    def _skip(r, e=None, ctx=""):
        skip_stats[r] += 1
        if skip_stats[r] <= MAX_PRINTED:
            print(f"  [SKIP][{r}] {ctx}{': '+repr(e) if e else ''}")
            if e is not None and skip_stats[r] == 1: traceback.print_exc()
            sys.stdout.flush()

    for pi, (id_rec, ood_rec) in enumerate(tqdm(pairs, desc="  CIS pairs")):
        try:
            id_inputs  = prepare_input_3frame(id_rec, processor, device,
                                               temporal_map=temporal_map, include_labels=False)
            ood_inputs = prepare_input_3frame(ood_rec, processor, device,
                                               temporal_map=temporal_map, include_labels=False)
        except Exception as e:
            _skip("prepare_exc", e, f"pair {pi}"); continue
        if id_inputs is None or ood_inputs is None:
            _skip("prepare_none", None, f"pair {pi}"); continue
        try:
            Y_t = generate_trajectory(model, processor, ood_inputs, max_new_tokens)
        except Exception as e:
            _skip("baseline_gen_exc", e, f"pair {pi}"); continue
        if Y_t is None:
            _skip("baseline_gen_parse", None, f"pair {pi}"); continue
        try:
            h_id_all,  _ = collect_hidden_states(model, id_inputs)
            h_ood_all, _ = collect_hidden_states(model, ood_inputs)
        except Exception as e:
            _skip("hidden_exc", e, f"pair {pi}"); continue

        for l in range(1, num_layers + 1):
            if l >= len(h_id_all) or l >= len(h_ood_all): continue
            h_id_l, h_ood_l = h_id_all[l], h_ood_all[l]
            try:
                p = patch_hidden(h_ood_l, h_id_l, "img", ood_inputs["input_ids"])
                Y_cf = patched_forward_and_generate(model, processor, ood_inputs, l, p, max_new_tokens)
                d = traj_displacement(Y_cf, Y_t)
                if d is not None: per_layer[l-1]["cis_img"].append(d)
            except Exception as e:
                _skip("patch_img_exc", e, f"pair {pi} layer {l}")
            torch.cuda.empty_cache()
            try:
                p = patch_hidden(h_ood_l, h_id_l, "txt", ood_inputs["input_ids"])
                Y_cf = patched_forward_and_generate(model, processor, ood_inputs, l, p, max_new_tokens)
                d = traj_displacement(Y_cf, Y_t)
                if d is not None: per_layer[l-1]["cis_txt"].append(d)
            except Exception as e:
                _skip("patch_txt_exc", e, f"pair {pi} layer {l}")
            torch.cuda.empty_cache()
            diff = torch.nn.functional.cosine_similarity(
                h_id_l.mean(dim=1).float(), h_ood_l.mean(dim=1).float()).item()
            if diff < 0.95: per_layer[l-1]["dpa_hits"] += 1
        if (pi + 1) % 10 == 0:
            print(f"  [{pi+1}/{len(pairs)}] skips: {dict(skip_stats)}")
            sys.stdout.flush()

    print(f"\n  Final skip stats: {dict(skip_stats)}")
    n_good = sum(len(per_layer[l]["cis_img"]) for l in range(num_layers))
    if n_good == 0:
        print(f"  [FATAL] 0 CIS samples collected. Inspect [SKIP] above.")
        sys.exit(3)

    out = {}
    for l in range(num_layers):
        r = per_layer[l]
        out[l] = {
            "cis_img_mean": float(np.mean(r["cis_img"])) if r["cis_img"] else 0.0,
            "cis_img_std":  float(np.std(r["cis_img"]))  if r["cis_img"] else 0.0,
            "cis_txt_mean": float(np.mean(r["cis_txt"])) if r["cis_txt"] else 0.0,
            "cis_txt_std":  float(np.std(r["cis_txt"]))  if r["cis_txt"] else 0.0,
            "dpa_proxy":    r["dpa_hits"] / max(len(pairs), 1),
            "n_samples":    len(r["cis_img"]),
            "layer_type":   "GEMMA2",
        }
    return out, dict(skip_stats)


def compute_partition(per_layer, tau_img=0.30, tau_txt=2.00):
    align, tune, freeze = [], [], []
    for l in sorted(per_layer.keys()):
        img, txt = per_layer[l]["cis_img_mean"], per_layer[l]["cis_txt_mean"]
        if img > tau_img and img >= txt:
            align.append(l)
        elif txt > tau_txt:
            tune.append(l)
        else:
            freeze.append(l)
    return {"phi_align": align, "phi_tune": tune, "phi_freeze": freeze,
            "thresholds": {"cis_img": tau_img, "cis_txt": tau_txt}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nuscenes",
                        choices=["nuscenes", "av2"])
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = PaliGemmaConfig()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    data_dir = NUS_DATA_DIR if args.dataset == "nuscenes" else AV2_DATA_DIR
    ckpt_dir = args.ckpt or str(CHECKPOINT_DIR / f"tvg_{args.dataset}" / "tvg_merged")
    out_dir = OUTPUT_DIR / f"phase1_cis_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  PaliGemma2-3B Phase 1 (CIS) on {args.dataset}")
    print(f"  ckpt: {ckpt_dir}")
    print(f"  K: {args.K}")
    print(f"{'='*70}")
    sys.stdout.flush()

    print(f"\n[Phase 1] Building temporal map ...")
    t0 = time.time()
    tm = build_temporal_map_for_dataset(args.dataset)
    if tm is not None:
        print(f"  temporal_map: {len(tm):,} samples ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

    print(f"\n[Phase 1] Loading {ckpt_dir} ...")
    sys.stdout.flush()
    processor = PaliGemmaProcessor.from_pretrained(ckpt_dir)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        ckpt_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    device = next(model.parameters()).device
    print(f"  device: {device}, memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  language layers: {len(_get_language_layers(model))}")
    sys.stdout.flush()

    id_records  = [json.loads(l) for l in open(data_dir / "test_id.jsonl")]
    ood_records = [json.loads(l) for l in open(data_dir / "test_ood.jsonl")]
    print(f"[Phase 1] Loaded {len(id_records)} ID + {len(ood_records)} OOD")
    pairs = build_paired_samples(id_records, ood_records, K=args.K, seed=cfg.seed)
    print(f"[Phase 1] Built {len(pairs)} scene-matched pairs")
    sys.stdout.flush()

    per_layer, skip_stats = run_cis(
        model, processor, pairs, device, NUM_LAYERS,
        temporal_map=tm, max_new_tokens=cfg.max_new_tokens)
    partition = compute_partition(per_layer, cfg.cis_threshold_img, cfg.cis_threshold_txt)

    results = {
        "dataset": args.dataset, "num_pairs": len(pairs),
        "per_layer": {str(l): v for l, v in per_layer.items()},
        "partition": partition, "skip_stats": skip_stats,
    }
    with open(out_dir / "paligemma_cis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"  PaliGemma2 CIS results ({args.dataset})")
    print("="*70)
    print(f"{'Layer':>5} {'CIS_img':>8} {'CIS_txt':>8} {'DPA':>6} {'n':>4}")
    print("-"*40)
    for l in range(NUM_LAYERS):
        r = per_layer[l]
        print(f"{l:5d} {r['cis_img_mean']:8.4f} {r['cis_txt_mean']:8.4f} "
              f"{r['dpa_proxy']:6.3f} {r['n_samples']:4d}")
    print("\nPartition:")
    for k, v in partition.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out_dir / 'paligemma_cis_results.json'}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

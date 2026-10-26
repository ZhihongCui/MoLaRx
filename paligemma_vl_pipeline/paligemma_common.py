"""Shared helpers for PaliGemma2 pipeline: input prep, temporal_map, masks."""
from pathlib import Path
from PIL import Image
import torch

from config_paligemma import (
    IMAGE_TOKEN_ID, TARGET_IMAGE_SIZE,
    TEMPORAL_PREFIX, INSTRUCTION,
)


# ════════════════════════════════════════════════════════════════
#  Temporal frame resolver (same pattern as llava_common)
# ════════════════════════════════════════════════════════════════

def resolve_three_frames(rec, temporal_map=None):
    """Return (prev_path, curr_path, next_path) or None."""
    if temporal_map is not None:
        token = rec.get("sample_token")
        if token and token in temporal_map:
            return temporal_map[token]
    ips = rec.get("image_paths")
    if isinstance(ips, (list, tuple)) and len(ips) == 3:
        return tuple(ips)
    ip = rec.get("image_path")
    if ip:
        return (ip, ip, ip)
    return None


# ════════════════════════════════════════════════════════════════
#  Input preparation (3 temporal frames, PaliGemma format)
# ════════════════════════════════════════════════════════════════

def prepare_input_3frame(rec, processor, device,
                         temporal_map=None, include_labels=True):
    """Build PaliGemma-compatible batch.

    PaliGemma prompt format:
        "<image><image><image>{prefix}\\n"  (user prompt)
        "{answer}"                            (supervised part, only present when training)

    When include_labels=True, we supervise only on the answer tokens
    (mask prefix tokens with -100).
    """
    three = resolve_three_frames(rec, temporal_map)
    if three is None:
        return None

    images = []
    for p in three:
        try:
            img = Image.open(p).convert("RGB").resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", TARGET_IMAGE_SIZE, color=(128, 128, 128))
        images.append(img)

    question = TEMPORAL_PREFIX + INSTRUCTION
    # PaliGemma expects a literal '<image>' placeholder per image in the text.
    # The processor replaces each with 256 image tokens internally.
    prefix = "<image><image><image>" + question + "\n"

    if include_labels:
        # Use PaliGemmaProcessor's built-in `suffix` API which correctly
        # masks prefix+images and supervises only on the answer tokens.
        # (Our earlier manual masking was buggy: it computed prefix_len
        # on the RAW tokenization where <image> is 1 token, but applied
        # the mask AFTER the processor expanded each <image> to 256
        # tokens, leaving the instruction text unmasked and training
        # the model to echo the prompt.)
        inputs = processor(
            text=prefix,
            images=images,
            suffix=rec["answer"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
    else:
        inputs = processor(
            text=prefix,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )

    inputs = {k: v.to(device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    return inputs


# ════════════════════════════════════════════════════════════════
#  nuScenes temporal_map (reuse logic from llava_common)
# ════════════════════════════════════════════════════════════════

def build_nuscenes_temporal_map(dataroot="data/nuscenes", version="v1.0-trainval"):
    import os
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    tm = {}
    for sample in nusc.sample:
        tok = sample['token']
        cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        curr_path = os.path.join(dataroot, cam['filename'])
        prev_path = curr_path
        if sample['prev']:
            try:
                p = nusc.get('sample', sample['prev'])
                pc = nusc.get('sample_data', p['data']['CAM_FRONT'])
                prev_path = os.path.join(dataroot, pc['filename'])
            except Exception: pass
        next_path = curr_path
        if sample['next']:
            try:
                n = nusc.get('sample', sample['next'])
                nc = nusc.get('sample_data', n['data']['CAM_FRONT'])
                next_path = os.path.join(dataroot, nc['filename'])
            except Exception: pass
        tm[tok] = (prev_path, curr_path, next_path)
    return tm


def build_temporal_map_for_dataset(dataset):
    if dataset == "nuscenes":
        return build_nuscenes_temporal_map()
    elif dataset == "av2":
        return None
    return None


# ════════════════════════════════════════════════════════════════
#  Image vs text masks (for CIS hooks)
# ════════════════════════════════════════════════════════════════

def get_image_mask(input_ids):
    if input_ids.dim() > 1:
        input_ids = input_ids[0]
    return (input_ids == IMAGE_TOKEN_ID)


def get_text_mask(input_ids, attention_mask=None):
    if input_ids.dim() > 1:
        input_ids = input_ids[0]
        if attention_mask is not None and attention_mask.dim() > 1:
            attention_mask = attention_mask[0]
    txt = (input_ids != IMAGE_TOKEN_ID)
    if attention_mask is not None:
        txt = txt & attention_mask.bool()
    return txt


def pool_modality(hidden_layer, input_ids, modality, attention_mask=None):
    if modality == "img":
        mask = get_image_mask(input_ids)
    elif modality == "txt":
        mask = get_text_mask(input_ids, attention_mask)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    h = hidden_layer[0]
    mask = mask.to(h.device)
    if mask.sum() == 0:
        return None
    return h[mask].mean(dim=0).float()

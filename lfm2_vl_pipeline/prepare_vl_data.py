"""
Prepare VL Trajectory Dataset from nuScenes — v11 Trajectory Planning
======================================================================

Extracts CAM_FRONT images + future 3-second trajectory waypoints.
Prompt: ego_speed + front_distance + lead_velocity (3 text variables).
Answer: 6 (x, y) waypoints in ego-frame metres at t+0.5s … t+3.0s.

Output JSONL files:
  - train.jsonl      (Singapore, class-balanced by accel sign)
  - val.jsonl        (Singapore, held-out)
  - test_id.jsonl    (Singapore, held-out  ~23 % of ID)
  - test_ood.jsonl   (Boston,    sampled to match test_id size)

Usage:
  python lfm2_vl_pipeline/prepare_vl_data.py
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

from config_vl import (
    NUSCENES_DATAROOT, NUSCENES_VERSION,
    DATA_DIR, OUTPUT_DIR,
    CAMERA_CHANNEL,
    ID_CITY, OOD_CITY,
    VULNERABILITY_MAP, WEATHER_KEYWORDS,
    VLConfig,
)

FUTURE_STEPS = 6          # 6 waypoints  →  t+0.5 … t+3.0 s
WAYPOINT_DT  = 0.5        # seconds between keyframes


# ════════════════════════════════════════════════════════════════
#  Ego pose helpers
# ════════════════════════════════════════════════════════════════

def get_ego_pose(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", sd["ego_pose_token"])


def ego_speed(nusc, sample) -> float:
    """Scalar ego speed (m/s) via centred finite diff on neighbouring poses."""
    if sample["prev"] == "" or sample["next"] == "":
        return None
    prev_s = nusc.get("sample", sample["prev"])
    next_s = nusc.get("sample", sample["next"])
    p0 = np.array(get_ego_pose(nusc, prev_s)["translation"][:2])
    p1 = np.array(get_ego_pose(nusc, next_s)["translation"][:2])
    t0 = get_ego_pose(nusc, prev_s)["timestamp"] / 1e6
    t1 = get_ego_pose(nusc, next_s)["timestamp"] / 1e6
    dt = t1 - t0
    if dt <= 0:
        return None
    return float(np.linalg.norm(p1 - p0) / dt)


def future_waypoints_ego_frame(nusc, sample):
    """
    Compute future FUTURE_STEPS waypoints in the current ego frame.

    Returns list of (dx, dy) tuples in metres, or None if not enough future
    frames exist.  dx = lateral (right +), dy = longitudinal (forward +).
    """
    # Collect future sample tokens
    future_samples = []
    tok = sample["next"]
    while tok != "" and len(future_samples) < FUTURE_STEPS:
        s = nusc.get("sample", tok)
        future_samples.append(s)
        tok = s["next"]

    if len(future_samples) < FUTURE_STEPS:
        return None

    # Current ego pose
    curr_pose = get_ego_pose(nusc, sample)
    origin    = np.array(curr_pose["translation"][:2])
    q_curr    = Quaternion(curr_pose["rotation"])

    # Rotation matrix: global → ego  (2-D)
    fwd_global = q_curr.rotate(np.array([1.0, 0.0, 0.0]))[:2]
    fwd_norm   = np.linalg.norm(fwd_global)
    if fwd_norm < 1e-8:
        return None
    fwd  = fwd_global / fwd_norm
    left = np.array([-fwd[1], fwd[0]])   # 90° CCW
    R    = np.stack([left, fwd], axis=0)  # (2, 2): row0=lateral, row1=long

    waypoints = []
    for fs in future_samples[:FUTURE_STEPS]:
        fp = get_ego_pose(nusc, fs)
        pos_global = np.array(fp["translation"][:2])
        delta_global = pos_global - origin
        dxdy = R @ delta_global          # (lateral, longitudinal)
        waypoints.append((round(float(dxdy[0]), 2),
                          round(float(dxdy[1]), 2)))
    return waypoints


def format_waypoints(wps) -> str:
    """Format list of (dx, dy) as model answer string."""
    return " ".join(f"({x},{y})" for x, y in wps)


# ════════════════════════════════════════════════════════════════
#  Closest annotation (for prompt context)
# ════════════════════════════════════════════════════════════════

def get_closest_annotation(nusc, sample_token: str):
    sample   = nusc.get("sample", sample_token)
    ld       = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", ld["ego_pose_token"])
    ego_xyz  = np.array(ego_pose["translation"][:2])

    best, best_dist = None, float("inf")
    for ann_token in sample["anns"]:
        ann  = nusc.get("sample_annotation", ann_token)
        vuln = VULNERABILITY_MAP.get(ann["category_name"], None)
        if vuln is None:
            continue
        obj_xyz = np.array(ann["translation"][:2])
        dist    = float(np.linalg.norm(obj_xyz - ego_xyz))
        if dist < best_dist:
            vel = 0.0
            if ann["next"] != "":
                ann_next   = nusc.get("sample_annotation", ann["next"])
                next_xyz   = np.array(ann_next["translation"][:2])
                next_s     = nusc.get("sample", ann_next["sample_token"])
                dt         = (next_s["timestamp"] - sample["timestamp"]) / 1e6
                if dt > 0:
                    dist_next = float(np.linalg.norm(next_xyz - ego_xyz))
                    vel = (dist - dist_next) / dt
            best_dist = dist
            best = {"distance": round(dist, 1), "velocity": round(vel, 1),
                    "vulnerability": vuln}
    return best


# ════════════════════════════════════════════════════════════════
#  Weather
# ════════════════════════════════════════════════════════════════

def parse_weather(description: str) -> str:
    desc_lower = description.lower()
    for label in ["rain", "fog", "night"]:
        for kw in WEATHER_KEYWORDS[label]:
            if kw in desc_lower:
                return label
    return "clear"


# ════════════════════════════════════════════════════════════════
#  Prompt construction
# ════════════════════════════════════════════════════════════════

def build_prompt(spd: float, closest) -> str:
    """Build text prompt from scalar ego speed + closest object info."""
    dist = closest["distance"] if closest else 50.0
    lead = closest["velocity"] if closest else 0.0
    return (
        f"ego_speed={spd:.1f} m/s, "
        f"front_distance={dist:.1f} m, "
        f"lead_velocity={lead:.1f} m/s. "
        "Predict future trajectory as 6 waypoints (x,y) in ego-frame "
        "metres at t+0.5s to t+3.0s."
    )


# ════════════════════════════════════════════════════════════════
#  Accel sign for balancing (same direction as classification)
# ════════════════════════════════════════════════════════════════

def traj_to_accel_label(wps) -> int:
    """Classify trajectory into 0=accel / 1=maintain / 2=decel by end speed."""
    if len(wps) < 2:
        return 1
    # Approximate end speed from last two waypoints
    dy1 = wps[-2][1] - wps[-3][1] if len(wps) >= 3 else wps[-1][1]
    dy2 = wps[-1][1] - wps[-2][1]
    diff = dy2 - dy1   # positive → accelerating, negative → decelerating
    if diff > 0.3:
        return 0
    elif diff < -0.3:
        return 2
    return 1


# ════════════════════════════════════════════════════════════════
#  Class balancing (same as before)
# ════════════════════════════════════════════════════════════════

def balance_classes(records: list, seed: int = 42) -> list:
    random.seed(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["_bal_label"], []).append(r)
    max_count = max(len(v) for v in by_class.values())
    balanced  = []
    for k in sorted(by_class.keys()):
        recs = by_class[k]
        balanced.extend(recs)
        if len(recs) < max_count:
            balanced.extend(random.choices(recs, k=max_count - len(recs)))
    random.shuffle(balanced)
    orig = {k: len(v) for k, v in by_class.items()}
    new  = Counter(r["_bal_label"] for r in balanced)
    print(f"  [Balance] Original: {orig}")
    print(f"  [Balance] After oversampling: {dict(new)}")
    print(f"  [Balance] {sum(orig.values())} -> {len(balanced)} records")
    return balanced


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    cfg = VLConfig()

    print("=" * 70)
    print(f"  VL Trajectory Data Prep  (v11)  —  nuScenes {NUSCENES_VERSION}")
    print("=" * 70)

    nusc = NuScenes(version=NUSCENES_VERSION,
                    dataroot=NUSCENES_DATAROOT, verbose=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = Counter()
    city_kf = Counter()

    for scene in nusc.scene:
        weather  = parse_weather(scene["description"])
        log      = nusc.get("log", scene["log_token"])
        location = log["location"]
        city     = "boston" if "boston" in location.lower() else "singapore"
        city_kf[city]   # just initialise

        sample_token = scene["first_sample_token"]
        while sample_token != "":
            sample = nusc.get("sample", sample_token)

            if CAMERA_CHANNEL not in sample["data"]:
                skipped["no_cam"] += 1
                sample_token = sample["next"]
                continue

            spd = ego_speed(nusc, sample)
            if spd is None:
                skipped["edge_frame"] += 1
                sample_token = sample["next"]
                continue

            wps = future_waypoints_ego_frame(nusc, sample)
            if wps is None:
                skipped["no_future"] += 1
                sample_token = sample["next"]
                continue

            cam_data = nusc.get("sample_data", sample["data"][CAMERA_CHANNEL])
            img_path = os.path.join(NUSCENES_DATAROOT, cam_data["filename"])
            closest  = get_closest_annotation(nusc, sample_token)

            records.append({
                "image_path":   img_path,
                "question":     build_prompt(spd, closest),
                "answer":       format_waypoints(wps),
                "city":         city,
                "location":     location,
                "weather":      weather,
                "sample_token": sample_token,
                "scene_name":   scene["name"],
                "ego_speed":    round(spd, 2),
                "front_distance": closest["distance"] if closest else 50.0,
                "lead_velocity":  closest["velocity"] if closest else 0.0,
                # label used only for class balancing — not in model input/output
                "_bal_label":   traj_to_accel_label(wps),
            })
            city_kf[city] += 1
            sample_token = sample["next"]

    print(f"\n  Collected {len(records)} valid keyframes")
    print(f"  Skipped:  {dict(skipped)}")
    print(f"  City distribution: {dict(city_kf)}")

    # ── City split ──
    id_records  = [r for r in records if r["city"] == ID_CITY]
    ood_records = [r for r in records if r["city"] == OOD_CITY]
    print(f"\n  ID  ({ID_CITY}):  {len(id_records)}")
    print(f"  OOD ({OOD_CITY}): {len(ood_records)}")

    # ── Train / val / test_id split ──
    random.seed(cfg.seed)
    random.shuffle(id_records)

    n       = len(id_records)
    n_train = int(n * cfg.train_ratio)
    n_val   = int(n * cfg.val_ratio)

    test_id_records = id_records[n_train + n_val:]

    # ── Cap OOD test to match ID test size ──
    test_ood_records = list(ood_records)
    max_ood = getattr(cfg, "test_ood_max_samples", -1)
    if max_ood > 0 and len(test_ood_records) > max_ood:
        random.seed(cfg.seed)
        test_ood_records = random.sample(test_ood_records, max_ood)
        print(f"\n  [Split] OOD capped: {len(ood_records)} -> {len(test_ood_records)} "
              f"(test_ood_max_samples={max_ood})")

    splits = {
        "train":    id_records[:n_train],
        "val":      id_records[n_train:n_train + n_val],
        "test_id":  test_id_records,
        "test_ood": test_ood_records,
    }

    # ── Balance training set ──
    print("\n  Balancing training set ...")
    splits["train"] = balance_classes(splits["train"], seed=cfg.seed)

    # ── Verify image paths ──
    missing = sum(
        1 for name, data in splits.items()
        for r in data if not os.path.isfile(r["image_path"])
    )
    if missing:
        print(f"\n  WARNING: {missing} image paths not found on disk "
              f"(expected if running locally).")

    # ── Save (strip internal _bal_label before writing) ──
    print()
    OUTPUT_KEYS = {k for k in list(splits["train"][0].keys()) if k != "_bal_label"}
    for name, data in splits.items():
        city_dist = Counter(r["city"] for r in data)
        print(f"  {name:>8}: {len(data)} records  city={dict(city_dist)}")
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps({k: v for k, v in r.items()
                                    if k != "_bal_label"}) + "\n")
        print(f"           -> {path}")

    # ── Save meta ──
    meta = {
        "version": "v11_trajectory",
        "nuscenes_version": NUSCENES_VERSION,
        "camera": CAMERA_CHANNEL,
        "id_city": ID_CITY,
        "ood_city": OOD_CITY,
        "future_steps": FUTURE_STEPS,
        "waypoint_dt_s": WAYPOINT_DT,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "test_ood_max_samples": max_ood,
        "id_keyframes": len(id_records),
        "ood_keyframes": len(ood_records),
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  VL Trajectory Data Preparation Complete")
    print(f"  Output: {DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

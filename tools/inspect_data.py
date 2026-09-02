#!/usr/bin/env python3
"""
Dump the structure of a training pickle to understand the data format.

Usage:
  python tools/inspect_data.py \
      --data_path /path/to/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
      --npy_samples 20
"""

import argparse
import pickle
import os
import numpy as np
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description="Inspect training pickle structure")
    p.add_argument("--data_path", required=True)
    p.add_argument("--npy_samples", type=int, default=20,
                   help="Number of .npy files to actually load and check shapes (default: 20)")
    return p.parse_args()


def describe(val, max_len=120):
    """Return a compact description of a value."""
    if isinstance(val, np.ndarray):
        return f"ndarray(shape={val.shape}, dtype={val.dtype})"
    if isinstance(val, list):
        if len(val) == 0:
            return "list[0]"
        inner = describe(val[0])
        return f"list[{len(val)}] of {inner}"
    if isinstance(val, dict):
        return f"dict with keys: {sorted(val.keys())}"
    if isinstance(val, str):
        if len(val) > max_len:
            return repr(val[:max_len] + "...")
        return repr(val)
    if isinstance(val, (int, float, bool)):
        return repr(val)
    return f"{type(val).__name__}: {val!r:.120}"


def main():
    args = parse_args()

    print(f"Loading {args.data_path} ...")
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} scenes\n")

    # ── Check data type ────────────────────────────────────────────────
    print(f"  top-level type: {type(data).__name__}")
    if isinstance(data, list):
        print(f"  first element type: {type(data[0]).__name__}")

    # ── Sample first few scenes ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE SCENES (first 3)")
    print("=" * 70)
    for i in range(min(3, len(data))):
        scene = data[i]
        print(f"\n── scene[{i}] ──")
        for k, v in scene.items():
            print(f"  {k}: {describe(v)}")

    # ── Aggregate keys across all scenes ────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY COVERAGE (all scenes)")
    print("=" * 70)
    all_keys = Counter()
    key_types = {}
    for scene in data:
        for k in scene.keys():
            all_keys[k] += 1
    for k in sorted(all_keys.keys()):
        print(f"  {k}: present in {all_keys[k]}/{len(data)} scenes")

    # ── Check types of each field ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIELD TYPES (sampled from first 100 scenes)")
    print("=" * 70)
    for k in sorted(all_keys.keys()):
        types_seen = Counter()
        for scene in data[:100]:
            v = scene.get(k)
            types_seen[describe(v, max_len=40)] += 1
        print(f"\n  {k}:")
        for desc, count in types_seen.most_common(5):
            print(f"    {desc}  ({count}×)")

    # ── Check image paths / npy files ───────────────────────────────────
    print("\n" + "=" * 70)
    print("IMAGE / NPY FILE CHECK")
    print("=" * 70)
    if "image" not in data[0]:
        print("  No 'image' key found — skipping.")
        return

    # Collect image paths
    all_image_paths = []
    for scene in data[:args.npy_samples]:
        paths = scene.get("image", [])
        all_image_paths.extend(paths if isinstance(paths, list) else [paths])

    print(f"  Sampled {len(all_image_paths)} image paths from first {args.npy_samples} scenes")
    if all_image_paths:
        print(f"  First path: {all_image_paths[0]}")
        print(f"  Path type: {type(all_image_paths[0]).__name__}")

    # Try to load .npy files and check shapes
    npy_shapes = Counter()
    npy_missing = 0
    npy_loaded = 0
    for p in all_image_paths:
        if not os.path.exists(p):
            npy_missing += 1
            continue
        try:
            arr = np.load(p)
            npy_shapes[arr.shape] += 1
            npy_loaded += 1
        except Exception as e:
            print(f"  Error loading {p}: {e}")

    print(f"\n  .npy files loaded: {npy_loaded}, missing: {npy_missing}")
    if npy_shapes:
        print(f"  .npy shapes found:")
        for shape, count in npy_shapes.most_common(10):
            print(f"    {shape}  ({count}×)")

    # ── Check action data ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ACTION DATA CHECK")
    print("=" * 70)
    if "action" not in data[0]:
        print("  No 'action' key found — skipping.")
        return

    action_shapes = Counter()
    for scene in data[:args.npy_samples]:
        a = scene.get("action")
        if isinstance(a, list):
            # Convert to numpy to check shape
            try:
                arr = np.array(a)
                action_shapes[arr.shape] += 1
            except Exception:
                action_shapes[f"list[{len(a)}] of {describe(a[0]) if a else 'empty'}"] += 1
        elif isinstance(a, np.ndarray):
            action_shapes[a.shape] += 1
        else:
            action_shapes[type(a).__name__] += 1

    print(f"  Action shapes from first {args.npy_samples} scenes:")
    for shape, count in action_shapes.most_common(10):
        print(f"    {shape}  ({count}×)")

    # ── Summary for seq length estimation ───────────────────────────────
    print("\n" + "=" * 70)
    print("SEQ LENGTH ESTIMATION SUMMARY")
    print("=" * 70)
    if npy_shapes:
        dominant_shape = npy_shapes.most_common(1)[0][0]
        print(f"  Dominant .npy shape: {dominant_shape}")
        if len(dominant_shape) == 2:
            h, w = dominant_shape
            print(f"  Visual tokens per frame: {h}×{w} = {h * w}")
        elif len(dominant_shape) == 3:
            frames, h, w = dominant_shape
            print(f"  Visual tokens per frame: {h}×{w} = {h * w}")
            print(f"  Frames per .npy: {frames}")
    if action_shapes:
        dominant_action = action_shapes.most_common(1)[0][0]
        print(f"  Dominant action shape: {dominant_action}")


if __name__ == "__main__":
    main()
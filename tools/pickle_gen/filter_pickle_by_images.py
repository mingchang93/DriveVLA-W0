#!/usr/bin/env python3
"""Filter a DriveVLA-W0 NavSim pickle to samples whose camera images exist.

The Qwen VLA dataset resolves each stored path to
``<data_root>/<log>/CAM_F0/<frame>.jpg`` and loads only
``image[cur_frame_idx]`` plus ``pre_1s_image[cur_frame_idx]``. This keeps the
samples where both files exist, so a partial OpenScene/NAVSIM camera download
can be used for a real-data smoke test.

Usage:
    python tools/pickle_gen/filter_pickle_by_images.py \
        --input  /mnt/sfs_turbo/models/DriveVLA-W0/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
        --data_root /mnt/sfs_turbo/lmc/navsim_openscene/trainval_sensor_blobs \
        --cur_frame_idx 3 \
        --output /mnt/sfs_turbo/lmc/navsim_openscene/navsim_trainval_subset.pkl
"""

import argparse
import os
import pickle

from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def resolve_image(path, data_root):
    """Mirror the dataset resolver; return an existing file or None."""
    if path.lower().endswith(IMAGE_EXTS) and os.path.exists(path):
        return path
    parts = path.split("/")
    filename = parts[-1][:-4] if parts[-1].endswith(".npy") else os.path.splitext(parts[-1])[0]
    rel = os.path.join(parts[-2], "CAM_F0", filename) if len(parts) >= 2 else filename
    for ext in IMAGE_EXTS:
        candidate = os.path.join(data_root, rel + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    p = argparse.ArgumentParser(description="Filter a NavSim pickle by image availability")
    p.add_argument("--input", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cur_frame_idx", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples from {args.input}")

    kept = []
    idx = args.cur_frame_idx
    for sample in tqdm(data, desc="Filtering"):
        ok = True
        for key in ("image", "pre_1s_image"):
            paths = sample.get(key)
            if not paths or idx >= len(paths) or not paths[idx]:
                ok = False
                break
            if resolve_image(paths[idx], args.data_root) is None:
                ok = False
                break
        if ok:
            kept.append(sample)
            if args.max_samples and len(kept) >= args.max_samples:
                break

    with open(args.output, "wb") as f:
        pickle.dump(kept, f)
    print(f"Kept {len(kept)} / {len(data)} samples -> {args.output}")


if __name__ == "__main__":
    main()

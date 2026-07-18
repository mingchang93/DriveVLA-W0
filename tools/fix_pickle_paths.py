#!/usr/bin/env python3
"""
Fix hardcoded absolute paths in DriveVLA-W0 pickle files.

The pickle files from the original repo store absolute paths to VQ code .npy files
(e.g. /mnt/nvme0n1p1/yingyan.li/...). This script replaces them with your local paths.

Usage:
    # Auto-detect new prefix from pickle location
    python tools/fix_pickle_paths.py data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl

    # Or specify explicitly
    python tools/fix_pickle_paths.py path/to/train.pkl \
        --new_prefix /data/lmc/DriveVLA-W0/data/navsim/processed_data
"""

import pickle
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fix hardcoded paths in DriveVLA-W0 pickle files")
    parser.add_argument("pkl_path", type=str, help="Path to the pickle file to fix")
    parser.add_argument("--old_prefix", type=str,
                        default="/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data",
                        help="Original prefix to replace (default: author's machine path)")
    parser.add_argument("--new_prefix", type=str, default=None,
                        help="New prefix to use (default: auto-detected from pickle location)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing")
    args = parser.parse_args()

    # Auto-detect new_prefix: go up 4 levels from the pickle's dir:
    #   <pickle_dir>  (meta/)
    #   parent        (processed_data/)
    #   grandparent   (navsim/)
    #   great-grandparent (data/)
    # That's the repo root relative to data. Append /data/navsim/processed_data
    if args.new_prefix is None:
        pkl_dir = Path(args.pkl_path).resolve().parent
        # Walk up until we find the repo root (contains scripts/, utils/, etc.)
        current = pkl_dir
        while current != current.parent:
            if (current / "scripts").is_dir() and (current / "utils").is_dir():
                break
            current = current.parent
        else:
            print("Could not find repo root. Specify --new_prefix explicitly.")
            sys.exit(1)
        args.new_prefix = str(current / "data" / "navsim" / "processed_data")

    pkl_path = Path(args.pkl_path)
    if not pkl_path.exists():
        print(f"Error: {pkl_path} not found")
        sys.exit(1)

    print(f"Pickle:     {pkl_path.resolve()}")
    print(f"Old prefix: {args.old_prefix}")
    print(f"New prefix: {args.new_prefix}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        print(f"Error: expected list, got {type(data)}")
        sys.exit(1)

    fixed = 0
    skipped = 0
    for i, item in enumerate(data):
        img_path = item.get("image_tokens_path", "")
        if not img_path:
            skipped += 1
            continue
        if img_path.startswith(args.old_prefix):
            new_path = img_path.replace(args.old_prefix, args.new_prefix, 1)
            if args.dry_run:
                print(f"  [{i}] {img_path}")
                print(f"   -> {new_path}")
            else:
                item["image_tokens_path"] = new_path
            fixed += 1

    print(f"\nEntries:     {len(data)}")
    print(f"Fixed:       {fixed}")
    print(f"Skipped:     {skipped}")

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to apply.")
        return

    out_path = pkl_path.parent / f"{pkl_path.stem}_fixed.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved:       {out_path}")
    print(f"\nNow run training with:\n"
          f"  --data_path {out_path}")


if __name__ == "__main__":
    main()

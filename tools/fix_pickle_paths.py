#!/usr/bin/env python3
"""
Fix hardcoded absolute paths in DriveVLA-W0 pickle files.

The pickle files store absolute paths to VQ code .npy files in list fields
(e.g. 'image' and 'pre_1s_image'). This script recursively replaces the old
prefix with your local prefix.

Usage:
    # Auto-detect new prefix from repo root
    python tools/fix_pickle_paths.py /path/to/train.pkl

    # Specify explicitly
    python tools/fix_pickle_paths.py /path/to/train.pkl \
        --new_prefix /data/models/DriveVLA-W0/data/navsim/processed_data

    # Preview only
    python tools/fix_pickle_paths.py /path/to/train.pkl --dry-run
"""

import pickle
import argparse
import os
import sys
from pathlib import Path


def replace_prefix_in_obj(obj, old_prefix, new_prefix, dry_run=False):
    """Recursively walk dicts/lists/strings and replace old_prefix with new_prefix."""
    count = 0
    if isinstance(obj, str):
        if old_prefix in obj:
            new = obj.replace(old_prefix, new_prefix, 1)
            if dry_run:
                print(f"  {obj[:80]}...")
                print(f"  → {new[:80]}...")
            count += 1
            return new, count
        return obj, 0
    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_v, c = replace_prefix_in_obj(v, old_prefix, new_prefix, dry_run)
            if c > 0:
                obj[k] = new_v
                count += c
        return obj, count
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_v, c = replace_prefix_in_obj(v, old_prefix, new_prefix, dry_run)
            if c > 0:
                obj[i] = new_v
                count += c
        return obj, count
    elif isinstance(obj, tuple):
        new_items = [replace_prefix_in_obj(v, old_prefix, new_prefix, dry_run) for v in obj]
        c = sum(x[1] for x in new_items)
        if c > 0:
            return tuple(x[0] for x in new_items), c
        return obj, 0
    return obj, 0


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

    if args.new_prefix is None:
        pkl_dir = Path(args.pkl_path).resolve().parent
        current = pkl_dir
        while current != current.parent:
            if (current / "scripts").is_dir() and (current / "utils").is_dir():
                break
            current = current.parent
        else:
            print("Error: could not find repo root. Specify --new_prefix explicitly.")
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

    print(f"Entries:    {len(data)}")

    if args.dry_run:
        print(f"\n--- Dry run: sample replacements ---")

    total_fixed, _ = replace_prefix_in_obj(data, args.old_prefix, args.new_prefix, args.dry_run)

    print(f"\nPaths fixed: {total_fixed}")

    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to apply.")
        return

    out_path = pkl_path.parent / f"{pkl_path.stem}_fixed.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    # Verify a sample
    if len(data) > 0 and isinstance(data[0], dict) and "image" in data[0] and data[0]["image"]:
        sample = data[0]["image"][0]
        exists = os.path.exists(sample)
        print(f"\nSample path: {sample}")
        print(f"File exists: {exists}")

    print(f"\nSaved:       {out_path}")
    print(f"Now train with: --data_path {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fix hardcoded paths in DriveVLA-W0 pickle files (recursive, handles lists)."""

import pickle, argparse, os, sys
from pathlib import Path

def _replace(obj, old, new):
    c = 0
    if isinstance(obj, str):
        if old in obj:
            return obj.replace(old, new, 1), 1
        return obj, 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            v2, c2 = _replace(v, old, new)
            if c2: obj[k] = v2; c += c2
        return obj, c
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            v2, c2 = _replace(v, old, new)
            if c2: obj[i] = v2; c += c2
        return obj, c
    if isinstance(obj, tuple):
        r = [_replace(v, old, new) for v in obj]
        c = sum(x[1] for x in r)
        return tuple(x[0] for x in r), c if c else obj
    return obj, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path")
    parser.add_argument("--old_prefix", default="/mnt/nvme0n1p1/yingyan.li/repo/VLA_Emu_Huawei/data/navsim/processed_data")
    parser.add_argument("--new_prefix", required=True)
    args = parser.parse_args()

    p = Path(args.pkl_path)
    with open(p, "rb") as f:
        data = pickle.load(f)

    total = 0
    for item in data:
        _, c = _replace(item, args.old_prefix, args.new_prefix)
        total += c

    out = p.parent / f"{p.stem}_fixed.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)

    print(total)

if __name__ == "__main__":
    main()

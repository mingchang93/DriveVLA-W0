#!/usr/bin/env python3
"""Replicate a NavSim pickle N times (throughput measurements only).

A small real-data subset gives one dataloader batch per epoch, so loading
cannot overlap with compute and the wall-clock step time is dominated by the
critical-path fetch. Replicating the sample list provides enough batches per
epoch for prefetching; the model sees the same scenes repeatedly, so do not use
the result for learning experiments.

Usage:
    python tools/pickle_gen/replicate_pickle.py \
        --input navsim_trainval_subset.pkl \
        --output navsim_trainval_subset_x20.pkl \
        --repeat 20
"""

import argparse
import pickle


def main():
    p = argparse.ArgumentParser(description="Replicate a NavSim pickle for throughput runs")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeat", type=int, default=20)
    args = p.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    out = data * args.repeat
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"Replicated {len(data)} x {args.repeat} = {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()

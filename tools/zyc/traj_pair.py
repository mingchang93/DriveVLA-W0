#!/usr/bin/env python3
"""Paired contrast of traj_metrics.py CSVs: <out dir>/<arm>.csv.

Same convention as submetrics.py: mean delta over the same scenes,
t = mean / (sd / sqrt(n)). First arm is the base; the rest are contrasted
against it. Reads only; writes nothing.

Usage: traj_pair.py <out dir> <base arm> <arm> [<arm> ...]
"""
import csv
import os
import sys


def load(out_dir, arm):
    with open(os.path.join(out_dir, arm + ".csv")) as f:
        return {r["scene"]: {k: float(v) for k, v in r.items() if k != "scene"} for r in csv.DictReader(f)}


def main(out_dir, base, arms):
    b = load(out_dir, base)
    print("%-34s %8s %8s %8s %8s %8s %8s" % ("arm", "dADE", "t", "dFDE", "t", "dhead", "t"))
    for a in arms:
        x = load(out_dir, a)
        common = sorted(set(b) & set(x))
        line = "%-34s" % a[:34]
        for k in ("ade", "fde", "head_deg"):
            d = [x[s][k] - b[s][k] for s in common]
            n = len(d)
            m = sum(d) / n
            sd = (sum((v - m) ** 2 for v in d) / (n - 1)) ** 0.5
            t = m / (sd / n ** 0.5) if sd else 0.0
            line += " %+8.4f %8.2f" % (m, t)
        print(line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])

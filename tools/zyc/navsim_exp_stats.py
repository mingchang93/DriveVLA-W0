#!/usr/bin/env python3
"""Per-arm stats from a navsim exp dir's score csv.

Prints mean PDMS and the mean of every per-scene component column, plus how many
scenes failed `no_at_fault_collisions` (i.e. collision rate). Reads only.

Usage: navsim_exp_stats.py <exp/eval_<tag>_full> [<dir> ...]
"""
import csv
import glob
import os
import sys

COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "ego_progress", "time_to_collision_within_bound", "comfort", "score"]


def main(dirs):
    print("%-30s %5s %7s %7s %7s %7s %7s %7s %6s" % (
        "arm", "n", "PDMS", "NC", "DAC", "DDC", "EP", "TTC", "coll"))
    for d in dirs:
        rows = []
        for c in sorted(glob.glob(os.path.join(d, "*", "*.csv"))):
            with open(c) as f:
                r = list(csv.DictReader(f))
            if r and "score" in r[0]:
                rows = r
        if not rows:
            print("%-30s no score rows" % os.path.basename(d.rstrip("/")))
            continue
        out = []
        for k in COLS:
            v = [float(x[k]) for x in rows if x.get(k) not in (None, "")]
            out.append(100.0 * sum(v) / len(v) if v else float("nan"))
        nc = out[0]
        print("%-30s %5d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %6.2f" % (
            (os.path.basename(d.rstrip("/")).replace("eval_", "").replace("_full", ""),
             len(rows), out[6], out[0], out[1], out[2], out[3], out[4], 100.0 - nc)))


if __name__ == "__main__":
    main(sys.argv[1:])

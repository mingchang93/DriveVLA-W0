#!/usr/bin/env python3
"""Mean PDMS from each navsim exp dir's score csv (column named 'score')."""
import csv
import glob
import os
import sys

for d in sys.argv[1:]:
    csvs = sorted(glob.glob(os.path.join(d, "*", "*.csv")))
    best = None
    for c in csvs:
        with open(c) as f:
            rows = list(csv.DictReader(f))
        if not rows or "score" not in rows[0]:
            continue
        vals = [float(r["score"]) for r in rows if r.get("score") not in (None, "")]
        if vals:
            best = (len(vals), sum(vals) / len(vals))
    name = os.path.basename(d.rstrip("/"))
    print("%-34s %s" % (name, "n=%d PDMS=%.5f" % best if best else "no score rows"))

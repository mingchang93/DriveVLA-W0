#!/usr/bin/env python3
"""Shapley attribution of a PDMS difference to its components.

This fork's score is multiplicative in the two hard metrics and a weighted mean
of the soft ones (pdm_scorer.py):

    score = NC * DAC * (5*EP + 5*TTC + 2*comfort) / 12

`driving_direction_compliance` is reported but has weight 0.0, so it contributes
nothing to the score; it is not a hard metric.

For each arm, each component is credited with its Shapley share of the per-scene
score gap against the base arm (components not in the coalition take the base
arm's value). The shares sum exactly to the gap by construction. Reads score
csvs only, writes nothing.

Usage: pdm_attrib.py <exp dir of base> <exp dir> [<exp dir> ...]
"""
import csv
import glob
import itertools
import math
import os
import sys

COMP = ["nc", "dac", "ep", "ttc", "comfort"]


def load(d):
    rows = []
    for c in sorted(glob.glob(os.path.join(d, "*", "*.csv"))):
        with open(c) as f:
            r = list(csv.DictReader(f))
        if r and "score" in r[0]:
            rows = [x for x in r if x.get("token") != "average"]
    out = {}
    for x in rows:
        try:
            out[x["token"]] = (
                float(x["no_at_fault_collisions"]),
                float(x["drivable_area_compliance"]),
                float(x["ego_progress"]),
                float(x["time_to_collision_within_bound"]),
                float(x["comfort"]),
            )
        except (TypeError, ValueError):
            continue
    return out


def score(v):
    nc, dac, ep, ttc, cf = v
    return nc * dac * (5 * ep + 5 * ttc + 2 * cf) / 12.0


def main(base_dir, arms):
    b = load(base_dir)
    n = len(COMP)
    wt = {}
    for k in range(n):
        for S in itertools.combinations(range(n), k):
            wt[S] = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
    m0 = len(b)
    base_mean = sum(score(v) for v in b.values()) / m0
    print("base %-28s PDMS %.5f  n=%d" % (os.path.basename(base_dir.rstrip("/")), base_mean, m0))
    print("%-28s %8s %9s | %s" % ("arm", "PDMS", "dPDMS", "  ".join("%8s" % c for c in COMP)))
    for a in arms:
        x = load(a)
        common = sorted(set(b) & set(x))
        m = len(common)
        arm_mean = sum(score(x[t]) for t in common) / m
        share = [0.0] * n
        for t in common:
            av, bv = x[t], b[t]
            for i in range(n):
                rest = [j for j in range(n) if j != i]
                for k in range(n):
                    for S in itertools.combinations(rest, k):
                        va = [av[i] if j == i or j in S else bv[j] for j in range(n)]
                        vb = [av[j] if j in S else bv[j] for j in range(n)]
                        share[i] += wt[S] * (score(va) - score(vb))
        gap = (arm_mean - base_mean)
        name = os.path.basename(a.rstrip("/")).replace("eval_", "").replace("_full", "")
        print("%-28s %8.5f %+9.5f | %s" % (
            name, arm_mean, gap, "  ".join("%+8.5f" % (s / m) for s in share)))
        print("%-28s %8s %9s | %s" % (
            "  share of gap", "", "", "  ".join(
                "%+8.1f%%" % (100.0 * s / m / gap if gap else 0.0) for s in share)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

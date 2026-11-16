#!/usr/bin/env python3
"""The seven navtest metrics, per arm, from the score csv + inference dumps.

  ADE / FDE / head / LDist   from the increments in `action` vs `action_gt_denorm`,
                             composed into poses by traj_metrics.scene_metrics
                             (8 predicted steps x 0.5 s, aligned to gt rows 3..10).
                             LDist = longitudinal axis (x).
  Agent Coll                 100 * (1 - mean no_at_fault_collisions), i.e. NAVSIM's
                             collision-with-agent test on the simulated rollout.
  Lane Cross                 100 * (1 - mean drivable_area_compliance). Proxy: an
                             ego-polygon vs drivable-map violation, NOT a test for
                             crossing a painted lane line.
  HB Coll                    collision AND a predicted deceleration of at least
                             HB_THRESHOLD m/s^2 between two 0.5 s steps (speed =
                             |(dx,dy)| / 0.5). This is one operationalisation of
                             "hard-braking collision", not a benchmark constant --
                             the threshold is a choice, and changing it rescales
                             the count.

Usage: metrics7.py <exp dir> <json_output dir> [<exp dir> <json_output dir> ...]
Reads both sources read-only. Prints one row per arm, then paired comparisons
against the first arm (t for the continuous metrics, exact McNemar for the binary).
"""
import csv
import glob
import json
import math
import os
import sys

from traj_metrics import scene_metrics

HB_THRESHOLD = 3.0  # m/s^2
DT = 0.5            # s between dumped steps


def max_decel(pred):
    """Largest deceleration of the predicted trajectory, m/s^2 (>= 0)."""
    if len(pred) < 2:
        return 0.0
    v = [math.hypot(p[0], p[1]) / DT for p in pred]
    return max([0.0] + [(v[i - 1] - v[i]) / DT for i in range(1, len(v))])


def load_csv(exp_dir, cols):
    path = ""
    for c in sorted(glob.glob(os.path.join(exp_dir, "*", "*.csv"))):
        with open(c) as f:
            rows = list(csv.DictReader(f))
        if rows and "score" in rows[0]:
            path = c
    if not path:
        raise SystemExit("no score csv under " + exp_dir)
    out = {}
    for r in rows:
        if r.get("token") == "average" or not r.get("score"):
            continue
        try:
            out[r["token"]] = [float(r[k]) for k in cols]
        except (TypeError, ValueError):
            continue
    return out


def load_json(json_dir):
    out = {}
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith(".json") or "traj_diff" in name:
            continue
        try:
            with open(os.path.join(json_dir, name)) as f:
                j = json.load(f)
            pred, gt = j["action"], j.get("action_gt_denorm")
            if not gt:
                continue
            out[name[:-5]] = (scene_metrics(pred, gt), max_decel(pred))
        except Exception:
            continue
    return out


def arm(exp_dir, json_dir):
    traj = load_json(json_dir)
    flags = load_csv(exp_dir, ["no_at_fault_collisions", "drivable_area_compliance"])
    common = sorted(set(traj) & set(flags))
    rows = []
    for t in common:
        (ade, fde, head, _hf, lon, _lf, _lat, _latf, _lb), decel = traj[t]
        nc, dac = flags[t]
        coll = nc < 0.5
        rows.append({
            "token": t, "ade": ade, "fde": fde, "head": head, "ldist": lon,
            "coll": 1.0 if coll else 0.0,
            "lane": 1.0 if dac < 0.5 else 0.0,
            "hb": 1.0 if (coll and decel >= HB_THRESHOLD) else 0.0,
            "decel": decel,
        })
    return rows, len(common)


def mean(rows, k):
    return sum(r[k] for r in rows) / len(rows)


def pct(rows, k):
    return 100.0 * mean(rows, k)


def paired_t(a, b, k):
    n = len(a)
    d = [a[i][k] - b[i][k] for i in range(n)]
    m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    return m, (m / (sd / math.sqrt(n)) if sd else 0.0)


def mcnemar(a, b, k):
    b_only = sum(1 for i in range(len(a)) if b[i][k] > 0.5 and a[i][k] < 0.5)
    a_only = sum(1 for i in range(len(a)) if a[i][k] > 0.5 and b[i][k] < 0.5)
    n = a_only + b_only
    if not n:
        return a_only, b_only, 1.0
    k_ = min(a_only, b_only)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k_ + 1)) / 2.0 ** n)
    return a_only, b_only, p


def main(args):
    if len(args) < 2 or len(args) % 2:
        raise SystemExit(__doc__)
    names, data = [], []
    for i in range(0, len(args), 2):
        rows, n = arm(args[i], args[i + 1])
        names.append(os.path.basename(args[i].rstrip("/")).replace("eval_", "").replace("_full", ""))
        data.append(rows)
        print("%-26s %5d scenes" % (names[-1], n))

    print()
    print("HB Coll threshold: predicted decel >= %.1f m/s^2 (%.1f s samples)" % (HB_THRESHOLD, DT))
    print("%-26s %7s %7s %7s %7s %10s %10s %10s" % (
        "arm", "ADE", "FDE", "head", "LDist", "AgentColl", "HBColl", "LaneCross"))
    for n, rows in zip(names, data):
        print("%-26s %7.4f %7.4f %7.4f %7.4f %9.3f%% %9.3f%% %9.3f%%" % (
            n, mean(rows, "ade"), mean(rows, "fde"), mean(rows, "head"),
            mean(rows, "ldist"), pct(rows, "coll"), pct(rows, "hb"), pct(rows, "lane")))
    for n, rows in zip(names, data):
        nc = sum(1 for r in rows if r["coll"])
        nh = sum(1 for r in rows if r["hb"])
        print("%-26s collisions %d, of which hard-braking %d (%.1f%% of collisions), "
              "max decel >=%.1f in %d scenes (%.2f%%)" % (
                  n, nc, nh, 100.0 * nh / nc if nc else 0.0, HB_THRESHOLD,
                  sum(1 for r in rows if r["decel"] >= HB_THRESHOLD),
                  100.0 * sum(1 for r in rows if r["decel"] >= HB_THRESHOLD) / len(rows)))

    if len(data) < 2:
        return
    base, rest = data[0], data[1:]
    print()
    print("paired vs %s (same scenes):" % names[0])
    for n, rows in zip(names[1:], rest):
        idx = {r["token"]: r for r in rows}
        keys = [r["token"] for r in base if r["token"] in idx]
        a = [idx[t] for t in keys]
        b = [r for r in base if r["token"] in idx]
        print("  %s  n=%d" % (n, len(keys)))
        for k, label in (("ade", "ADE"), ("fde", "FDE"), ("head", "head"), ("ldist", "LDist")):
            m, t = paired_t(a, b, k)
            print("    %-10s delta %+8.4f  t %+6.2f" % (label, m, t))
        for k, label in (("coll", "AgentColl"), ("hb", "HBColl"), ("lane", "LaneCross")):
            ao, bo, p = mcnemar(a, b, k)
            print("    %-10s arm-only %4d  base-only %4d  McNemar p %.4f" % (label, ao, bo, p))


if __name__ == "__main__":
    main(sys.argv[1:])

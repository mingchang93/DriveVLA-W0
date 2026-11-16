#!/usr/bin/env python3
"""Open-loop trajectory metrics from the per-scene VLA inference dumps.

Pred and GT are the 8 cumulative ego-frame poses (x, y, theta) at 0.5 s spacing
that inference/qwen/inference_action_navsim_vava.py writes as `action` /
`action_gt_denorm`. Nothing is re-inferred: this is post-processing of jsons.

  ADE    mean_t ||p_t - g_t||            FDE   ||p_8 - g_8||
  head   mean_t |dtheta| (deg)           headF |dtheta| at step 8 (deg)
  lon    mean_t |x_p - x_g| (m)          lonF  |x_p - x_g| at step 8 (m)   <- LDist
  lat    mean_t |y_p - y_g| (m)          latF  |y_p - y_g| at step 8 (m)
  lbias  mean_t (y_p - y_g)  (m, signed; + = pred right of GT in ego frame)

x is the ego longitudinal axis and y the lateral one in these dumps.

Usage: traj_metrics.py <out dir> <json_output dir> [<dir> ...]
Prints one row per arm and writes per-scene values to <out dir>/<arm>.csv.
Reads the arms' json_output dirs read-only; nothing is written beside them.
"""
import csv
import json
import os
import sys

STEPS = 8
D2R = 3.141592653589793 / 180.0


def scene_metrics(pred, gt):
    n = min(len(pred), len(gt), STEPS)
    d = [(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1], pred[i][2] - gt[i][2]) for i in range(n)]
    ade = sum((dx * dx + dy * dy) ** 0.5 for dx, dy, _ in d) / n
    fde = (d[-1][0] ** 2 + d[-1][1] ** 2) ** 0.5
    head = sum(abs(dd[2]) for dd in d) / n / D2R
    head_f = abs(d[-1][2]) / D2R
    lon = sum(abs(dd[0]) for dd in d) / n
    lon_f = abs(d[-1][0])
    lat = sum(abs(dd[1]) for dd in d) / n
    lat_f = abs(d[-1][1])
    lat_bias = sum(dd[1] for dd in d) / n
    return ade, fde, head, head_f, lon, lon_f, lat, lat_f, lat_bias


def q(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def run(d):
    rows, skipped = [], 0
    for p in sorted(os.listdir(d)):
        if not p.endswith(".json") or "traj_diff" in p:
            continue
        try:
            with open(os.path.join(d, p)) as f:
                j = json.load(f)
            pred, gt = j["action"], j.get("action_gt_denorm")
            if not gt:
                skipped += 1
                continue
            rows.append((p[:-5],) + scene_metrics(pred, gt))
        except Exception:
            skipped += 1
    return rows, skipped


def main(out_dir, dirs):
    os.makedirs(out_dir, exist_ok=True)
    print("%-34s %6s %7s %7s %7s %7s %7s %7s %7s %7s %7s" % (
        "arm", "n", "ADE", "FDE", "head", "headF", "lon", "lonF", "lat", "latF", "lbias"))
    for d in dirs:
        rows, skipped = run(d)
        name = os.path.basename(os.path.dirname(d.rstrip("/"))) or d
        if not rows:
            print("%-34s no usable json (skipped %d)" % (name, skipped))
            continue
        cols = list(zip(*[r[1:] for r in rows]))
        mean = [sum(c) / len(c) for c in cols]
        print("%-34s %6d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %+7.3f" % (
            (name, len(rows)) + tuple(mean)))
        for label, p in (("  median", 0.50), ("  p95", 0.95)):
            print("%-34s %6s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (
                (label, "") + tuple(q(c, p) for c in cols)))
        out = os.path.join(out_dir, name + ".csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "ade", "fde", "head_deg", "head_final_deg",
                        "lon", "lon_final", "lat", "lat_final", "lat_bias"])
            for r in rows:
                w.writerow([r[0]] + ["%.6f" % v for v in r[1:]])
        if skipped:
            print("%-34s %6d json skipped (no GT or unreadable)" % ("", skipped))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

#!/usr/bin/env python3
"""Open-loop trajectory metrics from the per-scene VLA inference dumps.

Pred and GT are *not* poses. Each row of `action` / `action_gt_denorm` is one
step's SE2 increment (dx, dy, dtheta) expressed in that step's ego frame --
`tools/pickle_gen/pickle_generation_navsim_pre_1s.py` builds it as
`rel_all[j, j+1]` "only relative", and the `pre_1s` variant shifts by 2 frames,
so the spacing is 0.5 s. 8 predicted increments = 4 s of future motion.

The GT window is 12 increments spanning [i-3, i+8], so gt rows 0-2 are the three
*past* intervals and the predicted horizon lines up with gt rows 3..10. Comparing
pred[i] against gt[i] instead is off by 1.5 s; measured by shift search over 200
scenes, mean |dx| error is 0.450 m at shift 0 and 0.230 m at shift +3.

Increments are composed into poses before any error is taken, because a
displacement in a rotated frame is not a displacement in the ego frame:

  pose[0] = (0,0,0),  pose[k+1] = pose[k] + R(theta[k]) . (dx,dy),  theta[k+1] = theta[k] + dtheta

  ADE    mean_t ||p_t - g_t|| for t = 1..8
  FDE    ||p_8 - g_8||
  head   mean_t |theta_p - theta_g| (deg)      headF  same at t = 8
  lon    mean_t |x_p - x_g| (m)                lonF   same at t = 8   <- LDist
  lat    mean_t |y_p - y_g| (m)                latF   same at t = 8
  lbias  mean_t (y_p - y_g)  (m, signed; + = pred right of GT in ego frame)

Usage: traj_metrics.py <out dir> <json_output dir> [<dir> ...]
Prints one row per arm and writes per-scene values to <out dir>/<arm>.csv.
Reads the arms' json_output dirs read-only; nothing is written beside them.
"""
import csv
import json
import math
import os
import sys

STEPS = 8
GT_OFFSET = 3   # gt rows 0-2 are past intervals; the horizon starts at row 3
D2R = 3.141592653589793 / 180.0


def to_poses(steps):
    """Compose per-step (dx, dy, dtheta) increments into ego-frame poses."""
    x = y = th = 0.0
    out = [(0.0, 0.0, 0.0)]
    for dx, dy, dth in steps:
        c, s = math.cos(th), math.sin(th)
        x += c * dx - s * dy
        y += s * dx + c * dy
        th += dth
        out.append((x, y, th))
    return out


def scene_metrics(pred, gt):
    n = min(len(pred), len(gt) - GT_OFFSET, STEPS)
    if n < 1:
        raise ValueError("trajectory too short")
    pp = to_poses(pred[:n])[1:]
    gp = to_poses(gt[GT_OFFSET:GT_OFFSET + n])[1:]
    d = [(pp[i][0] - gp[i][0], pp[i][1] - gp[i][1], pp[i][2] - gp[i][2]) for i in range(n)]
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


def selftest():
    """Straight-line cases with hand-computable answers."""
    past = [[9.0, 9.0, 0.9]] * GT_OFFSET          # gt rows 0-2 are past, must be ignored
    future = [[1.0, 0.0, 0.0]] * STEPS            # 1 m per 0.5 s, straight
    gt = past + future
    ade, fde, head, _hf, lon, _lf, lat, _lf2, _lb = scene_metrics(future, gt)
    assert (ade, fde, head, lon, lat) == (0.0, 0.0, 0.0, 0.0, 0.0), "identical horizon must be zero"

    off = [[1.5, 0.0, 0.0]] * STEPS               # +0.5 m per step -> position error 0.5*t
    ade, fde, _h, _hf, lon, _lf, _l, _l2, _lb = scene_metrics(off, gt)
    assert abs(ade - 0.5 * 4.5) < 1e-12, ("ADE", ade)
    assert abs(fde - 4.0) < 1e-12, ("FDE", fde)
    assert abs(lon - 0.5 * 4.5) < 1e-12 and abs(lat) < 1e-12, ("lon/lat", lon, lat)

    turn = [[1.0, 0.0, 0.1]] * STEPS              # 0.1 rad per step
    _a, _f, head, head_f, _lo, _lof, _la, _laf, _lbi = scene_metrics(turn, gt)
    assert abs(head - 0.1 * 4.5 / D2R) < 1e-9, ("head", head)
    assert abs(head_f - 0.8 / D2R) < 1e-9, ("head_f", head_f)
    print("selftest ok")


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
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(sys.argv[1], sys.argv[2:])

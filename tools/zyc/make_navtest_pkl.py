#!/usr/bin/env python3
"""Build the full navtest pickle + token yaml for the Qwen VLA inference runner.

Derived from lmc's `make_eval_smoke_pkl.py`, with two changes that matter for a
real evaluation rather than a smoke:

1. **Scene set comes from the metric cache, not from sampling.** The smoke builder
   walks every 25th frame and stops at `--num-scenes`. PDMS only scores scenes
   that exist in the metric cache, so the cache's metadata csv is the canonical
   token list; taking it from there guarantees
   pickle tokens == token_yaml == what `run_pdm_score.py` iterates.
   Pass `--metric-cache` to use it (recommended), or `--all-frames` to emit every
   eligible frame instead.

2. **Reports coverage.** Any cache token we cannot build a scene for is listed with
   the reason, so a silent shortfall cannot be mistaken for a complete run.

Schema, window geometry (WINDOW=12, CENTER=3), the `pre_1s` offset of 2 frames,
the `.npy` image-path convention and the normalization are kept byte-identical to
lmc's builder — the runner and dataset depend on all of them.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import yaml

WINDOW = 12
CENTER = 3
TEXTS = ["go left", "go straight", "go right", "unknown"]


def yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rel(origin, other):
    dx = other["xy"][0] - origin["xy"][0]
    dy = other["xy"][1] - origin["xy"][1]
    c, s = np.cos(-origin["yaw"]), np.sin(-origin["yaw"])
    dyaw = (other["yaw"] - origin["yaw"] + np.pi) % (2 * np.pi) - np.pi
    return [float(c * dx - s * dy), float(s * dx + c * dy), float(dyaw)]


def normalize(a, q01, q99):
    a = np.asarray(a, dtype=np.float64)
    z = 2.0 * (a - q01) / (q99 - q01) - 1.0
    return np.clip(z, -1.0, 1.0).tolist()


def load_cache_tokens(cache_dir):
    """Token set PDMS will score: one directory per log, one subdir per token."""
    import csv

    tokens = set()
    meta = os.path.join(cache_dir, "metadata")
    csvs = sorted(glob.glob(os.path.join(meta, "*.csv")))
    for c in csvs:
        with open(c) as f:
            for row in csv.DictReader(f):
                # 列名在不同 navsim 版本间有差异，挑第一个像路径的字段
                val = row.get("file_name") or row.get("cache_path") or next(iter(row.values()))
                if not val:
                    continue
                parts = str(val).strip("/").split("/")
                # 缓存布局是 <log>/<scenario_type>/<token>/metric_cache.pkl
                # 末段是文件名时 token 取倒数第二段，否则末段本身就是 token
                if len(parts) >= 2 and "." in parts[-1]:
                    tokens.add(parts[-2])
                elif parts:
                    tokens.add(parts[-1])
    if tokens:
        return tokens, "metadata csv (%d 个文件)" % len(csvs)

    # csv 不可用时回退到目录结构：<cache>/<log>/<token>/
    for log_dir in glob.glob(os.path.join(cache_dir, "*")):
        if not os.path.isdir(log_dir) or os.path.basename(log_dir) == "metadata":
            continue
        for tok_dir in glob.glob(os.path.join(log_dir, "*")):
            if os.path.isdir(tok_dir):
                tokens.add(os.path.basename(tok_dir))
    return tokens, "目录结构扫描"


def build_meta(frames):
    meta = []
    for f in frames:
        meta.append(
            {
                "token": f["token"],
                "xy": (float(f["ego2global_translation"][0]), float(f["ego2global_translation"][1])),
                "yaw": yaw(f["ego2global_rotation"]),
                "file": os.path.basename(f["cams"]["CAM_F0"]["data_path"]),
                "cmd": int(np.asarray(f["driving_command"]).reshape(-1).argmax()),
            }
        )
    return meta


def window(meta, i, n, log_name, q01, q99):
    """12 帧窗口，中心在 i-CENTER..;与 lmc 的 builder 逐字一致。"""
    idxs = list(range(i - CENTER, i - CENTER + WINDOW))
    actions, images, texts = [], [], []
    for j in idxs:
        if 0 <= j < n - 1:
            actions.append(normalize(rel(meta[j], meta[j + 1]), q01, q99))
        else:
            actions.append([0.0, 0.0, 0.0])
        if 0 <= j < n:
            images.append("%s/%s.npy" % (log_name, meta[j]["file"][:-4]))
            texts.append(TEXTS[meta[j]["cmd"]])
        else:
            images.append(None)
            texts.append(None)
    return actions, images, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", required=True, help="navsim_logs/test")
    ap.add_argument("--cam-root", required=True, help="sensor_blobs/test")
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--out-yaml", required=True)
    ap.add_argument("--metric-cache", default=None,
                    help="metric_cache 目录；给了就只产出缓存里的 token")
    ap.add_argument("--all-frames", action="store_true",
                    help="不限 token，逐帧产出（与 --metric-cache 互斥）")
    ap.add_argument("--stride", type=int, default=1,
                    help="--all-frames 时的帧间隔")
    args = ap.parse_args()

    stats = json.load(open(args.norm_stats))["norm_stats"]["libero"]
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])

    want = None
    if args.metric_cache:
        want, how = load_cache_tokens(args.metric_cache)
        print("metric cache token 数: %d  (来源: %s)" % (len(want), how))
        if not want:
            sys.exit("metric cache 里没找到 token，不能继续")
    elif not args.all_frames:
        sys.exit("必须给 --metric-cache 或 --all-frames 之一")

    scenes, tokens = [], []
    skip = Counter()
    logs = sorted(glob.glob(os.path.join(args.logs_dir, "*.pkl")))
    print("日志数: %d" % len(logs))

    for li, log_path in enumerate(logs, 1):
        log_name = os.path.basename(log_path)[:-4]
        cam_dir = os.path.join(args.cam_root, log_name, "CAM_F0")
        if not os.path.isdir(cam_dir):
            skip["无相机目录"] += 1
            continue
        frames = pickle.load(open(log_path, "rb"))
        n = len(frames)
        if n < WINDOW + CENTER:
            skip["日志过短"] += 1
            continue
        meta = build_meta(frames)

        rng = range(CENTER, n - (WINDOW - CENTER - 1), 1 if want else args.stride)
        for i in rng:
            cur, prev = meta[i], meta[i - 2]
            if want is not None and cur["token"] not in want:
                continue
            if not (os.path.exists(os.path.join(cam_dir, cur["file"]))
                    and os.path.exists(os.path.join(cam_dir, prev["file"]))):
                skip["缺图"] += 1
                continue
            a, im, tx = window(meta, i, n, log_name, q01, q99)
            pa, pim, ptx = window(meta, i - 2, n, log_name, q01, q99)
            scenes.append({
                "token": cur["token"], "text": tx, "image": im, "action": a,
                "pre_1s_text": ptx, "pre_1s_image": pim, "pre_1s_action": pa,
            })
            tokens.append(cur["token"])
        if li % 25 == 0:
            print("  进度 %d/%d 日志，已产出 %d 场景" % (li, len(logs), len(scenes)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pkl)), exist_ok=True)
    pickle.dump(scenes, open(args.out_pkl, "wb"))
    yaml.safe_dump({"tokens": tokens}, open(args.out_yaml, "w"))

    print("\n写出 %d 场景 -> %s" % (len(scenes), args.out_pkl))
    print("写出 %d token  -> %s" % (len(tokens), args.out_yaml))
    if skip:
        print("跳过统计: %s" % dict(skip))
    if want is not None:
        got = set(tokens)
        missing = want - got
        print("覆盖率: %d/%d = %.2f%%" % (len(got), len(want), 100.0 * len(got) / len(want)))
        if missing:
            print("未覆盖 %d 个 cache token，前 5 个: %s"
                  % (len(missing), list(sorted(missing))[:5]))
        dup = len(tokens) - len(got)
        if dup:
            print("警告: token 有 %d 个重复" % dup)
    if scenes:
        s = scenes[0]
        print("样例 token=%s  image[3]=%s" % (s["token"], s["image"][3]))
        print("样例 action[3]=%s" % s["action"][3])


if __name__ == "__main__":
    main()

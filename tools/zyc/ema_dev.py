# -*- coding: utf-8 -*-
"""按上个项目/客户的 EMA 口径算偏差。

口径照搬 /mnt/sfs_turbo/zyc/hif8/analysis/compare132/m8full.py：
    ema(d, dec=0.9)：e = d[first] 初始化，之后 e = 0.9*e + 0.1*x
    dev = 100*(EMA_hif8 - EMA_bf16) / EMA_bf16
    窗口从第 1001 步起（排除预热），原脚本注释写明「窗口从 1001 步」

我们的跑是 4000 步，所以窗口 = 1001..4000。
用法: python3 ema_dev.py <hif8_tag> <bf16_tag>
"""
import io
import math
import os
import re
import sys

R = "/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs"
WIN_FROM = 1001


def series(tag, key):
    """从 train.log 里按出现顺序抽某个损失，返回 {step: value}（步号从 1 起）。"""
    p = os.path.join(R, tag, "train.log")
    s = io.open(p, encoding="utf-8", errors="replace").read()
    vals = [float(x) for x in re.findall(key + r"[\"']?: ([0-9.eE+-]+)", s)]
    return {i + 1: v for i, v in enumerate(vals)}


def ema(d, dec=0.9):
    o = {}
    e = None
    for k in sorted(d):
        e = d[k] if e is None else dec * e + (1 - dec) * d[k]
        o[k] = e
    return o


def stat(v):
    n = len(v)
    if not n:
        return None
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return n, m, sd, sd / math.sqrt(n)


qt = sys.argv[1] if len(sys.argv) > 1 else "night_hif8_168"
bt = sys.argv[2] if len(sys.argv) > 2 else "night_bf16"

print("EMA 口径：dec=0.9，首值初始化，窗口从第 %d 步起" % WIN_FROM)
print("HiF8 臂 = %s   bf16 臂 = %s\n" % (qt, bt))
print("%-14s %8s %14s %12s %12s" % ("指标", "n", "EMA偏差", "标准差", "±标准误"))

for key, label in (("action_loss", "action_loss"), ("ross_loss", "ross_loss"), ("'loss", "loss(HF四舍五入)")):
    b = series(bt, key)
    q = series(qt, key)
    eb, eq = ema(b), ema(q)
    ks = [k for k in sorted(set(b) & set(q)) if k >= WIN_FROM and abs(eb[k]) > 1e-12]
    dev = [100.0 * (eq[k] - eb[k]) / eb[k] for k in ks]
    st = stat(dev)
    if not st:
        print("%-14s   无数据" % label)
        continue
    n, m, sd, se = st
    print("%-14s %8d %+13.4f%% %11.4f%% %11.4f%%" % (label, n, m, sd, se))

# 顺带给出逐步原始偏差，便于和 EMA 对照（上个项目发现 DriveLM 上两者差 3 倍）
print("\n对照：同窗口的逐步原始偏差（不做 EMA）")
for key, label in (("action_loss", "action_loss"), ("ross_loss", "ross_loss")):
    b = series(bt, key)
    q = series(qt, key)
    ks = [k for k in sorted(set(b) & set(q)) if k >= WIN_FROM and abs(b[k]) > 1e-12]
    dev = [100.0 * (q[k] - b[k]) / b[k] for k in ks]
    st = stat(dev)
    if st:
        n, m, sd, se = st
        print("%-14s %8d %+13.4f%% %11.4f%% %11.4f%%" % (label, n, m, sd, se))

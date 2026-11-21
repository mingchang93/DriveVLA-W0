# -*- coding: utf-8 -*-
"""四条臂 PDMS 的场景级统计：把"场景抽样误差"和"训练轨迹差异"分开。

为什么要做：四条臂的 PDMS 跨度只有 ±2%，而且量化越多反而越好，顺序违反直觉。
在把这个归因为"轨迹噪声"之前，必须先量清楚两件事：
  1. 12146 个场景均值的标准误（场景抽样带来的不确定度）
  2. 配对比较：同一批场景上，两条臂的逐场景差值的均值与标准误
     —— 配对能消掉"场景本身难易"这个共同项，比各自均值相减灵敏得多
如果配对差值显著偏离 0，说明两个模型确实不同（这点毫无疑问）；
但这**不等于**"量化导致了差异"——要回答后者，需要同配置换种子的噪声底，
而那个我们还没测。本脚本只负责把前者量准。
"""
import os
import sys

import numpy as np
import pandas as pd

W = "/mnt/sfs_turbo/zyc/drivevla-w0-zyc/navsim_workspace/exp"
ARMS = ["bf16", "hif8_168", "hif8_196", "hif8_edge3"]

frames = {}
for t in ARMS:
    base = os.path.join(W, "eval_all_%s_full" % t)
    csvs = []
    for root, _, files in os.walk(base):
        csvs += [os.path.join(root, f) for f in files if f.endswith(".csv")]
    if not csvs:
        print("  %s 没有 CSV" % t)
        continue
    df = pd.read_csv(sorted(csvs)[-1])
    frames[t] = df

if not frames:
    sys.exit("没读到任何 CSV")

k = list(frames)[0]
print("CSV 列名:", list(frames[k].columns)[:12])
print()

# 找出 token 列和分数列
def pick(df):
    cols = {c.lower(): c for c in df.columns}
    tok = cols.get("token") or cols.get("scene_token") or list(df.columns)[0]
    sc = cols.get("score") or cols.get("pdm_score") or cols.get("pdms")
    if sc is None:
        num = [c for c in df.columns if df[c].dtype.kind in "fc"]
        sc = num[-1] if num else None
    return tok, sc

tok, sc = pick(frames[k])
print("token 列 = %s   分数列 = %s" % (tok, sc))
print()

print("=== 各臂独立统计 ===")
series = {}
for t in ARMS:
    if t not in frames:
        continue
    df = frames[t].dropna(subset=[sc])
    v = df[sc].astype(float).values
    s = pd.Series(v, index=df[tok].values)
    series[t] = s
    sem = v.std(ddof=1) / np.sqrt(len(v))
    print("  %-11s n=%5d  均值=%.4f  标准差=%.4f  标准误=%.4f" % (t, len(v), v.mean(), v.std(ddof=1), sem))

print()
print("=== 与 bf16 的配对比较（同一批场景逐场景相减）===")
if "bf16" in series:
    b = series["bf16"]
    for t in ARMS:
        if t == "bf16" or t not in series:
            continue
        o = series[t]
        common = b.index.intersection(o.index)
        d = o.loc[common].values - b.loc[common].values
        sem = d.std(ddof=1) / np.sqrt(len(d))
        tstat = d.mean() / sem if sem > 0 else float("nan")
        same = int((d == 0).sum())
        print("  %-11s 共同场景=%5d  平均差=%+.4f  标准误=%.4f  t=%+.1f  逐场景完全相同=%d (%.1f%%)"
              % (t, len(common), d.mean(), sem, tstat, same, 100.0 * same / len(d)))

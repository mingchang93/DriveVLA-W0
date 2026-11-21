# -*- coding: utf-8 -*-
"""比较 HiF8 臂与 bf16 臂的逐步损失偏差。

口径（沿用上一阶段）：报「带符号的平均相对偏差」，并与同种子噪声底对比。
确定性修好之后，同配置两把跑是逐位一致的，噪声底 = 0，
所以这里看到的偏差全部来自量化本身。

用法: python3 cmp_arms.py <hif8_tag> <bf16_tag>
"""
import io
import os
import re
import sys

R = "/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs"


def grab(tag):
    p = os.path.join(R, tag, "train.log")
    if not os.path.exists(p):
        return None
    s = io.open(p, encoding="utf-8", errors="replace").read()
    return {
        "action": [float(x) for x in re.findall(r"action_loss.: ([0-9.eE+-]+)", s)],
        "ross": [float(x) for x in re.findall(r"ross_loss.: ([0-9.eE+-]+)", s)],
        "gn": [float(x) for x in re.findall(r"grad_norm.: ([0-9.eE+-]+)", s)],
    }


def report(name, q, b):
    n = min(len(q), len(b))
    if n == 0:
        print("  %-12s 无数据" % name)
        return
    devs = [(q[i] - b[i]) / abs(b[i]) for i in range(n) if abs(b[i]) > 1e-12]
    if not devs:
        print("  %-12s 基线全零，跳过" % name)
        return
    mean = sum(devs) / len(devs)
    var = sum((d - mean) ** 2 for d in devs) / max(len(devs) - 1, 1)
    sd = var ** 0.5
    print("  %-12s 带符号平均相对偏差 %+.4f%%  标准差 %.4f%%  n=%d"
          % (name, mean * 100, sd * 100, len(devs)))
    print("               首步 %+.4f%%   末步 %+.4f%%   绝对值最大 %.4f%%"
          % (devs[0] * 100, devs[-1] * 100, max(abs(d) for d in devs) * 100))


qt = sys.argv[1] if len(sys.argv) > 1 else "hif8_smoke"
bt = sys.argv[2] if len(sys.argv) > 2 else "bf16_ref20"
q, b = grab(qt), grab(bt)
print("=" * 68)
print("HiF8 臂 = %s   bf16 臂 = %s" % (qt, bt))
print("=" * 68)
if not q or not b:
    raise SystemExit("缺日志")
for key, nm in (("action", "action_loss"), ("ross", "ross_loss"), ("gn", "grad_norm")):
    report(nm, q[key], b[key])
print()
print("  逐步对照（action_loss）:")
n = min(len(q["action"]), len(b["action"]))
for i in range(n):
    x, y = q["action"][i], b["action"][i]
    print("    步%-3d hif8 %12.6f   bf16 %12.6f   %+.4f%%"
          % (i + 1, x, y, (x - y) / abs(y) * 100))
print()
print("  噪声底：同配置两把 bf16 已验证逐位一致（偏差恒为 0），")
print("  所以以上偏差全部来自量化，不需要再扣噪声。")

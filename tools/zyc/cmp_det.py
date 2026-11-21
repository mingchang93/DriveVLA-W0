# -*- coding: utf-8 -*-
"""比对同配置两把跑是否逐位一致。
判据用 grad_norm：HF Trainer 把 loss round 到 4 位小数（日志里就是 18.7769），
分辨率只有 1e-4，掩盖了真实分歧；grad_norm 是全精度打印的。
用法: python3 cmp_det.py <tagA> <tagB> [更多对...]"""
import io, os, re, sys
R = "/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs"
def grab(t):
    p = os.path.join(R, t, "train.log")
    if not os.path.exists(p): return None
    s = io.open(p, encoding="utf-8", errors="replace").read()
    return {"grad_norm": [float(x) for x in re.findall(r"grad_norm.: ([0-9.eE+-]+)", s)],
            "ross_loss": [float(x) for x in re.findall(r"ross_loss.: ([0-9.eE+-]+)", s)],
            "action_loss": [float(x) for x in re.findall(r"action_loss.: ([0-9.eE+-]+)", s)]}
def one(ta, tb):
    a, b = grab(ta), grab(tb)
    print("=" * 66); print("%s  vs  %s" % (ta, tb))
    if not a or not b or not a["grad_norm"] or not b["grad_norm"]:
        print("  缺日志或没跑出来"); return
    allsame = True
    for k in ("grad_norm", "ross_loss", "action_loss"):
        x, y = a[k], b[k]; n = min(len(x), len(y))
        if n == 0: print("  %-11s 无数据" % k); continue
        same = sum(1 for i in range(n) if x[i] == y[i])
        if same == n:
            print("  %-11s %d/%d 逐位完全相同  ✓" % (k, same, n)); continue
        allsame = False
        fd = next(i for i in range(n) if x[i] != y[i])
        rel = sorted(abs(x[i]-y[i])/max(abs(x[i]), 1e-12) for i in range(n) if x[i] != y[i])
        print("  %-11s %d/%d 相同  首次分歧第 %d 步  相对差 中位 %.2e 最大 %.2e"
              % (k, same, n, fd + 1, rel[len(rel)//2], rel[-1]))
    print("  grad_norm 逐条（全精度）:")
    for i in range(min(len(a["grad_norm"]), len(b["grad_norm"]))):
        u, v = a["grad_norm"][i], b["grad_norm"][i]
        print("    步%-2d %-20.10f %-20.10f %s" % (i+1, u, v,
              "相同" if u == v else "相对差 %.2e" % (abs(u-v)/abs(u))))
    print("  >>> %s" % ("逐位一致，确定性成立" if allsame else "仍有分歧"))
args = sys.argv[1:]
if not args:
    args = ["det5_eag_a", "det5_eag_b", "det5_ngc_a", "det5_ngc_b"]
for i in range(0, len(args) - 1, 2):
    one(args[i], args[i+1])

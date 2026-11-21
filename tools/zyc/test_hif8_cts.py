# -*- coding: utf-8 -*-
"""HiF8-CTS 线性层的数值自检：接进训练之前先确认前向/反向都对。

比的是「HiF8 量化矩阵乘」对「同 dtype 的普通矩阵乘」的相对偏差。
参照 fp32 结果作为真值，这样能分清偏差里有多少本来就是 bf16 自带的。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.quant.hif8_cts import (  # noqa: E402
    HiF8CTSLinear, convert_to_hif8_cts, hif8_available,
)

DEV = "npu:0"
torch.manual_seed(42)
torch.npu.manual_seed_all(42)

print("hif8 可用:", hif8_available())
if not hif8_available():
    raise SystemExit("环境缺原语，退出")


def rel(a, b):
    """a 相对 b 的相对误差（按 L2 范数）。"""
    a = a.float()
    b = b.float()
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def run_case(shape, K, M, tag):
    x_f32 = torch.randn(*shape, K, device=DEV, dtype=torch.float32)
    w_f32 = torch.randn(M, K, device=DEV, dtype=torch.float32) * (K ** -0.5)
    g_f32 = torch.randn(*shape, M, device=DEV, dtype=torch.float32)

    print("\n===== %s  x=%s  K=%d M=%d =====" % (tag, tuple(shape) + (K,), K, M))

    # --- fp32 真值 ---
    xf = x_f32.clone().requires_grad_(True)
    wf = w_f32.clone().requires_grad_(True)
    yf = F.linear(xf, wf)
    yf.backward(g_f32)

    # --- bf16 普通矩阵乘（基线）---
    xb = x_f32.bfloat16().clone().requires_grad_(True)
    wb = w_f32.bfloat16().clone().requires_grad_(True)
    yb = F.linear(xb, wb)
    yb.backward(g_f32.bfloat16())

    # --- bf16 + HiF8-CTS ---
    xh = x_f32.bfloat16().clone().requires_grad_(True)
    lin = nn.Linear(K, M, bias=False).to(DEV).bfloat16()
    with torch.no_grad():
        lin.weight.copy_(w_f32.bfloat16())
    qlin = HiF8CTSLinear.from_linear(lin)
    yh = qlin(xh)
    yh.backward(g_f32.bfloat16())

    print("  前向  bf16 vs fp32 : %.4e" % rel(yb, yf))
    print("  前向  hif8 vs fp32 : %.4e" % rel(yh, yf))
    print("  前向  hif8 vs bf16 : %.4e" % rel(yh, yb))
    print("  dx    bf16 vs fp32 : %.4e" % rel(xb.grad, xf.grad))
    print("  dx    hif8 vs fp32 : %.4e" % rel(xh.grad, xf.grad))
    print("  dw    bf16 vs fp32 : %.4e" % rel(wb.grad, wf.grad))
    print("  dw    hif8 vs fp32 : %.4e" % rel(qlin.weight.grad, wf.grad))
    print("  形状  y=%s dx=%s dw=%s" % (
        tuple(yh.shape), tuple(xh.grad.shape), tuple(qlin.weight.grad.shape)))
    assert yh.shape == yb.shape, "前向形状不符"
    assert xh.grad.shape == xb.grad.shape, "dx 形状不符"
    assert qlin.weight.grad.shape == wb.grad.shape, "dw 形状不符"
    assert torch.isfinite(yh).all(), "前向出现 NaN/Inf"
    assert torch.isfinite(xh.grad).all(), "dx 出现 NaN/Inf"
    assert torch.isfinite(qlin.weight.grad).all(), "dw 出现 NaN/Inf"


# 二维 / 三维（transformer 里就是 [B, L, C]）各测一组
run_case((4096,), 2048, 2048, "二维")
run_case((6, 1400), 2048, 2048, "三维 贴近真实 2VA 形状")
run_case((6, 1400), 2048, 6144, "三维 非方阵")

# --- bias 路径 ---
print("\n===== bias 路径 =====")
lin = nn.Linear(512, 512, bias=True).to(DEV).bfloat16()
q = HiF8CTSLinear.from_linear(lin)
x = torch.randn(8, 512, device=DEV, dtype=torch.bfloat16, requires_grad=True)
y = q(x)
y.sum().backward()
print("  bias.grad 形状 %s  有限 %s" % (tuple(q.bias.grad.shape), bool(torch.isfinite(q.bias.grad).all())))
assert q.bias.grad is not None, "bias 没有梯度"

# --- 整模替换 ---
print("\n===== 模块替换 =====")
m = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 64)),
).to(DEV).bfloat16()
conv, skip = convert_to_hif8_cts(m, apply=(), ignore=())
print("  换掉 %d 个，跳过 %d 个：%s" % (len(conv), len(skip), conv))
n_q = sum(1 for _ in m.modules() if isinstance(_, HiF8CTSLinear))
print("  实际 HiF8CTSLinear 数量: %d" % n_q)
assert n_q == 3, "替换数量不对"
out = m(torch.randn(4, 256, device=DEV, dtype=torch.bfloat16))
print("  前向通过，输出形状", tuple(out.shape))

print("\n全部通过")

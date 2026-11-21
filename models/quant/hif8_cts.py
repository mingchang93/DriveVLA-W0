# -*- coding: utf-8 -*-
"""HiF8 current-tensor-scaling（CTS）量化线性层。

移植自 MindSpeed 的
`mindspeed/te/pytorch/fp8/recipes/current_scaling_recipe.py` 里的
`TensorwiseMatMul`，去掉对 Megatron `get_args()` 和 TransformerEngine 的依赖，
改成可以直接在 HF Trainer + DeepSpeed 这条线上使用。

**CTS = Current Tensor Scaling**：每次前向/反向都用「当前张量」的 amax 现算
缩放因子（对应 DTS 用历史 amax 的延迟缩放）。粒度是 per-tensor。

量化只发生在矩阵乘的输入上：
  - 前向：x 和 w 各自量化成 hifloat8 再做 `npu_quant_matmul`
  - 反向：grads 量化后，与前向缓存的 w_q / x_q 相乘得到 dx / dw
主权重始终保持 bf16，优化器状态不受影响——这与 DeepSpeed 的 bf16 路径兼容。

动态范围按 MindSpeed 的做法分两档（见其 `FormatEnum`）：
  前向 x / w   `dst_type_max = 15`   (HIF8_15)
  反向 grads   `dst_type_max = 224`  (HIF8_224)
梯度分布比激活宽得多，所以给更大的范围。这两个值是照搬的，不是我们调的。

----------------------------------------------------------------------------
`HIF8_CTS_SYNC=1`：跨数据并行 rank 同步 amax（诊断用开关，默认关闭）
----------------------------------------------------------------------------
原实现每个 rank 从自己的 micro-batch 现算 amax，**各 rank 用各自的 scale**。
对固定划分格式（E4M3）这只是噪声；但 HiF8 是渐变精度格式，`dst_type_max=15`
的作用正是把分布主体放进 3-bit 尾数窗口，amax 一变整个张量在编码空间里的位置
就移、尾数精度跟着掉档（3-bit ε≈3.6% → 2-bit 7.2% → 1-bit 14.4%）。于是
「哪个 rank 的 micro-batch 里离群值更大，它的梯度就被量化得更粗」成为一个
**与数据相关的偏置**，all_reduce 平均不掉。

开关打开后改走「本地 amax -> 跨 rank ReduceOp.MAX -> 静态 npu_quantize」，
让所有 rank 用同一个 scale。约定沿用 hif8_dts.py：npu_quantize 是**除** scale，
npu_quant_matmul 再**乘**回同一个 scale。

两点与 DTS 的差别要知道：
  * DTS 把 all_reduce 放在 optimizer 的 post-step hook 里，刻意避开前向/反向；
    CTS 的语义要求现算，所以这里的集合通信落在前向/反向内部。各 rank 走同一张
    autograd 图、`need_dx`/`need_dw` 也一致，顺序相同不会死锁；但**重计算
    （gradient_checkpointing）会让前向跑两遍，开 GC 时不要用这个开关**。
  * 权重在纯 DP + ZeRO-1 下各 rank 本来就相同，那次规约是空操作，留着只为语义统一。

开销：每个 Linear 每步 3 次 1 元素规约（196 个 Linear ≈ 588 次），单次几十微秒，
相对 2.6 秒的步时不到 1%。
"""

import os

import torch
import torch.nn as nn

try:
    from models.quant import hif8_probe as _probe
except Exception:  # 探针缺失不影响训练
    _probe = None

try:
    import torch_npu
    _HIF8 = getattr(torch_npu, "hifloat8", None)
except Exception:  # 非 NPU 环境下允许 import，但不能用
    torch_npu = None
    _HIF8 = None

# 照搬 MindSpeed 的两档动态范围
FWD_DST_MAX = 15    # FormatEnum.HIF8_15
# 反向 dst_max。224 = 2^7.81 落在 HiF8 的 **2-bit** 尾数档（实测档位见
# hif8_bands.py：3-bit 档只覆盖 [2^-3, 2^4)）。CTS 的 scale 现算，顶端被
# 钉死在 dst_max 上，于是扛能量的大梯度常驻 2-bit、吃 7.2% 误差；DTS 用
# 跨 rank + 跨步 MAX 的 amax，顶端实测下沉到 2^4.3，能量主体进了 3-bit、
# 3.6% 误差。这个环境变量用来验证该归因（默认 224，与原实现逐字相同）。
BWD_DST_MAX = float(os.environ.get("HIF8_CTS_BWD_MAX", "224") or 224)

# 跨 rank 同步 amax 的开关。默认关闭 —— 关闭时代码路径与原实现逐字相同。
_MODE = os.environ.get("HIF8_CTS_SYNC", "0").strip().lower()
# "0"/""/"false"/"no" -> 原实现（npu_dynamic_quant，各 rank 本地 amax）
# "local"             -> 只换算子（npu_quantize + 本地 amax）。作为 "1" 的对照，
#                        把「换算子」这个变量单独摘出来。
# "1"/"true"/"yes"    -> 换算子 + 跨 rank amax MAX 同步
USE_QUANTIZE = _MODE in ("1", "true", "yes", "local")
SYNC_AMAX = _MODE in ("1", "true", "yes")
# amax 下界：npu_quantize 是除法，全零张量会产出 inf（npu_dynamic_quant 不需要这个）
_AMAX_EPS = 1e-12


def hif8_available() -> bool:
    ok = (
        torch_npu is not None
        and _HIF8 is not None
        and hasattr(torch_npu, "npu_dynamic_quant")
        and hasattr(torch_npu, "npu_quant_matmul")
    )
    if ok and USE_QUANTIZE:
        ok = hasattr(torch_npu, "npu_quantize")
    return ok


def _mm_kwargs():
    # hifloat8 走 npu_quant_matmul 时要显式告知两个输入的 dtype
    return {"x1_dtype": _HIF8, "x2_dtype": _HIF8}


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    """把 [..., K] 压成 [N, K]；已经是二维就原样返回。"""
    if t.dim() == 2:
        return t
    return t.reshape(-1, t.shape[-1])


def _quant(t: torch.Tensor, dst_max, role="?", label="?"):
    """量化到 hifloat8，返回 (码点, 反量化 scale)。

    默认走 npu_dynamic_quant（各 rank 自算 amax），与原实现一致。
    SYNC_AMAX 打开时走「本地 amax -> 跨 rank MAX -> npu_quantize」。
    探针（HIF8_PROBE）只读不改，关闭时零开销。
    """
    if not USE_QUANTIZE:
        q, s = torch_npu.npu_dynamic_quant(
            t, dst_type=_HIF8, quant_mode="pertensor", dst_type_max=dst_max
        )
        if _probe is not None and _probe.ENABLED:
            _probe.record(role, label, t, s, dst_max)
        return q, s

    amax = t.detach().abs().amax().to(torch.float32).reshape(1)
    if SYNC_AMAX and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX)
    amax = torch.clamp(amax, min=_AMAX_EPS)
    # scale 即 npu_quantize 所除、npu_quant_matmul 所乘的那个值
    scale = (amax / float(dst_max)).contiguous()
    q = torch_npu.npu_quantize(t, scale, None, _HIF8, -1, True)
    if _probe is not None and _probe.ENABLED:
        _probe.record(role, label, t, scale, dst_max)
    return q, scale


# ---- 只量化前向（HIF8_FWD_ONLY=1），反向走 bf16 ----
# 归属实验用：若只量化前向就几乎无损，则代价全在梯度量化上。
FWD_ONLY = os.environ.get("HIF8_FWD_ONLY", "0") not in ("0", "", "false", "False", "no")
_ONE_CACHE = {}


def _one(device):
    k = str(device)
    if k not in _ONE_CACHE:
        _ONE_CACHE[k] = torch.ones(1, dtype=torch.float32, device=device)
    return _ONE_CACHE[k]


def _dequant(q, s, out_dtype):
    """精确反量化。必须拆 scale：anti_quant(q, 很小的 s) 会在 fp16 里下溢，
    而 anti_quant(q, 1) 的格点值全部可精确表示，再在 float32 里乘 s。"""
    import torch_npu
    v = torch_npu.npu_anti_quant(q, _one(q.device))
    return (v.to(torch.float32) * s.to(torch.float32)).to(out_dtype)


class _HiF8CTSMatMul(torch.autograd.Function):
    """y = x @ w.T，三个矩阵乘的输入都按 per-tensor 现算缩放量化到 hifloat8。

    bias 不在这里处理：留给外层用普通加法，让 autograd 自然得到 bias 的梯度，
    省得在这里手写一遍求和。
    """

    @staticmethod
    def forward(ctx, x, weight, label="?"):
        x2 = _as_2d(x)
        mmk = _mm_kwargs()

        x_q, x_s = _quant(x2, FWD_DST_MAX, "x", label)
        w_q, w_s = _quant(weight, FWD_DST_MAX, "w", label)

        out = torch_npu.npu_quant_matmul(
            x_q, w_q.t(), w_s, pertoken_scale=x_s, output_dtype=x.dtype, **mmk
        )
        if x.dim() != 2:
            out = out.reshape(*x.shape[:-1], out.shape[-1])

        ctx.x_q, ctx.x_s, ctx.w_q, ctx.w_s = x_q, x_s, w_q, w_s
        ctx.out_dtype = x.dtype
        ctx.x_shape = tuple(x.shape)
        ctx.need_dx = x.requires_grad
        ctx.need_dw = weight.requires_grad
        ctx.label = label
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mmk = _mm_kwargs()
        g2 = _as_2d(grad_out)
        g_q, g_s = _quant(g2, BWD_DST_MAX, "g", getattr(ctx, "label", "?"))

        dx = dw = None
        if FWD_ONLY:
            # 反向不量化：用反量化后的 x/w 与未量化的 g 做 bf16 矩阵乘
            w_d = _dequant(ctx.w_q, ctx.w_s, ctx.out_dtype)
            dx_dw_g = g2.to(ctx.out_dtype)
            if ctx.need_dx:
                dx = torch.matmul(dx_dw_g, w_d)
                if len(ctx.x_shape) != 2:
                    dx = dx.reshape(*ctx.x_shape[:-1], dx.shape[-1])
            if ctx.need_dw:
                x_d = _dequant(ctx.x_q, ctx.x_s, ctx.out_dtype)
                dw = torch.matmul(dx_dw_g.t(), x_d)
            return dx, dw, None
        if ctx.need_dx:
            # 前向用的是 w_q.t()，所以这里用未转置的 w_q
            dx = torch_npu.npu_quant_matmul(
                g_q, ctx.w_q, ctx.w_s, pertoken_scale=g_s,
                output_dtype=ctx.out_dtype, **mmk
            )
            if len(ctx.x_shape) != 2:
                dx = dx.reshape(*ctx.x_shape[:-1], dx.shape[-1])
        if ctx.need_dw:
            dw = torch_npu.npu_quant_matmul(
                g_q.T, ctx.x_q, ctx.x_s, pertoken_scale=g_s,
                output_dtype=ctx.out_dtype, **mmk
            )
        return dx, dw, None


class HiF8CTSLinear(nn.Linear):
    """与 nn.Linear 等价的替换件，矩阵乘走 HiF8-CTS。

    权重和 bias 仍是原 dtype（bf16）的普通 Parameter，所以 DeepSpeed 的分片、
    优化器状态、checkpoint 保存全都照常工作——量化只发生在前向/反向的矩阵乘里。
    """

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "HiF8CTSLinear":
        new = cls(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            device="meta",
        )
        new.weight = lin.weight
        new.bias = lin.bias
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _HiF8CTSMatMul.apply(x, self.weight, getattr(self, "_hif8_label", "?"))
        if self.bias is not None:
            out = out + self.bias
        return out


# 量化范围：严格照客户规格（见 /mnt/sfs_turbo/zyc/hif8/hif8-cts-code-map.md 第 4 节）
#
#     quant_apply_modules:   ["model.language_model.layers.{*}"]
#     quant_ignored_modules: ["*lm_head", "*o_proj"]
#
# 即：只量化语言模型 28 层里的 q/k/v/gate/up/down_proj = 168 个；
#     o_proj 的 28 个排除；视觉塔的 104 个不量化；denoiser/vae 更不碰。
# 这不是我们的取舍，是对方定下的规格，改之前先确认。
DEFAULT_APPLY = ("model.language_model.layers.",)
DEFAULT_IGNORE = ("lm_head", "o_proj")
SMALL_LINEAR_DIM = 64   # 对应框架里的 _SMALL_LINEAR_DIM


def convert_to_hif8_cts(
    model: nn.Module,
    apply=DEFAULT_APPLY,
    ignore=DEFAULT_IGNORE,
    min_features: int = SMALL_LINEAR_DIM,
):
    """把符合范围的 nn.Linear 原地换成 HiF8CTSLinear。

    apply:         fqn 里含这些子串之一才量化（空元组表示不限制）
    ignore:        fqn 里含这些子串之一就排除，优先级高于 apply
    min_features:  in/out 任一小于它就跳过（小矩阵量化收益低、相对误差大）
    返回 (换掉的 fqn 列表, 被排除的 Linear fqn 列表)
    """
    if not hif8_available():
        raise RuntimeError("当前环境没有 torch_npu.hifloat8 / npu_dynamic_quant / npu_quant_matmul"
                           + ("（HIF8_CTS_SYNC=1 还需要 npu_quantize）" if SYNC_AMAX else ""))

    converted, skipped = [], []

    def _walk(mod, prefix):
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(p in full for p in ignore):
                    skipped.append(full)
                elif apply and not any(p in full for p in apply):
                    skipped.append(full)
                elif min(child.in_features, child.out_features) < min_features:
                    skipped.append(full)
                else:
                    _new = HiF8CTSLinear.from_linear(child)
                    _new._hif8_label = full
                    setattr(mod, name, _new)
                    converted.append(full)
            else:
                _walk(child, full)

    _walk(model, "")
    if converted and FWD_ONLY:
        # 必须 print：logger.info 到不了 train.log，开关就无从验证
        print("[HiF8-CTS] FWD_ONLY=1：只量化前向，反向用反量化的 x/w 做 bf16 "
              "矩阵乘，%d 个 Linear" % len(converted))
    if converted and (BWD_DST_MAX != 224.0 or FWD_DST_MAX != 15):
        # 必须 print 而不是 logger.info：后者到不了 train.log，开关就无从验证。
        print("[HiF8-CTS] dst_max 非默认：前向 %g 反向 %g（默认 15 / 224），"
              "%d 个 Linear" % (FWD_DST_MAX, BWD_DST_MAX, len(converted)))
    if SYNC_AMAX and converted:
        print("[HiF8-CTS] HIF8_CTS_SYNC=1：amax 跨 rank MAX 同步已启用，"
              "%d 个 Linear，每步 %d 次 1 元素规约" % (len(converted), 3 * len(converted)))
    return converted, skipped

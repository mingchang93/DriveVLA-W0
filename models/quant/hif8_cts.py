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
"""

import torch
import torch.nn as nn

try:
    import torch_npu
    _HIF8 = getattr(torch_npu, "hifloat8", None)
except Exception:  # 非 NPU 环境下允许 import，但不能用
    torch_npu = None
    _HIF8 = None

# 照搬 MindSpeed 的两档动态范围
FWD_DST_MAX = 15    # FormatEnum.HIF8_15
BWD_DST_MAX = 224   # FormatEnum.HIF8_224


def hif8_available() -> bool:
    return (
        torch_npu is not None
        and _HIF8 is not None
        and hasattr(torch_npu, "npu_dynamic_quant")
        and hasattr(torch_npu, "npu_quant_matmul")
    )


def _mm_kwargs():
    # hifloat8 走 npu_quant_matmul 时要显式告知两个输入的 dtype
    return {"x1_dtype": _HIF8, "x2_dtype": _HIF8}


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    """把 [..., K] 压成 [N, K]；已经是二维就原样返回。"""
    if t.dim() == 2:
        return t
    return t.reshape(-1, t.shape[-1])


class _HiF8CTSMatMul(torch.autograd.Function):
    """y = x @ w.T，三个矩阵乘的输入都按 per-tensor 现算缩放量化到 hifloat8。

    bias 不在这里处理：留给外层用普通加法，让 autograd 自然得到 bias 的梯度，
    省得在这里手写一遍求和。
    """

    @staticmethod
    def forward(ctx, x, weight):
        x2 = _as_2d(x)
        mmk = _mm_kwargs()

        x_q, x_s = torch_npu.npu_dynamic_quant(
            x2, dst_type=_HIF8, quant_mode="pertensor", dst_type_max=FWD_DST_MAX
        )
        w_q, w_s = torch_npu.npu_dynamic_quant(
            weight, dst_type=_HIF8, quant_mode="pertensor", dst_type_max=FWD_DST_MAX
        )

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
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mmk = _mm_kwargs()
        g2 = _as_2d(grad_out)
        g_q, g_s = torch_npu.npu_dynamic_quant(
            g2, dst_type=_HIF8, quant_mode="pertensor", dst_type_max=BWD_DST_MAX
        )

        dx = dw = None
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
        return dx, dw


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
        out = _HiF8CTSMatMul.apply(x, self.weight)
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
        raise RuntimeError("当前环境没有 torch_npu.hifloat8 / npu_dynamic_quant / npu_quant_matmul")

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
                    setattr(mod, name, HiF8CTSLinear.from_linear(child))
                    converted.append(full)
            else:
                _walk(child, full)

    _walk(model, "")
    return converted, skipped

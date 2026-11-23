#!/usr/bin/env python3
"""Parity checks for Qwen3-VL Ascend fused RMSNorm, RoPE, gate/up, and QKV."""

import copy
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise RuntimeError("torch_npu is required for the Qwen3-VL fused-op check") from exc

import reference.transformers.src.transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl
from reference.transformers.src.transformers.models.qwen2_5_vl import npu_fused_ops


def assert_close(actual, expected, name, rtol=1e-2, atol=1e-2):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    print(f"PASS {name}")


def reference_rms_norm(x, weight, eps):
    xf = x.float()
    return (weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def check_rms_norm(device):
    torch.manual_seed(1)
    x_actual = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_reference = x_actual.detach().clone().requires_grad_(True)
    weight_actual = torch.randn(64, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight_reference = weight_actual.detach().clone().requires_grad_(True)

    actual = npu_fused_ops.fused_rms_norm(x_actual, weight_actual, 1e-6)
    expected = reference_rms_norm(x_reference, weight_reference, 1e-6)
    assert actual is not None
    assert_close(actual, expected, "fused RMSNorm forward")
    actual.float().pow(2).mean().backward()
    expected.float().pow(2).mean().backward()
    assert_close(x_actual.grad, x_reference.grad, "fused RMSNorm input grad")
    assert_close(weight_actual.grad, weight_reference.grad, "fused RMSNorm weight grad")


def check_rope(device):
    torch.manual_seed(2)
    q_actual = torch.randn(2, 4, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    q_reference = q_actual.detach().clone().requires_grad_(True)
    cos = torch.randn(2, 1, 16, 64, device=device, dtype=torch.bfloat16)
    sin = torch.randn_like(cos)

    actual = npu_fused_ops.fused_rotary_mul(q_actual, cos, sin)
    expected = q_reference.float() * cos.float() + rotate_half(q_reference.float()) * sin.float()
    assert actual is not None
    assert_close(actual, expected, "fused RoPE forward")
    actual.float().pow(2).mean().backward()
    expected.float().pow(2).mean().backward()
    assert_close(q_actual.grad, q_reference.grad, "fused RoPE input grad")


def check_gate_up(device):
    config = SimpleNamespace(hidden_size=64, intermediate_size=128, hidden_act="silu")
    actual_mlp = qwen3_vl.Qwen3VLTextMLP(config).to(device=device, dtype=torch.bfloat16)
    reference_mlp = copy.deepcopy(actual_mlp)
    x_actual = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_reference = x_actual.detach().clone().requires_grad_(True)

    qwen3_vl._FUSED_GATEUP = True
    actual = actual_mlp(x_actual)
    qwen3_vl._FUSED_GATEUP = False
    old_swiglu = npu_fused_ops._ENABLE_SWIGLU
    npu_fused_ops._ENABLE_SWIGLU = False
    expected = reference_mlp(x_reference)
    npu_fused_ops._ENABLE_SWIGLU = old_swiglu

    assert_close(actual, expected, "fused gate/up forward")
    actual.float().pow(2).mean().backward()
    expected.float().pow(2).mean().backward()
    assert_close(x_actual.grad, x_reference.grad, "fused gate/up input grad")
    for (actual_name, actual_parameter), (_, expected_parameter) in zip(
        actual_mlp.named_parameters(), reference_mlp.named_parameters(), strict=True
    ):
        assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            f"fused gate/up grad {actual_name}",
        )


def check_qkv(device):
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        _attn_implementation="eager",
    )
    actual_attention = qwen3_vl.Qwen3VLTextAttention(config, layer_idx=0).to(
        device=device, dtype=torch.bfloat16
    )
    reference_attention = copy.deepcopy(actual_attention)
    hidden_actual = torch.randn(2, 17, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    hidden_reference = hidden_actual.detach().clone().requires_grad_(True)
    cos = torch.ones(2, 17, 16, device=device, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    position_embeddings = (cos, sin)

    qwen3_vl._FUSED_QKV = True
    actual = actual_attention(
        hidden_states=hidden_actual,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )[0]
    qwen3_vl._FUSED_QKV = False
    expected = reference_attention(
        hidden_states=hidden_reference,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )[0]

    assert_close(actual, expected, "fused QKV forward")
    actual.float().pow(2).mean().backward()
    expected.float().pow(2).mean().backward()
    assert_close(hidden_actual.grad, hidden_reference.grad, "fused QKV input grad")
    for (actual_name, actual_parameter), (_, expected_parameter) in zip(
        actual_attention.named_parameters(), reference_attention.named_parameters(), strict=True
    ):
        assert_close(actual_parameter.grad, expected_parameter.grad, f"fused QKV grad {actual_name}")


def main():
    required = {
        "ROSS_NPU_RMSNORM": "1",
        "ROSS_NPU_ROPE": "1",
        "ROSS_FUSED_GATEUP": "1",
        "ROSS_FUSED_QKV": "1",
    }
    missing = [name for name, value in required.items() if os.getenv(name) != value]
    if missing:
        raise RuntimeError(f"enable these flags before import: {', '.join(missing)}")
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required")

    device = torch.device(f"npu:{torch.npu.current_device()}")
    torch.npu.set_device(device)
    check_rms_norm(device)
    check_rope(device)
    check_gate_up(device)
    check_qkv(device)
    print("PASS all Qwen3-VL fused-op parity checks")


if __name__ == "__main__":
    main()

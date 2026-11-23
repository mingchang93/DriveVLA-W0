"""Optional Ascend-native fused ops for the Qwen2.5-VL / ROSS training path.

Each helper returns ``None`` when the op is unavailable or its env flag is off,
so every call site keeps its original PyTorch path as the fallback and the same
code still runs on CUDA/CPU.

Flags (set before the process starts):

    ROSS_NPU_RMSNORM=1 -> torch_npu.npu_rms_norm
    ROSS_NPU_SWIGLU=1 -> torch_npu.npu_swiglu
    ROSS_NPU_ROPE=1 -> torch_npu.npu_rotary_mul
    ROSS_NPU_ADD_RMSNORM=1 -> torch_npu.npu_add_rms_norm
    ROSS_NPU_ADD_RMSNORM_V2=1 -> torch_npu.npu_add_rms_norm_v2_functional
    ROSS_NPU_GN_SILU=1 -> torch_npu.npu_group_norm_silu
    ROSS_NPU_SILU=1 -> torch_npu.npu_silu
"""

import os

import torch

try:
    import torch_npu
except ImportError:  # CUDA/CPU builds
    torch_npu = None


def _npu_ready(op_name: str) -> bool:
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        return False
    return hasattr(torch_npu, op_name)


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _npu_tensor(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and x.device.type == "npu"


_ENABLE_RMSNORM = _flag("ROSS_NPU_RMSNORM")
_ENABLE_SWIGLU = _flag("ROSS_NPU_SWIGLU")
_ENABLE_ROPE = _flag("ROSS_NPU_ROPE")
_ENABLE_ADD_RMS = _flag("ROSS_NPU_ADD_RMSNORM")
_ENABLE_ADD_RMS_V2 = _flag("ROSS_NPU_ADD_RMSNORM_V2")
_ENABLE_GN_SILU = _flag("ROSS_NPU_GN_SILU")
_ENABLE_SILU = _flag("ROSS_NPU_SILU")

_HAS_RMSNORM = _npu_ready("npu_rms_norm")
_HAS_SWIGLU = _npu_ready("npu_swiglu")
_HAS_ROTARY = _npu_ready("npu_rotary_mul")
_HAS_ADD_RMS = _npu_ready("npu_add_rms_norm")
_HAS_ADD_RMS_V2 = _npu_ready("npu_add_rms_norm_v2_functional")
_HAS_GN_SILU = _npu_ready("npu_group_norm_silu")
_HAS_SILU = _npu_ready("npu_silu")

_BACKWARD_OK = {}
_WARNED = set()


def _warn_once(name: str, exc: Exception) -> None:
    if name not in _WARNED:
        _WARNED.add(name)
        print(f"[fused] {name} disabled at runtime: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")


def probe_backward_support(device_index: int = 0) -> dict:
    """Probe the optional fusions whose training support varies by build."""
    global _BACKWARD_OK
    if _BACKWARD_OK:
        return _BACKWARD_OK
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        _BACKWARD_OK = {"add_rms_norm": False, "group_norm_silu": False, "silu": False}
        return _BACKWARD_OK

    device = torch.device("npu", int(device_index))

    def _try(name, fn):
        try:
            x = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
            out = fn(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out.float().sum().backward()
            if x.grad is None or not bool(torch.isfinite(x.grad).all()):
                print(f"[fused] {name} backward unavailable: no gradient written")
                return False
            return True
        except Exception as exc:
            print(f"[fused] {name} backward unavailable: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            return False

    if _ENABLE_ADD_RMS and _HAS_ADD_RMS:
        gamma = torch.ones(64, device=device, dtype=torch.bfloat16)
        _BACKWARD_OK["add_rms_norm"] = _try(
            "add_rms_norm", lambda x: torch_npu.npu_add_rms_norm(x, x, gamma, 1e-6)
        )
    if _ENABLE_GN_SILU and _HAS_GN_SILU:
        gn_w = torch.ones(8, device=device, dtype=torch.bfloat16)
        gn_b = torch.zeros(8, device=device, dtype=torch.bfloat16)
        _BACKWARD_OK["group_norm_silu"] = _try(
            "group_norm_silu",
            lambda x: torch_npu.npu_group_norm_silu(x.reshape(2, 8, 8, 8), gn_w, gn_b, 2),
        )
    if _ENABLE_SILU and _HAS_SILU:
        _BACKWARD_OK["silu"] = _try("silu", lambda x: torch_npu.npu_silu(x))
    return _BACKWARD_OK


def backward_status() -> dict:
    return dict(_BACKWARD_OK)


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    if not (_ENABLE_RMSNORM and _HAS_RMSNORM and _npu_tensor(x)):
        return None
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor):
    if not (_ENABLE_SWIGLU and _HAS_SWIGLU and _npu_tensor(gate)):
        return None
    return torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)


def fused_rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if not (_ENABLE_ROPE and _HAS_ROTARY and _npu_tensor(x)):
        return None
    return torch_npu.npu_rotary_mul(x, cos, sin)


class _FusedAddRMSNormV2(torch.autograd.Function):
    """Fused add+RMSNorm with the missing gradient implemented explicitly."""

    @staticmethod
    def forward(ctx, x1, x2, weight, eps):
        rstd, out, summed = torch_npu.npu_add_rms_norm_v2_functional(x1, x2, weight, eps)
        ctx.save_for_backward(summed, rstd, weight)
        return out, summed

    @staticmethod
    def backward(ctx, dout, dsum):
        summed, rstd, weight = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(summed)
        if dsum is None:
            dsum = torch.zeros_like(summed)
        s = summed.float()
        r = rstd.float()
        w = weight.float()
        do = dout.float()
        xhat = s * r
        dw = (do * xhat).sum(dim=tuple(range(do.dim() - 1))).to(weight.dtype)
        dxhat = do * w
        ds = r * (dxhat - xhat * (dxhat * xhat).mean(dim=-1, keepdim=True))
        ds = (ds + dsum.float()).to(summed.dtype)
        return ds, ds, dw, None


def fused_add_rms_norm(x1: torch.Tensor, x2: torch.Tensor, weight: torch.Tensor, eps: float):
    """Return ``(normed(x1 + x2), x1 + x2)`` or ``None`` for the reference path."""
    if _ENABLE_ADD_RMS_V2 and _HAS_ADD_RMS_V2 and _npu_tensor(x1):
        if x1.dtype != x2.dtype:
            x2 = x2.to(x1.dtype)
        if weight.dtype != x1.dtype:
            weight = weight.to(x1.dtype)
        try:
            return _FusedAddRMSNormV2.apply(x1, x2, weight, eps)
        except Exception as exc:
            _warn_once("add_rms_norm_v2", exc)
            return None
    if not (
        _ENABLE_ADD_RMS
        and _HAS_ADD_RMS
        and _BACKWARD_OK.get("add_rms_norm", False)
        and _npu_tensor(x1)
    ):
        return None
    if x1.dtype != x2.dtype:
        x2 = x2.to(x1.dtype)
    if weight.dtype != x1.dtype:
        weight = weight.to(x1.dtype)
    try:
        normed, _rstd, summed = torch_npu.npu_add_rms_norm(x1, x2, weight, eps)
    except Exception as exc:
        _warn_once("add_rms_norm", exc)
        return None
    return normed, summed


def fused_silu(x: torch.Tensor):
    if not (_ENABLE_SILU and _HAS_SILU and _BACKWARD_OK.get("silu", False) and _npu_tensor(x)):
        return None
    try:
        return torch_npu.npu_silu(x)
    except Exception as exc:
        _warn_once("silu", exc)
        return None


def fused_group_norm_silu(x, weight, bias, num_groups: int, eps: float = 1e-5):
    if not (_ENABLE_GN_SILU and _HAS_GN_SILU and _BACKWARD_OK.get("group_norm_silu", False)):
        return None
    try:
        out = torch_npu.npu_group_norm_silu(x, weight, bias, num_groups, eps)
    except Exception as exc:
        _warn_once("group_norm_silu", exc)
        return None
    return out[0] if isinstance(out, (tuple, list)) else out


def fuse_unet_group_norm_silu(root) -> int:
    """Patch eligible diffusers ResNet blocks in place; state-dict keys are unchanged."""
    if not (_ENABLE_GN_SILU and _HAS_GN_SILU and _BACKWARD_OK.get("group_norm_silu", False)):
        return 0

    import torch.nn as nn

    def _patch_norm(norm, block):
        orig = norm.forward

        def forward(x, _norm=norm, _block=block, _orig=orig):
            _block._fused_gn_input = (x, _norm)
            return _orig(x)

        norm.forward = forward

    def _patch_silu(silu, block):
        orig = silu.forward

        def forward(x, _block=block, _orig=orig):
            pending = getattr(_block, "_fused_gn_input", None)
            _block._fused_gn_input = None
            if pending is not None:
                inp, norm = pending
                out = fused_group_norm_silu(inp, norm.weight, norm.bias, norm.num_groups, norm.eps)
                if out is not None:
                    return out
            return _orig(x)

        silu.forward = forward

    patched = 0
    for block in root.modules():
        if getattr(block, "time_embedding_norm", "default") != "default":
            continue
        norm1, norm2 = getattr(block, "norm1", None), getattr(block, "norm2", None)
        silu = getattr(block, "nonlinearity", None)
        if not (
            isinstance(norm1, nn.GroupNorm)
            and isinstance(norm2, nn.GroupNorm)
            and isinstance(silu, nn.SiLU)
        ):
            continue
        _patch_norm(norm1, block)
        _patch_norm(norm2, block)
        _patch_silu(silu, block)
        patched += 1
    return patched

#!/usr/bin/env python3
"""Per-step FLOPs counter for the Qwen2.5-VL / ROSS VLA training step.

Counts matmul work in the forward pass with module hooks:

    nn.Linear         2 * rows * in_features * out_features
    nn.Conv1d/2d/3d   2 * output.numel() * (in_channels // groups) * prod(kernel)
    attention         4 * B * H * Lq * Lk * D   (halved when is_causal)
                      or 4 * H * D * sum(L_i^2) when cu_seqlens segments it

Attention is costed from the shapes the module receives, so the number is the
same whether FA2, SDPA or eager executes it. Vision attention packs the whole
batch into one 2-D sequence and splits it by cu_seqlens; that path costs the
sum of per-segment dense attention rather than one dense matrix.

Trainable and frozen parameters accumulate into separate buckets so the caller
can apply the standard 3x / 1x forward+backward rule. Counters are per rank:
one rank is enough, because every rank runs the same per-device batch.

Device-agnostic by design (no torch_npu import). See
docs/superpowers/specs/2026-10-24-flops-counting-design.md.

Self-check: python utils/flops.py
"""

import os

import torch
import torch.nn as nn

__all__ = ["FlopsCounter"]

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _head_dim(module, heads):
    hd = getattr(module, "head_dim", None)
    if hd:
        return int(hd)
    for attr in ("q_proj", "to_q", "qkv"):
        proj = getattr(module, attr, None)
        if isinstance(proj, nn.Linear):
            out = proj.out_features // 3 if attr == "qkv" else proj.out_features
            return out // int(heads)
    return None


def _is_attention(module):
    if "attention" not in type(module).__name__.lower():
        return False
    heads = getattr(module, "num_heads", None) or getattr(module, "heads", None)
    return bool(heads) and _head_dim(module, heads) is not None


class FlopsCounter:
    """Accumulates forward FLOPs between reset() and snapshot()."""

    def __init__(self):
        self.mm_trainable = 0.0
        self.mm_frozen = 0.0
        # Linear-only buckets: the MFU convention of 408235117's Emu3 port
        # (nn.Linear hooks only, no Conv/attention). See npu-port.md Step 37.
        self.linear_trainable = 0.0
        self.linear_frozen = 0.0
        self.attn_trainable = 0.0       # causal-adjusted
        self.attn_frozen = 0.0
        self.attn_full_trainable = 0.0  # no masking adjustment
        self.attn_full_frozen = 0.0
        self.cumulative = 0.0           # per-card FLOPs over the whole run
        self.by_class = {}              # raw forward FLOPs per module class
        self._trainable = {}
        self._handles = []
        self._attached = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def attached(self):
        return self._attached

    def attach(self, model):
        """Register hooks once. Call after accelerator.prepare()/unwrap."""
        if self._attached:
            return
        seen = {}
        for module in model.modules():
            if isinstance(module, (nn.Linear,) + _CONV_TYPES):
                name = type(module).__name__
                seen[name] = seen.get(name, 0) + 1
                self._trainable[id(module)] = any(
                    p.requires_grad for p in module.parameters(recurse=False)
                )
                self._handles.append(module.register_forward_hook(self._module_hook))
            elif _is_attention(module):
                name = type(module).__name__
                key = f"{name}(is_causal={getattr(module, 'is_causal', None)!r})"
                seen[key] = seen.get(key, 0) + 1
                self._trainable[id(module)] = any(
                    p.requires_grad for p in module.parameters(recurse=True)
                )
                self._handles.append(
                    module.register_forward_hook(self._attn_hook, with_kwargs=True)
                )
        self._attached = True
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[flops] hooked modules: {seen}", flush=True)

    def reset(self):
        self.mm_trainable = self.mm_frozen = 0.0
        self.linear_trainable = self.linear_frozen = 0.0
        self.attn_trainable = self.attn_frozen = 0.0
        self.attn_full_trainable = self.attn_full_frozen = 0.0

    def accumulate(self):
        self.cumulative += self.flops_total

    # -- hooks -------------------------------------------------------------
    def _add(self, module, flops, is_attn, causal_scale=1.0):
        name = type(module).__name__
        self.by_class[name] = self.by_class.get(name, 0.0) + flops
        trainable = self._trainable.get(id(module), False)
        if is_attn:
            if trainable:
                self.attn_trainable += flops * causal_scale
                self.attn_full_trainable += flops
            else:
                self.attn_frozen += flops * causal_scale
                self.attn_full_frozen += flops
        elif trainable:
            self.mm_trainable += flops
        else:
            self.mm_frozen += flops

    def _module_hook(self, module, inputs, output):
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            if not in_features:
                return
            rows = inputs[0].numel() // in_features
            flops = 2.0 * rows * in_features * module.out_features
            if self._trainable.get(id(module), False):
                self.linear_trainable += flops
            else:
                self.linear_frozen += flops
        else:
            if not torch.is_tensor(output):
                return
            kernel = 1
            for size in module.kernel_size:
                kernel *= size
            flops = 2.0 * output.numel() * (module.in_channels // module.groups) * kernel
        self._add(module, flops, is_attn=False)

    def _attn_hook(self, module, args, kwargs, output):
        hidden = None
        if args and torch.is_tensor(args[0]):
            hidden = args[0]
        elif kwargs and torch.is_tensor(kwargs.get("hidden_states")):
            # The Qwen decoder calls attention with hidden_states= as a keyword.
            hidden = kwargs["hidden_states"]
        if hidden is None:
            return
        if hidden.dim() == 3:
            batch, lq = hidden.shape[0], hidden.shape[1]
        elif hidden.dim() == 2:          # Qwen vision: (seq_len, C)
            batch, lq = 1, hidden.shape[0]
        else:
            return

        heads = getattr(module, "num_heads", None) or getattr(module, "heads", None)
        head_dim = _head_dim(module, heads) if heads else None
        if not heads or not head_dim:
            return

        # Vision packs the batch into one sequence and splits it by cu_seqlens
        # (the block passes it as a keyword). Sum of per-segment dense
        # attention, not one dense matrix over the concatenation.
        cu = (kwargs or {}).get("cu_seqlens")
        if cu is None and len(args) > 1 and torch.is_tensor(args[1]) and args[1].dim() == 1:
            cu = args[1]

        if cu is not None and cu.numel() >= 2:
            lens = cu[1:].to(torch.int64) - cu[:-1].to(torch.int64)
            sum_sq = int((lens * lens).sum().item())
            flops = 4.0 * int(heads) * head_dim * sum_sq
        else:
            encoder = (kwargs or {}).get("encoder_hidden_states")
            if encoder is not None and torch.is_tensor(encoder):
                lk = encoder.shape[1] if encoder.dim() == 3 else encoder.shape[0]
            else:
                lk = lq
            flops = 4.0 * batch * int(heads) * lq * lk * head_dim
        causal = bool(getattr(module, "is_causal", False))
        self._add(module, flops, is_attn=True, causal_scale=0.5 if causal else 1.0)

    # -- results -----------------------------------------------------------
    @property
    def flops_linear(self):
        """Linear-only training FLOPs (3x trainable + 1x frozen)."""
        return 3.0 * self.linear_trainable + self.linear_frozen

    @property
    def flops_mm(self):
        return 3.0 * self.mm_trainable + self.mm_frozen

    @property
    def flops_attn(self):
        return 3.0 * self.attn_trainable + self.attn_frozen

    @property
    def flops_attn_full(self):
        return 3.0 * self.attn_full_trainable + self.attn_full_frozen

    @property
    def flops_total(self):
        return self.flops_mm + self.flops_attn

    def snapshot(self):
        return {
            "flops_linear": self.flops_linear,
            "flops_mm": self.flops_mm,
            "flops_attn": self.flops_attn,
            "flops_attn_full": self.flops_attn_full,
            "flops_total": self.flops_total,
        }


# ---------------------------------------------------------------------------
# Self-check: python utils/flops.py
# ---------------------------------------------------------------------------
def _selfcheck():
    # Linear(4 -> 8), input (2, 3, 4): rows = 6
    #   flops = 2 * 6 * 4 * 8 = 384
    c = FlopsCounter()
    lin = nn.Linear(4, 8)
    seq = nn.Sequential(lin)
    c.attach(seq)
    c.reset()
    seq(torch.zeros(2, 3, 4))
    assert c.flops_mm == 3.0 * 384, c.flops_mm

    # Frozen Linear lands in the 1x bucket
    c = FlopsCounter()
    lin = nn.Linear(4, 8)
    lin.weight.requires_grad_(False)
    lin.bias.requires_grad_(False)
    seq = nn.Sequential(lin)
    c.attach(seq)
    c.reset()
    seq(torch.zeros(2, 3, 4))
    assert c.flops_mm == 384, c.flops_mm

    # Conv3d(3 -> 5, k=2), input (2, 3, 4, 4, 4) -> output (2, 5, 3, 3, 3)
    #   numel = 270, flops = 2 * 270 * 3 * 8 = 12960
    c = FlopsCounter()
    conv = nn.Conv3d(3, 5, kernel_size=2)
    seq = nn.Sequential(conv)
    c.attach(seq)
    c.reset()
    seq(torch.zeros(2, 3, 4, 4, 4))
    assert c.flops_mm == 3.0 * 12960, c.flops_mm

    class FakeAttention(nn.Module):
        def __init__(self, heads, head_dim, causal):
            super().__init__()
            self.heads = heads
            self.head_dim = head_dim
            self.is_causal = causal
            self.to_q = nn.Linear(heads * head_dim, heads * head_dim)

        def forward(self, hidden_states, cu_seqlens=None, encoder_hidden_states=None):
            return self.to_q(hidden_states)

    def attention_flops(causal, shape, encoder_shape=None, cu_seqlens=None):
        c = FlopsCounter()
        attn = FakeAttention(heads=3, head_dim=5, causal=causal)
        c.attach(attn)
        c.reset()
        kwargs = {}
        if encoder_shape is not None:
            kwargs["encoder_hidden_states"] = torch.zeros(*encoder_shape)
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int32)
        attn(torch.zeros(*shape), **kwargs)
        return c.flops_attn_full, c.flops_attn

    # Dense causal: 4 * 2 * 3 * 4 * 4 * 5 = 1920 full, 960 causal-adjusted
    full, causal_adj = attention_flops(True, (2, 4, 15))
    assert full == 3.0 * 1920, full
    assert causal_adj == 3.0 * 960, causal_adj

    # Dense non-causal: both 1920
    full, causal_adj = attention_flops(False, (2, 4, 15))
    assert full == 3.0 * 1920 and causal_adj == 3.0 * 1920, (full, causal_adj)

    # Cross-attention: Lk = 6 -> 4 * 2 * 3 * 4 * 6 * 5 = 2880
    full, causal_adj = attention_flops(False, (2, 4, 15), encoder_shape=(2, 6, 15))
    assert full == 3.0 * 2880, full

    # 2-D input without segments: B=1, Lq=Lk=6 -> 4 * 1 * 3 * 6 * 6 * 5 = 2160
    full, causal_adj = attention_flops(False, (6, 15))
    assert full == 3.0 * 2160, full

    # Keyword call: the Qwen decoder passes hidden_states= to the attention
    c = FlopsCounter()
    attn = FakeAttention(heads=3, head_dim=5, causal=True)
    c.attach(attn)
    c.reset()
    attn(hidden_states=torch.zeros(2, 4, 15))
    assert c.flops_attn == 3.0 * 960, c.flops_attn
    assert c.flops_attn_full == 3.0 * 1920, c.flops_attn_full

    # Segmented (vision): cu_seqlens [0,4,6] over 6 tokens
    #   sum L^2 = 4^2 + 2^2 = 20 -> 4 * 1 * 3 * 5 * 20 = 1200  (not L^2 = 36)
    full, causal_adj = attention_flops(False, (6, 15), cu_seqlens=[0, 4, 6])
    assert full == 3.0 * 1200, full
    assert causal_adj == 3.0 * 1200, causal_adj

    # Total = mm + causal attention
    c = FlopsCounter()
    c.mm_trainable = 1.0
    c.mm_frozen = 2.0
    c.attn_trainable = 4.0
    c.attn_frozen = 8.0
    assert c.flops_mm == 5.0 and c.flops_attn == 20.0 and c.flops_total == 25.0
    c.accumulate()
    c.accumulate()
    assert c.cumulative == 50.0

    print("flops self-check: OK")


if __name__ == "__main__":
    _selfcheck()

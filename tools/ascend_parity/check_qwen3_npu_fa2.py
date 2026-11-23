#!/usr/bin/env python3
"""Parity check for the opt-in Qwen3-VL Ascend FA2 path."""

import copy
import os

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise RuntimeError("torch_npu is required for the Qwen3-VL FA2 parity check") from exc

import reference.transformers.src.transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl
from reference.transformers.src.transformers.integrations.npu_flash_attention import (
    ATTN_MASK_NPU_CACHE,
    get_attn_mask_npu,
    npu_flash_attn_func,
)
from reference.transformers.src.transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
)


def make_config():
    return Qwen3VLConfig(
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 4096,
            "use_cache": False,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": False,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 8,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        },
        image_token_id=15,
        video_token_id=14,
        vision_start_token_id=13,
        vision_end_token_id=12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def make_model(attn_implementation):
    config = make_config()
    config._attn_implementation = attn_implementation
    return qwen3_vl.Qwen3VLForConditionalGeneration(config).eval()


def run(model, input_ids, labels):
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        use_cache=False,
    )
    output.loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output, grads


def check_adapter(device):
    torch.manual_seed(11)
    lens = (37, 2048, 2053)
    for seq_len in lens:
        mask = get_attn_mask_npu(device, seq_len)
        assert mask.shape == (2048, 2048)
        assert mask.dtype == torch.bool
    assert len(ATTN_MASK_NPU_CACHE) >= 1

    seq_len = 2053
    query = torch.randn(1, seq_len, 2, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    reference = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    actual = npu_flash_attn_func(query, key, value, causal=True)
    reference.float().sum().backward()
    reference_grads = tuple(tensor.grad.clone() for tensor in (query, key, value))
    query.grad = key.grad = value.grad = None
    actual.float().sum().backward()
    actual_grads = tuple(tensor.grad.clone() for tensor in (query, key, value))

    torch.testing.assert_close(actual.float(), reference.float(), rtol=3e-3, atol=3e-3)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual_grad.float(), reference_grad.float(), rtol=5e-3, atol=5e-3)


def main():
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required for the Qwen3-VL FA2 parity check")
    device = torch.device(f"npu:{torch.npu.current_device()}")
    torch.npu.set_device(device)
    check_adapter(device)

    torch.manual_seed(7)
    sdpa = make_model("sdpa").to(device)
    fa2 = make_model("flash_attention_2").to(device)
    fa2.load_state_dict(copy.deepcopy(sdpa.state_dict()))

    input_ids = torch.tensor(
        [[3 + (index % 8) for index in range(65)] + [2]],
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[:, :-8] = -100

    expected, expected_grads = run(sdpa, input_ids, labels)
    actual, actual_grads = run(fa2, input_ids, labels)
    torch.testing.assert_close(actual.loss.float(), expected.loss.float(), rtol=5e-3, atol=5e-3)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        torch.testing.assert_close(
            actual_grads[name].float(),
            expected_grads[name].float(),
            rtol=1e-2,
            atol=1e-2,
        )

    assert os.getenv("ROSS_NPU_FA2", "0") == "1"
    print("PASS Qwen3-VL NPU FA2 mask, loss and gradient parity")


if __name__ == "__main__":
    main()

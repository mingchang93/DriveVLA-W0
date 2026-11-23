#!/usr/bin/env python3
"""Parity check for the opt-in Qwen3-VL valid-token CE path."""

import copy
import os

import torch

import reference.transformers.src.transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl
from reference.transformers.src.transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


def make_model():
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
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
    return qwen3_vl.Qwen3VLForConditionalGeneration(config).eval()


def run(model, input_ids, labels, logits_to_keep=0):
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        logits_to_keep=logits_to_keep,
        use_cache=False,
    )
    output.loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output, grads


def forward(model, input_ids, labels, logits_to_keep=0):
    return model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        logits_to_keep=logits_to_keep,
        use_cache=False,
    )


def main():
    enabled = os.getenv("ROSS_VALID_CE", "0") == "1"
    configured = getattr(qwen3_vl, "_VALID_TOKEN_CE", False)
    assert configured == enabled, f"ROSS_VALID_CE={int(enabled)} was not applied at module import"

    torch.manual_seed(7)
    candidate = make_model()
    reference = copy.deepcopy(candidate)
    input_ids = torch.tensor([[1, 3, 4, 5, 6]], dtype=torch.long)
    labels = torch.tensor([[-100, 3, -100, 5, 6]], dtype=torch.long)

    qwen3_vl._VALID_TOKEN_CE = False
    expected, expected_grads = run(reference, input_ids, labels)

    qwen3_vl._VALID_TOKEN_CE = enabled
    actual, actual_grads = run(candidate, input_ids, labels)
    assert (actual.logits is None) == enabled
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-5, atol=1e-6)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=1e-5, atol=1e-6)

    if enabled:
        qwen3_vl._VALID_TOKEN_CE = False
        empty_reference, _ = run(copy.deepcopy(candidate), input_ids, torch.full_like(labels, -100))
        qwen3_vl._VALID_TOKEN_CE = True
        empty_actual, _ = run(copy.deepcopy(candidate), input_ids, torch.full_like(labels, -100))
        assert empty_actual.logits is not None
        torch.testing.assert_close(empty_actual.loss, empty_reference.loss, rtol=1e-5, atol=1e-6, equal_nan=True)

        qwen3_vl._VALID_TOKEN_CE = False
        short_ids = input_ids[:, :1]
        short_labels = torch.tensor([[1]], dtype=torch.long)
        short_reference = forward(copy.deepcopy(candidate), short_ids, short_labels, logits_to_keep=1)
        qwen3_vl._VALID_TOKEN_CE = True
        short_actual = forward(copy.deepcopy(candidate), short_ids, short_labels, logits_to_keep=1)
        assert short_actual.logits is not None
        torch.testing.assert_close(short_actual.logits, short_reference.logits, rtol=0, atol=0)

    print(f"PASS ROSS_VALID_CE={int(enabled)} loss/gradient parity")


if __name__ == "__main__":
    main()

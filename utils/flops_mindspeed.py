#!/usr/bin/env python3
"""FLOPs/MFU by MindSpeed-MM's analytical method (for cross-project comparison).

Mirrors ``mindspeed_mm/fsdp/tools/flops_tool/flops_qwen3_vl.py`` from
Ascend/MindSpeed-MM (itself based on veomni, with fixes).

Scope is Qwen3-VL text + ViT, plus an optional measured ROSS SD UNet/VAE term.
``--include_recompute`` additionally counts the forward pass repeated by
gradient checkpointing.
"""

import argparse

from transformers import PretrainedConfig


class Qwen3VLFlopsCounter:
    """Estimate training FLOPs for dense and MoE Qwen3-VL models."""

    def __init__(self, config: PretrainedConfig):
        self.config = config

    @staticmethod
    def _get_moe_layer_count(config):
        num_experts = getattr(config, "num_experts", 0) or 0
        if num_experts <= 0:
            return 0
        sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
        mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or ())
        return sum(
            layer_idx not in mlp_only_layers and (layer_idx + 1) % sparse_step == 0
            for layer_idx in range(config.num_hidden_layers)
        )

    @staticmethod
    def _compute_attn_params(config):
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden_size // num_attention_heads

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        attn_linear_N = hidden_size * (q_size + k_size + v_size + q_size)
        attn_linear_N *= config.num_hidden_layers
        return attn_linear_N, head_dim, num_attention_heads

    @staticmethod
    def _text_linear_components(config):
        attn_linear_N, head_dim, num_attention_heads = Qwen3VLFlopsCounter._compute_attn_params(config)
        moe_layer_count = Qwen3VLFlopsCounter._get_moe_layer_count(config)
        dense_layer_count = config.num_hidden_layers - moe_layer_count

        dense_mlp_N = config.hidden_size * config.intermediate_size * 3
        mlp_N = dense_mlp_N * dense_layer_count
        if moe_layer_count:
            router_N = config.hidden_size * config.num_experts
            active_expert_N = (
                config.hidden_size * config.moe_intermediate_size * config.num_experts_per_tok * 3
            )
            mlp_N += (router_N + active_expert_N) * moe_layer_count

        lm_head_N = config.hidden_size * config.vocab_size
        return attn_linear_N, mlp_N, lm_head_N, head_dim, num_attention_heads

    @staticmethod
    def _estimate_text_flops(config, tokens_sum, batch_seqlens, include_recompute=False):
        attn_linear_N, mlp_N, lm_head_N, head_dim, num_attention_heads = (
            Qwen3VLFlopsCounter._text_linear_components(config)
        )
        linear_N = attn_linear_N + mlp_N + lm_head_N
        linear_flops = 6 * linear_N * tokens_sum
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attention_flops = (
            7 * seqlen_square_sum * head_dim * num_attention_heads * config.num_hidden_layers
        )
        total = linear_flops + attention_flops
        if include_recompute:
            total += 2 * (attn_linear_N + mlp_N) * tokens_sum
            total += 2 * seqlen_square_sum * head_dim * num_attention_heads * config.num_hidden_layers
        return total

    @staticmethod
    def _estimate_qwen3_vit_flop(
        images_seqlens,
        config,
        vit_freeze=False,
        include_recompute=False,
    ):
        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)
        num_heads = config.num_heads
        depth = config.depth
        dim = config.hidden_size
        mlp_hidden_dim = config.intermediate_size
        out_hidden_size = config.out_hidden_size

        head_dim = dim // num_heads
        spatial_merge_size = config.spatial_merge_size
        merger_hidden_size = dim * spatial_merge_size**2
        patch_embed_N = (
            dim * config.in_channels * config.temporal_patch_size
            * config.patch_size * config.patch_size
        )
        mlp_N = dim * mlp_hidden_dim * 2
        attn_linear_N = dim * (4 * dim)
        merger_N = (out_hidden_size + merger_hidden_size) * merger_hidden_size
        deepstack_merger_N = merger_N * len(
            getattr(config, "deepstack_visual_indexes", ()) or ()
        )
        block_dense_N = (mlp_N + attn_linear_N) * depth
        dense_N = patch_embed_N + block_dense_N + deepstack_merger_N + merger_N

        if vit_freeze:
            dense_rate, attn_rate = 2, 4
        else:
            dense_rate, attn_rate = 6, 14

        linear_flops = dense_rate * dense_N * tokens_sum
        seqlen_square_sum = sum(seqlen * seqlen for seqlen in images_seqlens)
        attention_flops = attn_rate * seqlen_square_sum * head_dim * num_heads * depth
        total = linear_flops + attention_flops
        if include_recompute and not vit_freeze:
            total += 2 * block_dense_N * tokens_sum
            total += 4 * seqlen_square_sum * head_dim * num_heads * depth
        return total

    def _estimate_qwen3_vl_flops(self, tokens_sum, batch_seqlens, **kwargs):
        include_recompute = kwargs.get("include_recompute", False)
        text_flops = self._estimate_text_flops(
            self.config.text_config,
            tokens_sum,
            batch_seqlens,
            include_recompute=include_recompute,
        )
        images_seqlens = kwargs.get("images_seqlens")
        vit_flops = (
            self._estimate_qwen3_vit_flop(
                images_seqlens,
                self.config.vision_config,
                vit_freeze=kwargs.get("vit_freeze", False),
                include_recompute=include_recompute,
            )
            if images_seqlens
            else 0
        )
        return text_flops + vit_flops

    def estimate_flops(self, batch_seqlens, step_time, **kwargs):
        tokens_sum = sum(batch_seqlens)
        return self._estimate_qwen3_vl_flops(tokens_sum, batch_seqlens, **kwargs) / step_time


def _measure_forward_flops(module, call, dense_coeff, attn_coeff):
    """Measure forward Linear/Conv/attention FLOPs and apply training rates."""
    import torch
    import torch.nn as nn

    dense = [0.0]
    attn = [0.0]
    handles = []

    def linear_hook(module, inputs, output):
        if inputs and torch.is_tensor(inputs[0]) and module.in_features:
            rows = inputs[0].numel() // module.in_features
            dense[0] += 2.0 * rows * module.in_features * module.out_features

    def conv_hook(module, inputs, output):
        if not torch.is_tensor(output):
            return
        kernel = 1
        for size in module.kernel_size:
            kernel *= size
        dense[0] += 2.0 * output.numel() * (module.in_channels // module.groups) * kernel

    def attn_hook(module, args, kwargs, output):
        hidden = args[0] if args and torch.is_tensor(args[0]) else kwargs.get("hidden_states")
        if hidden is None or not torch.is_tensor(hidden) or hidden.dim() != 3:
            return
        batch, query_len, channels = hidden.shape
        heads = getattr(module, "heads", None)
        if not heads:
            return
        head_dim = channels // int(heads)
        encoder = kwargs.get("encoder_hidden_states")
        key_len = encoder.shape[1] if torch.is_tensor(encoder) else query_len
        attn[0] += 4.0 * batch * int(heads) * query_len * key_len * head_dim

    for module in module.modules():
        name = type(module).__name__
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif "attention" in name.lower() and hasattr(module, "heads"):
            handles.append(module.register_forward_hook(attn_hook, with_kwargs=True))
    try:
        with torch.no_grad():
            call()
    finally:
        for handle in handles:
            handle.remove()
    return dense[0] * dense_coeff + attn[0] * attn_coeff


def measure_ross_flops(
    sd_path,
    batch=1,
    latent=(35, 63),
    z_channels=320,
    txt_len=77,
    txt_dim=768,
    device=None,
):
    """Measure SD1.5 UNet (trainable) and frozen VAE encode FLOPs."""
    import torch

    if device is None:
        try:
            import torch_npu  # noqa: F401

            device = "npu" if torch.npu.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    from diffusers import AutoencoderKL
    from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import (
        UNet2DConditionModel,
    )

    unet = UNet2DConditionModel.from_pretrained(f"{sd_path}/unet", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(f"{sd_path}/vae", torch_dtype=dtype).to(device).eval()

    height, width = latent
    sample = torch.randn(batch, 4, height, width, device=device, dtype=dtype)
    timestep = torch.full((batch,), 500, device=device, dtype=torch.long)
    encoder_hidden = torch.randn(batch, txt_len, txt_dim, device=device, dtype=dtype)
    z = torch.randn(batch, z_channels, height, width, device=device, dtype=dtype)

    unet_flops = _measure_forward_flops(
        unet,
        lambda: unet(
            sample,
            timestep,
            encoder_hidden,
            class_labels=None,
            return_dict=False,
            z=z,
        ),
        dense_coeff=3.0,
        attn_coeff=3.5,
    )
    vae_input = torch.randn(batch, 3, 280, 504, device=device, dtype=dtype)
    vae_flops = _measure_forward_flops(
        vae,
        lambda: vae.encode(vae_input),
        dense_coeff=2.0,
        attn_coeff=1.0,
    )
    return unet_flops + vae_flops


def get_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL FLOPs/MFU, MindSpeed-MM method")
    parser.add_argument(
        "--llm_seqlens",
        type=int,
        nargs="+",
        default=[1774],
        help="valid token lengths of ONE sample (text)",
    )
    parser.add_argument(
        "--vit_seqlens",
        type=int,
        nargs="+",
        default=None,
        help="ViT patch-token lengths of ONE sample (one entry per image)",
    )
    parser.add_argument("--hf_path", type=str, default="/mnt/sfs_turbo/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--gbs", type=int, default=1, help="samples in the measured step")
    parser.add_argument("--step_time", type=float, required=True, help="step time (s)")
    parser.add_argument("--vit_freeze", action="store_true")
    parser.add_argument(
        "--hardware_flops",
        type=float,
        default=432e12,
        help="per-device Cube-core peak FLOP/s (Ascend 950DT BF16)",
    )
    parser.add_argument(
        "--sd_path",
        type=str,
        default="/mnt/sfs_turbo/models/stable-diffusion-v1-5",
        help="SD1.5 directory holding unet/ and vae/",
    )
    parser.add_argument("--no_ross", action="store_true", help="exclude the ROSS UNet and VAE")
    parser.add_argument(
        "--include_recompute",
        action="store_true",
        help="also report MFU including gradient-checkpointing recomputation",
    )
    parser.add_argument("--latent_hw", type=int, nargs=2, default=[35, 63])
    return parser.parse_args()


def main():
    args = get_args()
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    counter = Qwen3VLFlopsCounter(Qwen3VLConfig.from_pretrained(args.hf_path))

    extra = 0.0
    if not args.no_ross:
        extra = measure_ross_flops(args.sd_path, latent=tuple(args.latent_hw))
        print(f"ROSS (UNet x3/x3.5 + VAE x2/x1) per sample: {extra:.4e} FLOPs")

    def report(label, per_sample):
        flops = (per_sample + extra) * args.gbs / args.device_num / args.step_time
        mfu = flops / args.hardware_flops
        print(f"flops/s per device ({label}): {flops:.4e}")
        print(f"MFU ({label}): {mfu * 100:.2f}%")
        return mfu

    useful = counter.estimate_flops(
        batch_seqlens=args.llm_seqlens,
        images_seqlens=args.vit_seqlens,
        vit_freeze=args.vit_freeze,
        step_time=1.0,
    )
    useful_mfu = report("useful", useful)
    if args.include_recompute:
        executed = counter.estimate_flops(
            batch_seqlens=args.llm_seqlens,
            images_seqlens=args.vit_seqlens,
            vit_freeze=args.vit_freeze,
            include_recompute=True,
            step_time=1.0,
        )
        print(f"recompute FLOPs per sample: {executed - useful:.4e}")
        executed_mfu = report("executed incl. recompute", executed)
        print(f"MFU delta: {(executed_mfu - useful_mfu) * 100:+.2f} pp")


if __name__ == "__main__":
    main()

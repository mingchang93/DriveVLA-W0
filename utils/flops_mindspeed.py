#!/usr/bin/env python3
"""FLOPs/MFU by MindSpeed-MM's analytical method (for cross-project comparison).

Mirrors ``mindspeed_mm/fsdp/tools/flops_tool/flops_qwen3_vl.py`` from
Ascend/MindSpeed-MM (itself based on veomni, with fixes). Kept as a separate
script so ``utils/flops.py`` (the project's measured counter) is unchanged.

Scope is exactly MindSpeed-MM's: Qwen3-VL **text + ViT** only.
- linears: 6 x params x tokens (3x forward: fwd + bwd),
- text attention: 7 x sum(L^2) x head_dim x heads x layers (F2 + B5, KV recompute),
- ViT attention: 14 x sum(L^2) x head_dim x heads x depth (F4 + B10), or 4 when frozen,
- full LM head over every token; embeddings excluded (table lookup).
It does NOT include the ROSS SD denoiser or action path, so its number is a
lower bound for our full training step but is the method other projects quote.

``--include_recompute`` additionally counts the forward pass that gradient
checkpointing re-runs for every checkpointed text decoder layer and ViT block
(the LM head, embeddings, patch embed, mergers and ROSS head are not
checkpointed), so the same method can report the executed-FLOPs MFU alongside
the useful-FLOPs MFU.

Usage (our canonical point: L=1774, per-device bs=24, one image = 3432 ViT
patch tokens; ``--gbs`` is the number of samples and the result is divided by
``--device_num``):

    python utils/flops_mindspeed.py \
        --hf_path /mnt/sfs_turbo/models/Qwen3-VL-8B-Instruct \
        --llm_seqlens 1774 --vit_seqlens 3432 3432 \
        --gbs 24 --device_num 1 --step_time 12.399 --hardware_flops 486e12
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
        """Weight counts for the text stack: (attn, mlp, lm_head, head_dim, heads)."""
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
            # Gradient checkpointing re-runs each decoder block's forward once.
            # The LM head sits outside the checkpointed blocks and is not rerun.
            total += 2 * (attn_linear_N + mlp_N) * tokens_sum
            total += 2 * seqlen_square_sum * head_dim * num_attention_heads * config.num_hidden_layers
        return total

    @staticmethod
    def _estimate_qwen3_vit_flop(images_seqlens, config, vit_freeze=False, include_recompute=False):
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
            # Only the ViT blocks are checkpointed; patch embed and mergers are not.
            total += 2 * block_dense_N * tokens_sum
            total += 4 * seqlen_square_sum * head_dim * num_heads * depth
        return total

    def _estimate_qwen3_vl_flops(self, tokens_sum, batch_seqlens, **kwargs):
        include_recompute = kwargs.get("include_recompute", False)
        text_flops = self._estimate_text_flops(
            self.config.text_config, tokens_sum, batch_seqlens, include_recompute=include_recompute
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
    """Forward FLOPs of ``call()`` with hooks, weighted by MindSpeed rates.

    dense_coeff applies to nn.Linear/nn.Conv*; attn_coeff to diffusers
    ``Attention`` modules (the score matmuls; their projections are Linear
    hooks). Returns FLOPs already multiplied by the coefficients.
    """
    import torch
    import torch.nn as nn

    dense = [0.0]
    attn = [0.0]
    handles = []

    def linear_hook(m, inputs, output):
        if inputs and torch.is_tensor(inputs[0]) and m.in_features:
            rows = inputs[0].numel() // m.in_features
            dense[0] += 2.0 * rows * m.in_features * m.out_features

    def conv_hook(m, inputs, output):
        if not torch.is_tensor(output):
            return
        k = 1
        for size in m.kernel_size:
            k *= size
        dense[0] += 2.0 * output.numel() * (m.in_channels // m.groups) * k

    def attn_hook(m, args, kwargs, output):
        hidden = args[0] if args and torch.is_tensor(args[0]) else kwargs.get("hidden_states")
        if hidden is None or not torch.is_tensor(hidden) or hidden.dim() != 3:
            return
        b, lq, c = hidden.shape
        heads = getattr(m, "heads", None)
        if not heads:
            return
        head_dim = c // int(heads)
        enc = kwargs.get("encoder_hidden_states")
        lk = enc.shape[1] if torch.is_tensor(enc) else lq
        attn[0] += 4.0 * b * int(heads) * lq * lk * head_dim

    for m in module.modules():
        name = type(m).__name__
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif "attention" in name.lower() and hasattr(m, "heads"):
            handles.append(m.register_forward_hook(attn_hook, with_kwargs=True))
    try:
        with torch.no_grad():
            call()
    finally:
        for h in handles:
            h.remove()
    return dense[0] * dense_coeff + attn[0] * attn_coeff


def measure_ross_flops(sd_path, batch=1, latent=(35, 63), z_channels=320,
                       txt_len=77, txt_dim=768, device=None):
    """Forward-measured SD1.5 UNet (trainable) + VAE encode (frozen) FLOPs.

    UNet: dense x3, attention x3.5 (MindSpeed trainable rates).
    VAE : dense x2, attention x1 (MindSpeed frozen rates).
    """
    import torch

    if device is None:
        try:
            import torch_npu  # noqa: F401
            device = "npu" if torch.npu.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import (
        UNet2DConditionModel,
    )
    from diffusers import AutoencoderKL

    unet = UNet2DConditionModel.from_pretrained(f"{sd_path}/unet", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(f"{sd_path}/vae", torch_dtype=dtype).to(device).eval()

    h, w = latent
    sample = torch.randn(batch, 4, h, w, device=device, dtype=dtype)
    timestep = torch.full((batch,), 500, device=device, dtype=torch.long)
    enc = torch.randn(batch, txt_len, txt_dim, device=device, dtype=dtype)
    z = torch.randn(batch, z_channels, h, w, device=device, dtype=dtype)

    unet_flops = _measure_forward_flops(
        unet,
        lambda: unet(sample, timestep, enc, class_labels=None, return_dict=False, z=z),
        dense_coeff=3.0,
        attn_coeff=3.5,
    )
    vae_in = torch.randn(batch, 3, 280, 504, device=device, dtype=dtype)
    vae_flops = _measure_forward_flops(
        vae, lambda: vae.encode(vae_in), dense_coeff=2.0, attn_coeff=1.0
    )
    return unet_flops + vae_flops


def get_args():
    p = argparse.ArgumentParser(description="Qwen3-VL FLOPs/MFU, MindSpeed-MM method")
    p.add_argument("--llm_seqlens", type=int, nargs="+", default=[1774],
                   help="valid token lengths of ONE sample (text)")
    p.add_argument("--vit_seqlens", type=int, nargs="+", default=None,
                   help="ViT patch-token lengths of ONE sample (one entry per image)")
    p.add_argument("--hf_path", type=str, default="/mnt/sfs_turbo/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--device_num", type=int, default=1)
    p.add_argument("--gbs", type=int, default=1, help="samples in the measured step")
    p.add_argument("--step_time", type=float, required=True, help="step time (s)")
    p.add_argument("--vit_freeze", action="store_true")
    p.add_argument("--hardware_flops", type=float, default=486e12,
                   help="per-device peak FLOP/s (default 486 TFLOPS)")
    p.add_argument("--sd_path", type=str,
                   default="/mnt/sfs_turbo/models/stable-diffusion-v1-5",
                   help="SD1.5 dir holding unet/ and vae/ (ROSS head)")
    p.add_argument("--no_ross", action="store_true",
                   help="exclude the ROSS SD UNet + VAE (stock Qwen3-VL scope)")
    p.add_argument("--include_recompute", action="store_true",
                   help="also report MFU counting the extra forward pass gradient checkpointing runs")
    p.add_argument("--latent_hw", type=int, nargs=2, default=[35, 63])
    return p.parse_args()


def main():
    args = get_args()
    # AutoConfig does not know qwen3_vl in our vendored 4.56 tree; use the
    # vendored config class directly.
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    counter = Qwen3VLFlopsCounter(Qwen3VLConfig.from_pretrained(args.hf_path))

    extra = 0.0
    if not args.no_ross:
        extra = measure_ross_flops(args.sd_path, latent=tuple(args.latent_hw))
        print(f"ROSS (UNet x3/x3.5 + VAE x2/x1) per sample: {extra:.4e} FLOPs")

    def _report(label, per_sample):
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
    useful_mfu = _report("useful", useful)
    if args.include_recompute:
        executed = counter.estimate_flops(
            batch_seqlens=args.llm_seqlens,
            images_seqlens=args.vit_seqlens,
            vit_freeze=args.vit_freeze,
            include_recompute=True,
            step_time=1.0,
        )
        print(f"recompute FLOPs per sample: {executed - useful:.4e}")
        exec_mfu = _report("executed incl. recompute", executed)
        print(f"MFU delta: {(exec_mfu - useful_mfu) * 100:+.2f} pp")


if __name__ == "__main__":
    main()

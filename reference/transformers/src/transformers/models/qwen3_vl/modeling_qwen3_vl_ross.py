"""ROSS head for Qwen3-VL.

Mirrors ``models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py``. The differences are
only the LM class/forward signature and the image-token grid: Qwen2.5-VL's
vision tower produced a fixed 10x18 = 180-token image, while Qwen3-VL's
patch-16 / merge-2 tower yields a different grid, so the (h, w) passed to the
denoiser is derived from ``image_grid_thw``.
"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange
from diffusers import AutoencoderKL

from .modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast,
)
from .configuration_qwen3_vl import Qwen3VLConfig
from ..qwen2_5_vl.modeling_ross.denoiser_sd import RossStableDiffusionXOmni


@dataclass
class Qwen3VLROSSOutput(Qwen3VLCausalLMOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None
    action_hidden_states: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    raw_pixel_values_vae: Optional[list] = None


class Qwen3VLConfigROSS(Qwen3VLConfig):
    """Qwen3-VL config plus the ROSS denoiser settings."""

    def __init__(
        self,
        enable_ross: bool = False,
        extract_image_hidden: bool = True,
        extract_action_hidden: bool = True,
        sd_model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_ross = enable_ross
        self.extract_image_hidden = extract_image_hidden
        self.extract_action_hidden = extract_action_hidden
        self.sd_model_path = sd_model_path


class Qwen3VLForConditionalGenerationROSS(Qwen3VLForConditionalGeneration):
    config_class = Qwen3VLConfigROSS

    def __init__(self, config: Qwen3VLConfigROSS):
        super().__init__(config)
        self.enable_ross = getattr(config, "enable_ross", True)
        self.extract_image_hidden = getattr(config, "extract_image_hidden", True)
        self.extract_action_hidden = getattr(config, "extract_action_hidden", True)
        sd_model_path = getattr(
            config, "sd_model_path", "pretrained_models/stable-diffusion-v1-5/unet"
        )
        # Qwen3-VL nests the text width under text_config.
        z_channel = config.text_config.hidden_size
        self.denoiser = RossStableDiffusionXOmni(
            unet_path=sd_model_path,
            z_channel=z_channel,
            mlp_depth=2,
            n_patches=180,
        )
        vae_path = sd_model_path.replace("/unet", "/vae")
        try:
            self.vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=True)
        except Exception:
            self.vae = AutoencoderKL.from_pretrained(vae_path, allow_pickle=True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae_shift_factor = (
            self.vae.config.shift_factor if self.vae.config.shift_factor is not None else 0.0
        )
        self.vae_scaling_factor = (
            self.vae.config.scaling_factor if self.vae.config.scaling_factor is not None else 1.0
        )
        self.post_init()

    def extract_hidden_with_masks(
        self,
        hidden: torch.Tensor,  # [B, L, C]
        image_masks: Optional[torch.Tensor] = None,  # [B, T, N, L] boolean
        action_masks: Optional[torch.Tensor] = None,  # [B, T, L] boolean
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract hidden states using structured boolean masks.

        Returns image_hidden [B, T, N, L_img, C] and action_hidden
        [B, T, L_act, C]. Copied from the Qwen2.5-VL ROSS implementation; it is
        model-agnostic.
        """
        B, L, C = hidden.shape
        image_hidden = None
        action_hidden = None

        if image_masks is not None:
            B_mask, T, N, L_mask = image_masks.shape
            if L_mask != L:
                if L_mask > L:
                    image_masks = image_masks[..., :L]
                    L_mask = L
                else:
                    image_masks = torch.nn.functional.pad(
                        image_masks, (0, L - L_mask), value=False
                    )
                    L_mask = L

            image_hidden_list = []
            for t in range(T):
                time_step_images = []
                for n in range(N):
                    batch_images = []
                    for b in range(B):
                        mask = image_masks[b, t, n]
                        if mask.any():
                            batch_images.append(hidden[b][mask])
                        else:
                            batch_images.append(
                                torch.zeros(0, C, device=hidden.device, dtype=hidden.dtype)
                            )
                    max_tokens = max(img.shape[0] for img in batch_images) if batch_images else 0
                    if max_tokens > 0:
                        padded_batch = []
                        for img in batch_images:
                            if img.shape[0] < max_tokens:
                                img = torch.cat(
                                    [
                                        img,
                                        torch.zeros(
                                            max_tokens - img.shape[0],
                                            C,
                                            device=img.device,
                                            dtype=img.dtype,
                                        ),
                                    ],
                                    dim=0,
                                )
                            padded_batch.append(img)
                        time_step_images.append(torch.stack(padded_batch, dim=0))
                    else:
                        time_step_images.append(
                            torch.zeros(B, 0, C, device=hidden.device, dtype=hidden.dtype)
                        )
                if time_step_images:
                    time_tensor = torch.stack(time_step_images, dim=0)  # [N, B, L_img, C]
                    image_hidden_list.append(time_tensor.permute(1, 0, 2, 3))  # [B, N, L, C]

            if image_hidden_list:
                max_l_img = max(t.size(2) for t in image_hidden_list)
                padded_time = []
                for t_tensor in image_hidden_list:
                    if t_tensor.size(2) < max_l_img:
                        t_tensor = torch.nn.functional.pad(
                            t_tensor, (0, 0, 0, max_l_img - t_tensor.size(2)), value=0
                        )
                    padded_time.append(t_tensor)
                image_hidden = torch.stack(padded_time, dim=1)  # [B, T, N, L, C]

        if action_masks is not None:
            B_mask, T, L_mask = action_masks.shape
            if L_mask != L:
                if L_mask > L:
                    action_masks = action_masks[..., :L]
                    L_mask = L
                else:
                    action_masks = torch.nn.functional.pad(
                        action_masks, (0, L - L_mask), value=False
                    )
                    L_mask = L

            action_hidden_list = []
            for t in range(T):
                batch_actions = []
                for b in range(B):
                    mask = action_masks[b, t]
                    if mask.any():
                        batch_actions.append(hidden[b][mask])
                    else:
                        batch_actions.append(
                            torch.zeros(0, C, device=hidden.device, dtype=hidden.dtype)
                        )
                max_tokens = max(act.shape[0] for act in batch_actions) if batch_actions else 0
                if max_tokens > 0:
                    padded_batch = []
                    for act in batch_actions:
                        if act.shape[0] < max_tokens:
                            act = torch.cat(
                                [
                                    act,
                                    torch.zeros(
                                        max_tokens - act.shape[0],
                                        C,
                                        device=act.device,
                                        dtype=act.dtype,
                                    ),
                                ],
                                dim=0,
                            )
                        padded_batch.append(act)
                    action_hidden_list.append(torch.stack(padded_batch, dim=0))
                else:
                    action_hidden_list.append(
                        torch.zeros(B, 0, C, device=hidden.device, dtype=hidden.dtype)
                    )

            if action_hidden_list:
                max_l_act = max(t.size(1) for t in action_hidden_list)
                padded_time = []
                for t_tensor in action_hidden_list:
                    if t_tensor.size(1) < max_l_act:
                        t_tensor = torch.nn.functional.pad(
                            t_tensor, (0, 0, 0, max_l_act - t_tensor.size(1)), value=0
                        )
                    padded_time.append(t_tensor)
                action_hidden = torch.stack(padded_time, dim=1)  # [B, T, L_act, C]

        return image_hidden, action_hidden

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[object] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        # ROSS-specific inputs
        raw_pixel_values_vae: Optional[torch.Tensor] = None,
        image_token_masks: Optional[torch.Tensor] = None,
        action_future_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden = outputs.hidden_states[-1]  # [B, L, C]

        def _pad_and_cat_list(mask_list: list, target_len: int) -> torch.Tensor:
            padded_list = []
            for m in mask_list:
                pad = target_len - m.shape[-1]
                if pad > 0:
                    m = torch.nn.functional.pad(m, (0, pad), mode="constant", value=0)
                padded_list.append(m)
            return torch.cat(padded_list, dim=0)

        img_max_len = 0
        if image_token_masks is not None:
            img_max_len = (
                max(m.shape[-1] for m in image_token_masks)
                if isinstance(image_token_masks, (list, tuple))
                else image_token_masks.shape[-1]
            )
        act_max_len = 0
        if action_future_masks is not None:
            act_max_len = (
                max(m.shape[-1] for m in action_future_masks)
                if isinstance(action_future_masks, (list, tuple))
                else action_future_masks.shape[-1]
            )
        target_length = max(img_max_len, act_max_len)

        if image_token_masks is not None and isinstance(image_token_masks, (list, tuple)):
            image_token_masks = _pad_and_cat_list(image_token_masks, target_length)
        if action_future_masks is not None and isinstance(action_future_masks, (list, tuple)):
            action_future_masks = _pad_and_cat_list(action_future_masks, target_length)

        image_hidden = None
        action_hidden = None
        if (self.extract_image_hidden or self.extract_action_hidden) and (
            image_token_masks is not None or action_future_masks is not None
        ):
            image_hidden, action_hidden = self.extract_hidden_with_masks(
                last_hidden,
                image_masks=image_token_masks if self.extract_image_hidden else None,
                action_masks=action_future_masks if self.extract_action_hidden else None,
            )

        # Conditions from t to predict t+1
        assert image_hidden.shape[1] == 2, "Currently only support 2 images for each sample"
        assert action_hidden.shape[1] == 2, "Currently only support 2 images for each sample"
        assert raw_pixel_values_vae.shape[0] == 2, "Currently only support 2 images for each sample"

        image_hidden = image_hidden[:, 0].squeeze(1)            # [B, N=1, L, C] -> [B, L, C]
        action_hidden = action_hidden.mean(2)[:, 0]             # [bsz, dim]
        cond_pixel_values = raw_pixel_values_vae[0]
        raw_pixel_values_vae = raw_pixel_values_vae[1]          # [bsz, 3, h, w]

        raw_pixel_values_vae = torch.nn.functional.interpolate(
            raw_pixel_values_vae, size=(280, 504), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            posterior = self.vae.encode(raw_pixel_values_vae).latent_dist
            z_q = (posterior.sample() - self.vae_shift_factor) * self.vae_scaling_factor

        # Qwen3-VL's vision tower uses patch 16 + 2x2 merge, so the per-image
        # token count is not 180; derive the spatial grid from grid_thw.
        merge = int(getattr(self.config.vision_config, "spatial_merge_size", 2))
        grid = image_grid_thw[0]
        h = int(grid[1]) // merge
        w = int(grid[2]) // merge

        with torch.amp.autocast(image_hidden.device.type, dtype=torch.float32):
            action_hidden = self.denoiser.ln_pre_a(action_hidden)
            image_hidden = self.denoiser.ln_pre(image_hidden)
            image_hidden = rearrange(image_hidden, "b (h w) c -> b c h w", h=h, w=w)
            ross_loss = self.denoiser(
                z=image_hidden.float(), target=z_q.float(), z_a=action_hidden.float()
            )

        self._last_logs = {
            "action_loss": float(outputs.loss.detach().cpu()),
            "ross_loss": float(ross_loss.mean().detach().cpu()),
        }

        outputs.loss = outputs.loss + ross_loss.mean()

        return Qwen3VLROSSOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", None),
            image_hidden_states=image_hidden.float(),
            action_hidden_states=action_hidden.float(),
            last_hidden_state=last_hidden,
            raw_pixel_values_vae=[cond_pixel_values, raw_pixel_values_vae],
        )


__all__ = [
    "Qwen3VLConfigROSS",
    "Qwen3VLForConditionalGenerationROSS",
    "Qwen3VLROSSOutput",
]

# Qwen3-VL backbone + ROSS head — architecture reference

**Branch:** `qwen3vl-backbone` · **Code:** `96952f6` · **Written:** 2026-10-28

Scope: the model trained by `scripts/scripts_train/train_qwen_vla_navsim.sh` on
the Qwen3-VL-8B checkpoint, i.e. `Qwen3VLForConditionalGenerationROSS`. Every
shape below is taken from the profiled best config (per-device bs=27,
`--max_pixels 903168`, `ROSS_FUSED_GATEUP=1`) unless marked *derived*.

---

## 1. What the model is

Qwen3-VL-8B is used **unmodified as the backbone**. ROSS subclasses it and adds
a next-frame denoising head. There is no separate text encoder and no
ControlNet-style branch: the diffusion head is conditioned on the LLM's own
feature map at the image-token positions.

| group | attribute | params | source of weights | state |
|---|---|---|---|---|
| Vision tower (27 ViT blocks + merger) | `model.visual` | 576.4M | Qwen3-VL checkpoint | trainable |
| Language model (36 layers, embed, norm) | `model.language_model` | 7568.4M | Qwen3-VL checkpoint | trainable |
| LM head (untied, 4096 × 151936) | `lm_head` | 622.3M | Qwen3-VL checkpoint | trainable |
| **Qwen backbone subtotal** | | **8767.1M (90.2%)** | | |
| Denoiser (SD1.5 UNet + ROSS adapters) | `denoiser` | 863.1M | UNet from SD1.5; adapters fresh | trainable |
| VAE encoder | `vae` | 83.7M | SD1.5 | **frozen** |
| **Total** | | **9713.9M** | | 9630.2M trainable |

The backbone is structurally untouched — no dropped layers, no re-width, nothing
replaced. The only change in the wrapper is `output_hidden_states=True` and
tapping the last hidden state.

Code boundary: [modeling_qwen3_vl.py:969-970](../reference/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L969-L970)
defines `visual` + `language_model`;
[modeling_qwen3_vl_ross.py:51-76](../reference/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl_ross.py#L51-L76)
adds `denoiser` and `vae`.

---

## 2. Data flow

```
Per device: 27 samples; each sample = 2 camera frames + prompt + 8 future action steps
==========================================================================================

 frame_t        frame_t+1              input_ids [27, 1772]
 1248x704       1248x704                   |
     |              |                       |
     +------+-------+                       |
            v                               v
 +----------------------+         +---------------------------+
 | VISION TOWER         |         | embed_tokens              |
 | model.visual         |         |  + image tokens (858/img)  |
 | 576.4M  trainable    |         +------------+--------------+
 |                      |                      |
 | Conv3d k=s=2x16x16   |                      |
 | 2 x 3432 patches     |                      |
 | 27 x ViT block       |                      |
 |   LayerNorm, attn    |                      |
 |   16 heads x 72      |                      |
 |   2D RoPE, full(TND) |                      |
 |   LayerNorm, MLP     |                      |
 |   1152 -> 4304 GELU  |                      |
 | deepstack @ 8/16/24  |                      |
 | 2x2 merge -> 4096    |                      |
 | out: 2 x 858 tokens  |                      |
 +----------+-----------+                      |
            | image_embeds                     |
            +----------------------------------+
                            |
                            v
 +----------------------------------------------------------------+
 | LANGUAGE MODEL   model.language_model   7568.4M  trainable      |
 |                                                                |
 | 36 x decoder layer:                                            |
 |   RMSNorm -> GQA (32 q / 8 kv, D=128, q_norm + k_norm)          |
 |            -> mRoPE -> FA2 TND causal -> o_proj                |
 |   RMSNorm -> SwiGLU 4096 -> 12288 -> 4096                       |
 |                                                                |
 | deepstack visual features added at layers 0, 1, 2              |
 | tokens: 27 x 1772 = 47,844   (0.2% padding)                     |
 +---------------+---------------+--------------------------------+
                 |                               |
                 v                               v
         +---------------+            last_hidden [27, 1772, 4096]
         | lm_head       |                       |
         | 622.3M        |     gather spans; keep span 0 (frame t)
         +-------+-------+                +------+------+
                 |                        v             v
                 v                 image t feats   action t feats
         CE (~5% of tokens)        [27,858,4096]    [27,4096]
                                        |               |
                                   grid 39x22      LN -> MLP_a
                                        |               |
                                  LN -> interp     -> 320-d
                                  MLP 4096->320        |
                                        |               |
                                        +---- cond -----+
                                     factor*z + factor_a*z_a
                                                |
  frame_t+1 --> VAE (FROZEN 83.7M, no_grad)     |
                280x504 -> z_q [27,4,35x63]     |
                          |                     |
                          v                     v
        +-------------------------------------------------------+
        | DENOISER   SD1.5 UNet + adapter MLPs    863.1M        |
        |   x_t     = add_noise(z_q, eps, t)                    |
        |   eps_hat = UNet(x_t, t, empty-prompt emb, cond)      |
        +--------------------------+----------------------------+
                                   v
                          MSE(eps_hat, eps), mean
                                   |
                                   v
                     TOTAL LOSS = CE + ROSS_MSE
```

**Note on the gather.** "gather spans; keep span 0" is a hidden-state gather,
not an attention mask. The collator marks the token positions between
`<|vision_start|>` and `<|vision_end|>` for each of the two image spans
([data_qwen_vla.py:196-211](../reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py#L196-L211)),
`extract_hidden_with_masks` collects **both** into `[B, 2, 1, L_img, C]`, and
the head then keeps span 0 (frame *t*) and action segment 0 before the
denoiser. The frame *t+1* image tokens stay in the LM sequence and are attended
normally — only the diffusion *condition* drops them. Frame *t+1* re-enters as
the VAE denoising target.

---

## 3. Input and intermediate sizes (measured, best config)

| item | value | measured from |
|---|---|---|
| batch | 27 samples/device (216 node-wide) | `--batch_size 27` |
| text sequence | **1772 tokens/sample** | text RoPE `27,32,1772,128` |
| text tokens/step | 47,844 padded / 47,744 valid | MatMul `47844,4096`; FA `47744,…` |
| padding | 0.2% (100 of 47,844) | padded vs unpadded FA inputs |
| images | 54 (2/sample) | ViT FA `185328 = 27×2×3432` |
| image size | 1248×704 px → 78×44 grid of 16×16 patches *(derived)* | 3432 patches × 256 px, 16:9 |
| ViT tokens/step | 185,328 pre-merge | ViT FA `185328,16,72` |
| tokens after merge | 858/image (2×2) | 3432 / 4 |
| VAE input | 280×504, latent 4×35×63 | interpolate in ROSS forward; UNet attn `27,8,2205,40`, 2205 = 35×63 |

Sequence composition: 2 × 858 = **1716 image tokens** (97% of the sequence) +
56 text/action tokens = 1772.

Attention shapes:

| | q | k / v | mask |
|---|---|---|---|
| text | `[47744, 32, 128]` TND | `[47744, 8, 128]` | `[2048, 2048]`, sparse_mode 3 |
| ViT | `[185328, 16, 72]` TND | same | none (dense) |

Large text linears, all on `[47844, 4096]` activations: `[24576, 4096]`
(fused gate/up), `[4096, 4096]` (q/o), `[4096, 12288]` (up/down). ViT MLP runs
at `[185328, 4304]`.

---

## 4. The two repeated blocks

```
Decoder layer (x36)                      ViT block (x27)
------------------------------------     ---------------------------------------
x -> RMSNorm -> q/k/v -> q_norm,k_norm   x -> LayerNorm -> attn 16h x 72
   -> mRoPE -> FA2 TND causal               (2D RoPE, full attention,
   -> o_proj -> (+) -> h                    cu_seqlens per image) -> (+) -> h
h -> RMSNorm -> gate/up 12288            h -> LayerNorm -> fc1 4304
   -> SwiGLU -> down 4096 -> (+) -> out     -> GELU-tanh -> fc2 1152 -> (+) -> out
```

---

## 5. Architecture notes

### Backbone is stock Qwen3-VL

Three Qwen3-VL-specific details drive both training and perf:

1. **Per-head q_norm / k_norm** — two extra RMSNorm-128 ops per layer before
   RoPE. This is why the text path shows four RMSNorm-family kernels per layer.
2. **Deepstack** — three extra mergers tap ViT blocks 8/16/24
   (`deepstack_visual_indexes`) and inject 4096-d visual features additively
   into decoder layers 0–2 (`_deepstack_process`, [modeling_qwen3_vl.py:934](../reference/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L934)).
3. **4× vision token reduction** (2×2 merge) — 3432 patches → 858 tokens per
   image, so 1716 of 1772 text tokens are image tokens.

### The ROSS head is a world model, not a policy head

Actions are supervised as ordinary FAST tokens through the same CE. The
denoiser separately asks: can the LLM's features of frame *t* plus the pooled
action state denoise the VAE latent of frame *t+1*?

- Text conditioning of the SD1.5 UNet is a **constant empty-prompt embedding**
  (`negative_prompt_sd15.pt`, repeated to batch) — no CLIP/text encoder runs.
  The real conditioning is `factor·z + factor_a·z_a`, with `factor` and
  `factor_a` **zero-initialized scalars** ([train_qwen_vla.py:613-616](../utils/train_qwen_vla.py#L613-L616)),
  i.e. a ControlNet-style gate that starts at zero.
- Conditioning uses **span 0 only** — `image_hidden[:, 0]` and
  `action_hidden.mean(2)[:, 0]`. The denoiser therefore never sees frame *t+1*
  features; the LLM does (teacher forcing), the denoiser does not.
- `image_hidden.shape[1] == 2` and `raw_pixel_values_vae.shape[0] == 2` are
  hard-asserted. The two-image layout is baked into the head; changing the
  camera count means editing the forward.
- The VAE is encode-only: `eval()` + `requires_grad_(False)` at construction
  and again in the trainer. The trace confirms no backward kernel ever touches
  it (convs at `280,504`: 5 fwd / 0 bwd; UNet at `35,63`: 53 fwd / 52 bwd).

---

## 6. The ROSS head in context: what it is for

The model is easiest to read as **one trunk with two heads**, only one of which
is the policy:

```
                  +----------------- Qwen3-VL trunk ------------------+
 frames + text -->| ViT 576M  -->  LLM 36 x 4096   7568M              |
                  |                  |                                |
                  |       last hidden [B, 1772, 4096]                 |
                  +-------+-----------------------------+-------------+
                          |                             |
                 text tokens                     image / action
                          v                             v
                  +----------------+      +---------------------------+
                  | POLICY head    |      | WORLD-MODEL head          |
                  | lm_head 622M   |      | VAE 84M (frozen)          |
                  |                |      |  t+1 -> z_q               |
                  | CE ~89 tokens  |      | UNet 859.5M + adapters    |
                  |                |      |  features_t, act_t        |
                  +-------+--------+      +-------------+-------------+
                          |                             |
                   action CE ~5.2                 eps MSE ~0.14
                          +--------------+--------------+
                                         v
                                loss = CE + MSE
```

### What each head is for

**The policy head is the product.** Actions are FAST tokens; the LM emits them;
the CE over ~89 labeled tokens per 1772-token sequence is what teaches driving.
This is the only head a deployed policy needs — although note that, as written,
the training forward always executes the ROSS path (its inputs are asserted, so
there is no branch that skips it).

**The ROSS head is auxiliary supervision with a physical meaning.** It asks:
given the trunk's features of the current frame and the action, can a pretrained
image generator reconstruct the next frame? Its gradient flows back through
`UNet → cond → mlp → last_hidden[image positions] → whole LLM → ViT`. It is not
a side branch that leaves the backbone alone — it is a second, different
training signal on the exact features the policy head reads.

### The role it plays, concretely

1. **It densifies supervision on the visual stream.** The CE touches 89 of 1772
   tokens; the ROSS term touches the 858 image tokens of frame *t* through a
   content-rich target (a 4×35×63 latent), not a class label. Mechanically,
   that is the main thing it adds.
2. **It anchors visual features to a generative latent space.** The target is
   the SD VAE's latent of frame *t+1*, so the trunk's image features must stay
   *decodable* into what happens next, not merely linearly separable — a
   different constraint than the action CE imposes.
3. **It makes the action conditioning load-bearing.** `cond = factor·z +
   factor_a·z_a` starts at zero and learns its own weight, so the model decides
   how much the action vector matters for predicting the future: a soft,
   learned coupling between the action stream and the visual world model.

### What it is not, and what is unmeasured

- **It is not how actions are produced.** No part of the deployed policy comes
  out of the denoiser.
- **~946M parameters (10% of the model) exist only to serve this objective** —
  the frozen VAE, the UNet, and the adapter MLPs. That is also why training
  needs the `stable-diffusion-v1-5` weights at all, and it is the VAE's 280×504
  conv work you see as the ~2.2% Conv bucket plus a share of the elementwise
  bucket. A leaner training loop would strip this path first.
- **No ablation exists.** Nothing in the repo measures whether ROSS helps
  driving (there is no CE-only run). The loss values show it is a *cheap* term —
  0.14 against 5.2 for CE — so the optimizer is essentially following the action
  gradient with ROSS as a regularizer. Whether that regularizer buys PDMS is an
  open empirical question, not something the code establishes.
- **Teacher forcing caveat.** The LM sequence already contains frame *t+1*'s
  image tokens, so what is tested is the denoiser's ability to predict the
  future from frame-*t* features, not the LM's ability to see the future. The
  head's condition is deliberately restricted to span 0 to keep that honest.

---

## 7. Inside the ROSS head: from features to a next frame

### 7.1 The training objective is noise prediction, not reconstruction

```
frame t+1 raw pixels (1248x704) -> interpolate 280x504
        |  VAE.encode, frozen, no_grad
        v
   z_q [B, 4, 35, 63]                  <- clean latent, never decoded in training
        |  eps ~ N(0,1), t ~ U{0..999}
        v
   x_t = scheduler.add_noise(z_q, eps, t)
        |  UNet(x_t, t, empty-prompt emb, cond)
        v
   eps_hat  ->  loss = MSE(eps_hat, eps).mean(), added to CE with weight 1
```

`prediction_type` comes from the scheduler config (epsilon, or velocity for
`v_prediction`). Nothing in the training graph produces an image: the VAE
decoder is never called, so there is no pixel, reconstruction or perceptual
loss.

### 7.2 Where the LLM steers the denoiser

```
LLM hidden @ frame t image tokens  [B, 858, 4096]
   -> reshape to grid (h=22, w=39) -> ln_pre (non-affine LayerNorm)
   -> bilinear to [B, 4096, 35, 63] -> mlp 4096->320 -> cond_z [B, 320, 35, 63]

action pooled [B, 4096] -> ln_pre_a -> mlp_a -> cond_a [B, 320, 1, 1]

cond = factor * cond_z + factor_a * cond_a          (factors zero-initialized)

UNet: x_t -> conv_in -> [B,320,35,63] + cond -> down/mid/up -> eps_hat
```

The conditioning is **one additive feature map right after `conv_in`**
([unet_2d_condition.py:1151-1152](../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/unet_2d_condition.py#L1151-L1152)): a
per-pixel, 320-channel bias on the UNet's own input representation. It is not a
ControlNet branch and not cross-attention — cross-attention receives a constant
empty-prompt embedding loaded from `negative_prompt_sd15.pt`. So the LLM
features are the only condition that varies.

### 7.3 Who learns from this loss

The loss is *defined* on the noise prediction, but it does not only train the
UNet:

| component | learns what |
|---|---|
| UNet (859.5M) | the denoising direction at 35x63x4, for every noise level `t` |
| `mlp` / `mlp_a` | projection of LLM / action features into the 320-channel conditioning space |
| `factor` / `factor_a` | scalar trust in the image / action condition, starting from 0 |
| `ln_pre` / `ln_pre_a` | nothing — `elementwise_affine=False`, no parameters |
| LLM + ViT | through `cond` only, i.e. via the image- and action-token hidden states |
| VAE encoder | nothing — frozen, `no_grad` |
| VAE decoder | nothing — outside the training graph |

The ROSS gradient enters the backbone **only at those two sets of token
positions** and then spreads through causal attention to earlier tokens. It puts
no direct pressure on text tokens, `lm_head`, or later positions — a different
footprint from the CE term, which is dense on the ~89 labeled action tokens but
reaches everything before them.

### 7.4 Inference is where pixels come back

`RossStableDiffusionXOmni.inference()` ([denoiser_sd.py:230-270](../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L230-L270))
runs the standard SD1.5 reverse loop: start from `N(0,1)` latents, call the UNet
for each of ~50 timesteps, `scheduler.step`, then divide by the VAE scaling
factor and `vae.decode` to a **280×504** image. CFG is supported via
`guidance_scale`; the file's `__main__` demo runs this with a dummy `z` and
saves `test_sd15.png`. This path is never exercised by training or evaluation.

### 7.5 Why predict noise at all

- **Direct frame regression fails.** MSE against the next frame is minimised by
  the mean of all plausible futures — a blur. Diffusion models the
  distribution and samples a particular future instead.
- **The labels are free and exact.** You added the noise, so the target is
  perfect at every step, with no annotation cost.
- **One network covers all noise levels.** Small `t` teaches fine detail, large
  `t` teaches global layout; `t` is sampled uniformly during training.
- **Noise and clean-latent prediction are the same parameterisation** (linear
  maps of each other given `x_t` and `t`); noise is better scaled at high `t`
  and is what the sampler consumes.
- **Latent space is 48× smaller**: 4×35×63 = 8,820 numbers versus 3×280×504 =
  423,360 pixels, and the frozen SD VAE plus a pretrained SD1.5 UNet already
  know that space — only the conditioning path is new.

Because training only ever scores the noise field, the quality of generated
frames is uncontrolled: there is no periodic decode, no FID/PSNR, and no
inference check in the loop. Running `inference()` on a few batches is the only
available way to see what the world model has learned.

---

## 8. Compute profile (best config, step 28, rank 0)

Full trace archived at `logs/profile_traces_20261028/profile_l1_gateup_bs27_20261028.tar.gz`
(2.2 GB → 272 MB) with a README; recorded in `npu-port.md`.

Stage 13.359–13.389 s on all 8 ranks, Computing 12.67–12.78 s, communication
0.237–0.340 s, **overlapped 0.000 s**, Free 0.35–0.40 s.

| category | time | share | attributable to |
|---|---|---|---|
| MatMul | 8.515 s | 64.7% | LLM gate/up+down and q/o (T=47,844), ViT MLP, merger |
| FlashAttention fwd+bwd | 2.316 s | 17.6% | 74% ViT dense attention, 26% text causal GQA |
| Elementwise/layout | 1.647 s | 12.5% | norms, SwiGLU/GELU, RoPE, unpad/pad, VAE/UNet activations |
| HCCL (788 launches) | 0.402 s | 3.1% | ZeRO-2 all-gather / all-reduce |
| Conv | 0.283 s | 2.2% | frozen VAE encode at 280×504 + UNet |

FA split: forward 1.128 s, backward 1.157 s. The vision tower is only 6.6% of
parameters but 74% of attention time — it runs dense O(L²) attention over 185k
tokens while the LLM's budget is mostly linear layers. The unpad/pad gathers
measure 0.099 s (`aclnnIndex` 0.046 + `aclnnIndexPut` 0.053) = 0.7% of the step.

---

## 9. Provenance

| claim | source |
|---|---|
| Parameter counts per group | safetensors header sums over `/mnt/sfs_turbo/models/Qwen3-VL-8B-Instruct` |
| Denoiser split (863.1M) | SD1.5 UNet header sum 859.5M; remainder = adapter MLPs, `pos_embed`, scalar gates |
| Total / trainable / frozen | run log `[PARAM] total=9713.9M, trainable=9630.2M`; `VAE: FROZEN` |
| Hidden sizes, depths, heads, deepstack indexes | checkpoint `config.json` (matches the official Qwen3-VL-8B-Instruct release) |
| Token counts, tensor shapes, kernel times | `kernel_details.csv` / `step_trace_time.csv`, step 28 |
| Freeze/zero-init behaviour | `utils/train_qwen_vla.py`, `modeling_qwen3_vl_ross.py`, `denoiser_sd.py` |

**Not verified here:** the exact per-module FLOP split (the profile is time, not
FLOPs), and the 1248×704 pixel size — the trace proves 3432 patches/image at
16 px, and 16:9 aspect implies a 78×44 grid.

# FLOPs counting for the Qwen-VL / ROSS VLA training step

Status: **draft v2, post-review** — nothing implemented yet.
Date: 2026-10-24
Scope: `utils/train_qwen_vla.py` + one new file `utils/flops.py`.

Revision note: v4 costs the vision attention by `cu_seqlens` segments (§2.2.1a)
instead of one dense matrix over the concatenated sequence, and shrinks the
counting protocol to one step at batch 2 scaled by 31 (§5). v3 added the
measured input length (§3.5: `L = 420`, measured on both the canonical subset
and the full pickle) and updated §3.2 — eager at the production batch is now
known to be memory-safe. v2 corrected §2.1's MFU
formula (per-card) and the conv formula. The vision tower's attention is counted
with the same generic dense formula as every other attention module — no
upstream model changes, no special-casing; see §7 for the resulting over-count.

---

## 1. Intent

Produce, for the training step actually run on the NPU box:

1. **Per-step FLOPs** — how much matmul work one optimizer micro-step performs.
2. **The step time it pairs with** — already logged as `step_time_full_s`.

MFU is derived from those two by hand; the tool neither computes nor assumes a
peak. Summing per-step FLOPs answers "how much compute has this project
consumed".

The reference document is `/mnt/sfs_turbo/lmc/flops_guide.md`. It is a
**methodology template only** — we are not reproducing OpenWAM's numbers, so
its conventions are ours to change where our model differs.

## 2. What is counted

### 2.1 Definitions — all numbers are per card

```
per_card_flops = 3 × forward_flops(trainable) + 1 × forward_flops(frozen)
```

3× / 1× is the standard forward+backward accounting: a parameter that receives
a gradient costs 2× extra in backward; a frozen one costs nothing extra.

**Scope boundary: the tool reports FLOPs and step time, not MFU.** There is no
peak constant in the code. The final MFU
(`per_card_flops / (step_time × peak_per_card)`) is computed by hand from those
two numbers, so the peak assumption stays outside the implementation.

Every counter lives on one rank and sees only that rank's own forward, so the
result is already a single-card number. It is **not** divided by world size —
and equally, `peak` is per card, not the cluster total.

**Only one rank is needed.** Every rank runs the same per-device batch through
a replicated model, so the counted work is identical on all of them. The
counting run is therefore launched with `--nproc_per_node 1`; that removes
HCCL entirely and makes the run short and cheap. It also means the counting
pickle only has to be a multiple of the per-device batch, not of
`world_size × batch`.

For cluster-wide figures (e.g. "the 62-step run consumed X PFLOPs across the
box"), multiply the per-card cumulative by `world_size`. State which one a
number is whenever it is reported; they differ by 8 on this box.

### 2.2 Components

| Component | Mechanism | Formula |
|---|---|---|
| `flops_mm` | forward hook on every `nn.Linear` | `2 · rows · in_features · out_features`, `rows = inputs[0].numel() // in_features` |
| `flops_conv` | forward hook on every `nn.Conv1d/2d/3d` | `2 · output.numel() · (in_ch / groups) · prod(kernel)` |
| `flops_attn` | forward hook on every attention module | see §2.2.1 |

`output.numel()` already contains `out_channels`, so there is no extra `· out_ch`.

#### 2.2.1 Attention — three different shapes, one hook

Modules are detected by duck typing (has `head_dim` and one of
`num_heads` / `heads`, class name contains `Attention`). They do **not** share a
cost model:

**(a) Qwen vision attention** — `Qwen2_5_VLVisionAttention`
([modeling_qwen2_5_vl.py:202](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L202)).

`hidden_states` is **2-D** `(seq_len, C)`
([:225](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L225))
— the whole batch's patches concatenated — and the tower splits it by
`cu_seqlens`, passed as a keyword from the block
([:312-318](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L312-L318)).
The executed cost is the sum of per-segment dense attention:

```
lens  = cu_seqlens[1:] - cu_seqlens[:-1]
flops = 4 · H · D · Σ lenᵢ²
```

Only blocks `[7, 15, 23, 31]` use the per-image `cu_seqlens`
(`fullatt_block_indexes`); the other 28 use `cu_window_seqlens`, whose segments
are 112-px windows → 64 patch tokens. `is_causal` is `False` for this module.
When `cu_seqlens` is absent, fall back to the dense `B=1, Lq=Lk=seq_len` term.

**(b) Qwen LLM attention** — `Qwen2_5_VLAttention`
([:633](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L633)).

`hidden_states` is `(B, L, C)`; dense, `Lk = Lq = L`:

```
flops = 4 · B · H · Lq · Lk · D
```

`H` is the **query** head count, so grouped-query attention is costed correctly
(every query head attends over `Lk`).

**(c) diffusers `Attention`** (UNet denoiser, and one mid-block inside the VAE).

`hidden_states` `(B, Lq, C)`; `encoder_hidden_states` `(B, Lk, C)` when present
(UNet cross-attention), else `Lk = Lq`:

```
flops = 4 · B · H · Lq · Lk · D
```

Read `is_causal` off the module when it exists — `diffusers==0.39.0`
([requirements_qwen.txt:54](../../../requirements_qwen.txt#L54)) has the
attribute, default `False` — rather than assuming its absence.

### 2.3 Causal masking

`is_causal` is already on the modules and free to read:
`False` on vision ([:214](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L214)),
`True` on the LLM ([:655](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L655)).

- `flops_attn_full` — no masking adjustment.
- `flops_attn_causal` — halved for `is_causal` modules. Both eager (which
  materialises the full matrix and masks afterwards) and FA2's causal kernel
  differ here; FA2 skips the upper triangle, so the halved figure is what the
  accelerated run executes.

Vision and diffusers attention are non-causal, so the two numbers differ only by
the LLM's share.

MFU is paired with `flops_attn_causal`. `flops_attn_full` is logged for
reference and for the validation in §5.

### 2.4 Trainable / frozen split

Each accumulator has two buckets, decided per module by
`any(p.requires_grad for p in module.parameters())`.

Visual encoder, LLM, `lm_head` and the UNet denoiser are trainable; `model.vae`
is frozen (`requires_grad_(False)` at
[train_qwen_vla.py:532-535](../../../utils/train_qwen_vla.py#L532-L535)). The
VAE runs under `torch.no_grad()`
([modeling_qwen2_5_vl_ross.py:351](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L351)),
but hooks still fire and its parameters are still marked frozen, so it lands in
the 1× bucket correctly. Its mid-block `Attention` therefore shows up in
`flops_attn_frozen`.

### 2.5 Scope: the whole training model

The count covers **everything in the forward**, not just the Qwen backbone. The
UNet denoiser is trainable and consumes a real share of the step; excluding it
would make the total misleading for the "compute consumed" question.

A per-tower breakdown (`visual.`, `model.`, `denoiser.`, `vae.`) is a name-prefix
split on the same hook pass if wanted later. Not built now.

## 3. Two paths, two runs

The requirement: count FLOPs on an **unaccelerated** path so no fused kernel can
hide work, and measure step time on the **accelerated** path.

### 3.1 Counting run — native path

`ROSS_COUNT_FLOPS=1` forces every `ROSS_NPU_*` fused-op flag to `0` **before**
`modeling_qwen2_5_vl_ross` is imported, because
[npu_fused_ops.py:45-51](../../../reference/transformers/src/transformers/models/qwen2_5_vl/npu_fused_ops.py#L45-L51)
reads them at import time. Effects: `probe_backward_support()` no-ops, all
`fused_*()` helpers return `None`, `fuse_unet_group_norm_silu()` returns 0.

The forcing is controlled by a separate flag, `ROSS_FLOPS_NATIVE` (default `1`
while counting), so the cross-check in §5.3 is still possible with fusions on.

**Honesty note:** for the current fusion set the native path changes the count by
zero. `npu_rms_norm`, `npu_swiglu`, `npu_rotary_mul`, `npu_silu`,
`npu_group_norm_silu` are normalisation/elementwise ops; none wraps a Linear or
Conv, and the module underneath still executes and still fires its hook. The
native path is a **guarantee against future fusions hiding matmul work**, not a
correction to today's number.

### 3.2 Attention kernel in the counting run

Measured input length is ~420 tokens (§3.5), so at production batch the eager
LLM score matrix is `B·H·L² = 62 · 28 · 420² ≈ 3.1e8` elements ≈ **612 MB** in
bf16. That fits, so the counting run may use **eager at the production batch**
and count attention directly instead of relying on the shape formula alone. The
vision tower's eager path splits by `cu_seqlens` into chunks of ≤360 tokens, so
it adds nothing measurable.

Re-check this before defaulting to eager if `L` grows or `max_pixels` changes:
the same expression at L=4096 is ~58 GB.

The timed run stays on FA2. The shape-based term (§2.2) reads `Lq`, `Lk`, `H`,
`D` and `cu_seqlens` from the module's own inputs, which are the same whatever
kernel executes, so FA2 hides nothing from the count.

A bs=2 eager run is still useful as the cheap formula cross-check (§5).

### 3.3 Timing run — accelerated path

A normal run (`ROSS_COUNT_FLOPS=0`, fusions on, FA2) supplies `step_time_full_s`,
which already exists in the log
([train_qwen_vla.py:310-317](../../../utils/train_qwen_vla.py#L310-L317)):
forward + backward plus the previous step's `optimizer.step()`/`zero_grad()`.

The counting run emits `flops_total` and logs its own step time, but that time
is **pessimistic by design** (native path) and is not the number to use. The
timed run's `step_time_full_s` is the other half of the pair; MFU is computed
from those two by hand (§2.1).

### 3.4 Batch size

Per-step FLOPs are exactly linear in batch size at fixed sequence length — every
term carries a `B` factor. The collator pads to the longest sequence *in the
batch* ([data_qwen_vla.py:1136](../../../reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py#L1136)),
so bs=2 and bs=62 see different `L`.

- Linear/Conv terms carry `L¹` → token-count scaling is exact.
- The attention terms carry `L²` (§2.2.1a) or `L²` per segment → token-count
  scaling is wrong; they need the production `L`.

bs=1 is impossible regardless: `squeeze()` at
[modeling_qwen2_5_vl_ross.py:339-340](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L339-L340)
collapses the batch dimension and breaks the `rearrange(..., h=10, w=18)`. bs=2
is the floor; the launch script already enforces it.

### 3.5 Fixed input length (measured, 2026-10-24)

Measured with the dataset's own `_get_item`, so the number is the real
post-preprocessing sequence length: images stubbed to the real grid
(`grid_thw=[1,20,36]` → 180 tokens), text and actions tokenized for real,
`cur_frame_idx=3`, `future_nums=8`, `pre_future_nums=2`,
`use_previous_actions=false`. Zero failures.

| dataset | n | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| `navsim_trainval_subset_x20.pkl` (canonical) | 9,340 | 408 | 413 | 416 | 419 | **420** |
| `navsim_emu_vla_256_144_trainval_pre_1s.pkl` (full) | 103,288 | 407 | 413 | 416 | 419 | **423** |

Longest training sample: index 356, token `6a185e2130c75743`.

**Decision: fix `L = 420`** for the canonical x20 set, `L = 424` if the full
pickle is ever used. The mode is 413, so 420 pads ~1.7% more tokens than the
median sample and ~3.4% more attention work by `L²` — negligible, and it makes
the FLOPs number reproducible across batches and runs.

Composition: 360 of the 408–423 tokens are the two image blocks (2 × 180), so
text + action + chat-template tokens are only ~48–63. That also explains why
the distribution is so tight.

Without a fixed length the collator pads each batch to its own max, so the
per-step number would drift between ~413 (most batches) and 420 (batches that
happen to contain a tail sample).

**Counting-run input — duplicate the longest samples.** The x20 subset is
exactly `data[:467] * 20`, and the longest scene (`6a185e2130c75743`) is index
356 replicated at `356 + 467k`. Every per-length count is a multiple of 20 for
this reason, and **there is only one unique scene at L=420**. Build the counting
pickle from that scene alone, replicated to a multiple of `ranks × batch`:

```python
import pickle
src = "/mnt/sfs_turbo/lmc/navsim_openscene/navsim_trainval_subset_x20.pkl"
d = pickle.load(open(src, "rb"))
P = len(d) // 20                                             # 467
sel = [d[i] for i in range(len(d)) if i % P == 356 % P]       # the 20 L=420 copies
pickle.dump(sel * 496, open(".../navsim_flops_L420.pkl", "wb"))   # 9,920 = 8 ranks x 62 x 20 steps
```

The total must be a multiple of the per-device batch and of
`world_size × per_device_batch_size`, otherwise the last batch is short and has
a different shape. The 9,920-entry file was sized for a 20-step bs=62 run
(9,920 = 8 × 62 × 20 = 160 × 62). The current protocol needs only one step at
bs=2, i.e. `world_size × 2 = 16` entries, so the existing file is more than
sufficient.

Caveats: every sample is identical, so the counting run's *step time* is not
meaningful (page-cache-hot image loads, no batch diversity). It is a FLOPs
instrument only — never use it for timing or learning. FLOPs themselves depend
only on shapes, so the duplication is sound.

## 4. Implementation

### 4.1 New file `utils/flops.py` (~140 lines, pure torch, device-agnostic)

```python
class FlopsCounter:
    def __init__(self): ...                    # zeroed buckets, handles list
    def attach(self, model) -> None            # one walk, register hooks
    def reset(self) -> None                    # called before compute_loss
    # per-forward buckets
    mm_trainable, mm_frozen                    # Linear + Conv
    attn_trainable, attn_frozen                # causal variant
    attn_full_trainable, attn_full_frozen
    # derived
    @property def flops_mm(self)               # 3·train + 1·frozen
    @property def flops_attn(self)             # causal variant
    @property def flops_attn_full(self)
    @property def flops_total(self)            # mm + causal attn
    # cumulative across steps
    def accumulate(self) -> None
    cumulative_total
```

Hooks are registered with `register_forward_hook(fn, with_kwargs=True)` so
`encoder_hidden_states` (diffusers cross-attention, passed by keyword) is
visible.

`attach()` is idempotent (guarded by `self._attached`).

No device branches: identical behaviour on CUDA and NPU, per
[CLAUDE.md](../../../CLAUDE.md). No `torch_npu` import.

**Self-check** — `python utils/flops.py` runs an assert-based `__main__` block
covering: `Linear`, `Conv3d` (hand-computed with `output.numel()`), a fake
causal and a fake non-causal attention, and a 2-D input treated as
`B=1`. This is the check that fails if any formula regresses.

### 4.2 Five wiring points in `utils/train_qwen_vla.py`

1. **Top of file, before the `reference.transformers` import (line ~163):** when
   `ROSS_COUNT_FLOPS=1` and `ROSS_FLOPS_NATIVE=1` (default), set
   `ROSS_NPU_RMSNORM`, `ROSS_NPU_SWIGLU`, `ROSS_NPU_ROPE`,
   `ROSS_NPU_ADD_RMSNORM`, `ROSS_NPU_GN_SILU`, `ROSS_NPU_SILU` to `"0"` and log
   the override.

2. **`RossTrainer.__init__`** ([line 232](../../../utils/train_qwen_vla.py#L232)):
   build `self._flops` if enabled, else `None`. Zero overhead when off.

3. **Override `compute_loss`** — the window the guide asks for:

   ```python
   def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
       if self._flops is not None and not self._flops.attached:
           # lazily, i.e. after accelerator.prepare() has run
           self._flops.attach(self.accelerator.unwrap_model(model))
       if self._flops is not None:
           self._flops.reset()
       loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
       if self._flops is not None:
           self._last_flops = self._flops.snapshot()
       return loss
   ```

   Reading between the forward and `accelerator.backward()` keeps
   gradient-checkpoint replay (which happens inside backward) out of the count.
   Checkpointing is on in the launch script, so this ordering is load-bearing.

4. **`RossTrainer.training_step`** ([line 284](../../../utils/train_qwen_vla.py#L284)):
   after the existing `super().training_step(...)`, add to the `logs` dict that
   feeds `_acc_sum` / `_log_mean_and_reset`:

   | key | meaning |
   |---|---|
   | `flops_mm` | Linear + Conv, 3×/1× applied |
   | `flops_attn` | attention, causal variant |
   | `flops_attn_full` | attention, no masking |
   | `flops_total` | `flops_mm + flops_attn` |
   | `seq_len` | padded sequence length of this micro-batch |

   plus `self._flops.accumulate()` for the running total.

5. **End of `train()`**: print cumulative per-card PFLOPs and the cluster-wide
   equivalent (`× world_size`), each labelled.

### 4.3 Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ROSS_COUNT_FLOPS` | `0` | `1` enables the counter |
| `ROSS_FLOPS_NATIVE` | `1` | `1` forces the fused ops off when counting; `0` keeps them on (for the §5.3 cross-check) |

No peak/TFLOPS variable exists — MFU is computed outside the tool (§2.1).

## 5. Validation

1. `python utils/flops.py` — formula self-check (§4.1), including the segmented
   and conv cases.
2. **Eager cross-check (bs=2, native, ~5 steps).** A `TorchFunctionMode` tallies
   `torch.matmul` during one step.
   - LLM: eager runs the dense matmuls at
     [:189](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L189)/[:196](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L196),
     so the tally must equal the dense term.
   - Vision splits per segment inside the module ([:270-290](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L270-L290)),
     so its tally will *not* match our dense term. Expected; see §7.
   - diffusers SDPA is not covered by the assert; the shape formula covers it.
3. **Count with fusions on vs off** (`ROSS_FLOPS_NATIVE=1` vs `0`, same batch,
   2 steps): `flops_mm` and `flops_total` must be bit-identical. This is the
   empirical form of §3.1's claim.
4. **Counting run (per-device batch 2, one step, `--gradient_checkpointing
   False`, `--attn_type sdpa`).** Input is the duplicated longest-sample pickle
   from §3.5, so the shape is exactly `[2, 420]`. FLOPs are **exactly** linear
   in `B` at fixed `L` (every term carries a `B` factor and the model has no
   batch-dependent normalisation), so multiply by 31 to get the bs=62 per-card
   number. One step is enough — the shape is constant, so there is nothing to
   average.

   Checkpointing is **off** for this run: no replay executes, so the 3×/1×
   count is the executed work with nothing to correct. sdpa keeps attention
   visible to the `FlopCounterMode` cross-check (§5.7).

   Use 8 ranks, not one: at `world_size=1` ZeRO-2 must hold the unsharded
   optimizer state and gradient partition, which does not fit even at bs=2. The
   counter is per-rank, so any rank reports the same number.

   Gates: `flops_total > flops_mm`, `flops_attn_frozen > 0` (VAE mid-block
   reachable), `seq_len == 420`. Ignore this run's step time.
5. **Timing run (production pickle, bs=62, accelerated, 100 steps, drop the first
   10, average 20–70)** for the authoritative step time.

6. **PyTorch counter cross-check (same run).** `ROSS_FLOPS_TORCH=1` wraps
   `compute_loss` in `torch.utils.flop_counter.FlopCounterMode` and logs
   `flops_torch_fwd`; the end-of-run print reports the hooks/torch ratio. It
   must agree to a few percent. On torch_npu 2.11 the counter sees the
   **forward only** — backward runs in `npu::` kernels (verified: CPU fwd+bwd
   384, NPU forward 128, NPU backward 0) — so the executed backward still comes
   from the 3×/1× factors.

Deliverables are `flops_total` (step 4) and mean `step_time_full_s` (step 5).
The MFU arithmetic is done outside the tool.

## 6. Open decisions for review

1. **Batch size for the production count.** The counting run, the timing run, and
   the counted `flops_attn` must share the same invocation (batch, `max_pixels`,
   `attn_type`) or the MFU pairing is meaningless. [npu-port.md](../../../npu-port.md)
   Step 23 makes **bs=62** canonical; confirm the counter run uses it.
2. **Headline attention variant** — causal (matches FA2, current plan) or full.
3. **Per-tower breakdown** — out of scope unless asked.
4. **Doc location** — currently `docs/superpowers/specs/`; repo convention is
   root-level `*.md` (`PRECISION_ALIGNMENT.md`, `npu-port.md`).

Resolved since v2: the working sequence length (measured, §3.5 — `L = 420`,
not the 4096 truncation cap). Removed: the peak question — the tool no longer
computes MFU, so no peak value is needed (§2.1).

## 7. Non-goals and known limits

- Not reproducing or comparing against OpenWAM's 39% / 49.2%.
- No model or kernel changes.
- Elementwise / normalisation / activation FLOPs (RMSNorm, SiLU, rotary,
  GroupNorm) are not counted. Deliberate under-count of total executed FLOPs.
- **3×/1× is a convention, and the counting run removes replay from the
  question.** Checkpointing is disabled for the counting run, so the counted
  forward is the only forward that executes and 3×/1× is exactly the work done.
  The *timed* run still uses checkpointing (bs=62 does not fit otherwise), so it
  executes one extra forward per checkpointed block and the resulting MFU is a
  convention number ~25–30% below raw hardware utilisation. That gap is a
  property of the timed run, not of the count.
- **One peak for mixed precision.** The denoiser runs under an fp32 autocast
  ([modeling_qwen2_5_vl_ross.py:355](../../../reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L355))
  and the VAE under `no_grad`; both are divided by the bf16 peak, which
  understates their utilisation. A per-dtype peak table is out of scope unless
  the denoiser proves to be a large share.
- Batch-size linearity holds at fixed `L` only; §3.4.

## 8. Files touched

| File | Change |
|---|---|
| `utils/flops.py` | new, ~140 lines incl. self-check |
| `utils/train_qwen_vla.py` | ~45 lines across 5 touch points |
| `scripts/scripts_train/train_qwen_vla_navsim.sh` | optional: pass `ROSS_COUNT_FLOPS` / `ROSS_FLOPS_NATIVE` through |

No new dependencies. No change to the CUDA path.

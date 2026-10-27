# Qwen2.5-VL NPU Port — Work Log

**Base commit:** `db45a6f7f2d6e966b6b94166a33dcf8e1014804e`
**Branches:** `qwen-npu-port` (port + perf work), `navsim-real-data` (real data)
**Scope:** train the Qwen2.5-VL (ROSS) VLA backbone on Ascend NPU.

## Data policy (2026-10-24)

**Every new learning run uses the real NAVSIM data.** The synthetic
`logs/synth_qwen_data` set is only for pipeline smoke tests; throughput runs
follow the fixed-length protocol noted below.

**Goal: maximize throughput (samples/sec).** Every new run must aim to beat
the best config — **57.52 samples/s median / 54.20 mean, MFU 35.7% median /
33.6% mean** at bs=48 with replay on, fused RMSNorm, fused RoPE and
`expandable_segments` (Step 35 basis: per-step `samples_per_s`, first 5 steps
dropped, fa2, ZeRO-2, valid CE).
Any configuration is fair game — batch size, ZeRO stage, offload, attention
backend, worker count — as long as it beats this number. The shared `lmc-w0`
shell exports `ROSS_STEP_TIMING=0`; override it to `1` so the median is
comparable.

**Current best config: `--batch_size 48`**, replay on, `ROSS_NPU_RMSNORM=1`,
`ROSS_NPU_ROPE=1`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,
`CPU_AFFINITY_CONF=1`, `ulimit -l unlimited` (Steps 33, 35; supersedes Step
29's 50.73/45.75 and Step 32's 53.99/50.25). The launcher now sets the whole
canonical env; existing environment values win for the ROSS_NPU_* flags.

```bash
export CPU_AFFINITY_CONF=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
--data_path /mnt/sfs_turbo/lmc/navsim_openscene/navsim_trainval_subset_x20.pkl \
--data_root /mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/trainval \
--batch_size 48 --num_workers 8 --pin_memory true --save_model false
```

Run `ulimit -l unlimited` first (Step 23): pinning needs it, and 8 workers is
the tested point at bs=48–62; more workers risk exhausting host pinned memory.

`--save_model` defaults to **false**: throughput runs write no checkpoint and no
final model (a 200-step run otherwise leaves ~155 GB: an end-of-run checkpoint
with optimizer state plus the 18 GB model). Set `--save_model true` only when
the weights are needed.

- `navsim_trainval_subset_x20.pkl` = the 467 scenes from camera shard 0,
  replicated 20× purely so the dataloader has multiple batches per epoch. As
  more shards are downloaded, regenerate the subset with
  `filter_pickle_by_images.py` (drop the replication once the retained count
  gives several batches per epoch) or use the full pickle at
  `/mnt/sfs_turbo/models/DriveVLA-W0/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl`.
- **FLOPs/MFU counting does not train on the above.** It uses
  `navsim_flops_L420.pkl` — the single longest scene (L=420) replicated to a
  fixed batch shape. Use it for shape-fixed FLOPs and per-step compute timing
  only; its wall time is optimistic (page-cache-hot, identical images) and it
  is never a learning set. See Step 24.
- `--pin_memory true` needs `ulimit -l unlimited` (the container's memlock is
  64 MB) and a bounded worker count; 8 workers is tested at bs=62. Pinning is
  the main on-stage free-time lever (Step 23).
- Step 20 has the data layout, download and validation details.

This document records every change made on the branch since the base commit, the
reason for it, and what is still outstanding.

---

## Accessing the NPU box

| Item | Value |
|---|---|
| SSH alias | `ott9-a57` |
| Host / user | `liteserver-hps-d235-00007.novalocal` / `root` |
| Repository | `/mnt/sfs_turbo/lmc/DriveVLA-W0` |
| tmux session | `lmc-w0` (training pane; `lmc-cann920` is a separate session) |
| Conda env | `drivevla` — active only inside `lmc-w0` |
| Accelerators | 8 × Ascend NPU (`/dev/davinci0..7`) |
| Qwen weights | `/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct` |
| SD1.5 weights | `/mnt/sfs_turbo/models/stable-diffusion-v1-5` |
| Synthetic data | `logs/synth_qwen_data` (4,096 scenes), `logs/synth_qwen_data_large` (8,192) |

### Connect and attach

```bash
ssh ott9-a57
tmux attach -t lmc-w0
```

### Run a command without attaching

```bash
ssh ott9-a57 "tmux send-keys -t lmc-w0 'cd /mnt/sfs_turbo/lmc/DriveVLA-W0 && python -c \"import torch; print(torch.__version__)\"' C-m"
ssh ott9-a57 "tmux capture-pane -t lmc-w0 -p | tail -20"
```

For automation, `ssh -o BatchMode=yes ott9-a57 '<command>'` works (key auth is
configured) and returns the command's stdout.

### Launching runs — always use `tmux`

**Launch every training/eval run inside the `lmc-w0` tmux session. Do not use
detached processes (`docker exec -d`, `nohup ... &`, background SSH jobs)
unless explicitly told otherwise.** The session runs inside the training
container with the env already active, and it is what the user monitors.

```bash
ssh ott9-a57 "tmux send-keys -t lmc-w0 'cd /mnt/sfs_turbo/lmc/DriveVLA-W0 && bash run_100.sh 2>&1 | tee logs/run.log' C-m"
ssh ott9-a57 "tmux capture-pane -t lmc-w0 -p | tail -20"
```

To stop a job, kill its processes (`pkill -f train_qwen_vla.py`) — do not send
`C-c` to the shared pane.

### Change workflow

Every code change that a run will use follows exactly one path:

1. modify locally
2. commit locally **only when the user says so** — never commit on your own
3. push **from this machine**
4. pull on the box
5. run, and add the run directory + base commit to the Run index

**The box is pull-only.** Never `git push` from it. **Never copy files to the
box** (`scp`, `rsync`, `tar | ssh`) either — every change must arrive through
commit → push → pull so there is a record of what actually ran. If this
machine cannot reach GitHub, stop and ask rather than routing around the
push.

### Environment caveat

A plain non-interactive SSH shell is **not** the training environment:

```text
ssh ott9-a57 'which python'   -> /usr/bin/python   (3.11.6, no torch)
```

The `drivevla` env is active only inside the `lmc-w0` tmux session
(`which python` → `/usr/local/python3.12.13/bin/python`,
`CONDA_PREFIX=/root/miniconda3/envs/drivevla`). Run Python/torch work through
tmux, or use the exact interpreter path from inside that session.

`/mnt/sfs_turbo` is shared with the SSH shell, so `git`, `ls`, `ps`, `du`, and
file inspection can be done directly over SSH.

### Verify the box

```bash
ssh ott9-a57 'ls /dev/davinci* | head; nproc'
ssh ott9-a57 "tmux send-keys -t lmc-w0 'python -c \"import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())\"' C-m"
```

Expected: eight devices and `(2.11.0+cpu, True, 8)` (verified 2026-10-25).

**Do not send `C-c`** to the `lmc-w0` pane unless you intend to kill the running
job — the session is shared.

---

## Step 0 — Baseline audit (no code change)

Entry point: [`utils/train_qwen_vla.py`](utils/train_qwen_vla.py).

Findings that drove the subsequent changes:

| Finding | Evidence |
|---|---|
| Backbone is **Qwen2.5-VL-7B**, not 3B | HF `liyingyan/DriveVLA-W0/Qwen_PDMS_85.3/config.json`: `hidden_size=3584`, `num_hidden_layers=28` (3B is `2048`/`36`). The 3B values in `argument.py` and the deleted launcher are stale defaults. |
| The model needs the **vendored transformers 4.56.0** | `reference/transformers/src/transformers/__init__.py:21`. `requirements.txt` pins `4.44.0`, which has no `qwen2_5_vl` module. The Qwen model/config/processor files are byte-identical to upstream `v4.56.0`; the only differences are the ROSS additions, a one-line import in `trainer.py`, and a ROSS reference in the Qwen `__init__.py`. |
| ROSS hardcodes a **10×18 = 180-token** image grid | `modeling_qwen2_5_vl_ross.py:68` (`n_patches=180`) and `:359` (`rearrange(... h=10, w=18)`). The default `max_pixels=28*28*576` produces 576 tokens for 16:9 images and crashes. |
| ROSS asserts **exactly 2 image/action segments** | `modeling_qwen2_5_vl_ross.py:335-337`. `use_previous_actions=true` with `cur_frame_idx>0` adds two more action segments and trips the assert. The old launcher avoided this with `cur_frame_idx=0`, which silently suppressed history. |
| Per-device batch must be **≥ 2** | `image_hidden[:, 0].squeeze()` at `:339` removes the batch dim when batch size is 1. |
| Training requires **Stable Diffusion 1.5** | UNet at `denoiser_sd.py:40`, VAE at `modeling_qwen2_5_vl_ross.py:71-81`, scheduler at `denoiser_sd.py:62`; `--sd_model_path` must end in `/unet` because paths are derived with `.replace("/unet", ...)`. |
| `decord` has **no aarch64 wheel** | Imported unconditionally by `qwenvl/dataset/data_qwen_vla.py` and `rope2d.py`, but `VideoReader` is never used on the VLA path. |

Image path resolution (`data_qwen_vla.py:391-424`): the pickle's
`.../<log>/<frame>.npy` is rewritten to
`<data_root>/<log>/CAM_F0/<frame>.jpg`.

---

## Step 1 — `1701619` Add Qwen2.5-VL VLA training script and requirements file

### 1a. `requirements_qwen.txt` (new)

Added a Qwen-specific environment file because `requirements.txt` is for the
Emu3 path and its `transformers==4.44.0` is incompatible.

Initial contents (evolved over commits `1701619`, `2088d65`, `877dfd8`; the
torch pins below were replaced by the 2.11.0 / CANN 9.1.0 stack in `02ea9d5` —
see `requirements_qwen.txt`):

- `-e ./reference/transformers` — editable 4.56.0 so `import transformers`
  resolves to the vendored patched tree and port edits are live.
- NPU torch stack: `torch==2.7.1`, `torchvision==0.22.1`,
  `torchaudio==2.7.1`, `torch_npu==2.7.1.post4`.
- `deepspeed==0.17.6`, `accelerate`, `tensorboard`, `numpy`, `pillow`,
  `pyyaml`, `tqdm`, `safetensors`.
- `diffusers`, `einops==0.8.1`, `scipy`.
- `decord` removed (see Step 4); `flash-attn` left commented because it is
  CUDA-only.

Install order is documented in the file header; the current header targets
CANN 9.1.0 (it was 8.5.1 in the original port), sourced from
`/usr/local/Ascend/ascend-toolkit/set_env.sh` before `torch_npu`.

### 1b. `scripts/scripts_train/train_qwen_vla_navsim.sh` (new)

The upstream launcher for this path had been deleted in `929171a`. It was
recovered for reference (`git show eb82f31:scripts/scripts_train/train_qwen_vla_navsim.sh`)
but not copied verbatim, because it passes flags that no longer exist and uses
stale defaults.

Differences from the deleted launcher:

| Old launcher | New launcher |
|---|---|
| `--model_name_or_path ...3B...` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| passes `--ross_loss_weight`, `--ross_grad_clip` | omitted — removed from `ModelArguments` at HEAD |
| `cur_frame_idx=0` | `3` (matches the pickle window `-3..+8`) |
| no `max_pixels` | `150528` → 180 tokens for 16:9 |
| no SD path validation | resolves `<parent>/unet`, validates `vae/` + `scheduler/` |
| no env preflight | checks `transformers` has Qwen2.5-VL; warns if `flash_attn` missing |
| `use_previous_actions=true` | `false` (Step 2) |

Features of the new launcher:

- `--device auto|cuda|npu`, forwarded via the `DEVICE` environment variable.
- NPU env: `INF_NAN_MODE_ENABLE=1`, `HCCL_DETERMINISTIC=TRUE`,
  `ASCEND_RT_VISIBLE_DEVICES`; CUDA: `NCCL_DETERMINISTIC`,
  `CUDA_DEVICE_MAX_CONNECTIONS`, `PYTORCH_CUDA_ALLOC_CONF`.
- `PYTHONPATH` includes the repo root (for `reference.*` imports) and
  `reference/Qwen2.5-VL/qwen-vl-finetune`.
- Supports `fa2` on NPU through the vendored flash-attention path; `sdpa` is
  the default until the A/B is recorded.
- Zero-stage shortcuts `--zero_stage 2|3` → `scripts/sft/zero{2,3}_offload.json`.
- Writes the full invocation to `<run_dir>/startup_command.txt`.
- Resume is automatic: `train_qwen_vla.py` detects `checkpoint-*` in the run dir.

---

## Step 2 — `f9a61e3` Update training script to disable previous actions and add synthetic dataset generation script

### 2a. `use_previous_actions` default changed to `false`

Reason: the ROSS forward asserts `action_hidden.shape[1] == 2`
(`modeling_qwen2_5_vl_ross.py:336`). `build_action_segments` adds four
segments when history is enabled and `cur_frame_idx > 0` (pre-user,
pre-assistant, cur-user, cur-assistant), so the assert fails. With history
disabled, exactly two assistant segments remain.

Comment added next to the default in the launcher.

### 2b. `tools/synth/synth_qwen_vla_data.py` (new)

Purpose: measure throughput without downloading the 1.24 TB NavSim camera set.

Generated schema matches `data_qwen_vla.py`:

- One scene = one training sample.
- `text` / `pre_1s_text`: `list[12]` strings.
- `image` / `pre_1s_image`: `list[12]` `.npy`-style paths.
- `action` / `pre_1s_action`: `[12, 3]` float32 in `[-1, 1]`.
- Images are **504×280**, i.e. exactly 10×18 = 180 Qwen vision tokens.
- Files resolve to `<data_root>/<log>/CAM_F0/<frame>.jpg`.

Generated artifact (not committed; `logs/` is gitignored):

```
logs/synth_qwen_data/
├── synth_navsim.pkl        # 256 scenes
└── images/<log>/CAM_F0/*.jpg  # 12 per scene = 3,072 files, ~12 MB
```

Verified: 256 samples, correct keys, first image resolves, resize grid = 180.

Throughput command:

```bash
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
    --data_path logs/synth_qwen_data/synth_navsim.pkl \
    --data_root logs/synth_qwen_data/images \
    --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5 \
    --model_name_or_path /mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct \
    --device npu --ngpus 8 --batch_size 2 \
    --max_steps 20 --logging_steps 1 \
    --save_steps 100000 --save_total_limit 1
```

Note: on the 8-rank NPU run the remote pkl contained 4,096 scenes and the
per-rank dataloader length was 256 (= 4096 / 8 / batch 2), so the ranks were
sharded. The generator default is 64 scenes (`--num_scenes`); regenerate with
`--num_scenes 4096` if you want the same sample count.

---

## Step 3 — `2088d65` Update requirements_qwen.txt to clarify installation instructions for torch stack on Ascend NPU

Added the NPU torch stack directly to the requirements file, per request:

```text
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
torch_npu==2.7.1.post4
```

Header updated with:

- CANN 8.5.1 is system-level; source `/usr/local/Ascend/ascend-toolkit/set_env.sh`.
  (Later replaced by the 2.11.0 / CANN 9.1.0 pins in `02ea9d5`.)
- CUDA users must replace these four lines with torch 2.4.0 / torchvision
  0.19.0 / torchaudio 2.4.0 and drop `torch_npu`.
- DeepSpeed's ZeRO CPU offload JIT-builds CPUAdam, so a C++ toolchain is needed.

Verified on PyPI: `torch_npu==2.7.1.post4` ships cp39–cp313 wheels for
`manylinux_2_28_{x86_64,aarch64}`; `torch==2.7.1` has both architectures.

---

## Step 4 — `877dfd8` Refactor decord import to handle ImportError and update requirements to clarify its optional status

Triggered by the remote install failing:

```
ERROR: No matching distribution found for decord
```

`decord` has no aarch64 wheel, and pip aborts the whole transaction on one
unresolvable requirement — so `transformers` was never installed.

Changes:

- [`data_qwen_vla.py:25-28`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py#L25-L28):
  guarded import; `VideoReader = None` on `ImportError`.
- [`rope2d.py:19-22`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/rope2d.py#L19-L22):
  same guard.
- `requirements_qwen.txt`: `decord` removed, with a note.

`VideoReader` is not referenced elsewhere in either file, so the guard is safe
for the image-only VLA path.

---

## Step 5 — NPU runtime fixes (committed as `2e40d2c`)

Triggered by the first real launch on the NPU host:

```
ModuleNotFoundError: No module named 'flash_attn'
  at qwenvl/train/trainer.py:7
```

Five fixes were applied so the training path can start on NPU and reach a
forward/backward pass:

| # | File | Change |
|---|---|---|
| 1 | [`qwenvl/train/trainer.py:7`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py#L7) | `flash_attn` import wrapped in `try/except ImportError` (`flash_attn_varlen_func = None`); `_flash_attention_forward` now raises a clear error only if `fa2` is actually used. `sdpa` runs without flash-attn. |
| 2 | [`modeling_ross/denoiser_sd.py:25-31`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L25-L31) | `negative_prompt_path` default changed from the author's `/mnt/vdb1/...` path to `None`, resolved relative to `__file__` (`negative_prompt_sd15.pt` next to the module). Added `import os`. |
| 3 | [`modeling_ross/denoiser_sd.py:81,195`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L81) | Both `torch.load(...)` calls now pass `map_location="cpu"` so the embeddings load on a machine without CUDA. |
| 4 | [`modeling_qwen2_5_vl_ross.py:355`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L355) | `torch.amp.autocast('cuda', ...)` → `torch.amp.autocast(image_hidden.device.type, ...)` (`npu` on Ascend, `cuda` on GPU). |
| 5 | [`utils/train_qwen_vla.py:280`](utils/train_qwen_vla.py#L280) | `torch.cuda.synchronize()` in `safe_save_model_for_hf_trainer` → existing `device_synchronize()` helper. |

Verification: these fixes were exercised by the successful 8-NPU run in Step 7.

---

## Step 6 — Missing `transformers.data` package (committed as `67292e7`)

The next launch got past the flash-attn guard and failed importing the Trainer:

```
ModuleNotFoundError: No module named 'transformers.data'
  at reference/transformers/src/transformers/trainer.py:59
```

Root cause: the vendored transformers checkout is a subset and omits the
upstream `data/` package entirely (`git log --all -- .../transformers/data`
returns nothing). `trainer.py:59` imports `DataCollator`,
`DataCollatorWithPadding`, and `default_data_collator` from it.

Fix:

- Added [`data/data_collator.py`](reference/transformers/src/transformers/data/data_collator.py)
  verbatim from upstream `transformers v4.56.0` (101 KB).
- Added [`data/__init__.py`](reference/transformers/src/transformers/data/__init__.py)
  as a minimal subset exporting only the data-collator symbols. Upstream's
  `__init__.py` also imports `.metrics` and `.processors`, which the Qwen path
  never uses; including them would pull in unused code/dependencies.
- Added a `.gitignore` negation for the vendored `data/` directory; the root
  `data` rule (meant for the project's dataset dir) was matching it.

`data_collator.py` imports only `..models.bert`, `..tokenization_utils_base`,
and `..utils`, all present in the vendored tree. Both files pass `py_compile`.

---

## Step 7 — First successful NPU training run

8 × Ascend NPU, synthetic data, 20 optimizer steps:

```bash
bash scripts/scripts_train/train_qwen_vla_navsim.sh \
    --data_path logs/synth_qwen_data/synth_navsim.pkl \
    --data_root logs/synth_qwen_data/images \
    --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5 \
    --model_name_or_path /mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct \
    --device npu --ngpus 8 --batch_size 2 \
    --max_steps 20 --logging_steps 1 \
    --save_steps 100000 --save_total_limit 1
```

Environment / model:

- `torch 2.7.1+cpu`, `torch_npu 2.7.1.post4`, `transformers 4.56.0`
- 9,238.5M params total, 9,154.8M trainable (7B LLM + vision + SD1.5 UNet)
- script estimate: ~187.6 GiB optimizer + master weights (ZeRO-3 CPU offload)
- dataset: 4,096 synthetic samples; per-rank dataloader length 256

Trainer output:

| Metric | Value |
|---|---|
| `train_runtime` | 304.88 s (includes the final checkpoint save) |
| `train_samples_per_second` | **1.05** |
| `train_steps_per_second` | **0.066** |
| `train_loss` | 13.28 |
| steady-state step time | **~8.4 s/step** (progress bar at 20/20 before saving) |
| `ross_loss` | ~0.97–1.01, stable, no NaN |
| `action_loss` | 19.8 → 7.6 (random labels — not a learning signal) |

Notes:

- The step time starts at ~12 s (warm-up) and settles at ~8.3–8.4 s/step.
  The end-to-end 0.066 steps/s is dominated by checkpoint I/O, not compute.
- Despite `--save_steps 100000`, the Trainer writes `checkpoint-20` at the end,
  then the script writes the final model. The run dir reached **~137 GB**
  (~18 GB final model + ~120 GB checkpoint with ZeRO optimizer state).
  For throughput-only runs, delete the checkpoint or add a
  `--save_strategy no` option (not exposed by the launcher yet).
- `use_previous_actions=false` was accepted; the 2-segment assert passed.

---

## Step 8 — ROSS loss diagnosis

The 20-step run logged `ross_loss` ≈ 0.97–1.01 with no downward trend. A
diagnostic run and checkpoint inspection showed why that run was not a valid
learning test.

**First-run checkpoint inspection** (`checkpoint-20`, compared against the base
weights):

| Weight group | Result |
|---|---|
| LLM / ViT | changed (max ≈ 2.4e-4 / 1.2e-4) |
| SD1.5 UNet | bit-identical to the original bf16 weights |
| `denoiser.factor` | 6.6e29 (anomalous) |
| `denoiser.factor_a` | 4e-10 |

**Controlled 100-step diagnostic** (`lr=1e-4`, `warmup=10`, `save_strategy=no`,
same seed/data):

| Metric | Value |
|---|---|
| `train_runtime` | 806.98 s / 100 steps = **8.07 s/step** |
| `train_samples_per_second` | **1.983** |
| `train_loss` | 5.82 |
| `ross_loss` | 0.0006–0.032, typical ~0.003, no divergence |
| `action_loss` | 19.85 → ~4.8 |
| UNet `conv_in.weight` | 8,934 / 11,520 elements changed, max 1.47e-3 |
| UNet mid-attention | 384,613 / 1,638,400 elements changed, max 9.77e-4 |
| `denoiser.factor` / `factor_a` | -5.0e-4 / 3.6e-4 (sane) |

Conclusion:

- The ROSS branch **does train**; its loss can drop to ~1e-3 on the synthetic
  data. The 20-step run never gave it a usable learning rate (warmup 50 meant
  the LR peaked at 2e-5 on the last step), and the UNet updates were below bf16
  save resolution.
- **Caveat (added in Step 10):** the synthetic `pre_1s_*` streams were
  byte-identical to the current-frame streams, so this objective was a
  near-identity reconstruction rather than `t → t+1`. The ~1e-3 loss does not
  demonstrate the intended ROSS task.
- The first run's `factor=6.6e29` was not reproduced. Treat that checkpoint as
  anomalous; if it recurs, restore the `ross_grad_clip` / NaN guards that
  existed in `eb82f31` (they were removed from HEAD).
- Use `lr` in the 5e-5–1e-4 range, `warmup_steps` 10–50, and several hundred
  steps for real training. For throughput-only runs, `--save_strategy no`
  avoids the 120 GB optimizer checkpoint (the final 18 GB model is still
  written).

---

## Step 9 — Larger-dataset confirmation (interrupted at 70/300)

Regenerated 8,192 scenes / 98,304 images (455 MB) at
`logs/synth_qwen_data_large`, then ran 300 steps with `lr=1e-4`,
`warmup=10`, `save_strategy=no`.

The run was **interrupted by SIGINT at step 70/300** (9:41 elapsed). Observed
before the interrupt:

| Metric | Value |
|---|---|
| `action_loss` | 20.39 → 5.21 |
| `ross_loss` | 0.00055–0.0129, typical ~0.002 |
| `grad_norm` | ~9.7 |
| LR at step 70 | 9.02e-5 |
| step time | 7.72 s/step |

This confirms the 100-step diagnostic on a 2× larger dataset: the ROSS loss is
stable, low (~2e-3), and three orders of magnitude below the first run's ~1.0.
The run directory is empty because the process was killed before the final
model save.

**Caveat (Step 10):** this run used the same degenerate pre/cur pairing as
Step 8 — `pre_1s_image == image` in every scene. The low loss therefore
reflects a near-identity reconstruction, not `t → t+1` prediction.

To complete the full 300 steps, rerun `run_confirm.sh` on the host (or the
equivalent command); the SIGINT came from the terminal, not the training code.

---

## Step 10 — Fix synthetic `pre_1s_*` and re-run (60 steps)

Auditing the synthetic data against the upstream generator found that every
scene had `pre_1s_image == image` and `pre_1s_action == action`:

- [`synth_qwen_vla_data.py`](tools/synth/synth_qwen_vla_data.py) copied the current-frame lists into the `pre_1s_*` fields (now fixed at [:85-87](tools/synth/synth_qwen_vla_data.py#L85-L87), self-check at [:91-98](tools/synth/synth_qwen_vla_data.py#L91-L98)).
- Upstream does not: [`pickle_generation_navsim_pre_1s.py:143-145`](tools/pickle_gen/pickle_generation_navsim_pre_1s.py#L143-L145) anchors `pre_1s_*` on `scene[i-2]`, and Phase 2 of that script explicitly repairs identical pairs ([:236](tools/pickle_gen/pickle_generation_navsim_pre_1s.py#L236), [:260-277](tools/pickle_gen/pickle_generation_navsim_pre_1s.py#L260-L277)).

Because the ROSS branch conditions on the pre segment and reconstructs the
current frame ([`modeling_qwen2_5_vl_ross.py:339-342`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py#L339-L342)),
the copy reduced the objective to "reconstruct the frame you conditioned on."

Fix (`e988fef`): `pre_1s_*` is now the same window anchored `PRE_SHIFT = 2`
frames earlier, with an assertion in the generator that fails if the pair
collapses back to a copy.

Verification (inside the training container, against the real dataset class):

| Check | Result |
|---|---|
| `pre_1s_image == image` | 0 / 4096 and 0 / 8192 after regeneration |
| shift alignment `pre[k] == image[k-2]` | all scenes |
| schema / image paths / 180-token grid | unchanged, 0 violations |
| ROSS segment asserts | image T=2, action T=2 |

Regenerated `logs/synth_qwen_data` (4,096 scenes) and
`logs/synth_qwen_data_large` (8,192) on the NPU host.

60-step run on the corrected data (`logs/ross_short_20261020`, lr 1e-4,
warmup 10, `save_strategy no`):

| Metric | Value |
|---|---|
| `train_runtime` | 523.3 s = **8.72 s/step** |
| `train_samples_per_second` | 1.835 |
| `ross_loss` | 0.972 → ~1.0, **flat across all 60 steps** |
| `action_loss` | 19.67 → ~5.0 |
| `grad_norm` | 67 → ~9 |
| final model | 18 GB |

The corrected objective sits at the epsilon-prediction baseline (~1.0, i.e.
predicting zero noise) and does not move within 60 steps. This confirms the
pipeline on non-degenerate inputs but is **not** a learning result; the earlier
~1e-3 losses were an artifact of the identity task. A few hundred steps are
needed to tell whether the ROSS branch fits the real objective.

**Corrected by Step 11:** the flat ~1.0 was not an insufficient-steps effect.
`factor` was frozen at a huge garbage value, which swamped the UNet input and
pinned the loss at the unconditional baseline. With the gate fixed, the first
step already reports `ross_loss` ≈ 0.0019 (Step 11). The ~1e-3 losses on the
degenerate data remain an artifact of the identity task, but the ~1.0 measured
here was the gate bug.

---

## Step 11 — ROSS conditioning gates were initialized from garbage

The two gates that scale the VLM conditioning into the SD1.5 UNet
([`denoiser_sd.py:53,60`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py#L53))
are bare `nn.Parameter`s created with `torch.tensor(0.)`. Values observed at
load:

| Run | `factor` | `factor_a` |
|---|---|---|
| 20-step (Step 7), saved | 6.6e29 | — |
| 60-step corrected, saved | −4.97e−13 | −1.78e24 |
| 100-step corrected, saved | −1.95e34 | 0.0069 |
| 3-step diagnostic | 6.36e−33 | 6.36e−33 |
| 3-step verification (after fix) | 3.9e−37 | **2.62e38** |

The `[GATE]` prints in `setup_vla_model_and_tokenizer` show the value is
already wrong immediately after `from_pretrained`, before DeepSpeed or the
Trainer exist.

Root cause:

- `from_pretrained` builds the model under `init_empty_weights()`
  ([`modeling_utils.py:4494`](reference/transformers/src/transformers/modeling_utils.py#L4494)),
  so the `torch.tensor(0.)` initializer is discarded and the parameter is
  created on the meta device.
- Neither gate exists in the Qwen checkpoint, so both are missing keys.
- `PreTrainedModel._init_weights` does not handle bare `nn.Parameter`s; its
  docstring says a model adding one must override it
  ([`modeling_utils.py:3028-3032`](reference/transformers/src/transformers/modeling_utils.py#L3028-L3032)).
- The missing keys are therefore materialized from uninitialized memory.

Consequences depended on the garbage value:

- Large (1e16–1e38): `factor * z` swamps the UNet input; the first GroupNorm
  normalizes the scale away, `ross_loss` sits at ~1.0, and the gradient with
  respect to the gate vanishes. The gate stays at that value for the whole run
  (confirmed: identical across all 100 steps of the Step 10 rerun).
- Tiny (≈1e−33): conditioning is effectively off, and the gates *do* move.

The conditioning tensors themselves were healthy throughout:
`z_absmean ≈ 0.18`, `za_absmean ≈ 0.23`.

Fix (`c024db1`): zero both gates explicitly after `from_pretrained`, restoring
the intended zero-init.

Verification — 3 steps after the fix (`logs/ross_gate_fixed_20261020`):

| Stage / step | `factor` | `factor_a` | `cond_absmean` | `ross_loss` |
|---|---|---|---|---|
| after `from_pretrained` | 3.9e−37 | 2.6e+38 | — | — |
| after zero-init | 0 | 0 | — | — |
| step 1 | 0 | 0 | 0 | 0.0019 |
| step 3 | −2.43e−06 | −6.89e−07 | 4.7e−07 | 0.0064 |

Both gates now start at zero and begin moving, so the gating is trainable.

The temporary `[GATE]` stage prints and the per-step `z_absmean` /
`za_absmean` / `cond_absmean` logging are still in the tree; remove them once
the fix is confirmed over a longer run.

---

## Step 12 — Throughput baseline (batch-size sweep)

Measured on 8 × Ascend 950DT with the current harness (ZeRO-3 CPU offload,
sdpa, bf16, gradient checkpointing, `model_max_length=4096`, 504×280 images,
synthetic 4,096-scene set). Code: `fbae548` (per-step timing instrumentation).

Sizes 2–16 ran 50 steps; 32–68 ran 20 steps (ceiling scan). Metrics are
per-step medians from the steady-state portion (first 5 steps dropped);
`step_time_s` is measured around a device sync.

| per-device bs | global batch | step median | samples/s median | samples/s max | peak HBM/device |
|---|---|---|---|---|---|
| 2 | 16 | 8.20 s | 1.96 | 2.02 | — |
| 4 | 32 | 8.76 s | 3.67 | 3.96 | — |
| 8 | 64 | 9.65 s | 6.64 | 7.20 | — |
| 16 | 128 | 11.39 s | 11.29 | 11.87 | — |
| 32 | 256 | 15.93 s | 16.14 | 16.60 | 96,550 MB (98.2%) |
| 64 | 512 | 27.96 s | 18.42 | 19.87 | 98,214 MB (99.9%) |
| 66 | 528 | 26.95 s | 19.75 | 20.66 | 93,471 MB (95.1%) |
| **68** | **544** | **27.11 s** | **20.49** | **21.32** | **98,285 MB (100.0%)** |
| 70 | 560 | OOM after 5 steps | — | — | 98,293 MB |
| 128 | 1024 | OOM at allocation (30.7 GiB) | — | — | — |

**Baseline:** the current harness peaks at **≈20.5 samples/s** (bs=68, global
batch 544). Recommended operating point is **bs=66** (19.75 samples/s, ~5 %
HBM headroom) or **bs=64** (18.42 samples/s) for longer runs. (Superseded by
Step 13 onward — these are ZeRO-3 + CPU-offload numbers.)

Notes:

- `bs=1` is not testable — the ROSS forward squeezes away the batch dim.
- Throughput flattens above bs=32: doubling the batch buys ~14 % (32→64) and
  ~11 % (64→68). The step is dominated by fixed per-step cost (ZeRO-3 CPU
  optimizer, VAE encode, SD1.5 UNet); activation memory limits the batch, not
  parameter count.
- `bs=70` completed 5 steps before OOM, so fitting at step 1 does not prove a
  size is stable. HBM figures are 20 s maxima and under-report transients
  (bs=64 read 99.9 % while bs=66 read 95.1 %), so treat everything above
  bs=64 as sitting at the wall.
- Each run writes an 18 GB final model; the harness has no skip-save flag.

Run directories: `logs/bs_sweep50_20261020/`, `logs/bs_ceiling_20261020/`,
`logs/bs_fine_20261020/`. `logs/bs_sweep_20261020/` is a killed 12-step
attempt and should be ignored.

---

## Step 13 — Throughput optimization (50-step protocol)

All runs: 8 × Ascend 950DT, **50 steps**, logged `train_samples_per_second`
(HF Trainer's wall-clock average, which includes data loading), same data and
recipe. Code: `fbae548` (per-step timing) and `c6f95aa` (ZeRO-2 config),
direct-copied onto a `23e62ee` HEAD.

| config | bs | workers | runtime | samples/s | vs baseline |
|---|---|---|---|---|---|
| ZeRO-3 + CPU offload | 68 | 12 | 1699.3 s | 16.006 | — |
| ZeRO-3 no offload | 48 | 12 | 980.2 s | 19.588 | +22% |
| ZeRO-3 no offload | 48 | 24 | 923.0 s | 20.803 | +30% |
| ZeRO-2 no offload | 24 | 12 | 453.0 s | 21.193 | +32% |
| ZeRO-2 no offload | 32 | 12 | 590.9 s | 21.663 | +35% |
| **ZeRO-2 no offload** | **24** | **24** | **441.4 s** | **21.750** | **+36%** |
| ZeRO-2 no offload | 32 | 24 | 601.5 s | 21.281 | +33% |

Findings:

- **Removing CPU offload is the dominant win.** The fixed per-step cost falls
  from ~27 s at bs=68 (offload) to ~3.5 s at bs=8 (no offload). The batch
  ceiling drops from 68 to 48 (ZeRO-3) / 32 (ZeRO-2) because parameters and
  Adam state move into HBM, but throughput still rises.
- **ZeRO-2 beats ZeRO-3.** Parameters are replicated, but there is no per-step
  parameter all-gather. At equal batch ZeRO-2 is consistently faster.
- **Gradient checkpointing is required at useful batch sizes.** Under this
  ZeRO-3 config, disabling it OOM'd at bs=8 on step 2 (88.5 GB allocated, then
  a 46 MB allocation fails). Under the later ZeRO-2/fa2/valid-CE config it fits
  only at bs=8 and is slower there — see Step 31.
- **More dataloader workers help, up to CPU saturation.** 8 ranks × 24 workers
  = 192 cores exactly; at bs=32 that oversubscribes and regresses (21.66 →
  21.28), while at bs=24 it improves (21.19 → 21.75).
- The last three rows are within ~2 % run-to-run noise.
- Remaining gap: `step_time_s` 7.36 s vs wall 8.83 s at the best config — ~1.5 s
  per step is data loading, not compute.

New defaults in
[`scripts/scripts_train/train_qwen_vla_navsim.sh`](scripts/scripts_train/train_qwen_vla_navsim.sh):
ZeRO-2 without offload (`scripts/sft/zero2.json`), per-device batch 24, 24
dataloader workers. `--zero_stage 2|3` now maps to `scripts/sft/zero{2,3}.json`;
the `*_offload.json` configs remain available via `--deepspeed_config`.
Step 14 supersedes the batch and attention defaults (fa2, batch 40).

The Step 12 numbers (ZeRO-3 + offload) are superseded by this table.

---

## Step 14 — Ascend fused ops and the FA2 baseline

The reference packages (`reference-packages/MindSpeed-MM`,
`reference-packages/DrivingSDK`) were reviewed for native Ascend replacements.
Neither is adoptable as a dependency: MindSpeed-MM has no DeepSpeed path (FSDP2
/ Megatron only) and its Qwen2.5-VL examples are absent from the checkout;
DrivingSDK's kernels are BEV / point-cloud ops, and its Qwen2.5-VL acceleration
is itself a set of manual `torch_npu` patches. The useful pieces are the thin
`torch_npu` wrappers, so this pass calls them directly.

**Metric.** `samples_per_s` = global batch ÷ `step_time_s`, where
`step_time_s` is the device-synced compute step. It excludes data loading; in
a long run (order 10^6 steps) that fixed contribution is negligible, so this
is the throughput figure used below.

**Result** — 50 steps each, ZeRO-2 without offload, bf16, gradient
checkpointing, bs=24 unless noted; per-step median after dropping the first 5
steps. Code: `9fed329`.

| run | bs | samples/s | vs base |
|---|---|---|---|
| base | 24 | 26.58 | — |
| `ROSS_NPU_RMSNORM` | 24 | 26.79 | +0.8% |
| `ROSS_NPU_SWIGLU` | 24 | 25.72 | −3.2% |
| `ROSS_NPU_ROPE` | 24 | 24.83 | −6.6% |
| `adamw_torch_fused` | 24 | 25.61 | −3.6% |
| base2 (A-B-A anchor) | 24 | 26.24 | — |
| **fa2** | 24 | **45.88** | **+75%** |
| **fa2** | 32 | **47.17** | **+80%** |
| **fa2** | **40** | **48.05** | **+83%** |
| fa2 | 48 | — | OOM at step 2 |
| fa2 | 42 | — | OOM at step 42 |

Base-vs-base drift on this metric is 1.3%, so the three negative leaf-op
results are real but small and none clears the keep threshold; RMSNorm is
noise. **FA2 is the win: +83% at bs=40.**

**New baseline:** ZeRO-2 no offload, `--attn_type fa2`, per-device batch 40
→ **48.22 samples/s** (per-step median over a 200-step run, 6.64 s/step, no
OOM), versus 26.58 for the sdpa baseline. The same config measured 48.05 over
50 steps. These are now the launcher defaults (`ATTN_TYPE=fa2`,
`BATCH_SIZE=40`; Step 17 raises the batch to 64 and enables valid-token CE). The leaf-op flags (`ROSS_NPU_RMSNORM`, `ROSS_NPU_SWIGLU`,
`ROSS_NPU_ROPE`) are left off; they did not pay for themselves. (RMSNorm and
RoPE are superseded by Steps 32/33: +6.8% and +6.3% on torch 2.11 at bs=48.)

Implementation:

- `reference/transformers/src/transformers/models/qwen2_5_vl/npu_fused_ops.py`
  — device-gated `npu_rms_norm` /
  `npu_swiglu` / `npu_rotary_mul` helpers, each behind an env flag and
  returning `None` on CUDA/CPU so the original path runs.
- `utils/train_qwen_vla.py` — `fa2` now maps to `flash_attention_2` and is
  allowed on NPU through the vendored `integrations/npu_flash_attention.py`.
- `reference/transformers/src/transformers/integrations/npu_flash_attention.py`
  — mask cache keyed by sequence length.
  **Hardware constraint:** on this CANN 9.1 / Ascend 950 build,
  `npu_fusion_attention` with `sparse_mode=2/3` fails kernel tiling unless the
  mask is **at least 2048×2048**, even for shorter sequences; the mask is
  clamped up to 2048 and grows for longer inputs.
- Launcher — `--optim` (e.g. `adamw_torch_fused`) and the `fa2` path.
- `tools/ascend_parity/check_fused_ops.py` — forward/backward parity for every
  op vs the torch reference; all pass at ≤3.3e-3 relative (bf16), including
  FA2 vs SDPA at 2.6e-3.

Commits: `5bd6f6b` (ops + FA2), `9fed329` (>=2048 mask).

200-step stability check at fa2/bs=40 (`logs/fa2_stab_20261021`): **passed**,
200/200 steps, no OOM, median **48.22 samples/s** (6.64 s/step). Larger
batches are not stable: bs=42 OOMs at step 42, bs=48 at step 2.

---

## Step 15 — L1 profiling collector

Ported the emu3 `NpuProfilerCallback` to the Qwen path. Opt-in via
`--npu_profiling`, with `--npu_profiler_level` selecting:

- level 0: NPU activity only;
- level 1 (L1): CPU + NPU activities, `record_shapes=True`,
  `ProfilerLevel.Level1`.

The callback starts on `on_train_begin`, calls `prof.step()` on `on_step_end`
and stops on `on_train_end`. Schedule: `wait=1, warmup=1, active=1, repeat=1,
skip_first=26`, so a trace is emitted after ~step 29. Traces land in
`<output_dir>/npu_profile_level1/`.

Run `logs/profile_l1_20261021` (fa2, bs=40, 32 steps, `save_strategy no`):
8 rank directories, 2.6 GB, each `ASCEND_PROFILER_OUTPUT/` containing
`trace_view.json` (~159 MB), `op_statistic.csv`, `operator_details.csv`,
`kernel_details.csv`, `api_statistic.csv`, `step_trace_time.csv`,
`communication.json` and the msprof databases.

Traced step 28 (`step_trace_time.csv`):

| metric | time |
|---|---|
| Computing | 6.239 s |
| Communication (not overlapped) | 0.373 s |
| Overlapped | 0.000 s |
| Free | 0.322 s |
| total | 6.933 s |

Per-kernel breakdown (rank 0, step 28, 6.98 s kernel time):

| category | time | share |
|---|---|---|
| MatMul (LLM + ViT + UNet) | 3.80 s | 54% |
| Elementwise/layout (Mul, Add, Cast, Transpose, ReduceSum, Square, Concat) | ~1.06 s | 15% |
| Communication (`HcclLaunch` 0.370, allReduce 0.088, allGather 0.056) | 0.74 s | 11% |
| FlashAttention fwd+bwd | 0.44 s | 6% |
| Conv (UNet/VAE/ViT) | 0.40 s | 6% |
| LM head matmuls | 0.13 s | 2% |
| LogSoftmax + NLL fwd/bwd | 0.02 s | 0.3% |

Findings:

- FA2 cut attention to ~6%; matmuls dominate, so the remaining compute is the
  LLM/ViT/UNet body, not the head.
- The 151k-vocab head is ~133 ms/step (forward + two backward matmuls) and the
  CE kernels ~21 ms. Valid-token CE is therefore **a memory lever** (it removes
  the `[16920, 152064]` fp32 `logits.float()` transient, ~10.3 GB per rank),
  not a large compute lever.
- Communication: 389 collectives during backward (0.1–1.7 ms each) plus
  `HcclLaunch` ~0.370 s. The step tail is 7.8 + 1.0 + 22.8 + 17.4 ms of
  allreduce followed by a 56.0 ms all-gather. Overlap is zero.
- The tail all-gather is ZeRO-2's reassembly of updated parameter shards. It is
  a step-boundary operation and cannot be overlapped in ZeRO-2.
- Rank-to-rank differences in collective duration are arrival-time skew
  (stragglers), not uneven gradient volume: this is pure data parallelism with
  replicated parameters, not DDP and not model parallelism. The 56 ms
  all-gather is 0.8% of the step, not a redesign target.

Profiler overhead: the run averaged 19.5 s/step while profiling was active;
only the active window (step ~28) is traced, and the per-kernel timings above
come from that step.

Commits: `c01713b` (callback + launcher flags).

---

## Step 16 — Ascend runtime env A/B

Env-only variants on the Step 14 baseline (fa2, bs=40, ZeRO-2, 50 steps,
`ASCEND_LAUNCH_BLOCKING=0`). Metric: per-step `samples_per_s` median after
dropping the first 5 steps. Runs: `logs/env_ab_20261022`,
`logs/env_ab2_20261022`.

| tag | env | samples/s | vs base |
|---|---|---|---|
| base | `HCCL_DETERMINISTIC=TRUE` | 47.94 | — |
| comb | + `COMBINED_ENABLE=1` | 48.02 | +0.2% |
| msr | + `MULTI_STREAM_MEMORY_REUSE=2` | 48.20 | +0.6% |
| tq1 | + `TASK_QUEUE_ENABLE=1` | 48.14 | +0.4% |
| detoff | no `HCCL_DETERMINISTIC` | 48.16 | +0.5% |
| nodet | no determinism + `COMBINED_ENABLE` + `MULTI_STREAM_MEMORY_REUSE` + `expandable_segments` | 48.00 | +0.1% |
| base2 | anchor | 48.14 | +0.4% |
| tq2 | `TASK_QUEUE_ENABLE=2` | — | failed before training (empty run dir) |

Conclusion: none of these runtime knobs changes throughput beyond the 0.4%
anchor drift. `TASK_QUEUE_ENABLE=2` is unusable on this build — it fails before
model load, leaving an empty run directory; value 1 works but is neutral.
(Superseded on the recreated container: `=2` now runs, still neutral and with
zero overlap — Step 36.)
Determinism-off is also neutral, so `HCCL_DETERMINISTIC` can stay on for
correctness runs at no measurable cost.

The remaining communication cost is structural (Step 15: 389 collectives plus
`HcclLaunch` overhead), which these env switches do not reduce. Reducing the
collective count requires bucket tuning (`reduce_bucket_size`), and the larger
win remains the memory/batch levers.

Re-checked at the then-current best config (`fa2`, bs=64, `ROSS_VALID_CE=1`): the
combined-knobs + determinism-off run measured **50.84 samples/s**, identical to
the baseline's 50.84 (−0.01 %, step 10.07 s both). The conclusion is unchanged
at the new operating point; no further env tuning was done.

---

## Step 17 — Valid-token CE compaction (C1a)

Implemented behind `ROSS_VALID_CE=1` (launcher `--valid_token_ce`). The LM loss
projects only the labeled positions instead of the full `[N, vocab]` matrix and
returns `logits=None` on that path. It falls back to the original path for
`logits_to_keep > 0`, a supplied `shift_labels`, or a batch with no valid
labels.

Why: measured on a real batch, only **5.35 %** of positions are labeled
(181 / 3384 in an 8-sample batch; 22–24 per ~421-token sample). Everything else
is system/user prompt, the two 180-token image blocks, the assistant role
prefix and the turn terminator. The fp32 `logits.float()` transient drops from
`[16920, 152064] × 4 B ≈ 10.3 GB` to `[~914, 152064] × 4 B ≈ 0.55 GB` per rank.

Throughput (per-step `samples_per_s` median, first 5 steps dropped):

| run | bs | steps | samples/s | step median |
|---|---|---|---|---|
| base20 (C1a off) | 40 | 20 | 48.11 | 6.65 s |
| c1a | 40 | 50 | 49.38 | 6.48 s |
| c1a | 48 | 50 | 50.10 | 7.67 s |
| c1a | 56 | 50 | 50.23 | 8.92 s |
| c1a | 64 | 50 | 50.84 | 10.07 s |
| c1a | 72 | 50 | 50.88 | 11.32 s |
| **c1a** | **64** | **200** | **50.91** | **10.06 s** |

Findings:

- C1a at bs=40: +2.6 %. The profile predicted this — the LM head is ~2 % of
  the step, so the saving is memory, not compute.
- The batch ceiling moves from 40 to at least 72 for 50 steps (bs=42 previously
  OOM'd at step 42).
- Throughput is flat above 64: step time scales almost linearly with batch
  (6.48 → 11.32 s for 40 → 72). The run is now compute-bound.
- bs=72 is not stable over 200 steps: that run died at the first logging HCCL
  `all_gather` on rank 3 (error 207001) while the same config passed 50 steps.
  bs=64 completed 200/200.
- **New best: C1a + bs=64 → 50.91 samples/s** (+5.6 % over the 48.22 Step 14
  baseline). These are now the launcher defaults: `ATTN_TYPE=fa2`,
  `BATCH_SIZE=64`, `VALID_TOKEN_CE=true` (export `ROSS_VALID_CE=0` to disable).

Parity (20 steps, unshuffled, identical start):

- losses are bit-identical at steps 0–1 and diverge to ~1 % by step 19 (max
  7.4e-2 at one step);
- weights after 20 steps: relL2 2.5e-4 (UNet `conv_in`) to 6.6e-3 (layer0
  `q_proj`), max abs ~1e-3 (bf16 ulp); gate scalars max abs 6e-4. No structural
  difference — bf16 reduction-order noise amplified by the optimizer.

Commits: `5f163a6`.

---

## Step 18 — Elementwise-op fusion plan (add+RMSNorm, GN+SiLU, SiLU)

Profile hot spots (per rank per step, supplied in-session; same shape as the
Step 15 L1 trace):

| # | op | core | calls | ms | share |
|---|---|---|---|---|---|
| 1 | MatMulV3 | AI_CORE | 2,000 | 3,200.5 | 48.43% |
| 2 | MatMulV3 | MIX_AIC | 1,835 | 596.3 | 9.02% |
| 3 | Mul | AI_VECTOR | 2,840 | 428.4 | 6.48% |
| 4 | HcclLaunch | AI_CPU | 389 | 370.8 | 5.61% |
| 5 | FlashAttentionScoreGrad | MIX_AIC | 92 | 228.7 | 3.46% |
| 6 | FlashAttentionScore | MIX_AIC | 185 | 193.1 | 2.92% |
| 7 | Cast | AI_VECTOR | 4,988 | 220.8 | 3.34% |
| 8 | Add | AI_VECTOR | 2,096 | 216.3 | 3.27% |
| 9 | Conv2DV2 | AI_CORE | 112 | 169.3 | 2.56% |
| 10 | Transpose | AI_VECTOR | 1,035 | 119.0 | 1.80% |

Compute is 66% (MatMul 57.5% + FA 6.4% + Conv 2.6%); elementwise/layout is
14.9%; communication 5.6%. **Ceiling:** removing every elementwise op is ~15%;
realistically 3–6% after fusion overhead.

### Why Cast / Add / Mul are this large

1. **RMSNorm fallback** (`Qwen2RMSNorm`): `to(fp32) → pow → mean → rsqrt →
   mul → to(bf16) → *weight` = 2 Cast + 3 Mul + 1 Add + 1 Reduce per call.
   ~177 norm calls/step (LLM 28×2 with checkpoint replay, ViT 32×2, `ln_q`).
2. **RoPE**: `q*cos + rotate_half(q)*sin` plus `cat`/neg = 4 Mul + 2 Add per
   layer per q/k; ~240 Mul + 120 Add per step.
3. **SwiGLU**: `silu(gate)*up` = Sigmoid + 2 Mul per MLP; 28 layers × replay.
4. **Residual/bias adds**: 2 residual adds per block plus q/k/v biases across
   LLM, ViT and UNet ≈ 700–900 Adds.
5. **SD1.5 UNet**: GroupNorm computes in fp32 (2 Casts + affine Mul/Add per
   norm) and every SiLU is `x*sigmoid(x)`.
6. **ROSS fp32 block**: `autocast(fp32)` plus `.float()` on image/action
   hidden and `z_q` at `modeling_qwen2_5_vl_ross.py:355-360`.
7. **Optimizer**: `adamw_torch` runs the per-tensor path over ~1,000+
   parameter tensors with bf16 params + fp32 master weights — likely the
   largest *count* bucket of Cast/Add/Mul (parameter-shaped, memory-bound).

Verify the split from the L1 `operator_details.csv`: parameter-shaped rows are
optimizer, activation-shaped rows are the model.

### Kernels to migrate

| op | replaces | call site | status |
|---|---|---|---|
| `npu_add_rms_norm` | residual Add + post-attention RMSNorm | LLM decoder layer (1 of 2 norms/layer) | **new**, flag `ROSS_NPU_ADD_RMSNORM=1` |
| `npu_group_norm_silu` | GroupNorm + SiLU | diffusers `ResnetBlock2D` norm1/norm2 pairs | **new**, flag `ROSS_NPU_GN_SILU=1` |
| `npu_silu` | `x*sigmoid(x)` | LLM/ViT MLP activation, UNet SiLU | **new**, flag `ROSS_NPU_SILU=1` |
| `npu_rms_norm` | whole RMSNorm fallback | all norms | already wired, +0.8% (Step 14) |
| `npu_swiglu` | `silu(gate)*up` | LLM/ViT MLP | already wired, −3.2% (Step 14) |
| `npu_rotary_mul` | rotate_half + 2 Mul + Add | q/k RoPE | already wired, −6.6% (Step 14) |
| `npu_apply_adam_w` | per-tensor AdamW kernels | optimizer (`--optim adamw_torch_fused`) | retest on 2.11; was −3.6% |
| `npu_confusion_transpose` | transpose+reshape | attention/vision layouts | deferred (≤1%, layout risk) |
| `npu_fast_gelu` / `npu_layer_norm_eval` | — | — | not applicable (SiLU model; eval-only) |

**Feasibility gate:** several torch_npu fused ops are inference-only. The new
`probe_backward_support()` in `npu_fused_ops.py` runs one forward+backward per
op before training and disables any op whose backward fails. The setup logs
`[fused] backward support: {...}` and the fused ResNet block count.

### Implementation

- `reference/transformers/.../qwen2_5_vl/npu_fused_ops.py` — `fused_add_rms_norm`,
  `fused_silu`, `fused_group_norm_silu`, `probe_backward_support`,
  `fuse_unet_group_norm_silu` (patches `ResnetBlock2D` instance methods so the
  state_dict keys are unchanged).
- `modeling_qwen2_5_vl.py` — decoder layer uses add+RMSNorm; both MLP forwards
  use `fused_silu` when SwiGLU is off.
- `utils/train_qwen_vla.py` — probe + UNet GN+SiLU fusion after the UNet loads.
- `tools/ascend_parity/check_fused_ops.py` — forward/backward parity for the
  three new ops.

### A/B protocol

50 steps each, same seed/data/config as Step 17 (fa2, bs=64, valid CE, ZeRO-2,
`adamw_torch`); metric = per-step `samples_per_s` median after dropping the
first 5 steps.

| run | env |
|---|---|
| base | no new flags |
| add_rms | `ROSS_NPU_ADD_RMSNORM=1` |
| gn_silu | `ROSS_NPU_GN_SILU=1` |
| silu | `ROSS_NPU_SILU=1` |
| all | all three |
| adamw_fused | `--optim adamw_torch_fused` on the best of the above |

### Feasibility result (2026-10-23, torch_npu 2.11 / CANN 9.1.0)

`tools/ascend_parity/check_fused_ops.py` on the old box:

| op | forward | backward |
|---|---|---|
| `npu_add_rms_norm` | PASS (rel 4.4e-3) | **unavailable** — `derivative for npu::npu_add_rms_norm is not implemented`. The `v2_functional` variant can be trained with a hand-written backward (Step 36) |
| `npu_group_norm_silu` | returns `(y, mean, rstd)` | **unavailable** — no autograd kernel; backward leaves `grad=None` |
| `npu_silu` | **unavailable** — `Current device only support aclnn operator, but ... npu_silu do not have aclnn implementation` | — |
| `npu_rms_norm` / `npu_swiglu` / `npu_rotary_mul` | PASS | PASS |

All three new gates therefore stay disabled at runtime; the probe treats
`grad is None` as failure, and the helpers fall back to the reference path.
No A/B runs were launched for them: the flagged code would be identical to the
base run.

The only remaining lever from the profile analysis was the optimizer.
`--optim adamw_torch_fused` measured **50.60 samples/s / 10.13 s** median
(steps 6–50) versus **50.85 / 10.08 s** for `adamw_torch` — −0.5%, within
run-to-run noise. No win; the launcher default stays `adamw_torch`.
Run: `logs/oldbox_torch211_adamwfused_20261023`.

The existing fused ops keep their Step 14 verdicts: RMSNorm neutral (+0.8%),
SwiGLU −3.2%, RoPE −6.6%, so none is enabled by default. (Superseded for
RMSNorm and RoPE by Steps 32/33: +6.8% and +6.3% on torch 2.11 at bs=48;
SwiGLU stands.)

---

## Step 19 — Batch-size stability sweep

Step-time outliers (one 12–19 s step per 50) motivated a stability sweep.
Protocol: 50 steps, same config as Step 17; per-step `step_time_s` over steps
6–50; mean/std/CV plus the count of steps > 1.5× the median.

| bs | mean (s) | std | CV | median | max | outliers |
|---|---|---|---|---|---|---|
| 64 | 10.462 | 1.029 | 9.84% | 10.054 | 14.24 | 0 |
| **62** | 9.887 | 0.564 | **5.71%** | 9.749 | 13.20 | 0 |
| 60 | 9.770 | 1.145 | 11.72% | 9.481 | 16.50 | 1 |
| 58 | 9.633 | 1.440 | 14.95% | 9.204 | 16.93 | 1 |
| 56 | 9.166 | 0.907 | 9.89% | 8.877 | 12.52 | 0 |
| **62 (repeat)** | 9.845 | 0.397 | **4.03%** | 9.745 | 12.04 | 0 |

**bs=62 is the most stable** (CV 4.0–5.7% across two runs) and its throughput
median (50.88–50.90 samples/s) is tied with bs=64 (50.93). bs=56 has the
lowest median step time (8.88 s) but roughly twice the variance. Use bs=62 for
long runs that care about step-time stability; bs=64 remains fine for peak
throughput (the difference is noise).

Runs: `logs/bs_stab_20261023/bs{64,62,60,58,56}` and `bs62_rep`.

---

## Step 20 — Real NAVSIM data for training (branch `navsim-real-data`)

The Qwen VLA dataset resolves each pickle path to
`<data_root>/<log>/CAM_F0/<frame>.jpg` and loads only `image[cur_frame_idx]`
plus `pre_1s_image[cur_frame_idx]`. The HF trainval pickle
(`navsim_emu_vla_256_144_trainval_pre_1s.pkl`, 103,288 samples, already
action-normalized) therefore needs only the camera JPEGs — no VQ codes and no
NuPlan.

**Source.** OpenScene v1.1 trainval camera shards (the NAVSIM sensor blobs).
`huggingface.co` resolves to IPv6 only on this box; use `hf-mirror.com`
(the download script defaults to it). Metadata is 7.0 GB; camera shards are
4.7 GB × 200 ≈ **940 GB**.

**Layout gotcha.** Each shard extracts to
`openscene-v1.1/sensor_blobs/trainval/<log>/CAM_F0/<frame>.jpg`. Pass
`.../sensor_blobs/trainval` as `--data_root`, not `.../sensor_blobs`.

**Validate with one shard.**

```bash
bash tools/pickle_gen/download_openscene_camera.sh \
    --out /mnt/sfs_turbo/lmc/navsim_openscene --start 0 --end 0
python tools/pickle_gen/filter_pickle_by_images.py \
    --input /mnt/sfs_turbo/models/DriveVLA-W0/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
    --data_root /mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/trainval \
    --output /mnt/sfs_turbo/lmc/navsim_openscene/navsim_trainval_subset.pkl
```

Shard 0: 22,360 JPEGs, **467 / 103,288** samples kept (both the current and
pre_1s frames exist).

**Smoke run** (`logs/navsim_real_smoke_20261023`, 10 steps, bs=8, real images):
action_loss 20.7 → 11.2, `ross_loss` **0.16–0.26** (versus ~0.002 on the
near-identity synthetic data), 1.66–3.0 s/step. JPEG loading and the 180-token
16:9 resize work.

**Full download.** Single-stream throughput was ~3.2 MB/s (24 min/shard), so
200 shards serially would take ~80 h. Run 4–8 shards in parallel to bring that
to roughly 10–20 h. The 7 GB metadata tarball
(`openscene_metadata_trainval.tgz`) is only needed to regenerate the pickle
from `navsim_logs` with `pickle_generation_navsim_mini.py`; the filtered-HF-
pickle path above does not need it.

**Throughput on the 20× subset (throughput only, not a learning run).**
`navsim_trainval_subset_x20.pkl` = 9,340 entries (467 × 20), bs=62, 200 steps:

| metric | value |
|---|---|
| `step_time_s` median (steps 6–200) | 9.995 s (mean 10.27, CV 13.85%, 6 outliers, max 21.4) |
| samples/s median | 49.62 |
| wall clock | 14.4 s/step (bar), 15.2 s/step incl. final save |
| `train_loss` | 2.08 (overfits the 467 unique scenes; ignore) |

Synthetic bs=62 for comparison: 9.745 s median, CV 4.03%. Real data is ~2.5%
slower in compute and noisier (JPEG loading stalls even with prefetch).

Two host-memory traps with a prefetching real-data loader:
1. The 467-sample subset gives `dataloader len: 1`, so loading cannot overlap
   (48.6 s/step wall). Replicating the list fixes it.
2. `ulimit -l` is 64 MB in this container; `pin_memory` exhausts it
   (`aclrtMallocHostWithCfg` 207001). Raising the limit lets the kernel OOM-kill
   the workers instead. **Superseded by Step 23:** with `ulimit -l unlimited`
   and 8 workers, pinning is safe and is the current canonical setting.

New from `ed9eec9`: every run also logs `step_time_full_s` /
`samples_per_s_full` = forward/backward + `optimizer.step()` + `zero_grad()`
(device-synced), so the wall-clock gap is measured rather than inferred.

**CPU affinity A/B (2026-10-24, 50 steps each, real 20× subset, bs=62):**

| tag | step median | mean | CV | samples/s |
|---|---|---|---|---|
| base (`CPU_AFFINITY_CONF=0`) | 9.821 s | 10.151 s | 12.88% | 50.50 |
| **`CPU_AFFINITY_CONF=1`** | 9.809 s | **9.877 s** | **2.38%** | **50.57** |

Same throughput, ~5× lower step-time variance. `CPU_AFFINITY_CONF=1` is now
part of the canonical run config. (The `step_time_full_s` values were identical
to `step_time_s` in this A/B: the vendored Trainer calls `self.optimizer.step()`
directly rather than `Trainer.optimizer_step`, so the first implementation
never fired. Fixed to time `on_pre_optimizer_step` → `on_step_end`.)

---

## Step 21 — Re-profile on real data (torch 2.11, bs=62)

Run `logs/profile_l1_real_20261024` (real 20× subset, bs=62, fa2,
`CPU_AFFINITY_CONF=1`, 8 workers, `pin_memory=false`, no weights saved),
L1 trace at step 28.

`step_trace_time.csv`: Computing **9.273 s**, Communication (not overlapped)
0.266 s, Free **1.875 s**, total stage 11.414 s. Unprofiled step: 9.8 s
compute, ~15.8 s wall.

| op | ms | share | calls |
|---|---|---|---|
| MatMulV3 (3 kernel forms) | 5,657 | 55.5% | 2,181 |
| HcclLaunch | 655 | 6.4% | 389 |
| Mul | 629 | 6.2% | 2,808 |
| Add | 351 | 3.4% | 2,030 |
| FlashAttentionScoreGrad | 351 | 3.4% | 92 |
| FlashAttentionScore | 300 | 2.9% | 185 |
| Conv2D | 295 | 2.9% | 182 |
| Cast | 243 | 2.4% | 3,320 |
| Conv backward | 284 | 2.8% | 195 |
| Transpose | 120 | 1.2% | 818 |
| SiLU fwd+grad | 122 | 1.2% | 400 |
| Div (RealDiv) | 100 | 1.0% | 274 |
| ReduceSum / Square / Concat | 196 | 1.9% | — |

Findings:

- **MatMul is still 55.5%**; FA 6.3%, Conv 5.6%. Elementwise/layout is 17.3%
  (Step 15: 15%). The compute shape of the step has not changed.
- **The optimizer is only 3.8–5 ms per step** (new callback metric). The earlier
  guess that AdamW dominated the elementwise profile was wrong — the wall-clock
  gap is dataloader/logging, not the optimizer.
- fp32 Mul (365 ms) exceeds bf16 Mul (303 ms): the fp32 RMSNorm/RoPE/LayerNorm
  fallbacks. `npu_rms_norm` targets this but remains neutral (Step 14).
- INT64 Mul: **621 calls, 0.8 ms** — mRoPE/window index arithmetic (see the
  earlier trace analysis); 0.1% of Mul time.
- **Free rose from 0.32 s (synthetic) to 1.88 s (real JPEGs)**: the remaining
  overhead is the data path. With 8 workers the loader still leaves bubbles;
  worker count is constrained by the 64 MB memlock/pin issue.

---

## Step 22 — Three-way profile: batch size vs code revision

All traces: real 20× subset, fa2, `CPU_AFFINITY_CONF=1`, L1 profiler, traced
step 28, CANN 9.1.0 / torch_npu 2.11.0.

| trace | code | bs | Computing | Comm | Free | Stage |
|---|---|---|---|---|---|---|
| A. HEAD | `f78fd40` | 62 | 9.273 s | 0.266 s | 1.875 s | 11.414 s |
| B. HEAD | `f78fd40` | 40 | 6.028 s | 0.279 s | 1.273 s | 7.579 s |
| C. Step 15 | `c01713b` | 40 | 6.195 s | 0.269 s | 0.358 s | 6.822 s |

Kernel shares (A/B/C): MatMul 55.5/53.8/54.3%, FA 6.7/6.5/6.4%, Hccl
6.4/7.0/6.9%, Mul 6.2/5.9/5.8%, Conv 5.7/5.8/5.7%, Add 3.4/3.1/3.1%,
Cast 3.0/3.1/3.2%, Transpose 1.3/1.4/1.4%.

**Batch-size delta (A vs B).** Compute scales linearly with batch
(9.273/6.028 = 1.538 vs 62/40 = 1.55): no per-sample compute advantage at
bs=62. The throughput gain is fixed-cost amortization — communication per
sample 4.3 ms (bs62) vs 7.0 ms (bs40), plus the ~4 ms optimizer step.

**Code delta (B vs C).** HEAD computes 2.7% faster (6.028 vs 6.195 s), from
the valid-token CE compaction (Step 17). The old commit shows much less free
time (0.358 vs 1.273 s), but that arm ran with 2 workers and `pin_memory`
(the only configuration that fit) while HEAD used 8 workers without pinning, so
the free-time difference is data-path configuration, not code. The `c01713b`
run also needed `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` at bs=40:
without it the full-vocab LM loss OOMs (9.5 GiB contiguous request with
62.9 GiB resident).

Saved weights from all previous runs were deleted during this step
(`logs/` 3,046 GB → 13 GB); `--save_model false` keeps it that way.
Traces are archived in `logs/profile_traces_20261024/` (see the README there).

---

## Step 23 — On-stage free time

All runs: HEAD code, real 20× subset, fa2, affinity on, L1 trace, step 28
(where applicable); bs=40 unless the row says bs=62.

| variant | Computing | Comm | Free | Stage |
|---|---|---|---|---|
| unpinned, 8 workers | 6.028 s | 0.279 s | **1.273 s** | 7.579 s |
| `pin_memory`, 2 workers | 6.061 s | 0.230 s | 0.420 s | 6.710 s |
| `pin_memory`, 8 workers | 6.052 s | 0.271 s | **0.347 s** | 6.670 s |
| `pin_memory`, 8 workers, gate diagnostics removed + `ROSS_STEP_TIMING=0` | 6.054 s | 0.250 s | 0.349 s | 6.652 s |
| `pin_memory`, 8 workers, **bs=62** | 9.227 s | 0.249 s | 0.521 s | 9.996 s |

Findings:

- **Pinned host memory is the free-time lever**: 1.273 → 0.347 s (−73%) and
  stage 7.579 → 6.670 s (−12%), with compute unchanged. Pageable H2D stalls
  the device between kernels; pinned buffers DMA asynchronously.
- Removing the five scalar gate `.cpu()` syncs and the step-boundary timing
  syncs did **not** change on-stage free (0.347 → 0.349 s, noise). The gate
  syncs are microseconds (scalars); the timing syncs sit outside the traced
  step. They still matter for host-side overlap/wall, not for free.
- Pinning requires `ulimit -l unlimited` (64 MB memlock) and a bounded worker
  count; 24 workers with pinning OOM'd the host earlier. 8 workers is safe at
  bs=40; bs=62 with pinning was not yet profiled.
- The residual ~0.35 s is launch/dispatch and profiler overhead. The only
  remaining on-stage lever is fewer kernels (fusion), which Step 14 measured
  as neutral or negative on this stack.
- Note: pinning lowers on-stage free but the wall was ~unchanged — the idle
  moves off-stage (dataloader wait). Wall reduction needs faster data delivery,
  not more sync removal.

**Per-sample on-stage (pinned):** bs=40 → 6.670/40 = **166.8 ms**; bs=62 →
9.996/62 = **161.2 ms**. bs=62 is ~3% better because comm/launch/fixed costs
amortize; free per sample is flat (8.7 vs 8.4 ms) and computing per sample is
148.8 vs 151.3 ms. The canonical config is therefore bs=62 + pinning.
(Superseded by Step 29: bs=48 with the same pinning.)

---

## Step 24 — Fixed input length and the FLOPs counting input

Prerequisite for the FLOPs/MFU work (design:
[`docs/superpowers/specs/2026-10-24-flops-counting-design.md`](docs/superpowers/specs/2026-10-24-flops-counting-design.md)).
Per-step FLOPs depend on the padded sequence length, which the collator sets to
the batch max, so the number drifts with batch composition. This step measures
the real length and pins it.

**Method.** Ran the dataset's own `_get_item` over every sample with the
canonical arguments (`cur_frame_idx=3`, `future_nums=8`, `pre_future_nums=2`,
`use_previous_actions=false`, `max_pixels=150528`), text and FAST actions
tokenized for real, image loading stubbed to the verified real grid
(`grid_thw=[1,20,36]` → 180 tokens; confirmed by loading one real image).
Zero failures across 112,628 samples.

| dataset | n | min | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| `navsim_trainval_subset_x20.pkl` | 9,340 | 408 | 413 | 416 | 419 | **420** |
| `navsim_emu_vla_256_144_trainval_pre_1s.pkl` | 103,288 | 407 | 413 | 416 | 419 | **423** |

Findings:

- 360 of the ~413 tokens are the two image blocks; text + action + template is
  only ~48–63 tokens. The distribution is tight for that reason — nothing like
  the `model_max_length=4096` truncation cap.
- The x20 subset is exactly `data[:467] * 20`; every per-length count is a
  multiple of 20. The longest scene is token `6a185e2130c75743` at index 356,
  replicated at `356 + 467k`, so there is only **one unique scene at L=420**.
- **Decision: fix L=420** for the canonical set (424 if the full pickle is ever
  used). Versus the median sample that is +1.7% tokens and +3.4% attention work
  — negligible, and it removes run-to-run variation.

**Counting input artifact.** `navsim_flops_L420.pkl` (not committed;
regenerate with the recipe below) contains only the L=420 scene, replicated to
`8 ranks × 62 × 20 steps = 9,920` entries. Verified: every measured length is
420 and each rank gets 1,240 samples = exactly 20 batches of 62, so no short
tail batch changes the shape.

```python
import pickle
src = "/mnt/sfs_turbo/lmc/navsim_openscene/navsim_trainval_subset_x20.pkl"
d = pickle.load(open(src, "rb"))
P = len(d) // 20                                             # 467
sel = [d[i] for i in range(len(d)) if i % P == 356 % P]       # the 20 L=420 copies
pickle.dump(sel * 496, open(".../navsim_flops_L420.pkl", "wb"))
```

Caveats: all samples are identical, so this pickle's **wall** time is
meaningless (page-cache-hot image loads) — never use it for learning. Its
per-step **compute** time is valid, because every step has the same shapes;
that is the Step 27/29 throughput basis. FLOPs depend only on shapes, so it is
sound as a counting instrument. The total must be a multiple of
`world_size × per_device_batch_size`.

Related: at L=420 with bs=62 the eager LLM score matrix is `62 · 28 · 420² ≈
3.1e8` elements ≈ 612 MB bf16, so an eager, production-shaped counting run is
memory-feasible; the timed run stays on FA2.

---

## Step 25 — First FLOPs count (fixed-length batch, `f404d21`)

> **Superseded by Step 26.** These numbers cost the vision attention as one
> dense matrix over the concatenated patch sequence; the counter now segments
> by `cu_seqlens`, and the protocol is one step at bs=2 (×31) rather than 20
> steps at bs=62. Kept for the record.

Run `logs/flops_count_20261024_v3`: 8 × Ascend 950DT, bs=62, L=420,
`ROSS_COUNT_FLOPS=1`, `ROSS_VALID_CE=1`, fa2, ZeRO-2, 20 steps, no profiling,
input `/mnt/sfs_turbo/lmc/navsim_openscene/navsim_flops_L420.pkl`.

| quantity | per card per step |
|---|---|
| `flops_mm` (Linear + Conv, 3× / 1×) | **1.4757e15** |
| `flops_attn` (causal variant, 3× / 1×) | **3.9249e15** |
| `flops_total` | **5.4006e15** |
| cumulative over 20 steps | 1.0801e17 (108.01 PFLOPs/card) |

`seq_len` logged 420 on all 20 steps and every FLOPs value is bit-identical
step to step, as expected for a fixed-shape batch.

**Decomposition.** Attention is 72.7 % of the total and 99.8 % of that is the
vision tower, counted dense over the concatenated patch sequence (the accepted
over-count from Step 24): 32 blocks × `4·16·80·89,280²` ≈ 1.306e15 forward
(×3 = 3.918e15). The LLM contributes 6.6e12 (causal-halved). Segmented by
`cu_seqlens` the vision term would be ~6.1e12 forward — ~214× smaller — and the
total would fall from 5.40e15 to ~1.49e15. `flops_attn == flops_attn_full` to
five significant figures, because the causal halving only touches the LLM's
0.2 % share.

**Findings.**

- **Single-rank counting does not fit at bs=62.** At `world_size=1` ZeRO-2 must
  hold the full gradient partition; backward OOMs allocating 17.05 GiB with
  82.47 GiB already resident. The 8-rank layout is what makes bs=62 fit. Since
  the counter is per-rank and every rank sees the same shapes, the 8-rank run is
  the practical way to count — read rank 0's log.
- **`ROSS_VALID_CE=1` is mandatory** at bs=62. Without it the full
  `[26040, 152064]` fp32 logits transient (≈15.8 GiB) OOMs. It is also the
  production setting, so the count matches what actually runs.
- The earlier async-mode A/B failure was the same missing flag, not the feature:
  it ran without `ROSS_VALID_CE=1` under the L1 profiler and hit the LM-head OOM.
- Step time was **not** logged in this run: the shared tmux shell still had
  `ROSS_STEP_TIMING=0` exported from Step 23. That is fine — the counting run's
  own time is the native-path time and must not be used for MFU; pair
  `flops_total` with a production run's `step_time_full_s`.

---

## Step 26 — Segmented attention, one step at bs=2 (`a350933`)

Run `logs/flops_b2_buckets_20261024`: 8 × 950DT, per-device bs=2, **one step**,
`ROSS_COUNT_FLOPS=1`, `ROSS_VALID_CE=1`, fa2, ZeRO-2, fixed-length pickle.

| per card, bs=2, 1 step | ×31 → bs=62 |
|---|---|
| `flops_mm` 4.7603e13 | **1.4757e15** |
| `flops_attn` 6.3435e11 | **1.9665e13** |
| `flops_attn_full` 8.4677e11 | **2.6250e13** |
| `flops_total` 4.8237e13 | **1.4954e15** |

The ×31 mm value is exactly the bs=62 value from Step 25's run — batch linearity
at fixed L holds to the digit, so one step at bs=2 replaces 20 at bs=62.

**Bugs found and fixed on the way** (all `a350933` / `865ef27`):

- **Attention was not counted at all.** The Qwen decoder calls
  `self.self_attn(hidden_states=..., ...)` **by keyword**, and the hook only
  read positional args, so all 28 LLM attention modules were silently skipped.
  `flops_attn == flops_attn_full` was the symptom (only non-causal vision/UNet
  attention was counted). The hook now accepts `kwargs["hidden_states"]`, with a
  self-check case for the keyword form.
- **Vision attention was densified.** Now costed by `cu_seqlens`: per-image
  segments (720 patch tokens) for the 4 full-attention blocks and 64-token
  windows for the other 28. Attention drops from 3.92e15 to 1.97e13 at bs=62.

**Reading the diagnostic prints.** `[flops] by class` and `[flops] buckets`
accumulate over the whole run and the run logged two forwards (the print is 2×
the per-step value), so use the logged `flops_*` keys, not the prints. The
logged per-step value is anchored: 4.7603e13 = 2 samples × 418 tokens ×
2 × 28.45e9 effective params.

**Single rank** still does not fit: at `world_size=1` ZeRO-2 holds the
unsharded optimizer state and gradient partition regardless of batch size. Use
8 ranks; the counter is per-rank and identical.

---

## Step 27 — Best config on the fixed-length set + MFU (`7109a22`)

Run `logs/bestcfg_flopsdata_20261024`: canonical config (bs=62, fa2, ZeRO-2,
8 workers, pinning, `CPU_AFFINITY_CONF=1`, `ROSS_VALID_CE=1`, no counting, no
profiler), 100 steps on `navsim_flops_L420.pkl`.

| metric | value |
|---|---|
| `step_time_full_s`, steps 20–70, **mean** | **10.858 s** |
| median (stable plateau) | 9.662 s |
| min / max | 9.651 / 19.961 s |
| outliers > 1.5× median | 7 / 51 (14.5–20.0 s) |
| samples/s per rank at the plateau | 51.3 |

Most steps are 9.65–9.67 s; seven spikes inflate the mean (CV 24% vs the 4–5%
Step 19 measured on the real subset). Host load during the run was ordinary
(44/192 cores, 973/1502 GB), so the spikes are device/collective skew.

**MFU** with `flops_total = 1.4954e15` per card per step (Step 26) and peak
**486 TFLOPS**:

| step time used | MFU |
|---|---|
| mean 10.858 s | **28.3 %** |
| median 9.662 s | **31.8 %** |

## Step 28 — Communication/compute overlap A/B (`7109a22`)

The earlier async A/B failure (Step 25 note) was **not** the env: it ran without
`ROSS_VALID_CE=1` and OOM'd on the LM-head logits. Facts about the knob on
torch_npu 2.11:

- `torch.npu.set_async_mode` does **not exist** (removed). `torch.npu.SyncLaunchStream`
  docstring confirms the default stream *does* enable the task queue.
- `torch_npu/_init/core/optional_features.py` forces `TASK_QUEUE_ENABLE=0` only
  on an **interactive** command line, so torchrun scripts already run async.
- The lever is therefore `TASK_QUEUE_ENABLE=0|1`; `set_option` rejects it.

A/B: canonical config, `navsim_flops_L420.pkl`, 32 steps, L1 trace at step 28,
`ROSS_STEP_TIMING=0`.

| metric (step 28) | sync (`TASK_QUEUE_ENABLE=0`) | async (`=1`) |
|---|---|---|
| Computing | 9.227 s | 9.218 s |
| Communication (not overlapped) | 0.215 s | 0.205 s |
| **Overlapped** | **0.000 s** | **0.000 s** |
| Free | 0.400 s | 0.375 s |
| Stage | 9.842 s | 9.799 s |

**Finding: there is no communication/compute overlap in either mode.** The
whole communication time is reported as not-overlapped in both arms
(`Overlapped = 0.000 s`), and async mode only shaves ~25 ms of free time and
~10 ms of communication — 0.4 % of stage, within noise. Host-side async
dispatch cannot create device-side CoC; that needs multi-stream execution or a
different collective schedule (Step 16's `COMBINED_ENABLE` was also neutral).
DeepSpeed's `overlap_comm: true` is set in `zero2.json` but produces no traced
overlap here.

---

## Step 29 — Batch-size stability sweep on the fixed-length set (`02a694f`)

200 steps per size (bs=62 reused from Step 27 at 100 steps), no profiling,
canonical otherwise (fa2, ZeRO-2, pinning, valid-token CE). FLOPs per card
scale linearly: `1.4954e15 × bs/62`. Peak 486 TFLOPS.

| bs | steps | mean s | median s | CV | outliers | samples/s (median) | samples/s (mean) | MFU (median) | MFU (mean) |
|---|---|---|---|---|---|---|---|---|---|
| 62 | 100 | 10.858 | 9.662 | 24.2% | 7 | **51.33** | 45.68 | **31.8%** | 28.3% |
| 48 | 200 | 8.394 | 7.570 | 19.2% | 4 | 50.73 | **45.75** | 31.5% | **28.4%** |
| 40 | 200 | 7.088 | 6.404 | 21.4% | 5 | 49.97 | 45.15 | 31.0% | 28.0% |
| 32 | 200 | 5.760 | 5.220 | 17.4% | 2 | 49.04 | 44.44 | 30.4% | 27.6% |
| 24 | 200 | 4.491 | 4.418 | **16.3%** | **1** | 43.46 | 42.75 | 27.0% | 26.5% |

(20–70 window; outliers = steps > 1.5× the window median.)

Findings:

- **Lowering the batch reduces variance but does not remove the spikes.** CV
  goes 24.2% → 16.3% from bs=62 to 24, and outlier count 7 → 1, but a few
  10–20 s steps appear at every size. The spikes are not batch-size artefacts.
- **Mean-based MFU is flat above bs=32** (27.6–28.4%, best at bs=48) and drops
  at bs=24 (26.5%). Median-based MFU is 30–32% for bs≥32.
- **Per-sample step cost rises as bs falls**: 0.156 s/sample at bs=62 → 0.184
  at bs=24 (fixed per-step cost amortized over fewer samples), which is exactly
  why throughput and MFU fall at the bottom of the sweep.
- **Throughput:** 51.3 samples/s at bs=62, 50.7 at 48, 50.0 at 40, 49.0 at 32,
  43.5 at 24 (median-based; mean-based ~42.8–45.8 throughout).
- If the goal is a *stable* number, bs=24 is the tightest (CV 16.3%); if it is
  *throughput/MFU*, bs=48–62 is better and the mean is the honest figure.

**Current best config: bs=48** — highest mean MFU (28.4%) with 50.7 samples/s,
and a lower CV than bs=62 (19.2% vs 24.2%). Adopted in the Data policy section.

---

## Step 30 — Executed-backward count + PyTorch cross-check (`5b2a053`)

**Protocol: every FLOPs measurement is now two runs** — the same command with
`--gradient_checkpointing False` and with `True`:

| | replay off (`flops_exec_20261025`) | replay on (`flops_replay_on_20261025`) |
|---|---|---|
| logged `flops_total` (read before backward) | 4.8237e13 | 4.8237e13 |
| hooks raw forward (whole run) | 1.6927e13 | 3.2444e13 |
| torch `FlopCounterMode` forward | 1.6645e13 | 1.6645e13 |
| hooks / torch | **1.017** | **1.949** |
| replayed forward (hooks delta) | 0 | **1.5517e13** |
| executed (logged + replayed) | 4.8237e13 | **6.3754e13** |
| ×31 → bs=62 | 1.4954e15 | **1.9764e15** |
| MFU @486, median 9.662 s | 31.8 % | **42.1 %** |
| MFU @486, mean 10.858 s | 28.3 % | **37.5 %** |

The logged values are identical by construction — the snapshot is taken before
`backward()`. The replay is visible only in the hook totals, because the hooks
are **forward** hooks and checkpoint replay re-runs the forward inside
`backward()`; the backward gradient math is never a forward call, so neither the
hooks nor `FlopCounterMode` see it (that is the 2× in the 3× rule). The
hooks/torch ratio is therefore the diagnostic: ~1.0 means no replay, ~1.9 means
replay was counted (≈0.92 of a forward — the VAE is `no_grad` and part of the
graph is not checkpointed).

Original run notes follow. Run `logs/flops_exec_20261025`: bs=2, 8 ranks,
**sdpa**, **gradient checkpointing off**, 1 step,
`ROSS_COUNT_FLOPS=1 ROSS_FLOPS_TORCH=1`.

| quantity | value |
|---|---|
| hooks raw forward | 1.6927e13 |
| `FlopCounterMode` forward | 1.6645e13 |
| hooks / torch | **1.017** |
| `flops_mm` (3× / 1×) | 4.7603e13 |
| `flops_attn` (causal) | 6.3435e11 |
| `flops_total` | 4.8237e13 |

Findings:

- **The two independent forward counts agree to 1.7 %** — the hook formulas are
  validated (hooks are slightly higher; likely addmm bias handling).
- **`FlopCounterMode` cannot see the NPU backward.** Verified: CPU fwd+bwd = 384,
  NPU forward = 128, NPU backward = **0** (torch_npu runs backward in `npu::`
  kernels outside aten dispatch). So `di-qwen-mot`'s executed-backward method is
  not reproducible on torch_npu 2.11; the executed backward stays the 3×/1×
  factor. With checkpointing off that factor is exact for GEMM (attention would
  be 3.5×, +0.2 % of the total).
- **The counting run disables gradient checkpointing** (`--gradient_checkpointing
  False`). No replay is executed, so `flops_total` is the executed-work number
  directly — there is nothing to correct for. Checkpointing is only re-enabled
  for the timed run (no batch ≥16 fits without it under ZeRO-2 — Step 31).
- The logged `flops_*` values are read before backward, so they are unaffected
  even if checkpointing is left on; only the `by_class` diagnostic accumulates
  replay. A warning is printed in that case.

---

## Step 31 — Replay on/off versus the throughput target (negative result)

Question: does disabling gradient checkpointing beat the bs=48 replay-on
baseline on samples/sec? All runs: 8×950DT, `navsim_flops_L420.pkl`, 50 steps,
fa2, valid CE, pinning (`ulimit -l unlimited`), `CPU_AFFINITY_CONF=1`, no
weights saved, code `6575814`; only the batch size, ZeRO stage and the flag
differ.

The shared `lmc-w0` shell still exported `ROSS_STEP_TIMING=0`, so these runs
report HF wall-clock (`train_runtime`), not the per-step median the target uses.
The on/off pairs are comparable to each other, not to the 50.73 target.

| run | bs | ZeRO | replay | result | runtime | s/step | samples/s (global) |
|---|---|---|---|---|---|---|---|
| `ab_replay_on_bs48_20261025` | 48 | 2 | on | 50/50 | 497.77 s | 9.955 | 38.57 |
| `ab_replay_off_bs48_20261025` | 48 | 2 | off | OOM, step 1 | — | — | — |
| `ab_replay_off_bs40_20261025` | 40 | 2 | off | OOM, step 1 | — | — | — |
| `ab_replay_off_bs32_20261025` | 32 | 2 | off | OOM, step 1 | — | — | — |
| `ab_replay_off_bs16_20261025` | 16 | 2 | off | OOM, step 1 | — | — | — |
| `ab_replay_off_bs8_20261025` | 8 | 2 | off | 50/50 | 154.35 s | 3.087 | 20.73 |
| `ab_replay_on_bs8_20261025` | 8 | 2 | on | 50/50 | 102.50 s | 2.050 | 31.22 |
| `abz3_replay_off_bs48_20261025` | 48 | 3 | off | killed at goal reset | — | — | — |

Findings:

- **Replay off does not fit above bs=8 under ZeRO-2.** Every failure is on the
  first forward, in the ViT rotary path, with 90.0–90.3 GiB of the 91.75 GiB
  HBM ceiling already allocated. The ceiling is between 8 and 16.
- **At the only size where it fits, replay off is slower**: 3.087 vs 2.050
  s/step at bs=8 (1.51×), and 20.73 vs 31.22 samples/s.
- **ZeRO-3 cannot rescue it.** Step 13 measured replay-on at 19.59 samples/s
  (12 workers) with ZeRO-3 at bs=48; a replay-off arm there would be no faster
  and stays far below the 50.73 target. The ZeRO-3 bs=48 off run was stopped
  when the goal was restated and produced no result.
- Direction closed: replay off is not a throughput candidate.

Why off is slower at equal batch is not measured (no memory trace was taken).
The likely cause is that the uncheckpointed backward holds the full activation
set and runs at ~90 GiB with ~1.4 GiB headroom, while at bs=8 there is too
little compute to amortize the extra activation traffic.

---

## Step 32 — Fused RMSNorm re-test on the 2.11 stack (adopted)

Step 14's +0.8% verdict for `npu_rms_norm` was measured on torch 2.7.1 at
bs=24. Re-tested on the current stack: torch 2.11, bs=48, replay on, fa2,
ZeRO-2, valid CE, pinning, `CPU_AFFINITY_CONF=1`, fixed-length set, 50 steps,
per-step `step_time_s` with the first 5 dropped. A-B-A to separate signal from
drift.

| arm | median s | mean s | CV | outliers | samples/s (median) | samples/s (mean) |
|---|---|---|---|---|---|---|
| A1 base | 7.614 | 8.221 | 14.2% | 2 | 50.43 | 46.71 |
| B `ROSS_NPU_RMSNORM=1` | **7.112** | 7.641 | 18.8% | 3 | **53.99** | 50.25 |
| A3 base | 7.587 | 8.159 | 13.9% | 2 | 50.61 | 47.06 |

Base-vs-base drift is 0.36%. Effect: median step time −6.4%, throughput
**+6.8%** (mean-based +7.2%). Losses match (3.6658 / 3.6690 / 3.6647), so it
is the same work. MFU on the logged-FLOPs basis
(`1.4954e15 × 48/62 = 1.1577e15` per card per step): **33.5% median / 31.2%
mean**.

The fused path is live: `Qwen2RMSNorm.forward` calls `fused_rms_norm`
([`qwen2/modeling_qwen2.py:198`](reference/transformers/src/transformers/models/qwen2/modeling_qwen2.py#L198)),
and Qwen2.5-VL imports that class. The run's `[fused] backward support: {}` is
expected — the probe covers only add-RMSNorm, GN+SiLU and SiLU, the ops with
no NPU backward kernel.

**Adopted:** the launcher exports `ROSS_NPU_RMSNORM=1` (an existing env value
wins, so A/Bs can set `0`). This supersedes Step 14's verdict for RMSNorm;
SwiGLU and RoPE keep their negative results.

Runs: `ab_rmsnorm_{base,on,base2}_20261026`.

---

## Step 33 — Re-testing the rejected fused kernels (RoPE adopted)

Re-ran the backward probe (`tools/ascend_parity/check_fused_ops.py`) and an
A-B-C-A throughput test on torch 2.11 at bs=48, with RMSNorm on as the
baseline. 50 steps per arm, fa2, ZeRO-2, valid CE, pinning, fixed-length set,
first 5 steps dropped.

Probe result — the three feasibility rejects are unchanged:

| op | forward | backward |
|---|---|---|
| `npu_add_rms_norm` | PASS | **unavailable** — `derivative for npu::npu_add_rms_norm is not implemented` |
| `npu_group_norm_silu` | PASS | **unavailable** — no autograd kernel (finite check fails) |
| `npu_silu` | unavailable | — (no aclnn implementation) |

Throughput A-B-C-A:

| arm | flags | median s | mean s | samples/s (median) | samples/s (mean) |
|---|---|---|---|---|---|
| A base | RMS | 7.098 | 7.764 | 54.10 | 49.46 |
| B | RMS + SWIGLU | 7.286 | 7.862 | 52.70 | 48.84 |
| C | RMS + ROPE | **6.676** | 7.114 | **57.52** | 53.98 |
| D base | RMS | 7.096 | 7.603 | 54.11 | 50.50 |

Base drift is 0.03% (7.098 vs 7.096), so both deltas are real.

- **SwiGLU: rejected again** (+2.4% step time, −2.6% throughput); the old
  −3.2% verdict holds.
- **RoPE: flipped to a win** (−5.9% step time, +6.3% median throughput,
  +8.0% mean-based). Step 14's −6.6% was on torch 2.7.1 at bs=24 — the same
  stack-transfer effect as RMSNorm (Step 32). Fusion is live in the vision
  tower ([`modeling_qwen2_5_vl.py:151-153`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L151-L153))
  and the LLM ([`:624-627`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L624-L627)).
- Losses agree across arms (3.6709 / 3.6782 / 3.6712 / 3.6712).

**Adopted:** the launcher exports `ROSS_NPU_ROPE=1` alongside
`ROSS_NPU_RMSNORM=1` (existing env values win). New best: **57.52 samples/s
median / 53.98 mean, MFU 35.7% median / 33.5% mean** — +13.4% over the 50.73
target.

Runs: `ab_fused_{base1,swiglu,rope,base2}_20261026`.

---

## Step 34 — L1 profile of the fused best config

Run `logs/profile_l1_best_20261026`: 32 steps, L1 profiler, real 20× subset,
bs=48, replay on, fused RMSNorm + RoPE, fa2, ZeRO-2, valid CE, pinning,
`CPU_AFFINITY_CONF=1`, code `9c79b54`. Traced step 28. Archive:
`logs/profile_traces_20261026/profile_l1_best_bs48_rmsrope_20261026.tar.gz`
(2.3 GB → 286 MB).

Step 28, rank 0: Computing 6.296 s, Communication (not overlapped) 4.317 s,
Free 1.581 s, Stage 12.194 s. All ranks agree on computing (6.27–6.30 s).

Kernel mix (rank 0, 6.688 s total op time):

| category | time | share |
|---|---|---|
| MatMul (AI_CORE + MIX_AIC) | 4.365 s | 65.3% |
| FlashAttention fwd+bwd | 0.504 s | 7.5% |
| Conv fwd/bwd | 0.450 s | 6.7% |
| Elementwise/layout | 0.516 s | 7.7% |
| HCCL launch kernels (389) | 0.392 s | 5.9% |

The fusions land where the pre-fusion profile predicted: elementwise/layout
17.3% → 7.7% versus Step 21, with Cast 243 → 70 ms, Mul 629 → 141 ms and Add
351 → 128 ms. MatMul's share rises to 65.3% only because the other buckets
shrank; absolute MatMul is lower at bs=48 than at bs=62.

Caveat: step 28 is a spike step (Stage 12.19 s vs ~7.1 s typical), and its
4.32 s communication is peer-wait, not HCCL work — the HCCL kernels are
0.392 s over the same 389 launches as Step 21. Use this trace for kernel
attribution, not for the communication split; a clean CoC number needs a
non-spike step or a multi-step active window.

---

## Step 35 — Collective stalls: expandable segments (adopted)

Step 34's trace showed a 12.194 s step whose 4.317 s of communication was pure
wait: 389 collectives, transit ≈ 0, and a few allReduces at ~0.5 s on every
rank. The previous pinned profile (Step 23) had 0.261 s over the same 389
collectives. The one config difference was
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` — Step 23 had it, Step 34
did not.

Re-profiled with it (`logs/profile_l1_best_es_20261026`, same config):

| trace | collectives | comm elapsed | max single | step 28 stage |
|---|---|---|---|---|
| without expandable segments | 389 | 4.317 s | 595.8 ms | 12.194 s |
| with expandable segments | 389 | **0.234 s** | 47.0 ms | **6.996 s** |

Computing is unchanged (6.296 s), so the stalls are allocator fragmentation at
specific backward allocations — not bandwidth and not a straggler.

Unprofiled 50-step A/B on the best config (`ab_es_on_20261026` against the
Step 33 RoPE arm):

| arm | median s | mean s | CV | outliers | samples/s (median) | samples/s (mean) |
|---|---|---|---|---|---|---|
| without | 6.676 | 7.114 | 20.7% | 3 | 57.52 | 53.98 |
| with | 6.677 | 7.085 | 15.8% | 2 | 57.51 | 54.20 |

Median throughput is unchanged; the mean gains 0.4% and variance drops.

**Adopted:** the launcher now sets the whole canonical NPU env — `CPU_AFFINITY_CONF=1`,
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`, `ulimit -l unlimited`,
`ASCEND_LAUNCH_BLOCKING=0`, and `ROSS_STEP_TIMING=1` — so ad-hoc launches
cannot silently miss them.

Trace: `logs/profile_traces_20261026/profile_l1_best_bs48_rmsrope_es_20261026.tar.gz`.

---

## Step 36 — v2 add-RMSNorm, selective CE, TASK_QUEUE_ENABLE=2 (all neutral)

Three experiments borrowed from the Emu3 port (`408235117/DriveVLA-W0-950DT`,
branch `emu3-npu-port`; see Step 33 for the fused-kernel re-test that did pay
off).

### 36a. `npu_add_rms_norm_v2_functional` with a custom backward

The v2 op exists on torch_npu 2.11 but registers **no autograd kernel**: the
residual and the norm weight receive no gradient (`x.grad` only, plus a
"silently incorrect" warning). `ROSS_NPU_ADD_RMSNORM_V2=1` wraps its forward in
a `torch.autograd.Function` that derives the backward from the kernel's
`summed`/`rstd` outputs. Parity (`check_fused_ops.py`): y rel 4.4e-3,
dx1/dx2 4.98e-3, dw 1.41e-3 — all inside bf16 tolerance. The path engaged with
no fallback warnings.

### 36b. Selective checkpoint CE

`ROSS_SELECTIVE_CE=1` wraps the valid-token cross-entropy in a selective
checkpoint (save matmul, recompute exp/sub), mirroring the Emu3 port. Their win
came from a 14.4 GB `exp()` transient in a chunked 184K-vocab CE; our compacted
CE only materialises ~0.55 GB.

### A-B-C-A (50 steps, bs=48, RMSNorm + RoPE + expandable segments)

| arm | median s | mean s | CV | outliers | samples/s (median) | samples/s (mean) | peak HBM |
|---|---|---|---|---|---|---|---|
| base1 | 6.658 | 7.274 | 21.8% | 3 | 57.67 | 52.79 | 63.5 GB |
| ADD_RMSNORM_V2 | 6.748 | 7.136 | 15.3% | 2 | 56.90 | 53.81 | 63.5 GB |
| SELECTIVE_CE | 6.658 | 7.146 | 18.3% | 3 | 57.68 | 53.74 | 63.5 GB |
| base2 | 6.649 | 7.097 | 16.9% | 2 | 57.75 | 54.11 | 63.5 GB |

Base drift 0.14% median / ~2.5% mean. **Add-RMSNorm v2 is −1.3% median** — the
fp32 math in the hand-written backward costs more than the fused residual add
saves. **Selective CE is neutral** and peak HBM is unchanged, so there is
nothing to reclaim. Both flags stay off by default.

### 36c. `TASK_QUEUE_ENABLE=2` (retried)

Step 16 found `=2` crashed before model loading. On the recreated container it
runs, but is neutral: 6.639 s median / 57.84 samples/s versus 6.649 / 57.75 for
the adjacent base, against 0.14% drift. The L1 trace shows why — step 28 has
`Overlapped = 0.000 s` on all 8 ranks (Computing 6.299 s, Comm 0.248 s, Free
0.412 s, Stage 6.960 s), the same zero-overlap result as `=0`/`=1` in Step 28.

Runs: `abx_{base1,v2add,selce,base2,tq2}_20261026`, `profile_l1_tq2_20261026`.
Trace: `logs/profile_traces_20261026/profile_l1_tq2_bs48_rmsrope_es_20261026.tar.gz`.

---

## Run index

Every run records the commit it is based on. "Code" is the commit whose tree
the run actually executed; where a directly copied file was used, both the
file's commit and the box `HEAD` are noted.

| Run directory | Step | Code | Steps |
|---|---|---|---|
| `train_qwen_vla_20261019_091442` | 7 | `67292e7` | 20 |
| `diag_ross_lr1e-4` | 8 | `67292e7` | 100 |
| `ross_confirm_300` | 9 | `67292e7` | 70 / 300 (interrupted) |
| `ross_short_20261020` | 10 | `e988fef` | 60 |
| `ross_100_20261020` | 10 | `e988fef` | 100 |
| `ross_100_gatelog_20261020` | 11 | `89edc4e` file on `c409ca4` HEAD (direct copy) | 100 (killed) |
| `ross_gatelog2_20261020` | 11 | `89edc4e` file on `c409ca4` HEAD (direct copy) | 100 |
| `ross_gate_diag_20261020` | 11 | `016a83f` | 3 |
| `ross_gate_fixed_20261020` | 11 | `c024db1` file on `016a83f` HEAD (direct copy) | 3 |
| `bs_sweep50_20261020/bs{2,4,8,16}` | 12 | `fbae548` file on `016a83f` HEAD (direct copy) | 50 |
| `bs_ceiling_20261020/bs{32,64,128}` | 12 | `fbae548` file on `016a83f` HEAD (direct copy) | 20 (128 OOM) |
| `bs_fine_20261020/bs{66,68,70}` | 12 | `fbae548` file on `016a83f` HEAD (direct copy) | 20 (70 OOM at step 5) |
| `bs_nooff3_20261021/bs{8,16,24,32,48}` | 13 | `fbae548` file on `23e62ee` HEAD (direct copy) | 20 |
| `bs_zero2_20261021/bs{8,16,24,32}` | 13 | `c6f95aa` file on `23e62ee` HEAD (direct copy) | 20 |
| `bs_nockpt_20261021/bs8` | 13 | `fbae548` file on `23e62ee` HEAD (direct copy) | OOM at step 2 |
| `bs_base50_20261021/bs68` | 13 | `fbae548` file on `23e62ee` HEAD (direct copy) | 50 |
| `bs_z3_50_20261021/bs48`, `bs_w24_50_20261021/bs48` | 13 | `fbae548` file on `23e62ee` HEAD (direct copy) | 50 |
| `bs_z2_50_20261021/bs{24,32}`, `bs_z2_w24_50_20261021/bs{24,32}` | 13 | `c6f95aa` file on `23e62ee` HEAD (direct copy) | 50 |
| `fused_ab_20261021/{base,rmsnorm,swiglu,rope,adamw,fa2}` | 14 | `9fed329` | 50 |
| `fused_ab2_20261021/{base2,fa2_32,fa2_40,fa2_48}` | 14 | `9fed329` | 50 (48 OOM at step 2) |
| `fa2_fine_20261021/fa2_42` | 14 | `9fed329` | OOM at step 42 |
| `fa2_stab_20261021/fa2_40_200` | 14 | `9fed329` | 200 |
| `profile_l1_20261021/navsim` | 15 | `c01713b` | 32 (L1 trace, step 28) |
| `env_ab_20261022/{base,comb,msr,base2}` | 16 | `c01713b` | 50 |
| `env_ab_20261022/{tq2,all3,all3_perf}` | 16 | `c01713b` | failed before training |
| `env_ab2_20261022/{tq1,detoff,nodet}` | 16 | `c01713b` | 50 |
| `c1a_20261022/{base20,c1a20,c1a40,c1a48,c1a56,c1a64,c1a72}` | 17 | `5f163a6` | 20 / 50 |
| `c1a_stab_20261022/c1a64_stab` | 17 | `5f163a6` | 200 |
| `c1a_stab_20261022/c1a72_stab` | 17 | `5f163a6` | failed (HCCL gather, rank 3) |
| `env_ab_c1a_20261023/nodet` | 16 | `b91ddd6` | 50 |
| `oldbox_torch211_50step_20261023/navsim` | 18 | `5a7c11a` | 50 (torch 2.11 re-run, 50.85 samples/s) |
| `oldbox_torch211_adamwfused_20261023/navsim` | 18 | `9573d02` | 50 (`adamw_torch_fused`, 50.60 samples/s) |
| `bs_stab_20261023/bs{64,62,60,58,56}` | 19 | `c49bfb1` | 50 each (stability sweep) |
| `bs_stab_20261023/bs62_rep` | 19 | `c49bfb1` | 50 (bs=62 repeat) |
| `navsim_real_smoke_20261023/navsim` | 20 | `237c2ac` | 10 (real data, 467 samples, bs=8) |
| `navsim_real_200step_x20_20261024/navsim` | 20 | `19781ee` | 200 (real 20× subset, bs=62, throughput) |
| `affinity_20261024/base` | 20 | `b875769` | 50 (`CPU_AFFINITY_CONF=0`) |
| `affinity_20261024/aff` | 20 | `b875769` | 50 (`CPU_AFFINITY_CONF=1`) |
| `profile_l1_real_20261024/navsim` | 21 | `67765d9` | 35 (L1 trace, step 28, no weights) |
| `profile_l1_real_bs40_20261024/navsim` | 22 | `f78fd40` | 35 (L1 trace, bs=40, no weights) |
| `profile_l1_c01713b_bs40_20261024/navsim` | 22 | `c01713b` | 35 (L1 trace, Step 15 code, bs=40; weights deleted) |
| `flops_count_20261024_v3` | 25 | `f404d21` | 20 (FLOPs count, bs=62, L=420, no weights) |
| `flops_b2_final_20261024` | 26 | `a350933` | 1 (FLOPs count, bs=2, L=420, no weights) |
| `flops_b2_buckets_20261024` | 26 | `bc64c73` | 1 (same, with the bucket diagnostic) |
| `bestcfg_flopsdata_20261024` | 27 | `7109a22` | 100 (best config, fixed-length set, no weights) |
| `coc_sync_20261024` | 28 | `7109a22` | 32 (L1 trace, `TASK_QUEUE_ENABLE=0`) |
| `coc_async_20261024` | 28 | `7109a22` | 32 (L1 trace, `TASK_QUEUE_ENABLE=1`) |
| `bsstab_{48,40,32,24}_20261024` | 29 | `02a694f` | 200 each (no profiling) |
| `flops_exec_20261025` | 30 | `5b2a053` | 1 (bs=2, no ckpt, sdpa, torch cross-check) |
| `flops_replay_on_20261025` | 30 | `5b2a053` | 1 (same, checkpointing on) |
| `ab_replay_on_bs48_20261025` | 31 | `6575814` | 50 (wall-clock metric) |
| `ab_replay_off_bs48_20261025` | 31 | `6575814` | OOM step 1 |
| `ab_replay_off_bs40_20261025` | 31 | `6575814` | OOM step 1 |
| `ab_replay_off_bs32_20261025` | 31 | `6575814` | OOM step 1 |
| `ab_replay_off_bs16_20261025` | 31 | `6575814` | OOM step 1 |
| `ab_replay_off_bs8_20261025` | 31 | `6575814` | 50 |
| `ab_replay_on_bs8_20261025` | 31 | `6575814` | 50 |
| `abz3_replay_off_bs48_20261025` | 31 | `6575814` | killed (goal reset) |
| `ab_rmsnorm_base_20261026` | 32 | `6575814` | 50 |
| `ab_rmsnorm_on_20261026` | 32 | `6575814` | 50 |
| `ab_rmsnorm_base2_20261026` | 32 | `6575814` | 50 |
| `ab_fused_base1_20261026` | 33 | `ff1169d` | 50 |
| `ab_fused_swiglu_20261026` | 33 | `ff1169d` | 50 |
| `ab_fused_rope_20261026` | 33 | `ff1169d` | 50 |
| `ab_fused_base2_20261026` | 33 | `ff1169d` | 50 |
| `profile_l1_best_20261026` | 34 | `9c79b54` | 32 (L1 trace, step 28, no weights) |
| `profile_l1_best_es_20261026` | 35 | `9c79b54` | 32 (L1 trace, same + expandable segments) |
| `ab_es_on_20261026` | 35 | `9c79b54` | 50 |
| `abx_base1_20261026` | 36 | `7c769e7` | 50 |
| `abx_v2add_20261026` | 36 | `7c769e7` | 50 |
| `abx_selce_20261026` | 36 | `7c769e7` | 50 |
| `abx_base2_20261026` | 36 | `7c769e7` | 50 |
| `abx_tq2_20261026` | 36 | `7c769e7` | 50 |
| `profile_l1_tq2_20261026` | 36 | `7c769e7` | 32 (L1 trace, step 28, `TASK_QUEUE_ENABLE=2`) |

`train_qwen_vla_20261019_085720` and `_090116` are earlier failed attempts
with no `trainer_state.json`.

---

## Current status

### Verified locally

- Launcher parses (`bash -n`), `--help` works, missing-argument validation works.
- Synthetic data: 256 scenes, correct pickle schema, image path resolution,
  504×280 → 180 tokens.
- `pre_1s_*` is a 2-frame-shifted view of the window, not a copy (Step 10);
  0 / 4096 and 0 / 8192 duplicate pairs after regeneration on the NPU host.
- The vendored Qwen model/config/processor files are identical to upstream
  `transformers v4.56.0`; `trainer.py` differs by one import style and the
  Qwen `__init__.py` adds the ROSS module reference.
- `diffusers 0.41.0` still exports every symbol the custom UNet imports
  (`FusedAttnProcessor2_0`, `USE_PEFT_BACKEND`, `FromOriginalModelMixin`,
  `get_mid_block`, `retrieve_timesteps`, `rescale_noise_cfg`).
- The five NPU runtime fixes from Step 5 pass `py_compile`.
- The added `transformers/data` files pass `py_compile`.

### Verified on the remote NPU host

- At the time of Step 7: `torch==2.7.1`, `torchvision==0.22.1`,
  `torchaudio==2.7.1`, `torch_npu==2.7.1.post4`; `torch.npu.is_available()` →
  True, 8 devices. The box now reports `torch 2.11.0+cpu` / `torch_npu 2.11.0`
  (verified 2026-10-25).
- `transformers.data` + `transformers.trainer` import successfully after
  Step 6 (no reinstall needed).
- **20-step training on 8 NPU completed successfully** (Step 7): all five
  fixes exercised, `ross_loss` stable around 1.0, model saved.
- **60-step training on 8 NPU with corrected synthetic data** completed
  (Step 10): 8.72 s/step, ROSS loss flat at ~1.0, model saved.
- `diffusers 0.41.0` + the custom UNet, DeepSpeed ZeRO-3 CPU offload +
  `torch_npu`, and `torch.amp.autocast('npu')` all worked at runtime.
- The synthetic 504×280 images produced the expected 180-token grid.

---

## Remaining port work

Core NPU training works (Step 7). What remains:

| # | File:line | Problem | Planned fix |
|---|---|---|---|
| 1 | [`inference/qwen/inference_action_navsim_vava.py:100-104`](inference/qwen/inference_action_navsim_vava.py#L100-L104) | inference hardcodes `backend='nccl'` and `torch.cuda` | use `hccl` / `torch.npu` when NPU; only needed for evaluation |
| 2 | `use_previous_actions=true` | ROSS asserts 2 action segments | currently disabled; if history is required, generalize the assert/extraction to handle 4 segments |
| 3 | launcher | `--gradient_checkpointing True` is hardcoded; no off switch | **Save half resolved** by `67765d9`: `--save_model false` (default) sets `--save_strategy no` and skips the final model. The checkpointing switch is optional — replay off does not beat the target (Step 31) |
| 4 | training recipe | 20 steps / warmup 50 is not a valid learning test | use lr 5e-5–1e-4, warmup 10–50, several hundred steps (Steps 8, 10); on corrected data the ROSS loss is still at baseline after 60 steps (Step 10) |

Resolved by Step 7 (no longer open): `diffusers 0.41.0` UNet compatibility,
DeepSpeed ZeRO-3 + `torch_npu`, device-aware autocast, and the 180-token grid.

---

## Appendix — core port files (Steps 1–6, 24; later steps list their own files)

| File | Status | Step |
|---|---|---|
| [`requirements_qwen.txt`](requirements_qwen.txt) | new | 1, 3, 4 |
| [`scripts/scripts_train/train_qwen_vla_navsim.sh`](scripts/scripts_train/train_qwen_vla_navsim.sh) | new | 1, 2 |
| [`tools/synth/synth_qwen_vla_data.py`](tools/synth/synth_qwen_vla_data.py) | new | 2, 10 |
| [`reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/data_qwen_vla.py) | modified | 4 |
| [`reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/rope2d.py`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/dataset/rope2d.py) | modified | 4 |
| [`reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py`](reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/trainer.py) | modified | 5 |
| [`reference/transformers/.../modeling_ross/denoiser_sd.py`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_ross/denoiser_sd.py) | modified | 5 |
| [`reference/transformers/.../modeling_qwen2_5_vl_ross.py`](reference/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl_ross.py) | modified | 5 |
| [`utils/train_qwen_vla.py`](utils/train_qwen_vla.py) | modified | 5 |
| [`reference/transformers/src/transformers/data/__init__.py`](reference/transformers/src/transformers/data/__init__.py) | new | 6 |
| [`reference/transformers/src/transformers/data/data_collator.py`](reference/transformers/src/transformers/data/data_collator.py) | new (upstream v4.56.0) | 6 |
| [`.gitignore`](.gitignore) | modified | 6 |
| [`docs/superpowers/specs/2026-10-24-flops-counting-design.md`](docs/superpowers/specs/2026-10-24-flops-counting-design.md) | new (FLOPs/MFU design) | 24 |
| [`npu-port.md`](npu-port.md) | new | this document |

Generated but not committed: `logs/synth_qwen_data/` (gitignored) and
`/mnt/sfs_turbo/lmc/navsim_openscene/navsim_flops_L420.pkl` (Step 24; regenerate
from the recipe there).

---

## Appendix — ott9-a38: 16 × Ascend910 bring-up (2026-10-27)

A second box, `ott9-a38` (`liteserver-ott9-a3-00008`): 8 cards × 2 chips =
**16 Ascend910**, 64 GB HBM per chip, 2 TB host RAM, container `drivevla-w0`
(CANN 9.1.0 image), host network namespace. Same repo path and `lmc-w0` tmux
session name as the 950DT box.

### Setup

- **Repo:** git smart-HTTP is filtered from this box (`info/refs` times out);
  codeload tarballs work. Final state is a shallow **SSH** clone
  (`git@github.com:mingchang93/DriveVLA-W0.git`, `navsim-real-data`) at
  `6066202`, so pulls work normally. The tarball snapshot is kept as
  `DriveVLA-W0-snapshot-6066202`.
- **Env:** `pip install -r requirements_qwen.txt` with
  `-i https://mirrors.aliyun.com/pypi/simple` (the configured Huawei mirror
  returns 429). Verified `torch 2.11.0+cpu`, `torch_npu 2.11.0`,
  `transformers 4.56.0`, `torch.npu.device_count() == 16`.
- **Weights:** Qwen2.5-VL-7B and SD1.5 via `modelscope download`.
- **Data:** official OpenScene metadata, **not** liyingyan's pickle —
  `openscene_metadata_trainval.tgz` (6.6 GB, 8-way parallel range download from
  hf-mirror) for scene logs, plus camera shard 0 via
  `tools/pickle_gen/download_openscene_camera.sh --start 0 --end 0`. Seven
  shard-0 scenes intersect the metadata; `pickle_generation_navsim_mini.py`
  produced 2,795 samples, all kept by `filter_pickle_by_images.py`, ×20 =
  55,900 for throughput.

### Two generator fixes (both pushed)

| commit | bug |
|---|---|
| `4e44e41` | `NormStats` defined inside `main()` → `pickle.dump` raised "Can't pickle local object" before the main pickle was written |
| `57f194b` | stored image paths kept the metadata's `<log>/CAM_F0/<frame>.jpg`; the dataloader and filter insert `CAM_F0` themselves, so paths doubled and filtered to 0 samples |

### Environment gotchas (each cost a failed run)

1. **HCCL interface.** Auto-selection picked the wrong NIC; rank 11 timed out
   in `HcclAllreduce`. Fix: `HCCL_SOCKET_IFNAME=enp23s0f3` (the host's
   default-route NIC, 172.168.4.212/24) plus `HCCL_CONNECT_TIMEOUT=120`.
2. **Hugepages starved host RAM.** `HugePages_Total: 480000 × 2 MB = 960 GB`
   was reserved (free but unavailable to normal allocations), leaving ~1 TB for
   the 16 ranks' checkpoint-load transients. The kernel OOM-killed ranks, which
   surfaced as rank SIGSEGV/SIGKILL rather than a clean allocator error. Fix:
   `sysctl -w vm.nr_hugepages=0`, after which 1.99 TB is available.
3. **Device OOM also presents as SIGSEGV** on this CANN/Soc: bs=48 dies in the
   first step on the 64 GB chips with no allocator message. Use the `[mem]`
   peak print plus a sweep, not the exception type.

### Throughput — 16 ranks, 50 steps, real subset

fa2, ZeRO-2, valid CE, fused RMSNorm + RoPE, `expandable_segments`, pinning.

| bs | median s | mean s | CV | outliers | samples/s (median) | samples/s (mean) | peak HBM |
|---|---|---|---|---|---|---|---|
| **32** | **8.507** | 8.586 | 3.8% | 0 | **60.18** | 59.64 | 45.3 GB |
| 40 | 10.668 | 10.717 | 3.6% | 0 | 59.99 | 59.72 | 50.1 GB |
| 48 | — | — | — | — | SIGSEGV in step 1 | — | — |

bs=32 is the best median and leaves ~19 GB headroom; bs=40 is within noise.
Step-time CV is ~3.7% with **zero outliers**, versus 14–25% on the 8×950DT box.
60.18 samples/s beats the 950DT best (57.52 at bs=48) with twice the chips and a
smaller per-device batch.

Runs: `a38_bs32_v2_20261027`, `a38_bs40_20261027`, `a38_bs48_v3_20261027`
(failed), `a38_diag_20261027` (2-step bs=8, blocking mode).

# Precision Alignment Guide — NPU vs GPU Deterministic Reproducibility

How to make an NPU training run produce the **same loss curve, same gradients, and same
checkpoint** as the equivalent GPU run — step by step.

Based on the [Ascend msprobe Training Precision Debug Guide](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md).

---

## Phase 1 — Pre-Alignment Environment Verification

Before any code change, verify the two environments are comparable.

### 1.1 Hyperparameters & Launch Arguments

Diff the launch command or config file between NPU and GPU. Every argument except
the device selector must be identical:

| Check | Pass if |
|-------|---------|
| `--seed` | Same value on both platforms |
| `--batch_size` | Same per-device batch size on both platforms |
| `--learning_rate` | Same value on both platforms |
| `--max_steps` / `--num_epochs` | Same step count or epoch count |
| `--precision` / `--fp` | Use `fp32` for initial alignment; `bf16`/`fp16` only after fp32 passes |
| `--gradient_accumulation_steps` | Same value on both platforms |
| `--warmup_steps` / `--lr_scheduler` | Same scheduling on both platforms |
| `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` | Same optimizer config |

### 1.2 Library Versions

```bash
# Run on both NPU and GPU machines — diff the output
pip list 2>/dev/null | grep -iE "torch|deepspeed|npu|cann|flash|transformer"
```

| Check | NPU | GPU |
|-------|-----|-----|
| `torch` | Latest stable with `torch_npu` support | Latest stable with CUDA support |
| `torch_npu` | Installed and version matches `torch` | N/A |
| `deepspeed` | Same git branch/commit | Same git branch/commit |
| `transformers` | Same version | Same version |
| `flash-attn` | Not installed (NPU uses SDPA or eager) | Installed (CUDA-only) |

> **Version mismatch is the #1 cause of precision divergence.** Align these before moving to Phase 2.

### 1.3 Upgrade to Latest (NPU Side)

```bash
# Update CANN, driver, and torch_npu to latest compatible versions
pip install --upgrade torch_npu
# Check CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null
```

### 1.4 Model Structure Identity

```python
# Run on both NPU and GPU — the output must be identical
print(model)
# or for HuggingFace-style models:
print(model.config)
```

### 1.5 Weight Initialization Identity

| Check | How |
|-------|-----|
| Same pretrained checkpoint | Load from the same directory / model hub ID on both platforms |
| Same random init | Use identical `--seed` plus all Phase 2 seed fixes |

---

## Phase 2 — Fix Randomness

Every source of non-determinism must be pinned. Implement these in your training
entrypoint before model creation or data loading.

### 2.1 Essential Random Seeds

```python
import os
import random
import numpy as np
import torch

def set_all_seeds(seed: int):
    """Set every seed that affects reproducibility. Call BEFORE model init."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Platform-specific device seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)
    except ImportError:
        pass
```

### 2.2 Disable Dropout

```python
def disable_dropout(model: torch.nn.Module):
    """Set p=0 on every dropout layer. Call AFTER model init."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.p = 0.0
```

### 2.3 Disable Data Shuffling

```python
from torch.utils.data import DataLoader, SequentialSampler

# Use SequentialSampler instead of default RandomSampler
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=SequentialSampler(train_dataset),   # deterministic order
    num_workers=num_workers,
    pin_memory=True,
)
```

> **Critical:** even with all seeds fixed, `RandomSampler` produces different
> sequences on NPU vs GPU because the sampler's internal RNG state diverges across
> platforms. Always use `SequentialSampler` for alignment runs.

### 2.4 Check Dataset `__getitem__` for RNG Calls

Audit `Dataset.__getitem__` for any `random.randint()`, `random.random()`, or
`np.random.*` calls. These are safe *if* the seeds in Section 2.1 are called before
the dataset is created. If the dataset uses its own unseeded RNG instance, seed it
explicitly.

---

## Phase 3 — Enable Deterministic Computation

### 3.1 Operator Determinism

```python
def enable_deterministic_computation():
    """Enable deterministic algorithms on both platforms. Call BEFORE model init."""
    # warn_only=True: NPU does not support deterministic mode for all operators
    # (e.g. grid_sample, some attention variants). Without this flag,
    # unsupported operators raise RuntimeError.
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

> `warn_only=True` is intentional. It logs warnings for operators that cannot be
> made deterministic on NPU. Those operators are your first suspects if loss
> diverges.

### 3.2 Communication Determinism

```python
# NPU collective communication (HCCL)
os.environ.setdefault("HCCL_DETERMINISTIC", "TRUE")

# GPU collective communication (NCCL)
if torch.cuda.is_available():
    os.environ.setdefault("NCCL_DETERMINISTIC", "TRUE")
    os.environ.setdefault("NCCL_CROSS_NIC", "1")
```

> **For single-device alignment, communication determinism is irrelevant.** Start
> with 1 device per platform before scaling to multi-device.

### 3.3 Non-Saturation Mode (NPU)

```python
# NPU by default may saturate overflow (clamp to max/min) instead of producing
# Inf/NaN. This makes overflow invisible in loss curves.
os.environ.setdefault("INF_NAN_MODE_ENABLE", "1")
```

This guarantees NPU matches GPU behavior: overflow → `Inf`/`NaN`.

---

## Phase 4 — Align the Data Pipeline

### 4.1 Verify Input Tensors Are Identical

The fastest way to confirm data determinism: dump the first batch on both platforms.

```python
# Add temporarily after DataLoader creation
batch = next(iter(train_dataloader))
torch.save(batch, "first_batch.pt")
```

Then diff on a machine with both files:
```python
import torch
npu_batch = torch.load("first_batch_npu.pt")
gpu_batch = torch.load("first_batch_gpu.pt")
for k in npu_batch:
    if not torch.equal(npu_batch[k], gpu_batch[k]):
        print(f"MISMATCH: {k}")
    else:
        print(f"OK: {k}")
```

### 4.2 Check for Platform-Dependent Preprocessing

| Pattern | Risk | Fix |
|---------|------|-----|
| `os.listdir()` / `os.walk()` without `sorted()` | File ordering differs by filesystem | Wrap in `sorted()` |
| `set()` / `dict` iteration | Hash ordering differs without `PYTHONHASHSEED` | Section 2.1 covers this |
| `random.*` / `np.random.*` in `__getitem__` | Safe if seeds set before dataset creation | Verify Section 2.1 runs first |
| Data augmentation (`torchvision.transforms`) with random params | Each platform's RNG may diverge | Disable augmentation for alignment |

---

## Phase 5 — First-Step Loss Comparison

Run with `--max_steps 1` (or equivalent) and compare the first loss value.

### 5.1 Evaluate

| Observation | Meaning | Action |
|-------------|---------|--------|
| Step 0 loss **identical** (within 0.01%) | Forward pass aligned | Proceed to multi-step (Phase 6) |
| Step 0 loss **differs > 0.01%** | Forward pass diverges | See Section 5.2 |
| Step 0 loss **NaN / Inf on NPU only** | Overflow on NPU | Check `INF_NAN_MODE_ENABLE=1`; see Phase 7 |

### 5.2 Forward Pass Diverges — Operator Bisect

1. Dump forward tensors for step 0 on both platforms (use Ascend msprobe Precision
   Collection Tool or manual hook-based dumping).
2. Find the **first** operator where output differs despite identical input.
3. Common NPU non-deterministic operators:
   - Fused attention (`npu_fusion_attention`) → fall back to SDPA or eager
   - `grid_sample` → move to CPU or raise to FP32
   - `torch.randn` on-device → generate on CPU then `.to(device)`
   - Fused layernorm → replace with small-op layernorm

### 5.3 First-Step OK, Later Steps Diverge

This means the **backward pass** (gradient computation) differs.

1. Dump step N-1 backward tensors + step N forward tensors.
2. Bisect the backward graph the same way as the forward pass.
3. Check optimizer state — Adam moments can accumulate differently across platforms.
   - Temporarily switch to SGD (`momentum=0`) to isolate the optimizer.
   - Verify `beta1`, `beta2`, `epsilon` are identical on both platforms.

---

## Phase 6 — Multi-Step and Long-Stable Comparison

### 6.1 Multi-Step (10–100 steps)

Run enough steps to see trend. Compare:

| Metric | What to look for |
|--------|------------------|
| Loss curve | Same shape, same values per step |
| Gradient norm | Same magnitude and trend |
| Learning rate | Same schedule and per-step values |

### 6.2 Long-Stable Training (1000+ steps)

If loss curves track for a few hundred steps then diverge:

| Pattern | Likely cause | Action |
|---------|-------------|--------|
| **GradNorm spikes → Loss diverges** | Gradient accumulation error | Dump gradient tensors; bisect backward |
| **Loss drifts gradually** | Cumulative matmul/conv precision drift | Try `CLOSE_MATMUL_K_SHIFT=1` (NPU) |
| **Sudden NaN after many steps** | Overflow accumulation | See Phase 7 |

---

## Phase 7 — NaN / Overflow Diagnosis

### 7.1 Universal Mitigations

| Action | How |
|--------|-----|
| Enable non-saturation mode (NPU) | `export INF_NAN_MODE_ENABLE=1` |
| Enable stream sync for debug (NPU) | `export ASCEND_LAUNCH_BLOCKING=1` |
| Disable fused attention | Switch to `eager` or `sdpa` attention backend |
| Raise precision | Switch from `fp16`/`bf16` to `fp32` |
| Simplify DeepSpeed | Remove `overlap_comm`, `overlap_optimizer`; try Zero-2 instead of Zero-3 |
| Disable optimizer temporarily | Set LR=0 to isolate optimizer from forward/backward |

### 7.2 Find the First NaN

1. Make the NaN reproducible: fix all seeds (Phase 2), disable shuffle (Section 2.3).
2. Bisect by step to find the exact step where NaN first appears.
3. Register forward hooks to locate the exact operator:

```python
def detect_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
        print(f"NaN in {module.__class__.__name__} at step {global_step}")
```

### 7.3 Root Cause by NaN Location

| NaN first appears in | Suspect |
|----------------------|---------|
| **Weight tensor** | Previous step's gradient corrupted the optimizer → NaN weight |
| **Input to operator** | Upstream operator not tracked by hooks → trace code |
| **Output of operator** | This operator itself → single-operator precision bug |

---

## Phase 8 — Framework-Level Isolation

### 8.1 DeepSpeed

| Check | Why |
|-------|-----|
| `bucket_size` identical on both platforms | Different bucket sizes → different all-reduce patterns → different grad precision |
| Zero stage identical | Zero-2 and Zero-3 split gradients differently |
| CPU offload disabled | Offload adds fp32 ↔ fp16 conversions that differ between platforms |
| Overlap features disabled | `overlap_comm` and `overlap_optimizer` reorder operations non-deterministically |

### 8.2 Optimizer Isolation

Set LR=0 so forward and backward run but weights never change. If the loss matches
with LR=0 but diverges with normal LR, the optimizer is the source.

### 8.3 Scale Reduction

Reduce complexity to the minimum that still reproduces the issue:

- 1 device per platform (no distributed communication)
- Small batch size (2–4)
- Few steps (10–100)
- If MoE: reduce expert count or disable temporarily
- If large model: reduce layer count via config

---

## Phase 9 — Single Operator Verification

When a specific operator is suspected:

1. **Extract the exact input tensors** from the NPU run at that operator.
2. **Run the operator in isolation** on NPU, GPU, and CPU with the same input.
3. **Compare Euclidean distance to CPU as reference**:

```
NPU_output vs CPU_ref:  euclidean_distance = 1.3e-6
GPU_output vs CPU_ref:  euclidean_distance = 8.2e-7   (baseline)
```

4. If NPU-to-CPU distance significantly exceeds GPU-to-CPU distance → operator has
   a platform-specific precision issue.

### Repair Options (Try in Order)

| # | Method | How |
|---|--------|-----|
| 1 | Raise precision | FP16/BF16 → FP32 for that operator only |
| 2 | Move to CPU | Transfer inputs to CPU, run operator, transfer output back to device |
| 3 | Replace fused operator | Fused attention → eager; fused layernorm → small-op layernorm |
| 4 | Contact Ascend support | Submit single-operator reproduction script with input tensors |

---

## Quick Checklist

```
[ ] Phase 1: Environment
    [ ] All hyperparameters identical (diff launch scripts/configs)
    [ ] Library versions aligned (pip list diff)
    [ ] NPU CANN / driver / torch_npu up to date
    [ ] Model structure identical (print(model) diff)
    [ ] Weight init identical (same checkpoint or same seed)

[ ] Phase 2: Fix Randomness
    [ ] PYTHONHASHSEED set
    [ ] random.seed(seed) called
    [ ] np.random.seed(seed) called
    [ ] torch.manual_seed(seed) called
    [ ] torch.cuda.manual_seed_all / torch.npu.manual_seed_all called
    [ ] cudnn deterministic + benchmark disabled (GPU)
    [ ] TF32 disabled (GPU)
    [ ] CUBLAS_WORKSPACE_CONFIG set (GPU)
    [ ] Dropout disabled (all layers p=0)
    [ ] Data shuffling disabled (SequentialSampler)

[ ] Phase 3: Deterministic Computation
    [ ] torch.use_deterministic_algorithms(True, warn_only=True)
    [ ] HCCL_DETERMINISTIC=TRUE (NPU)
    [ ] NCCL_DETERMINISTIC=TRUE (GPU)
    [ ] NCCL_CROSS_NIC=1 (GPU)
    [ ] INF_NAN_MODE_ENABLE=1 (NPU)

[ ] Phase 4: Data Pipeline
    [ ] First batch identical on both platforms (dump & diff)
    [ ] No platform-dependent file ordering (sorted() on os.listdir/glob)
    [ ] No unseeded RNG in Dataset.__getitem__
    [ ] Data augmentation disabled for alignment run

[ ] Phase 5: First-Step Loss
    [ ] Step 0 loss matches (< 0.01% difference)
    [ ] If not: operator bisect to find first divergence
    [ ] If step 1+ diverges: backward pass bisect + optimizer isolation

[ ] Phase 6: Multi-Step
    [ ] Loss curve matches for 100 steps
    [ ] Gradient norm curve matches
    [ ] No sudden divergence after N steps

[ ] Phase 7: NaN / Overflow (if applicable)
    [ ] INF_NAN_MODE_ENABLE=1 active
    [ ] First NaN step identified
    [ ] NaN source (weight / input / output) identified
    [ ] Fused attention disabled (eager backend)
    [ ] Optimizer isolated (LR=0 test)

[ ] Phase 8: Framework Isolation
    [ ] DeepSpeed overlap features disabled
    [ ] DeepSpeed bucket_size + zero stage identical
    [ ] Optimizer isolated (LR=0 test)
    [ ] Scale reduced (1 device, small batch, few steps)

[ ] Phase 9: Single Operator
    [ ] Suspect operator identified
    [ ] Isolated test: NPU vs CPU Euclidean distance ≤ GPU vs CPU
```

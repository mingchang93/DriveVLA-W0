# Setup: GPU / NPU

## 1. System Prerequisites

<details open>
<summary><b>GPU</b></summary>

```bash
# CUDA 12.4+
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.5.7
```
</details>

<details>
<summary><b>NPU</b></summary>

```bash
# CANN 8.5.1 + torch_npu
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install torch_npu==2.7.1.post4
```
</details>

## 2. Shared Dependencies

```bash
pip install -r requirements.txt
pip install deepspeed scipy tensorboard wandb
```

`flash-attn` in `requirements.txt` is CUDA-only — NPU falls back to SDPA automatically.

## 3. Data Preparation

```bash
apt-get update && apt-get install -y unzip

# Set these to match your layout
DATA_ROOT=/path/to/datasets
REPO_ROOT=/path/to/DriveVLA-W0

# Unpack VQ code zips
cd "$DATA_ROOT"
unzip train_vp_codes.zip
unzip test_vq_codes.zip

cd "$REPO_ROOT"

# Fix pickle paths for the local machine
python tools/fix_pickle_paths.py \
    "$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

python tools/fix_pickle_paths.py \
    "$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s.pkl" \
    --old_prefix /mnt/vdb1/yingyan.li/repo/VLA/data/navsim/processed_data \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"
```

Set `DATA_ROOT` and `REPO_ROOT` to match your machine. The fix is only needed once per download — it rewrites the data-root prefix stored inside each pickle.

## 4. Launch Training

```bash
# GPU (default)
bash scripts/scripts_train/train_base_ar_withou_moe.sh

# NPU
bash scripts/scripts_train/train_base_ar_withou_moe.sh --device npu

# GPU with explicit device flag
bash scripts/scripts_train/train_base_ar_withou_moe.sh --device cuda
```

### Example: NPU debug run

```bash
MODEL_ROOT=/data/models

bash scripts/scripts_train/train_base_ar_withou_moe.sh \
    --model_name_or_path "$MODEL_ROOT/Emu3-Stage1" \
    --data_path "$MODEL_ROOT/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl" \
    --test_data_path "$MODEL_ROOT/DriveVLA-W0/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl" \
    --ngpus 8 \
    --batch_size 1 \
    --max_steps 200 \
    --save_steps 100 \
    --fp fp16 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --deterministic \
    --device npu
```

See `scripts/scripts_train/train_base_ar_withou_moe.sh --help` for all options.

#!/usr/bin/env bash
#
# fp32 experiments on CUDA: 100/200 steps
#
# Usage:
#   bash scripts/run_experiments_cuda_fp32.sh
#
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_SCRIPT="$ROOT/scripts/scripts_train/train_base_ar_withou_moe.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$ROOT/logs/train_base_ar_${TIMESTAMP}"

MODEL_PATH="/data/models/Emu3-Stage1"
DATA_PATH="/data/models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl"
TEST_DATA_PATH="/data/models/DriveVLA-W0/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl"

COMMON_ARGS="
    --model_name_or_path $MODEL_PATH
    --data_path $DATA_PATH
    --test_data_path $TEST_DATA_PATH
    --output_dir $OUTPUT_DIR
    --ngpus 8
    --batch_size 1
    --warmup_steps 0
    --logging_steps 1
    --device cuda
    --log_data_hash
    --deterministic
    --shuffle_train_data false
    --eval_strategy no
    --eval_steps 10000
    --skip_inference
"

echo "============================================"
echo "CUDA fp32 experiments — output: $OUTPUT_DIR"
echo "============================================"

# Experiment 1: fp32, 100 steps
echo ""
echo "=== [1/2] fp32, 100 steps ==="
bash "$TRAIN_SCRIPT" \
    $COMMON_ARGS \
    --fp fp32 \
    --max_steps 100 \
    --save_steps 50 \
    --exp_name fp32_100steps

echo "Waiting 10 minutes for GPU cleanup..."
sleep 600

# Experiment 2: fp32, 200 steps
echo ""
echo "=== [2/2] fp32, 200 steps ==="
bash "$TRAIN_SCRIPT" \
    $COMMON_ARGS \
    --fp fp32 \
    --max_steps 200 \
    --save_steps 100 \
    --exp_name fp32_200steps

echo ""
echo "============================================"
echo "All CUDA fp32 experiments done. Results: $OUTPUT_DIR"
echo "============================================"
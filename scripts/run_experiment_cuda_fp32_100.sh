#!/usr/bin/env bash
#
# CUDA fp32 100-step experiment
#
# Usage:
#   bash scripts/run_experiment_cuda_fp32_100.sh
#
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_SCRIPT="$ROOT/scripts/scripts_train/train_base_ar_withou_moe.sh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$ROOT/logs/train_base_ar_${TIMESTAMP}"

MODEL_PATH="/data/models/Emu3-Stage1"
DATA_PATH="/data/models/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl"
TEST_DATA_PATH="/data/models/DriveVLA-W0/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl"

echo "============================================"
echo "CUDA fp32 500 steps — output: $OUTPUT_DIR"
echo "============================================"

bash "$TRAIN_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --ngpus 8 \
    --batch_size 1 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --device cuda \
    --log_data_hash \
    --deterministic \
    --shuffle_train_data false \
    --eval_strategy no \
    --eval_steps 10000 \
    --skip_inference \
    --constant_lr \
    --fp fp32 \
    --max_steps 500 \
    --save_steps 250 \
    --exp_name fp32_500steps

echo ""
echo "CUDA fp32 500 steps done. Results: $OUTPUT_DIR"
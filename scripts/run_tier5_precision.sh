#!/usr/bin/env bash
#
# Tier 5: fp32 vs bf16 comparison
#
# Usage:
#   bash scripts/run_tier5_precision.sh --device cuda
#   bash scripts/run_tier5_precision.sh --device npu \
#       --project_root /path/to/project \
#       --data_root /path/to/data \
#       --model_root /path/to/models
#
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_SCRIPT="$ROOT/scripts/scripts_train/train_base_ar_withou_moe.sh"

# ── Parse ──────────────────────────────────────────────────────────
DEVICE=""
PROJECT_ROOT=""
DATA_ROOT=""
MODEL_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)       DEVICE="$2";       shift 2 ;;
    --project_root) PROJECT_ROOT="$2"; shift 2 ;;
    --data_root)    DATA_ROOT="$2";    shift 2 ;;
    --model_root)   MODEL_ROOT="$2";   shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [ "$DEVICE" != "cuda" ] && [ "$DEVICE" != "npu" ]; then
  echo "Usage: $0 --device cuda|npu [--project_root <path>] [--data_root <path>] [--model_root <path>]"
  exit 1
fi

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/data/models/DriveVLA-W0}"
DATA_ROOT="${DATA_ROOT:-/data/models/DriveVLA-W0}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"

MODEL_PATH="$MODEL_ROOT/Emu3-Stage1"
TRAIN_PKL="$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
TEST_PKL="$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s.pkl"
TRAIN_PKL_FIXED="${TRAIN_PKL%.pkl}_fixed.pkl"
TEST_PKL_FIXED="${TEST_PKL%.pkl}_fixed.pkl"

# ── Fix pickle paths ───────────────────────────────────────────────
echo "=== Fixing pickle paths ==="
python "$ROOT/tools/fix_pickle_paths.py" \
    "$TRAIN_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

python "$ROOT/tools/fix_pickle_paths.py" \
    "$TEST_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

[ -f "$TRAIN_PKL_FIXED" ] && mv "$TRAIN_PKL_FIXED" "$TRAIN_PKL"
[ -f "$TEST_PKL_FIXED" ] && mv "$TEST_PKL_FIXED" "$TEST_PKL"
echo "=== Pickle paths fixed ==="

# ── Common flags ───────────────────────────────────────────────────
COMMON=(
  --model_name_or_path "$MODEL_PATH"
  --data_path "$TRAIN_PKL"
  --test_data_path "$TEST_PKL"
  --ngpus 8
  --logging_steps 1
  --device "$DEVICE"
  --log_data_hash
  --deterministic
  --shuffle_train_data false
  --eval_strategy no
  --eval_steps 10000
  --save_steps 1000
  --skip_inference
  --max_steps 400
  --warmup_steps 300
  --batch_size 1
  --learning_rate 1e-5
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="$ROOT/logs/tier5_precision_${DEVICE}_${TIMESTAMP}"

echo "============================================"
echo "Tier 5 precision comparison — device: $DEVICE"
echo "Output base: $BASE_OUT"
echo "============================================"

# ── fp32 ───────────────────────────────────────────────────────────
echo ""
echo "=== Tier 5: fp32 ==="
bash "$TRAIN_SCRIPT" \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT" \
  --exp_name tier5_fp32 \
  --fp fp32

# ── bf16 ───────────────────────────────────────────────────────────
echo ""
echo "=== Tier 5: bf16 ==="
bash "$TRAIN_SCRIPT" \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT" \
  --exp_name tier5_bf16 \
  --fp bf16

echo ""
echo "=== Done ($DEVICE) ==="
echo "Results: $BASE_OUT"
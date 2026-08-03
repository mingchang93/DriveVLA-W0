#!/usr/bin/env bash
#
# Alignment experiments: 4 tiers × 2 devices = 8 runs
#
# Usage:
#   bash scripts/run_alignment_experiments.sh --device cuda
#   bash scripts/run_alignment_experiments.sh --device npu \
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

# ── Paths: use CLI args if given, otherwise defaults ───────────────
PROJECT_ROOT="${PROJECT_ROOT:-/data/models/DriveVLA-W0}"
DATA_ROOT="${DATA_ROOT:-/data/models/DriveVLA-W0}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"

MODEL_PATH="$MODEL_ROOT/Emu3-Stage1"
TRAIN_PKL="$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl"
TEST_PKL="$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl"
TRAIN_PKL_FIXED="${TRAIN_PKL%.pkl}_fixed.pkl"
TEST_PKL_FIXED="${TEST_PKL%.pkl}_fixed.pkl"

# ── Fix pickle paths for the local machine ─────────────────────────
echo "=== Fixing pickle paths ==="
python "$ROOT/tools/fix_pickle_paths.py" \
    "$TRAIN_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

python "$ROOT/tools/fix_pickle_paths.py" \
    "$TEST_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

# Move fixed pickles into place (idempotent)
[ -f "$TRAIN_PKL_FIXED" ] && mv "$TRAIN_PKL_FIXED" "$TRAIN_PKL"
[ -f "$TEST_PKL_FIXED" ] && mv "$TEST_PKL_FIXED" "$TEST_PKL"
echo "=== Pickle paths fixed ==="

# ── Common flags (all experiments) ─────────────────────────────────
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
  --fp fp32
  --save_steps 100
  --skip_inference
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="$ROOT/logs/alignment_${DEVICE}_${TIMESTAMP}"

echo "============================================"
echo "Alignment experiments — device: $DEVICE"
echo "Output base: $BASE_OUT"
echo "============================================"

# ── Tier 0: warmup=200 only ────────────────────────────────────────
echo ""
echo "=== Tier 0: warmup=200, max_grad_norm=5.0, lr=1e-5, batch=1 ==="
bash "$TRAIN_SCRIPT" \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT" \
  --exp_name tier0_warmup200 \
  --max_steps 300 \
  --warmup_steps 200 \
  --batch_size 1 \
  --learning_rate 1e-5

# ── Tier 1: + symmetric clipping ───────────────────────────────────
echo ""
echo "=== Tier 1: + max_grad_norm=2.0, lr=2e-5 ==="
bash "$TRAIN_SCRIPT" \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT" \
  --exp_name tier1_clip2_lr2e5 \
  --max_steps 300 \
  --warmup_steps 200 \
  --batch_size 1 \
  --max_grad_norm 2.0 \
  --learning_rate 2e-5

# ── Tier 2: + larger batch ─────────────────────────────────────────
echo ""
echo "=== Tier 2: + batch_size=2 ==="
bash "$TRAIN_SCRIPT" \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT" \
  --exp_name tier2_batch2 \
  --max_steps 150 \
  --warmup_steps 100 \
  --batch_size 2 \
  --max_grad_norm 2.0 \
  --learning_rate 2e-5

# ── Tier 3: + adam_epsilon ─────────────────────────────────────────
echo ""
echo "=== Tier 3: + adam_epsilon=1e-4 ==="
bash "$TRAIN_SCRIPT" \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT" \
  --exp_name tier3_eps1e4 \
  --max_steps 150 \
  --warmup_steps 100 \
  --batch_size 2 \
  --max_grad_norm 2.0 \
  --learning_rate 2e-5 \
  --adam_epsilon 1e-4

echo ""
echo "=== All 4 tiers done for $DEVICE ==="
echo "Results: $BASE_OUT"
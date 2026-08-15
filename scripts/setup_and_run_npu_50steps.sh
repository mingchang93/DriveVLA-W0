#!/usr/bin/env bash
#
# Quick NPU training smoke test: train only, 50 steps.
# Skips data prep (pickles must already be fixed) and inference.
#
# Usage:
#   bash scripts/setup_and_run_npu_50steps.sh \
#       --project_root /path/to/DriveVLA-W0 \
#       --data_root /path/to/datasets \
#       --model_root /path/to/models
#
set -e

# ============================================================
# Parse required arguments
# ============================================================
PROJECT_ROOT=""
DATA_ROOT=""
MODEL_ROOT=""
BATCH_SIZE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_root) PROJECT_ROOT="$2"; shift 2 ;;
    --data_root)    DATA_ROOT="$2";    shift 2 ;;
    --model_root)   MODEL_ROOT="$2";   shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
    --help|-h)
      echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path> [--batch_size <int>]"
      echo ""
      echo "Required:"
      echo "  --project_root   Path to the DriveVLA-W0 repo"
      echo "  --data_root      Path to datasets (pickles + VQ code zips)"
      echo "  --model_root     Path to pretrained models (Emu3-Stage1, etc.)"
      echo ""
      echo "Optional:"
      echo "  --batch_size     Per-GPU train batch size (default 1)"
      echo ""
      echo "Note: runs 50 steps only; no data prep, no inference."
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path>"
      exit 1
      ;;
  esac
done

if [ -z "$PROJECT_ROOT" ] || [ -z "$DATA_ROOT" ] || [ -z "$MODEL_ROOT" ]; then
  echo "ERROR: --project_root, --data_root, and --model_root are required."
  echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path>"
  exit 1
fi

cd "$PROJECT_ROOT"

# ============================================================
# Derived paths
# ============================================================
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/scripts_train/train_base_ar_withou_moe.sh"
MODEL_PATH="$MODEL_ROOT/Emu3-Stage1"
TRAIN_PKL="$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/logs/npu_50steps_${TIMESTAMP}"

echo "============================================"
echo "DriveVLA-W0 — NPU Quick Train (50 steps)"
echo "============================================"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  DATA_ROOT:    $DATA_ROOT"
echo "  MODEL_ROOT:   $MODEL_ROOT"
echo "  output:       $OUTPUT_DIR"
echo ""

# ============================================================
# Train (50 steps only)
# ============================================================
echo "=== Training (50 steps) ==="

bash "$TRAIN_SCRIPT" \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$TRAIN_PKL" \
    --output_dir "$OUTPUT_DIR" \
    --ngpus 8 \
    --batch_size "$BATCH_SIZE" \
    --warmup_steps 100 \
    --logging_steps 1 \
    --device npu \
    --log_data_hash \
    --deterministic \
    --shuffle_train_data false \
    --eval_strategy no \
    --eval_steps 10000 \
    --fp bf16 \
    --max_steps 50 \
    --save_steps 200 \
    --skip_inference \
    --exp_name bf16_50steps

echo ""
echo "============================================"
echo "NPU quick train complete."
echo "  Output: $OUTPUT_DIR"
echo "============================================"

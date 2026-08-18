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
BATCH_SIZES="1"
ASCEND_DEVICES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_root)    PROJECT_ROOT="$2";    shift 2 ;;
    --data_root)       DATA_ROOT="$2";       shift 2 ;;
    --model_root)      MODEL_ROOT="$2";      shift 2 ;;
    --batch_sizes)     BATCH_SIZES="$2";     shift 2 ;;
    --ascend_devices)  ASCEND_DEVICES="$2";  shift 2 ;;
    --help|-h)
      echo "Usage: $0 --project_root <path> --data_root <path> --model_root <path> [--batch_sizes <int,int,...>] [--ascend_devices <dev_list>]"
      echo ""
      echo "Required:"
      echo "  --project_root    Path to the DriveVLA-W0 repo"
      echo "  --data_root       Path to datasets (pickles + VQ code zips)"
      echo "  --model_root      Path to pretrained models (Emu3-Stage1, etc.)"
      echo ""
      echo "Optional:"
      echo "  --batch_sizes     Comma-separated list of per-GPU batch sizes (default 1)"
      echo "  --ascend_devices  Comma-separated NPU device list, e.g. 0,1,2,3 (default: all 8)"
      echo ""
      echo "Note: runs 50 steps per batch size; no data prep, no inference."
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

# One rank per visible NPU when --ascend_devices restricts the device set
# (ASCEND_RT_VISIBLE_DEVICES remaps physical ids to logical 0..N-1, so a
# 8-rank launch on fewer visible devices dies with "Invalid device ID").
if [ -n "$ASCEND_DEVICES" ]; then
  NGPUS=$(($(echo "$ASCEND_DEVICES" | tr -cd ',' | wc -c) + 1))
  export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICES"
else
  NGPUS=8
fi

# NPU memory allocator: avoid pre-caching large blocks
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

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
echo "  PROJECT_ROOT:   $PROJECT_ROOT"
echo "  DATA_ROOT:      $DATA_ROOT"
echo "  MODEL_ROOT:     $MODEL_ROOT"
echo "  batch_sizes:    $BATCH_SIZES"
echo "  ascend_devices: ${ASCEND_DEVICES:-all 8}"
echo "  ngpus:          $NGPUS"
echo "  output:         $OUTPUT_DIR"
echo ""

# ============================================================
# Train (50 steps per batch size)
# ============================================================
IFS=',' read -ra BS_ARRAY <<< "$BATCH_SIZES"
for bs in "${BS_ARRAY[@]}"; do
  echo "=== Training (batch_size=$bs) ==="

  bash "$TRAIN_SCRIPT" \
      --model_name_or_path "$MODEL_PATH" \
      --data_path "$TRAIN_PKL" \
      --output_dir "$OUTPUT_DIR" \
      --ngpus "$NGPUS" \
      --batch_size "$bs" \
      --warmup_steps 100 \
      --logging_steps 1 \
      --device npu \
      --log_data_hash \
      --deterministic \
      --shuffle_train_data false \
      --eval_strategy no \
      --save_strategy no \
      --eval_steps 10000 \
      --fp bf16 \
      --max_steps 50 \
      --save_steps 200 \
      --skip_inference \
      --no_save_weights \
      --exp_name "bf16_50steps_bs${bs}"

  echo ""
done
echo "============================================"
echo "NPU quick train complete."
echo "  Output: $OUTPUT_DIR"
echo "============================================"

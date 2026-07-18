#!/usr/bin/env bash
#
# One-shot SFT launcher for DriveVLA-W0 on pre-Ampere GPUs (V100, etc.)
#
# Usage:
#   bash run_sft.sh                    # defaults
#   bash run_sft.sh --fp fp16 --ngpus 4 --batch_size 4
#

set -e

ROOT=$(cd "$(dirname "$0")" && pwd)

# ============================================================
# Parse args
# ============================================================
FP="bf16"
NGPUS=8
BATCH_SIZE=6
MASTER_PORT=23457
EXP_NAME="sft_base_ar"
SKIP_INFERENCE=false
DATA_DIR="$ROOT/data/navsim/processed_data/meta"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp)                FP="$2";         shift 2 ;;
    --ngpus)             NGPUS="$2";       shift 2 ;;
    --batch_size)        BATCH_SIZE="$2";  shift 2 ;;
    --master_port)       MASTER_PORT="$2"; shift 2 ;;
    --exp_name)          EXP_NAME="$2";    shift 2 ;;
    --skip_inference)    SKIP_INFERENCE=true; shift ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --fp            <str>   Precision: bf16, fp16, or fp32 (default: bf16)"
      echo "  --ngpus         <int>   Number of GPUs (default: 8)"
      echo "  --batch_size    <int>   Per-device batch size (default: 6)"
      echo "  --master_port   <int>   Distributed port (default: 23457)"
      echo "  --exp_name      <str>   Experiment name (default: sft_base_ar)"
      echo "  --skip_inference        Skip inference after training"
      exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# ============================================================
# Step 0: symlink train/ -> utils/
# ============================================================
ln -sf "$ROOT/utils" "$ROOT/train"

export PYTHONPATH="$ROOT:$ROOT/reference/Emu3:$PYTHONPATH"

# ============================================================
# Step 1: Fix pickle paths
# ============================================================
echo "=== Step 1: Fixing pickle paths ==="
TRAIN_PKL="$DATA_DIR/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
TEST_PKL="$DATA_DIR/navsim_emu_vla_256_144_test_pre_1s.pkl"
TRAIN_PKL_FIXED="$DATA_DIR/navsim_emu_vla_256_144_trainval_pre_1s_fixed.pkl"
TEST_PKL_FIXED="$DATA_DIR/navsim_emu_vla_256_144_test_pre_1s_fixed.pkl"

for pkl in "$TRAIN_PKL" "$TEST_PKL"; do
  if [ ! -f "$pkl" ]; then
    echo "ERROR: $pkl not found. Download it first."
    exit 1
  fi
done

python "$ROOT/tools/fix_pickle_paths.py" "$TRAIN_PKL"
python "$ROOT/tools/fix_pickle_paths.py" "$TEST_PKL"

echo ""

# ============================================================
# Step 2: Set precision
# ============================================================
case "$FP" in
  bf16)  FP_FLAGS="--bf16 True --fp16 False" ;;
  fp16)  FP_FLAGS="--bf16 False --fp16 True" ;;
  fp32)  FP_FLAGS="--bf16 False --fp16 False" ;;
  *)     echo "ERROR: --fp must be bf16, fp16, or fp32"; exit 1 ;;
esac

echo "=== Step 2: Training ==="
echo "  precision:  $FP"
echo "  ngpus:      $NGPUS"
echo "  batch_size: $BATCH_SIZE"
echo "  output:     $ROOT/logs/$EXP_NAME"
echo "  train pkl:  $TRAIN_PKL_FIXED"
echo "  test pkl:   $TEST_PKL_FIXED"
echo ""

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    train/train_moe.py \
    --model_name_or_path "$ROOT/pretrained_models/Emu3-Stage1" \
    --model_config_path "$ROOT/configs/moe_fast_video.json" \
    --actions_format fast \
    --action_tokenizer_path "$ROOT/configs/fast" \
    --deepspeed "$ROOT/scripts/sft/zero3_offload.json" \
    --output_dir "$ROOT/logs/${EXP_NAME}" \
    $FP_FLAGS \
    --tf32 False \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 1e-6 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --data_path "$TRAIN_PKL_FIXED" \
    --max_steps 4000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_steps 50 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --frames 1 \
    --action_frames 8 \
    --max_position_embeddings 1400 \
    --seed 42 \
    --attn_type sdpa \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --eval_strategy no \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_action False \
    --actions True \
    --use_gripper False \
    --driving True \
    --evaluation_strategy steps \
    --eval_steps 400 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --use_previous_actions True \
    --report_to tensorboard \
    --data_type navsim_vava

echo ""
echo "=== Training finished ==="

# ============================================================
# Step 3: Inference (optional)
# ============================================================
if [ "$SKIP_INFERENCE" = false ]; then
  echo ""
  echo "=== Step 3: Inference ==="
  echo "  checkpoint: $ROOT/logs/${EXP_NAME}"
  echo "  test data:  $TEST_PKL_FIXED"
  echo ""

  torchrun --nproc_per_node=${NGPUS} \
    inference/vla/inference_action_navsim_with_previous_action_last_VAVA.py \
    --emu_hub "$ROOT/logs/${EXP_NAME}" \
    --output_dir "$ROOT/logs/${EXP_NAME}/json_output" \
    --train_meta_pkl "$TEST_PKL_FIXED" \
    --input_num_frame "1"

  echo "=== Inference done ==="
  echo "Results at: $ROOT/logs/${EXP_NAME}/json_output"
else
  echo "Skipping inference (--skip_inference)."
fi

echo ""
echo "All done."

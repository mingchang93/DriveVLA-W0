#!/usr/bin/env bash
#
# Base AR training (without MoE) for DriveVLA-W0
#
# Usage:
#   bash scripts/scripts_train/train_base_ar_withou_moe.sh \
#       --model_name_or_path ./pretrained_models/Emu3-Stage1 \
#       --data_path ./data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl
#
# All paths have sensible defaults — you can override any of them.
# Run without arguments to use defaults (assuming standard repo layout):
#   bash scripts/scripts_train/train_base_ar_withou_moe.sh
#

set -e

# ============================================================
# Defaults — adjust to your repo layout
# ============================================================
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

DEFAULT_MODEL_NAME_OR_PATH="$ROOT/pretrained_models/Emu3-Stage1"
DEFAULT_MODEL_CONFIG_PATH="$ROOT/configs/moe_fast_video.json"
DEFAULT_ACTION_TOKENIZER_PATH="$ROOT/configs/fast"
DEFAULT_DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero3_offload.json"
DEFAULT_DATA_PATH="$ROOT/data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
DEFAULT_TEST_DATA_PATH="$ROOT/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl"
DEFAULT_OUTPUT_DIR="$ROOT/logs/train_base_ar"
DEFAULT_INPUT_NUM_FRAME="1"

# ============================================================
# Parse input arguments
# ============================================================
MODEL_NAME_OR_PATH="$DEFAULT_MODEL_NAME_OR_PATH"
MODEL_CONFIG_PATH="$DEFAULT_MODEL_CONFIG_PATH"
ACTION_TOKENIZER_PATH="$DEFAULT_ACTION_TOKENIZER_PATH"
DEEPSPEED_CONFIG="$DEFAULT_DEEPSPEED_CONFIG"
DEEPSPEED_CONFIG_EXPLICIT=false
DATA_PATH="$DEFAULT_DATA_PATH"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
TEST_DATA_PATH="$DEFAULT_TEST_DATA_PATH"
NGPUS=8
MASTER_PORT=23457
BATCH_SIZE=6
EXP_NAME="train_base_ar"
INPUT_NUM_FRAME="$DEFAULT_INPUT_NUM_FRAME"
SKIP_INFERENCE=false
FP="bf16"
ATTN_TYPE="sdpa"
MAX_STEPS=4000
SAVE_STEPS=2000
EVAL_STRATEGY="no"
EVAL_STEPS=400
SEED=42
DETERMINISTIC=false
DET_FLAG=""
LOGGING_STEPS=10
WARMUP_STEPS=50
ZERO_STAGE=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)     MODEL_NAME_OR_PATH="$2";       shift 2 ;;
    --model_config_path)      MODEL_CONFIG_PATH="$2";        shift 2 ;;
    --action_tokenizer_path)  ACTION_TOKENIZER_PATH="$2";    shift 2 ;;
    --deepspeed_config)       DEEPSPEED_CONFIG="$2";  DEEPSPEED_CONFIG_EXPLICIT=true;  shift 2 ;;
    --zero_stage)             ZERO_STAGE="$2";               shift 2 ;;
    --data_path)              DATA_PATH="$2";                shift 2 ;;
    --test_data_path)         TEST_DATA_PATH="$2";           shift 2 ;;
    --output_dir)             OUTPUT_DIR="$2";               shift 2 ;;
    --ngpus)                  NGPUS="$2";                    shift 2 ;;
    --master_port)            MASTER_PORT="$2";              shift 2 ;;
    --batch_size)             BATCH_SIZE="$2";               shift 2 ;;
    --exp_name)               EXP_NAME="$2";                 shift 2 ;;
    --input_num_frame)        INPUT_NUM_FRAME="$2";          shift 2 ;;
    --fp)                     FP="$2";                       shift 2 ;;
    --attn_type)              ATTN_TYPE="$2";                shift 2 ;;
    --max_steps)              MAX_STEPS="$2";                shift 2 ;;
    --save_steps)             SAVE_STEPS="$2";               shift 2 ;;
    --eval_strategy)          EVAL_STRATEGY="$2";            shift 2 ;;
    --eval_steps)             EVAL_STEPS="$2";               shift 2 ;;
    --seed)                   SEED="$2";                     shift 2 ;;
    --deterministic)          DETERMINISTIC=true;            shift ;;
    --logging_steps)          LOGGING_STEPS="$2";            shift 2 ;;
    --warmup_steps)           WARMUP_STEPS="$2";             shift 2 ;;
    --skip_inference)         SKIP_INFERENCE=true;           shift ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options (all optional, defaults in parentheses):"
      echo "  --model_name_or_path       <path>  ($DEFAULT_MODEL_NAME_OR_PATH)"
      echo "  --model_config_path        <path>  ($DEFAULT_MODEL_CONFIG_PATH)"
      echo "  --action_tokenizer_path    <path>  ($DEFAULT_ACTION_TOKENIZER_PATH)"
      echo "  --deepspeed_config         <path>  ($DEFAULT_DEEPSPEED_CONFIG)"
      echo "  --zero_stage               <int>   (3) — shortcut: 2→zero2_offload, 3→zero3_offload"
      echo "  --data_path                <path>  ($DEFAULT_DATA_PATH)"
      echo "  --test_data_path           <path>  ($DEFAULT_TEST_DATA_PATH)"
      echo "  --output_dir               <path>  ($DEFAULT_OUTPUT_DIR)"
      echo "  --ngpus                    <int>   (8)"
      echo "  --master_port              <int>   (23457)"
      echo "  --batch_size               <int>   (6)"
      echo "  --exp_name                 <str>   (train_base_ar)"
      echo "  --input_num_frame          <int>   (1)"
      echo "  --fp                       <str>   (bf16) — bf16, fp16, or fp32"
      echo "  --attn_type                <str>   (sdpa) — sdpa, fa2, or eager"
      echo "  --max_steps                <int>   (4000)"
      echo "  --save_steps               <int>   (2000)"
      echo "  --eval_strategy            <str>   (no) — no, steps, or epoch"
      echo "  --eval_steps               <int>   (400) — used when eval_strategy=steps"
      echo "  --seed                     <int>   (42)"
      echo "  --deterministic                   Strict reproducibility (NPU vs GPU debug)"
      echo "  --logging_steps            <int>   (10)"
      echo "  --warmup_steps             <int>   (50)"
      echo "  --skip_inference                   Skip inference after training"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

# Convert boolean flags to CLI arguments
[ "$DETERMINISTIC" = true ] && DET_FLAG="--deterministic"

# Resolve --zero_stage shorthand to config path.
# --deepspeed_config takes priority if explicitly given.
if [ "$DEEPSPEED_CONFIG_EXPLICIT" = false ]; then
  case "$ZERO_STAGE" in
    2) DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero2_offload.json" ;;
    3) DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero3_offload.json" ;;
    *) echo "ERROR: --zero_stage must be 2 or 3 (got '$ZERO_STAGE')"; exit 1 ;;
  esac
fi

# ============================================================
# Fix: symlink train/ → utils/
# ============================================================
ln -sf "$ROOT/utils" "$ROOT/train"

export PYTHONPATH="$ROOT:$ROOT/reference/Emu3:$PYTHONPATH"

# Reduce GPU memory fragmentation (suggested by PyTorch's OOM message)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# Verify paths
# ============================================================
echo "=== Training config ==="
echo "  model_name_or_path:      $MODEL_NAME_OR_PATH"
echo "  model_config_path:       $MODEL_CONFIG_PATH"
echo "  action_tokenizer_path:   $ACTION_TOKENIZER_PATH"
echo "  zero_stage:              $ZERO_STAGE"
echo "  deepspeed_config:        $DEEPSPEED_CONFIG"
echo "  data_path:               $DATA_PATH"
echo "  output_dir:              $OUTPUT_DIR"
echo "  test_data_path:          $TEST_DATA_PATH"
echo "  ngpus:                   $NGPUS"
echo "  batch_size:              $BATCH_SIZE"
echo "  master_port:             $MASTER_PORT"
echo "  fp:                      $FP"
echo "  attn_type:               $ATTN_TYPE"
echo "  max_steps:               $MAX_STEPS"
echo "  save_steps:              $SAVE_STEPS"
echo "  eval_strategy:           $EVAL_STRATEGY"
echo "  eval_steps:              $EVAL_STEPS"
echo "  seed:                    $SEED"
echo "  deterministic:           $DETERMINISTIC"
echo "  logging_steps:           $LOGGING_STEPS"
echo "  warmup_steps:            $WARMUP_STEPS"
echo "  skip_inference:          $SKIP_INFERENCE"
echo ""

for p in "$MODEL_NAME_OR_PATH" "$MODEL_CONFIG_PATH" "$ACTION_TOKENIZER_PATH" "$DEEPSPEED_CONFIG" "$DATA_PATH"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: $p not found. Override via --$(
      case "$p" in
        "$MODEL_NAME_OR_PATH")     echo "model_name_or_path" ;;
        "$MODEL_CONFIG_PATH")      echo "model_config_path" ;;
        "$ACTION_TOKENIZER_PATH")  echo "action_tokenizer_path" ;;
        "$DEEPSPEED_CONFIG")       echo "deepspeed_config" ;;
        "$DATA_PATH")              echo "data_path" ;;
      esac
    ) or place the file at the default path."
    exit 1
  fi
done

echo "=== Setting precision: $FP ==="
case "$FP" in
  bf16)  FP_FLAGS="--bf16 True --fp16 False" ;;
  fp16)  FP_FLAGS="--bf16 False --fp16 True" ;;
  fp32)  FP_FLAGS="--bf16 False --fp16 False" ;;
  *)     echo "ERROR: --fp must be bf16, fp16, or fp32 (got '$FP')"; exit 1 ;;
esac

# ============================================================
# Launch training
# ============================================================
torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    train/train_moe.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --model_config_path "$MODEL_CONFIG_PATH" \
    --actions_format fast \
    --action_tokenizer_path "$ACTION_TOKENIZER_PATH" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
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
    --data_path "$DATA_PATH" \
    --max_steps "$MAX_STEPS" \
    --dataloader_num_workers 12 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_steps "$WARMUP_STEPS" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --frames 1 \
    --action_frames 8 \
    --max_position_embeddings 1400 \
    --seed "$SEED" \
    $DET_FLAG \
    --attn_type "$ATTN_TYPE" \
    --logging_steps "$LOGGING_STEPS" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_action False \
    --actions True \
    --use_gripper False \
    --driving True \
    --evaluation_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --use_previous_actions True \
    --report_to tensorboard \
    --data_type navsim_vava

# ============================================================
# Inference (skipped with --skip_inference)
# ============================================================
if [ "$SKIP_INFERENCE" = false ]; then
  echo ""
  echo "=== Running inference on test set ==="
  echo "  checkpoint:   ${OUTPUT_DIR}/${EXP_NAME}"
  echo "  test_data:    ${TEST_DATA_PATH}"
  echo "  output:       ${OUTPUT_DIR}/${EXP_NAME}/json_output"
  echo ""

  torchrun --nproc_per_node=${NGPUS} \
    inference/vla/inference_action_navsim_with_previous_action_last_VAVA.py \
    --emu_hub "${OUTPUT_DIR}/${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}/${EXP_NAME}/json_output" \
    --train_meta_pkl "${TEST_DATA_PATH}" \
    --input_num_frame "${INPUT_NUM_FRAME}"

  echo "=== Inference done ==="
  echo "Results at: ${OUTPUT_DIR}/${EXP_NAME}/json_output"
else
  echo ""
  echo "Skipping inference (--skip_inference)."
fi

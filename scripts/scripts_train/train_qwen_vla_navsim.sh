#!/usr/bin/env bash
#
# Qwen VLA (ROSS) training for DriveVLA-W0
#
# Usage:
#   bash scripts/scripts_train/train_qwen_vla_navsim.sh \
#       --sensor_blobs /path/to/sensor_blobs \
#       --data_path /path/to/train.pkl
#
# Run without arguments to use defaults.
# Use --help for all options.
#

set -e

# ============================================================
# Defaults — adjust to your repo layout
# ============================================================
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

DEFAULT_SENSOR_BLOBS="$HOME/navsim_workspace/dataset/sensor_blobs/trainval"
DEFAULT_NAVSIM_LOGS="$HOME/navsim_workspace/dataset/navsim_logs/trainval"
DEFAULT_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_SD_MODEL_PATH="$ROOT/pretrained_models/stable-diffusion-v1-5"
DEFAULT_ACTION_TOKENIZER_PATH="$ROOT/configs/fast"
DEFAULT_DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero3_offload.json"
# Pickle files live in the repo (or anywhere — override via --data_path / --test_data_path)
DEFAULT_DATA_PATH="$ROOT/data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
DEFAULT_TEST_DATA_PATH="$ROOT/data/navsim/processed_data/meta/navsim_emu_vla_256_144_test_pre_1s.pkl"
# Data root for camera images — derived from NavSim workspace + split
DEFAULT_DATA_ROOT="$DEFAULT_SENSOR_BLOBS"
DEFAULT_OUTPUT_DIR="$ROOT/logs/train_qwen_vla_$TIMESTAMP"
DEFAULT_DATASET_TYPE="navsim2va_ross"

# ============================================================
# Parse input arguments
# ============================================================
MODEL_NAME_OR_PATH="$DEFAULT_MODEL_NAME_OR_PATH"
SENSOR_BLOBS="$DEFAULT_SENSOR_BLOBS"
NAVSIM_LOGS="$DEFAULT_NAVSIM_LOGS"
SD_MODEL_PATH="$DEFAULT_SD_MODEL_PATH"
ACTION_TOKENIZER_PATH="$DEFAULT_ACTION_TOKENIZER_PATH"
DEEPSPEED_CONFIG="$DEFAULT_DEEPSPEED_CONFIG"
DEEPSPEED_CONFIG_EXPLICIT=false
DATA_PATH="$DEFAULT_DATA_PATH"
TEST_DATA_PATH="$DEFAULT_TEST_DATA_PATH"
DATA_ROOT="$DEFAULT_DATA_ROOT"
DATASET_TYPE="$DEFAULT_DATASET_TYPE"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
NGPUS=8
MASTER_PORT=23458
BATCH_SIZE=2
EXP_NAME="train_qwen_vla_navsim"
FP="bf16"
ATTN_TYPE="sdpa"
DEVICE="auto"
MAX_STEPS=4000
SAVE_STEPS=2000
EVAL_STRATEGY="no"
EVAL_STEPS=400
SEED=42
SHUFFLE_TRAIN_DATA=true
DETERMINISTIC=false
DET_FLAG=""
LOG_DATA_HASH=false
HASH_FLAG=""
LOGGING_STEPS=10
WARMUP_STEPS=50
ZERO_STAGE=3
MODEL_MAX_LENGTH=4096
FUTURE_NUMS=8
TUNE_MM_LLM=true
TUNE_MM_MLP=true
TUNE_MM_VISION=true
USE_PREVIOUS_ACTIONS=true
CUR_FRAME_IDX=0
LEARNING_RATE="5e-5"
MAX_GRAD_NORM="5.0"
SKIP_INFERENCE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)     MODEL_NAME_OR_PATH="$2";       shift 2 ;;
    --sensor_blobs)           SENSOR_BLOBS="$2";             shift 2 ;;
    --navsim_logs)            NAVSIM_LOGS="$2";               shift 2 ;;
    --sd_model_path)          SD_MODEL_PATH="$2";            shift 2 ;;
    --action_tokenizer_path)  ACTION_TOKENIZER_PATH="$2";    shift 2 ;;
    --deepspeed_config)       DEEPSPEED_CONFIG="$2";  DEEPSPEED_CONFIG_EXPLICIT=true;  shift 2 ;;
    --zero_stage)             ZERO_STAGE="$2";               shift 2 ;;
    --data_path)              DATA_PATH="$2";                shift 2 ;;
    --test_data_path)         TEST_DATA_PATH="$2";           shift 2 ;;
    --data_root)              DATA_ROOT="$2";                shift 2 ;;
    --dataset_type)           DATASET_TYPE="$2";             shift 2 ;;
    --output_dir)             OUTPUT_DIR="$2";               shift 2 ;;
    --ngpus)                  NGPUS="$2";                    shift 2 ;;
    --master_port)            MASTER_PORT="$2";              shift 2 ;;
    --batch_size)             BATCH_SIZE="$2";               shift 2 ;;
    --exp_name)               EXP_NAME="$2";                 shift 2 ;;
    --fp)                     FP="$2";                       shift 2 ;;
    --attn_type)              ATTN_TYPE="$2";                shift 2 ;;
    --device)                 DEVICE="$2";                   shift 2 ;;
    --max_steps)              MAX_STEPS="$2";                shift 2 ;;
    --save_steps)             SAVE_STEPS="$2";               shift 2 ;;
    --eval_strategy)          EVAL_STRATEGY="$2";            shift 2 ;;
    --eval_steps)             EVAL_STEPS="$2";               shift 2 ;;
    --seed)                   SEED="$2";                     shift 2 ;;
    --shuffle_train_data)     SHUFFLE_TRAIN_DATA="$2";       shift 2 ;;
    --deterministic)          DETERMINISTIC=true;            shift ;;
    --log_data_hash)          LOG_DATA_HASH=true;            shift ;;
    --logging_steps)          LOGGING_STEPS="$2";            shift 2 ;;
    --warmup_steps)           WARMUP_STEPS="$2";             shift 2 ;;
    --model_max_length)       MODEL_MAX_LENGTH="$2";         shift 2 ;;
    --future_nums)            FUTURE_NUMS="$2";              shift 2 ;;
    --tune_mm_llm)            TUNE_MM_LLM="$2";              shift 2 ;;
    --tune_mm_mlp)            TUNE_MM_MLP="$2";              shift 2 ;;
    --tune_mm_vision)         TUNE_MM_VISION="$2";           shift 2 ;;
    --use_previous_actions)   USE_PREVIOUS_ACTIONS="$2";     shift 2 ;;
    --cur_frame_idx)          CUR_FRAME_IDX="$2";            shift 2 ;;
    --learning_rate)          LEARNING_RATE="$2";            shift 2 ;;
    --max_grad_norm)          MAX_GRAD_NORM="$2";            shift 2 ;;
    --skip_inference)         SKIP_INFERENCE=true;           shift ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options (all optional, defaults in parentheses):"
      echo "  --model_name_or_path       <path>  ($DEFAULT_MODEL_NAME_OR_PATH)"
      echo "  --sensor_blobs             <path>  ($DEFAULT_SENSOR_BLOBS) — camera image root (data_root)"
      echo "  --navsim_logs              <path>  ($DEFAULT_NAVSIM_LOGS) — NavSim annotation logs dir"
      echo "  --sd_model_path            <path>  ($DEFAULT_SD_MODEL_PATH) — parent dir with unet/ and vae/ subdirs"
      echo "  --action_tokenizer_path    <path>  ($DEFAULT_ACTION_TOKENIZER_PATH)"
      echo "  --deepspeed_config         <path>  ($DEFAULT_DEEPSPEED_CONFIG)"
      echo "  --zero_stage               <int>   (3) — shortcut: 2→zero2_offload, 3→zero3_offload"
      echo "  --data_path                <path>  ($DEFAULT_DATA_PATH)"
      echo "  --test_data_path           <path>  ($DEFAULT_TEST_DATA_PATH) — used for post-training inference"
      echo "  --data_root                <path>  (derived from --sensor_blobs) — camera image root"
      echo "  --dataset_type             <str>   (navsim2va_ross) — or nuplan2va_ross"
      echo "  --output_dir               <path>  ($DEFAULT_OUTPUT_DIR)"
      echo "  --ngpus                    <int>   (8)"
      echo "  --master_port              <int>   (23458)"
      echo "  --batch_size               <int>   (2) — Qwen+SD heavier than Emu3"
      echo "  --exp_name                 <str>   (train_qwen_vla_navsim)"
      echo "  --fp                       <str>   (bf16) — bf16, fp16, or fp32"
      echo "  --attn_type                <str>   (sdpa) — sdpa, fa2, or eager. FA2= CUDA-only"
      echo "  --device                   <str>   (auto) — auto, cuda, or npu"
      echo "  --max_steps                <int>   (4000)"
      echo "  --save_steps               <int>   (2000)"
      echo "  --eval_strategy            <str>   (no) — no, steps, or epoch"
      echo "  --eval_steps               <int>   (400) — used when eval_strategy=steps"
      echo "  --seed                     <int>   (42)"
      echo "  --shuffle_train_data       <bool>  (true) — true=shuffle, false=deterministic order"
      echo "  --deterministic                    Strict reproducibility (NPU vs GPU debug)"
      echo "  --log_data_hash                    Log SHA256 hash per batch for cross-platform data verification"
      echo "  --logging_steps            <int>   (10)"
      echo "  --warmup_steps             <int>   (50)"
      echo "  --model_max_length         <int>   (4096)"
      echo "  --future_nums              <int>   (8) — future action steps to predict"
      echo "  --tune_mm_llm              <bool>  (true) — train language model"
      echo "  --tune_mm_mlp              <bool>  (true) — train vision-language merger"
      echo "  --tune_mm_vision           <bool>  (true) — train vision encoder"
      echo "  --use_previous_actions     <bool>  (true) — include historical actions"
      echo "  --cur_frame_idx            <int>   (0) — current frame index"
      echo "  --learning_rate            <str>   (5e-5)"
      echo "  --max_grad_norm            <str>   (5.0)"
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

# Derive data_root from sensor_blobs (honour explicit --data_root if set)
if [ "$DATA_ROOT" = "$DEFAULT_DATA_ROOT" ]; then
  DATA_ROOT="$SENSOR_BLOBS"
fi

# Convert boolean flags to CLI arguments
[ "$DETERMINISTIC" = true ] && DET_FLAG="--deterministic"
[ "$LOG_DATA_HASH" = true ] && HASH_FLAG="--log_data_hash"
# Data shuffling: true → shuffle (default), false → deterministic order (NPU/GPU alignment)
SHUFFLE_FLAG="--dataloader_shuffle $SHUFFLE_TRAIN_DATA"

# ============================================================
# Device-specific environment (NPU vs CUDA)
# ============================================================
if [ "$DEVICE" = "npu" ]; then
  # FA2 is CUDA-only; force sdpa on NPU
  [ "$ATTN_TYPE" = "fa2" ] && ATTN_TYPE="sdpa"
  # NPU communication backend (HCCL) — NCCL is CUDA-only
  export HCCL_DETERMINISTIC=TRUE
  # Non-saturation mode: overflow → Inf/NaN (matches GPU default behavior)
  export INF_NAN_MODE_ENABLE=1
  # Stream sync for memory corruption diagnosis (uncomment to enable)
  # export ASCEND_LAUNCH_BLOCKING=1
  # Ascend EP / CANN tuning (optional)
  : "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
elif [ "$DEVICE" = "cuda" ]; then
  export NCCL_DEBUG=INFO
  export NCCL_BLOCKING_WAIT=1
  export NCCL_DETERMINISTIC=TRUE
  export CUDA_DEVICE_MAX_CONNECTIONS=1
else
  # "auto" — let Python detect; skip device-specific vars
  :
fi

# ============================================================
# DeepSpeed config resolution (--zero_stage shortcut)
# ============================================================
if [ "$DEEPSPEED_CONFIG_EXPLICIT" = false ]; then
  case "$ZERO_STAGE" in
    2) DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero2_offload.json" ;;
    3) DEEPSPEED_CONFIG="$ROOT/scripts/sft/zero3_offload.json" ;;
    *) echo "ERROR: --zero_stage must be 2 or 3 (got '$ZERO_STAGE')"; exit 1 ;;
  esac
fi

# ============================================================
# PYTHONPATH — Qwen-VL needs its reference subtree on the path
# (No symlink needed — train_qwen_vla.py manages its own paths)
# ============================================================
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune:$PYTHONPATH"

# Reduce GPU memory fragmentation (suggested by PyTorch OOM message)
# Only applies to CUDA — harmless but meaningless on NPU
if [ "$DEVICE" != "npu" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

# Forward device preference to Python
export DEVICE

# ============================================================
# Resolve SD model sub-paths
# ============================================================
# User passes the parent directory (e.g. stable-diffusion-v1-5).
# The Python script expects the UNet path and derives the VAE path
# via .replace("/unet", "/vae"), so we pass <sd_model_path>/unet.
SD_MODEL_UNET_PATH="$SD_MODEL_PATH/unet"
SD_MODEL_VAE_PATH="$SD_MODEL_PATH/vae"

# ============================================================
# Verify paths
# ============================================================
echo "=== Qwen VLA Training config ==="
echo "  model_name_or_path:      $MODEL_NAME_OR_PATH"
echo "  sensor_blobs:            $SENSOR_BLOBS"
echo "  navsim_logs:             $NAVSIM_LOGS"
echo "  sd_model_path:           $SD_MODEL_PATH"
echo "    → unet:                $SD_MODEL_UNET_PATH"
echo "    → vae:                 $SD_MODEL_VAE_PATH"
echo "  action_tokenizer_path:   $ACTION_TOKENIZER_PATH"
echo "  zero_stage:              $ZERO_STAGE"
echo "  deepspeed_config:        $DEEPSPEED_CONFIG"
echo "  data_path:               $DATA_PATH"
echo "  test_data_path:          $TEST_DATA_PATH"
echo "  data_root:               $DATA_ROOT"
echo "  dataset_type:            $DATASET_TYPE"
echo "  output_dir:              ${OUTPUT_DIR}/${EXP_NAME}"
echo "  ngpus:                   $NGPUS"
echo "  batch_size:              $BATCH_SIZE"
echo "  master_port:             $MASTER_PORT"
echo "  fp:                      $FP"
echo "  attn_type:               $ATTN_TYPE"
echo "  device:                  $DEVICE"
echo "  max_steps:               $MAX_STEPS"
echo "  save_steps:              $SAVE_STEPS"
echo "  eval_strategy:           $EVAL_STRATEGY"
echo "  eval_steps:              $EVAL_STEPS"
echo "  seed:                    $SEED"
echo "  shuffle_train_data:      $SHUFFLE_TRAIN_DATA"
echo "  deterministic:           $DETERMINISTIC"
echo "  log_data_hash:           $LOG_DATA_HASH"
echo "  logging_steps:           $LOGGING_STEPS"
echo "  warmup_steps:            $WARMUP_STEPS"
echo "  model_max_length:        $MODEL_MAX_LENGTH"
echo "  future_nums:             $FUTURE_NUMS"
echo "  tune_mm_llm:             $TUNE_MM_LLM"
echo "  tune_mm_mlp:             $TUNE_MM_MLP"
echo "  tune_mm_vision:          $TUNE_MM_VISION"
echo "  use_previous_actions:    $USE_PREVIOUS_ACTIONS"
echo "  cur_frame_idx:           $CUR_FRAME_IDX"
echo "  learning_rate:           $LEARNING_RATE"
echo "  max_grad_norm:           $MAX_GRAD_NORM"
echo "  skip_inference:          $SKIP_INFERENCE"
echo ""

for p in "$MODEL_NAME_OR_PATH" "$ACTION_TOKENIZER_PATH" "$DEEPSPEED_CONFIG" "$DATA_PATH" "$DATA_ROOT"; do
  # Skip HF Hub model names (contain '/', not local paths)
  if [ "$p" = "$MODEL_NAME_OR_PATH" ] && [[ "$p" != /* ]] && [[ "$p" != ./* ]]; then
    echo "[Skip] model_name_or_path=$p (non-local, assumed HF Hub)"
    continue
  fi
  if [ ! -e "$p" ]; then
    echo "ERROR: $p not found."
    case "$p" in
      "$ACTION_TOKENIZER_PATH")  hint="--action_tokenizer_path" ;;
      "$DEEPSPEED_CONFIG")       hint="--deepspeed_config" ;;
      "$DATA_PATH")              hint="--data_path" ;;
      "$DATA_ROOT")              hint="--data_root" ;;
      *)                         hint="" ;;
    esac
    [ -n "$hint" ] && echo "Override via $hint or place the file at the default path."
    exit 1
  fi
done

# Verify SD model sub-paths separately
if [ ! -e "$SD_MODEL_UNET_PATH" ]; then
  echo "ERROR: $SD_MODEL_UNET_PATH not found."
  echo "  Expected: <sd_model_path>/unet/config.json"
  echo "  Download:  huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir $SD_MODEL_PATH"
  exit 1
fi
if [ ! -e "$SD_MODEL_VAE_PATH" ]; then
  echo "ERROR: $SD_MODEL_VAE_PATH not found."
  echo "  Expected: <sd_model_path>/vae/config.json"
  echo "  Download:  huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir $SD_MODEL_PATH"
  exit 1
fi

# ============================================================
# Precision flags
# ============================================================
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
echo ""
echo "=== Launching Qwen VLA training ==="
echo "  Command: torchrun utils/train_qwen_vla.py"
echo ""

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    utils/train_qwen_vla.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --sd_model_path "$SD_MODEL_UNET_PATH" \
    --dataset_type "$DATASET_TYPE" \
    --data_path "$DATA_PATH" \
    --data_root "$DATA_ROOT" \
    --action_tokenizer_path "$ACTION_TOKENIZER_PATH" \
    --actions_format fast \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
    $FP_FLAGS \
    --tf32 False \
    --attn_type "$ATTN_TYPE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0.1 \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --max_steps "$MAX_STEPS" \
    --dataloader_num_workers 12 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_steps "$WARMUP_STEPS" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --future_nums "$FUTURE_NUMS" \
    --seed "$SEED" \
    $SHUFFLE_FLAG \
    $DET_FLAG \
    $HASH_FLAG \
    --logging_steps "$LOGGING_STEPS" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --use_previous_actions "$USE_PREVIOUS_ACTIONS" \
    --tune_mm_llm "$TUNE_MM_LLM" \
    --tune_mm_mlp "$TUNE_MM_MLP" \
    --tune_mm_vision "$TUNE_MM_VISION" \
    --cur_frame_idx "$CUR_FRAME_IDX" \
    --action_dim 3 \
    --report_to tensorboard

echo ""
echo "=== Training complete ==="
echo "Checkpoint at: ${OUTPUT_DIR}/${EXP_NAME}"

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

  NORM_STATS_PATH="$ROOT/configs/normalizer_navsim_trainval/norm_stats.json"
  TOKEN_YAML="$ROOT/inference/navsim/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"

  torchrun --nproc_per_node=${NGPUS} \
    inference/qwen/inference_action_navsim_vava.py \
    --qwen_hub "${OUTPUT_DIR}/${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}/${EXP_NAME}/json_output" \
    --test_data_pkl "${TEST_DATA_PATH}" \
    --action_tokenizer_path "${ACTION_TOKENIZER_PATH}" \
    --token_yaml "${TOKEN_YAML}" \
    --norm_stats_path "${NORM_STATS_PATH}" \
    --raw_img_root "${SENSOR_BLOBS}" \
    --cur_frame_idx "${CUR_FRAME_IDX}" \
    --future_nums "${FUTURE_NUMS}" \
    --action_dim 3 \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --bf16 \
    --use_previous_actions \
    --save_gt

  echo "=== Inference done ==="
  echo "Results at: ${OUTPUT_DIR}/${EXP_NAME}/json_output"
else
  echo ""
  echo "Skipping inference (--skip_inference)."
fi
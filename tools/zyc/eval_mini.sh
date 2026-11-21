#!/bin/bash
# 单个 ckpt 走完整评测：推理 -> PDMS。mini 口径 21 场景，用来验链路通不通。
#   用法: bash eval_mini.sh <ckpt目录> <标签> [卡数]
# 图像在 sensor_blobs/test，pkl 里存的是 .npy 相对路径，
# 推理侧按 raw_img_root/<log>/CAM_F0/<同名>.jpg 取原图。
set -u
CKPT=$1; TAG=$2; NP=${3:-2}
D=/mnt/sfs_turbo/zyc/DriveVLA-W0
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
OUT=$W/navsim_workspace/eval_$TAG
LOG_NAME=$(cat $W/navsim_workspace/mini_eval/log_name.txt)
mkdir -p $OUT/json_output

cd $D || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$D:$D/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800

echo "=== [1/2] 推理 $TAG（$NP 卡，21 场景）==="
time torchrun --nproc_per_node=$NP --master_port=25301 \
  inference/qwen/inference_action_navsim_vava.py \
  --qwen_hub "$CKPT" \
  --output_dir "$OUT/json_output" \
  --test_data_pkl "$W/navsim_workspace/mini_eval/mini.pkl" \
  --action_tokenizer_path configs/fast \
  --token_yaml "$W/navsim_workspace/mini_eval/mini_tokens.yaml" \
  --norm_stats_path configs/normalizer_navsim_trainval/norm_stats.json \
  --cur_frame_idx 3 --future_nums 8 --action_dim 3 --model_max_length 1400 \
  --image_cam_subdir CAM_F0 \
  --raw_img_root /mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/test \
  --use_previous_actions --bf16 --save_gt
echo "推理产出 $(ls $OUT/json_output | wc -l) 个文件"

echo "=== [2/2] PDMS $TAG ==="
source $D/setenv_zyc.sh
CONDA=/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python
time $CONDA $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  train_test_split.scene_filter.log_names="[$LOG_NAME]" \
  agent=emu_vla_agent \
  agent.experiment_path="$OUT/json_output" \
  experiment_name="eval_$TAG" \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache"
echo "EVALDONE_$TAG"

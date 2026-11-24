#!/bin/bash
# 单个 ckpt 走完整评测：推理 -> PDMS。
#   用法: bash eval_ckpt.sh <ckpt目录> <标签> <mini|full> [卡数]
#
# mini = 单个 log 的 21 场景，约 1.5 分钟，用来确认链路通不通（已实测走通）
# full = navtest 全量 12146 场景，是真正用于对比的指标
#
# 图像在 sensor_blobs/test；pkl 里存的是 .npy 相对路径，
# 推理侧按 raw_img_root/<log>/CAM_F0/<同名>.jpg 取原图。
# PDMS 用 conda 的 navsim 环境（py3.9），推理用训练那套 venv-hif8。
set -u

# 跳过开关：存在 .skip_eval 就直接退出。用来让正在跑的 chain 脚本
# 训练完就放卡、不接评测——不用改那些正在被 bash 边读边执行的脚本。
# 删掉这个文件即可恢复评测。
if [ -f /mnt/sfs_turbo/zyc/drivevla-w0-zyc/.skip_eval ]; then
  echo "[eval_ckpt] 检测到 .skip_eval，跳过评测"
  exit 0
fi
CKPT=$1; TAG=$2; MODE=${3:-mini}; NP=${4:-8}
D=/mnt/sfs_turbo/zyc/DriveVLA-W0
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
OUT=$W/navsim_workspace/eval_${TAG}_${MODE}

# Full evals share navsim_workspace/exp and its metric cache across nodes.
# Serialize them so concurrent chains cannot contaminate each other's PDMS.
# flock is not propagated across the shared filesystem, so use atomic mkdir.
if [ "$MODE" = full ]; then
  LOCKDIR=$W/navsim_workspace/.eval_full.lockdir
  mkdir -p "$(dirname "$LOCKDIR")"
  while ! mkdir "$LOCKDIR" 2>/dev/null; do
    echo "[eval_ckpt] waiting for full-eval lock: $LOCKDIR"
    sleep 10
  done
  printf 'host=%s pid=%s started=%s\n' \
    "$(uname -n)" "$$" "$(date '+%F %T %Z')" > "$LOCKDIR/owner"
  trap 'rm -f "$LOCKDIR/owner"; rmdir "$LOCKDIR" 2>/dev/null' EXIT
  echo "[eval_ckpt] acquired full-eval lock: $(cat "$LOCKDIR/owner")"
fi

# 清空再建：PDMS 会读 experiment_path 下所有 json，残file（上一次被 kill 的评测）
# 会让它读到半截或重复的文件并报 JSONDecodeError。推理会重新产出全部场景，清空无损失。
rm -rf "$OUT/json_output"
mkdir -p "$OUT/json_output"

if [ "$MODE" = "mini" ]; then
  PKL=$W/navsim_workspace/mini_eval/mini.pkl
  YAML=$W/navsim_workspace/mini_eval/mini_tokens.yaml
  LOG_NAME=$(cat $W/navsim_workspace/mini_eval/log_name.txt)
  FILTER="train_test_split.scene_filter.log_names=[$LOG_NAME]"
else
  PKL=$W/navsim_workspace/navtest_qwen_2va.pkl
  YAML=$W/navsim_workspace/navtest_tokens.yaml
  FILTER=""
fi

cd "$D" || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$D:$D/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800

echo "=== [1/2] 推理 $TAG ($MODE, $NP 卡) ==="
time torchrun --nproc_per_node="$NP" --master_port=$((25300 + RANDOM % 80)) \
  inference/qwen/inference_action_navsim_vava.py \
  --qwen_hub "$CKPT" \
  --output_dir "$OUT/json_output" \
  --test_data_pkl "$PKL" \
  --action_tokenizer_path configs/fast \
  --token_yaml "$YAML" \
  --norm_stats_path configs/normalizer_navsim_trainval/norm_stats.json \
  --cur_frame_idx 3 --future_nums 8 --action_dim 3 --model_max_length 1400 \
  --image_cam_subdir CAM_F0 \
  --raw_img_root /mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/test \
  --use_previous_actions --bf16 --save_gt
echo "推理产出 $(ls "$OUT/json_output" | wc -l) 个文件"

echo "=== [2/2] PDMS $TAG ($MODE) ==="
source "$D/setenv_zyc.sh"
CONDA=/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python
# 注：日志里会出现 ray 的 "No module named 'acl'" —— 那是它探测 NPU 类型的
# 良性告警，已被捕获，不影响评分。别把它当失败。
time $CONDA "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
  train_test_split=navtest \
  $FILTER \
  agent=emu_vla_agent \
  agent.experiment_path="$OUT/json_output" \
  experiment_name="eval_${TAG}_${MODE}" \
  metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache"
echo "EVALDONE_${TAG}_${MODE}"

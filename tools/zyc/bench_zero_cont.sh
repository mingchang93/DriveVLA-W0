#!/bin/bash
# 续跑：z1a（100 步，已完成 191.64s）之后的三把。
#   z1b  20 步  —— 只为复验「关掉 ASCEND_LAUNCH_BLOCKING 后确定性仍成立」，不参与计时
#   z0   100 步 —— 计时
#   z2   100 步 —— 计时
#
# 计时口径：冷启动跑满 100 步，取 HF 的 train_runtime / 100，三档同口径。
# 显存：训练中每 15 秒采一次 npu-smi，取峰值。
set -u
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs

sample_hbm() {
  while true; do
    npu-smi info 2>/dev/null | awk '/^\| [0-9]+ +[0-9]+ +\/ +[0-9]+/ {print}' >> "$1" 2>/dev/null
    npu-smi info 2>/dev/null | awk '/Ascend/ {print}' >> "$1" 2>/dev/null
    sleep 15
  done
}

one() {  # tag ds steps
  local tag=$1 ds=$2 steps=$3
  local out="$R/bench_$tag"
  rm -rf "$out"
  echo "[$(date '+%H:%M:%S')] >>> $tag ds=$ds steps=$steps"
  rm -f "/tmp/hbm_$tag.txt"
  sample_hbm "/tmp/hbm_$tag.txt" &
  local sp=$!
  DS="$ds" BS=6 GA=1 STEPS="$steps" LOGSTEP=10 NPROC=8 \
    PORT=$((23890 + RANDOM % 40)) HCCL_RANGE=20700-20799 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/bench_$tag.log" 2>&1
  kill $sp 2>/dev/null
  echo "[$(date '+%H:%M:%S')] <<< $tag $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" | tail -1)"
  sleep 20
}

one z1b scripts/sft/zero1.json 20
one z0  scripts/sft/zero0.json 100
one z2  scripts/sft/zero2.json 100
echo "[$(date '+%H:%M:%S')] BENCH2DONE"

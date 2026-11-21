#!/bin/bash
# 100 步速度基准：mbs6 八卡，ZeRO-0 / 1 / 2 哪个快。
#
# 口径统一：全部从冷启动跑满 100 步，取 HF 的 train_runtime / 100。
# （上次的教训：不能拿 tqdm 的全程均值去比别人的稳态值，预热会把结果带偏。）
# 同配置两把跑的速度噪声底实测约 1.7%，差距小于这个数就不算差别。
#
# ZeRO-1 跑两把（z1a / z1b）：既给计时，又顺带复验关掉
# ASCEND_LAUNCH_BLOCKING 之后确定性是否仍然成立。
#
# 显存：训练中途每 15 秒采一次 npu-smi，取各卡峰值。
set -u
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs

sample_hbm() {   # $1=输出文件；后台跑，训练结束后被杀
  while true; do
    npu-smi info 2>/dev/null \
      | grep -oE '\| [0-9]+ +[0-9]+ +/ +[0-9]+' >> "$1" 2>/dev/null
    npu-smi info 2>/dev/null | awk '/^\| [0-9]+ +Ascend/ {print $0}' >> "$1" 2>/dev/null
    sleep 15
  done
}

one() {  # $1=tag  $2=deepspeed配置
  local tag=$1 ds=$2
  local out="$R/bench_$tag"
  echo "[$(date '+%H:%M:%S')] >>> $tag  ds=$ds"
  rm -f "/tmp/hbm_$tag.txt"
  sample_hbm "/tmp/hbm_$tag.txt" &
  local sp=$!
  DS="$ds" BS=6 GA=1 STEPS=100 LOGSTEP=10 NPROC=8 \
    PORT=$((23850 + RANDOM % 40)) HCCL_RANGE=20600-20699 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/bench_$tag.log" 2>&1
  kill $sp 2>/dev/null
  echo "[$(date '+%H:%M:%S')] <<< $tag  $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" | tail -1)"
  sleep 20
}

one z1a scripts/sft/zero1.json
one z1b scripts/sft/zero1.json
one z0  scripts/sft/zero0.json
one z2  scripts/sft/zero2.json
echo "[$(date '+%H:%M:%S')] BENCHDONE"

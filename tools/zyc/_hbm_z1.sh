#!/bin/bash
# 测 ZeRO-0 / ZeRO-2 在 8 卡 mbs6 下的 HBM 峰值。
# npu-smi 的 HBM 在每张卡块的第二行，形如 "  4787  / 98304"，单卡 96 GB。
# 必须在训练进行中采样：跑完之后只剩 rank0 没释放，采到的是假数。
set -u
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs

sample() {   # $1=输出文件
  while true; do
    npu-smi info 2>/dev/null \
      | grep -oE '[0-9]+ +/ +98304' \
      | grep -oE '^[0-9]+' \
      | paste -sd' ' >> "$1"
    sleep 10
  done
}

one() {  # tag ds
  local tag=$1 ds=$2
  echo "[$(date '+%H:%M:%S')] >>> $tag"
  rm -f "/tmp/hbmz1_$tag.txt"
  sample "/tmp/hbmz1_$tag.txt" &
  local sp=$!
  DS="$ds" BS=6 GA=1 STEPS=30 LOGSTEP=10 NPROC=8 \
    PORT=$((23940 + RANDOM % 30)) HCCL_RANGE=20800-20899 OUT="$R/hbm_$tag" \
    bash "$T/run_sft_navsim.sh" > "/tmp/hbmz1_$tag.log" 2>&1
  kill $sp 2>/dev/null
  echo "[$(date '+%H:%M:%S')] <<< $tag"
  sleep 15
}

one z1 scripts/sft/zero1.json
echo "[$(date '+%H:%M:%S')] HBMDONE"

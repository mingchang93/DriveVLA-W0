#!/bin/bash
# 真实训练配置（8 卡 ZeRO-1 bs6）下复验逐位一致。8 卡会占满整机，所以串行。
set -u
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
for tag in a b; do
  p=$((23800 + RANDOM % 50))
  echo "[$(date "+%H:%M:%S")] >>> det21_$tag"
  DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=12 LOGSTEP=1 NPROC=8 \
    PORT=$p HCCL_RANGE=20400-20499 OUT="$R/det21_$tag" \
    bash $T/run_sft_navsim.sh > /tmp/det21_$tag.log 2>&1
  echo "[$(date "+%H:%M:%S")] <<< det21_$tag"
  sleep 20
done
echo "[$(date "+%H:%M:%S")] DET20DONE"

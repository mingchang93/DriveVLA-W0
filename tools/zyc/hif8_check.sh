#!/bin/bash
# HiF8 臂的两项验收：
#   1. 自身逐位可复现（两把单卡 12 步，并行在卡 0/1）
#      —— 对比口径要求两条臂各自都确定，否则偏差里混着噪声
#   2. 8 卡 100 步吞吐（与 bf16 的 191.64s 同口径：冷启动、LOGSTEP=1、ZeRO-1、bs6）
set -u
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs

# --- 1. 确定性对，两张卡并行 ---
for x in a:0:23990:21100-21119 b:1:23992:21120-21139; do
  IFS=: read tag card port hccl <<< "$x"
  ( HIF8=cts ASCEND_RT_VISIBLE_DEVICES=$card DS=scripts/sft/zero1.json \
      BS=6 GA=1 STEPS=12 LOGSTEP=1 NPROC=1 PORT=$port HCCL_RANGE=$hccl \
      OUT="$R/hif8_det_$tag" bash "$T/run_sft_navsim.sh" \
      > "/tmp/hif8_det_$tag.log" 2>&1 ) &
done
wait
echo "[$(date '+%H:%M:%S')] 确定性对跑完"

# --- 2. 8 卡 100 步计时 ---
HIF8=cts DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=100 LOGSTEP=1 NPROC=8 \
  PORT=23996 HCCL_RANGE=21200-21299 OUT="$R/hif8_speed100" \
  bash "$T/run_sft_navsim.sh" > /tmp/hif8_speed100.log 2>&1
echo "[$(date '+%H:%M:%S')] 8卡100步 $(grep -oE "'train_runtime': [0-9.]+" "$R/hif8_speed100/train.log" | tail -1)"
echo "[$(date '+%H:%M:%S')] HIF8CHECKDONE"

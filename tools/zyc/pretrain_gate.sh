#!/bin/bash
# 预训练自动等卡启动门：卡一释放就立刻起，不用人盯。
#   用法: ARM=bf16|hif8 bash pretrain_gate.sh
#
# 同事 14:30 释放机器，所以这里只"等"不"杀"——轮询到 8 卡全空才启动。
# 为什么不预约时刻：上一次按时刻挂起的任务链在 5 小时睡眠窗口里不明原因死掉了，
# 改成轮询条件更稳，而且卡提前空出来就能提前开始。
#
# 数据用现成的 nuPlan pickle（5385 段）。8000 步在这份数据上约 138 个 epoch，
# 会过拟合，但那只影响绝对 PDMS，不影响 bf16 vs HiF8 的对比——两条臂看到
# 完全相同的数据、相同的重放次数。全量 pickle 要改 step1~4 里写死的
# /mnt/vdb1 路径，时间不可控，不挡在这条路径上。
#
# 启动时带 stdout 重定向：脚本内部用 `| tee -a train.log` 写日志，
# 若 stdout 是失效 fd（docker exec -d + setsid 的情况），tee 会退出、
# 连文件那份都写不成。今天 eval_all / run_one / extract_v2 的日志
# 全部只剩一两行就是这个原因。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
ARM=${ARM:?必须指定 ARM=bf16 或 hif8}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/pretrain_gate_${ARM}.log
say() { echo "[$(date +'%F %T')] [$ARM] $*" >> "$LOG"; }

case "$ARM" in
  bf16) HIF8=""    ; TAG=pretrain_nuplan_bf16 ;;
  hif8) HIF8=cts   ; TAG=pretrain_nuplan_hif8_168 ;;
  *) echo "未知 ARM=$ARM"; exit 1 ;;
esac

say "等卡（需要 8 张全空），每 60 秒查一次"
waited=0
while true; do
  free=$(npu-smi info 2>/dev/null | grep -c 'No running processes')
  if [ "${free:-0}" -ge 8 ]; then say "8 卡就绪（等了 $waited 分钟）"; break; fi
  [ $((waited % 15)) -eq 0 ] && say "  空闲 ${free:-?}/8，继续等（已等 $waited 分钟）"
  sleep 60; waited=$((waited+1))
  if [ "$waited" -gt 600 ]; then say "!! 等了 10 小时仍无卡，放弃"; exit 1; fi
done

OUT=$R/$TAG
say ">>> 启动 8000 步预训练  hif8='${HIF8:-off}'  输出 $OUT"
HIF8="$HIF8" STEPS=8000 BS=12 GA=1 NPROC=8 LOGSTEP=1 \
  PORT=$((24600 + RANDOM % 40)) OUT="$OUT" \
  bash "$T/run_stage1_pretrain.sh" > "/tmp/pretrain_${ARM}.log" 2>&1
rc=$?
say "<<< 训练结束 rc=$rc $(grep -oE "'train_runtime': [0-9.]+" "$OUT/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$OUT"/*.safetensors 2>/dev/null | wc -l)"
say "PRETRAINDONE_${ARM}"

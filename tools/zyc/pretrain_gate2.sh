#!/bin/bash
# 预训练自动等卡启动门（抢锁版）：四台都挂，哪两台先空出卡就哪两台跑。
#
# 为什么用抢锁：不知道同事先释放哪几台。四台都挂门，各自等自己的卡空出来，
# 然后用 mkdir（原子操作）抢一个臂——先到先得，拿到 bf16 的跑 bf16，
# 拿到 hif8 的跑 hif8，两个臂都被领走后其余机器的门自行退出。
# 这样无论哪两台先释放，预训练都能第一时间跑上，不用人盯。
#
# 只等不杀：同事 14:30 释放，不需要抢占。
# 不用预约时刻：上次按时刻挂起的任务链在 5 小时睡眠窗口里不明原因死掉，
# 轮询条件更稳，而且卡提前空出来就能提前开始。
#
# 数据用现成的 nuPlan pickle（5385 段）。8000 步约 138 个 epoch，会过拟合，
# 但那只影响绝对 PDMS，不影响 bf16 vs HiF8 的对比——两条臂数据与重放次数相同。
#
# 日志用 >> 直接追加，不经过 tee：docker exec -d + setsid 启动后 stdout 的 fd
# 失效，tee 写 stdout 失败就退出，连文件那份也写不成（今天多个日志只剩
# 一两行就是这个原因）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
CLAIM=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/.pretrain_claim
H=$(cat /proc/sys/kernel/hostname 2>/dev/null || echo unknown)
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/pretrain_gate.log
say() { echo "[$(date +'%F %T')] [$H] $*" >> "$LOG"; }

mkdir -p "$CLAIM"

say "等卡（需 8 张全空），每 60 秒一次"
waited=0
while true; do
  free=$(npu-smi info 2>/dev/null | grep -c 'No running processes')
  if [ "${free:-0}" -ge 8 ]; then break; fi
  [ $((waited % 15)) -eq 0 ] && say "空闲 ${free:-?}/8，已等 $waited 分钟"
  sleep 60; waited=$((waited+1))
  if [ "$waited" -gt 600 ]; then say "等了 10 小时仍无卡，放弃"; exit 1; fi
done
say "8 卡就绪（等了 $waited 分钟），开始抢臂"

ARM=""
for a in bf16 hif8; do
  if mkdir "$CLAIM/$a" 2>/dev/null; then ARM=$a; break; fi
done
if [ -z "$ARM" ]; then say "两个臂都已被别的机器领走，本机退出"; exit 0; fi
say "抢到 $ARM"

case "$ARM" in
  bf16) HIF8=""  ; TAG=pretrain_nuplan_bf16 ;;
  hif8) HIF8=cts ; TAG=pretrain_nuplan_hif8_168 ;;
esac
OUT=$R/$TAG

say ">>> 启动 8000 步预训练  hif8='${HIF8:-off}'  输出 $OUT"
HIF8="$HIF8" STEPS=8000 BS=12 GA=1 NPROC=8 LOGSTEP=1 \
  PORT=$((24600 + RANDOM % 40)) OUT="$OUT" \
  bash "$T/run_stage1_pretrain.sh" > "/tmp/pretrain_${ARM}.log" 2>&1
rc=$?
say "<<< $ARM 训练结束 rc=$rc $(grep -oE "'train_runtime': [0-9.]+" "$OUT/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$OUT"/*.safetensors 2>/dev/null | wc -l) $(grep -oE '量化 [0-9]+ 个' "$OUT/train.log" 2>/dev/null | head -1)"
say "PRETRAINDONE_${ARM}"

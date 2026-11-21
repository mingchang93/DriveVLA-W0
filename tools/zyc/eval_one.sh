#!/bin/bash
# 评一个 aligned ckpt：先 mini 确认链路，出分再跑 full。
#   用法: TAG=aligned_bf16 bash eval_one.sh
# full = navtest 全量 12146 场景，是用于对比的指标；mini 21 场景只验链路。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
TAG=${TAG:?}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/eval_${TAG}.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

ck=$R/$TAG
if [ ! -f "$ck/model.safetensors.index.json" ]; then say "没有 ckpt，退出"; exit 1; fi
for mode in mini full; do
  say ">>> $TAG $mode 开始"
  t0=$(date +%s)
  bash "$T/eval_ckpt.sh" "$ck" "$TAG" "$mode" 8 > "/tmp/ev_${TAG}_${mode}.log" 2>&1
  s=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${TAG}_${mode}.log" | tail -1 | grep -oE '[0-9.]+$')
  n=$(grep -oE 'valid results: [0-9]+' "/tmp/ev_${TAG}_${mode}.log" | tail -1)
  say "<<< $TAG $mode PDMS=${s:-失败}  $n  耗时 $(( ($(date +%s)-t0)/60 )) 分钟"
  if [ -z "$s" ]; then say "    $mode 没出分，停在这里（日志 /tmp/ev_${TAG}_${mode}.log）"; break; fi
done
say "EVALDONE_${TAG}"

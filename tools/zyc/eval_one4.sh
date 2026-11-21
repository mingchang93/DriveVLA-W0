#!/bin/bash
# 评一个 aligned ckpt，卡数可配（并行跑两把时各用 4 卡）。
#   用法: TAG=aligned_bf16 NP=4 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash eval_one4.sh
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
TAG=${TAG:?}; NP=${NP:-8}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/eval_${TAG}.log
say() { echo "[$(date +'%F %T')] [$NP卡 dev=${ASCEND_RT_VISIBLE_DEVICES:-all}] $*" >> "$LOG"; }

ck=$R/$TAG
[ -f "$ck/model.safetensors.index.json" ] || { say "没有 ckpt，退出"; exit 1; }
for mode in mini full; do
  say ">>> $TAG $mode 开始"
  t0=$(date +%s)
  bash "$T/eval_ckpt.sh" "$ck" "$TAG" "$mode" "$NP" > "/tmp/ev_${TAG}_${mode}.log" 2>&1
  s=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${TAG}_${mode}.log" | tail -1 | grep -oE '[0-9.]+$')
  n=$(grep -oE 'valid results: [0-9]+' "/tmp/ev_${TAG}_${mode}.log" | tail -1)
  say "<<< $TAG $mode PDMS=${s:-失败}  $n  耗时 $(( ($(date +%s)-t0)/60 )) 分钟"
  [ -n "$s" ] || { say "    $mode 没出分，停（日志 /tmp/ev_${TAG}_${mode}.log）"; break; }
done
say "EVALDONE_${TAG}"

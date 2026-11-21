#!/bin/bash
# 接棒跑 2B 的 DTS 实验。
#   用法: WAIT_RUN=w7b_hif8_196 TAG=aligned_hif8_196_dts IGN=lm_head bash queue_dts.sh
#
# 两道闸：
#   1) 本机那把 7B 跑完（看 train.log 的 TRAIN_EXIT）
#   2) a54 上的 DTS 20 步冒烟已通过（看 runs/dts_smoke/train.log 的 TRAIN_EXIT=0）
# 第二道的理由：DTS 是今天新写的，没在真模型上跑过。冒烟没过就不起，
# 日志写清楚，别赔掉一整把 4000 步。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
WAIT_RUN=${WAIT_RUN:?}
TAG=${TAG:?}
IGN=${IGN:-lm_head,o_proj}
INTERVAL=${INTERVAL:-10}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/queue_${TAG}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "===== 接棒器：等 $WAIT_RUN 跑完 + DTS 冒烟通过，再起 $TAG（interval=$INTERVAL）====="

tick=0
while true; do
  ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/$WAIT_RUN/train.log" 2>/dev/null | tail -1)
  [ -n "$ex" ] && break
  tick=$((tick+1))
  [ $((tick % 20)) -eq 1 ] && say "等本机：$WAIT_RUN $(tail -1 "$R/$WAIT_RUN/train.log" 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+/4000' | tail -1)"
  sleep 60
done
say ">>> $WAIT_RUN 已结束（$ex）"

# 第二道闸：DTS 冒烟
tick=0
while true; do
  sm=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/dts_smoke/train.log" 2>/dev/null | tail -1)
  if [ -n "$sm" ]; then
    if [ "$sm" = "TRAIN_EXIT=0" ]; then say ">>> DTS 冒烟已通过，继续"; break; fi
    say "!!! DTS 冒烟结果是 $sm —— 不起 $TAG，卡让出来。排查看 /tmp/one_dts_smoke.log"
    say "QUEUEABORT_$TAG"
    exit 1
  fi
  tick=$((tick+1))
  [ $((tick % 10)) -eq 1 ] && say "等 a54 的 DTS 冒烟出结果…"
  sleep 60
  if [ "$tick" -gt 180 ]; then say "!!! 等了 3 小时冒烟还没结果，放弃"; say "QUEUEABORT_$TAG"; exit 1; fi
done

for i in $(seq 1 30); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done
say "    卡上进程 $(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')，起 $TAG"

docker exec -d zyc-w0 env TAG="$TAG" HIF8=dts IGN="$IGN" HIF8_INTERVAL="$INTERVAL" \
  MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  setsid bash "$T/run_one.sh"
sleep 150
say "    启动后训练进程 $(docker exec zyc-w0 bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"')  日志：$(tail -1 "$W/one_$TAG.log" 2>/dev/null)"
say "QUEUEDONE_$TAG"

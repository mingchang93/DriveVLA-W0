#!/bin/bash
# 等前一把训练写出 TRAIN_EXIT，再起下一把。
#   用法: WAIT_RUN=w7b_bf16 TAG=aligned_hif8_132 HIF8=cts IGN=... MODEL=... bash queue_next.sh
#
# 为什么要等而不是直接排在链里：原来的夜链是串行的，现在把其中几棒拆到
# 别的机器并行跑，链已经被摘掉了，需要一个轻量的接棒器。
# 判据用 train.log 里的 TRAIN_EXIT 行——它是 run_sft_navsim.sh 收尾时写的，
# 比看进程可靠（进程可能还在收尾存 ckpt）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
WAIT_RUN=${WAIT_RUN:?}
TAG=${TAG:?}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/queue_${TAG}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "===== 接棒器：等 $WAIT_RUN 跑完，再起 $TAG ====="
tick=0
while true; do
  ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/$WAIT_RUN/train.log" 2>/dev/null | tail -1)
  [ -n "$ex" ] && break
  tick=$((tick+1))
  [ $((tick % 20)) -eq 1 ] && say "等待中：$WAIT_RUN $(tail -1 "$R/$WAIT_RUN/train.log" 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+/4000' | tail -1)"
  sleep 60
done
say ">>> $WAIT_RUN 已结束（$ex），等卡释放"

# 等卡真的空出来：上一把可能还在存 ckpt
for i in $(seq 1 30); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done
say "    卡上进程 $(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')，起 $TAG"

docker exec -d zyc-w0 env TAG="$TAG" HIF8="${HIF8:-}" IGN="${IGN:-lm_head,o_proj}" \
  MODEL="${MODEL:-/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct}" \
  setsid bash "$T/run_one.sh"
sleep 120
say "    启动后训练进程 $(docker exec zyc-w0 bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"')  日志：$(tail -1 "$W/one_$TAG.log" 2>/dev/null)"
say "QUEUEDONE_$TAG"

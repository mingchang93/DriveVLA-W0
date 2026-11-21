#!/bin/bash
# 等本机那把训练跑完，再用 8 卡评它一把（只评一个 tag，不成对）。
#   用法: WAIT_TAG=aligned_hif8_132 bash eval_one_q.sh
#
# 为什么要单 tag 版：原来的成对接棒器会让两台机器评到同一个 ckpt。
#   a51 已经在评 aligned_hif8_168 -> a54 就不该再评它
#   a52 接的是 w7b_bf16           -> a53 就不该再评它
# 重复评一遍要多花 20 多分钟的卡，而且两份结果写同一个 experiment 目录会打架。
#
# 结束判据：train.log 里有 TRAIN_EXIT 且容器里没有训练进程。
# 只看日志行会被骗：同一个输出目录被两次运行共用时，先前那把的收尾行会追加进
# 新日志里（a54 的 w7b_hif8_132 就这样误判过一次）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
R=$W/runs
WAIT_TAG=${WAIT_TAG:?}
NP=${NP:-8}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/evalq1_${WAIT_TAG}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

ntrain() { docker exec zyc-w0 bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"' 2>/dev/null | tr -d ' \r'; }

say "===== 单把评测接棒器：等 $WAIT_TAG 跑完，再 $NP 卡评它 ====="
tick=0
while true; do
  ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/$WAIT_TAG/train.log" 2>/dev/null | tail -1)
  if [ -n "$ex" ] && [ "$(ntrain)" = "0" ]; then
    say "   $WAIT_TAG 结束（$ex，且无训练进程）"
    break
  fi
  tick=$((tick+1))
  if [ $((tick % 20)) -eq 1 ]; then
    say "   等 $WAIT_TAG $(tr -d '\r' < "$R/$WAIT_TAG/train.log" 2>/dev/null | grep -oE '[0-9]+/4000' | tail -1)  训练进程 $(ntrain)"
  fi
  sleep 60
  [ "$tick" -gt 600 ] && { say "!! 等 10 小时未完，放弃"; exit 1; }
done

say ">>> 等卡释放"
for i in $(seq 1 45); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

[ -f "$R/$WAIT_TAG/model.safetensors.index.json" ] || { say "!! $WAIT_TAG 没有 ckpt，不评"; exit 1; }
say ">>> 起 $WAIT_TAG 评测（$NP 卡）"
docker exec -d zyc-w0 env TAG="$WAIT_TAG" NP="$NP" setsid bash "$T/eval_one4.sh"
sleep 90
say "    $(tail -1 "$W/eval_$WAIT_TAG.log" 2>/dev/null)"
say "EVALQ1DONE_$WAIT_TAG"

#!/bin/bash
# 等本机当前评测结束，再评下一个。
#   用法: WAIT_TAG=w7b_hif8_196 NEXT_TAG=aligned_hif8_168 bash chain_eval.sh
#
# 为什么要接：a53/a54 还在训练，等它们腾卡才评就白等两小时。
# a51/a52 已经空出来，把剩下没评的两把接在它们当前这把后面。
# 判据用 eval_<tag>.log 里的 EVALDONE_ 标记（eval_one4.sh 收尾时写）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
R=$W/runs
WAIT_TAG=${WAIT_TAG:?}
NEXT_TAG=${NEXT_TAG:?}
NP=${NP:-8}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/chaineval_${NEXT_TAG}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "===== 等 $WAIT_TAG 评完，再评 $NEXT_TAG（$NP 卡）====="
tick=0
while ! grep -q "EVALDONE_$WAIT_TAG" "$W/eval_$WAIT_TAG.log" 2>/dev/null; do
  tick=$((tick+1))
  [ $((tick % 20)) -eq 1 ] && say "等待中：$(tail -1 "$W/eval_$WAIT_TAG.log" 2>/dev/null | cut -c1-110)"
  sleep 60
  [ "$tick" -gt 300 ] && { say "!! 等 5 小时未完，放弃"; exit 1; }
done
say ">>> $WAIT_TAG 评完，等卡释放"
for i in $(seq 1 45); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

[ -f "$R/$NEXT_TAG/model.safetensors.index.json" ] || { say "!! $NEXT_TAG 没有 ckpt"; exit 1; }
say ">>> 起 $NEXT_TAG 评测"
docker exec -d zyc-w0 env TAG="$NEXT_TAG" NP="$NP" setsid bash "$T/eval_one4.sh"
sleep 90
say "    $(tail -1 "$W/eval_$NEXT_TAG.log" 2>/dev/null)"
say "CHAINEVALDONE_$NEXT_TAG"

#!/bin/bash
# a54 串行链：7B 冒烟（bf16 + HiF8-168，ZeRO-2，50 步）→ 两把都过再并行起 lr1e-5 评测。
#
# 评测为什么能并行：8 卡劈成 4+4（ASCEND_RT_VISIBLE_DEVICES），
# eval_ckpt.sh 的 master_port 本身是随机的，输出目录与 experiment_name 按 tag 区分，
# 两把互不干扰。推理是数据并行切场景，4 卡只是慢一倍，但两把并行总时长不变。
#
# 严格门控：冒烟任一把没有 TRAIN_EXIT=0 就不进评测——避免在流程还没跑通时
# 白占一小时的卡。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/chain7b.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

say "===== 链开始：7B 冒烟 -> lr1e-5 评测 ====="
bash "$T/smoke7b.sh"

ok=1
for tag in bf16 hif8_168; do
  ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/smoke7b_$tag/train.log" 2>/dev/null | tail -1)
  say "  7B $tag -> ${ex:-无}"
  [ "$ex" = "TRAIN_EXIT=0" ] || ok=0
done

if [ "$ok" != 1 ]; then
  say "!!! 7B 冒烟没全过，按门控不进评测。卡已释放。"
  say "CHAIN7BDONE"
  exit 1
fi

say ">>> 两把 7B 都跑通，并行起 lr1e-5 评测（4+4 卡）"
( ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TAG=aligned_bf16      NP=4 bash "$T/eval_one4.sh" ) &
p1=$!
( ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 TAG=aligned_hif8_196  NP=4 bash "$T/eval_one4.sh" ) &
p2=$!
wait $p1 $p2
say "----- 评测结果 -----"
for tag in aligned_bf16 aligned_hif8_196; do
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  say "  $tag  mini=${m:-无}  full=${f:-无}"
done
say "CHAIN7BDONE"

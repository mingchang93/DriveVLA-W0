#!/bin/bash
# 并行评两把 lr1e-5 的 ckpt：8 卡劈成 4+4。
# 上一次失败是 json_output 里混了被 kill 的评测残file，eval_ckpt.sh 现在推理前会清空。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/eval_pair.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

say "===== 并行评测 lr1e-5 两把（4+4 卡）====="
( ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TAG=aligned_bf16     NP=4 bash "$T/eval_one4.sh" ) & p1=$!
( ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 TAG=aligned_hif8_196 NP=4 bash "$T/eval_one4.sh" ) & p2=$!
wait $p1 $p2
say "----- 结果 -----"
for tag in aligned_bf16 aligned_hif8_196; do
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  nf=$(grep -oE 'valid results: [0-9]+' "/tmp/ev_${tag}_full.log" 2>/dev/null | tail -1)
  say "  $tag  mini=${m:-无}  full=${f:-无}  $nf"
done
say "EVALPAIRDONE"

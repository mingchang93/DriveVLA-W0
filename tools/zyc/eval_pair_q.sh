#!/bin/bash
# 等本机两把训练都跑完，再 4+4 卡并行评测。
#   用法: TAG1=w7b_hif8_196 TAG2=aligned_hif8_196_dts bash eval_pair_q.sh
#
# 并行而不是串行：8 卡劈成 4+4，两把同时跑，总时长≈一把的时长。
# eval_ckpt.sh 的 master_port 本身随机，输出目录与 experiment_name 按 tag 区分，互不干扰。
# 每把都是先 mini（21 场景，约 1.5 分钟）出分才接 full（12146 场景）——
# 7B 的评测我们没跑过，真有问题会在 mini 阶段暴露，不会赔掉两小时。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
TAG1=${TAG1:?}; TAG2=${TAG2:?}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/evalq_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

wait_run() {
  local t=$1 tick=0
  while true; do
    ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/$t/train.log" 2>/dev/null | tail -1)
    [ -n "$ex" ] && { say "   $t 结束（$ex）"; return 0; }
    tick=$((tick+1))
    [ $((tick % 30)) -eq 1 ] && say "   等 $t $(tail -1 "$R/$t/train.log" 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+/4000' | tail -1)"
    sleep 60
    [ "$tick" -gt 600 ] && { say "!! 等 $t 超过 10 小时，放弃"; return 1; }
  done
}

say "===== 评测接棒器：等 $TAG1 与 $TAG2 都跑完 ====="
wait_run "$TAG1" || { say "EVALQABORT"; exit 1; }
wait_run "$TAG2" || { say "EVALQABORT"; exit 1; }

say ">>> 两把都完成，等卡释放"
for i in $(seq 1 45); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

ok=0
for t in "$TAG1" "$TAG2"; do
  [ -f "$R/$t/model.safetensors.index.json" ] && ok=$((ok+1)) || say "!! $t 没有 ckpt，不评"
done
[ "$ok" = 0 ] && { say "两把都没 ckpt"; say "EVALQABORT"; exit 1; }

say ">>> 并行评测（4+4 卡）：$TAG1 / $TAG2"
( ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TAG="$TAG1" NP=4 bash "$T/eval_one4.sh" ) & p1=$!
( ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 TAG="$TAG2" NP=4 bash "$T/eval_one4.sh" ) & p2=$!
wait $p1 $p2

say "----- 结果 -----"
for t in "$TAG1" "$TAG2"; do
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${t}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${t}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  ns=$(grep -oE 'Number of successful scenarios: [0-9]+' "/tmp/ev_${t}_full.log" 2>/dev/null | tail -1)
  say "  $t  mini=${m:-无}  full=${f:-无}  $ns"
done
say "EVALQDONE_${MACH}"

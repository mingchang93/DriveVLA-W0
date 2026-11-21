#!/bin/bash
# 四个 ckpt 的评测全部放在本机：每个占 2 张卡，四个并行用满 8 卡。
#
# 为什么这样排：评测里只有推理用卡，PDMS 评分是纯 CPU 的。实测口径
# （mini 21 场景 2 卡 36.7 秒，其中约 25 秒是模型加载）反推全量 12146 场景：
#   2 卡约 115 分钟 / 8 卡约 29 分钟
# 四个各用 2 卡并行，总耗时约 115 分钟，但立刻腾出另外三台机器；
# 若四台各用 8 卡串行，虽然单个只要 29 分钟，却要占满四台。
#
# 分两阶段：先四个 mini 并行（约 2 分钟）确认 ckpt 能加载、链路没坏，
# 出分了再跑 full。mini 不过就不跑 full，省得白占两小时。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/eval_all.log
say() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }

TAGS="bf16 hif8_168 hif8_196 hif8_edge3"
declare -A DEV=( [bf16]="0,1" [hif8_168]="2,3" [hif8_196]="4,5" [hif8_edge3]="6,7" )

say "===== 阶段一：四个 mini 并行（各 2 卡）====="
for t in $TAGS; do
  ck=$R/night_$t
  if [ ! -f "$ck/model.safetensors.index.json" ]; then say "  $t 没有 ckpt，跳过"; continue; fi
  ( ASCEND_RT_VISIBLE_DEVICES=${DEV[$t]} \
      bash "$T/eval_ckpt.sh" "$ck" "all_$t" mini 2 > "/tmp/evalall_${t}_mini.log" 2>&1 ) &
  say "  起 $t mini  卡 ${DEV[$t]}"
  sleep 8     # 错开启动，避免 torchrun 随机端口撞车
done
wait
say "----- mini 结果 -----"
ok=""
for t in $TAGS; do
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/evalall_${t}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  say "  $t mini PDMS=${m:-失败}"
  [ -n "$m" ] && ok="$ok $t"
done

if [ -z "$ok" ]; then say "mini 全部失败，不跑 full"; say "EVALALLDONE"; exit 1; fi

say "===== 阶段二：full 并行（各 2 卡，约 115 分钟）====="
for t in $ok; do
  ck=$R/night_$t
  ( ASCEND_RT_VISIBLE_DEVICES=${DEV[$t]} \
      bash "$T/eval_ckpt.sh" "$ck" "all_$t" full 2 > "/tmp/evalall_${t}_full.log" 2>&1 ) &
  say "  起 $t full  卡 ${DEV[$t]}"
  sleep 10
done
wait

say "===== 汇总 ====="
for t in $TAGS; do
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/evalall_${t}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/evalall_${t}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  n=$(grep -oE 'Number of successful scenarios: [0-9]+' "/tmp/evalall_${t}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  say "  $(printf %-11s $t)  mini=${m:-?}   full=${f:-?}   full场景数=${n:-?}"
done
say "EVALALLDONE"

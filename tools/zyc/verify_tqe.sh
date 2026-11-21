#!/bin/bash
# 验 TASK_QUEUE_ENABLE=1 是否还能保持逐位一致，并顺带量它的吞吐代价。
#
# 背景：profiling 显示单步 2204 ms 里 889 ms 散在约 15,650 个亚毫秒空隙里，
# 是算子下发跟不上的特征。TASK_QUEUE_ENABLE=0 关掉了异步任务队列，
# 主机必须同步逐个下发；24,000 kernel/步 × 几十微秒 ≈ 889 ms，与实测吻合。
# 这个开关当初和另外八个确定性变量一起加的，从没逐个验过。而确定性的真正根因
# 后来查明是播种顺序，不是环境变量 —— 所以它可能是白付的 40%。
#
# 五把 50 步（同机、同配置、背靠背，排除机器与时段差异）：
#   tqe1_a / tqe1_b           bf16,  TASK_QUEUE_ENABLE=1   -> 比对逐位一致
#   tqe0_a                    bf16,  TASK_QUEUE_ENABLE=0   -> 吞吐参照
#   tqe1_h196_a / _b          HiF8-196, TQE=1              -> 量化路径也要验
#
# 判据用 grad_norm 与我们自己打的 action_loss/ross_loss，不用 HF 四舍五入到
# 4 位的 loss —— 那把尺子只有 1e-4 分辨率，会把真实分歧盖掉。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/verify_tqe.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

one() {   # tag  TQE  HIF8  IGN
  local tag=$1 tqe=$2 hif8=$3 ign=$4
  local out=$R/$tag
  rm -rf "$out"
  say ">>> $tag  TASK_QUEUE_ENABLE=$tqe  hif8=${hif8:-off}"
  docker exec zyc-w0 env \
    TAG="$tag" HIF8="$hif8" IGN="$ign" STEPS=50 \
    TASK_QUEUE_ENABLE="$tqe" \
    MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
    bash "$T/run_one.sh"
  local ex rt
  ex=$(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE 'TRAIN_EXIT=[0-9]+' | tail -1)
  rt=$(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE "'train_runtime': [0-9.]+" | tail -1)
  say "<<< $tag $ex  $rt  环境确认 $(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE 'TASK_QUEUE_ENABLE=[0-9]' | head -1)"
  sleep 15
}

say "===== 验 TASK_QUEUE_ENABLE=1 的确定性与吞吐代价（同机背靠背）====="

one tqe1_a      1 ""    lm_head,o_proj
one tqe1_b      1 ""    lm_head,o_proj
one tqe0_a      0 ""    lm_head,o_proj
one tqe1_h196_a 1 cts   lm_head
one tqe1_h196_b 1 cts   lm_head

say "----- 逐位比对 -----"
python3 "$T/cmp_det.py" tqe1_a tqe1_b tqe1_h196_a tqe1_h196_b >> "$LOG" 2>&1

say "----- 吞吐对照（50 步 train_runtime）-----"
for t in tqe1_a tqe1_b tqe0_a tqe1_h196_a tqe1_h196_b; do
  rt=$(tr -d '\r' < "$R/$t/train.log" 2>/dev/null | grep -oE "'train_runtime': [0-9.]+" | tail -1 | grep -oE '[0-9.]+')
  tq=$(tr -d '\r' < "$R/$t/train.log" 2>/dev/null | grep -oE 'TASK_QUEUE_ENABLE=[0-9]' | head -1)
  [ -n "$rt" ] && say "  $(printf '%-14s' "$t")  $tq  runtime ${rt}s  $(python3 -c "print('%.3f s/step' % (${rt}/50))")"
done
say "VERIFYTQEDONE"

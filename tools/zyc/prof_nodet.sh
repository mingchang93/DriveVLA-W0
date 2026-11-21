#!/bin/bash
# a53：采非确定性模式下的 profiling，bf16 / CTS-196 / DTS-196 三把。
#
# 和已有的确定性 profile 对照，回答两个问题：
#   1. 确定性到底付了多少吞吐？（已知的两笔：TASK_QUEUE_ENABLE=0 导致的
#      同步下发约 889 ms/步、F.interpolate 被分解后 aten::index 回退 AI_CPU
#      约 267 ms/步，合计超过单步 2204 ms 的一半）
#   2. 关掉确定性后，HiF8 的相对收益是否变得可测？确定性模式下 HiF8 相关算子
#      只占设备耗时 4.56%，净省约 36 ms —— 被一个被刻意拖慢一倍的基线淹没了。
#
# 用 run_sft_nodet.sh（run_sft_navsim.sh 的副本 + DETERMINISM 总开关）。
# 之所以用副本：a54 正在执行原文件，bash 边读边执行，改它会炸
# （上次给原文件加 SEED 时就把正在跑的四把搞成了假的 TRAIN_EXIT=1）。
#
# profiler 配置与确定性那两把完全一致（skip20/wait1/warmup1/active3、Level1、
# 只采 rank0），否则两边不可比。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/prof_nodet.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

one() {   # tag  HIF8  IGN
  local tag=$1 hif8=$2 ign=$3
  local out=$R/$tag
  local pdir=$W/$tag
  rm -rf "$out" "$pdir"
  mkdir -p "$pdir"
  say ">>> $tag  hif8=${hif8:-off}  DETERMINISM=off"
  docker exec zyc-w0 env \
    MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
    HIF8="$hif8" HIF8_IGNORE="$ign" HIF8_INTERVAL=10 \
    DETERMINISM=off \
    DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=30 LOGSTEP=1 NPROC=8 \
    PORT=$((25200 + RANDOM % 40)) HCCL_RANGE=24700-24799 OUT="$out" \
    HIF8_PROFILE=1 HIF8_PROFILE_DIR="$pdir" \
    bash "$T/run_sft_nodet.sh" > "/tmp/$tag.log" 2>&1
  local ex rt
  ex=$(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE 'TRAIN_EXIT=[0-9]+' | tail -1)
  rt=$(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE "'train_runtime': [0-9.]+" | tail -1)
  say "<<< $tag $ex  $rt"
  say "    环境: $(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE '确定性 DETERMINISM=.*' | head -1 | cut -c1-150)"
  say "    量化: $(tr -d '\r' < "$out/train.log" 2>/dev/null | grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' | head -1)"
  say "    profile: $(du -sh "$pdir" 2>/dev/null | cut -f1)"
  sleep 20
}

say "===== a53：非确定性 profiling（bf16 / CTS-196 / DTS-196，各 30 步）====="
one nodet_bf16      ""   lm_head,o_proj
one nodet_cts196    cts  lm_head
one nodet_dts196    dts  lm_head

say "----- 吞吐对照（30 步 train_runtime）-----"
for t in nodet_bf16 nodet_cts196 nodet_dts196; do
  rt=$(tr -d '\r' < "$R/$t/train.log" 2>/dev/null | grep -oE "'train_runtime': [0-9.]+" | tail -1 | grep -oE '[0-9.]+')
  [ -n "$rt" ] && say "  $(printf '%-14s' "$t")  ${rt}s  $(python3 -c "print('%.3f s/step' % (${rt}/30))")"
done
say "    对照：确定性模式下 prof_bf16 / prof_hif8_196 的 30 步 runtime 见各自 train.log"
say "PROFNODETDONE"

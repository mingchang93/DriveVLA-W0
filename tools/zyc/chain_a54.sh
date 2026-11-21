#!/bin/bash
# a54 夜链：03:00 清卡 -> hif8_168 4000 步 -> hif8_132 4000 步 -> 两把 4+4 卡并行全量评测。
#
# 为什么是这两档（都在官方超参 lr1e-5 下，与 aligned_bf16 / aligned_hif8_196 逐键可比）：
#   hif8_168  ignore=lm_head,o_proj                      客户规格，交付表上空着的那一格
#   hif8_132  再加 layers.0./1./2./25./26./27.           前后三层留 bf16（edge3）
# ZeRO 档位用 zero1，与已完成的 aligned_bf16 / aligned_hif8_196 一致，否则不可比。
#
# 不用定时 sleep：之前 03:00 的定时链在 5 小时长 sleep 里静默死掉过，
# 看日志以为在等，其实进程早没了。这里改成轮询时钟 + 每半小时心跳，
# 而且卡要是提前空出来也不会提前开跑（必须到点），避免抢同事的卡。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/chain_a54.log
STEPS=${STEPS:-4000}
DS=${DS:-scripts/sft/zero1.json}
START_HOUR=${START_HOUR:-3}
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

hbm_start() {
  local f=$W/hbm_$1.txt
  : > "$f"
  ( while true; do
      npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +98304' | awk '{print $1}' | sort -rn | head -1 >> "$f"
      sleep 20
    done ) >/dev/null 2>&1 &
  echo $!
}
hbm_report() {
  local f=$W/hbm_$1.txt
  [ -s "$f" ] || { echo "无采样"; return; }
  awk '{s+=$1; if($1>m) m=$1; n++} END{printf "峰值 %d MB (%.0f%%) / 均值 %d MB", m, m*100/98304, s/n}' "$f"
}

say "===== a54 夜链挂起，等 ${START_HOUR}:00 ====="
if [ "${GATED:-0}" = "1" ]; then
  say "由宿主侧 night_gate 放行（等时与清卡已在宿主完成）"
  if [ "${DRYRUN:-0}" = "1" ]; then
    say "DRYRUN=1：门闸分支已验证，不训练，就此退出"
    say "  工具目录可见: $(ls -d "$T" 2>/dev/null || echo 不可见)"
    say "  输出根目录可见: $(ls -d "$R" 2>/dev/null || echo 不可见)"
    say "  数据脚本可见: $(ls "$T/run_sft_navsim.sh" 2>/dev/null || echo 不可见)"
    say "DRYRUNDONE"
    exit 0
  fi
else
  tgt=$(date -d "today ${START_HOUR}:00:00" +%s)
  [ "$(date +%s)" -ge "$tgt" ] && tgt=$(date -d "tomorrow ${START_HOUR}:00:00" +%s)
  say "兜底等待：目标 $(date -d "@$tgt" +'%F %T')"
  tick=0
  while [ "$(date +%s)" -lt "$tgt" ]; do
    tick=$((tick+1))
    [ $((tick % 30)) -eq 1 ] && say "心跳：现在 $(date +%H:%M)，还剩 $(( (tgt-$(date +%s))/60 )) 分钟"
    sleep 60
  done
  say ">>> 到点，开始清卡"
  LOG="$LOG" bash "$T/kill_cards.sh"
fi

train() {  # tag  IGNORE
  local tag=$1 ign=$2
  local out=$R/aligned_$tag
  rm -rf "$out"
  say ">>> 训练 $tag  ignore=$ign"
  local hp; hp=$(hbm_start "$tag")
  HIF8=cts HIF8_IGNORE="$ign" \
    DS="$DS" BS=6 GA=1 STEPS=$STEPS LOGSTEP=1 NPROC=8 \
    PORT=$((24900 + RANDOM % 40)) HCCL_RANGE=24000-24099 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/night_$tag.log" 2>&1
  kill "$hp" 2>/dev/null
  say "<<< $tag $(grep -oE 'TRAIN_EXIT=[0-9]+' "$out/train.log" 2>/dev/null | tail -1)  步数 $(grep -oE "[0-9]+/$STEPS" "$out/train.log" 2>/dev/null | tail -1)  $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1)"
  say "    显存 $(hbm_report "$tag")   $(grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$out/train.log" 2>/dev/null | head -1)"
  say "    ckpt $(ls "$out"/*.safetensors 2>/dev/null | wc -l) 片 $(du -sh "$out" 2>/dev/null | cut -f1)"
  say "    首步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | head -1)"
  say "    末步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | tail -1)"
  sleep 30
}

train hif8_168 "lm_head,o_proj"
train hif8_132 "lm_head,o_proj,layers.0.,layers.1.,layers.2.,layers.25.,layers.26.,layers.27."

say "----- 两把并行全量评测（4+4 卡）-----"
ok=0
for tag in hif8_168 hif8_132; do
  [ -f "$R/aligned_$tag/model.safetensors.index.json" ] && ok=$((ok+1)) || say "  $tag 没有 ckpt"
done
if [ "$ok" = 0 ]; then say "两把都没 ckpt，不评测"; say "CHAINA54DONE"; exit 1; fi

( ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TAG=aligned_hif8_168 NP=4 bash "$T/eval_one4.sh" ) & p1=$!
( ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 TAG=aligned_hif8_132 NP=4 bash "$T/eval_one4.sh" ) & p2=$!
wait $p1 $p2

say "----- 结果 -----"
for tag in aligned_hif8_168 aligned_hif8_132; do
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  ns=$(grep -oE 'Number of successful scenarios: [0-9]+' "/tmp/ev_${tag}_full.log" 2>/dev/null | tail -1)
  say "  $tag  full PDMS=${f:-无}   $ns"
done
say "  对照 aligned_bf16 = 0.8120328047184969（已测，12146 场景）"
say "CHAINA54DONE"

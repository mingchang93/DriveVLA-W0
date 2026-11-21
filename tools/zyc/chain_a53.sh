#!/bin/bash
# a53 夜链：03:00 清卡 -> 7B bf16 4000 步 -> 7B hif8_168 4000 步。只训不评。
#
# 底模用官方的 Qwen2.5-VL-7B-Instruct（官方 Qwen_PDMS_85.3 用的就是它），
# 我们此前一直是 Qwen3-VL-2B。换骨干只换 --model_name_or_path：
# train_qwen_vla.py 按 config 的 model_type 自动分家族
#   qwen2_5_vl -> Qwen2_5_VLForConditionalGenerationROSS + flash_attention_2
#
# 冒烟（50 步）已验过：两种配置都跑通、量化 168 自检通过、
# bf16 与 hif8_168 各自逐位一致、ZeRO-1 下显存峰值 58 G / 96 G = 59%。
# 单步 2.48 s，4000 步约 2.8 小时一把。
#
# 为什么不排评测：7B 推理比 2B 慢多少还没量过，排进去会挤掉第二把训练。
# ckpt 存下来（每把约 18 G，4 片），评测另找窗口。
#
# 不用定时 sleep：之前 03:00 的定时链在长 sleep 里静默死掉过。轮询时钟 + 心跳。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
M7B=/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct
LOG=$W/chain_a53.log
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

say "===== a53 夜链挂起，等 ${START_HOUR}:00（7B，$STEPS 步，$(basename $DS .json)）====="
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

train7b() {  # tag  HIF8  IGNORE
  local tag=$1 hif8=$2 ign=$3
  local out=$R/w7b_$tag
  rm -rf "$out"
  say ">>> 7B $tag  hif8=${hif8:-off}"
  local hp; hp=$(hbm_start "w7b_$tag")
  MODEL="$M7B" HIF8="$hif8" HIF8_IGNORE="$ign" \
    DS="$DS" BS=6 GA=1 STEPS=$STEPS LOGSTEP=1 NPROC=8 \
    PORT=$((24950 + RANDOM % 40)) HCCL_RANGE=24100-24199 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/w7b_$tag.log" 2>&1
  kill "$hp" 2>/dev/null
  local ex; ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$out/train.log" 2>/dev/null | tail -1)
  say "<<< 7B $tag $ex  步数 $(grep -oE "[0-9]+/$STEPS" "$out/train.log" 2>/dev/null | tail -1)  $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1)"
  say "    显存 $(hbm_report "w7b_$tag")   $(grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$out/train.log" 2>/dev/null | head -1)"
  say "    ckpt $(ls "$out"/*.safetensors 2>/dev/null | wc -l) 片 $(du -sh "$out" 2>/dev/null | cut -f1)"
  say "    首步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | head -1)"
  say "    末步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | tail -1)"
  if [ "$ex" != "TRAIN_EXIT=0" ]; then
    say "    前几条错误："
    grep -nE 'RuntimeError|ValueError|AssertionError|out of memory|No module' "/tmp/w7b_$tag.log" 2>/dev/null | head -5 | sed 's/^/      /' >> "$LOG"
  fi
  sleep 30
}

train7b bf16     ""   lm_head,o_proj
train7b hif8_168 cts  lm_head,o_proj

say "----- 汇总 -----"
for tag in bf16 hif8_168; do
  say "  7B $tag  $(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/w7b_$tag/train.log" 2>/dev/null | tail -1)  ckpt $(ls "$R/w7b_$tag"/*.safetensors 2>/dev/null | wc -l) 片"
done
say "CHAINA53DONE"

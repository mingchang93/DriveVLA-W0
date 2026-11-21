#!/bin/bash
# 7B 冒烟：用官方底模 Qwen2.5-VL-7B-Instruct 跑通 SFT 流程，bf16 与 HiF8-168 两种配置。
#
# 为什么是 7B：官方 Qwen_PDMS_85.3 的 training_args.bin 里底模是 Qwen2.5-VL-7B，
# 我们一直跑的是 Qwen3-VL-2B。换骨干只需换 --model_name_or_path：
# train_qwen_vla.py 按 config 的 model_type 自动分家族
# （qwen2_5_vl -> Qwen2_5_VLForConditionalGenerationROSS + flash_attention_2）。
#
# ZeRO-2 而不是 ZeRO-1：ZeRO-1 只切优化器状态，梯度每卡仍各存一份完整的
# （7B bf16 约 15 G）。ZeRO-2 把梯度也切开，是 7B 该用的档。
# 单卡 HBM 96 G，理论够，但没实测过，所以带 BS 回退：6 -> 3 -> 1。
#
# 量化范围不用改：Qwen2.5-VL 的子模块命名与 Qwen3-VL 一致
# （model.language_model.layers.*），且同为 28 层，所以 ignore=lm_head,o_proj
# 仍然是 168 个 Linear。脚本会打印自检结果，不靠推断。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
M7B=/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/smoke7b.log
STEPS=${STEPS:-50}
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

say "===== 7B 冒烟开始（$STEPS 步，ZeRO-2）====="
say "底模 $M7B  $(du -sh $M7B 2>/dev/null | cut -f1)"

# 后台记录 HBM 峰值，训练结束后汇报
hbmlog=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/smoke7b_hbm.txt
: > "$hbmlog"
( while true; do
    npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +98304' | awk '{print $1}' | sort -rn | head -1 >> "$hbmlog"
    sleep 20
  done ) &
HBMPID=$!
trap 'kill $HBMPID 2>/dev/null' EXIT

one() {  # tag  HIF8  IGNORE
  local tag=$1 hif8=$2 ign=$3
  local out=$R/smoke7b_$tag
  for bs in 6 3 1; do
    rm -rf "$out"
    say ">>> $tag  bs=$bs  hif8=${hif8:-off}"
    : > "$hbmlog"
    MODEL="$M7B" HIF8="$hif8" HIF8_IGNORE="$ign" \
      DS=scripts/sft/zero2.json BS=$bs GA=1 STEPS=$STEPS LOGSTEP=1 NPROC=8 \
      PORT=24810 HCCL_RANGE=23700-23799 OUT="$out" \
      bash "$T/run_sft_navsim.sh" > "/tmp/smoke7b_$tag.log" 2>&1
    local ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$out/train.log" 2>/dev/null | tail -1)
    local peak=$(sort -rn "$hbmlog" 2>/dev/null | head -1)
    local q=$(grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$out/train.log" 2>/dev/null | head -1)
    local st=$(grep -oE "[0-9]+/$STEPS" "$out/train.log" 2>/dev/null | tail -1)
    if [ "$ex" = "TRAIN_EXIT=0" ]; then
      say "<<< $tag 跑通  bs=$bs  步数 $st  HBM峰值 ${peak:-?} / 98304 MB  $q"
      say "    ckpt $(ls "$out"/*.safetensors 2>/dev/null | wc -l) 片 $(du -sh "$out" 2>/dev/null | cut -f1)  $(grep -oE "'"'"'train_runtime'"'"': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1)"
      say "    首步 $(grep -oE '{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | head -1)"
      return 0
    fi
    # 判断是不是显存不够——是就降 bs 重试，别的错直接报
    if grep -qiE 'out of memory|OOM|NPU out of memory|RuntimeError: NPU' "/tmp/smoke7b_$tag.log"; then
      say "!!! $tag bs=$bs 显存不足（HBM峰值 ${peak:-?} MB），降 bs 重试"
      continue
    fi
    say "!!! $tag bs=$bs 失败（非显存问题，$ex），前几条错误："
    grep -nE 'Error|Traceback|raise |Exception|assert' "/tmp/smoke7b_$tag.log" | head -6 | sed 's/^/      /' >> "$LOG"
    return 1
  done
  say "!!! $tag 连 bs=1 都跑不起来"
  return 1
}

one bf16     ""   lm_head,o_proj
sleep 30
one hif8_168 cts  lm_head,o_proj
say "SMOKE7BDONE"

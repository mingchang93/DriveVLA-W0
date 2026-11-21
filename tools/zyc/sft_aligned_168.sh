#!/bin/bash
# a54 上补跑 HiF8-168（客户配置）的 aligned 4000 步。
# 这一臂之前被取消了，所以 aligned 组少了客户规格那一档，无法与 lr 8e-5 轮次对照。
# 超参与 aligned_bf16 / aligned_hif8_196 完全相同，只差 hif8_ignore：
#   lm_head,o_proj -> 168（客户规格，o_proj 不量化）
#   lm_head        -> 196
# 只训不评：评测开关 .skip_eval 仍在，评测等确认后单独发。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/sft_aligned_168.log
say() { echo "[$(date +%F) $(date +%T)] $*" >> "$LOG"; }

tag=hif8_168; out=$R/aligned_$tag
say ">>> 训练 $tag（量化 168，客户配置）"
HIF8=cts HIF8_IGNORE=lm_head,o_proj \
  DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=4000 LOGSTEP=1 NPROC=8 \
  PORT=24792 HCCL_RANGE=23500-23599 OUT="$out" \
  bash "$T/run_sft_navsim.sh" > /tmp/aligned_$tag.log 2>&1
rc=$?
say "<<< 训练 $tag rc=$rc $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$out"/*.safetensors 2>/dev/null | wc -l) $(grep -oE "量化 [0-9]+ 个" "$out/train.log" 2>/dev/null | head -1)"
say "    首步 $(grep -oE "\{.action_loss.: [0-9.]+" "$out/train.log" 2>/dev/null | head -1 | grep -oE "[0-9.]+$")  末步 $(grep -oE "\{.action_loss.: [0-9.]+" "$out/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")"
say "SFT168DONE"

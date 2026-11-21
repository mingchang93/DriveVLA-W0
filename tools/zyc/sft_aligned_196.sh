#!/bin/bash
# a53 上跑 HiF8-196（aligned 配置），跑完接 mini + full 评测。
# 与 a54 的 bf16 / HiF8-168 用完全相同的超参，只差 hif8_ignore：
#   lm_head        -> 196（o_proj 也量化）
#   lm_head,o_proj -> 168（客户规格）
# 超参已对齐官方 Qwen_PDMS_85.3：lr 1e-5 / warmup 200 / wd 0.001 / cosine / max_len 8192
# 日志用 >> 直写文件，不经过 tee（docker exec -d + setsid 后 stdout fd 失效，
# tee 会连文件那份一起写不成）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/sft_aligned_196.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

tag=hif8_196; out=$R/aligned_$tag
say ">>> 训练 $tag（量化 196）"
HIF8=cts HIF8_IGNORE=lm_head \
  DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=4000 LOGSTEP=1 NPROC=8 \
  PORT=24780 HCCL_RANGE=23300-23399 OUT="$out" \
  bash "$T/run_sft_navsim.sh" > /tmp/aligned_$tag.log 2>&1
say "<<< 训练 $tag rc=$? $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$out"/*.safetensors 2>/dev/null | wc -l) $(grep -oE '量化 [0-9]+ 个' "$out/train.log" 2>/dev/null | head -1)"

for mode in mini full; do
  [ -f "$out/model.safetensors.index.json" ] || { say "没有 ckpt，跳过评测"; break; }
  say ">>> 评测 $tag $mode"
  bash "$T/eval_ckpt.sh" "$out" "aligned_$tag" "$mode" 8 > /tmp/evalal_${tag}_${mode}.log 2>&1
  s=$(grep -oE 'Final average score of valid results: [0-9.]+' /tmp/evalal_${tag}_${mode}.log | tail -1 | grep -oE '[0-9.]+$')
  say "<<< 评测 $tag $mode PDMS=${s:-失败}"
  [ "$mode" = mini ] && [ -z "$s" ] && { say "mini 没出分，跳过 full"; break; }
done
say "SFT196DONE"

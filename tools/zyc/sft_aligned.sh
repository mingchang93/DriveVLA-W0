#!/bin/bash
# 对齐官方超参后重跑 SFT：a54 上串行跑 bf16 和 HiF8-168，各 4000 步，跑完接评测。
#
# 超参已对齐 liyingyan/DriveVLA-W0 的 Qwen_PDMS_85.3（training_args.bin）：
#   lr 1e-5 · warmup 200 · weight_decay 0.001 · scheduler cosine · model_max_length 8192
# 原先那几个值来自 Emu3 的 train_base_ar_withou_moe.sh——repo 里没有 Qwen 脚本，
# 当时只能借 Emu3 的。
#
# 只用 a54（其余三台是 lh 的 vllm 在跑，按指示不动）。所以串行而非并行。
#
# 日志用 >> 直接写文件，不经过 tee：今天多个脚本的日志只剩一两行，根因是
# docker exec -d + setsid 后 stdout 的 fd 失效，tee 写 stdout 失败就退出，
# 连文件那份也写不成。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/sft_aligned.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

say "===== 对齐官方超参后重跑 SFT（a54 串行）====="
say "lr 1e-5 / warmup 200 / wd 0.001 / cosine / max_len 8192 / 全局 48"

one() {  # tag  HIF8  HIF8_IGNORE
  local tag=$1 hif8=$2 ign=$3
  local out="$R/aligned_$tag"
  say ">>> 训练 $tag  hif8='${hif8:-off}'"
  HIF8="$hif8" HIF8_IGNORE="$ign" \
    DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=4000 LOGSTEP=1 NPROC=8 \
    PORT=$((24700 + RANDOM % 40)) HCCL_RANGE=23000-23199 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/aligned_$tag.log" 2>&1
  local rc=$?
  say "<<< 训练 $tag rc=$rc $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$out"/*.safetensors 2>/dev/null | wc -l) $(grep -oE '量化 [0-9]+ 个' "$out/train.log" 2>/dev/null | head -1)"
  say "    首步 $(grep -oE "\{.action_loss.: [0-9.]+" "$out/train.log" 2>/dev/null | head -1 | grep -oE '[0-9.]+$')  末步 $(grep -oE "\{.action_loss.: [0-9.]+" "$out/train.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
  sleep 45
}

one bf16     ""   lm_head,o_proj
one hif8_168 cts  lm_head,o_proj

# ---- 评测：先 mini 确认链路，出分再跑 full ----
say "----- 评测 -----"
for tag in bf16 hif8_168; do
  ck="$R/aligned_$tag"
  [ -f "$ck/model.safetensors.index.json" ] || { say "  $tag 没有 ckpt，跳过"; continue; }
  for mode in mini full; do
    say ">>> 评测 $tag $mode"
    bash "$T/eval_ckpt.sh" "$ck" "aligned_$tag" "$mode" 8 > "/tmp/evalal_${tag}_${mode}.log" 2>&1
    s=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/evalal_${tag}_${mode}.log" | tail -1 | grep -oE '[0-9.]+$')
    say "<<< 评测 $tag $mode PDMS=${s:-失败}"
    [ "$mode" = mini ] && [ -z "$s" ] && { say "  mini 没出分，跳过 full"; break; }
  done
done

say "===== 汇总 ====="
for tag in bf16 hif8_168; do
  say "  $tag  mini=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/evalal_${tag}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')  full=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/evalal_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
done
say "SFTALIGNEDDONE"

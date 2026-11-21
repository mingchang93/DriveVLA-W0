#!/bin/bash
# a54 串行链（7B）：
#   1) bf16      50 步  ——  能不能拉起 + 显存
#   2) hif8_168  50 步  ——  同上，且核对量化层数
#   3) bf16 / hif8_168 各再跑一把同配置  ——  验确定性（逐位一致）
#   4) 两把都过 -> 并行起 lr1e-5 的两个评测（4+4 卡）
#
# 底模换成官方的 Qwen2.5-VL-7B-Instruct（官方 Qwen_PDMS_85.3 用的就是它），
# 我们此前一直跑 Qwen3-VL-2B。换骨干只换 --model_name_or_path：
# train_qwen_vla.py 按 config 的 model_type 自动分家族
#   qwen2_5_vl -> Qwen2_5_VLForConditionalGenerationROSS + flash_attention_2
#
# ZeRO-2 而非 ZeRO-1：ZeRO-1 不切梯度，7B 每卡要各存一份约 15 G 的完整梯度。
# 单卡 HBM 96 G，带 bs 回退 6 -> 3 -> 1，免得一次 OOM 就白跑。
#
# 确定性判据用 grad_norm 与我们自己打的 action_loss/ross_loss：
# HF Trainer 把 loss round 到 4 位小数，拿它比对等于用 1e-4 的尺子。
#
# 显存：每把单独记一份采样，报峰值与均值（采样取 8 卡里的最大值）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
M7B=/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct
LOG=$W/chain7b.log
STEPS=${STEPS:-50}
DS=${DS:-scripts/sft/zero2.json}   # ZeRO 档位，整条链统一
#   zero1 -> 只切优化器状态，梯度每卡各存一份完整的（约 +13 G）
#   zero2 -> 梯度也切开，实测 7B bs6 峰值 54.7 G / 96 G = 56%
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

hbm_start() {   # $1=标签；后台采样，8 卡取最大
  local f=$W/hbm_$1.txt
  : > "$f"
  ( while true; do
      npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +98304' \
        | awk '{print $1}' | sort -rn | head -1 >> "$f"
      sleep 10
    done ) >/dev/null 2>&1 &
  echo $!
}
hbm_report() {  # $1=标签
  local f=$W/hbm_$1.txt
  [ -s "$f" ] || { echo "无采样"; return; }
  awk '{s+=$1; if($1>m) m=$1; n++} END{printf "峰值 %d MB (%.0f%%) / 均值 %d MB / 采样 %d 次", m, m*100/98304, s/n, n}' "$f"
}

run() {  # tag  HIF8  IGNORE  BSLIST
  local tag=$1 hif8=$2 ign=$3 bslist=$4
  local out=$R/smoke7b_$tag
  for bs in $bslist; do
    rm -rf "$out"
    say ">>> $tag  bs=$bs  hif8=${hif8:-off}"
    local hp; hp=$(hbm_start "$tag")
    MODEL="$M7B" HIF8="$hif8" HIF8_IGNORE="$ign" \
      DS="$DS" BS=$bs GA=1 STEPS=$STEPS LOGSTEP=1 NPROC=8 \
      PORT=$((24810 + RANDOM % 40)) HCCL_RANGE=23700-23799 OUT="$out" \
      bash "$T/run_sft_navsim.sh" > "/tmp/smoke7b_$tag.log" 2>&1
    kill "$hp" 2>/dev/null
    local ex; ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$out/train.log" 2>/dev/null | tail -1)
    if [ "$ex" = "TRAIN_EXIT=0" ]; then
      echo "$bs" > "$W/bs_$tag.txt"
      say "<<< $tag 跑通  bs=$bs  步数 $(grep -oE "[0-9]+/$STEPS" "$out/train.log" | tail -1)"
      say "    显存 $(hbm_report "$tag")"
      say "    量化 $(grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$out/train.log" | head -1)$(grep -oE '数量自检[^，]*' "$out/train.log" | head -1)"
      say "    用时 $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" | tail -1)  ckpt $(ls "$out"/*.safetensors 2>/dev/null | wc -l) 片 $(du -sh "$out" 2>/dev/null | cut -f1)"
      say "    首步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" | head -1)"
      return 0
    fi
    if grep -qiE 'out of memory|NPU out of memory|HBM' "/tmp/smoke7b_$tag.log"; then
      say "!!! $tag bs=$bs 显存不足（$(hbm_report "$tag")），降 bs 重试"
      continue
    fi
    say "!!! $tag bs=$bs 失败（非显存，$ex），前几条："
    grep -nE 'RuntimeError|ValueError|AssertionError|No module' "/tmp/smoke7b_$tag.log" | head -5 | sed 's/^/      /' >> "$LOG"
    return 1
  done
  say "!!! $tag 连最小 bs 都起不来"
  return 1
}

say "===== 7B 链开始（$STEPS 步，$(basename $DS .json)，底模 $(du -sh $M7B | cut -f1)）====="

run bf16     ""   lm_head,o_proj "6 3 1" || { say "CHAIN7BDONE"; exit 1; }
sleep 20
run hif8_168 cts  lm_head,o_proj "6 3 1" || { say "CHAIN7BDONE"; exit 1; }
sleep 20

# ---- 确定性：同配置各再跑一把，bs 必须与第一把一致 ----
say "----- 确定性复跑（同配置第二把）-----"
for tag in bf16 hif8_168; do
  bs=$(cat "$W/bs_$tag.txt" 2>/dev/null || echo 6)
  h=$([ "$tag" = bf16 ] && echo "" || echo cts)
  out=$R/smoke7b_${tag}_det
  rm -rf "$out"
  say ">>> ${tag}_det  bs=$bs"
  hp=$(hbm_start "${tag}_det")
  MODEL="$M7B" HIF8="$h" HIF8_IGNORE=lm_head,o_proj \
    DS="$DS" BS=$bs GA=1 STEPS=$STEPS LOGSTEP=1 NPROC=8 \
    PORT=$((24860 + RANDOM % 40)) HCCL_RANGE=23800-23899 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/smoke7b_${tag}_det.log" 2>&1
  kill "$hp" 2>/dev/null
  say "<<< ${tag}_det $(grep -oE 'TRAIN_EXIT=[0-9]+' "$out/train.log" 2>/dev/null | tail -1)  显存 $(hbm_report "${tag}_det")"
  sleep 20
done

say "----- 逐位比对 -----"
python3 "$T/cmp_det.py" smoke7b_bf16 smoke7b_bf16_det smoke7b_hif8_168 smoke7b_hif8_168_det >> "$LOG" 2>&1

# ---- 评测 ----
ok=1
for tag in bf16 hif8_168; do
  ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/smoke7b_$tag/train.log" 2>/dev/null | tail -1)
  [ "$ex" = "TRAIN_EXIT=0" ] || ok=0
done
if [ "$ok" != 1 ]; then say "!!! 冒烟未全过，不进评测"; say "CHAIN7BDONE"; exit 1; fi

say ">>> 并行起 lr1e-5 评测（4+4 卡）"
( ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 TAG=aligned_bf16     NP=4 bash "$T/eval_one4.sh" ) & p1=$!
( ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 TAG=aligned_hif8_196 NP=4 bash "$T/eval_one4.sh" ) & p2=$!
wait $p1 $p2
say "----- lr1e-5 评测结果 -----"
for tag in aligned_bf16 aligned_hif8_196; do
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/ev_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
  say "  $tag  mini=${m:-无}  full=${f:-无}"
done
say "CHAIN7BDONE"

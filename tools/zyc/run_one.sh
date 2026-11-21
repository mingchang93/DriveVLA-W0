#!/bin/bash
# 跑单把训练，带显存采样与结果汇总。给临时腾出来的机器用。
#   用法: TAG=w7b_hif8_168 HIF8=cts IGN=lm_head,o_proj MODEL=<底模> bash run_one.sh
#
# 和夜链里的 train()/train7b() 做同样的事，只是不串成链——
# 这样可以把某一把从串行队列里摘出来、挪到别的机器并行跑，缩短总时长。
# 输出目录与标签必须与原队列一致，否则后续评测和对照会对不上。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
TAG=${TAG:?}; HIF8=${HIF8:-}; IGN=${IGN:-lm_head,o_proj}
MODEL=${MODEL:-/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct}
SEED=${SEED:-42}
STEPS=${STEPS:-4000}
DS=${DS:-scripts/sft/zero1.json}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/one_${TAG}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

out=$R/$TAG
f=$W/hbm_${TAG}.txt
: > "$f"
( while true; do
    npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +98304' | awk '{print $1}' | sort -rn | head -1 >> "$f"
    sleep 20
  done ) >/dev/null 2>&1 &
hp=$!
trap 'kill $hp 2>/dev/null' EXIT

rm -rf "$out"
say ">>> $TAG  底模=$(basename "$MODEL")  hif8=${HIF8:-off}  ignore=$IGN  seed=$SEED  $(basename $DS .json)"
MODEL="$MODEL" HIF8="$HIF8" HIF8_IGNORE="$IGN" SEED="$SEED" HIF8_INTERVAL="${HIF8_INTERVAL:-10}" \
  DS="$DS" BS=6 GA=1 STEPS=$STEPS LOGSTEP=1 NPROC=8 \
  PORT=$((25000 + RANDOM % 60)) HCCL_RANGE=24300-24399 OUT="$out" \
  bash "$T/run_sft_navsim.sh" > "/tmp/one_$TAG.log" 2>&1
kill $hp 2>/dev/null

ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$out/train.log" 2>/dev/null | tail -1)
say "<<< $TAG $ex  步数 $(grep -oE "[0-9]+/$STEPS" "$out/train.log" 2>/dev/null | tail -1)  $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1)"
say "    显存 $(awk '{s+=$1; if($1>m) m=$1; n++} END{printf "峰值 %d MB (%.0f%%) / 均值 %d MB", m, m*100/98304, s/n}' "$f" 2>/dev/null)"
say "    $(grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$out/train.log" 2>/dev/null | head -1)  ckpt $(ls "$out"/*.safetensors 2>/dev/null | wc -l) 片 $(du -sh "$out" 2>/dev/null | cut -f1)"
say "    首步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | head -1)"
say "    末步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$out/train.log" 2>/dev/null | tail -1)"
[ "$ex" = "TRAIN_EXIT=0" ] || { say "    前几条错误："; grep -nE 'RuntimeError|ValueError|AssertionError|out of memory' "/tmp/one_$TAG.log" 2>/dev/null | head -5 | sed 's/^/      /' >> "$LOG"; }
say "ONEDONE_$TAG"

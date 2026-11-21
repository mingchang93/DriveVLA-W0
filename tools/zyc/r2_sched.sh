#!/bin/bash
# r2: 定点起跑（默认 23:15，第9参数可改），重跑六臂（ZYC 工作区 + lmc venv-hif8，与 11-15/16 基线同环境）。
# 每台机：第1臂 = bf16 预训练(复用 pt_ptlr5e5) + 量化 SFT；第2臂 = DTS 预训练 8000@5e-5 + 量化 SFT 4000@1e-5。
# 用法: r2_sched.sh <role> <ignore> <pt_sft_tag> <chain_tag> <port_sft> <port_pt> <hr_pt> <hr_sft>
set -u
export TZ=Asia/Shanghai
ROLE=${1:?role}
IG=${2:?ignore}
PT_TAG=${3:?sft tag}
CH_TAG=${4:?chain tag}
PORT_SFT=${5:?port}
PORT_PT=${6:?port}
HR_PT=${7:?range}
HR_SFT=${8:?range}

T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/r2_${ROLE}.log
START=${9:-2315}
say(){ echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }
occ(){ npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+/{c++} END{print c+0}'; }

say "===== r2 / $ROLE : $START 定点起跑 · ignore=$IG ====="
say "    代码 $T/.. · 输出 $W/runs · 容器 zyc-w0 · 解释器 lmc/venv-hif8（与基线一致）"

# --- 定点 23:15 ---
while [ "$(date +%H%M)" -lt "$START" ]; do sleep 5; done
say "到点 $START，当前占卡 $(occ)"

# --- a52 释放 NPU（只杀进程，不碰容器）---
if [ "$ROLE" = "a52" ]; then
  say "释放占卡进程（按用户指定命令，不动容器）"
  pkill -9 vllm; pkill -9 VLLM; pkill -9 python; pkill -9 -f mooncake; pkill -9 -f transfer_engine; pkill -9 mooncake-master
  say "pkill 已发，等 HBM 回落"
fi

w=0
while [ "$(occ)" -ne 0 ] && [ "$w" -lt 120 ]; do
  sleep 5; w=$((w+1))
  [ $((w % 6)) -eq 0 ] && say "    等卡：$(occ) 个进程（$((w*5))s）"
done

# 清理没落地的情形：再补一次（仅 a52），然后交给脚本自带的占卡闸判断
if [ "$(occ)" -ne 0 ]; then
  say "!! 仍有 $(occ) 个进程占卡"
  if [ "$ROLE" = "a52" ]; then
    say "    再发一次 pkill，再等 300s"
    pkill -9 vllm; pkill -9 VLLM; pkill -9 python; pkill -9 -f mooncake; pkill -9 -f transfer_engine; pkill -9 mooncake-master
    w=0; while [ "$(occ)" -ne 0 ] && [ "$w" -lt 60 ]; do sleep 5; w=$((w+1)); done
    say "    之后占卡 $(occ)"
  fi
fi
say ">>> 起跑，占卡 $(occ)"

# --- 第 1 臂：复用 pt_ptlr5e5，只做量化 SFT + full 评测 ---
say ">>> 第1臂 bf16/DTS：SFT(init=pt_ptlr5e5, ignore=$IG) + eval"
PT_DIR=pt_ptlr5e5 TAG="$PT_TAG" HIF8_SFT=dts IG_SFT="$IG" HIF8_INTERVAL=10 \
  PORT_SFT="$PORT_SFT" HR_SFT="$HR_SFT" bash "$T/sft_from_pt.sh" >> "$LOG" 2>&1
say "<<< 第1臂结束 rc=$?"

# --- 第 2 臂：DTS 预训练 8000@5e-5 -> 量化 SFT 4000@1e-5 -> eval ---
say ">>> 第2臂 DTS/DTS：预训练 8000@5e-5 + SFT 4000@1e-5 (ignore=$IG)"
TAG="$CH_TAG" HIF8_PT=dts HIF8_SFT=dts IG_PT="$IG" IG_SFT="$IG" HIF8_INTERVAL=10 \
  SAVE_STEPS_PT=1000 PORT_PT="$PORT_PT" PORT_SFT="$PORT_SFT" HR_PT="$HR_PT" HR_SFT="$HR_SFT" \
  bash "$T/chain_ptlr.sh" >> "$LOG" 2>&1
say "<<< 第2臂结束 rc=$?"
say "R2_${ROLE}_ALLDONE"

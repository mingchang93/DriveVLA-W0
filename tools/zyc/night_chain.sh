#!/bin/bash
# 夜间任务链：03:00（本地 CST）启动，四把 4000 步 SFT，跑完接评测。
#
# 容器是 UTC、宿主机是 CST。不显式指定时区的话 "明天 03:00" 会按 UTC 算，
# 实际落在本地 11:00，晚 8 小时。epoch 是绝对的，所以统一用上海时区来算。
set -u
export TZ=Asia/Shanghai

T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/night_chain.log
START_HHMM=${START_HHMM:-03:00}

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---------- 1. 等到预定时刻 ----------
now=$(date +%s)
tgt=$(date -d "today $START_HHMM" +%s 2>/dev/null)
if [ -z "$tgt" ] || [ "$tgt" -le "$now" ]; then
  tgt=$(date -d "tomorrow $START_HHMM" +%s)
fi
say "任务链挂起，将在 $(date -d "@$tgt" '+%F %T %Z') 启动（还有 $(( (tgt-now)/60 )) 分钟）"
while [ "$(date +%s)" -lt "$tgt" ]; do sleep 60; done
say "到点，开始"

# ---------- 2. 清空占卡进程 ----------
# 按指示清卡。不用 pkill 盲杀：从 npu-smi 的进程表取 PID，
# 杀之前把完整命令行写进日志，留个凭据。
say "----- 清理占卡进程 -----"
npu_pids() { npu-smi info 2>/dev/null | awk -F'|' '{gsub(/ /,"",$3); if ($3 ~ /^[0-9]+$/) print $3}' | sort -u; }
pids=$(npu_pids)
if [ -z "$pids" ]; then
  say "  八张卡本来就是空的"
else
  for p in $pids; do
    say "  kill $p : $(ps -o cmd= -p "$p" 2>/dev/null | cut -c1-140)"
    kill "$p" 2>/dev/null
  done
  sleep 20
  for p in $(npu_pids); do say "  仍在，强杀 $p"; kill -9 "$p" 2>/dev/null; done
  sleep 15
fi
say "  清理后空闲卡数: $(npu-smi info 2>/dev/null | grep -c 'No running processes')/8"

# ---------- 3. 四把训练 ----------
# ignore 与量化数量的对应（语言模型 28 层 × 7 类 = 196 个 Linear）：
#   lm_head                 -> 196   o_proj 也量化
#   lm_head,o_proj          -> 168   客户规格基线
#   lm_head,o_proj,前后三层 -> 132   edge3：前三层 + 后三层留 bf16
EDGE3_IGN="lm_head,o_proj,layers.0.,layers.1.,layers.2.,layers.25.,layers.26.,layers.27."

train_one() {  # tag  HIF8  HIF8_IGNORE
  local tag=$1 hif8=$2 ign=$3
  local out="$R/night_$tag"
  say ">>> 训练 $tag  hif8='${hif8:-off}' ignore='$ign'"
  HIF8="$hif8" HIF8_IGNORE="$ign" \
    DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=4000 LOGSTEP=1 NPROC=8 \
    PORT=$((24100 + RANDOM % 60)) HCCL_RANGE=22000-22199 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/night_$tag.log" 2>&1
  say "<<< 训练 $tag rc=$? $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$out"/*.safetensors 2>/dev/null | wc -l) 量化=$(grep -oE '量化 [0-9]+ 个' "$out/train.log" 2>/dev/null | head -1)"
  sleep 60
}

train_one bf16        ""   lm_head,o_proj
train_one hif8_168    cts  lm_head,o_proj
train_one hif8_196    cts  lm_head
train_one hif8_edge3  cts  "$EDGE3_IGN"

# ---------- 4. 评测 ----------
# 先 mini（21 场景，约 1.5 分钟，已实测走通）确认 ckpt 能加载、链路没坏，
# 再跑 full（navtest 全量 12146 场景），后者才是用于对比的指标。
# mini 失败就跳过该 ckpt 的 full，省得白跑。
say "----- 开始评测 -----"
for tag in bf16 hif8_168 hif8_196 hif8_edge3; do
  ck="$R/night_$tag"
  if [ ! -f "$ck/model.safetensors.index.json" ]; then
    say "  $tag 没有 ckpt，跳过"; continue
  fi
  say ">>> 评测 $tag (mini)"
  bash "$T/eval_ckpt.sh" "$ck" "night_$tag" mini 8 > "/tmp/eval_${tag}_mini.log" 2>&1
  m=$(grep -oE "Final average score of valid results: [0-9.]+" "/tmp/eval_${tag}_mini.log" | tail -1)
  say "<<< 评测 $tag (mini) $m"
  if [ -z "$m" ]; then say "  $tag mini 没出分，跳过 full"; continue; fi
  say ">>> 评测 $tag (full)"
  bash "$T/eval_ckpt.sh" "$ck" "night_$tag" full 8 > "/tmp/eval_${tag}_full.log" 2>&1
  f=$(grep -oE "Final average score of valid results: [0-9.]+" "/tmp/eval_${tag}_full.log" | tail -1)
  say "<<< 评测 $tag (full) $f"
done

say "----- 汇总 -----"
for tag in bf16 hif8_168 hif8_196 hif8_edge3; do
  say "  $tag  mini=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/eval_${tag}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')  full=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/eval_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
done
say "NIGHTCHAINDONE"

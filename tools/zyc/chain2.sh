#!/bin/bash
# 夜间任务链（双机版）：03:00 CST 启动，本机跑两把 4000 步 SFT，跑完接评测。
#   用法: ROLE=a52 bash chain2.sh    -> bf16 + hif8_168
#         ROLE=a54 bash chain2.sh    -> hif8_196 + hif8_edge3
#
# 容器是 UTC、宿主机是 CST。不显式指定时区的话 "明天 03:00" 会按 UTC 算，
# 实际落在本地 11:00，晚 8 小时。epoch 是绝对的，所以统一用上海时区。
set -u
export TZ=Asia/Shanghai

T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
ROLE=${ROLE:?必须指定 ROLE=a52 或 a54}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/chain_${ROLE}.log
START_HHMM=${START_HHMM:-03:00}

say() { echo "[$(date '+%F %T')] [$ROLE] $*" | tee -a "$LOG"; }

# ---------- 1. 等到预定时刻 ----------
now=$(date +%s)
tgt=$(date -d "today $START_HHMM" +%s 2>/dev/null)
if [ -z "$tgt" ] || [ "$tgt" -le "$now" ]; then tgt=$(date -d "tomorrow $START_HHMM" +%s); fi
say "挂起，将在 $(date -d "@$tgt" '+%F %T %Z') 启动（还有 $(( (tgt-now)/60 )) 分钟）"
while [ "$(date +%s)" -lt "$tgt" ]; do sleep 60; done
say "到点"

# ---------- 2. 清卡：只杀 lh 的，别人的不动 ----------
npu_pids() { npu-smi info 2>/dev/null | awk -F'|' '{gsub(/ /,"",$3); if ($3 ~ /^[0-9]+$/) print $3}' | sort -u; }
owner() {   # 把 pid 映射到容器名，判断归属
  local cid
  cid=$(grep -oE 'docker-[a-f0-9]{12}' /proc/$1/cgroup 2>/dev/null | head -1 | sed 's/docker-//')
  [ -n "$cid" ] && docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null | awk -v c="$cid" '$1==c{print $2}' | head -1
}

say "----- 清卡（规则：只杀 lh 的）-----"
blocked=0
for p in $(npu_pids); do
  o=$(owner "$p"); o=${o:-未知}
  if echo "$o" | grep -qiE '(^|[^a-z])lh([^a-z]|$)|^lh'; then
    say "  kill $p 容器=$o : $(ps -o cmd= -p "$p" 2>/dev/null | cut -c1-110)"
    kill "$p" 2>/dev/null
  else
    say "  保留 $p 容器=$o（不是 lh 的，按规则不动）"
    blocked=1
  fi
done
sleep 25

free=$(npu-smi info 2>/dev/null | grep -c 'No running processes')
say "  当前空闲卡 $free/8"
if [ "$free" -lt 8 ]; then
  say "  !! 卡没空出来，等待中（每 5 分钟查一次，最多 3 小时）"
  waited=0
  while [ "$free" -lt 8 ] && [ "$waited" -lt 180 ]; do
    sleep 300; waited=$((waited+5))
    free=$(npu-smi info 2>/dev/null | grep -c 'No running processes')
    [ $((waited % 15)) -eq 0 ] && say "  已等 $waited 分钟，空闲 $free/8"
  done
  if [ "$free" -lt 8 ]; then
    say "  !! 等了 3 小时仍不足 8 卡，放弃本次任务链。需要人工介入。"
    say "CHAINABORT_${ROLE}"; exit 1
  fi
fi
say "  八卡就绪"

# ---------- 3. 两把训练 ----------
# ignore 与量化数量（语言模型 28 层 × 7 类 = 196 个 Linear）：
#   lm_head                 -> 196   o_proj 也量化
#   lm_head,o_proj          -> 168   客户规格基线
#   lm_head,o_proj,前后三层 -> 132   edge3：前三层 + 后三层留 bf16
EDGE3_IGN="lm_head,o_proj,layers.0.,layers.1.,layers.2.,layers.25.,layers.26.,layers.27."

train_one() {  # tag  HIF8  HIF8_IGNORE
  local tag=$1 hif8=$2 ign=$3
  local out="$R/night_$tag"
  say ">>> 训练 $tag hif8='${hif8:-off}' ignore='$ign'"
  HIF8="$hif8" HIF8_IGNORE="$ign" \
    DS=scripts/sft/zero1.json BS=6 GA=1 STEPS=4000 LOGSTEP=1 NPROC=8 \
    PORT=$((24100 + RANDOM % 60)) HCCL_RANGE=22000-22199 OUT="$out" \
    bash "$T/run_sft_navsim.sh" > "/tmp/night_$tag.log" 2>&1
  say "<<< 训练 $tag rc=$? $(grep -oE "'train_runtime': [0-9.]+" "$out/train.log" 2>/dev/null | tail -1) ckpt=$(ls "$out"/*.safetensors 2>/dev/null | wc -l) $(grep -oE '量化 [0-9]+ 个' "$out/train.log" 2>/dev/null | head -1)"
  sleep 60
}

if [ "$ROLE" = "a52" ]; then
  TAGS="bf16 hif8_168"
  train_one bf16     ""   lm_head,o_proj
  train_one hif8_168 cts  lm_head,o_proj
else
  TAGS="hif8_196 hif8_edge3"
  train_one hif8_196   cts  lm_head
  train_one hif8_edge3 cts  "$EDGE3_IGN"
fi

# ---------- 4. 评测本机这两把 ----------
# 先 mini（21 场景，约 1.5 分钟，已实测走通）确认 ckpt 能加载，
# 出分了再跑 full（navtest 全量 12146 场景）——后者才是对比用的指标。
say "----- 评测 -----"
for tag in $TAGS; do
  ck="$R/night_$tag"
  [ -f "$ck/model.safetensors.index.json" ] || { say "  $tag 没有 ckpt，跳过"; continue; }
  say ">>> 评测 $tag mini"
  bash "$T/eval_ckpt.sh" "$ck" "night_$tag" mini 8 > "/tmp/eval_${tag}_mini.log" 2>&1
  m=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/eval_${tag}_mini.log" | tail -1 | grep -oE '[0-9.]+$')
  say "<<< 评测 $tag mini PDMS=${m:-失败}"
  [ -n "$m" ] || { say "  mini 没出分，跳过 full"; continue; }
  say ">>> 评测 $tag full"
  bash "$T/eval_ckpt.sh" "$ck" "night_$tag" full 8 > "/tmp/eval_${tag}_full.log" 2>&1
  f=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/eval_${tag}_full.log" | tail -1 | grep -oE '[0-9.]+$')
  say "<<< 评测 $tag full PDMS=${f:-失败}"
done

say "----- 本机汇总 -----"
for tag in $TAGS; do
  say "  $tag  mini=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/eval_${tag}_mini.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')  full=$(grep -oE 'Final average score of valid results: [0-9.]+' "/tmp/eval_${tag}_full.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
done
say "CHAINDONE_${ROLE}"

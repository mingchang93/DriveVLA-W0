#!/bin/bash
# Run one DTS-168 interval-10 chain at 23:30 CST.
# Usage: schedule_dts168_i10_2330.sh <node> <seed> <tag> <port-pt> <port-sft> <hccl-pt> <hccl-sft>
set -u
export TZ=Asia/Shanghai

EXPECTED_NODE=${1:?node}
SEED=${2:?seed}
TAG=${3:?tag}
PORT_PT=${4:?port_pt}
PORT_SFT=${5:?port_sft}
HR_PT=${6:?hccl_pt}
HR_SFT=${7:?hccl_sft}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
T="$ROOT/tools/zyc"
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/schedule_${TAG}_2330.log
OUT=$W/schedule_${TAG}_2330.out
LOCK=$W/.schedule_${TAG}_2330.lockdir
TARGET_HUMAN="2026-11-26 23:30:00"

say() { echo "[$(date '+%F %T')] [$TAG] $*" | tee -a "$LOG"; }
die() { say "ABORT: $*"; exit 1; }

host=$(hostname -s)
[ "$host" = "$EXPECTED_NODE" ] || die "wrong host: got $host expected $EXPECTED_NODE"
mkdir -p "$W"
mkdir "$LOCK" 2>/dev/null || die "scheduler lock exists: $LOCK"
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT
printf 'host=%s seed=%s tag=%s started=%s\n' \
  "$host" "$SEED" "$TAG" "$(date '+%F %T %Z')" > "$LOCK/owner"

if [ "${DRY_RUN:-0}" = "1" ]; then
  say "DRY_RUN node=$host seed=$SEED tag=$TAG ports=$PORT_PT/$PORT_SFT hccl=$HR_PT/$HR_SFT"
  exit 0
fi

target=$(date -d "$TARGET_HUMAN" +%s) || die "bad target time"
now=$(date +%s)
[ "$now" -le $((target + 120)) ] || die "started too late: $(date '+%F %T')"
while [ "$now" -lt "$target" ]; do
  sleep 0.2
  now=$(date +%s)
done
say "23:30 reached on $host"

container_state() {
  docker inspect -f '{{.State.Status}}' zyc-w0 2>/dev/null || echo missing
}

npu_pids() {
  npu-smi info 2>/dev/null |
    awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+ +\|/ {print $4}' |
    sort -un
}

occupied() {
  npu-smi info 2>/dev/null |
    awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+/{c++} END{print c+0}'
}

matching_pids() {
  local pattern=$1
  ps -eo pid=,ppid=,args= |
    awk -v re="$pattern" -v self="$$" -v parent="$PPID" '
      $0 ~ /(^|[ ])tmux([ ]|$)/ {next}
      $0 ~ /systemd-run|schedule_dts168/ {next}
      $1 != self && $1 != parent && $0 ~ re {print $1}' |
    sort -un
}

terminate_pids() {
  local label=$1
  shift
  [ "$#" -gt 0 ] || return 0
  say "TERM $label: $*"
  ps -o pid,ppid,user,etime,args -p "$(IFS=,; echo "$*")" | tee -a "$LOG" || true
  kill -TERM "$@" 2>/dev/null || true
  sleep 3
  local alive=()
  local pid
  for pid in "$@"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  if [ "${#alive[@]}" -gt 0 ]; then
    say "KILL $label survivors: ${alive[*]}"
    kill -KILL "${alive[@]}" 2>/dev/null || true
  fi
}

state=$(container_state)
[ "$state" = "running" ] || die "zyc-w0 must already be running; state=$state"
say "container running; it will not be stopped or restarted"

if [ -n "$(npu_pids)" ] || [ "$(occupied)" -ne 0 ]; then
  say "NPU busy at target: pids=$(npu_pids | tr '\n' ' ') occupied=$(occupied)"
  for round in 1 2 3; do
    mapfile -t pids < <(matching_pids \
      'chain_ptlr(_seed)?[.]sh|run_one[.]sh|run_stage1_pretrain(_seed)?[.]sh|run_sft_navsim[.]sh|eval_ckpt[.]sh')
    terminate_pids supervisors "${pids[@]}"

    mapfile -t pids < <(matching_pids \
      'train_qwen_vla[.]py|inference_action_navsim_vava[.]py|run_pdm_score[.]py|torchrun|VLLMWorker|vllm serve|vllm[.]entrypoints|api_server')
    terminate_pids workers "${pids[@]}"

    mapfile -t pids < <(npu_pids)
    terminate_pids npu-owners "${pids[@]}"
    sleep 3

    if [ -z "$(npu_pids)" ] && [ "$(occupied)" -eq 0 ]; then
      say "NPUs cleared after round $round"
      break
    fi
    say "cleanup round $round left pids=$(npu_pids | tr '\n' ' ') occupied=$(occupied)"
  done
fi

[ -z "$(npu_pids)" ] && [ "$(occupied)" -eq 0 ] || die "NPU cleanup failed"
state=$(container_state)
[ "$state" = "running" ] || die "container not running before launch: $state"

cd "$ROOT" || die "checkout missing"
say "launch DTS-168 interval-10 seed=$SEED from $ROOT"
env -u HIF8_DTS_STATIC -u HIF8_DTS_RESEED -u HIF8_DTS_CLOCK_SRC \
  -u HIF8_DTS_HISTORY -u HIF8_DTS_ALGO -u HIF8_DTS_STAT \
  -u HIF8_DTS_STAT_N -u HIF8_DTS_STAT_ROLES -u HIF8_PROBE \
  -u HIF8_DW -u HIF8_ELEM -u HIF8_STRUCT -u HIF8_COH -u HIF8_PROFILE \
  SEED="$SEED" TAG="$TAG" \
  HIF8_PT=dts HIF8_SFT=dts \
  IG_PT=lm_head,o_proj IG_SFT=lm_head,o_proj HIF8_INTERVAL=10 \
  DATALOADER_PIN_MEMORY=False \
  PORT_PT="$PORT_PT" PORT_SFT="$PORT_SFT" HR_PT="$HR_PT" HR_SFT="$HR_SFT" \
  bash "$T/chain_ptlr_seed.sh" >> "$OUT" 2>&1
rc=$?
say "CHAIN_EXIT=$rc"
exit "$rc"

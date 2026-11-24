#!/bin/bash
# Wait for 23:15 CST, clear running training/inference processes by PID, then
# launch one full-chain HiF8 preset. Containers are never stopped or restarted.
set -u
export TZ=Asia/Shanghai

PRESET=${1:?usage: schedule_dts_i200_2315.sh <preset.sh> <tag>}
TAG=${2:?usage: schedule_dts_i200_2315.sh <preset.sh> <tag>}
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
SFT=$W/runs/sft_${TAG}
TARGET_HUMAN=${TARGET_HUMAN:-'2026-11-23 23:15:00'}
LATE_GRACE=${LATE_GRACE:-600}
TARGET_EPOCH=$(date -d "$TARGET_HUMAN" +%s) || exit 1
LOG=$W/schedule_${TAG}_2315.log
OUT=$W/schedule_${TAG}_2315.out
LOCK=$W/.schedule_${TAG}.lock

say() { echo "[$(date '+%F %T')] [$TAG] $*" | tee -a "$LOG"; }

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "scheduler lock already exists: $LOCK" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

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
    awk -v re="$pattern" -v self="$$" -v parent="$PPID" \
      '$0 ~ /(^|[ ])tmux([ ]|$)/ {next}
       $0 ~ /schedule_dts_i200_2315/ {next}
       $1 != self && $1 != parent && $0 ~ re {print $1}' |
    sort -un
}

terminate_pids() {
  local label=$1
  shift
  [ "$#" -gt 0 ] || return 0

  say "TERM $label: $*"
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

clear_jobs() {
  local pids

  # Stop supervisors first so they do not respawn their workers.
  mapfile -t pids < <(matching_pids \
    'chain_ptlr(_seed)?[.]sh|run_one[.]sh|run_stage1_pretrain(_seed)?[.]sh|run_sft_navsim[.]sh|eval_ckpt[.]sh')
  terminate_pids supervisors "${pids[@]}"

  mapfile -t pids < <(matching_pids \
    'train_qwen_vla[.]py|inference_action_navsim_vava[.]py|run_pdm_score[.]py|torchrun|VLLMWorker|vllm serve|vllm[.]entrypoints|api_server')
  terminate_pids workers "${pids[@]}"

  mapfile -t pids < <(npu_pids)
  terminate_pids npu-owners "${pids[@]}"
}

state=$(container_state)
say "scheduled target=$TARGET_HUMAN container=$state"
[ "$state" = running ] || { say "ABORT: zyc-w0 is $state"; exit 1; }

now=$(date +%s)
if [ "$now" -gt $((TARGET_EPOCH + LATE_GRACE)) ]; then
  say "ABORT: scheduler started too late"
  exit 1
fi

while :; do
  now=$(date +%s)
  [ "$now" -ge "$TARGET_EPOCH" ] && break
  sleep 1
done

say "23:15 reached: job processes=$(ps -eo args= | grep -E 'train_qwen_vla|torchrun|VLLMWorker|vllm|run_pdm_score' | grep -vc grep || true) npu=$([ -n "$(npu_pids)" ] && echo busy || echo free)"

for round in 1 2 3; do
  clear_jobs
  sleep 3
  if [ -z "$(npu_pids)" ] && [ "$(occupied)" -eq 0 ]; then
    say "NPU processes cleared after cleanup round $round"
    break
  fi
  say "cleanup round $round left NPU processes: $(npu_pids | tr '\n' ' ')"
done

if [ -n "$(npu_pids)" ] || [ "$(occupied)" -ne 0 ]; then
  say "ABORT: NPU processes survived cleanup: $(npu_pids | tr '\n' ' ')"
  exit 1
fi

state=$(container_state)
say "pre-launch container=$state; starting preset=$PRESET"
[ "$state" = running ] || { say "ABORT: zyc-w0 is $state"; exit 1; }

env \
  -u HIF8_DTS_STATIC \
  -u HIF8_DTS_RESEED \
  -u HIF8_DTS_CLOCK_SRC \
  -u HIF8_DTS_HISTORY \
  -u HIF8_DTS_ALGO \
  -u HIF8_DTS_STAT \
  -u HIF8_DTS_STAT_N \
  -u HIF8_DTS_STAT_ROLES \
  -u HIF8_PROBE \
  -u HIF8_DW \
  -u HIF8_ELEM \
  -u HIF8_STRUCT \
  -u HIF8_COH \
  -u HIF8_PROFILE \
  bash "$PRESET" >> "$OUT" 2>&1
rc=$?
say "preset exited rc=$rc"

json_count() {
  find "$W/navsim_workspace/eval_${TAG}_full/json_output" \
    -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

model_json_count() {
  find "$W/navsim_workspace/eval_${TAG}_full/json_output" \
    -maxdepth 1 -type f -name '*.json' -printf '%f\n' 2>/dev/null |
    grep -vc '_'
}

csv_count() {
  find "$W/navsim_workspace/exp/eval_${TAG}_full" \
    -maxdepth 2 -type f -name '*.csv' 2>/dev/null | wc -l
}

eval_complete() {
  [ "$(model_json_count)" -ge 12146 ] && [ "$(csv_count)" -gt 0 ]
}

if ! eval_complete && [ -s "$SFT/model.safetensors.index.json" ]; then
  say "integrated eval incomplete: total_json=$(json_count) model_json=$(model_json_count) csv=$(csv_count); running explicit full eval"
  docker exec zyc-w0 bash "$T/eval_ckpt.sh" "$SFT" "$TAG" full 8 \
    > "$W/post_eval_${TAG}.log" 2>&1
  eval_rc=$?
  say "explicit eval exited rc=$eval_rc total_json=$(json_count) model_json=$(model_json_count) csv=$(csv_count)"
  [ "$eval_rc" -eq 0 ] || rc=$eval_rc
fi

if eval_complete; then
  say "FINAL_EVAL_OK total_json=$(json_count) model_json=$(model_json_count) csv=$(csv_count)"
else
  say "FINAL_EVAL_FAILED total_json=$(json_count) model_json=$(model_json_count) csv=$(csv_count)"
  [ "$rc" -ne 0 ] || rc=1
fi

exit "$rc"

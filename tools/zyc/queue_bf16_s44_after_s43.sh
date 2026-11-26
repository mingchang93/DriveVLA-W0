#!/bin/bash
# Wait for the existing bf16 seed42/seed43 queue, then run seed44 with the
# same fixed data seed and pair-sampling settings.
set -u
export TZ=Asia/Shanghai

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
QRUN=/mnt/sfs_turbo/lmc/runs
QLOG=/mnt/sfs_turbo/lmc/logs/pt_pair_aligned_bf16_s42_s43_20261126.log
LOG=/mnt/sfs_turbo/lmc/logs/pt_pair_aligned_bf16_s44_after_s43.log
OUT=$QRUN/pt_aligned_bf16_pair_s44
S42=$QRUN/pt_aligned_bf16_pair_s42/train.log
S43=$QRUN/pt_aligned_bf16_pair_s43/train.log
LOCK=/mnt/sfs_turbo/lmc/logs/.pt_pair_bf16_s44.lockdir

say() { echo "[$(date '+%F %T %Z')] [bf16-s44-after-s43] $*"; }
die() { say "ABORT: $*"; exit 1; }

mkdir "$LOCK" 2>/dev/null || die "lock exists: $LOCK"
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT
printf 'host=%s started=%s\n' \
  "$(hostname -s)" "$(date '+%F %T %Z')" > "$LOCK/owner"

say "waiting for existing seed42/43 queue marker: $QLOG"
QLOG_START_LINES=$(wc -l < "$QLOG" 2>/dev/null || echo 0)
[ "$(docker inspect -f '{{.State.Status}}' zyc-w0 2>/dev/null || echo missing)" = running ] ||
  die "zyc-w0 not running"
[ ! -e "$OUT/train.log" ] || die "output already exists: $OUT"

while :; do
  if grep -q 'PAIR_PRETRAIN_QUEUE_DONE' "$QLOG" 2>/dev/null; then
    break
  fi
  if tail -n "+$((QLOG_START_LINES + 1))" "$QLOG" 2>/dev/null |
      grep -Eq 'ARM_RESULT seed=(42|43) TRAIN_EXIT=[1-9]'; then
    die "existing queue reported a failure"
  fi
  if ! tmux has-session -t lmc-a53-pt-pair 2>/dev/null; then
    die "existing queue tmux session disappeared before completion marker"
  fi
  sleep 60
done

grep -q 'TRAIN_EXIT=0' "$S42" || die "seed42 did not finish successfully"
grep -q 'TRAIN_EXIT=0' "$S43" || die "seed43 did not finish successfully"
say "seed42 and seed43 both completed successfully"

npu_pids() {
  npu-smi info 2>/dev/null |
    awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+ +\|/ {print $4}' |
    sort -un
}
while [ -n "$(npu_pids)" ]; do
  say "waiting for NPUs: $(npu_pids | tr '\n' ' ')"
  sleep 60
done

cd "$ROOT" || die "checkout missing"
say "launch bf16 pair seed44 data_seed42"
DATALOADER_PIN_MEMORY=False PAIR_SAMPLING=1 DATA_SEED=42 SEED=44 \
  OUT="$OUT" STEPS=8000 SAVE_STRATEGY=no SAVE_STEPS=1000 \
  NPROC=8 BS=12 GA=1 GC=False PORT=23993 HCCL_RANGE=17900-17999 \
  bash tools/zyc/run_stage1_pretrain_seed.sh >> "$LOG" 2>&1
rc=$?
ex=$(grep -aoE 'TRAIN_EXIT=[0-9]+' "$OUT/train.log" 2>/dev/null | tail -1)
say "seed44 launcher_exit=$rc ${ex:-missing}"
[ "$ex" = "TRAIN_EXIT=0" ] || exit 1

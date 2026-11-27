#!/bin/bash
# Sequential a54 queue:
#   1. zero-shot full navtest evals for three nuPlan pretrain checkpoints
#   2. seed 42/43/44 NAVSIM SFT from pt_ptlr5e5, following the
#      sft_r2_dts168_lr5e5 recipe exactly (DTS-168, interval 10), followed by
#      a full navtest eval for each SFT arm
#
# The queue waits for all eight NPUs to be free. It never kills or shares a
# device with another container.
set -u
export TZ=Asia/Shanghai

W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
PY=/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python
LOG=$W/queue_pretrain_eval_sft_from_ptlr5e5_dts168_i10_a54.log
LOCK=$W/.queue_pretrain_eval_sft_from_ptlr5e5_dts168_i10.lockdir
POLL=${POLL:-300}

PT_BASE=$W/runs/pt_ptlr5e5
PT_S43=$W/runs/pt_bf16_s43
PT_S44=$W/runs/pt_bf16_s44

say() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }
die() { say "!! QUEUE_FAILED: $*"; exit 1; }

occupied() {
  npu-smi info 2>/dev/null |
    awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+/{c++} END{print c+0}'
}

wait_free() {
  local n
  n=$(occupied)
  if [ "$n" -eq 0 ]; then
    say "NPUs free"
    return 0
  fi
  say "waiting for NPUs: $n processes still occupy cards"
  while :; do
    sleep "$POLL"
    n=$(occupied)
    if [ "$n" -eq 0 ]; then
      say "NPUs free"
      return 0
    fi
    say "waiting for NPUs: $n processes still occupy cards"
  done
}

check_ckpt() {
  local dir=$1 label=$2
  for f in config.json model.safetensors.index.json tokenizer.json \
           preprocessor_config.json video_preprocessor_config.json; do
    [ -s "$dir/$f" ] || die "$label missing $f"
  done
  "$PY" "$W/ckcheck.py" "$dir" "$label" | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "$label checkpoint validation failed"
}

score_value() {
  grep -ahoE "Final average score of valid results: [0-9.]+" "$1" 2>/dev/null |
    tail -1 | grep -oE "[0-9.]+$"
}

success_count() {
  grep -ahoE "Number of successful scenarios: [0-9]+" "$1" 2>/dev/null |
    tail -1 | grep -oE "[0-9]+$"
}

run_full_eval() {
  local ckpt=$1 tag=$2
  local eval_dir=$W/navsim_workspace/eval_${tag}_full
  local out=/tmp/queue_eval_${tag}.out
  local retry=/tmp/queue_eval_retry_${tag}.out
  local pdms succ

  [ ! -e "$eval_dir" ] || die "eval directory already exists: $eval_dir"
  wait_free
  check_ckpt "$ckpt" "$tag"
  say ">>> full navtest eval: $tag"
  docker exec zyc-w0 bash "$T/eval_ckpt.sh" "$ckpt" "$tag" full 8 > "$out" 2>&1
  pdms=$(score_value "$out")
  succ=$(success_count "$out")

  if [ -z "$pdms" ] || [ "$succ" != "12146" ]; then
    say "first PDMS pass incomplete (successful=${succ:-missing}); retrying scoring with 64 workers"
    wait_free
    docker exec zyc-w0 bash -c '
      set -e
      cd /mnt/sfs_turbo/zyc/DriveVLA-W0
      source setenv_zyc.sh
      exec /mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python \
        "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
        train_test_split=navtest \
        agent=emu_vla_agent \
        agent.experiment_path="$1" \
        experiment_name="$2" \
        metric_cache_path="$NAVSIM_EXP_ROOT/metric_cache" \
        worker.threads_per_node=64
    ' bash "$eval_dir/json_output" "eval_${tag}_full" > "$retry" 2>&1
    pdms=$(score_value "$retry")
    succ=$(success_count "$retry")
  fi

  [ "$succ" = "12146" ] || die "$tag scored ${succ:-no} successful scenarios"
  [ -n "$pdms" ] || die "$tag produced no PDMS"
  say "<<< eval OK: $tag  successful=$succ  PDMS=$pdms"
}

run_sft() {
  local seed=$1 tag=$2 port=$3 hccl_range=$4
  local sft_dir=$W/runs/sft_${tag}
  local out=/tmp/queue_sft_${tag}.out
  local ex steps q clock refreshes

  [ ! -e "$sft_dir" ] || die "SFT directory already exists: $sft_dir"
  wait_free
  check_ckpt "$PT_BASE" "pt_ptlr5e5"
  say ">>> SFT from pt_ptlr5e5  seed=$seed  DTS-168 interval=10  tag=$tag"
  docker exec zyc-w0 env \
    MODEL="$PT_BASE" \
    STEPS=4000 BS=6 GA=1 GC=False NPROC=8 LOGSTEP=1 SEED="$seed" \
    DATALOADER_PIN_MEMORY=True \
    HIF8=dts HIF8_IGNORE=lm_head,o_proj HIF8_INTERVAL=10 \
    DS=scripts/sft/zero2.json \
    OUT="$sft_dir" PORT="$port" HCCL_RANGE="$hccl_range" \
    bash "$T/run_sft_navsim.sh" > "$out" 2>&1

  ex=$(grep -aoE 'TRAIN_EXIT=[0-9]+' "$sft_dir/train.log" 2>/dev/null | tail -1)
  steps=$(grep -ac 'learning_rate' "$sft_dir/train.log" 2>/dev/null || true)
  q=$(grep -aoE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$sft_dir/train.log" 2>/dev/null | head -1)
  clock=$(grep -aoE "clock_src': 'trainer'" "$sft_dir/train.log" 2>/dev/null | head -1)
  refreshes=$(grep -ac 'points sampled' "$sft_dir/train.log" 2>/dev/null || true)

  say "<<< SFT seed=$seed  exit=$ex  steps=$steps  ${q:-no-quant-count}  clock=${clock:-missing}  refreshes=$refreshes"
  [ "$ex" = "TRAIN_EXIT=0" ] || die "$tag SFT failed; see $sft_dir/train.log"
  [ "$steps" = "4000" ] || die "$tag did not reach 4000 steps"
  [ "$refreshes" -gt 0 ] || die "$tag has no DTS refreshes"

  for f in video_preprocessor_config.json chat_template.json configuration.json preprocessor_config.json; do
    if [ ! -s "$sft_dir/$f" ] && [ -s "$PT_BASE/$f" ]; then
      cp "$PT_BASE/$f" "$sft_dir/$f"
      say "copied $f from pt_ptlr5e5"
    fi
  done
  check_ckpt "$sft_dir" "sft_$tag"
  wait_free
  run_full_eval "$sft_dir" "$tag"
}

mkdir "$LOCK" 2>/dev/null || { echo "queue already active: $LOCK"; exit 1; }
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT
printf 'host=%s pid=%s started=%s\n' "$(hostname -s)" "$$" "$(date '+%F %T %Z')" > "$LOCK/owner"

say "===== QUEUE START host=$(hostname -s) ====="
say "1) zero-shot evals: pt_ptlr5e5, pt_bf16_s43, pt_bf16_s44"
say "2) SFT from pt_ptlr5e5: seeds 42/43/44, exact sft_r2_dts168_lr5e5 recipe"
say "current NPU occupation: $(occupied)"

run_full_eval "$PT_BASE" pt_ptlr5e5
run_full_eval "$PT_S43" pt_bf16_s43
run_full_eval "$PT_S44" pt_bf16_s44

run_sft 42 ptlr5e5_dts168_i10_s42 24642 18642-18652
run_sft 43 ptlr5e5_dts168_i10_s43 24643 18653-18663
run_sft 44 ptlr5e5_dts168_i10_s44 24644 18664-18674

say "===== QUEUE COMPLETE ====="

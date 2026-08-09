#!/usr/bin/env bash
#
# Alignment experiments: 5 tiers × 2 devices
#
# Usage:
#   bash scripts/run_alignment_experiments.sh --device cuda
#   bash scripts/run_alignment_experiments.sh --device npu --tiers 0,2,4
#   bash scripts/run_alignment_experiments.sh --device npu --tiers all \
#       --project_root /path/to/project \
#       --data_root /path/to/data \
#       --model_root /path/to/models
#
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_SCRIPT="$ROOT/scripts/scripts_train/train_base_ar_withou_moe.sh"

# ── Parse ──────────────────────────────────────────────────────────
DEVICE=""
PROJECT_ROOT=""
DATA_ROOT=""
MODEL_ROOT=""
TIERS="all"
FP="fp32"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)       DEVICE="$2";       shift 2 ;;
    --project_root) PROJECT_ROOT="$2"; shift 2 ;;
    --data_root)    DATA_ROOT="$2";    shift 2 ;;
    --model_root)   MODEL_ROOT="$2";   shift 2 ;;
    --tiers)        TIERS="$2";        shift 2 ;;
    --fp)           FP="$2";           shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [ "$DEVICE" != "cuda" ] && [ "$DEVICE" != "npu" ]; then
  echo "Usage: $0 --device cuda|npu [--tiers 0,1,2,3,4|all] [--fp fp32|bf16] [--project_root <path>] [--data_root <path>] [--model_root <path>]"
  exit 1
fi

if [ "$FP" != "fp32" ] && [ "$FP" != "bf16" ]; then
  echo "Invalid --fp value: $FP (must be fp32 or bf16)"
  exit 1
fi

# Resolve tiers to a space-separated list
if [ "$TIERS" = "all" ]; then
  SELECTED_TIERS="-2 -1 0 1 2 3 4 5 6"
else
  SELECTED_TIERS=$(echo "$TIERS" | tr ',' ' ')
fi

# ── Paths: use CLI args if given, otherwise defaults ───────────────
PROJECT_ROOT="${PROJECT_ROOT:-/data/models/DriveVLA-W0}"
DATA_ROOT="${DATA_ROOT:-/data/models/DriveVLA-W0}"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"

MODEL_PATH="$MODEL_ROOT/Emu3-Stage1"
TRAIN_PKL="$DATA_ROOT/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
TEST_PKL="$DATA_ROOT/navsim_emu_vla_256_144_test_pre_1s.pkl"
TRAIN_PKL_FIXED="${TRAIN_PKL%.pkl}_fixed.pkl"
TEST_PKL_FIXED="${TEST_PKL%.pkl}_fixed.pkl"

# ── Fix pickle paths for the local machine ─────────────────────────
echo "=== Fixing pickle paths ==="
python "$ROOT/tools/fix_pickle_paths.py" \
    "$TRAIN_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

python "$ROOT/tools/fix_pickle_paths.py" \
    "$TEST_PKL" \
    --new_prefix "$DATA_ROOT/data/navsim/processed_data"

# Move fixed pickles into place (idempotent)
[ -f "$TRAIN_PKL_FIXED" ] && mv "$TRAIN_PKL_FIXED" "$TRAIN_PKL"
[ -f "$TEST_PKL_FIXED" ] && mv "$TEST_PKL_FIXED" "$TEST_PKL"
echo "=== Pickle paths fixed ==="

# ── Common flags (all experiments) ─────────────────────────────────
COMMON=(
  --model_name_or_path "$MODEL_PATH"
  --data_path "$TRAIN_PKL"
  --test_data_path "$TEST_PKL"
  --ngpus 8
  --logging_steps 1
  --device "$DEVICE"
  --log_data_hash
  --deterministic
  --shuffle_train_data false
  --eval_strategy no
  --eval_steps 10000
  --fp "$FP"
  --save_steps 1000
  --skip_inference
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUT="$ROOT/logs/alignment_${DEVICE}_${FP}_${TIMESTAMP}"

echo "============================================"
echo "Alignment experiments — device: $DEVICE, fp: $FP"
echo "Selected tiers: $SELECTED_TIERS"
echo "Output base: $BASE_OUT"
echo "============================================"

# ── Helper: check if a tier is selected ────────────────────────────
selected() {
  for t in $SELECTED_TIERS; do
    [ "$t" = "$1" ] && return 0
  done
  return 1
}

# ── Tier 0: warmup=200, batch=1, lr=1e-5, max_grad_norm=5.0 ───────
if selected 0; then
  echo ""
  echo "=== Tier 0: warmup=200, batch=1, lr=1e-5, max_grad_norm=5.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier0_warmup200 \
    --max_steps 300 \
    --warmup_steps 200 \
    --batch_size 1 \
    --learning_rate 1e-5
fi

# ── Tier 1: + symmetric clipping (lr=2e-5, clip=2.0) ──────────────
if selected 1; then
  echo ""
  echo "=== Tier 1: warmup=200, batch=1, lr=2e-5, max_grad_norm=2.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier1_clip2_lr2e5 \
    --max_steps 300 \
    --warmup_steps 200 \
    --batch_size 1 \
    --max_grad_norm 2.0 \
    --learning_rate 2e-5
fi

# ── Tier 2: + larger batch (batch=2, lr=2e-5, clip=2.0) ──────────
if selected 2; then
  echo ""
  echo "=== Tier 2: warmup=100, batch=2, lr=2e-5, max_grad_norm=2.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier2_batch2 \
    --max_steps 150 \
    --warmup_steps 100 \
    --batch_size 2 \
    --max_grad_norm 2.0 \
    --learning_rate 2e-5
fi

# ── Tier 3: + adam_epsilon (batch=2, lr=2e-5, clip=2.0, eps=1e-4) ─
if selected 3; then
  echo ""
  echo "=== Tier 3: warmup=100, batch=2, lr=2e-5, clip=2.0, eps=1e-4 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier3_eps1e4 \
    --max_steps 150 \
    --warmup_steps 100 \
    --batch_size 2 \
    --max_grad_norm 2.0 \
    --learning_rate 2e-5 \
    --adam_epsilon 1e-4
fi

# ── Tier 4: Tier 0 smoothness + Tier 2 batch (batch=2, lr=1e-5) ───
if selected 4; then
  echo ""
  echo "=== Tier 4: warmup=100, batch=2, lr=1e-5, max_grad_norm=5.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier4_batch2_lr1e5 \
    --max_steps 150 \
    --warmup_steps 100 \
    --batch_size 2 \
    --learning_rate 1e-5
fi

# ── Tier 5: longer warmup (warmup=300, batch=1, lr=1e-5) ──────────
if selected 5; then
  echo ""
  echo "=== Tier 5: warmup=300, batch=1, lr=1e-5, max_grad_norm=5.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier5_warmup300 \
    --max_steps 400 \
    --warmup_steps 300 \
    --batch_size 1 \
    --learning_rate 1e-5
fi

# ── Tier 6: low LR + large epsilon (batch=2, lr=1e-5, eps=1e-4) ──
if selected 6; then
  echo ""
  echo "=== Tier 6: warmup=100, batch=2, lr=1e-5, eps=1e-4 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier6_lr1e5_eps1e4 \
    --max_steps 150 \
    --warmup_steps 100 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --adam_epsilon 1e-4
fi

# ── Tier 7: Tier 0 but warmup=100 (batch=1, lr=1e-5) ──────────────
if selected 7; then
  echo ""
  echo "=== Tier 7: warmup=100, batch=1, lr=1e-5, max_grad_norm=5.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier7_warmup100 \
    --max_steps 300 \
    --warmup_steps 100 \
    --batch_size 1 \
    --learning_rate 1e-5
fi

# ── Tier 8: Tier 7 but warmup=50 (batch=1, lr=1e-5) ──────────────
if selected 8; then
  echo ""
  echo "=== Tier 8: warmup=50, batch=1, lr=1e-5, max_grad_norm=5.0 ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier8_warmup50 \
    --max_steps 300 \
    --warmup_steps 50 \
    --batch_size 1 \
    --learning_rate 1e-5
fi

# ── Tier 9: Tier 7 + 600 steps + inference ─────────────────────────
if selected 9; then
  echo ""
  echo "=== Tier 9: warmup=100, batch=1, lr=1e-5, max_grad_norm=5.0, 600 steps, inference ==="

  # Filter out --skip_inference from COMMON
  COMMON_INFER=()
  for arg in "${COMMON[@]}"; do
    [ "$arg" != "--skip_inference" ] && COMMON_INFER+=("$arg")
  done

  bash "$TRAIN_SCRIPT" \
    "${COMMON_INFER[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier9_warmup100_600steps \
    --max_steps 600 \
    --warmup_steps 100 \
    --batch_size 1 \
    --learning_rate 1e-5
fi

# ── Tier -1: quick profiling run with msprobe ────────────
if selected -1; then
  echo ""
  echo "=== Tier -1: profiling run, batch=1, lr=1e-5, 100 steps ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier_neg1_profile \
    --max_steps 100 \
    --warmup_steps 0 \
    --batch_size 1 \
    --learning_rate 1e-5

  # Summarize submodule timing with cross-rank variance analysis
  echo ""
  echo "=== Tier -1: summarizing submodule times ==="
  SUBMODULE_DIR="$BASE_OUT/tier_neg1_profile/submodule_times"
  COMBINED="$BASE_OUT/tier_neg1_profile/submodule_times_combined.jsonl"
  python3 "$ROOT/scripts/summarize_submodule_times.py" \
    --submodule_dir "$SUBMODULE_DIR" \
    --combine "$COMBINED"
  echo "Combined data: $COMBINED"
fi

# ── Tier -2: no gradient clipping (batch=1, lr=1e-5, 100 steps) ───
if selected -2; then
  echo ""
  echo "=== Tier -2: gradient clipping disabled, batch=1, lr=1e-5, 100 steps ==="
  bash "$TRAIN_SCRIPT" \
    "${COMMON[@]}" \
    --output_dir "$BASE_OUT" \
    --exp_name tier_neg2_noclip \
    --max_steps 100 \
    --warmup_steps 0 \
    --batch_size 1 \
    --max_grad_norm 0 \
    --learning_rate 1e-5
fi

echo ""
echo "=== Done ($DEVICE) ==="
echo "Results: $BASE_OUT"
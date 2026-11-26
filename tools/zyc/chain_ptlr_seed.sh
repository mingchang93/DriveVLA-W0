#!/bin/bash
# 预训练 lr 扫描：预训练 bf16 @ $LR_PT -> SFT bf16 @ 1e-5 -> full 评测
#
# 唯一变量是预训练 lr。SFT 固定 1e-5。
#
# 1e-5 的依据（三方一致，2026-11-14 从官方 HF 直接核对）：
#   上游 HF 仓库的 Qwen_PDMS_85.3/
#     training_args.bin   learning_rate=1e-05 · wd 0.001 · warmup 200 · ZeRO-3 · ML 8192
#     trainer_state.json  峰值 lr 9.99986e-06 @ step 210 · max_steps 4000 · bs6 · gbs 48
#     output_dir 名       ross_sd15_img_action_qwen25vl_fast_147456_4_000_2va_lr1e-5_unfreezevit
#   该 ckpt 的 config.json：Qwen2_5_VLForConditionalGenerationROSS · enable_ross=True
#     · sd_model_path=stable-diffusion-v1-5/unet，即与我们同族的 ROSS 变体（7B）。
#   注：run_sft_navsim.sh 里「官方实测 lr=2e-4、非 ROSS」那条注释与上述产物不符。
#
# 为什么弃用 2e-4 做预训练：
#   实测 2e-4 显著劣于纯 SFT 基线，八个预训练组合
#   无一例外。2e-4 的唯一出处是官方 Emu3-8B/VQ/6VA 预训练脚本（gbs 96/warmup 400/8k 步），
#   官方从未给出 Qwen 任一阶段的预训练 lr。
#
# 与既有 chain_bf16_full.sh 的差异仅两处：预训练 lr；开中间 ckpt（save_total_limit 5）。
# 其余（数据、GC=False、步数、batch、seed、抽帧默认）全部一致，保证可同表比较。
set -u
export TZ=Asia/Shanghai
# lr 已定死，不再由外部传入（queue 传进来的 LR_PT 会被覆盖）。
# 实测（gbs 96）：5e-5 为峰值；2.5e-5 偏低，1e-4 与 2e-4 明显更差。
# 阶段二 SFT 的 LR=1e-5 是官方 ROSS 版（pintang 镜像）的值，也已写死。
LR_PT=5e-5
SEED=${SEED:-42}
TAG=${TAG:?必须给标签}
PORT_PT=${PORT_PT:-23861}
PORT_SFT=${PORT_SFT:-23871}
HR_PT=${HR_PT:-17300-17399}
HR_SFT=${HR_SFT:-17400-17499}
# ---- HiF8 量化开关（本变体新增）----
# HIF8_PT / HIF8_SFT：空=bf16，dts=delayed tensor scaling，cts=current
# IG_PT / IG_SFT：lm_head = 量化 196 个 Linear；lm_head,o_proj = 168 个
HIF8_PT=${HIF8_PT:-}
HIF8_SFT=${HIF8_SFT:-}
IG_PT=${IG_PT:-lm_head}
IG_SFT=${IG_SFT:-lm_head}
SAVE_STEPS_PT=${SAVE_STEPS_PT:-1000}
HIF8_INTERVAL=${HIF8_INTERVAL:-10}
DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY:-True}
# DTS 的 recipe 开关。默认全空 = 沿用代码默认（延迟缩放 history_len=1 / most_recent）。
# 必须显式写进下面 docker exec 的 env 行，否则不会透进容器、会被静默忽略。
#   HIF8_DTS_STATIC=1                      静态缩放（实测最好，loss 偏差 +0.52%）
#   HIF8_DTS_HISTORY=16 HIF8_DTS_ALGO=max  跟随 amax 真实衰减、滤掉单步极值噪声
HIF8_DTS_STATIC=${HIF8_DTS_STATIC:-}
HIF8_DTS_HISTORY=${HIF8_DTS_HISTORY:-}
HIF8_DTS_ALGO=${HIF8_DTS_ALGO:-}
#   HIF8_DTS_STAT=p99                      scale 用非零元素的 p99 而不是 amax
HIF8_DTS_STAT=${HIF8_DTS_STAT:-}
HIF8_DTS_STAT_N=${HIF8_DTS_STAT_N:-}
HIF8_DTS_STAT_ROLES=${HIF8_DTS_STAT_ROLES:-}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
T="$ROOT/tools/zyc"
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
PT=$W/runs/pt_${TAG}
SFT=$W/runs/sft_${TAG}
LOG=$W/chain_${TAG}.log
PY=/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python3

say() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }
die() { say "!! 链条中断: $*"; say "CHAIN_${TAG}_FAILED"; exit 1; }
occupied() { npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+/{c++} END{print c+0}'; }
hbm_max()  { npu-smi info 2>/dev/null | grep -oE '[0-9]+ *\/ *98304' | grep -oE '^[0-9]+' | sort -rn | head -1; }
wait_free(){ local w=0; while [ "$(occupied)" -ne 0 ] && [ "$w" -lt 60 ]; do sleep 5; w=$((w+1)); done; say "    卡已释放（等 $((w*5))s）"; }
# 步数用 learning_rate 计数 —— epoch 在我们打的行和 HF 打的行里都出现，数它会双计
nstep(){ tr -d '\r' < "$1/train.log" 2>/dev/null | grep -c "learning_rate"; }

check_ckpt(){
  local d=$1 name=$2
  for f in config.json model.safetensors.index.json tokenizer.json preprocessor_config.json video_preprocessor_config.json; do
    [ -s "$d/$f" ] || die "$name 缺 $f"
  done
  "$PY" "$W/ckcheck.py" "$d" "$name" | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "$name ckpt 校验未通过"
}

n=$(occupied)
say "===== 预训练 lr=$LR_PT 全链条：预训练 8000 -> SFT 4000(lr 1e-5) -> full 评测 ====="
say "开跑前占卡 $n 个，HBM 最高 $(hbm_max)"
if [ "$n" -ne 0 ]; then
  npu-smi info 2>/dev/null | awk '/Process id/{f=1} f' | head -12 | tee -a "$LOG"
  die "有 $n 个进程占卡（不抢别人的卡）"
fi
say "配置：预训练 lr=$LR_PT · SFT lr=1e-5(官方值) · GC=False 两阶段统一 · SAVE_STEPS=$SAVE_STEPS_PT"
say "量化：预训练 HIF8=[${HIF8_PT:-bf16}] ignore=[$IG_PT] · SFT HIF8=[${HIF8_SFT:-bf16}] ignore=[$IG_SFT] · interval=$HIF8_INTERVAL"
say "数据：官方 pickle 18,191 段；抽帧用默认（原实现）"
say "预计约 9.2 小时"

rm -f "$W/chain_${TAG}_hbm.txt"
# 坑 #6：后台采样进程必须重定向 stdout，否则 $(...) 永不返回
( while true; do hbm_max >> "$W/chain_${TAG}_hbm.txt"; sleep 120; done ) >/dev/null 2>&1 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

# ---------- 阶段一：预训练 ----------
rm -rf "$PT"
say ">>> 阶段一 预训练 8000 步（lr $LR_PT，mb12x8xga1=96）"
( for mark in 50 400 1000 2000 4000 6000; do
    while :; do
      [ "$(nstep "$PT")" -ge "$mark" ] && break
      grep -q TRAIN_EXIT "$PT/train.log" 2>/dev/null && exit 0
      sleep 120
    done
    al=$(tr -d '\r' < "$PT/train.log" | grep -oE "action_loss.: [0-9.]+" | tail -1 | grep -oE '[0-9.]+')
    gn=$(tr -d '\r' < "$PT/train.log" | grep -oE "grad_norm.: [0-9.e+-]+" | tail -1 | grep -oE '[0-9.e+-]+$')
    lr=$(tr -d '\r' < "$PT/train.log" | grep -oE "learning_rate.: [0-9.e+-]+" | tail -1 | grep -oE '[0-9.e+-]+$')
    echo "[$(date +'%F %T')]    预训练 $mark 步: action_loss=$al grad_norm=$gn lr=$lr" >> "$LOG"
  done ) >/dev/null 2>&1 &
P1=$!
docker exec zyc-w0 env \
  MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  STEPS=8000 BS=12 GA=1 GC=False NPROC=8 LOGSTEP=1 SEED="$SEED" \
  DATALOADER_PIN_MEMORY="$DATALOADER_PIN_MEMORY" \
  LR="$LR_PT" SAVE_STRATEGY=steps SAVE_STEPS="$SAVE_STEPS_PT" \
  HIF8="$HIF8_PT" HIF8_IGNORE="$IG_PT" HIF8_INTERVAL="$HIF8_INTERVAL" \
  HIF8_DTS_STATIC="$HIF8_DTS_STATIC" HIF8_DTS_HISTORY="$HIF8_DTS_HISTORY" HIF8_DTS_ALGO="$HIF8_DTS_ALGO" \
  HIF8_DTS_STAT="$HIF8_DTS_STAT" HIF8_DTS_STAT_N="$HIF8_DTS_STAT_N" HIF8_DTS_STAT_ROLES="$HIF8_DTS_STAT_ROLES" \
  OUT="$PT" PORT="$PORT_PT" HCCL_RANGE="$HR_PT" \
  bash "$T/run_stage1_pretrain_seed.sh" > /tmp/chain_pt_${TAG}.out 2>&1
kill "$P1" 2>/dev/null
ex=$(tr -d '\r' < "$PT/train.log" 2>/dev/null | grep -oE 'TRAIN_EXIT=[0-9]+' | tail -1)
rt=$(tr -d '\r' < "$PT/train.log" 2>/dev/null | grep -oE "train_runtime.: [0-9.]+" | tail -1 | grep -oE '[0-9.]+')
say "<<< 阶段一 $ex   $(nstep "$PT") 步   runtime ${rt:-?}s"
[ "$ex" = "TRAIN_EXIT=0" ] || die "阶段一退出码不是 0，看 $PT/train.log"
pk=$(tr -d '\r' < "$PT/train.log" | grep -oE "learning_rate.: [0-9.e+-]+" | grep -oE '[0-9.e+-]+$' | sort -g | tail -1)
say "    实际峰值 lr=$pk（期望约 $LR_PT）"
say "    补的 processor 配置: $(tr -d '\r' < "$PT/train.log" | grep -c '补 processor 配置') 个"
check_ckpt "$PT" "阶段一 ckpt"
wait_free

# ---------- 阶段二：SFT ----------
rm -rf "$SFT"
say ">>> 阶段二 SFT 4000 步（lr 1e-5，bs6，起始权重=阶段一输出）"
( for mark in 50 500 1000 2000 3000; do
    while :; do
      [ "$(nstep "$SFT")" -ge "$mark" ] && break
      grep -q TRAIN_EXIT "$SFT/train.log" 2>/dev/null && exit 0
      sleep 120
    done
    al=$(tr -d '\r' < "$SFT/train.log" | grep -oE "action_loss.: [0-9.]+" | tail -1 | grep -oE '[0-9.]+')
    gn=$(tr -d '\r' < "$SFT/train.log" | grep -oE "grad_norm.: [0-9.e+-]+" | tail -1 | grep -oE '[0-9.e+-]+$')
    echo "[$(date +'%F %T')]    SFT $mark 步: action_loss=$al grad_norm=$gn" >> "$LOG"
  done ) >/dev/null 2>&1 &
P2=$!
docker exec zyc-w0 env \
  MODEL="$PT" \
  STEPS=4000 BS=6 GA=1 GC=False NPROC=8 LOGSTEP=1 SEED="$SEED" \
  DATALOADER_PIN_MEMORY="$DATALOADER_PIN_MEMORY" \
  LR=1e-5 \
  HIF8="$HIF8_SFT" HIF8_IGNORE="$IG_SFT" HIF8_INTERVAL="$HIF8_INTERVAL" \
  HIF8_DTS_STATIC="$HIF8_DTS_STATIC" HIF8_DTS_HISTORY="$HIF8_DTS_HISTORY" HIF8_DTS_ALGO="$HIF8_DTS_ALGO" \
  HIF8_DTS_STAT="$HIF8_DTS_STAT" HIF8_DTS_STAT_N="$HIF8_DTS_STAT_N" HIF8_DTS_STAT_ROLES="$HIF8_DTS_STAT_ROLES" \
  DS=scripts/sft/zero2.json \
  OUT="$SFT" PORT="$PORT_SFT" HCCL_RANGE="$HR_SFT" \
  bash "$T/run_sft_navsim.sh" > /tmp/chain_sft_${TAG}.out 2>&1
kill "$P2" 2>/dev/null
ex=$(tr -d '\r' < "$SFT/train.log" 2>/dev/null | grep -oE 'TRAIN_EXIT=[0-9]+' | tail -1)
say "<<< 阶段二 $ex   $(nstep "$SFT") 步"
say "    起始权重确认: $(tr -d '\r' < "$SFT/train.log" | grep -oE "pt_${TAG}" | head -1)"
[ "$ex" = "TRAIN_EXIT=0" ] || die "阶段二退出码不是 0，看 $SFT/train.log"
# run_sft_navsim.sh 不补 processor 配置 —— chain_bf16 的阶段二就是死在这一步，
# 它训练成功（TRAIN_EXIT=0，4000 步）但缺 video_preprocessor_config.json 被判失败。
for f in video_preprocessor_config.json chat_template.json configuration.json preprocessor_config.json; do
  if [ ! -s "$SFT/$f" ] && [ -s "$PT/$f" ]; then
    cp "$PT/$f" "$SFT/$f" && say "    从阶段一补 $f"
  fi
done
check_ckpt "$SFT" "阶段二 ckpt"
wait_free

# ---------- 阶段三：full 评测 ----------
say ">>> 阶段三 full 评测（navtest 12,146 场景）"
# 必须进容器：eval_ckpt.sh 的 torchrun 来自 lmc 的 venv，其 python3 在宿主机上是断链
docker exec zyc-w0 bash "$T/eval_ckpt.sh" "$SFT" "$TAG" full 8 > /tmp/chain_eval_${TAG}.out 2>&1
rc=$?
# 判据用实际分数，不看退出码 —— eval_ckpt.sh 末尾的 echo EVALDONE_ 无条件执行
pdms=$(grep -ahoE "Final average score of valid results: [0-9.]+" /tmp/chain_eval_${TAG}.out 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
nj=$(ls -1 "$W/navsim_workspace/eval_${TAG}_full/json_output" 2>/dev/null | wc -l)
say "<<< 阶段三 退出码 $rc  推理产出 $nj 个 json  PDMS=${pdms:-未出分}"
if [ -z "$pdms" ]; then
  tail -25 /tmp/chain_eval_${TAG}.out 2>/dev/null | sed 's/^/      /' | tee -a "$LOG"
  die "阶段三没出分"
fi

say "===== 链条完成 ====="
"$PY" - "$pdms" "$LR_PT" <<'PYEOF' | tee -a "$LOG"
import os, sys
v=float(sys.argv[1]); lr=sys.argv[2]
spec=os.environ.get("PDMS_REFS","")
# 基线不写死在脚本里。要自动对比就传 PDMS_REFS="名称=数值;名称=数值"
ref=[]
for item in spec.split(";"):
    if "=" in item:
        n,_,b=item.partition("=")
        try: ref.append((n.strip(), float(b)))
        except ValueError: pass
print("  本把 预训练 lr=%s + SFT lr1e-5   PDMS = %.6f" % (lr, v))
print()
for name,b in ref:
    print("  vs %-36s %.6f   差 %+.6f (%+.2f%%)" % (name, b, v-b, 100*(v-b)/b))
print()
if not ref:
    print("  （未提供基线；设 PDMS_REFS=\"名称=数值;...\" 可自动对比）")
PYEOF
say "  HBM 峰值 $(sort -rn "$W/chain_${TAG}_hbm.txt" 2>/dev/null | head -1) / 98304 MB"
say "  阶段一 $PT"
say "  阶段二 $SFT"
wait_free
say "CHAIN_${TAG}_DONE"

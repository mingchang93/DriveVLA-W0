#!/bin/bash
# 从指定的预训练权重起，跑 SFT（可配量化）+ full 评测。即链条的阶段二、三。
#
# 为什么不跑完整链条：bf16 @ lr 5e-5 的预训练产物 pt_ptlr5e5 已在盘上，
# 而且 bf16 基线用的正是这份权重。复用它意味着
# **起始权重完全相同、只有 SFT 的量化这一个变量**，比重跑预训练更干净，
# 还省 5.5 小时。（确定性全开 + 同 seed 下重跑预训练本应逐位一致，重跑是纯浪费。）
#
# 用法（全部通过环境变量）：
#   PT_DIR   起始权重目录名（runs/ 下）
#   TAG      本次标签
#   HIF8_SFT 空=bf16 / dts / cts
#   IG_SFT   lm_head = 量化 196 个 Linear；lm_head,o_proj = 168 个
#   PORT_SFT / HR_SFT / HIF8_INTERVAL 可选
set -u
export TZ=Asia/Shanghai
PT_DIR=${PT_DIR:?必须给起始权重目录名}
TAG=${TAG:?必须给标签}
HIF8_SFT=${HIF8_SFT:-}
IG_SFT=${IG_SFT:-lm_head}
HIF8_INTERVAL=${HIF8_INTERVAL:-10}
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
PORT_SFT=${PORT_SFT:-24601}
HR_SFT=${HR_SFT:-18500-18599}

T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
PT=$W/runs/$PT_DIR
SFT=$W/runs/sft_${TAG}
LOG=$W/sftpt_${TAG}.log
PY=/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python3

say(){ echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }
die(){ say "!! 中断: $*"; say "SFTPT_${TAG}_FAILED"; exit 1; }
occ(){ npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f && /^\| [0-9]+ +\| +[0-9]+/{c++} END{print c+0}'; }
# 步数用 learning_rate 计数 —— 'epoch' 在我们打的行和 HF 打的行里都出现，数它会双计
nstep(){ tr -d '\r' < "$1/train.log" 2>/dev/null | grep -c "learning_rate"; }

say "===== SFT（起始权重=$PT_DIR）+ full 评测 ====="
say "  量化 HIF8=[${HIF8_SFT:-bf16}] ignore=[$IG_SFT] interval=$HIF8_INTERVAL · SFT lr 1e-5（脚本内写死）"

n=$(occ)
say "开跑前占卡 $n 个"
if [ "$n" -ne 0 ]; then
  npu-smi info 2>/dev/null | awk '/Process id/{f=1} f' | head -10 | tee -a "$LOG"
  die "有 $n 个进程占卡（不抢别人的卡）"
fi

for f in config.json model.safetensors.index.json tokenizer.json preprocessor_config.json video_preprocessor_config.json; do
  [ -s "$PT/$f" ] || die "起始权重缺 $f"
done
"$PY" "$W/ckcheck.py" "$PT" "起始权重 $PT_DIR" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "起始权重校验未通过"

rm -rf "$SFT"
say ">>> SFT 4000 步（lr 1e-5，bs6，GC=False，ZeRO-2）"
docker exec zyc-w0 env \
  MODEL="$PT" \
  STEPS=4000 BS=6 GA=1 GC=False NPROC=8 LOGSTEP=1 SEED=42 \
  HIF8="$HIF8_SFT" HIF8_IGNORE="$IG_SFT" HIF8_INTERVAL="$HIF8_INTERVAL" \
  HIF8_DTS_STATIC="$HIF8_DTS_STATIC" HIF8_DTS_HISTORY="$HIF8_DTS_HISTORY" HIF8_DTS_ALGO="$HIF8_DTS_ALGO" \
  HIF8_DTS_STAT="$HIF8_DTS_STAT" HIF8_DTS_STAT_N="$HIF8_DTS_STAT_N" HIF8_DTS_STAT_ROLES="$HIF8_DTS_STAT_ROLES" \
  DS=scripts/sft/zero2.json \
  OUT="$SFT" PORT="$PORT_SFT" HCCL_RANGE="$HR_SFT" \
  bash "$T/run_sft_navsim.sh" > "/tmp/sftpt_${TAG}.out" 2>&1
ex=$(tr -d '\r' < "$SFT/train.log" 2>/dev/null | grep -oE 'TRAIN_EXIT=[0-9]+' | tail -1)
q=$(tr -d '\r' < "$SFT/train.log" 2>/dev/null | grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' | head -1)
say "<<< SFT $ex   $(nstep "$SFT") 步   ${q:-（bf16，无量化）}"
[ "$ex" = "TRAIN_EXIT=0" ] || die "SFT 退出码不是 0，看 $SFT/train.log"

# run_sft_navsim.sh 不补 processor 配置 —— 链条脚本的 ckpt 门槛会因此判失败
for f in video_preprocessor_config.json chat_template.json configuration.json preprocessor_config.json; do
  if [ ! -s "$SFT/$f" ] && [ -s "$PT/$f" ]; then cp "$PT/$f" "$SFT/$f" && say "    从起始权重补 $f"; fi
done
"$PY" "$W/ckcheck.py" "$SFT" "$TAG ckpt" | tee -a "$LOG"
w=0; while [ "$(occ)" -ne 0 ] && [ "$w" -lt 60 ]; do sleep 5; w=$((w+1)); done
say "    卡已释放"

say ">>> full 评测（navtest 12,146 场景）"
# 必须进容器：eval_ckpt.sh 的 torchrun 来自 lmc 的 venv，其 python3 在宿主机上是断链
docker exec zyc-w0 bash "$T/eval_ckpt.sh" "$SFT" "$TAG" full 8 > "/tmp/eval_${TAG}.out" 2>&1
# 判据用实际分数，不看退出码 —— eval_ckpt.sh 末尾的 echo 是无条件执行的
pdms=$(grep -ahoE "Final average score of valid results: [0-9.]+" "/tmp/eval_${TAG}.out" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
nj=$(ls -1 "$W/navsim_workspace/eval_${TAG}_full/json_output" 2>/dev/null | wc -l)
say "<<< 评测  推理产出 $nj 个 json  PDMS=${pdms:-未出分}"
[ -z "$pdms" ] && { tail -25 "/tmp/eval_${TAG}.out" | sed 's/^/      /' | tee -a "$LOG"; die "没出分"; }

say "----- 判读 -----"
"$PY" - "$pdms" <<'PYEOF' | tee -a "$LOG"
import os, sys
v = float(sys.argv[1])
# 基线不写死在脚本里。要自动对比就传 PDMS_REFS="名称=数值;名称=数值"
refs = []
for item in os.environ.get("PDMS_REFS", "").split(";"):
    if "=" in item:
        n, _, b = item.partition("=")
        try:
            refs.append((n.strip(), float(b)))
        except ValueError:
            pass
print("  本把 bf16 预训练(5e-5) + 量化 SFT   PDMS = %.6f" % v)
print()
for name, b in refs:
    print("  vs %-42s %.6f   差 %+.6f (%+.2f%%)" % (name, b, v - b, 100 * (v - b) / b))
print()
print("  这一把隔离的是「只在 SFT 用量化」：起始权重与 bf16 那把完全相同，")
print("  所以与 bf16 基线的差就是 SFT 阶段量化的全部代价；")
print("  再与全程量化那把比，可看出预训练阶段的量化是否另有代价。")
if not refs:
    print("  （未提供基线；设 PDMS_REFS=\"名称=数值;...\" 可自动对比）")
PYEOF
w=0; while [ "$(occ)" -ne 0 ] && [ "$w" -lt 60 ]; do sleep 5; w=$((w+1)); done
say "SFTPT_${TAG}_DONE"

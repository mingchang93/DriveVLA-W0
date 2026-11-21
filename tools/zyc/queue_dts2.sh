#!/bin/bash
# 等本机评测结束 -> 跑重写后的 DTS 20 步冒烟 -> 过了再跑 2B hif8_196 DTS 4000 步。
#   用法: WAIT_EVAL=aligned_hif8_168 bash queue_dts2.sh
#
# 为什么要先冒烟：DTS 刚按参考实现重写过（换成 npu_quant_max 采样、窗口末尾采样
# 下窗口生效、跨 rank MAX、scale 下界钳位、optimizer post-step hook 做时钟），
# 这五处都没在真模型上跑过。20 步约 2 分钟，比赔掉一整把 4000 步便宜得多。
#
# 冒烟要核三件事（不只是「没崩」）：
#   1) TRAIN_EXIT=0
#   2) 量化层数 196（ignore=lm_head）
#   3) state() 里 interval=10 / sample_last=1 / history_len=1 / most_recent，
#      且 points = 196 x 3 = 588（每个 Linear 三个点：x、w、grad）
#      —— 点数对不上说明量化点没按预期注册，那 DTS 的调度就是空转
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
WAIT_EVAL=${WAIT_EVAL:?}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/queue_dts2.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "===== 等 $WAIT_EVAL 评完 -> DTS 冒烟 -> 2B hif8_196 DTS 4000 步 ====="
tick=0
while ! grep -q "EVALDONE_$WAIT_EVAL" "$W/eval_$WAIT_EVAL.log" 2>/dev/null; do
  tick=$((tick+1))
  [ $((tick % 15)) -eq 1 ] && say "   等评测：$(tail -1 "$W/eval_$WAIT_EVAL.log" 2>/dev/null | cut -c1-100)"
  sleep 60
  [ "$tick" -gt 300 ] && { say "!! 等 5 小时未完，放弃"; exit 1; }
done
say ">>> $WAIT_EVAL 评完，等卡释放"
for i in $(seq 1 45); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

# ---- 冒烟 ----
say ">>> DTS 冒烟（2B，20 步，interval=10，ignore=lm_head -> 应 196 层）"
docker exec zyc-w0 env TAG=dts2_smoke HIF8=dts IGN=lm_head HIF8_INTERVAL=10 STEPS=20 \
  MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  bash "$T/run_one.sh"
L=$R/dts2_smoke/train.log
ex=$(tr -d '\r' < "$L" 2>/dev/null | grep -oE 'TRAIN_EXIT=[0-9]+' | tail -1)
qn=$(tr -d '\r' < "$L" 2>/dev/null | grep -oE '量化 [0-9]+ 个 Linear' | head -1)
st=$(tr -d '\r' < "$L" 2>/dev/null | grep -oE '\[HiF8-DTS\] .*' | head -1)
say "<<< 冒烟 $ex   $qn"
say "    state: $st"
say "    首步 $(tr -d '\r' < "$L" 2>/dev/null | grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' | head -1)"
say "    末步 $(tr -d '\r' < "$L" 2>/dev/null | grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' | tail -1)"

if [ "$ex" != "TRAIN_EXIT=0" ]; then
  say "!!! 冒烟没过，不起 4000 步。前几条错误："
  docker exec zyc-w0 bash -c "grep -aE 'RuntimeError|ValueError|AssertionError|TypeError|AttributeError|aclnn' /tmp/one_dts2_smoke.log | head -8" | sed 's/^/      /' >> "$LOG"
  say "QUEUEDTS2ABORT"
  exit 1
fi
if ! echo "$st" | grep -q "'points': 588"; then
  say "!!! state 里的量化点数不是 588（196 层 x 3 个角色），调度可能在空转 —— 不起 4000 步"
  say "QUEUEDTS2ABORT"
  exit 1
fi
say "    冒烟通过且点数正确，起 4000 步"

sleep 30
for i in $(seq 1 30); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

# ---- 正式跑 ----
say ">>> 2B hif8_196 DTS 4000 步（interval=10）"
docker exec -d zyc-w0 env TAG=aligned_hif8_196_dts HIF8=dts IGN=lm_head HIF8_INTERVAL=10 \
  MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  setsid bash "$T/run_one.sh"
sleep 150
say "    启动后训练进程 $(docker exec zyc-w0 bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"')"
say "    $(tail -1 "$W/one_aligned_hif8_196_dts.log" 2>/dev/null)"
say "QUEUEDTS2DONE"

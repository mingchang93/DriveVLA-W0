#!/bin/bash
# a54 接棒：等 2B hif8_168 跑完 -> 先用 20 步验 HiF8-DTS 走不走得通 -> 再跑 7B hif8_132。
#
# 为什么要先插一个 20 步的 DTS 冒烟：a51/a52 的第二棒都是 DTS（约 06:15 接棒），
# DTS 是今天新写的代码，没在真模型上跑过。a54 约 05:11 先空出来，
# 用 2 分钟在这里验一遍，比到时候直接赔掉一整把 4000 步划算。
# 冒烟只看三件事：能跑、量化层数对、刷新间隔确实在生效（看 scale 刷新次数）。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
R=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/queue_a54.log
MACH=$(hostname -s 2>/dev/null || echo unknown)
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "===== a54 接棒器：等 aligned_hif8_168 -> DTS 冒烟 -> 7B hif8_132 ====="
tick=0
while true; do
  ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/aligned_hif8_168/train.log" 2>/dev/null | tail -1)
  [ -n "$ex" ] && break
  tick=$((tick+1))
  [ $((tick % 20)) -eq 1 ] && say "等待中：aligned_hif8_168 $(tail -1 "$R/aligned_hif8_168/train.log" 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+/4000' | tail -1)"
  sleep 60
done
say ">>> aligned_hif8_168 结束（$ex），等卡释放"
for i in $(seq 1 30); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

# ---- 1) DTS 冒烟：20 步，2B，间隔 10 ----
say ">>> DTS 冒烟（2B，20 步，interval=10）"
docker exec zyc-w0 env TAG=dts_smoke HIF8=dts IGN=lm_head,o_proj HIF8_INTERVAL=10 STEPS=20 \
  MODEL=/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  bash "$T/run_one.sh"
smoke_ex=$(grep -oE 'TRAIN_EXIT=[0-9]+' "$R/dts_smoke/train.log" 2>/dev/null | tail -1)
say "<<< DTS 冒烟 $smoke_ex  $(grep -oE '\[HiF8\] 模式[^）]*）?[^「]*' "$R/dts_smoke/train.log" 2>/dev/null | head -1)"
say "    量化 $(grep -oE '量化 [0-9]+ 个 Linear（未量化 [0-9]+ 个）' "$R/dts_smoke/train.log" 2>/dev/null | head -1)"
say "    首步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$R/dts_smoke/train.log" 2>/dev/null | head -1)"
say "    末步 $(grep -oE '\{.action_loss.: [0-9.]+, .ross_loss.: [0-9.]+' "$R/dts_smoke/train.log" 2>/dev/null | tail -1)"
if [ "$smoke_ex" != "TRAIN_EXIT=0" ]; then
  say "!!! DTS 冒烟没过，前几条错误："
  grep -nE 'RuntimeError|ValueError|AssertionError|TypeError|AttributeError' /tmp/one_dts_smoke.log 2>/dev/null | head -6 | sed 's/^/      /' >> "$LOG"
  say "    a51/a52 的 DTS 第二棒不要起——先修。"
else
  say "    DTS 可用，a51/a52 的第二棒可以照常接。"
fi
sleep 30
for i in $(seq 1 30); do
  n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
  [ "${n:-9}" = "0" ] && break
  sleep 20
done

# ---- 2) 7B hif8_132 ----
say ">>> 7B hif8_132（4000 步）"
docker exec -d zyc-w0 env TAG=w7b_hif8_132 HIF8=cts \
  IGN=lm_head,o_proj,layers.0.,layers.1.,layers.2.,layers.25.,layers.26.,layers.27. \
  MODEL=/mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct \
  setsid bash "$T/run_one.sh"
sleep 150
say "    启动后训练进程 $(docker exec zyc-w0 bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"')  日志：$(tail -1 "$W/one_w7b_hif8_132.log" 2>/dev/null)"
say "QUEUEA54DONE"

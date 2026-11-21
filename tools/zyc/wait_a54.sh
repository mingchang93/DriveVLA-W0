#!/bin/bash
# 守候 a54：每 5 分钟看一次卡，空了就起 7B 链（冒烟 -> 确定性 -> lr1e-5 评测）。
#
# 为什么要连续两次判空才动手：同事的任务在阶段切换时会短暂放开卡，
# 单次采样判空容易抢到别人正要继续用的卡。两次间隔 5 分钟，代价只是最多多等 5 分钟。
#
# 判空口径：npu-smi 的进程行数为 0。显存不用作主判据——进程退出后显存回收有延迟，
# 空闲基线本身也有约 38 G（4.7 G/卡）。
#
# 日志一律用 >> 直写文件，不走 tee：这个脚本是 setsid 后台起的，
# stdout 的 fd 会失效，tee 一死连文件那份也写不成（之前踩过三次）。
set -u
export TZ=Asia/Shanghai
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/wait_a54.log
CHAIN=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc/chain7b2.sh
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

nproc_on_card() { npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}'; }
who_on_card() {
  npu-smi info 2>/dev/null | grep -oE '^\| [0-9]+ +\| +[0-9]{4,}' | awk '{print $4}' \
    | sort -u | head -1 | while read -r p; do ps -o args= -p "$p" 2>/dev/null | cut -c1-70; done
}

say "===== 守候 a54 开始（每 5 分钟一次，连续两次判空才起）====="
free_streak=0
tick=0
while true; do
  n=$(nproc_on_card)
  tick=$((tick+1))
  if [ "${n:-9}" = "0" ]; then
    free_streak=$((free_streak+1))
    say "第 $tick 次：卡上 0 进程（连续判空 $free_streak/2）"
    if [ "$free_streak" -ge 2 ]; then
      say ">>> a54 已空，起 7B 链（ZeRO-2，按你的选择不换 ZeRO-3）"
      docker exec -d zyc-w0 setsid bash "$CHAIN"
      sleep 60
      say "    链状态：$(tail -1 $W/chain7b.log 2>/dev/null)"
      say "WAITA54DONE 已交棒给 chain7b2"
      exit 0
    fi
  else
    [ "$free_streak" -gt 0 ] && say "第 $tick 次：又被占上了，判空计数清零"
    free_streak=0
    # 每半小时报一次占用者，免得日志太吵
    if [ $((tick % 6)) -eq 1 ]; then
      say "第 $tick 次：$n 个进程占卡 — $(who_on_card)"
    fi
  fi
  sleep 300
done

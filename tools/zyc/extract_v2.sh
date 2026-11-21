#!/bin/bash
# 带验证的解压，校验与下一个包的解压重叠执行。
#   用法: MACH=a51 bash extract_v2.sh "5 6"
#
# 为什么要验证：`tar -xf` 返回 0 只保证归档结构完整、写入没报错；普通 tar
# 没有内容校验和，不保证落盘内容逐字节正确。`tar -df` 重读归档逐成员比对
# 内容/大小/时间戳，是真正的内容校验，但要再读一遍（约 40 分钟/包）。
#
# 重叠的做法：解完第 N 个就把它的校验丢到后台，立刻开始解第 N+1 个。
# 校验是只读、解压是读+写，IO 画像不同；且校验不额外占磁盘。
# 限制最多 1 个校验在飞（开下一个前先 wait 上一个），避免堆积。
# 代价：同一时刻盘上会多留一个待校验的归档，比纯串行多占约 110 GB。
#
# 包放在 _hold（我把还没开始的从 _tmp 挪过去了），这样旧的 extract2.sh
# 找不到文件会跳过并正常退出，不用杀进程、不浪费已完成的工作。
set -u
export TZ=Asia/Shanghai
HOLD=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_hold
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_v_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" | tee -a "$LOG"; }

verify_and_delete() {   # $1 = 包编号
  local i=$1 f=$HOLD/nuplan-v1.1_train_camera_${i}.zip
  local t0=$(date +%s)
  say "    [校验] camera_$i 开始 tar -df"
  local out rc nd
  out=$(cd "$DST" && tar -df "$f" --strip-components=1 2>&1); rc=$?
  nd=$(echo "$out" | grep -c .)
  local mins=$(( ($(date +%s)-t0)/60 ))
  if [ $rc -eq 0 ] && [ "$nd" = "0" ]; then
    say "    [校验] camera_$i 通过（${mins} 分钟，零差异）"
    rm -f "$f" && say "    [校验] camera_$i 已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  else
    say "    [校验] camera_$i 不通过 rc=$rc 差异 $nd 处，包保留。前几处："
    echo "$out" | head -5 | sed 's/^/        /' | tee -a "$LOG"
  fi
}

say "挂起：等本机 extract2.sh 退出"
while pgrep -f 'extract2\.sh' > /dev/null 2>&1; do sleep 60; done
while pgrep -f 'tar -xf .*camera' > /dev/null 2>&1; do sleep 60; done
say "旧队列已结束，开始（解压与校验重叠）。队列: $1"

vpid=""
for i in $1; do
  f=$HOLD/nuplan-v1.1_train_camera_${i}.zip
  if [ ! -f "$f" ]; then say "camera_$i 不在 _hold，跳过"; continue; fi
  say ">>> camera_$i ($(du -h "$f" | cut -f1)) 解压"
  t0=$(date +%s)
  tar -xf "$f" --strip-components=1 -C "$DST" 2>>"$LOG"
  rc=$?
  if [ $rc -ne 0 ]; then say "!!! camera_$i 解压失败 rc=$rc，包保留，跳过"; continue; fi
  say "<<< camera_$i 解压完成，用时 $(( ($(date +%s)-t0)/60 )) 分钟；校验转后台，立刻解下一个"

  # 最多 1 个校验在飞
  [ -n "$vpid" ] && wait "$vpid" 2>/dev/null
  verify_and_delete "$i" &
  vpid=$!
done
[ -n "$vpid" ] && wait "$vpid" 2>/dev/null
say "队列完成 EXTRACTVDONE_${MACH}"

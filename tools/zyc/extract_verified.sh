#!/bin/bash
# 带验证的解压：解完先用 tar -df 逐文件比对，比对干净才删压缩包。
#   用法: MACH=a51 bash extract_verified.sh "5 6"
#
# 为什么要加这一步：`tar -xf` 返回 0 只保证归档结构完整、写入没报错，
# 普通 tar 没有内容校验和，不保证落盘内容逐字节正确。
# `tar -df` 会重读归档，把每个成员和磁盘上的文件比对内容、大小、时间戳，
# 这是真正的内容校验。代价是再读一遍归档（约 40 分钟/包）。
#
# 包放在 _hold（我把还没开始的 9 个从 _tmp 挪过去了），这样旧的 extract2.sh
# 找不到文件会跳过并正常退出，不需要杀进程、不浪费已完成的工作。
# （本来想用 chattr +i 保护，实测这个文件系统不支持。）
#
# 严格一个一个：先等本机 extract2.sh 整个退出再开工。
set -u
export TZ=Asia/Shanghai
H=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_hold
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_v_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" | tee -a "$LOG"; }

say "挂起：等本机 extract2.sh 退出"
while pgrep -f 'extract2\.sh' > /dev/null 2>&1; do sleep 60; done
while pgrep -f 'tar -xf .*camera' > /dev/null 2>&1; do sleep 60; done
say "本机旧队列已结束，开始带验证解压。队列: $1"

for i in $1; do
  f=$H/nuplan-v1.1_train_camera_${i}.zip
  if [ ! -f "$f" ]; then say "camera_$i 不在 _hold，跳过"; continue; fi
  sz=$(du -h "$f" | cut -f1)

  say ">>> camera_$i ($sz) 解压"
  t0=$(date +%s)
  tar -xf "$f" --strip-components=1 -C "$DST" 2>>"$LOG"
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then say "!!! camera_$i 解压失败 rc=$rc，包保留，跳过"; continue; fi
  say "    解压完成，用时 $(( (t1-t0)/60 )) 分钟，开始 tar -df 内容比对"

  # -C 要和解压时一致；--strip-components 对 -d 同样适用
  dout=$(cd "$DST" && tar -df "$f" --strip-components=1 2>&1)
  drc=$?
  t2=$(date +%s)
  # tar -d 对"归档里有、磁盘上也有且一致"的成员不输出任何内容；
  # 任何差异都会打印一行。时间戳差异也会报，但我们是刚解出来的，不该有。
  nd=$(echo "$dout" | grep -c . )
  if [ $drc -eq 0 ] && [ "$nd" = "0" ]; then
    say "<<< camera_$i 验证通过（比对用时 $(( (t2-t1)/60 )) 分钟，零差异）"
    rm -f "$f" && say "    已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  else
    say "!!! camera_$i 验证不通过 rc=$drc 差异 $nd 处，包保留。前几处："
    echo "$dout" | head -5 | sed 's/^/      /' | tee -a "$LOG"
  fi
done
say "队列完成 EXTRACTVDONE_${MACH}"

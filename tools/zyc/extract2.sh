#!/bin/bash
# 分片解压：每台机器只处理指定的几个包，四台并行。
#   用法: bash extract2.sh "3 4 5 6"
#
# 相比 v1 的两处改动：
#   1. 去掉 nice/ionice —— v1 用最低优先级，实测只有 31 MB/s，17 个包要 17 小时。
#      用户确认拖慢训练没关系，所以按正常优先级跑。
#   2. 按列表分片，四台机器各取不同的包并行，避免重复和冲突。
#
# 包名是 .zip，实际是 POSIX tar（S3 对象就是这样）。包内带一层
# nuplan-v1.1_train_camera_N/，而目标 images/ 下直接是 <log>/CAM_*/，
# 所以必须 --strip-components=1。
#
# 解压成功（tar 退出 0）才删压缩包，保持磁盘占用平稳。
# 磁盘守护（diskguard.sh）会在余量不足时 SIGSTOP 这些 tar。
set -u
export TZ=Asia/Shanghai
Z=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
H=$(hostname -s)
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_${H}.log
KEEP=${KEEP:-0}

say() { echo "[$(date '+%F %T')] [$H] $*" | tee -a "$LOG"; }

mkdir -p "$DST"
LIST="$1"
say "开始，队列: $LIST，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"

for i in $LIST; do
  f=$Z/nuplan-v1.1_train_camera_${i}.zip
  if [ ! -f "$f" ]; then say "  camera_$i 不在（可能已解压），跳过"; continue; fi
  sz=$(du -h "$f" | cut -f1)
  say ">>> camera_$i ($sz)"
  t0=$(date +%s)
  tar -xf "$f" --strip-components=1 -C "$DST" 2>>"$LOG"
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    say "<<< camera_$i 成功，用时 $(( (t1-t0)/60 )) 分钟"
    if [ "$KEEP" != 1 ]; then rm -f "$f" && say "    已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"; fi
  else
    say "!!! camera_$i 失败 rc=$rc，包保留"
  fi
done
say "队列完成 EXTRACTDONE_${H}"

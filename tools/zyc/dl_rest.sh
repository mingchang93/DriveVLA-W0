#!/bin/bash
# 下 train_camera 的剩余包（29–42），带磁盘余量闸门。
#   用法: MACH=a51 bash dl_rest.sh "29 30 31 32 33 34 35"
#
# 为什么之前没下这些：我当初探 S3 只探到 28 就停了，把"29 个包"当成全部。
# 实际 train_camera 到 42，共 43 个，少下了 14 个（约 1.55 TB）。
#
# 磁盘闸门：开下一个包前要求余量 > MIN_FREE（默认 500 GB）。
# 现在盘上压着 2.3 TB 已解压但未删的归档，校验正在并行回收，
# 所以这里会随着空间释放自然推进，不会把盘填满。
# 单个包最大 153 GB，500 GB 的阈值留了三倍余量。
#
# 串行：一次只下一个，下完再下下一个——下一半的包没法用。
# 续传：curl -C -，中断后重跑同一条命令即可接着下。
set -u
export TZ=Asia/Shanghai
D=/mnt/sfs_turbo/zyc/nuplan-camera-assist
BASE=https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/sensor_blobs/train_set
MACH=${MACH:-unknown}
MIN_FREE=${MIN_FREE:-500}
LOG=$D/progress_rest_${MACH}.log
mkdir -p "$D/_tmp" "$D/_logs" "$D/_done"
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

free_gb() { df -BG /mnt/sfs_turbo | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

say "队列: $1，磁盘闸门 ${MIN_FREE}G"
for i in $1; do
  name=nuplan-v1.1_train_camera_${i}.zip
  if [ -f "$D/_done/$name.done" ]; then say "camera_$i 已完成，跳过"; continue; fi

  # 等磁盘腾出空间
  waited=0
  while [ "$(free_gb)" -lt "$MIN_FREE" ]; do
    [ $((waited % 30)) -eq 0 ] && say "  等磁盘：余量 $(free_gb)G < ${MIN_FREE}G（已等 $waited 分钟）"
    sleep 60; waited=$((waited+1))
    if [ "$waited" -gt 720 ]; then say "!! 等了 12 小时磁盘仍不足，放弃 camera_$i"; break 2; fi
  done

  say ">>> camera_$i 开始（余量 $(free_gb)G）"
  t0=$(date +%s)
  ok=0
  for attempt in 1 2 3 4 5 6; do
    curl -L -C - --fail \
      --retry 10 --retry-delay 20 --retry-all-errors \
      --speed-time 300 --speed-limit 10240 --connect-timeout 30 \
      -o "$D/_tmp/$name" "$BASE/$name" >> "$D/_logs/camera_${i}.log" 2>&1
    if [ $? -eq 0 ]; then ok=1; break; fi
    say "    第 $attempt 次中断，已下 $(du -h "$D/_tmp/$name" 2>/dev/null | cut -f1)，30 秒后续传"
    sleep 30
  done

  if [ "$ok" = 1 ]; then
    # 下完核对字节数，和 S3 的 content-length 比
    loc=$(stat -c %s "$D/_tmp/$name")
    rem=$(curl -sI --max-time 20 "$BASE/$name" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
    if [ -n "$rem" ] && [ "$loc" = "$rem" ]; then
      touch "$D/_done/$name.done"
      say "<<< camera_$i 完成并字节核对通过（$(( loc/1073741824 )) GB，$(( ($(date +%s)-t0)/60 )) 分钟）"
    else
      say "!!! camera_$i 字节数不符（本地 $loc / 远端 ${rem:-?}），不标记完成"
    fi
  else
    say "!!! camera_$i 六次都失败，放弃"
  fi
done
say "队列完成 DLRESTDONE_${MACH}"

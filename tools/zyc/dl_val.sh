#!/bin/bash
# 下 nuPlan 的 val_set 相机包（val_camera_0..11，共 12 个、约 803 GB）。
#   用法: MACH=a51 bash dl_val.sh "0 1 2"
#
# 为什么要下：官方预训练的数据量（由 emu3_pre_state.json 的 8000 步 / 44.199 epoch
# 与 gbs 96 反推）是 17,376 片段。我们实测每 log 产 14.52 个片段，而 train_set
# 43 个包的上限只有 1,105 个 log = 16,023 片段 —— 只用 train_set 无论如何凑不到。
# 公开的另一来源就是 val_set（约 188 个 log），train+val 约 18,774 片段，量级对上。
# 而且 val 的 db 我们早就有了（当时下了九个 train 包 + 一个 val 包，
# nuPlan 在 db 层面本来就是合在 splits/trainval 里的），缺的只有相机图。
#
# 串行：一次只下一个，下完再下下一个 —— 下一半的包没法用。
# 续传：curl -C -，中断后重跑同一条命令即可接着下。
# 下完核字节数：和 S3 的 content-length 比，一致才打 .done 标记。
set -u
export TZ=Asia/Shanghai
D=/mnt/sfs_turbo/zyc/nuplan-camera-assist
BASE=https://motional-nuplan.s3.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set
MACH=${MACH:-unknown}
MIN_FREE=${MIN_FREE:-400}
LOG=$D/progress_val_${MACH}.log
mkdir -p "$D/_tmp" "$D/_logs" "$D/_done"
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

free_gb() { df -BG /mnt/sfs_turbo | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }

say "队列: $1，磁盘闸门 ${MIN_FREE}G"
for i in $1; do
  name=nuplan-v1.1_val_camera_${i}.zip
  if [ -f "$D/_done/$name.done" ]; then say "val_camera_$i 已完成，跳过"; continue; fi

  waited=0
  while [ "$(free_gb)" -lt "$MIN_FREE" ]; do
    [ $((waited % 30)) -eq 0 ] && say "  等磁盘：余量 $(free_gb)G < ${MIN_FREE}G（已等 $waited 分钟）"
    sleep 60; waited=$((waited+1))
    if [ "$waited" -gt 720 ]; then say "!! 等了 12 小时磁盘仍不足，放弃 val_camera_$i"; break 2; fi
  done

  say ">>> val_camera_$i 开始（余量 $(free_gb)G）"
  t0=$(date +%s)
  ok=0
  for attempt in 1 2 3 4 5 6; do
    curl -L -C - --fail \
      --retry 10 --retry-delay 20 --retry-all-errors \
      --speed-time 300 --speed-limit 10240 --connect-timeout 30 \
      -o "$D/_tmp/$name" "$BASE/$name" >> "$D/_logs/val_camera_${i}.log" 2>&1
    if [ $? -eq 0 ]; then ok=1; break; fi
    say "    第 $attempt 次中断，已下 $(du -h "$D/_tmp/$name" 2>/dev/null | cut -f1)，30 秒后续传"
    sleep 30
  done

  if [ "$ok" = 1 ]; then
    loc=$(stat -c %s "$D/_tmp/$name")
    rem=$(curl -sI --max-time 20 "$BASE/$name" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
    if [ -n "$rem" ] && [ "$loc" = "$rem" ]; then
      touch "$D/_done/$name.done"
      say "<<< val_camera_$i 完成并字节核对通过（$(( loc/1073741824 )) GB，$(( ($(date +%s)-t0)/60 )) 分钟）"
    else
      say "!!! val_camera_$i 字节数不符（本地 $loc / 远端 ${rem:-?}），不标记完成"
    fi
  else
    say "!!! val_camera_$i 六次都失败，放弃"
  fi
done
say "队列完成 DLVALDONE_${MACH}"

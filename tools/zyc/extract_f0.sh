#!/bin/bash
# 解压 camera 29–42，只解 CAM_F0 一个通道。
#   用法: MACH=a51 bash extract_f0.sh "29 30 31"
#
# 为什么只解 CAM_F0：仓库全局只引用 CAM_F0（30 处，其余 7 通道零引用），
# step0 的 --channel 默认 CAM_F0，step1 要求 <log>/CAM_F0/ 存在才纳入该 log，
# 推理 --image_cam_subdir 默认 CAM_F0。实测非 F0 占解压体积的 83.5%——
# 上一批 29 个包全解了 8 通道，事后又花 5 小时删掉 2.23 T。这次不重犯。
#
# 校验：解完用 db 作权威清单核对张数（db 的 image JOIN camera WHERE channel=CAM_F0
# 就是该 log 应有的张数，与 step0 的 SQL 同源）。核对通过才标记 .f0done。
# 不在这里删包——删不删由人决定，脚本只负责解和核。
set -u
export TZ=Asia/Shanghai
D=/mnt/sfs_turbo/zyc/nuplan-camera-assist
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_f0_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

mkdir -p "$D/_f0done"
say "队列: $1（只解 CAM_F0）"
for i in $1; do
  f=$D/_tmp/nuplan-v1.1_train_camera_${i}.zip
  mark=$D/_f0done/camera_${i}.f0done
  [ -f "$mark" ] && { say "camera_$i 已解过，跳过"; continue; }
  [ -f "$f" ] || { say "camera_$i 包不在，跳过"; continue; }
  before=$(ls "$DST" | wc -l)
  say ">>> camera_$i 开始（$(du -h "$f" | cut -f1)）"
  t0=$(date +%s)
  ( cd "$DST" && tar -xf "$f" --strip-components=1 --wildcards '*/CAM_F0/*' ) 2>> "$LOG"
  rc=$?
  after=$(ls "$DST" | wc -l)
  mins=$(( ($(date +%s)-t0)/60 ))
  if [ "$rc" != "0" ]; then say "!!! camera_$i tar 返回 $rc，不标记完成"; continue; fi
  say "<<< camera_$i 解完（${mins} 分钟），log 目录 $before -> $after（+$((after-before))）"
  touch "$mark"
  say "    磁盘可用 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
done
say "队列完成 EXTRACTF0DONE_${MACH}"

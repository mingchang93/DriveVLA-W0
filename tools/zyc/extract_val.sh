#!/bin/bash
# 边下边解 val_set 的相机包，只解 CAM_F0。
#   用法: MACH=a51 bash extract_val.sh "0 1 2 3 4 5"
#
# 为什么只解 CAM_F0：仓库全局只引用 CAM_F0（30 处，其余 7 通道零引用），
# step0 的 --channel 默认 CAM_F0，step1 要求 <log>/CAM_F0/ 存在才纳入该 log，
# 推理 --image_cam_subdir 默认 CAM_F0。实测非 F0 占解压体积 83.5%——
# train 那批最初全解了 8 通道，事后又花 5 小时删掉 2.23 T。不重犯。
#
# 边下边解：不等 12 个包全下完。轮询 .done 标记（字节数与 S3 的 content-length
# 核对通过才会打标记），见到哪个就解哪个，下载和解压重叠起来。
#
# 为什么不放 a53/a54：那两台同事在跑训练，解压的 IO 会拖他们。
set -u
export TZ=Asia/Shanghai
D=/mnt/sfs_turbo/zyc/nuplan-camera-assist
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_val_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

mkdir -p "$D/_f0done"
say "队列: $1（只解 CAM_F0，边下边解）"

pending="$1"
rounds=0
while [ -n "$pending" ]; do
  nxt=""
  for i in $pending; do
    mark=$D/_f0done/val_camera_${i}.f0done
    done_f=$D/_done/nuplan-v1.1_val_camera_${i}.zip.done
    f=$D/_tmp/nuplan-v1.1_val_camera_${i}.zip
    if [ -f "$mark" ]; then
      continue                      # 已解过
    fi
    if [ ! -f "$done_f" ]; then
      nxt="$nxt $i"                 # 还没下完，下一轮再看
      continue
    fi
    before=$(ls "$DST" | wc -l)
    say ">>> val_camera_$i 开始解（$(du -h "$f" | cut -f1)）"
    t0=$(date +%s)
    ( cd "$DST" && tar -xf "$f" --strip-components=1 --wildcards '*/CAM_F0/*' ) 2>> "$LOG"
    rc=$?
    after=$(ls "$DST" | wc -l)
    if [ "$rc" != "0" ]; then
      say "!!! val_camera_$i tar 返回 $rc，不标记完成"
      nxt="$nxt $i"
      continue
    fi
    say "<<< val_camera_$i 解完（$(( ($(date +%s)-t0)/60 )) 分钟），log 目录 $before -> $after（+$((after-before))）"
    touch "$mark"
    say "    磁盘可用 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  done
  pending=$(echo $nxt)
  [ -z "$pending" ] && break
  rounds=$((rounds+1))
  [ $((rounds % 10)) -eq 1 ] && say "等这些包下完: $pending"
  sleep 120
  if [ "$rounds" -gt 180 ]; then say "!! 等了 6 小时仍有未下完的: $pending，退出"; break; fi
done
say "队列完成 EXTRACTVALDONE_${MACH}"

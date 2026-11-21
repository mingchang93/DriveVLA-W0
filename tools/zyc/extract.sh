#!/bin/bash
# 把我们下的 nuPlan 相机包解压到共享数据集目录。
#
# 关键点：文件名虽然是 .zip，实际是 POSIX tar（S3 对象就是这样，lmc 那边
# 索性命名成 .tar）。包内带一层顶层目录 nuplan-v1.1_train_camera_N/，
# 而目标 images/ 下直接是 <log>/CAM_*/，所以必须 --strip-components=1，
# 否则会多出一层、和 lmc 已解压的那 374 个 log 对不上。
#
# 解压成功（tar 退出 0）才删压缩包——和 lmc 的做法一致，保持磁盘占用平稳。
# jpg 压缩率低，解压前后体积接近，所以净增长约等于零，峰值只多占单个包。
#
# a51 同时在跑训练，所以用最低调度优先级 + 空闲 IO 优先级，尽量不抢资源。
set -u
Z=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract.log
KEEP=${KEEP:-0}        # KEEP=1 则解压后保留压缩包

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

mkdir -p "$DST"
say "开始解压，源 $Z -> $DST（解压后$([ "$KEEP" = 1 ] && echo 保留 || echo 删除)压缩包）"
say "  待处理 $(ls $Z/*.zip 2>/dev/null | wc -l) 个包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"

for f in $(ls $Z/*.zip 2>/dev/null | sort -V); do
  b=$(basename "$f")
  n=${b%.zip}                     # nuplan-v1.1_train_camera_N
  sz=$(du -h "$f" | cut -f1)
  say ">>> $b ($sz)"
  before=$(ls "$DST" | wc -l)
  t0=$(date +%s)
  nice -n 19 ionice -c3 tar -xf "$f" --strip-components=1 -C "$DST" 2>>"$LOG"
  rc=$?
  t1=$(date +%s)
  after=$(ls "$DST" | wc -l)
  if [ $rc -eq 0 ]; then
    say "<<< $b 解压成功，用时 $(( (t1-t0)/60 )) 分钟，images 下 log 数 $before -> $after"
    if [ "$KEEP" != 1 ]; then rm -f "$f" && say "    已删压缩包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"; fi
  else
    say "!!! $b 解压失败 rc=$rc，压缩包保留，跳过"
  fi
done

say "全部处理完，images 下共 $(ls $DST | wc -l) 个 log，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
say "EXTRACTDONE"

#!/bin/bash
# 接力补解 camera_2。
#
# 为什么需要它：camera_2 原本由旧 extract.sh 的 tar 在解，我杀掉它的父进程后
# tar 失去了所有活祖先、被进程组回收，解压中途断了（压缩包完好、images 里留下
# 部分文件）。而我分片时假定它由那个孤儿负责，16 个包的队列里漏了它。
#
# 严格一个一个：先等本机当前队列的 tar 全部结束，再开工，不和它们并行。
# tar -xf 默认覆盖，所以残留的部分文件会被正确重写，不需要先清理。
set -u
export TZ=Asia/Shanghai
Z=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_pickup.log
say() { echo "[$(date +%F %T)] $*" | tee -a "$LOG"; }

say "接力脚本挂起，等本机当前解压队列结束"
while pgrep -f tar -xf .*camera > /dev/null; do sleep 120; done
say "本机解压队列已空，开始补解 camera_2"

f=$Z/nuplan-v1.1_train_camera_2.zip
if [ ! -f "$f" ]; then say "camera_2 的包不在了，跳过"; exit 0; fi
t0=$(date +%s)
tar -xf "$f" --strip-components=1 -C "$DST" 2>>"$LOG"
rc=$?
if [ $rc -eq 0 ]; then
  say "camera_2 解压成功，用时 $(( ($(date +%s)-t0)/60 )) 分钟"
  rm -f "$f" && say "已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
else
  say "camera_2 解压失败 rc=$rc，包保留"
fi
say "PICKUPDONE"

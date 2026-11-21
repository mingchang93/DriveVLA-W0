#!/bin/bash
# 接力补解 camera_2（第二版）。
#
# 背景：camera_2 原由旧 extract.sh 的 tar 在解，我杀掉其父进程后，tar 失去了
# 所有活祖先、被 docker exec 的进程组回收，解压中断（压缩包完好）。而我分片时
# 假定它由那个孤儿负责，16 个包的队列里漏了它，所以要补。
#
# v1 的错误：用"有没有 tar 进程"判断本机队列是否结束。那有竞态——
# 一个包刚结束、下一个还没启动的瞬间是空窗，它就误以为队列空了，
# 结果和 camera_8 并行跑起来，违背"一个一个"的要求。
# v2 改成等调度脚本 extract2.sh 整个退出，那是可靠信号，没有空窗。
#
# 重解是安全的：tar -xf 逐文件独立解出、默认覆盖，中断时写了一半的文件
# 会被完整重写，最终状态正确；代价只是已解部分要重做。
set -u
export TZ=Asia/Shanghai
Z=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract_pickup.log
say() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG"; }

say "接力脚本挂起：等本机 extract2.sh 调度器退出后再补解 camera_2"
while pgrep -f 'extract2\.sh' > /dev/null 2>&1; do sleep 120; done
# 再确认一次没有任何 tar 在跑，双保险
while pgrep -f 'tar -xf .*camera' > /dev/null 2>&1; do sleep 60; done
say "本机队列已全部结束，开始补解 camera_2"

f=$Z/nuplan-v1.1_train_camera_2.zip
if [ ! -f "$f" ]; then say "camera_2 的包已不在，跳过"; say "PICKUPDONE"; exit 0; fi
t0=$(date +%s)
tar -xf "$f" --strip-components=1 -C "$DST" 2>>"$LOG"
rc=$?
if [ $rc -eq 0 ]; then
  say "camera_2 成功，用时 $(( ($(date +%s)-t0)/60 )) 分钟"
  rm -f "$f"
  say "已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
else
  say "camera_2 失败 rc=$rc，包保留"
fi
say "PICKUPDONE"

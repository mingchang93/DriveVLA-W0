#!/bin/bash
# 磁盘守护：盯着共享盘余量，快满了就暂停解压，而不是等人来救。
#
# 为什么是暂停不是杀掉：tar 被 SIGSTOP 后进程还在、已写入的文件完好，
# 空间回收后 SIGCONT 就能接着解；杀掉则这一包要重来，而且脚本会误判成
# 解压失败。暂停期间训练不受影响（训练是 NPU 瓶颈，解压是 IO）。
#
# 阈值：低于 LOW 暂停，回到 HIGH 以上才恢复（留迟滞，避免反复抖动）。
# 单个包解压峰值约 125 GB，所以 LOW 给 400 GB 有足够余量。
set -u
export TZ=Asia/Shanghai
LOW=${LOW:-400}      # GB
HIGH=${HIGH:-600}    # GB
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/diskguard.log
say()   { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
alert() { echo "[$(date '+%F %T')] ALERT $*" | tee -a "$LOG"; }

paused=0
say "磁盘守护启动：低于 ${LOW}G 暂停解压，回到 ${HIGH}G 以上恢复"
while true; do
  avail=$(df -BG /mnt/sfs_turbo 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}')
  pids=$(pgrep -f 'tar -xf .*camera' 2>/dev/null)

  if [ -z "${avail:-}" ]; then
    alert "读不到磁盘余量"; sleep 60; continue
  fi

  if [ "$paused" = 0 ] && [ "$avail" -lt "$LOW" ]; then
    if [ -n "$pids" ]; then
      for p in $pids; do kill -STOP "$p" 2>/dev/null; done
      paused=1
      alert "磁盘只剩 ${avail}G，已暂停解压（pid: $(echo $pids | tr '\n' ' ')）。训练不受影响。需要清理空间后才会自动恢复。"
    else
      alert "磁盘只剩 ${avail}G，但没找到解压进程"
    fi
  elif [ "$paused" = 1 ] && [ "$avail" -ge "$HIGH" ]; then
    for p in $pids; do kill -CONT "$p" 2>/dev/null; done
    paused=0
    say "磁盘回到 ${avail}G，已恢复解压"
  fi

  # 训练没在跑就没必要再守了（解压也结束了）
  if [ -z "$pids" ] && grep -q EXTRACTDONE /mnt/sfs_turbo/zyc/drivevla-w0-zyc/extract.log 2>/dev/null; then
    say "解压已全部完成，守护退出（最终余量 ${avail}G）"; exit 0
  fi

  say "余量 ${avail}G  解压进程 $(echo "$pids" | grep -c . ) 个  暂停=$paused"
  sleep 60
done

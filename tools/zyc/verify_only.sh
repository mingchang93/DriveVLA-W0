#!/bin/bash
# 只做校验 + 删包（包已解压好了，不重解）。
#   用法: MACH=a51 bash verify_only.sh "5 6"
#
# 修的是 extract_v2.sh 里的判据 bug：
#   `tar -df <包> --strip-components=1` 在内容完全一致时也会输出一行
#       tar: : Warning: Cannot stat: No such file or directory
#   原因是顶层目录被剥成空名字，tar 对空名字 stat 不到——完全无害。
#   原判据要求"输出行数为 0"，于是每个包都被误判为校验不通过、归档永不删除，
#   磁盘一路降到 1.4 T。实测 --strip-components 对 -d 是生效的，
#   per-file 比对全部通过。
#
# 另一个坑：`tar -d` 的返回码不可信。对照实验里所有文件都缺失，rc 仍是 0。
# 所以只能解析输出，不能靠 rc。
#
# 新判据：滤掉那行空名字警告后，任何剩余输出都算真差异。
# 真差异的形态是 `<file>: Mod time differs` / `Size differs` /
# `Contents differ` / `Warning: Cannot stat`（带文件名的）。
#
# 日志：不用 `echo | tee`。今天所有"日志只有一两行"的怪事都是同一个原因——
# 这些脚本用 docker exec -d + setsid 启动、stdout 没重定向，setsid 脱离后
# fd 失效，tee 写 stdout 失败就退出，文件那一份也没写成。
# 这里直接用 >> 追加到文件，不经过 tee。
set -u
export TZ=Asia/Shanghai
HOLD=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_hold
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/verify_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "开始校验队列: $1"
for i in $1; do
  f=$HOLD/nuplan-v1.1_train_camera_${i}.zip
  if [ ! -f "$f" ]; then say "camera_$i 不在 _hold（可能已通过并删除），跳过"; continue; fi
  say ">>> camera_$i 开始 tar -df"
  t0=$(date +%s)
  out=$(cd "$DST" && tar -df "$f" --strip-components=1 2>&1)
  # 滤掉顶层空名字那行无害警告
  real=$(echo "$out" | grep -v '^tar: : Warning: Cannot stat' | grep -c .)
  mins=$(( ($(date +%s)-t0)/60 ))
  if [ "$real" = "0" ]; then
    say "<<< camera_$i 校验通过（${mins} 分钟，真差异 0 处）"
    rm -f "$f"
    say "    已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  else
    say "!!! camera_$i 校验不通过，真差异 $real 处，包保留。前几处："
    echo "$out" | grep -v '^tar: : Warning: Cannot stat' | head -8 | sed 's/^/      /' >> "$LOG"
  fi
done
say "队列完成 VERIFYDONE_${MACH}"

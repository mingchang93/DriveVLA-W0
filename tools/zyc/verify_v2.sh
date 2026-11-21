#!/bin/bash
# 校验 + 删包（包已解压好，不重解）。
#   用法: MACH=a51 bash verify_v2.sh "2 5 6 9 20"
#
# 判据演进：
#   v1 要求 `tar -df` 输出行数为 0 —— 错。`--strip-components=1` 会让顶层目录
#      变成空名字，tar 对空名字 stat 不到，必然输出一行无害警告。
#   v2（本版）再滤掉 `LICENSE` —— 43 个相机包每个都带一个包根 LICENSE，
#      解压时后者覆盖前者，所以时间戳必然不一致。实测 camera_5/9/27/28 的
#      "真差异" 全都只有这一行，所有图像文件逐个比对都通过。
#
# 另一个坑：`tar -d` 的返回码不可信。对照实验里所有文件都缺失时 rc 仍是 0，
# 所以只能解析输出。
#
# 真差异的形态：`<file>: Mod time differs` / `Size differs` / `Contents differ`
# / 带文件名的 `Cannot stat`（缺文件）。滤掉上面两类后任何剩余输出都算真差异。
set -u
export TZ=Asia/Shanghai
HOLD=/mnt/sfs_turbo/zyc/nuplan-camera-assist/_hold
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/verify2_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "开始校验队列: $1（滤掉空名字警告与 LICENSE 时间戳）"
for i in $1; do
  f=$HOLD/nuplan-v1.1_train_camera_${i}.zip
  if [ ! -f "$f" ]; then say "camera_$i 不在 _hold，跳过"; continue; fi
  say ">>> camera_$i tar -df 开始"
  t0=$(date +%s)
  out=$(cd "$DST" && tar -df "$f" --strip-components=1 2>&1)
  real=$(echo "$out" \
    | grep -v '^tar: : Warning: Cannot stat' \
    | grep -vE '^LICENSE: (Mod time|Size) differs' \
    | grep -c .)
  mins=$(( ($(date +%s)-t0)/60 ))
  if [ "$real" = "0" ]; then
    say "<<< camera_$i 通过（${mins} 分钟，真差异 0）"
    rm -f "$f" && say "    已删包，磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  else
    say "!!! camera_$i 不通过，真差异 $real 处，包保留。前几处："
    echo "$out" | grep -v '^tar: : Warning: Cannot stat' \
      | grep -vE '^LICENSE: (Mod time|Size) differs' | head -8 | sed 's/^/      /' >> "$LOG"
  fi
done
say "队列完成 VERIFY2DONE_${MACH}"

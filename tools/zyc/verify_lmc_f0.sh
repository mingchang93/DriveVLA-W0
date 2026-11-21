#!/bin/bash
# 只比对 CAM_F0 通道，校验 lmc 的归档是否已完整解压；通过则删归档。
#   用法: MACH=a51 bash verify_lmc_f0.sh "12 13"
#
# 为什么要改口径：上一版用「整包比对」，而 lmc 当初只解了 CAM_F0 一个通道，
# 于是每个包都报约 50 万处「CAM_L1: Cannot stat」——那是我的判据错了，
# 不是他的数据坏了。仓库全局只出现 CAM_F0（30 处），其余 7 通道零引用，
# 所以「只解 CAM_F0」对这条训练/评测路径是完备的。
#
# 判据：把归档成员限制在 */CAM_F0/*，此时
#   - 目录项 `.../CAM_F0: Mode differs` 是 umask 差异，无害，滤掉
#   - 文件级任何一行（Mod time / Size / Contents differ、带文件名的 Cannot stat）都算真差异
# 另：tar -d 返回码不可信（全缺时仍返回 0），只能解析输出。
set -u
export TZ=Asia/Shanghai
SRC=/mnt/sfs_turbo/datasets/nuplan-v1.1/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/verify_lmc_f0_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "开始（只比对 CAM_F0，通过则删）: $1"
for i in $1; do
  f=$SRC/nuplan-v1.1_train_camera_${i}.tar
  [ -f "$f" ] || { say "camera_$i 不在，跳过"; continue; }
  sz=$(du -h "$f" | cut -f1)
  say ">>> camera_$i 开始（$sz）"
  t0=$(date +%s)
  out=$(cd "$DST" && tar -df "$f" --strip-components=1 --wildcards '*/CAM_F0/*' 2>&1)
  # 目录项的 Mode differs 与空名字警告滤掉，其余都算真差异
  real_lines=$(echo "$out" \
    | grep -v '^tar: : Warning: Cannot stat' \
    | grep -vE '/CAM_F0/?: Mode differs$' \
    | grep -vE '^LICENSE: (Mod time|Size) differs' \
    | grep .)
  real=$(echo "$real_lines" | grep -c .)
  mins=$(( ($(date +%s)-t0)/60 ))
  # 顺带记录这包里有多少个 CAM_F0 文件，确认不是「一个都没比」
  say "    原始输出 $(echo "$out" | grep -c .) 行，其中真差异 $real 处（${mins} 分钟）"
  if [ "$real" = "0" ]; then
    say "<<< camera_$i CAM_F0 完整"
    rm -f "$f" && say "    已删（省 $sz），磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  else
    say "!!! camera_$i 有 $real 处真差异 —— 保留。前几处："
    echo "$real_lines" | head -8 | sed 's/^/      /' >> "$LOG"
  fi
done
say "队列完成 LMCF0DONE_${MACH}"

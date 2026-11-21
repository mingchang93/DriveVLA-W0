#!/bin/bash
# 校验 lmc 的 camera 12–19，通过则删除归档。
#   用法: MACH=a51 bash verify_lmc2.sh "12 13"
#
# 授权：用户转达 lmc 同意删除这些压缩包，并要求先校验。
# 这是不可逆操作（约 1.2 TB，重下要数小时），所以判据取严：
#   只有 tar -df 的「真差异」为 0 才删；任何差异都保留并把前几处写进日志。
#
# 为什么必须先校验：之前我只从每个归档开头抽几个 log 名、看 images/ 下有没有
# 同名目录，就说"已解压可以删"——那只证明开头解过，既不验内容也不验完整性。
# tar -df 是逐成员比对内容/大小/时间戳。
#
# 判据的两处例外（都已验证无害）：
#   1. `tar: : Warning: Cannot stat` —— --strip-components=1 把顶层目录剥成
#      空名字，tar 对空名字 stat 不到，必然出现这一行。
#   2. `LICENSE: Mod time differs` —— 43 个相机包每个都带一个包根 LICENSE，
#      解压时后者覆盖前者，时间戳必然不一致。
# 另：tar -d 的返回码不可信（所有文件缺失时仍返回 0），只能解析输出。
#
# 先等本机 verify_v2 队列结束，避免叠加 IO（训练与下载在同一个网络盘上）。
set -u
export TZ=Asia/Shanghai
SRC=/mnt/sfs_turbo/datasets/nuplan-v1.1/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/verify_lmc_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "挂起：等本机 verify_v2 队列结束"
while pgrep -f 'verify_v2\.sh' > /dev/null 2>&1; do sleep 120; done
say "开始校验 lmc 的包（通过则删）: $1"

for i in $1; do
  f=$SRC/nuplan-v1.1_train_camera_${i}.tar
  [ -f "$f" ] || { say "camera_$i 不在，跳过"; continue; }
  sz=$(du -h "$f" | cut -f1)
  say ">>> camera_$i tar -df 开始（$sz）"
  t0=$(date +%s)
  out=$(cd "$DST" && tar -df "$f" --strip-components=1 2>&1)
  real=$(echo "$out" \
    | grep -v '^tar: : Warning: Cannot stat' \
    | grep -vE '^LICENSE: (Mod time|Size) differs' \
    | grep -c .)
  mins=$(( ($(date +%s)-t0)/60 ))
  if [ "$real" = "0" ]; then
    say "<<< camera_$i 完整，真差异 0（${mins} 分钟）"
    rm -f "$f" && say "    已删（省 $sz），磁盘剩余 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
  else
    say "!!! camera_$i 有 $real 处真差异（${mins} 分钟）—— 保留，需重解。前几处："
    echo "$out" | grep -v '^tar: : Warning: Cannot stat' \
      | grep -vE '^LICENSE: (Mod time|Size) differs' | head -8 | sed 's/^/      /' >> "$LOG"
  fi
done
say "队列完成 VERIFYLMCDONE_${MACH}"

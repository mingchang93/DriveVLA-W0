#!/bin/bash
# 校验 lmc 的 camera 12–19（只读，不删）。
#   用法: MACH=a51 bash verify_lmc.sh "12 13"
#
# 为什么要做：我之前只从每个归档开头抽几个 log 名、看 images/ 下有没有同名目录，
# 据此说"已解压、可以删"——那只证明开头解过，既不验内容也不验完整性。
# 要给出"可以删"的结论必须跑 tar -df 逐成员比对。
#
# 这些是 lmc 目录下的文件，本脚本只读：跑完只报结论，**不删任何东西**。
# 删不删由他定。
#
# 先等本机我们自己那 9 个的校验队列结束，避免叠加 IO（训练在同一个网络盘上）。
#
# 判据同 verify_v2：滤掉顶层空名字警告和 LICENSE 时间戳差异
# （43 个包都带一个包根 LICENSE，解压时后者覆盖前者，必然不一致）。
# tar -d 的返回码不可信（所有文件缺失时仍返回 0），只能解析输出。
set -u
export TZ=Asia/Shanghai
SRC=/mnt/sfs_turbo/datasets/nuplan-v1.1/_tmp
DST=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
MACH=${MACH:-unknown}
LOG=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/verify_lmc_${MACH}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

say "挂起：等本机 verify_v2 队列结束"
while pgrep -f 'verify_v2\.sh' > /dev/null 2>&1; do sleep 120; done
say "开始校验 lmc 的包（只读，不删）: $1"

for i in $1; do
  f=$SRC/nuplan-v1.1_train_camera_${i}.tar
  [ -f "$f" ] || { say "camera_$i 不在，跳过"; continue; }
  say ">>> camera_$i tar -df 开始（$(du -h "$f" | cut -f1)）"
  t0=$(date +%s)
  out=$(cd "$DST" && tar -df "$f" --strip-components=1 2>&1)
  real=$(echo "$out" | grep -v '^tar: : Warning: Cannot stat' | grep -vE '^LICENSE: (Mod time|Size) differs' | grep -c .)
  mins=$(( ($(date +%s)-t0)/60 ))
  if [ "$real" = "0" ]; then
    say "<<< camera_$i 完整，真差异 0（${mins} 分钟）—— 可安全删除，省 $(du -h "$f" | cut -f1)"
  else
    say "!!! camera_$i 有 $real 处真差异（${mins} 分钟）—— 不可删，需重解。前几处："
    echo "$out" | grep -v '^tar: : Warning: Cannot stat' | grep -vE '^LICENSE: (Mod time|Size) differs' | head -8 | sed 's/^/      /' >> "$LOG"
  fi
done
say "队列完成 VERIFYLMCDONE_${MACH}"

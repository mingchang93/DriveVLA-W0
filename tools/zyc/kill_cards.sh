#!/bin/bash
# 清空本机 NPU：先记录占卡者的命令行，再按进程组停。
# 用户在 2026-11-12 明确授权「3 点那会先 kill 占卡程序」。
# 记录命令行是为了事后能说清停了谁的任务。
# 不用 pkill -f <模式>：模式串会出现在自己的命令行里，会把自己杀掉。
set -u
LOG=${LOG:-/mnt/sfs_turbo/zyc/drivevla-w0-zyc/kill_cards.log}
say() { echo "[$(date +'%F %T')] [kill_cards] $*" >> "$LOG"; }

pids_on_card() {
  npu-smi info 2>/dev/null | grep -oE '^\| [0-9]+ +\| +[0-9]{4,}' | awk '{print $4}' | sort -u
}

n=$(pids_on_card | wc -l)
if [ "$n" = "0" ]; then say "卡上无进程，无需清理"; exit 0; fi

say "卡上有 $n 个进程，先记录："
for p in $(pids_on_card); do
  say "    PID $p  $(ps -o etime=,args= -p "$p" 2>/dev/null | cut -c1-110)"
done

for sig in TERM KILL; do
  for p in $(pids_on_card); do
    pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$pg" ] && kill -"$sig" -"$pg" 2>/dev/null
    kill -"$sig" "$p" 2>/dev/null
  done
  sleep 10
done

# 显存回收有延迟，等一会儿再确认
for i in 1 2 3 4 5 6; do
  left=$(pids_on_card | wc -l)
  [ "$left" = "0" ] && break
  sleep 10
done
say "清理后卡上进程 $(pids_on_card | wc -l)，合计占用 $(npu-smi info 2>/dev/null | grep -oE '[0-9]+ +/ +98304' | awk '{s+=$1} END{print s+0}') MB（空闲基线约 38000）"

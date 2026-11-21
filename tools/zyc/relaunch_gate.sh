#!/bin/bash
# 重挂本机的夜链门闸：停旧的、清日志、起新的、自证在跑。
#   用法: CHAIN=chain_a54.sh bash relaunch_gate.sh
# 写成脚本是因为内联 ssh 命令的引号嵌套已经坑过一次：
# 旧门闸被杀掉、新的没起来，而我靠 grep 计数自检又被自己的命令行干扰，没看出来。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
CHAIN=${CHAIN:?需要 CHAIN=chain_a54.sh}
base=${CHAIN%.sh}

echo "== 停旧门闸 =="
old=$(ps -eo pid=,args= | grep night_gate.sh | grep -v relaunch_gate | grep -v " grep " | awk '{print $1}')
if [ -n "$old" ]; then
  echo "   旧 PID: $old"
  for p in $old; do kill -KILL "$p" 2>/dev/null; done
  sleep 2
else
  echo "   无旧门闸"
fi

rm -f "$W/gate_$base.log" "$W/$base.log"
echo "== 起新门闸 =="
setsid nohup bash "$T/night_gate.sh" >/dev/null 2>&1 </dev/null &
sleep 8

echo "== 自证 =="
ps -eo pid=,ppid=,etime=,args= | grep night_gate.sh | grep -v relaunch_gate | grep -v " grep " | head -2
echo "-- 门闸日志 --"
cat "$W/gate_$base.log" 2>/dev/null

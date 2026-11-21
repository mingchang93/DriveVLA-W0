#!/bin/bash
# 门闸交接链路的演练桩：不碰卡、不训练，只验证
#   docker exec -d zyc-w0 env GATED=1 setsid bash <脚本>
# 这条路径能否：进到容器、拿到环境变量、脱离父进程、把日志写进共享盘。
# 之所以要单独验：真链一旦在 03:00 放行失败，等于白等一夜。
set -u
export TZ=Asia/Shanghai
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/gate_selftest.log
echo "[$(date +'%F %T')] 桩已启动 GATED=${GATED:-未传}  PPID=$PPID  SID=$(ps -o sid= -p $$ | tr -d ' ')" >> "$LOG"
echo "[$(date +'%F %T')] 容器内可见共享盘: $(ls -d /mnt/sfs_turbo/zyc/DriveVLA-W0 2>/dev/null || echo 不可见)" >> "$LOG"
echo "[$(date +'%F %T')] venv python: $(/mnt/sfs_turbo/lmc/venv-hif8/bin/python -c 'import torch,torch_npu;print(torch.__version__, torch_npu.npu.device_count())' 2>&1 | tail -1)" >> "$LOG"
echo "[$(date +'%F %T')] npu-smi 在容器里可用: $(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +Ascend') 张卡" >> "$LOG"
sleep 20
echo "[$(date +'%F %T')] 20 秒后仍存活（说明没被父进程带走）" >> "$LOG"
echo "[$(date +'%F %T')] SELFTESTDONE" >> "$LOG"

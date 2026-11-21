#!/bin/bash
# 包装器：固定 ROLE，避开 docker exec 的嵌套引号问题
export ROLE=a52
exec bash /mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc/chain2.sh

# DriveVLA-W0 / NAVSIM 环境变量（zyc 副本）
export DRIVEVLA_ROOT=/mnt/sfs_turbo/zyc/DriveVLA-W0
export NAVSIM_DEVKIT_ROOT="$DRIVEVLA_ROOT/inference/navsim/navsim"
export OPENSCENE_DATA_ROOT=/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1
export NUPLAN_MAPS_ROOT=/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/nuplan-maps-v1.0
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/navsim_workspace/exp
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$DRIVEVLA_ROOT:${PYTHONPATH:-}"

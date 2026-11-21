#!/bin/bash
# 删掉已解压图像里的非 CAM_F0 通道（约 2.9 TB），只留 CAM_F0。
#
# 授权与依据：用户要求"对照 repo、hf、论文看看有没有用，没用确实可以删掉"，四处对照结果：
#   我们的 repo      只出现 CAM_F0（30 处），其余 7 通道零引用
#   lmc 参考分支     CAM_F0 5320 处，其余通道 0
#   官方 HF 的 config/state  无任何 CAM_ / num_cameras / n_views / multi_view
#   论文 + 官方 GitHub       benchmark 表里 DriveVLA-W0 标 "1x Cam"（对比方法才是 3x Cam + L）
# 管道层面也是写死的：step0 --channel 默认 CAM_F0；step1 要求 <log>/CAM_F0/ 存在才纳入；
# 推理 --image_cam_subdir 默认 CAM_F0；dataset 的 _construct_image_path 硬编码 CAM_F0。
# 唯一用多相机的 transfuser_features.py 是 TransFuser baseline，读 navsim 自己的
# sensor_blobs，不碰我们解压的 nuplan train_set。
#
# 不可逆，所以：
#   1) 先落清单（每个 log 每个通道的文件数与字节），万一要恢复知道缺什么
#   2) 开工前确认没有解压在跑，避免和正在写的目录打架
#   3) 只删 CAM_* 里名字不是 CAM_F0 的目录，不碰 CAM_F0、不碰 log 根下的其他文件
#   4) 删完复核 CAM_F0 的文件数没变
set -u
export TZ=Asia/Shanghai
IMG=/mnt/sfs_turbo/datasets/nuplan-v1.1/images
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/drop_nonf0.log
MAN=$W/drop_nonf0_manifest.tsv
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

# 2) 有解压在跑就别动
if pgrep -x tar > /dev/null 2>&1 || pgrep -x unzip > /dev/null 2>&1; then
  say "有 tar/unzip 在跑，为免和正在写的目录打架，本次不执行"
  exit 1
fi

say "===== 删非 CAM_F0 通道 开始 ====="
say "开工前磁盘: $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}') 可用"

# 1) 落清单 + 统计
: > "$MAN"
nlog=0
say "正在落清单（每 log 每通道的文件数/字节）…"
for d in "$IMG"/*/; do
  [ -d "$d" ] || continue
  lg=$(basename "$d")
  for c in "$d"CAM_*/; do
    [ -d "$c" ] || continue
    ch=$(basename "$c")
    cnt=$(find "$c" -maxdepth 1 -name '*.jpg' | wc -l)
    byt=$(du -sb "$c" 2>/dev/null | cut -f1)
    printf "%s\t%s\t%s\t%s\n" "$lg" "$ch" "$cnt" "$byt" >> "$MAN"
  done
  nlog=$((nlog+1))
  [ $((nlog % 100)) -eq 0 ] && { say "  清单 $nlog 个 log…"; }
done
say "清单完成：$nlog 个 log，$(wc -l < "$MAN") 条通道记录 -> $MAN"
awk -F'\t' '{if ($2=="CAM_F0") {f+=$4; fc+=$3} else {o+=$4; oc+=$3}} END{
  printf "  CAM_F0  %9.1f GB  %8d 张\n  非 F0   %9.1f GB  %8d 张  占 %.1f%%\n",
    f/1073741824, fc, o/1073741824, oc, o*100/(f+o)}' "$MAN" >> "$LOG"

# 3) 删除
say "开始删除非 CAM_F0 通道…"
del=0
for d in "$IMG"/*/; do
  [ -d "$d" ] || continue
  for c in "$d"CAM_*/; do
    [ -d "$c" ] || continue
    ch=$(basename "$c")
    [ "$ch" = "CAM_F0" ] && continue
    rm -rf "$c"
    del=$((del+1))
  done
  [ $((del % 700)) -eq 0 ] && [ "$del" -gt 0 ] && say "  已删 $del 个通道目录，磁盘可用 $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}')"
done
say "删除完成：$del 个通道目录"

# 4) 复核 CAM_F0 没被碰
say "复核 CAM_F0…"
bad=0; tot=0
while IFS=$'\t' read -r lg ch cnt byt; do
  [ "$ch" = "CAM_F0" ] || continue
  now=$(find "$IMG/$lg/CAM_F0" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
  tot=$((tot+1))
  if [ "$now" != "$cnt" ]; then
    bad=$((bad+1))
    [ "$bad" -le 5 ] && say "  !! $lg CAM_F0 从 $cnt 变成 $now"
  fi
done < "$MAN"
say "复核：$tot 个 log 的 CAM_F0，张数变化的 $bad 个"
say "剩余非 F0 目录: $(find "$IMG" -maxdepth 2 -type d -name 'CAM_*' ! -name CAM_F0 | wc -l)"
say "收工磁盘: $(df -h /mnt/sfs_turbo | tail -1 | awk '{print $4}') 可用"
say "DROPNONF0DONE"

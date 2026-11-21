#!/bin/bash
# 为全部 1105 个 log 建 db 软链，然后跑 step0 把缺的 json 补齐。
#
# 为什么要 db_subset：step0 会递归遍历 --db-root 下所有 .db，而全量 db 有 14571 个，
# 其中绝大多数我们没有对应图片（会被 place_image 判为 missing 后跳过），
# 白读一遍要一个多小时。上次就是用软链子集的做法（当时 373 个）。
#
# 为什么 blob-root 和 image-dir 都指 images：图片已经解压就位，
# place_image 对已存在的目标直接返回 exists，不会动文件，step0 只读 db 写 json。
set -u
export TZ=Asia/Shanghai
N=/mnt/sfs_turbo/datasets/nuplan-v1.1
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
LOG=$W/step0_rerun.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

say "===== step0 补齐开始 ====="
mkdir -p "$N/_full/db_subset"
linked=0; nodb=0
for d in "$N"/images/*/; do
  lg=$(basename "$d")
  [ -d "$N/images/$lg/CAM_F0" ] || continue
  src="$N/db/$lg.db"
  if [ ! -f "$src" ]; then nodb=$((nodb+1)); continue; fi
  dst="$N/_full/db_subset/$lg.db"
  [ -L "$dst" ] || { ln -s "$src" "$dst" 2>/dev/null && linked=$((linked+1)); }
done
say "db 软链：新增 $linked，总计 $(ls "$N/_full/db_subset" | wc -l)，缺 db 的 log $nodb"
say "现有 json $(ls "$N/_full/json" | wc -l) 个，目标 $(ls -d "$N"/images/*/ | wc -l) 个"

say ">>> 跑 step0（只读 db 写 json，不动图片）"
t0=$(date +%s)
cd /mnt/sfs_turbo/zyc/DriveVLA-W0 && \
python3 tools/pickle_gen/nuplan_pickle_gen/step0_export_nuplan.py \
  --db-root "$N/_full/db_subset" \
  --blob-root "$N/images" \
  --image-dir "$N/images" \
  --json-dir "$N/_full/json" \
  --channel CAM_F0 --link symlink >> "$W/step0_rerun_raw.log" 2>&1
rc=$?
say "<<< step0 rc=$rc，耗时 $(( ($(date +%s)-t0)/60 )) 分钟，json 现有 $(ls "$N/_full/json" | wc -l) 个"
say "    原始输出尾部：$(tail -3 "$W/step0_rerun_raw.log" | tr '\n' ' | ')"
say "STEP0DONE"

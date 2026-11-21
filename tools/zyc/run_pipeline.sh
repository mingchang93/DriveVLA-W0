#!/bin/bash
# 跑 nuPlan 预训练 pickle 流水线（step1 -> 2 -> 3a -> 3b -> 4），用改好路径的副本。
#
# 副本在 /mnt/sfs_turbo/zyc/drivevla-w0-zyc/pipeline/，由 prep_pipeline.py 生成：
# 路径指向我们的目录、产物写 _full2（保留 _full 里那份 373-log 的旧 pickle 作退路）、
# step1 去掉 VQ 门槛、EXCLUDE_DIR 指到按 metric_cache 建的 navtest 排除目录。
#
# step0 已单独跑完：1105/1105 个 json，frames=3192986 missing_images=0 skipped=0。
#
# 每步都记耗时与产物规模。任何一步非 0 退出就停，不往下走——
# 后面几步都吃前面的产物，带着半截中间结果继续只会产出一个说不清的 pickle。
set -u
export TZ=Asia/Shanghai
P=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/pipeline
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
N=/mnt/sfs_turbo/datasets/nuplan-v1.1
LOG=$W/pipeline.log
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

cd /mnt/sfs_turbo/zyc/DriveVLA-W0 || exit 1
PY=/mnt/sfs_turbo/lmc/venv-hif8/bin/python
[ -x "$PY" ] || PY=python3

run_step() {   # 文件名  说明
  local f=$1 desc=$2
  say ">>> $desc（$f）"
  local t0=$(date +%s)
  "$PY" "$P/$f" >> "$W/pipeline_raw.log" 2>&1
  local rc=$?
  say "<<< $desc rc=$rc，耗时 $(( ($(date +%s)-t0)/60 )) 分钟"
  if [ "$rc" != "0" ]; then
    say "!!! $desc 失败，停止流水线。原始输出尾部："
    tail -15 "$W/pipeline_raw.log" | sed 's/^/      /' >> "$LOG"
    exit 1
  fi
}

say "===== 预训练 pickle 流水线开始（1105 个 log）====="
: > "$W/pipeline_raw.log"

run_step step1_segment_videos.py "step1 切片段"
say "    video_segments.json: $(du -h $N/_full2/intermediate/video_segments.json 2>/dev/null | cut -f1)"

run_step step2_generate_actions_seq_optimized.py "step2 生成 action/waypoint"
say "    actions: $(ls $N/_full2/intermediate/actions 2>/dev/null | wc -l) 个  waypoints: $(ls $N/_full2/intermediate/waypoints 2>/dev/null | wc -l) 个"

run_step step3a_analyze_displacement_distribution.py "step3a 位移分布/阈值"
run_step step3b_generate_commands.py "step3b 生成 command"
say "    commands: $(ls $N/_full2/intermediate/commands 2>/dev/null | wc -l) 个"

run_step step4_merge_pickle.py "step4 合并 pickle"
PKL=$N/_full2/output/nuplan_processed_data.pkl
say "    pickle: $(du -h $PKL 2>/dev/null | cut -f1)"

# 收尾自检：片段数、覆盖 log 数、以及确认 navtest 的 10 个 log 真的被排除了
"$PY" - >> "$LOG" 2>&1 <<'PYEOF'
import os, pickle
N = "/mnt/sfs_turbo/datasets/nuplan-v1.1"
W = "/mnt/sfs_turbo/zyc/drivevla-w0-zyc"
p = N + "/_full2/output/nuplan_processed_data.pkl"
d = pickle.load(open(p, "rb"))
items = d if isinstance(d, list) else list(d.values())
logs = set()
for x in items:
    im = x.get("image") if isinstance(x, dict) else None
    if im:
        s = im[0] if isinstance(im, (list, tuple)) else im
        logs.add(str(s).split("/")[0])
navtest = set(os.listdir(W + "/navtest_exclude"))
leak = sorted(logs & navtest)
print("    片段数 %d   覆盖 log %d" % (len(items), len(logs)))
print("    与 navtest 的交集: %d 个%s" % (len(leak), ("  " + ", ".join(leak[:3])) if leak else "（泄漏已过滤干净）"))
old = N + "/_full/output/nuplan_processed_data.pkl"
if os.path.exists(old):
    o = pickle.load(open(old, "rb"))
    print("    对照旧 pickle: 片段 %d -> %d（%+.1f%%）"
          % (len(o), len(items), 100.0 * (len(items) - len(o)) / len(o)))
PYEOF

say "PIPELINEDONE"

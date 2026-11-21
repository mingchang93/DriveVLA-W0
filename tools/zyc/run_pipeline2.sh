#!/bin/bash
# 跑 nuPlan 预训练 pickle 流水线（step1 -> 2 -> 3a -> 3b -> 4）。
#
# 用 conda 的 navsim 解释器，不用训练 venv：step2 需要 pyquaternion
# （四元数转位姿，真依赖，不能用替身），训练 venv 没有、conda navsim 有。
# 已用 ast + find_spec 在 conda 解释器下复查过全部 step 的顶层 import，缺失 0。
#
# step1 的产物若已存在且够大就跳过：它跑了 15 分 40 秒、产出 450 MB
# （1095 个 log，对比 373 个 log 时的 160 MB，比例合理），没必要重跑。
# 判据用文件大小而不是存在性 —— 第一轮 step1 "成功"过一次，产出却是空的 4 KB
# （逐帧要求 VQ 的 .npy 存在，而全盘没有 VQ，于是 315 万帧全判无效）。
#
# 任何一步非 0 退出就停：后面几步都吃前面的产物，带着半截中间结果继续
# 只会产出一个说不清的 pickle。
set -u
export TZ=Asia/Shanghai
P=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/pipeline
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
N=/mnt/sfs_turbo/datasets/nuplan-v1.1
LOG=$W/pipeline.log
PY=/mnt/sfs_turbo/zyc/miniconda3/envs/navsim/bin/python
say() { echo "[$(date +'%F %T')] $*" >> "$LOG"; }

cd /mnt/sfs_turbo/zyc/DriveVLA-W0 || exit 1
export PYTHONPATH="/mnt/sfs_turbo/zyc/DriveVLA-W0:/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/pickle_gen:/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/pickle_gen/nuplan_pickle_gen:${PYTHONPATH:-}"

run_step() {   # 文件名  说明
  local f=$1 desc=$2
  say ">>> $desc（$f）"
  local t0=$(date +%s)
  "$PY" "$P/$f" >> "$W/pipeline_raw.log" 2>&1
  local rc=$?
  say "<<< $desc rc=$rc，耗时 $(( ($(date +%s)-t0)/60 )) 分钟"
  if [ "$rc" != "0" ]; then
    say "!!! $desc 失败，停止流水线。原始输出尾部："
    tr -d '\r' < "$W/pipeline_raw.log" | grep -vE '^(处理序列|验证完整性)' | tail -12 | sed 's/^/      /' >> "$LOG"
    exit 1
  fi
}

say "===== 预训练 pickle 流水线（conda navsim 解释器）====="
say "python: $("$PY" -V 2>&1)"

SEG=$N/_full2/intermediate/video_segments.json
seg_mb=0
[ -f "$SEG" ] && seg_mb=$(( $(stat -c %s "$SEG") / 1048576 ))
if [ "$seg_mb" -gt 100 ]; then
  say ">>> step1 跳过：产物已存在且为 ${seg_mb} MB（大于 100 MB 的下限）"
else
  say ">>> step1 需要重跑：产物 ${seg_mb} MB"
  run_step step1_segment_videos.py "step1 切片段"
  say "    video_segments.json: $(du -h "$SEG" 2>/dev/null | cut -f1)"
fi

n_act=$(ls $N/_full2/intermediate/actions 2>/dev/null | wc -l)
if [ "$n_act" -gt 10000 ]; then
  say ">>> step2 跳过：actions 已有 $n_act 个（大于 10000 的下限）"
else
  say ">>> step2 需要重跑：actions 只有 $n_act 个"
  run_step step2_generate_actions_seq_optimized.py "step2 生成 action/waypoint"
fi
say "    actions: $(ls $N/_full2/intermediate/actions 2>/dev/null | wc -l) 个  waypoints: $(ls $N/_full2/intermediate/waypoints 2>/dev/null | wc -l) 个"

run_step step3a_analyze_displacement_distribution.py "step3a 位移分布/阈值"
run_step step3b_generate_commands.py "step3b 生成 command"
say "    commands: $(ls $N/_full2/intermediate/commands 2>/dev/null | wc -l) 个"

run_step step4_merge_pickle.py "step4 合并 pickle"
PKL=$N/_full2/output/nuplan_processed_data.pkl
say "    pickle: $(du -h $PKL 2>/dev/null | cut -f1)"

# 收尾自检：片段数、覆盖 log 数、navtest 泄漏是否真被过滤、与旧 pickle 的对比
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
# 官方 Emu3 预训练 state 反推的规模：8000 步 / 44.199 epoch x 全局 batch 96 = 17376
print("    官方反推规模 17376 片段，我们现在是它的 %.0f%%" % (100.0 * len(items) / 17376))
PYEOF

say "PIPELINEDONE"

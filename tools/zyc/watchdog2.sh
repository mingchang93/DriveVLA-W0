#!/bin/bash
# 看门狗 v2：盯四台机器的训练 + 解压 + 磁盘。每 15 分钟一轮，只在出问题时告警。
#
# v1 的教训：它只会按既定规则检查"训练卡住/非零退出/磁盘不足"，
# 却没有"该启动却迟迟没启动"这一条，于是眼看着两条链在 03:00 没起来、
# 如实记了 3 小时"阶段: 挂起"却一声没吭。v2 补上了存活性检查：
# 训练日志超过 N 分钟没更新就告警，不管它自称处于什么阶段。
set -u
export TZ=Asia/Shanghai
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
R=$W/runs
LOG=$W/watchdog.log
STATE=$W/.watchdog_state
STALL_MIN=${STALL_MIN:-12}     # 训练日志多少分钟没更新算卡住

note()  { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
alert() { echo "[$(date '+%F %T')] ALERT $*" | tee -a "$LOG"; }

declare -A TAG=( [a51]=hif8_196 [a52]=bf16 [a53]=hif8_168 [a54]=hif8_edge3 )

note "看门狗 v2 启动（每 15 分钟，只报问题）"
while true; do
  # ---- 磁盘 ----
  avail=$(df -BG /mnt/sfs_turbo 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}')
  if [ -n "${avail:-}" ]; then
    [ "$avail" -lt 500 ] && alert "磁盘只剩 ${avail}G"
  else
    alert "读不到磁盘余量"
  fi

  # ---- 磁盘守护是否报过警 ----
  if [ -f "$W/diskguard.log" ]; then
    recent=$(find "$W/diskguard.log" -newermt "-16 minutes" 2>/dev/null)
    [ -n "$recent" ] && grep -h "ALERT" "$W/diskguard.log" 2>/dev/null | tail -1 | grep -q . && {
      last_ag=$(grep -h "ALERT" "$W/diskguard.log" | tail -1)
      echo "$last_ag" | grep -qE "$(date '+%F %H')" && alert "磁盘守护: $(echo "$last_ag" | cut -c1-120)"
    }
  fi

  # ---- 四台训练 ----
  for role in a51 a52 a53 a54; do
    t=${TAG[$role]}
    rl=$W/run_${role}.log
    tl=$R/night_${t}/train.log

    [ -f "$rl" ] || { alert "$role 的 run 日志不存在（任务没起来？）"; continue; }
    grep -q "RUNABORT_${role}" "$rl" && { alert "$role 放弃启动（卡不足）"; continue; }
    if grep -q "RUNDONE_${role}" "$rl"; then
      note "$role ($t) 已全部完成: $(grep -E 'PDMS=' "$rl" | tail -2 | tr '\n' ' ' | cut -c1-160)"
      continue
    fi

    # 训练已结束？看有没有 ckpt 和非零退出
    if grep -q "<<< 训练 $t" "$rl"; then
      line=$(grep "<<< 训练 $t" "$rl" | tail -1)
      echo "$line" | grep -q "rc=0"   || alert "$role 训练非零退出: $(echo "$line" | cut -c1-130)"
      echo "$line" | grep -q "ckpt=0" && alert "$role 训练没产出 ckpt: $(echo "$line" | cut -c1-130)"
      note "$role ($t) 训练已完，进入评测: $(tail -1 "$rl" | cut -c1-120)"
      continue
    fi

    # 训练中：存活性 + 步数推进
    if [ ! -f "$tl" ]; then alert "$role ($t) 训练日志还没出现"; continue; fi
    stale=$(find "$tl" -mmin +$STALL_MIN 2>/dev/null)
    step=$(grep -oE '[0-9]+/4000' "$tl" 2>/dev/null | tail -1 | cut -d/ -f1)
    if [ -n "$stale" ]; then
      alert "$role ($t) 训练日志 ${STALL_MIN} 分钟没更新，停在第 ${step:-?}/4000 步"
    else
      prev=$(grep "^${role} " "$STATE" 2>/dev/null | awk '{print $2}')
      [ -n "$prev" ] && [ -n "$step" ] && [ "$step" = "$prev" ] && \
        alert "$role ($t) 步数 15 分钟没推进，停在第 $step 步"
      note "$role ($t) 第 ${step:-?}/4000 步"
    fi
    grep -v "^${role} " "$STATE" 2>/dev/null > "$STATE.new"
    echo "${role} ${step:-0}" >> "$STATE.new"
    mv -f "$STATE.new" "$STATE"
  done

  # ---- 解压 ----
  if [ -f "$W/extract.log" ]; then
    if grep -q EXTRACTDONE "$W/extract.log"; then
      note "解压已全部完成"
    else
      left=$(ls /mnt/sfs_turbo/zyc/nuplan-camera-assist/_tmp/*.zip 2>/dev/null | wc -l)
      # 解压在 a51、看门狗在 a52，进程表各机独立，pgrep 看不到对面。
      # 改用共享盘上 extract.log 的更新时间判断死活。
      dead=$(find "$W/extract.log" -mmin +90 2>/dev/null)
      note "解压：剩 $left 个包，images 下 $(ls /mnt/sfs_turbo/datasets/nuplan-v1.1/images 2>/dev/null | wc -l) 个 log，日志更新于 $(date -r "$W/extract.log" '+%H:%M')"
      [ -n "$dead" ] && [ "$left" -gt 0 ] && alert "还剩 $left 个包未解压，但 extract.log 已 90 分钟没更新"
    fi
  fi

  sleep 900
done

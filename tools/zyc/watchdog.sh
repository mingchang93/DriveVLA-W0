#!/bin/bash
# 夜链看门狗：每 15 分钟自查一次，不依赖任何外部会话。
#
# 为什么要有它：我的定时唤醒是会话级的——会话不在前台运行时根本不会触发，
# 昨晚 03:07/03:08 两个就没响。链本身能跑，但"链死了没人续"这个缺口得由
# 服务器自己兜住。
#
# 它做四件事：
#   1) 把状态写进一行，便于一眼看全（status 文件总是最新一行）
#   2) 链死了（卡上无训练进程、链日志没写完）就重新放行续跑，
#      续跑前先把已完成那把的 train.log 留存——链会 rm -rf 同名输出目录
#   3) 卡被别的进程占住导致链起不来，按既定流程腾卡（授权见 kill_cards.sh 注释）
#   4) 磁盘余量低于阈值就记告警，避免 ckpt 写不下把训练搞断
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
CHAIN=${CHAIN:?需要 CHAIN=chain_a54.sh}
CTR=${CTR:-zyc-w0}
MIN_FREE_G=${MIN_FREE_G:-300}
MACH=$(hostname -s 2>/dev/null || echo unknown)
base=${CHAIN%.sh}
LOG=$W/watchdog_$base.log
STATUS=$W/status_$base.txt
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

ntrain() { docker exec "$CTR" bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"' 2>/dev/null | tr -d ' \r'; }
ninfer() { docker exec "$CTR" bash -c 'ps -eo args= | grep -c "[i]nference_action"' 2>/dev/null | tr -d ' \r'; }
nchain() { docker exec "$CTR" bash -c "ps -eo args= | grep -c '[c]hain_a5'" 2>/dev/null | tr -d ' \r'; }
ncard()  { npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}'; }
freeg()  { df -BG /mnt/sfs_turbo | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; }
step() {
  for t in "$@"; do
    s=$(docker exec "$CTR" bash -c "tail -1 /tmp/$t.log 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+/4000' | tail -1")
    [ -n "$s" ] && { echo "$t $s"; return; }
  done
  echo "-"
}

say "===== 看门狗启动（每 15 分钟，链=$CHAIN）====="
while true; do
  done_marker=$(grep -cE 'CHAINA5[34]DONE' "$W/$base.log" 2>/dev/null)
  tr_n=$(ntrain); inf_n=$(ninfer); ch_n=$(nchain); card_n=$(ncard); fg=$(freeg)
  if [ "$CHAIN" = "chain_a54.sh" ]; then
    cur=$(step night_hif8_132 night_hif8_168)
  else
    cur=$(step w7b_hif8_168 w7b_bf16)
  fi
  line="[$(date +'%F %T')] $MACH 训练进程=$tr_n 推理=$inf_n 链=$ch_n 卡=$card_n 余量=${fg}G 当前=$cur 完成标记=$done_marker"
  echo "$line" > "$STATUS"
  echo "$line" >> "$LOG"

  # 链已完成：收工
  if [ "${done_marker:-0}" -gt 0 ]; then
    say "链已写完成标记，看门狗退出"
    exit 0
  fi

  # 磁盘告警
  if [ "${fg:-9999}" -lt "$MIN_FREE_G" ]; then
    say "!! 磁盘余量 ${fg}G 低于 ${MIN_FREE_G}G —— ckpt 可能写不下"
  fi

  # 链死了：没有训练也没有推理，链进程也没了，但完成标记还没写
  if [ "${tr_n:-0}" = "0" ] && [ "${inf_n:-0}" = "0" ] && [ "${ch_n:-0}" = "0" ]; then
    say "!! 链不在了但没有完成标记 —— 判定为异常中断，准备续起"
    say "   链日志末 3 行：$(tail -3 "$W/$base.log" 2>/dev/null | tr '\n' ' | ')"
    # 留存已完成那把的 train.log：链重跑会 rm -rf 同名输出目录
    ts=$(date +%H%M%S)
    for d in "$W"/runs/aligned_hif8_168 "$W"/runs/aligned_hif8_132 "$W"/runs/w7b_bf16 "$W"/runs/w7b_hif8_168; do
      [ -f "$d/train.log" ] && cp "$d/train.log" "$W/rescued_$(basename "$d")_$ts.train.log" 2>/dev/null
    done
    say "   已留存现有 train.log（前缀 rescued_*_$ts）"
    # 卡被别人占着就先腾出来
    if [ "${card_n:-0}" -gt 0 ]; then
      say "   卡上有 $card_n 个进程，先腾卡"
      LOG="$LOG" bash "$T/kill_cards.sh"
      sleep 15
    fi
    docker exec -d "$CTR" env GATED=1 setsid bash "$T/$CHAIN"
    sleep 180
    say "   续起后训练进程=$(ntrain) 卡=$(ncard)"
  fi

  sleep 900
done

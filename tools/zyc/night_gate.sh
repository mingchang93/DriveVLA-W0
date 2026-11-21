#!/bin/bash
# 宿主侧门闸：等到下一个 START_HOUR 点 -> 清卡 -> 放行容器里的夜链 -> 确认真的跑起来了。
#   用法: CHAIN=chain_a54.sh bash night_gate.sh
#
# 分两层的原因：训练要在容器 zyc-w0 里跑（venv/torch 在那边），
# 但清卡必须在宿主做——容器看不到别的容器的进程，在里面 kill 清不掉。
#
# 等待判据用绝对时间戳，不比小时数。
# 曾经写成 while [ "$(date +%-H)" -lt 3 ]，20 点时 20 < 3 为假，一次没等就放行，
# 20:57 就把训练起来了，差 3 分钟撞上同事用卡。
#
# 也不用 at/cron 的长 sleep：上一晚 03:00 的定时链就是在 5 小时 sleep 里静默死掉的，
# 日志看着像在等、其实进程早没了。每分钟醒一次 + 每半小时心跳，断了一眼可见。
#
# 放行之后还要自证：docker exec -d 是异步的，返回 0 不代表链真的起来了
# （容器被重启、镜像里少东西、脚本语法错，都会静默失败）。
# 所以放行后查链日志与训练进程，不对就重试一次，仍不对就把失败原因写清楚。
set -u
export TZ=Asia/Shanghai
T=/mnt/sfs_turbo/zyc/DriveVLA-W0/tools/zyc
W=/mnt/sfs_turbo/zyc/drivevla-w0-zyc
CHAIN=${CHAIN:?需要 CHAIN=chain_a54.sh}
START_HOUR=${START_HOUR:-3}
CTR=${CTR:-zyc-w0}
MACH=$(hostname -s 2>/dev/null || echo unknown)
LOG=$W/gate_${CHAIN%.sh}.log
CHAINLOG=$W/${CHAIN%.sh}.log
say() { echo "[$(date +'%F %T')] [$MACH] $*" >> "$LOG"; }

train_running() {
  docker exec "$CTR" bash -c 'ps -eo args= | grep -c "[t]rain_qwen_vla"' 2>/dev/null | tr -d ' \r'
}
chain_running() {
  docker exec "$CTR" bash -c "ps -eo args= | grep -c '[c]hain_a5'" 2>/dev/null | tr -d ' \r'
}
handoff() {
  # 容器没在跑就先拉起来，否则 docker exec 必败
  local st; st=$(docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null)
  if [ "$st" != "true" ]; then
    say "!! 容器 $CTR 不在运行（state=${st:-查不到}），尝试 docker start"
    docker start "$CTR" >/dev/null 2>&1
    sleep 20
    st=$(docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null)
    say "   docker start 后 state=${st:-查不到}"
    [ "$st" = "true" ] || { say "!! 容器拉不起来，放行失败"; return 1; }
  fi
  docker exec -d "$CTR" env GATED=1 setsid bash "$T/$CHAIN"
  return 0
}

target=$(date -d "today ${START_HOUR}:00:00" +%s 2>/dev/null)
now=$(date +%s)
[ -z "$target" ] && { say "!! date -d 不可用，放弃（不敢盲放行）"; exit 1; }
[ "$now" -ge "$target" ] && target=$(date -d "tomorrow ${START_HOUR}:00:00" +%s)

say "===== 门闸挂起 ====="
say "现在 $(date +'%F %T')，目标 $(date -d "@$target" +'%F %T')，等 $(( (target-now)/60 )) 分钟"
say "到点后：清卡 -> 放行 $CHAIN -> 自证已启动"

tick=0
while [ "$(date +%s)" -lt "$target" ]; do
  tick=$((tick+1))
  if [ $((tick % 30)) -eq 1 ]; then
    say "心跳 $(date +%H:%M)：还剩 $(( (target-$(date +%s))/60 )) 分钟，卡上进程 $(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')"
  fi
  sleep 60
done

# 清到干净为止：对方若有守护进程会把任务重新拉起，清一次不够。
# 授权来自用户「有同事进程也 kill 必须跑起来」。每轮都重新枚举占卡 PID。
clear_cards_hard() {
  local round
  for round in 1 2 3 4 5; do
    local n; n=$(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')
    if [ "${n:-0}" = "0" ]; then
      [ "$round" = "1" ] && say "   卡本来就是空的" || say "   第 $round 轮后卡已清空"
      return 0
    fi
    say "   第 $round 轮：卡上仍有 $n 个进程，继续清"
    LOG="$LOG" bash "$T/kill_cards.sh"
    sleep 15
  done
  say "!! 清了 5 轮仍有 $(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}') 个进程占卡——可能有守护进程在拉起，训练大概率起不来"
  return 1
}

say ">>> 到点（$(date +'%F %T')），清卡"
clear_cards_hard

for attempt in 1 2 3; do
  say ">>> 第 $attempt 次放行 $CHAIN"
  [ "$attempt" -gt 1 ] && { say "   重试前再清一次卡"; clear_cards_hard; }
  if ! handoff; then say "!! 放行调用本身失败"; sleep 30; continue; fi
  # 链起来要先加载 16G 模型，给足 4 分钟
  ok=0
  for i in $(seq 1 24); do
    sleep 10
    c=$(chain_running); t=$(train_running)
    if [ "${t:-0}" -gt 0 ]; then
      say "<<< 已确认启动：链进程 ${c:-?}，训练进程 $t，卡上进程 $(npu-smi info 2>/dev/null | grep -cE '^\| [0-9]+ +\| +[0-9]{4,}')"
      ok=1; break
    fi
  done
  if [ "$ok" = 1 ]; then
    say "链日志首行：$(head -1 "$CHAINLOG" 2>/dev/null)"
    say "链日志末行：$(tail -1 "$CHAINLOG" 2>/dev/null)"
    say "GATEDONE_$CHAIN"
    exit 0
  fi
  say "!! 4 分钟内没看到训练进程。链进程=$(chain_running)  链日志末行：$(tail -1 "$CHAINLOG" 2>/dev/null)"
  say "   容器内链的 stdout（若有）：$(docker exec "$CTR" bash -c "tail -3 /tmp/night_hif8_168.log /tmp/w7b_bf16.log 2>/dev/null | tail -3" 2>/dev/null)"
done

say "!!! 两次放行都没能确认启动——需要人工介入。排查顺序："
say "    1) docker ps 看 $CTR 在不在"
say "    2) docker exec $CTR bash -n $T/$CHAIN 看语法"
say "    3) cat $CHAINLOG 看链自己写了什么"
say "GATEFAILED_$CHAIN"

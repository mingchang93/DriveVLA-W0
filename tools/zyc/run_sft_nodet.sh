#!/bin/bash
# NAVSIM 2VA SFT（无 nuPlan 预训练，从通用 Qwen3-VL-2B 直接训）
#
# 对应论文 Table 5 的 `—/2VA = 80.7` 那一档。repo 没有"不跑预训练"的脚本
# （`train_base_ar_withou_moe.sh` 的 --model_name_or_path 指向阶段一的输出目录），
# 所以超参取自该脚本，再按"冷启动"调两处，逐条依据写在下面。
#
# ── 取自 repo `scripts/scripts_train/train_base_ar_withou_moe.sh` ──
#   max_steps 4000          （NAVSIM 上 1.86 epoch，103,288 场景 ÷ (6×8)）
#   learning_rate 8e-5
#   min_learning_rate 1e-6 / weight_decay 0.1 / max_grad_norm 5.0
#   adam β(0.9, 0.95) / eps 1e-6 / lr_scheduler cosine_with_min_lr
#   （null_prompt_prob 0.15 不传：那是 Emu3 的参数，Qwen 的 DataArguments 里没有
#     这个字段，整个 Qwen 路径零引用——不是漏传，是这条线没有这个机制。）
#   max_position_embeddings 1400  ← 训练脚本的直接证据，不只是推理脚本
#   gradient_accumulation_steps 1 / seed 42 / dataloader_num_workers 12
#   世界模型开：apply_loss_on_only_vision True / only_action False
#     —— 五个 repo 脚本（阶段一 + 四个阶段二）全部如此，世界模型从未被关过。
#        推理侧不受影响：评测 runner 加载普通 Qwen3VLForConditionalGeneration，
#        denoiser.*/vae.* 以 unexpected-key 忽略。已用 20 步 ckpt 实测走通
#        （21 场景 → 21 JSON → PDMS 41.13）。
#
# ── 两处按冷启动调整 ──
#   warmup_steps 50 / per_device_train_batch_size 6
#     六个官方脚本审计结论：预训练一律 warmup 400 + bs4，四个 SFT 脚本一律
#     warmup 50（无例外）+ bs 6 或 12。早前我把 warmup 改成 400 的理由是
#     "冷启动要更长 warmup"，那是推断、repo 无依据，且证据上也站不住：
#     lmc 的 smoke 用 warmup 2 + lr 5e-5 正常收敛，今早用 warmup 400 + lr 2e-4
#     也正常——模型对 warmup 不敏感。故一律按 repo。
#     全局 batch = 6 × 8 × 1 = 48，与 repo 一致。
#
# ── tf32 ──
#   repo 是 True，Ascend 上没有 tf32，设 False（lmc 的脚本也是 False）。
set -u

ROOT=/mnt/sfs_turbo/zyc/DriveVLA-W0
OUT=${OUT:-/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/sft_navsim_2va_bf16}
DATA=/mnt/sfs_turbo/zjj/datasets/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl
CAM=/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/trainval
NPROC=${NPROC:-8}
BS=${BS:-6}
GA=${GA:-1}
STEPS=${STEPS:-4000}
PORT=${PORT:-23501}
HCCL_RANGE=${HCCL_RANGE:-17500-17599}
DS=${DS:-scripts/sft/zero2.json}   # 可换 zero0/zero1/zero2_offload
LOGSTEP=${LOGSTEP:-1}                # 逐步打日志：loss 曲线与确定性比对都要用
MODEL=${MODEL:-/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct}   # 骨干
#   Qwen3-VL-2B-Instruct   -> 我们一直跑的那套（model_type=qwen3_vl）
#   Qwen2.5-VL-7B-Instruct -> 官方 Qwen_PDMS_85.3 的底模（model_type=qwen2_5_vl）
# train_qwen_vla.py 读 config 的 model_type 自动选家族与 attn 实现：
#   qwen3_vl -> Qwen3VLForConditionalGenerationROSS + sdpa
#   其他     -> Qwen2_5_VLForConditionalGenerationROSS + flash_attention_2
HIF8=${HIF8:-}                      # 空=bf16 基线臂；cts=HiF8 current-tensor-scaling
HIF8_INTERVAL=${HIF8_INTERVAL:-10}   # DTS 的刷新间隔，仅 --hif8 dts 时生效
SEED=${SEED:-42}                     # 同配置换种子可测 run-to-run 噪声底
# 重计算开关。默认 False —— 用户决定预训练与 SFT 从现在起都关重计算。
# 关掉快 8.7%（实测 3.134 -> 2.863 s/步），HBM 32% -> 68%，仍有余量。
# 注意：开/关重计算不逐位一致（分叉起于 float32 最低位，13 步内 grad_norm
# 放大到 5.3e-2），所以关掉后的结果与此前开着跑的那批不可同表比较。
GC=${GC:-False}
HIF8_IGNORE=${HIF8_IGNORE:-lm_head,o_proj}   # 决定量化数量：
#   lm_head                  -> 196   o_proj 也量化
#   lm_head,o_proj           -> 168   客户规格基线
#   lm_head,down_proj        -> 168   swapDownO，bf16 预算从 o_proj 换到 down_proj
#   lm_head,o_proj,down_proj -> 140   偏差最低，量化率 71%

cd "$ROOT" || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ── 确定性：必须在 torchrun 之前 export ──
# 代码里 set_reproducibility() 也会 setdefault 这几个，但那时 torch/torch_npu/HCCL
# 已经初始化完了，对已建立的通信域可能不生效；PYTHONHASHSEED 在解释器启动后设
# 也只影响 fork 出来的 dataloader worker。所以在这里 export，让每个 rank 从
# 一开始就带着。代码里的 setdefault 保留作兜底。
# 确定性：必须在 torchrun 之前 export。
# 来源：lmc 的 emu3-npu-port 分支 PRECISION_ALIGNMENT.md 第 3 章（593 行的精度对齐指南）。
# 我们最初只设了前两个，结果两把同配置跑从第 3-4 步就分歧；后六个是关键补充，
# 其中 CLOSE_MATMUL_K_SHIFT 和 TASK_QUEUE_ENABLE=0 最可能是元凶——matmul 的 K 轴
# 重排会改变累加顺序，异步任务队列会重排算子派发，两者都解释"前向相同、反向分歧"。
export HCCL_DETERMINISTIC=TRUE        # 集合通信归约顺序固定
export INF_NAN_MODE_ENABLE=1          # 非饱和：溢出产生 Inf/NaN（对 FP8 尤其关键）
export CLOSE_MATMUL_K_SHIFT=1         # 关 MatMul 的 K 轴重排（改变累加顺序）
export ATB_MATMUL_SHUFFLE_K_ENABLE=0  # 关 ATB matmul 的 shuffle
export ACL_OP_DETERMINISTIC=1         # 昇腾算子库确定性模式
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-0}   # 关异步任务队列（异步派发会重排算子）
# 默认仍是 0（行为不变）。置 1 可测它到底是不是确定性的必要条件：
# profiling 显示它占单步时间约 40%（889/2204 ms，散在约 15,650 个亚毫秒空隙里），
# 而确定性的真正根因是播种顺序，不是这些环境变量。
export FLAGS_npu_storage_format=0     # 关私有存储格式
export ASCEND_LAUNCH_BLOCKING=0       # 同步派发只在对齐排查时需要；确定性已由
                                      # 「建模前播种」解决，关掉它换回吞吐。
                                      # 关掉后的确定性由 bench_zero_100step.sh
                                      # 的 z1a/z1b 两把 100 步复验。
# ── 总开关：DETERMINISM=off 时把上面的确定性设置翻到性能默认值 ──
# 默认 on，行为不变。off 只用于量「确定性的代价」：profiling 显示
# 单步 2204 ms 里，889 ms 是 TASK_QUEUE_ENABLE=0 导致的同步下发开销，
# 267 ms 是 F.interpolate 被分解后 aten::index 回退 AI_CPU，合计超过一半。
# INF_NAN_MODE_ENABLE 不在此列：它是数值语义（溢出给 Inf/NaN 而非饱和），
# 对 FP8 关键，翻掉会改变算法本身。
DETERMINISM=${DETERMINISM:-on}
DET_FLAG=True
if [ "$DETERMINISM" = "off" ]; then
  export HCCL_DETERMINISTIC=FALSE
  export CLOSE_MATMUL_K_SHIFT=0
  export ATB_MATMUL_SHUFFLE_K_ENABLE=1
  export ACL_OP_DETERMINISTIC=0
  export TASK_QUEUE_ENABLE=1
  export FLAGS_npu_storage_format=1
  DET_FLAG=False
fi

export PYTHONHASHSEED=42

export HCCL_NPU_SOCKET_PORT_RANGE=$HCCL_RANGE
export HCCL_CONNECT_TIMEOUT=1800
ulimit -l unlimited 2>/dev/null || true
mkdir -p "$OUT"

echo "[$(date '+%F %T')] SFT 启动于 $(hostname -s)" | tee "$OUT/train.log"
echo "  卡数 $NPROC · bs $BS · ga $GA · 全局 batch $((BS*NPROC*GA)) · $STEPS 步 · ds=$DS" | tee -a "$OUT/train.log"
echo "  数据 $DATA" | tee -a "$OUT/train.log"
echo "  backbone $MODEL" | tee -a "$OUT/train.log"
echo "  hif8=${HIF8:-off} ignore=$HIF8_IGNORE interval=$HIF8_INTERVAL seed=$SEED" | tee -a "$OUT/train.log"
echo "  确定性 DETERMINISM=$DETERMINISM deterministic=$DET_FLAG HCCL=$HCCL_DETERMINISTIC ACL_OP=$ACL_OP_DETERMINISTIC TASK_QUEUE=$TASK_QUEUE_ENABLE STORAGE_FMT=$FLAGS_npu_storage_format K_SHIFT=$CLOSE_MATMUL_K_SHIFT" | tee -a "$OUT/train.log"


torchrun --nproc_per_node="$NPROC" --master_port="$PORT" utils/train_qwen_vla.py \
  --model_name_or_path "$MODEL" \
  --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5/unet \
  --dataset_type navsim2va_ross \
  --data_path "$DATA" \
  --data_root "$CAM" \
  --action_tokenizer_path configs/fast \
  --hif8 "$HIF8" \
  --hif8_ignore "$HIF8_IGNORE" \
  --hif8_interval "$HIF8_INTERVAL" \
  --actions_format fast --action_dim 3 \
  --enable_world_model True \
  --cur_frame_idx 3 --future_nums 8 --use_previous_actions True \
  --max_pixels 147456 --min_pixels 12544 --model_max_length 8192 \
  --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision True \
  --deepspeed "$DS" \
  --output_dir "$OUT" \
  --bf16 True --tf32 False --optim adamw_torch \
  --learning_rate 1e-5 --weight_decay 0.001 \
  --lr_scheduler_type cosine \
  --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
  --max_steps "$STEPS" --warmup_steps 200 \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --gradient_checkpointing "$GC" \
  --dataloader_num_workers 12 --dataloader_pin_memory True --dataloader_shuffle True \
  --seed "$SEED" --deterministic "$DET_FLAG" --remove_dropout True \
  --save_strategy no \
  --eval_strategy no --logging_steps "$LOGSTEP" \
  --report_to tensorboard 2>&1 | tee -a "$OUT/train.log"

echo "TRAIN_EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/train.log"
echo "[$(date '+%F %T')] 结束" | tee -a "$OUT/train.log"
du -sh "$OUT" 2>/dev/null | tee -a "$OUT/train.log"

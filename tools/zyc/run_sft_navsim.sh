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
#   null_prompt_prob 0.15   （两个 repo 脚本都有，早前的脚本漏了这一条）
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

cd "$ROOT" || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ── 确定性：必须在 torchrun 之前 export ──
# 代码里 set_reproducibility() 也会 setdefault 这几个，但那时 torch/torch_npu/HCCL
# 已经初始化完了，对已建立的通信域可能不生效；PYTHONHASHSEED 在解释器启动后设
# 也只影响 fork 出来的 dataloader worker。所以在这里 export，让每个 rank 从
# 一开始就带着。代码里的 setdefault 保留作兜底。
export HCCL_DETERMINISTIC=TRUE      # 多卡 all-reduce 的归约顺序固定
export INF_NAN_MODE_ENABLE=1        # 非饱和模式：溢出产生 Inf/NaN 而非饱和。
                                    # 对 FP8 尤其关键——饱和会把溢出静默吃掉，
                                    # 正好掩盖我们要找的量化问题。
export PYTHONHASHSEED=42

export HCCL_NPU_SOCKET_PORT_RANGE=$HCCL_RANGE
export HCCL_CONNECT_TIMEOUT=1800
ulimit -l unlimited 2>/dev/null || true
mkdir -p "$OUT"

echo "[$(date '+%F %T')] SFT 启动于 $(hostname -s)" | tee "$OUT/train.log"
echo "  卡数 $NPROC · bs $BS · ga $GA · 全局 batch $((BS*NPROC*GA)) · $STEPS 步" | tee -a "$OUT/train.log"
echo "  数据 $DATA" | tee -a "$OUT/train.log"
echo "  确定性 HCCL_DETERMINISTIC=$HCCL_DETERMINISTIC INF_NAN_MODE_ENABLE=$INF_NAN_MODE_ENABLE PYTHONHASHSEED=$PYTHONHASHSEED" | tee -a "$OUT/train.log"

torchrun --nproc_per_node="$NPROC" --master_port="$PORT" utils/train_qwen_vla.py \
  --model_name_or_path /mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5/unet \
  --dataset_type navsim2va_ross \
  --data_path "$DATA" \
  --data_root "$CAM" \
  --action_tokenizer_path configs/fast \
  --actions_format fast --action_dim 3 \
  --cur_frame_idx 3 --future_nums 8 --use_previous_actions True \
  --max_pixels 147456 --min_pixels 12544 --model_max_length 1400 \
  --null_prompt_prob 0.15 \
  --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision True \
  --deepspeed scripts/sft/zero2.json \
  --output_dir "$OUT" \
  --bf16 True --tf32 False --optim adamw_torch \
  --learning_rate 8e-5 --weight_decay 0.1 \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
  --max_steps "$STEPS" --warmup_steps 50 \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --gradient_checkpointing True \
  --dataloader_num_workers 12 --dataloader_pin_memory True --dataloader_shuffle True \
  --seed 42 --deterministic True --remove_dropout True \
  --save_strategy no \
  --eval_strategy no --logging_steps 10 \
  --report_to tensorboard 2>&1 | tee -a "$OUT/train.log"

echo "TRAIN_EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/train.log"
echo "[$(date '+%F %T')] 结束" | tee -a "$OUT/train.log"
du -sh "$OUT" 2>/dev/null | tee -a "$OUT/train.log"

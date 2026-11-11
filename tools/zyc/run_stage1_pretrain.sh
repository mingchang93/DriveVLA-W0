#!/bin/bash
# 阶段一 · 预训练（2VA，联合损失 action + ross）
#
# 决定依据：
#   - 2VA 而非 6VA：开源的 ViT/扩散头三条 assert 硬限两帧（原作者 `3c10d99 init` 写的），
#     且作者自己的 Qwen checkpoint 名是 `..._2va_nuplan_pretrained_8k_2va`，两头都是 2VA。
#     6VA 只在 Emu3/VQ 那条线上有脚本。
#   - model_max_length 1400：我们是 **2VA**，不是 6VA。
#     ViT/扩散分支（ROSS 头）硬限两帧——modeling_qwen2_5_vl_ross.py:335-337 有三条
#     assert image_hidden.shape[1] == 2，是原作者 commit 3c10d99 init 写的；
#     Qwen 路径只有 navsim2va_ross / nuplan2va_ross 两个 dataset，都是 2VA；
#     作者自己的 ckpt 名 ..._2va_nuplan_pretrained_8k_2va 两头也都是 2VA。
#     6VA 只存在于 Emu3/VQ 分支（train_nuplan_6va_*.sh + Emu3DrivingNuplan6VADataset），
#     那边的 max_position_embeddings=4000 是为装六对 VA，与我们无关。
#   - max_pixels 147456：取自开源 inference/qwen/inference_2va.sh，
#     不用 lmc smoke 的 150528 / 2048。bf16 与后续 hif8 两条臂必须用同一套。
#   - 与 Emu3 释放的预训练配置对齐（`train_nuplan_6va_multi_v0.2_master.sh`）：
#       lr 2e-4 · warmup 400 · 8000 步 · 全局 batch 96
#     那边是 3 机 × 8 卡 × mb4 = 96（WORLD_SIZE=3）。我们单机 8 卡，用
#     mb12 × 8 × ga1 = 96 等价；mb12 跑不下就 BS=6 GA=2（同样 96）。
#   - 窗口是 2VA、model_max_length 1400：Emu3 那边 6VA 用 4000，与我们无关。
#   - 原注释（已不适用）：lr 2e-4 / max_steps 8000 / warmup 400 / bs4 取自 6VA
#     预训练脚本的值（lmc smoke 用的 5e-5 是阶段二带 action expert 那组的值，不是预训练的）。
#   - use_previous_actions True：开源两个训练脚本与 inference_2va.sh 都是 True，
#     关掉会让训练与推理的序列结构不一致。
#   - 不存中间 ckpt：已验证 save_strategy=no 在训练结束时仍写出完整权重
#     （1569 个张量、6.77 GB、denoiser.* 与 vae.* 都在），阶段二够用。
#   - 动作头：无 action expert（train_qwen_vla.py 零处引用 models/policy_head），
#     对应开源 train_base_ar_withou_moe.sh 那一种；阶段二的 lr 应为 8e-5。
#   - 数据用官方全量 trainval pickle（103,288 场景）。已实测：1192 个 log 全在 lmc
#     下载的 1310 个里，随机 300 场景的 7199 张图 100% 命中。
set -u

ROOT=/mnt/sfs_turbo/zyc/DriveVLA-W0
OUT=${OUT:-/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/pretrain_nuplan_2va_bf16}
NPROC=${NPROC:-8}
BS=${BS:-12}            # 微批；跑不下退 BS=6 GA=2，全局 batch 不变
GA=${GA:-1}
STEPS=${STEPS:-8000}
DATA=/mnt/sfs_turbo/zjj/datasets/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl
CAM=/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/trainval

cd "$ROOT" || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# a52 上还有 zhaobo / wy / lh / k8s 的容器，给本作业一个独立的 HCCL 端口段，
# 避免撞端口（HiF8 阶段踩过：两个作业共用默认段会在 HcclBroadcast 直接失败）。
# 确定性：必须在 torchrun 之前 export（代码里的 setdefault 对已初始化的 HCCL 不生效）
export HCCL_DETERMINISTIC=TRUE
export INF_NAN_MODE_ENABLE=1
export PYTHONHASHSEED=42
export HCCL_NPU_SOCKET_PORT_RANGE=17100-17199
export HCCL_CONNECT_TIMEOUT=1800
ulimit -l unlimited 2>/dev/null || true
mkdir -p "$OUT"

echo "[$(date '+%F %T')] 阶段一启动于 $(hostname -s)" | tee "$OUT/train.log"
echo "  数据: $DATA" | tee -a "$OUT/train.log"
echo "  输出: $OUT" | tee -a "$OUT/train.log"
echo "  卡 $NPROC · mb $BS · ga $GA · 全局 batch $((BS*NPROC*GA)) · $STEPS 步" | tee -a "$OUT/train.log"

torchrun --nproc_per_node="$NPROC" --master_port=${PORT:-23493} utils/train_qwen_vla.py \
  --model_name_or_path /mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5/unet \
  --dataset_type navsim2va_ross \
  --data_path "$DATA" \
  --data_root "$CAM" \
  --action_tokenizer_path configs/fast \
  --actions_format fast --action_dim 3 \
  --cur_frame_idx 3 --future_nums 8 --use_previous_actions True \
  --max_pixels 147456 --min_pixels 12544 --model_max_length 1400 \
  --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision True \
  --deepspeed scripts/sft/zero2.json \
  --output_dir "$OUT" \
  --bf16 True --tf32 False --optim adamw_torch \
  --learning_rate 2e-4 --weight_decay 0.1 \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
  --max_steps "$STEPS" --warmup_steps 400 \
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
ls -d "$OUT"/checkpoint-* 2>/dev/null | tee -a "$OUT/train.log"

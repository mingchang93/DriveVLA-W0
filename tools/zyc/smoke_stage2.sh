#!/bin/bash
# 阶段二开关 smoke：验证 --enable_world_model false 这条从未跑过的路径。
# 预期：world_model: False / 不实例化 denoiser+vae / 只有 action_loss 无 ross_loss /
#       可训练参数掉到约 2129M（砍掉 860M UNet）/ 能存出 ckpt。
set -u
ROOT=/mnt/sfs_turbo/zyc/DriveVLA-W0
OUT=/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/smoke_stage2_wmoff
cd "$ROOT" || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE=17200-17299
export HCCL_CONNECT_TIMEOUT=1800
ulimit -l unlimited 2>/dev/null || true
rm -rf "$OUT"; mkdir -p "$OUT"
echo "[$(date '+%F %T')] 阶段二开关 smoke" | tee "$OUT/train.log"
torchrun --nproc_per_node=8 --master_port=23495 utils/train_qwen_vla.py \
  --model_name_or_path /mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct \
  --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5/unet \
  --enable_world_model false \
  --dataset_type navsim2va_ross \
  --data_path /mnt/sfs_turbo/lmc/navsim_openscene/navsim_trainval_subset.pkl \
  --data_root /mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/trainval \
  --action_tokenizer_path configs/fast \
  --actions_format fast --action_dim 3 \
  --cur_frame_idx 3 --future_nums 8 --use_previous_actions True \
  --max_pixels 147456 --min_pixels 12544 --model_max_length 1400 \
  --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision True \
  --deepspeed scripts/sft/zero2.json \
  --output_dir "$OUT" \
  --bf16 True --tf32 False --optim adamw_torch \
  --learning_rate 8e-5 --weight_decay 0.1 \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
  --max_steps 20 --warmup_steps 2 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 --dataloader_pin_memory True --dataloader_shuffle True \
  --seed 42 --save_strategy no --eval_strategy no --logging_steps 1 \
  --report_to tensorboard 2>&1 | tee -a "$OUT/train.log"
echo "TRAIN_EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/train.log"

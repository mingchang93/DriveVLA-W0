#!/usr/bin/env bash
# Push per-device batch size until OOM, sampling HBM. Untracked runner.
cd /mnt/sfs_turbo/lmc/DriveVLA-W0
export DEVICE=npu HCCL_DETERMINISTIC=TRUE INF_NAN_MODE_ENABLE=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PWD/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"

HBM_LOG=logs/bs_ceiling_hbm_20261020.log
: > "$HBM_LOG"
( while :; do
    echo "SAMPLE $(date -u +%H:%M:%S)"
    npu-smi info 2>/dev/null | grep -E "/ 98304"
    sleep 20
  done ) >> "$HBM_LOG" 2>&1 &
SAMPLER=$!

for BS in 32 64 128 256; do
  echo "===== BATCH SIZE $BS  start $(date -u +%H:%M:%S) ====="
  torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=$((23600+BS)) \
    utils/train_qwen_vla.py \
    --model_name_or_path /mnt/sfs_turbo/models/Qwen2.5-VL-7B-Instruct \
    --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5/unet \
    --dataset_type navsim2va_ross \
    --data_path logs/synth_qwen_data/synth_navsim.pkl \
    --data_root logs/synth_qwen_data/images \
    --action_tokenizer_path "$PWD/configs/fast" \
    --actions_format fast --action_dim 3 --cur_frame_idx 3 --future_nums 8 \
    --use_previous_actions false \
    --max_pixels 150528 --min_pixels 12544 --model_max_length 4096 \
    --tune_mm_llm true --tune_mm_mlp true --tune_mm_vision true \
    --deepspeed "$PWD/scripts/sft/zero3_offload.json" \
    --output_dir "$PWD/logs/bs_ceiling_20261020/bs$BS/navsim" \
    --bf16 True --fp16 False --tf32 False --attn_type sdpa \
    --learning_rate 1e-4 --weight_decay 0.1 \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
    --max_steps 20 --warmup_steps 1 \
    --per_device_train_batch_size $BS --gradient_accumulation_steps 1 \
    --gradient_checkpointing True --dataloader_num_workers 12 --dataloader_shuffle true \
    --seed 42 --save_strategy no --eval_strategy no --logging_steps 1 --report_to none
  rc=$?
  echo "===== BATCH SIZE $BS  end $(date -u +%H:%M:%S) rc=$rc ====="
  if [ $rc -ne 0 ]; then echo "===== STOPPING: bs=$BS failed rc=$rc ====="; break; fi
done
kill $SAMPLER 2>/dev/null
echo "===== CEILING DONE ====="

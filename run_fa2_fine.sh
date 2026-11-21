#!/usr/bin/env bash
# Fine FA2 batch scan 42/44/46. Untracked runner.
cd /mnt/sfs_turbo/lmc/DriveVLA-W0 || exit 1
export DEVICE=npu HCCL_DETERMINISTIC=TRUE INF_NAN_MODE_ENABLE=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PWD/reference/Qwen2.5-VL/qwen-vl-finetune${PYTHONPATH:+:$PYTHONPATH}"
OFF="ROSS_NPU_RMSNORM=0 ROSS_NPU_SWIGLU=0 ROSS_NPU_ROPE=0"

run_one () {
  local tag="$1"; local bs="$2"
  echo "===== $tag  bs=$bs attn=fa2 start $(date -u +%H:%M:%S) ====="
  env $OFF torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=23950 \
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
    --deepspeed "$PWD/scripts/sft/zero2.json" \
    --output_dir "$PWD/logs/fa2_fine_20261021/$tag/navsim" \
    --bf16 True --fp16 False --tf32 False \
    --attn_type fa2 --optim adamw_torch \
    --learning_rate 1e-4 --weight_decay 0.1 \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
    --max_steps 50 --warmup_steps 1 \
    --per_device_train_batch_size "$bs" --gradient_accumulation_steps 1 \
    --gradient_checkpointing True --dataloader_num_workers 24 --dataloader_shuffle true \
    --seed 42 --save_strategy no --eval_strategy no --logging_steps 1 --report_to none
  local rc=$?
  echo "===== $tag  end $(date -u +%H:%M:%S) rc=$rc ====="
  return $rc
}

run_one fa2_42 42 || { echo "===== STOP ====="; echo "===== FA2_FINE_DONE ====="; exit 0; }
run_one fa2_44 44 || { echo "===== STOP ====="; echo "===== FA2_FINE_DONE ====="; exit 0; }
run_one fa2_46 46 || { echo "===== STOP ====="; echo "===== FA2_FINE_DONE ====="; exit 0; }
echo "===== FA2_FINE_DONE ====="

#!/bin/bash
# 阶段一 · 预训练（2VA，联合损失 action + ross）
#
# 决定依据：
#   - 2VA 而非 6VA：开源的 ViT/扩散头三条 assert 硬限两帧（原作者 `3c10d99 init` 写的），
#     且作者自己的 Qwen checkpoint 名是 `..._2va_nuplan_pretrained_8k_2va`，两头都是 2VA。
#     6VA 只在 Emu3/VQ 那条线上有脚本。
#   - model_max_length 8192：和我们的 SFT（run_sft_navsim.sh）以及官方 Qwen 2VA SFT
#     的 training_args.bin 一致。注意这跟 2VA/6VA 无关 —— 之前这里写 1400 是把两个
#     不同的参数混了：navsim 脚本的 --max_position_embeddings 1400（模型 config 的
#     位置编码数量）和 inference_2va.sh 的 MODEL_MAX_LEN=1400（推理侧 tokenizer 上限）。
#     预训练在 1400 截断、紧接的 SFT 在 8192 不截断，这个组合肯定是错的。
#     2VA 本身的依据仍然成立，见下：
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
#   - 窗口是 2VA。Emu3 那边 6VA 的 --max_position_embeddings 4000 与我们无关，
#     而且那同样不是 model_max_length。
#   - 原注释（已不适用）：lr 2e-4 / max_steps 8000 / warmup 400 / bs4 取自 6VA
#     预训练脚本的值（lmc smoke 用的 5e-5 是阶段二带 action expert 那组的值，不是预训练的）。
#   - use_previous_actions True：开源两个训练脚本与 inference_2va.sh 都是 True，
#     关掉会让训练与推理的序列结构不一致。
#   - 不存中间 ckpt：已验证 save_strategy=no 在训练结束时仍写出完整权重
#     （1569 个张量、6.77 GB、denoiser.* 与 vae.* 都在），阶段二够用。
#   - 动作头：无 action expert（train_qwen_vla.py 零处引用 models/policy_head），
#     对应开源 train_base_ar_withou_moe.sh 那一种；阶段二的 lr 应为 8e-5。
#   - 数据就是官方 pickle 本身（18,191 段）。我们自己复现的 _full3 是它的严格超集
#     （18,257 段，官方的 18,191 段一条不缺，多 66 段），log 集合完全相同。
#     「官方 17,376 段」是早前的错误记录，官方 Training.md 并无此数，实测 18,191。
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${OUT:-/mnt/sfs_turbo/zyc/drivevla-w0-zyc/runs/pretrain_nuplan_2va_bf16}
MODEL=${MODEL:-/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct}   # 骨干；也是补 processor 配置的来源
NPROC=${NPROC:-8}
BS=${BS:-12}            # 微批；跑不下退 BS=6 GA=2，全局 batch 不变
GA=${GA:-1}
# 重计算开关。默认 True（行为不变）。关掉省第二遍 forward，代价是激活全存、显存涨。
# HiF8 路径关掉还能省第二遍量化 —— profiling 显示那是 HiF8 主机开销的一部分来源。
GC=${GC:-False}
# 从已废弃的 run_stage1_pretrain_lrtest.sh 搬过来的参数（链条脚本依赖）。
# 默认值与本脚本原先写死的行为一致，所以不传参数的旧调用方行为不变。
# lr 不做参数：已定死 5e-5（实测 PDMS 0.8547；2.5e-5 得 0.8293、
# 1e-4 得 0.7054、2e-4 得 0.6803）。
SAVE_STRATEGY=${SAVE_STRATEGY:-no}
SAVE_STEPS=${SAVE_STEPS:-1000}
STEPS=${STEPS:-8000}
HIF8=${HIF8:-}                      # 空=bf16 基线臂；cts=HiF8 current-tensor-scaling
HIF8_INTERVAL=${HIF8_INTERVAL:-10}   # DTS 的 amax 刷新间隔，仅 --hif8 dts 生效。
# 之前漏传这个参数，--hif8 dts 会用代码默认间隔而不是 10，等于跑了另一个配置。
HIF8_IGNORE=${HIF8_IGNORE:-lm_head,o_proj}   # 168 = 客户规格（排除 lm_head + o_proj）
LOGSTEP=${LOGSTEP:-1}
PIN_MEMORY=${DATALOADER_PIN_MEMORY:-True}
DS=${DS:-scripts/sft/zero1.json}   # 与 SFT 同档位；ZeRO-0 在 mb12 下显存放不下
# nuPlan 2VA 预训练数据：官方 pickle 本身，18,191 段 / 1,310 个 log。
# 规模对齐：官方预训练是 17,376 段（论文、repo、HF 三处一致），我们 +5.1%。
# 构成 = nuPlan camera train_set 43 包 + val_set 12 包全解压。只用 train_set 时
# 封顶 1,095 log / 15,895 段，比官方少 8.5%，这就是当初要补下 val_set 的原因。
# navtest 的 136 个评测 log 已按官方 EXCLUDE_DIR 机制排除，不会泄漏评测集。
# 动作已在 step4 里归一化过（normalize_actions 在 pickle.dump 之前），
# 训练侧不读归一化器（train_qwen_vla.py 与 data_qwen_vla.py 零引用）；
# 归一化统计量在 _full3/normalizer_nuplan_01，评测反归一化时才要。
# 退路：_full2（15,895 段）和 _full（5,385 段）都原样保留。
DATA=${DATA:-/mnt/sfs_turbo/zyc/official_pretrain_meta/meta/nuplan_processed_data_ndarray.pkl}
# 评测反归一化用官方那份：/mnt/sfs_turbo/zyc/DriveVLA-W0/configs/normalizer_nuplan
# （训练侧不读它；用 _full3 自己算的那份会和训练数据尺度不一致）
# pickle 里是 `<log>/<hash>.npy`，dataset 会拼成 data_root/<log>/CAM_F0/<hash>.jpg
CAM=${CAM:-/mnt/sfs_turbo/datasets/nuplan-v1.1/images}

cd "$ROOT" || exit 1
source /mnt/sfs_turbo/lmc/venv-hif8/bin/activate
export PYTHONPATH="$ROOT:$ROOT/reference/Qwen2.5-VL/qwen-vl-finetune"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# a52 上还有 zhaobo / wy / lh / k8s 的容器，给本作业一个独立的 HCCL 端口段，
# 避免撞端口（HiF8 阶段踩过：两个作业共用默认段会在 HcclBroadcast 直接失败）。
# 确定性：必须在 torchrun 之前 export（代码里的 setdefault 对已初始化的 HCCL 不生效）
export HCCL_DETERMINISTIC=TRUE
export INF_NAN_MODE_ENABLE=1
export CLOSE_MATMUL_K_SHIFT=1         # 关 MatMul 的 K 轴重排（改变累加顺序）
export ATB_MATMUL_SHUFFLE_K_ENABLE=0  # 关 ATB matmul 的 shuffle
export ACL_OP_DETERMINISTIC=1         # 昇腾算子库确定性模式
export TASK_QUEUE_ENABLE=0            # 关异步任务队列（异步派发会重排算子）
export FLAGS_npu_storage_format=0     # 关私有存储格式
export ASCEND_LAUNCH_BLOCKING=0       # 同步派发只在对齐排查时需要，平时关掉换吞吐
export PYTHONHASHSEED=42
HCCL_RANGE=${HCCL_RANGE:-17100-17199}
export HCCL_NPU_SOCKET_PORT_RANGE=$HCCL_RANGE
export HCCL_CONNECT_TIMEOUT=1800
ulimit -l unlimited 2>/dev/null || true
mkdir -p "$OUT"

echo "[$(date '+%F %T')] 阶段一启动于 $(hostname -s)" | tee "$OUT/train.log"
echo "  数据: $DATA" | tee -a "$OUT/train.log"
echo "  输出: $OUT" | tee -a "$OUT/train.log"
echo "  卡 $NPROC · mb $BS · ga $GA · 全局 batch $((BS*NPROC*GA)) · $STEPS 步 · 重计算 $GC" | tee -a "$OUT/train.log"

SEED=${SEED:-42}
echo "  随机种子: SEED=$SEED（确定性已开：同 seed 跨机器逐位一致，所以换 seed 是拿到方差样本的唯一办法）" | tee -a "$OUT/train.log"
DATA_ARGS=()
PAIR_SAMPLING=${PAIR_SAMPLING:-0}
if [ "$PAIR_SAMPLING" = "1" ]; then
  DATA_SEED=${DATA_SEED:-42}
  DATA_ARGS+=(--nuplan_pair_sampling True --data_seed "$DATA_SEED")
  echo "  数据采样: pair-level, DATA_SEED=$DATA_SEED 固定 pair 集合, SEED=$SEED 只改变其顺序" | tee -a "$OUT/train.log"
elif [ -n "${DATA_SEED:-}" ]; then
  DATA_ARGS+=(--data_seed "$DATA_SEED")
  echo "  数据采样: scene-level, DATA_SEED=$DATA_SEED" | tee -a "$OUT/train.log"
else
  echo "  数据采样: scene-level, historical default" | tee -a "$OUT/train.log"
fi
torchrun --nproc_per_node="$NPROC" --master_port=${PORT:-23493} utils/train_qwen_vla.py \
  --model_name_or_path "$MODEL" \
  --sd_model_path /mnt/sfs_turbo/models/stable-diffusion-v1-5/unet \
  --dataset_type nuplan2va_ross \
  --hif8 "$HIF8" \
  --hif8_ignore "$HIF8_IGNORE" \
  --hif8_interval "$HIF8_INTERVAL" \
  --data_path "$DATA" \
  --data_root "$CAM" \
  "${DATA_ARGS[@]}" \
  --action_tokenizer_path configs/fast \
  --actions_format fast --action_dim 3 \
  --cur_frame_idx 3 --future_nums 8 --use_previous_actions True \
  --max_pixels 147456 --min_pixels 12544 --model_max_length 8192 \
  --tune_mm_llm True --tune_mm_mlp True --tune_mm_vision True \
  --deepspeed "$DS" \
  --output_dir "$OUT" \
  --bf16 True --tf32 False --optim adamw_torch \
  --learning_rate 5e-5 --weight_decay 0.1 \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
  --max_grad_norm 5.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-6 \
  --max_steps "$STEPS" --warmup_steps 400 \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" \
  --gradient_checkpointing "$GC" \
  --dataloader_num_workers 12 --dataloader_pin_memory "$PIN_MEMORY" --dataloader_shuffle True \
  --seed "$SEED" --deterministic True --remove_dropout True \
  --save_strategy "$SAVE_STRATEGY" --save_steps "$SAVE_STEPS" --save_total_limit 5 \
  --eval_strategy no --logging_steps "$LOGSTEP" \
  --report_to tensorboard 2>&1 | tee -a "$OUT/train.log"

TRAIN_RC=${PIPESTATUS[0]}
echo "TRAIN_EXIT=$TRAIN_RC" | tee -a "$OUT/train.log"

# 把 HF 保存漏掉的 processor 配置从骨干目录补进来，让 $OUT 成为可直接
# --model_name_or_path 加载的完整模型目录（阶段二 SFT 就是这么用它的）。
# 只补白名单里的小 json，且只在缺失时补；绝不碰 *.safetensors。
if [ "$TRAIN_RC" = "0" ] && [ -d "$MODEL" ]; then
  for f in video_preprocessor_config.json chat_template.json configuration.json preprocessor_config.json; do
    if [ ! -e "$OUT/$f" ] && [ -f "$MODEL/$f" ]; then
      cp "$MODEL/$f" "$OUT/$f" && echo "  补 processor 配置: $f" | tee -a "$OUT/train.log"
    fi
  done
fi
echo "[$(date '+%F %T')] 结束" | tee -a "$OUT/train.log"
du -sh "$OUT" 2>/dev/null | tee -a "$OUT/train.log"
ls -d "$OUT"/checkpoint-* 2>/dev/null | tee -a "$OUT/train.log"

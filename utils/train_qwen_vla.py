#!/usr/bin/env python3
"""
Unified training script for Qwen-VL VLA model
Combines Qwen-VL's multimodal capabilities with VLA action prediction
"""
import os
import sys
import logging
import pathlib
import pickle
from pathlib import Path
import torch

try:  # registers the "npu" backend on Ascend; ImportError on CUDA-only hosts
    import torch_npu  # noqa: F401
except ImportError:
    pass
import transformers
from typing import Dict
from collections import defaultdict

# Add paths to import Qwen-VL components
qwen_vl_path = Path(__file__).parent.parent / "reference" / "Qwen2.5-VL" / "qwen-vl-finetune"
sys.path.append(str(qwen_vl_path))

# Ensure trainer monkey patches (optimizer grouping, print helpers) are applied
import qwenvl.train.trainer  # noqa: F401

# Optional utilities from trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: F401

# Import Qwen-VL components
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoTokenizer,
    AutoProcessor,

)

from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_ross import (
    Qwen2_5_VLConfigROSS, 
    Qwen2_5_VLForConditionalGenerationROSS,
)

from reference.transformers.src.transformers.trainer import Trainer



from qwenvl.dataset.data_qwen_vla import (
    make_supervised_data_module_navsim2_vla_ross,
    make_supervised_data_module_nuplan2_vla_ross
)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# Import VLA-specific components
sys.path.append(str(qwen_vl_path / "qwenvl" / "utils"))
from token_utils import smart_load_model_and_tokenizer, prepare_action_tokenizer_mapping, check_and_add_vla_tokens

local_rank = None


# 数据集无条件产出的 ROSS 专用键。阶段二用普通 Qwen3VL 类，它的 forward 把
# 未知键塞进 **kwargs 再往下传，会在 decoder 层炸，所以必须显式剥掉。
_ROSS_ONLY_KEYS = (
    "raw_pixel_values_vae",
    "image_token_masks",
    "action_future_masks",
    "frame_image_counts",
)


class RossTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self._strip_ross_keys = bool(kwargs.pop("strip_ross_keys", False))
        super().__init__(*args, **kwargs)
        self._acc_sum = defaultdict(float)
        self._acc_n = 0
        self._micro_in_accum = 0  # 当前累积里的 micro-step 计数

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        if self._strip_ross_keys:
            for k in _ROSS_ONLY_KEYS:
                inputs.pop(k, None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)

    def _log_mean_and_reset(self):
        if self._acc_n == 0:
            return
        mean_logs = {k: (s / float(self._acc_n)) for k, s in self._acc_sum.items()}

        # 仅主进程写
        acc = getattr(self, "accelerator", None)
        is_main = acc.is_main_process if acc is not None else self.is_world_process_zero()
        if is_main:
            self.log(mean_logs)

        self._acc_sum.clear()
        self._acc_n = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 累积 forward 写入的 _last_logs
        logs = getattr(model, "_last_logs", None)
        if logs:
            for k, v in logs.items():
                try:
                    self._acc_sum[k] += float(v)
                except Exception:
                    pass
            self._acc_n += 1

        # 统计当前累积内的 micro-step，并在到达 optimizer step 时输出
        self._micro_in_accum += 1
        gas = max(1, int(self.args.gradient_accumulation_steps))
        if self._micro_in_accum % gas == 0:
            self._log_mean_and_reset()

        return loss
    
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# ===================== 确定性与固定种子 =====================
# HiF8 精度对比要求两条臂只差量化这一个变量。`--seed` 只让 HF Trainer 设
# python/numpy/torch 的 RNG，不保证算子级数值可复现（NPU 的 reduction 顺序、
# 原子加、非确定性 kernel 都会引入逐步差异）。下面这套照搬 zjj 在 Emu3 路径上
# 验证过的实现，不依赖 msprobe（venv-hif8 未安装）。
import random as _random
import numpy as _np

try:
    import torch_npu as _torch_npu  # noqa: F401
    _DEV_TYPE = "npu" if torch.npu.is_available() else "cpu"
except Exception:
    _DEV_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def _device_manual_seed_all(seed: int):
    if _DEV_TYPE == "npu":
        torch.npu.manual_seed_all(seed)
    elif _DEV_TYPE == "cuda":
        torch.cuda.manual_seed_all(seed)


def set_reproducibility(seed: int, deterministic: bool = True):
    """固定种子 + 开确定性。返回 seed 便于调用方串联。

    注意 PYTHONHASHSEED 必须在任何 dict/set 操作之前生效；在进程内 setdefault
    只能影响后续 fork 出来的子进程（dataloader worker），主进程的 hash 种子
    已经定了。要完全严格，得在 launcher 里 export。这里仍然设，因为
    dataloader_num_workers>0 时 worker 会继承。
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    _device_manual_seed_all(seed)

    if deterministic:
        # warn_only=True：遇到没有确定性实现的算子时告警而不是抛错，
        # 否则 Qwen3-VL / SD UNet 里个别算子会直接把训练打断。
        torch.use_deterministic_algorithms(True, warn_only=True)
        if _DEV_TYPE == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            os.environ.setdefault("NCCL_DETERMINISTIC", "TRUE")
        # HCCL 确定性通信（NPU）
        os.environ.setdefault("HCCL_DETERMINISTIC", "TRUE")
        # 非饱和模式：溢出产生 Inf/NaN 而不是饱和到最大值，与 GPU 行为一致。
        # 这一条对 FP8 尤其重要——饱和模式会把溢出静默吃掉，掩盖量化问题。
        os.environ.setdefault("INF_NAN_MODE_ENABLE", "1")
    return seed


def disable_model_dropout(model: torch.nn.Module) -> None:
    """把所有 Dropout 的 p 置 0。

    Qwen3-VL 的 config 里 attention_dropout=0.1，SD UNet 里也有 dropout。
    两条臂即使同种子，dropout mask 也可能因算子调度不同而发散，所以显式关掉。
    等价于 msprobe.pytorch.seed_all(rm_dropout=True)。
    """
    n = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            if module.p != 0.0:
                module.p = 0.0
                n += 1
    rank0_print(f"[Reproducibility] 关掉 {n} 个 Dropout")
# ===================== 确定性设置结束 =====================


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def setup_vla_model_and_tokenizer(model_args, training_args):
    """
    Complete VLA model setup: load model, configure training parameters, and set up tokenizer
    """
    
    import json as _json
    _cfg_path = os.path.join(model_args.model_name_or_path, "config.json")
    _family = "qwen25"
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as _cfg_f:
            if _json.load(_cfg_f).get("model_type") == "qwen3_vl":
                _family = "qwen3vl"
    model_type = "qwen3vl_ross" if _family == "qwen3vl" else "qwen2.5vl_ross"
    # 阶段开关:True=阶段一(联合损失,带世界模型);False=阶段二(只动作损失)
    _wm = bool(getattr(model_args, "enable_world_model", True))
    rank0_print(f"[model] backbone family: {_family}  world_model: {_wm}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if _family == "qwen3vl":
        from reference.transformers.src.transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig,
        )
        from reference.transformers.src.transformers.models.qwen3_vl.modeling_qwen3_vl_ross import (
            Qwen3VLConfigROSS,
            Qwen3VLForConditionalGenerationROSS,
        )
        from reference.transformers.src.transformers.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLProcessor,
        )

        base_cfg = Qwen3VLConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
        )
        if not _wm:
            # 阶段二:普通因果 LM 类。ROSS 的 forward 无条件走 denoiser 分支,
            # 且 assert image_hidden.shape[1] == 2,没有 masks 时必然失败,
            # 所以这里换类而不是给 ROSS 类加开关(评测 runner 用的也是这个类)。
            from reference.transformers.src.transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )
            model_class = Qwen3VLForConditionalGeneration
            ross_cfg = base_cfg
        else:
            model_class = Qwen3VLForConditionalGenerationROSS
            cfg_kwargs = base_cfg.to_dict()
            if "enable_ross" in cfg_kwargs:
                ross_cfg = Qwen3VLConfigROSS(**cfg_kwargs)
            else:
                ross_cfg = Qwen3VLConfigROSS(
                    **cfg_kwargs,
                    enable_ross=True,
                    extract_image_hidden=True,
                    extract_action_hidden=True,
                    sd_model_path=model_args.sd_model_path,
                )
        # The vendored NPU FA2 path is wired to Qwen2.5-VL's attention classes.
        attn_impl = "sdpa"
    else:
        model_class = Qwen2_5_VLForConditionalGenerationROSS
        base_cfg = Qwen2_5_VLConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
        )
        if "enable_ross" in base_cfg.to_dict().keys():
            ross_cfg = Qwen2_5_VLConfigROSS(
                    **base_cfg.to_dict(),
            )
        else:
            ross_cfg = Qwen2_5_VLConfigROSS(
                    **base_cfg.to_dict(),
                    enable_ross=True,
                    extract_image_hidden=True,
                    extract_action_hidden=True,
                    sd_model_path=model_args.sd_model_path,
                )
        attn_impl = "flash_attention_2"
    model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=ross_cfg,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_impl,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
    )
    if _wm:
        from diffusers import AutoencoderKL
        from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import UNet2DConditionModel
        model.vae = AutoencoderKL.from_pretrained(model_args.sd_model_path.replace("/unet", "/vae"), torch_dtype=model.dtype)
        model.vae.eval()
        model.vae.requires_grad_(False)
        model.denoiser.unet = UNet2DConditionModel.from_pretrained(model_args.sd_model_path, torch_dtype=model.dtype)
        model.denoiser.unet.train()
        model.denoiser.unet.requires_grad_(True)
    else:
        rank0_print("World model: DISABLED (阶段二,只动作损失;不实例化 denoiser/vae)")

    tokenizer, model, _ = check_and_add_vla_tokens(tokenizer, model)

    # 确定性：固定种子 + 关 dropout（HiF8 对比的前置条件）
    _det = bool(getattr(model_args, "deterministic", True))
    set_reproducibility(int(getattr(training_args, "seed", 42)), deterministic=_det)
    rank0_print(f"[Reproducibility] seed={training_args.seed} deterministic={_det} device={_DEV_TYPE}")
    if bool(getattr(model_args, "remove_dropout", True)):
        disable_model_dropout(model)
    
    rank0_print(f"Loaded {model_class.__name__} from {model_args.model_name_or_path}")

    if _family == "qwen3vl":
        image_processor = Qwen3VLProcessor.from_pretrained(
            model_args.model_name_or_path
        ).image_processor
    else:
        image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Configure which parts of the model to train
    # Vision encoder
    if getattr(model_args, 'tune_mm_vision', True):
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
        rank0_print("Vision encoder: TRAINABLE")
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
        rank0_print("Vision encoder: FROZEN")
    
    # Vision-language connector (merger)
    if getattr(model_args, 'tune_mm_mlp', True):
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
        rank0_print("Vision-language merger: TRAINABLE")
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
        rank0_print("Vision-language merger: FROZEN")
    
    # Language model
    if getattr(model_args, 'tune_mm_llm', True):
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
        rank0_print("Language model: TRAINABLE")
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        rank0_print("Language model: FROZEN")
    
    # Denoiser / VAE 只在阶段一存在
    if _wm:
        for n, p in model.denoiser.named_parameters():
            p.requires_grad = True
        rank0_print("Denoiser: TRAINABLE")

        for n, p in model.vae.named_parameters():
            p.requires_grad = False
        rank0_print("VAE: FROZEN")
    
    def count(m):
        tot = sum(p.numel() for p in m.parameters())
        trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return tot, trn

    tot, trn = count(model)
    if local_rank in (0, None):
        print(f"[PARAM] total={tot/1e6:.1f}M, trainable={trn/1e6:.1f}M")

    # 粗估初始化上界（单卡、ZeRO-3 之前）
    bytes_per_param = 2   # bf16 权重
    adam_state = 16       # Adam m,v 各 8B
    master_fp32 = 4       # FP32 master
    rough = trn*(bytes_per_param+adam_state+master_fp32)/(1024**3)
    print(f"[EST] optimizer+master upper bound ~= {rough:.1f} GiB")

    return model, tokenizer, image_processor, model_type

def setup_vla_data_args(data_args, image_processor, model_type):
    """Setup data arguments for VLA training"""
    # Standard Qwen-VL data args
    data_args.image_processor = image_processor
    data_args.model_type = model_type
    if hasattr(data_args, 'max_pixels'):
        data_args.image_processor.max_pixels = getattr(data_args, 'max_pixels', 1280*28*28)
    if hasattr(data_args, 'min_pixels'):
        data_args.image_processor.min_pixels = getattr(data_args, 'min_pixels', 256*28*28)
        
    data_args.image_processor.size = {
        "longest_edge": data_args.max_pixels,
        "shortest_edge": data_args.min_pixels
    }
    
    # VLA-specific args (with defaults)
    data_args.use_actions = getattr(data_args, 'use_actions', True)
    data_args.actions_format = getattr(data_args, 'actions_format', 'fast')
    data_args.action_tokenizer_path = getattr(data_args, 'action_tokenizer_path', None)
    data_args.action_dim = getattr(data_args, 'action_dim', 3)  # steering, acceleration, braking
    
    # Driving-specific args
    data_args.use_previous_actions = getattr(data_args, 'use_previous_actions', False)
    data_args.cur_frame_idx = getattr(data_args, 'cur_frame_idx', 3)
    
    rank0_print(f"VLA Data Config:")
    rank0_print(f"  - use_actions: {data_args.use_actions}")
    rank0_print(f"  - actions_format: {data_args.actions_format}")
    rank0_print(f"  - action_tokenizer_path: {data_args.action_tokenizer_path}")
    rank0_print(f"  - action_dim: {data_args.action_dim}")
    
    return data_args

def train():
    global local_rank
    
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    rank0_print("="*50)
    rank0_print("Qwen-VL VLA Training Setup")
    rank0_print("="*50)
    
    # Setup model and tokenizer (includes all training configuration)
    model, tokenizer, image_processor, model_type = setup_vla_model_and_tokenizer(model_args, training_args)
    
    # Setup data
    data_args = setup_vla_data_args(data_args, image_processor, model_type)
    
    if data_args.dataset_type == "navsim2va_ross":
        data_module = make_supervised_data_module_navsim2_vla_ross(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "nuplan2va_ross":
        data_module = make_supervised_data_module_nuplan2_vla_ross(tokenizer=tokenizer, data_args=data_args)
    else:
        raise ValueError(f"Unsupported dataset type: {data_args.dataset_type}")
    rank0_print("Using standard VLA data format")
    
        
    # Create trainer
    trainer = RossTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        strip_ross_keys=not bool(getattr(model_args, "enable_world_model", True)),
        **data_module
    )
    
    # Print dataloader len
    if torch.distributed.get_rank() == 0:
        rank0_print(f"dataloader len:")
        rank0_print(f"{len(trainer.get_train_dataloader())}")
        rank0_print(f"{training_args.gradient_accumulation_steps}")
        print("len(train_dataset) =", len(data_module["train_dataset"]))
        print("per_device_train_batch_size =", training_args.per_device_train_batch_size)
        print("TrainingArguments.world_size =", training_args.world_size)
        print("torch.distributed.get_world_size() =", torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training from scratch...")
        trainer.train()
    
    # Save final model
    trainer.save_state()
    
    # Save image processor
    if hasattr(data_args, 'image_processor'):
        data_args.image_processor.save_pretrained(training_args.output_dir)
    
    # Save tokenizer (with new VLA tokens)
    tokenizer.save_pretrained(training_args.output_dir)
    
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    rank0_print("="*50)
    rank0_print("Training completed successfully!")
    rank0_print(f"Model saved to: {training_args.output_dir}")
    rank0_print("="*50)

if __name__ == "__main__":
    train()
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_type: Optional[str] = field(default="qwen2.5vl")
    sd_model_path: Optional[str] = field(default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    model_type: str = field(default="qwen2.5vl")
    sd_model_path: Optional[str] = field(default=None)
    # 阶段开关:True=阶段一(联合损失,带扩散世界模型);False=阶段二(只动作损失)
    enable_world_model: bool = field(default=True)
    # 确定性：HiF8 与 bf16 两条臂必须只差量化这一个变量
    deterministic: bool = field(default=True)
    remove_dropout: bool = field(default=True)
    # HiF8 量化："" 关闭；"cts" 用 current-tensor-scaling
    hif8: str = field(default="")
    # 量化范围，逗号分隔的 fqn 子串。默认照客户规格：
    #   只量化 language_model 的 28 层，排除 lm_head 与 o_proj
    #   （见 /mnt/sfs_turbo/zyc/hif8/hif8-cts-code-map.md 第 4 节，应为 168 个）
    hif8_apply: str = field(default="model.language_model.layers.")
    hif8_ignore: str = field(default="lm_head,o_proj")
    # DTS（延迟缩放）的刷新间隔：每 interval 步重算一次 amax 并刷新 scale，
    # 其余步复用缓存的 scale。仅在 --hif8 dts 时生效；cts 每步现算，与它无关。
    hif8_interval: int = field(default=10)
    # 阶段开关:True=阶段一(联合损失,带扩散世界模型);False=阶段二(只动作损失)
    enable_world_model: bool = field(default=True)

@dataclass
class DataArguments:
    dataset_type: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    
    # VLA-specific arguments
    data_path: Optional[str] = field(default=None)
    use_actions: bool = field(default=True)
    actions_format: str = field(default="fast")
    action_tokenizer_path: Optional[str] = field(default=None)
    action_dim: int = field(default=3)
    use_previous_actions: bool = field(default=False)
    cur_frame_idx: int = field(default=0)
    future_nums: int = field(default=8)
    data_root: Optional[str] = field(default=None)
    # NuPlan only: expose every valid (scene, anchor) pair as one dataset item.
    # The Trainer's seedable sampler then controls pair order with --data_seed.
    nuplan_pair_sampling: bool = field(
        default=False,
        metadata={
            "help": "Select the NuPlan training-pair budget with data_seed, "
                    "then permute that fixed subset with --seed."
        },
    )
    nuplan_pair_sampling_mode: str = field(
        default="disjoint_2va",
        metadata={
            "help": "NuPlan pair sampling mode: 'disjoint_2va' selects 2VA "
                    "windows whose individual VA pairs do not overlap; "
                    "'anchor' preserves the historical anchor-only sampler."
        },
    )
    nuplan_holdout_validation: bool = field(
        default=False,
        metadata={
            "help": "Create a leakage-safe NuPlan validation dataset by holding "
                    "out complete logs from the training split."
        },
    )
    nuplan_validation_log_fraction: float = field(
        default=0.05,
        metadata={"help": "Fraction of NuPlan logs reserved for validation."},
    )
    nuplan_validation_seed: int = field(
        default=20261127,
        metadata={
            "help": "Seed for the fixed log-level holdout and validation-pair "
                    "selection. Independent of model and training-data seeds."
        },
    )
    nuplan_validation_pair_count: int = field(
        default=4096,
        metadata={"help": "Number of 2VA windows in the validation dataset."},
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    dataloader_shuffle: bool = field(default=True)
    save_only_model: bool = field(default=False, metadata={"help": "Save only model weights, not optimizer states"})
    min_learning_rate: Optional[float] = field(default=None)

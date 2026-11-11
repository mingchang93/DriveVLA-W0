#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 Qwen 训练入口加确定性与固定种子（HiF8 精度对比的硬前置条件）。

现状：`utils/train_qwen_vla.py` 里一处 seed/determinism 设置都没有，只靠
`--seed 42` 让 HF Trainer 的 set_seed() 设 python/numpy/torch 的 RNG。
那只保证数据顺序和初始化可复现，不保证算子级数值可复现——NPU 上的 reduction
顺序、原子加、非确定性 kernel 都会引入逐步差异。而我们要测的 bf16 vs HiF8
偏差量级可能和那个噪声同阶。

做法照搬 zjj 在 Emu3 路径上验证过的实现（`utils/train_moe.py` 的
set_reproducibility + disable_model_dropout），但不依赖 msprobe
（venv-hif8 里没装）。参考昇腾官方的精度对齐指南：
https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/best_practices/train_debug_guide.md

额外处理一件 Qwen 特有的事：Qwen3-VL 的 config 里 attention_dropout=0.1，
两条臂即使同种子，dropout mask 也可能因算子调度不同而发散，所以要显式关掉。

改三处：
  1. 文件头部插入 set_reproducibility / disable_model_dropout / _device_type
  2. train() 开头调用 set_reproducibility(training_args.seed)
  3. 模型建好后调用 disable_model_dropout(model)
新增两个开关（默认开，保持可关）：
  --deterministic true|false       确定性算法 + HCCL_DETERMINISTIC
  --remove_dropout true|false      把所有 Dropout 的 p 置 0
幂等：已打过则跳过。
"""
import io
import os
import sys

ROOT = "/mnt/sfs_turbo/zyc/DriveVLA-W0"
TRN = os.path.join(ROOT, "utils/train_qwen_vla.py")
ARG = os.path.join(ROOT, "reference/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/argument.py")

applied, skipped = [], []


def patch(path, old, new, tag, probe):
    s = io.open(path, encoding="utf-8").read()
    if probe in s:
        skipped.append(tag)
        return
    if old not in s:
        raise SystemExit("FAIL %s: anchor not found\n---\n%s\n---" % (tag, old[:220]))
    io.open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    applied.append(tag)


# ---------------- 1. 两个开关 ----------------
patch(
    ARG,
    """    # 阶段开关:True=阶段一(联合损失,带扩散世界模型);False=阶段二(只动作损失)
    enable_world_model: bool = field(default=True)""",
    """    # 阶段开关:True=阶段一(联合损失,带扩散世界模型);False=阶段二(只动作损失)
    enable_world_model: bool = field(default=True)
    # 确定性：HiF8 与 bf16 两条臂必须只差量化这一个变量
    deterministic: bool = field(default=True)
    remove_dropout: bool = field(default=True)""",
    "1/3 ModelArguments 加 deterministic / remove_dropout",
    "remove_dropout: bool = field",
)

# ---------------- 2. 工具函数 ----------------
HELPERS = '''

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
'''

patch(
    TRN,
    """def rank0_print(*args):
    if local_rank == 0:
        print(*args)""",
    """def rank0_print(*args):
    if local_rank == 0:
        print(*args)

""" + HELPERS.strip() + "\n",
    "2/3 插入 set_reproducibility / disable_model_dropout",
    "def set_reproducibility(",
)

# ---------------- 3. 调用点：模型建好后关 dropout ----------------
patch(
    TRN,
    """    tokenizer, model, _ = check_and_add_vla_tokens(tokenizer, model)""",
    """    tokenizer, model, _ = check_and_add_vla_tokens(tokenizer, model)

    # 确定性：固定种子 + 关 dropout（HiF8 对比的前置条件）
    _det = bool(getattr(model_args, "deterministic", True))
    set_reproducibility(int(getattr(training_args, "seed", 42)), deterministic=_det)
    rank0_print(f"[Reproducibility] seed={training_args.seed} deterministic={_det} device={_DEV_TYPE}")
    if bool(getattr(model_args, "remove_dropout", True)):
        disable_model_dropout(model)""",
    "3/3 调用 set_reproducibility + disable_model_dropout",
    "[Reproducibility] seed=",
)

print("patched:")
for a in applied:
    print("  +", a)
for s in skipped:
    print("  = %s (already applied)" % s)
print("\n[usage] default on; disable with --deterministic false --remove_dropout false")

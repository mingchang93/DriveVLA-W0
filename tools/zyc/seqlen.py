# -*- coding: utf-8 -*-
"""量一下 NAVSIM 2VA 样本的真实 token 长度，判断 model_max_length=1400 会不会截断。

为什么要查：collator 是静默截断的，没有任何警告——
    input_ids = input_ids[:, : self.tokenizer.model_max_length]
而动作 token 在序列末尾（BOA + action_ids + EOA 拼在图像和文本之后），
一旦超长，切掉的正是训练目标本身。官方用 8192，我们用 1400（抄自 Emu3 的
2VA 脚本，但 Emu3 视觉是离散 VQ 码字，Qwen 是连续 ViT 特征，token 数由
max_pixels 决定，两者不可类比）。

只用 CPU：加载 tokenizer 和 image_processor，不加载模型权重。
"""
import os
import sys

import numpy as np
import torch

D = "/mnt/sfs_turbo/zyc/DriveVLA-W0"
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(D, "reference/Qwen2.5-VL/qwen-vl-finetune"))

from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

MODEL = "/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct"
PKL = "/mnt/sfs_turbo/zjj/datasets/DriveVLA-W0/navsim_emu_vla_256_144_trainval_pre_1s.pkl"
ROOT = "/mnt/sfs_turbo/lmc/navsim_openscene/openscene-v1.1/sensor_blobs/trainval"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
tok.model_max_length = 10 ** 6          # 先设超大，量真实长度，不让它截断

# 不用 check_and_add_vla_tokens：它要真模型（会调 get_input_embeddings）。
# 这里只需要 tokenizer 侧的两个特殊 token 和属性，手动加等效。


class _Args:
    pass


a = _Args()
a.data_path = PKL
a.data_root = ROOT
a.actions_format = "fast"
a.action_tokenizer_path = os.path.join(D, "configs/fast")
a.action_dim = 3
a.cur_frame_idx = 3
a.future_nums = 8
a.use_previous_actions = True
a.use_actions = True
a.max_pixels = 147456
a.min_pixels = 12544
a.image_processor = proc.image_processor
a.model_type = "qwen3_vl"
a.dataset_type = "navsim2va_ross"

tok.add_special_tokens({"additional_special_tokens": ["<action_start>", "<action_end>"]})
tok.boa_token = "<action_start>"
tok.eoa_token = "<action_end>"
print("词表大小", len(tok))

from qwenvl.dataset.data_qwen_vla import LazySupervisedNavsim2VAROSSDataset as DS  # noqa: E402

ds = DS(tokenizer=tok, data_args=a)
print("数据集大小", len(ds))

lens = []
for i in np.linspace(0, len(ds) - 1, N).astype(int):
    try:
        s = ds[int(i)]
        lens.append(int(s["input_ids"].shape[-1]))
    except Exception as e:
        print("  样本 %d 失败: %r" % (i, e))
        if len(lens) == 0 and i > 5:
            break

if not lens:
    sys.exit("没取到任何样本")
lens = np.array(lens)
print("\n抽样 %d 个样本的 token 长度" % len(lens))
print("  最小 %d  中位 %.0f  均值 %.0f  最大 %d" % (lens.min(), np.median(lens), lens.mean(), lens.max()))
for thr in (1400, 2048, 4096, 8192):
    over = int((lens > thr).sum())
    print("  超过 %-5d 的比例: %5.1f%%  (%d/%d)" % (thr, 100.0 * over / len(lens), over, len(lens)))
print("\n结论: model_max_length=1400 %s" % ("会截断，必须改大" if (lens > 1400).any() else "不会截断"))

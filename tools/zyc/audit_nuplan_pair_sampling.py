#!/usr/bin/env python3
"""Construct the real NuPlan dataset at a fixed pair budget and compare seeds."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_ROOT = REPO_ROOT / "reference/Qwen2.5-VL/qwen-vl-finetune"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(QWEN_ROOT))

from qwenvl.dataset.data_qwen_vla import LazySupervisedNuplan2VAROSSDataset  # noqa: E402
from qwenvl.train.argument import DataArguments  # noqa: E402
from qwenvl.utils.token_utils import VLA_SPECIAL_TOKENS  # noqa: E402


def load_tokenizer_and_processor(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(VLA_SPECIAL_TOKENS.values())}
    )
    tokenizer.boa_token = VLA_SPECIAL_TOKENS["boa_token"]
    tokenizer.eoa_token = VLA_SPECIAL_TOKENS["eoa_token"]
    tokenizer.model_max_length = 8192
    tokenizer.padding_side = "right"

    config_path = Path(model_path) / "config.json"
    with config_path.open() as handle:
        model_type = json.load(handle).get("model_type", "")
    if model_type == "qwen3_vl":
        from reference.transformers.src.transformers.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLProcessor,
        )

        processor = Qwen3VLProcessor.from_pretrained(model_path)
        dataset_model_type = "qwen3vl_ross"
    else:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        dataset_model_type = "qwen2.5vl_ross"
    return tokenizer, processor.image_processor, dataset_model_type


def build_dataset(args, tokenizer, image_processor, model_type, model_seed, sample_count):
    data_args = DataArguments()
    data_args.dataset_type = "nuplan2va_ross"
    data_args.data_path = args.data_path
    data_args.data_root = args.data_root
    data_args.action_tokenizer_path = str(REPO_ROOT / "configs/fast")
    data_args.actions_format = "fast"
    data_args.action_dim = 3
    data_args.use_actions = True
    data_args.use_previous_actions = True
    data_args.max_pixels = args.max_pixels
    data_args.min_pixels = args.min_pixels
    data_args.image_processor = image_processor
    data_args.image_processor.max_pixels = args.max_pixels
    data_args.image_processor.min_pixels = args.min_pixels
    data_args.image_processor.size = {
        "longest_edge": args.max_pixels,
        "shortest_edge": args.min_pixels,
    }
    data_args.model_type = model_type
    data_args.seed = args.data_seed
    data_args.model_seed = model_seed
    data_args.nuplan_pair_sampling = True
    data_args.nuplan_pair_sampling_mode = args.sampling_mode
    data_args.nuplan_pair_sample_count = sample_count

    dataset = LazySupervisedNuplan2VAROSSDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    summary = {
        "sampling_mode": dataset.pair_sampling_mode,
        "data_seed": args.data_seed,
        "model_seed": model_seed,
        "dataset_len": len(dataset),
        "valid_pairs": dataset.num_pairs,
        "selected_pairs": dataset.num_pair_samples,
        "unique_selected_pairs": int(np.unique(dataset.pair_indices).size),
        "va_pair_slots": dataset.num_va_pair_slots,
        "unique_va_pairs": dataset.num_unique_va_pairs,
        "duplicate_va_pairs": dataset.num_duplicate_va_pairs,
        "duplicate_va_pair_slots": dataset.num_duplicate_va_pair_slots,
        "max_va_pair_multiplicity": dataset.max_va_pair_multiplicity,
        "max_disjoint_pair_windows": dataset.max_disjoint_pair_windows,
        "pair_sampling_scanned_windows": dataset.pair_sampling_scanned_windows,
        "set_sha256": dataset.pair_set_sha256,
        "order_sha256": dataset.pair_order_sha256,
        "va_pair_set_sha256": dataset.va_pair_set_sha256,
        "va_pair_order_sha256": dataset.va_pair_order_sha256,
        "first_pair_ids": dataset.pair_indices[:5].tolist(),
        "last_pair_ids": dataset.pair_indices[-5:].tolist(),
    }
    del dataset
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/sfs_turbo/models/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--data-path",
        default="/mnt/sfs_turbo/zyc/official_pretrain_meta/meta/nuplan_processed_data_ndarray.pkl",
    )
    parser.add_argument(
        "--data-root",
        default="/mnt/sfs_turbo/datasets/nuplan-v1.1/images",
    )
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[43, 44])
    parser.add_argument(
        "--sampling-mode",
        choices=("anchor", "disjoint_2va"),
        default="disjoint_2va",
    )
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--per-device-batch", type=int, default=12)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--min-pixels", type=int, default=12544)
    args = parser.parse_args()

    sample_count = (
        args.max_steps
        * args.per_device_batch
        * args.world_size
        * args.grad_accum
    )
    tokenizer, image_processor, model_type = load_tokenizer_and_processor(args.model)

    summaries = []
    for model_seed in args.model_seeds:
        summary = build_dataset(
            args,
            tokenizer,
            image_processor,
            model_type,
            model_seed,
            sample_count,
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    if len(summaries) < 2:
        raise SystemExit("at least two model seeds are required")
    base = summaries[0]
    for summary in summaries[1:]:
        assert summary["set_sha256"] == base["set_sha256"]
        assert summary["order_sha256"] != base["order_sha256"]
        assert summary["va_pair_set_sha256"] == base["va_pair_set_sha256"]
        assert summary["va_pair_order_sha256"] != base["va_pair_order_sha256"]
        assert summary["unique_selected_pairs"] == sample_count
        assert summary["dataset_len"] == sample_count
        if args.sampling_mode == "disjoint_2va":
            assert summary["va_pair_slots"] == sample_count * 2
            assert summary["unique_va_pairs"] == sample_count * 2
            assert summary["duplicate_va_pairs"] == 0
            assert summary["duplicate_va_pair_slots"] == 0
            assert summary["max_va_pair_multiplicity"] == 1
    print("PAIR_SAMPLING_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

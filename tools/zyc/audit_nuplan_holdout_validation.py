#!/usr/bin/env python3
"""Audit the log-level NuPlan holdout and its pair disjointness."""

import argparse
import gc
import json
import pickle
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_ROOT = REPO_ROOT / "reference/Qwen2.5-VL/qwen-vl-finetune"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(QWEN_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_nuplan_pair_sampling import load_tokenizer_and_processor  # noqa: E402
from qwenvl.dataset.data_qwen_vla import (  # noqa: E402
    LazySupervisedNuplan2VAROSSDataset,
    nuplan_log_id,
    select_nuplan_validation_logs,
)
from qwenvl.train.argument import DataArguments  # noqa: E402


def data_args_for(
    args,
    image_processor,
    model_type,
    data_seed,
    model_seed,
):
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
    data_args.seed = data_seed
    data_args.model_seed = model_seed
    data_args.nuplan_pair_sampling = True
    data_args.nuplan_pair_sampling_mode = "disjoint_2va"
    data_args.nuplan_pair_sample_count = args.train_pairs
    data_args.nuplan_holdout_validation = True
    data_args.nuplan_validation_log_fraction = args.validation_log_fraction
    data_args.nuplan_validation_seed = args.validation_seed
    data_args.nuplan_validation_pair_count = args.validation_pairs
    return data_args


def build_dataset(
    args,
    tokenizer,
    image_processor,
    model_type,
    shared_data,
    validation_logs,
    role,
    data_seed,
    model_seed,
):
    return LazySupervisedNuplan2VAROSSDataset(
        tokenizer=tokenizer,
        data_args=data_args_for(
            args,
            image_processor,
            model_type,
            data_seed,
            model_seed,
        ),
        dataset_role=role,
        shared_data=shared_data,
        validation_logs=validation_logs,
    )


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
    parser.add_argument("--train-pairs", type=int, default=768000)
    parser.add_argument("--validation-pairs", type=int, default=4096)
    parser.add_argument("--validation-log-fraction", type=float, default=0.05)
    parser.add_argument("--validation-seed", type=int, default=20261127)
    parser.add_argument("--max-pixels", type=int, default=147456)
    parser.add_argument("--min-pixels", type=int, default=12544)
    args = parser.parse_args()

    tokenizer, image_processor, model_type = load_tokenizer_and_processor(args.model)
    with open(args.data_path, "rb") as handle:
        shared_data = pickle.load(handle)
    validation_logs = select_nuplan_validation_logs(
        shared_data,
        args.validation_log_fraction,
        args.validation_seed,
    )

    train = build_dataset(
        args,
        tokenizer,
        image_processor,
        model_type,
        shared_data,
        validation_logs,
        "train",
        42,
        42,
    )
    validation = build_dataset(
        args,
        tokenizer,
        image_processor,
        model_type,
        shared_data,
        validation_logs,
        "validation",
        42,
        42,
    )
    validation_other_seed = build_dataset(
        args,
        tokenizer,
        image_processor,
        model_type,
        shared_data,
        validation_logs,
        "validation",
        100,
        100,
    )

    train_logs = {nuplan_log_id(scene) for scene in train.data}
    validation_scene_logs = {nuplan_log_id(scene) for scene in validation.data}
    assert train_logs.isdisjoint(validation_scene_logs)
    assert validation_scene_logs == set(validation_logs)
    assert validation.dataset_role == "validation"
    assert train.pair_sampling_mode == "disjoint_2va"
    assert validation.pair_sampling_mode == "disjoint_2va"
    assert len(train) == args.train_pairs
    assert len(validation) == args.validation_pairs
    assert train.num_duplicate_va_pairs == 0
    assert validation.num_duplicate_va_pairs == 0
    assert validation.num_unique_va_pairs == args.validation_pairs * 2
    assert validation.pair_set_sha256 == validation_other_seed.pair_set_sha256
    assert validation.va_pair_set_sha256 == validation_other_seed.va_pair_set_sha256
    assert validation.pair_order_sha256 == validation_other_seed.pair_order_sha256
    assert validation.va_pair_order_sha256 == validation_other_seed.va_pair_order_sha256

    train_va_keys = train.selected_va_pair_keys()
    validation_va_keys = validation.selected_va_pair_keys()
    overlap = np.intersect1d(
        train_va_keys,
        validation_va_keys,
        assume_unique=True,
    )
    assert overlap.size == 0

    summary = {
        "train_logs": len(train_logs),
        "validation_logs": len(validation_scene_logs),
        "train_scenes": len(train.data),
        "validation_scenes": len(validation.data),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "train_unique_va_pairs": train.num_unique_va_pairs,
        "validation_unique_va_pairs": validation.num_unique_va_pairs,
        "train_duplicate_va_pairs": train.num_duplicate_va_pairs,
        "validation_duplicate_va_pairs": validation.num_duplicate_va_pairs,
        "train_validation_va_overlap": int(overlap.size),
        "train_set_sha256": train.pair_set_sha256,
        "validation_set_sha256": validation.pair_set_sha256,
        "validation_va_set_sha256": validation.va_pair_set_sha256,
        "validation_order_sha256": validation.pair_order_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("HOLDOUT_VALIDATION_AUDIT_PASS", flush=True)

    del validation_other_seed, validation, train
    gc.collect()


if __name__ == "__main__":
    main()

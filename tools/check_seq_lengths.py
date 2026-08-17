#!/usr/bin/env python3
"""
Check actual tokenized sequence lengths in a training pickle to determine
the minimum viable --max_position_embeddings.

Usage:
  python tools/check_seq_lengths.py \
      --data_path data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
      --model_name_or_path /path/to/Emu3-Stage1 \
      --max_samples 500

The script loads the dataset normally (same code path as training), records
the non-padding token count per sample, and prints a histogram + percentiles
so you can pick the smallest --max_position_embeddings that covers your data.
"""

import argparse
import sys
import os
import numpy as np

# ── path setup (mirrors train_moe.py) ──────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, os.path.join(_parent_dir, "reference", "Emu3"))
sys.path.insert(0, _parent_dir)

from emu3.mllm import Emu3Tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Check actual tokenized sequence lengths")
    p.add_argument("--data_path", required=True, help="Path to the training .pkl file")
    p.add_argument("--model_name_or_path", required=True,
                   help="Path to tokenizer / model hub dir (e.g. pretrained_models/Emu3-Stage1)")
    p.add_argument("--max_samples", type=int, default=500,
                   help="Cap on number of samples to check (default: 500)")
    p.add_argument("--model_max_length", type=int, default=1400,
                   help="Model max length used by tokenizer (default: 1400)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load pickle ────────────────────────────────────────────────────
    import pickle
    print(f"Loading {args.data_path} ...")
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} scenes in pickle")

    # ── Tokenizer ──────────────────────────────────────────────────────
    tokenizer = Emu3Tokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    print(f"  tokenizer model_max_length = {tokenizer.model_max_length}")

    # ── Minimal tokenization (text only, no image tokens) ──────────────
    # We tokenize just the text prompt to get a lower bound.  The actual
    # sequence also includes visual tokens and action tokens, so we also
    # do a full pass with visual tokens below.
    text_lengths = []
    for i, scene in enumerate(data):
        if i >= args.max_samples:
            break
        prompt = scene.get("text", "")
        text_input = tokenizer.bos_token + prompt
        ids = tokenizer(text_input, padding=False, return_tensors="pt")["input_ids"][0]
        text_lengths.append(len(ids))

    text_lengths = np.array(text_lengths)
    print(f"\n── Text-only token counts (n={len(text_lengths)}) ──")
    print(f"  min={text_lengths.min()}, max={text_lengths.max()}, "
          f"mean={text_lengths.mean():.0f}, median={np.median(text_lengths):.0f}")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p}: {int(np.percentile(text_lengths, p))}")

    # ── Estimate visual token overhead ─────────────────────────────────
    # Each image frame is encoded as H×W visual tokens, each formatted as
    # "<|visual token XXXXXX|>" (a single BPE token).  Plus the framing
    # tokens: boi, "{frames}*{h}*{w}", img_token, eof_token, eoi_token.
    # For a typical navsim sample: frames=1, h=18, w=32 → 576 visual tokens
    # plus ~5 framing tokens → ~581 tokens per frame.
    #
    # Action tokens: 8 action_frames × 3 action_dim = 24 tokens + BOA + EOA
    # → ~26 tokens.
    print(f"\n── Estimated total (text + 1 frame @ 18×32 + 8×3 actions) ──")
    VISUAL_TOKENS_PER_FRAME = 18 * 32   # h × w from resolution
    FRAMING_TOKENS = 5                   # boi, dims, img, eof, eoi
    ACTION_TOKENS = 8 * 3 + 2            # action_frames × action_dim + BOA + EOA
    estimated = text_lengths + VISUAL_TOKENS_PER_FRAME + FRAMING_TOKENS + ACTION_TOKENS
    print(f"  min={estimated.min()}, max={estimated.max()}, "
          f"mean={estimated.mean():.0f}, median={np.median(estimated):.0f}")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p}: {int(np.percentile(estimated, p))}")

    # ── Recommendation ─────────────────────────────────────────────────
    p99 = int(np.percentile(estimated, 99))
    p100 = int(estimated.max())
    # Round up to nearest 64 for efficient attention
    rounded = ((p100 + 63) // 64) * 64
    print(f"\n── Recommendation ──")
    print(f"  p99 coverage:  --max_position_embeddings {p99}")
    print(f"  p100 coverage: --max_position_embeddings {p100}")
    print(f"  p100 + 64-align: --max_position_embeddings {rounded}")
    print(f"  current setting: --max_position_embeddings {args.model_max_length}")
    savings_pct = (1 - rounded / args.model_max_length) * 100
    if savings_pct > 0:
        print(f"  memory savings vs current: ~{savings_pct:.0f}% of activation memory")


if __name__ == "__main__":
    main()
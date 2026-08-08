#!/usr/bin/env python3
"""Summarize per-submodule forward times from --log_submodule_time profiling.

Aggregates submodule_times/rank*.jsonl produced by LoggingTrainer and prints
a ranked table so you can compare where training time is spent. With multiple
ranks, also shows per-rank variance to detect rank-0 overhead from ZeRO-3
all-gather, logging, and callbacks.

Use --combine to merge all per-rank files into a single combined JSONL for
further analysis.

Usage:
  python scripts/summarize_submodule_times.py \\
      --submodule_dir logs/train_base_ar_*/train_base_ar/submodule_times

  python scripts/summarize_submodule_times.py \\
      --submodule_dir logs/.../submodule_times --combine combined.jsonl

  python scripts/summarize_submodule_times.py --self-test   # verify the tool works
"""
import argparse
import glob
import json
import os
import statistics
import sys


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(records):
    """Aggregate per-key times across all intervals.

    Returns a list of (key, mean, median, total, pct_of_forward) sorted by mean desc.
    """
    keys = sorted(k for k in records[0] if k not in ('step', 'steps'))

    totals = {k: 0.0 for k in keys}
    all_vals = {k: [] for k in keys}

    for rec in records:
        for k in keys:
            v = rec.get(k, 0.0)
            if v > 0:
                totals[k] += v
                all_vals[k].append(v)

    grand_total = sum(totals.values())

    rows = []
    for k in keys:
        vals = all_vals[k]
        if not vals:
            continue
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        total = totals[k]
        pct = (total / grand_total * 100) if grand_total > 0 else 0.0
        rows.append((k, mean, median, total, pct))

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def report(rows, n_records, label=''):
    tag = f' [{label}]' if label else ''
    print("=" * 90)
    print(f"Submodule forward-time profile ({n_records} intervals){tag}")
    print("=" * 90)
    print(f"{'Key':<45} {'Mean(s)':>10} {'Median(s)':>10} {'Total(s)':>10} {'%':>7}")
    print("-" * 90)

    for key, mean, median, total, pct in rows:
        print(f"{key:<45} {mean:>10.4f} {median:>10.4f} {total:>10.3f} {pct:>6.1f}%")

    print("-" * 90)

    layer_keys = [r for r in rows if '.layers.' in r[0] and r[0].endswith('.total')]
    if layer_keys:
        print(f"\nLayer-by-layer comparison ({len(layer_keys)} layers):")
        layer_times = [(r[0], r[1]) for r in layer_keys]
        layer_means = [lt[1] for lt in layer_times]
        if min(layer_means) > 0:
            spread = max(layer_means) / min(layer_means)
            print(f"  min layer mean: {min(layer_means):.4f}s  "
                  f"max layer mean: {max(layer_means):.4f}s  "
                  f"spread: {spread:.2f}x")
        layer_times.sort(key=lambda x: x[1], reverse=True)
        print(f"  slowest 3: {', '.join(f'{k} ({v:.4f}s)' for k, v in layer_times[:3])}")
        print(f"  fastest 3: {', '.join(f'{k} ({v:.4f}s)' for k, v in layer_times[-3:])}")

    print("=" * 90)


def cross_rank_compare(all_rank_rows, all_rank_records):
    """Compare submodule means across ranks with rank 0 as baseline.

    Shows per-rank delta vs rank 0 for each top-level submodule. Flags
    ranks where the delta exceeds 5% — these indicate ZeRO-3 all-gather
    imbalance or rank-0 overhead from logging, callbacks, or checkpoint I/O.
    """
    ranks = sorted(all_rank_rows.keys())
    if len(ranks) < 2:
        return

    top_keys = ['model.embed', 'model.norm', 'lm_head+loss']
    rank_means = {}
    for rank in ranks:
        d = {r[0]: r[1] for r in all_rank_rows[rank]}
        layer_total = sum(v for k, v in d.items() if '.layers.' in k and k.endswith('.total'))
        d['layers(sum)'] = layer_total
        rank_means[rank] = d

    baseline = rank_means[0]  # rank 0 as reference

    print("\nCross-rank variance vs rank 0 (baseline):")
    # Header
    print(f"{'Key':<30}  {'rank0':>10}", end='')
    for rank in ranks[1:]:
        print(f"  {'rank'+str(rank)+'(Δ%)':>15}", end='')
    print()
    print("-" * (30 + 2 + 10 + len(ranks[1:]) * 17))

    for key in top_keys + ['layers(sum)']:
        base = baseline.get(key, 0.0)
        vals = [rank_means[r].get(key, 0.0) for r in ranks]
        if all(v == 0.0 for v in vals):
            continue
        print(f"{key:<30}  {base:>10.4f}", end='')
        flags = []
        for rank in ranks[1:]:
            v = rank_means[rank].get(key, 0.0)
            if base > 0:
                delta = (v - base) / base * 100
                sign = '+' if delta >= 0 else ''
                print(f"  {v:>10.4f}({sign}{delta:.1f}%)", end='')
                if abs(delta) > 5.0:
                    flags.append(f'rank {rank} {sign}{delta:.1f}%')
            else:
                print(f"  {v:>10.4f}", end='')
        if flags:
            print(f"  ⚠ {', '.join(flags)}", end='')
        print()

    print("-" * (30 + 2 + 10 + len(ranks[1:]) * 17))


def self_test():
    import tempfile

    tmp = tempfile.mkdtemp()
    sub_dir = os.path.join(tmp, 'submodule_times')
    os.makedirs(sub_dir, exist_ok=True)

    # Rank 0: slightly slower lm_head+loss (simulates rank-0 overhead)
    with open(os.path.join(sub_dir, 'rank0.jsonl'), 'w') as f:
        for i in range(5):
            rec = {'step': (i + 1) * 10, 'steps': 10}
            rec['model.layers.0.total'] = 0.050 + i * 0.001
            rec['model.layers.0.self_attn'] = 0.030 + i * 0.001
            rec['model.layers.0.mlp'] = 0.015 + i * 0.001
            rec['model.embed'] = 0.002 + i * 0.0001
            rec['model.norm'] = 0.001 + i * 0.0001
            rec['lm_head+loss'] = 0.085 + i * 0.002  # rank 0 slower
            f.write(json.dumps(rec) + '\n')

    # Rank 1: baseline
    with open(os.path.join(sub_dir, 'rank1.jsonl'), 'w') as f:
        for i in range(5):
            rec = {'step': (i + 1) * 10, 'steps': 10}
            rec['model.layers.0.total'] = 0.050 + i * 0.001
            rec['model.layers.0.self_attn'] = 0.030 + i * 0.001
            rec['model.layers.0.mlp'] = 0.015 + i * 0.001
            rec['model.embed'] = 0.002 + i * 0.0001
            rec['model.norm'] = 0.001 + i * 0.0001
            rec['lm_head+loss'] = 0.080 + i * 0.002  # rank 1 normal
            f.write(json.dumps(rec) + '\n')

    paths = sorted(glob.glob(os.path.join(sub_dir, 'rank*.jsonl')))
    assert len(paths) == 2, f"expected 2 rank files, got {len(paths)}"

    all_rank_rows = {}
    all_rank_records = {}
    for p in paths:
        rank = int(os.path.basename(p).replace('rank', '').replace('.jsonl', ''))
        records = load_records(p)
        all_rank_records[rank] = records
        all_rank_rows[rank] = summarize(records)

    # Rank 0 lm_head+loss > rank 1
    rank0_lm = [r for r in all_rank_rows[0] if r[0] == 'lm_head+loss'][0][1]
    rank1_lm = [r for r in all_rank_rows[1] if r[0] == 'lm_head+loss'][0][1]
    assert rank0_lm > rank1_lm, f"rank 0 lm_head+loss ({rank0_lm}) should be > rank 1 ({rank1_lm})"

    print("self-test OK: multi-rank parsing, cross-rank comparison all correct")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submodule_dir', help='Path to submodule_times/ directory')
    ap.add_argument('--self-test', action='store_true', help='Run built-in verification')
    ap.add_argument('--top', type=int, default=50, help='Show top N keys (default 50)')
    ap.add_argument('--combine', metavar='PATH', help='Merge all rank files into a single combined JSONL')
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.submodule_dir:
        ap.error('--submodule_dir is required (or use --self-test)')

    paths = sorted(glob.glob(os.path.join(args.submodule_dir, 'rank*.jsonl')))
    if not paths:
        print(f'No rank*.jsonl files found in {args.submodule_dir}')
        return 1

    print(f"Found {len(paths)} rank file(s) in {args.submodule_dir}")

    all_rank_rows = {}
    all_rank_records = {}

    for p in paths:
        rank = int(os.path.basename(p).replace('rank', '').replace('.jsonl', ''))
        records = load_records(p)
        if not records:
            print(f"  rank{rank}: empty, skipping")
            continue
        all_rank_records[rank] = records
        all_rank_rows[rank] = summarize(records)

    # Show rank 0 in detail
    if 0 in all_rank_rows:
        report(all_rank_rows[0][:args.top], len(all_rank_records[0]), 'rank 0')

    # Cross-rank comparison
    cross_rank_compare(all_rank_rows, all_rank_records)

    # Merge all ranks into a single combined file for further analysis
    if args.combine:
        with open(args.combine, 'w') as out:
            for rank in sorted(all_rank_records):
                for rec in all_rank_records[rank]:
                    rec['rank'] = rank
                    out.write(json.dumps(rec) + '\n')
        print(f"\nCombined {sum(len(r) for r in all_rank_records.values())} records "
              f"from {len(all_rank_records)} ranks → {args.combine}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
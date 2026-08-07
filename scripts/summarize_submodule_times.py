#!/usr/bin/env python3
"""Summarize per-submodule forward times from --log_submodule_time profiling.

Aggregates the submodule_times/rank0.jsonl produced by LoggingTrainer and
prints a ranked table so you can compare where training time is spent.

Usage:
  python scripts/summarize_submodule_times.py \\
      --submodule_log logs/train_base_ar_*/train_base_ar/submodule_times/rank0.jsonl

  python scripts/summarize_submodule_times.py --self-test   # verify the tool works
"""
import argparse
import json
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
    # Collect all keys except 'step' and 'steps'
    keys = sorted(k for k in records[0] if k not in ('step', 'steps'))

    # Compute per-interval forward time as sum of all submodule times
    # except embed (which is already included in layer total via the forward flow).
    # Actually, each submodule time is additive — they sum to the full forward time.
    # But layers.total includes self_attn + mlp + layernorms, and embed + norm + lm_head+loss
    # are separate. So total per step ≈ sum of all recorded keys.
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

    rows.sort(key=lambda r: r[1], reverse=True)  # mean descending
    return rows


def report(rows, records):
    print("=" * 90)
    print(f"Submodule forward-time profile ({len(records)} intervals)")
    print("=" * 90)
    print(f"{'Key':<45} {'Mean(s)':>10} {'Median(s)':>10} {'Total(s)':>10} {'%':>7}")
    print("-" * 90)

    for key, mean, median, total, pct in rows:
        print(f"{key:<45} {mean:>10.4f} {median:>10.4f} {total:>10.3f} {pct:>6.1f}%")

    print("-" * 90)

    # Group by layer for quick comparison
    layer_keys = [r for r in rows if '.layers.' in r[0] and r[0].endswith('.total')]
    if layer_keys:
        print(f"\nLayer-by-layer comparison (32 layers):")
        layer_times = [(r[0], r[1]) for r in layer_keys]
        layer_means = [lt[1] for lt in layer_times]
        if min(layer_means) > 0:
            spread = max(layer_means) / min(layer_means)
            print(f"  min layer mean: {min(layer_means):.4f}s  "
                  f"max layer mean: {max(layer_means):.4f}s  "
                  f"spread: {spread:.2f}x")

        # Slowest 3 layers
        layer_times.sort(key=lambda x: x[1], reverse=True)
        print(f"  slowest 3: {', '.join(f'{k} ({v:.4f}s)' for k, v in layer_times[:3])}")
        print(f"  fastest 3: {', '.join(f'{k} ({v:.4f}s)' for k, v in layer_times[-3:])}")

    print("=" * 90)


def self_test():
    """Synthetic records — verifies parsing, aggregation, ranking."""
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'rank0.jsonl')

    # Simulate 5 intervals, 3 layers, each with total > self_attn + mlp (layernorm gap)
    with open(path, 'w') as f:
        for i in range(5):
            rec = {'step': (i + 1) * 10, 'steps': 10}
            # Layer 0: attn-heavy
            rec['model.layers.0.total'] = 0.050 + i * 0.001
            rec['model.layers.0.self_attn'] = 0.030 + i * 0.001
            rec['model.layers.0.mlp'] = 0.015 + i * 0.001
            # Layer 1: mlp-heavy
            rec['model.layers.1.total'] = 0.045 + i * 0.001
            rec['model.layers.1.self_attn'] = 0.015 + i * 0.001
            rec['model.layers.1.mlp'] = 0.025 + i * 0.001
            # Embed
            rec['model.embed'] = 0.002 + i * 0.0001
            # Norm
            rec['model.norm'] = 0.001 + i * 0.0001
            # lm_head+loss
            rec['lm_head+loss'] = 0.080 + i * 0.002
            f.write(json.dumps(rec) + '\n')

    records = load_records(path)
    assert len(records) == 5, f"expected 5 records, got {len(records)}"

    rows = summarize(records)
    keys = [r[0] for r in rows]

    # lm_head+loss should be the biggest (0.08s+)
    assert 'lm_head+loss' in keys, "lm_head+loss missing"
    idx_lm = keys.index('lm_head+loss')
    assert rows[idx_lm][1] > 0.07, f"lm_head+loss mean too low: {rows[idx_lm][1]}"

    # layer 0 should be > layer 1 (it's attn-heavy, and self_attn > mlp in this test)
    # Actually layer 0 total = 0.050 + avg, layer 1 total = 0.045 + avg, so layer 0 > layer 1
    idx_l0 = [i for i, r in enumerate(rows) if r[0] == 'model.layers.0.total'][0]
    idx_l1 = [i for i, r in enumerate(rows) if r[0] == 'model.layers.1.total'][0]
    assert idx_l0 < idx_l1, "layer 0 should rank higher than layer 1"

    print("self-test OK: parsing, aggregation, ranking all correct")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submodule_log', help='Path to submodule_times/rank0.jsonl')
    ap.add_argument('--self-test', action='store_true', help='Run built-in verification')
    ap.add_argument('--top', type=int, default=50, help='Show top N keys (default 50)')
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.submodule_log:
        ap.error('--submodule_log is required (or use --self-test)')

    records = load_records(args.submodule_log)
    if not records:
        print('No records found in', args.submodule_log)
        return 1

    rows = summarize(records)
    report(rows[:args.top], records)
    return 0


if __name__ == '__main__':
    sys.exit(main())
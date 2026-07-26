#!/usr/bin/env python3
"""
Compare inference outputs from two runs (e.g. GPU vs NPU).

Primary metric: action-space ADE between GPU and NPU decoded actions.
Also reports token-level agreement for diagnostic purposes.

Usage:
  python scripts/scripts_infer/cmp_inference_outputs.py <dir_a> <dir_b>
"""
import json, os, sys
import numpy as np


def relative_to_absolute(actions):
    """Convert relative (dx, dy, dtheta) to absolute (x, y) positions."""
    positions = np.zeros((len(actions), 2))
    x, y, theta = 0.0, 0.0, 0.0
    for i, row in enumerate(actions):
        dx, dy = row[0], row[1]
        dtheta = row[2] if len(row) >= 3 else 0.0
        x += dx * np.cos(theta) - dy * np.sin(theta)
        y += dx * np.sin(theta) + dy * np.cos(theta)
        theta += dtheta
        positions[i] = [x, y]
    return positions


def compute_ade(pred, gt):
    """Average Displacement Error: mean L2 distance over all time steps."""
    pred_pos = relative_to_absolute(pred)
    gt_pos = relative_to_absolute(gt)
    return np.linalg.norm(pred_pos - gt_pos, axis=1).mean()


def main():
    dir_a, dir_b = sys.argv[1], sys.argv[2]
    files_a = sorted(f for f in os.listdir(dir_a) if f.endswith(".json"))
    files_b = sorted(f for f in os.listdir(dir_b) if f.endswith(".json"))
    common = sorted(set(files_a) & set(files_b))

    print(f"Files: A={len(files_a)}, B={len(files_b)}, common={len(common)}")

    gpu_npu_ade = []
    gpu_gt_ade = []
    npu_gt_ade = []
    token_match = 0
    token_total = 0
    sample_match = 0
    skipped = 0

    for fname in common:
        with open(os.path.join(dir_a, fname)) as f:
            da = json.load(f)
        with open(os.path.join(dir_b, fname)) as f:
            db = json.load(f)

        # Token-level
        ta = da.get("raw_tokens", [])
        tb = db.get("raw_tokens", [])
        n = min(len(ta), len(tb))
        token_total += n
        token_match += sum(1 for i in range(n) if ta[i] == tb[i])
        if ta == tb:
            sample_match += 1

        # Action-level
        aa = np.array(da.get("action", []))
        ab = np.array(db.get("action", []))
        gt = np.array(da.get("action_gt_denorm", []))
        if aa.size == 0 or ab.size == 0 or aa.shape != ab.shape:
            skipped += 1
            continue
        if np.allclose(aa, 0) or np.allclose(ab, 0):
            skipped += 1
            continue

        gpu_npu_ade.append(compute_ade(aa, ab))
        if gt.size > 0 and gt.shape == aa.shape:
            gpu_gt_ade.append(compute_ade(aa, gt))
            npu_gt_ade.append(compute_ade(ab, gt))

    # --- Report ---
    n = len(gpu_npu_ade)
    print(f"\nValid samples (non-zero actions, both sides): {n}  (skipped: {skipped})")

    if n == 0:
        print("\nNo valid samples to compare. Model may be too undertrained to decode actions.")
        print(f"Token match rate: {token_match}/{token_total} = {100*token_match/max(token_total,1):.2f}%")
        print(f"Sample match rate: {sample_match}/{len(common)} = {100*sample_match/max(len(common),1):.2f}%")
        return

    gpu_npu_ade = np.array(gpu_npu_ade)

    print(f"\n{'='*55}")
    print(f"  GPU vs NPU — Action-space comparison")
    print(f"{'='*55}")
    print(f"  Mean ADE (GPU↔NPU):  {gpu_npu_ade.mean():.6f} m")
    print(f"  Median ADE:             {np.median(gpu_npu_ade):.6f} m")
    print(f"  Min ADE:                {gpu_npu_ade.min():.6f} m")
    print(f"  Max ADE:                {gpu_npu_ade.max():.6f} m")
    print(f"  Samples with ADE < 0.01m:  {(gpu_npu_ade < 0.01).sum()}/{n}")
    print(f"  Samples with ADE < 0.001m: {(gpu_npu_ade < 0.001).sum()}/{n}")

    if len(gpu_gt_ade) > 0:
        gpu_gt_ade = np.array(gpu_gt_ade)
        npu_gt_ade = np.array(npu_gt_ade)
        print(f"\n{'='*55}")
        print(f"  Reference: GPU & NPU vs Ground Truth")
        print(f"{'='*55}")
        print(f"  GPU Mean ADE vs GT:  {gpu_gt_ade.mean():.6f} m")
        print(f"  NPU Mean ADE vs GT:  {npu_gt_ade.mean():.6f} m")
        print(f"  GPU↔GT vs NPU↔GT correlation:  {np.corrcoef(gpu_gt_ade, npu_gt_ade)[0,1]:.6f}")

    print(f"\n{'='*55}")
    print(f"  Token-level (diagnostic)")
    print(f"{'='*55}")
    print(f"  Token match rate:     {token_match}/{token_total} = {100*token_match/max(token_total,1):.2f}%")
    print(f"  Sample match rate:    {sample_match}/{len(common)} = {100*sample_match/max(len(common),1):.2f}%")

    # Verdict
    print(f"\n{'='*55}")
    if sample_match == len(common):
        print("✓ Bit-exact match across all samples.")
    elif gpu_npu_ade.mean() < 0.01:
        print("✓ GPU and NPU actions nearly identical (mean ADE < 0.01 m).")
    elif gpu_npu_ade.mean() < 0.1:
        print("⚠ Minor action differences (mean ADE < 0.1 m) — likely FP rounding.")
    else:
        print("✗ Significant divergence — check training reproducibility.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Compare inference outputs from two runs (e.g. GPU vs NPU).

Generates metrics, plots, and a summary table covering:
  - Action-space ADE (GPU↔NPU, GPU↔GT, NPU↔GT)
  - Per-dimension MAE (dx, dy, dtheta)
  - Per-timestep ADE (where divergence grows)
  - Token-level agreement
  - Plots saved to <output_dir>/cmp_plots/

Usage:
  python scripts/scripts_infer/cmp_inference_outputs.py <dir_a> <dir_b> [--output_dir <dir>]
"""
import json, os, sys, argparse
import numpy as np

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


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
    pred_pos = relative_to_absolute(pred)
    gt_pos = relative_to_absolute(gt)
    return np.linalg.norm(pred_pos - gt_pos, axis=1).mean()


def compute_fde(pred, gt):
    pred_pos = relative_to_absolute(pred)
    gt_pos = relative_to_absolute(gt)
    return np.linalg.norm(pred_pos[-1] - gt_pos[-1])


def compute_per_step_ade(pred, gt):
    """ADE at each timestep: (T,) array of L2 distances."""
    pred_pos = relative_to_absolute(pred)
    gt_pos = relative_to_absolute(gt)
    return np.linalg.norm(pred_pos - gt_pos, axis=1)


def plot_ade_histogram(gpu_npu_ade, gpu_gt_ade, npu_gt_ade, out_dir):
    """Histogram of ADE distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bins = 30

    axes[0].hist(gpu_npu_ade, bins=bins, color="purple", alpha=0.7, edgecolor="black")
    axes[0].axvline(np.mean(gpu_npu_ade), color="red", linestyle="--", label=f"Mean={np.mean(gpu_npu_ade):.4f}")
    axes[0].set_title("GPU ↔ NPU ADE")
    axes[0].set_xlabel("ADE (m)")
    axes[0].legend()

    axes[1].hist(gpu_gt_ade, bins=bins, color="blue", alpha=0.7, edgecolor="black")
    axes[1].axvline(np.mean(gpu_gt_ade), color="red", linestyle="--", label=f"Mean={np.mean(gpu_gt_ade):.4f}")
    axes[1].set_title("GPU ↔ GT ADE")
    axes[1].set_xlabel("ADE (m)")
    axes[1].legend()

    axes[2].hist(npu_gt_ade, bins=bins, color="green", alpha=0.7, edgecolor="black")
    axes[2].axvline(np.mean(npu_gt_ade), color="red", linestyle="--", label=f"Mean={np.mean(npu_gt_ade):.4f}")
    axes[2].set_title("NPU ↔ GT ADE")
    axes[2].set_xlabel("ADE (m)")
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "ade_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_step_ade(per_step_ade_list, out_dir):
    """Mean per-timestep ADE with std band."""
    steps = np.array(per_step_ade_list)  # (N, T)
    mean = steps.mean(axis=0)
    std = steps.std(axis=0)
    t = np.arange(1, len(mean) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, mean, "b-o", markersize=6, label="GPU↔NPU ADE")
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, color="b")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Per-Timestep GPU↔NPU ADE (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "per_step_ade.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_action_scatter(aa, ab, out_dir):
    """Scatter GPU vs NPU for each action dimension (dx, dy, dtheta)."""
    dims = ["dx", "dy", "dtheta"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, label) in enumerate(zip(axes, dims)):
        gpu_vals = aa[:, :, i].ravel()
        npu_vals = ab[:, :, i].ravel()
        ax.scatter(gpu_vals, npu_vals, alpha=0.3, s=4)
        vmin = min(gpu_vals.min(), npu_vals.min())
        vmax = max(gpu_vals.max(), npu_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1, label="y=x")
        ax.set_xlabel(f"GPU {label}")
        ax.set_ylabel(f"NPU {label}")
        ax.set_title(f"{label} — GPU vs NPU")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "action_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trajectory_samples(aa_list, ab_list, gt_list, out_dir, n_samples=4):
    """Overlay GPU, NPU, and GT trajectories for a few samples."""
    n = min(n_samples, len(aa_list))
    indices = np.linspace(0, len(aa_list) - 1, n, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        gpu_pos = relative_to_absolute(aa_list[idx])
        npu_pos = relative_to_absolute(ab_list[idx])
        gt_pos = relative_to_absolute(gt_list[idx]) if idx < len(gt_list) else None

        ax.plot(gpu_pos[:, 0], gpu_pos[:, 1], "b-o", markersize=4, label="GPU")
        ax.plot(npu_pos[:, 0], npu_pos[:, 1], "g--s", markersize=4, label="NPU")
        if gt_pos is not None:
            ax.plot(gt_pos[:, 0], gt_pos[:, 1], "k-^", markersize=4, label="GT")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Sample {idx}")
        ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "trajectory_samples.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare GPU vs NPU inference outputs")
    parser.add_argument("dir_a", help="First inference output directory")
    parser.add_argument("dir_b", help="Second inference output directory")
    parser.add_argument("--output_dir", "-o", default=None, help="Directory for plots (default: dir_a/cmp_plots)")
    args = parser.parse_args()

    dir_a, dir_b = args.dir_a, args.dir_b
    files_a = sorted(f for f in os.listdir(dir_a) if f.endswith(".json"))
    files_b = sorted(f for f in os.listdir(dir_b) if f.endswith(".json"))
    common = sorted(set(files_a) & set(files_b))

    print(f"Files: A={len(files_a)}, B={len(files_b)}, common={len(common)}")

    out_dir = args.output_dir or os.path.join(dir_a, "cmp_plots")
    os.makedirs(out_dir, exist_ok=True)

    # Collectors
    gpu_npu_ade = []
    gpu_gt_ade = []
    npu_gt_ade = []
    gpu_npu_fde = []
    per_step_ade_list = []
    dim_mae = {0: [], 1: [], 2: []}  # dx, dy, dtheta
    all_aa = []  # all GPU actions for scatter
    all_ab = []  # all NPU actions
    all_gt = []  # all GT actions
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
        gpu_npu_fde.append(compute_fde(aa, ab))
        per_step_ade_list.append(compute_per_step_ade(aa, ab))

        for dim in range(3):
            dim_mae[dim].append(np.mean(np.abs(aa[:, dim] - ab[:, dim])))

        all_aa.append(aa)
        all_ab.append(ab)
        if gt.size > 0 and gt.shape == aa.shape:
            gpu_gt_ade.append(compute_ade(aa, gt))
            npu_gt_ade.append(compute_ade(ab, gt))
            all_gt.append(gt)

    # --- Text Report ---
    n = len(gpu_npu_ade)
    print(f"\nValid samples: {n}  (skipped: {skipped})")

    if n == 0:
        print("\nNo valid samples to compare. Model may be too undertrained to decode actions.")
        print(f"Token match rate: {token_match}/{token_total} = {100*token_match/max(token_total,1):.2f}%")
        print(f"Sample match rate: {sample_match}/{len(common)} = {100*sample_match/max(len(common),1):.2f}%")
        return

    gpu_npu_ade = np.array(gpu_npu_ade)
    gpu_npu_fde = np.array(gpu_npu_fde)

    # ---- Summary Table ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE — GPU vs NPU Inference Comparison")
    print(f"{'='*70}")
    print(f"  {'Metric':<40} {'Value':>25}")
    print(f"  {'-'*65}")
    print(f"  {'Samples compared':<40} {n:>25d}")
    print(f"  {'Samples skipped (zero actions)':<40} {skipped:>25d}")
    print(f"  {'':<40} {'':>25}")
    print(f"  {'--- Action-space metrics ---':<40} {'':>25}")
    print(f"  {'Mean ADE (GPU↔NPU)':<40} {gpu_npu_ade.mean():>25.6f} m")
    print(f"  {'Median ADE (GPU↔NPU)':<40} {np.median(gpu_npu_ade):>25.6f} m")
    print(f"  {'Min ADE (GPU↔NPU)':<40} {gpu_npu_ade.min():>25.6f} m")
    print(f"  {'Max ADE (GPU↔NPU)':<40} {gpu_npu_ade.max():>25.6f} m")
    print(f"  {'Std ADE (GPU↔NPU)':<40} {gpu_npu_ade.std():>25.6f} m")
    print(f"  {'Mean FDE (GPU↔NPU)':<40} {gpu_npu_fde.mean():>25.6f} m")
    print(f"  {'Samples ADE < 0.01m':<40} {(gpu_npu_ade < 0.01).sum():>25d} / {n}")
    print(f"  {'Samples ADE < 0.001m':<40} {(gpu_npu_ade < 0.001).sum():>25d} / {n}")
    print(f"  {'':<40} {'':>25}")
    print(f"  {'--- Per-dimension MAE (GPU↔NPU) ---':<40} {'':>25}")
    for dim, label in [(0, "dx"), (1, "dy"), (2, "dtheta")]:
        vals = np.array(dim_mae[dim])
        print(f"  {'  ' + label + ' MAE':<40} {vals.mean():>25.6f}")
    print(f"  {'':<40} {'':>25}")
    print(f"  {'--- Token-level (diagnostic) ---':<40} {'':>25}")
    print(f"  {'Token match rate':<40} {token_match:>25d} / {token_total} ({100*token_match/max(token_total,1):.2f}%)")
    print(f"  {'Sample match rate (bit-exact)':<40} {sample_match:>25d} / {len(common)} ({100*sample_match/max(len(common),1):.2f}%)")

    if len(gpu_gt_ade) > 0:
        gpu_gt_ade = np.array(gpu_gt_ade)
        npu_gt_ade = np.array(npu_gt_ade)
        print(f"  {'':<40} {'':>25}")
        print(f"  {'--- Reference: vs Ground Truth ---':<40} {'':>25}")
        print(f"  {'GPU Mean ADE vs GT':<40} {gpu_gt_ade.mean():>25.6f} m")
        print(f"  {'NPU Mean ADE vs GT':<40} {npu_gt_ade.mean():>25.6f} m")
        print(f"  {'GPU↔GT vs NPU↔GT correlation':<40} {np.corrcoef(gpu_gt_ade, npu_gt_ade)[0,1]:>25.6f}")
        gt_gap = abs(gpu_gt_ade.mean() - npu_gt_ade.mean())
        print(f"  {'|GPU ADE - NPU ADE| vs GT':<40} {gt_gap:>25.6f} m")

    # Verdict
    print(f"\n{'='*70}")
    if sample_match == len(common):
        print("✓ VERDICT: Bit-exact match across all samples.")
    elif gpu_npu_ade.mean() < 0.01:
        print("✓ VERDICT: GPU and NPU actions nearly identical (mean ADE < 0.01 m).")
    elif gpu_npu_ade.mean() < 0.1:
        print("⚠ VERDICT: Minor action differences (mean ADE < 0.1 m) — likely FP rounding.")
    else:
        print("✗ VERDICT: Significant divergence — check training reproducibility.")

    # Per-timestep breakdown
    print(f"\n{'='*70}")
    print(f"  PER-TIMESTEP ADE BREAKDOWN (GPU↔NPU)")
    print(f"{'='*70}")
    steps = np.array(per_step_ade_list)
    print(f"  {'Step':<8} {'Mean ADE':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*56}")
    for t in range(steps.shape[1]):
        print(f"  {t+1:<8} {steps[:,t].mean():>12.6f} {steps[:,t].std():>12.6f} {steps[:,t].min():>12.6f} {steps[:,t].max():>12.6f}")

    # --- Plots ---
    if HAS_MPL:
        print(f"\n{'='*70}")
        print(f"  Generating plots → {out_dir}/")
        print(f"{'='*70}")

        plot_ade_histogram(gpu_npu_ade, gpu_gt_ade if len(gpu_gt_ade) > 0 else np.array([]),
                           npu_gt_ade if len(npu_gt_ade) > 0 else np.array([]), out_dir)
        plot_per_step_ade(per_step_ade_list, out_dir)

        all_aa_arr = np.stack(all_aa)  # (N, 8, 3)
        all_ab_arr = np.stack(all_ab)
        plot_action_scatter(all_aa_arr, all_ab_arr, out_dir)

        if len(all_gt) > 0:
            plot_trajectory_samples(all_aa, all_ab, all_gt, out_dir)

        print(f"\n  All plots saved to: {out_dir}/")
    else:
        print("\n  (matplotlib not available — skipping plots. Install with: pip install matplotlib)")


if __name__ == "__main__":
    main()
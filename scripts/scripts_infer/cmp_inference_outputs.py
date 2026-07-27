#!/usr/bin/env python3
"""
Compare inference outputs from two runs (e.g. GPU vs NPU) and against ground truth.

Generates metrics, plots, and a summary report covering:
  - Action-space ADE / FDE (GPU↔NPU, GPU↔GT, NPU↔GT)
  - Per-dimension MAE (dx, dy, dtheta) for all three pairs
  - Per-timestep ADE breakdown for all three pairs
  - Token-level agreement
  - Plots saved to <output_dir>/cmp_plots/
  - Report saved to <output_dir>/report.txt

Usage:
  python scripts/scripts_infer/cmp_inference_outputs.py <dir_a> <dir_b> [--output_dir <dir>]
"""
import json, os, sys, argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def relative_to_absolute(actions):
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
    pred_pos = relative_to_absolute(pred)
    gt_pos = relative_to_absolute(gt)
    return np.linalg.norm(pred_pos - gt_pos, axis=1)


# ---- Plotting ----

def plot_ade_histogram(gpu_npu_ade, gpu_gt_ade, npu_gt_ade, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bins = 30

    datasets = [
        (gpu_npu_ade, "purple", "GPU ↔ NPU ADE"),
        (gpu_gt_ade, "blue", "GPU ↔ GT ADE"),
        (npu_gt_ade, "green", "NPU ↔ GT ADE"),
    ]
    for ax, (data, color, title) in zip(axes, datasets):
        if len(data) == 0:
            continue
        ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean={np.mean(data):.4f}")
        ax.set_title(title)
        ax.set_xlabel("ADE (m)")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "ade_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_step_ade(gpu_npu_steps, gpu_gt_steps, npu_gt_steps, out_dir):
    """Per-timestep ADE: GPU↔NPU, GPU↔GT, NPU↔GT on one plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    series = [
        (gpu_npu_steps, "purple", "GPU↔NPU"),
        (gpu_gt_steps, "blue", "GPU↔GT"),
        (npu_gt_steps, "green", "NPU↔GT"),
    ]
    for steps_list, color, label in series:
        if len(steps_list) == 0:
            continue
        steps = np.array(steps_list)
        mean = steps.mean(axis=0)
        std = steps.std(axis=0)
        t = np.arange(1, len(mean) + 1)
        ax.plot(t, mean, "-o", color=color, markersize=5, label=label)
        ax.fill_between(t, mean - std, mean + std, alpha=0.12, color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Per-Timestep ADE (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "per_step_ade.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_action_scatter(a_vals, b_vals, label_a, label_b, dim_labels, out_dir, suffix):
    """Scatter A vs B for each action dimension."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, dlabel) in enumerate(zip(axes, dim_labels)):
        x = a_vals[:, :, i].ravel()
        y = b_vals[:, :, i].ravel()
        ax.scatter(x, y, alpha=0.3, s=4)
        vmin = min(x.min(), y.min())
        vmax = max(x.max(), y.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1, label="y=x")
        ax.set_xlabel(f"{label_a} {dlabel}")
        ax.set_ylabel(f"{label_b} {dlabel}")
        ax.set_title(f"{dlabel} — {label_a} vs {label_b}")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"action_scatter_{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trajectory_samples(aa_list, ab_list, gt_list, label_a, label_b, out_dir, n_samples=4):
    n = min(n_samples, len(aa_list))
    indices = np.linspace(0, len(aa_list) - 1, n, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        a_pos = relative_to_absolute(aa_list[idx])
        b_pos = relative_to_absolute(ab_list[idx])
        gt_pos = relative_to_absolute(gt_list[idx]) if idx < len(gt_list) else None

        ax.plot(a_pos[:, 0], a_pos[:, 1], "b-o", markersize=4, label=label_a)
        ax.plot(b_pos[:, 0], b_pos[:, 1], "g--s", markersize=4, label=label_b)
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


# ---- Report helpers ----

def print_metric_section(title, metrics):
    """Print a list of (name, value) pairs as a formatted section."""
    print(f"\n  {'--- ' + title + ' ---':<40} {'':>25}")
    for name, value in metrics:
        print(f"  {name:<40} {value:>25}")


def print_per_step_table(title, steps_list):
    """Print per-timestep ADE breakdown table."""
    if len(steps_list) == 0:
        return
    print(f"\n  {'--- ' + title + ' ---':<40} {'':>25}")
    steps = np.array(steps_list)
    print(f"  {'Step':<8} {'Mean ADE':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*56}")
    for t in range(steps.shape[1]):
        print(f"  {t+1:<8} {steps[:,t].mean():>12.6f} {steps[:,t].std():>12.6f} {steps[:,t].min():>12.6f} {steps[:,t].max():>12.6f}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Compare GPU vs NPU inference outputs and vs GT")
    parser.add_argument("dir_a", help="First inference output directory (e.g. GPU)")
    parser.add_argument("dir_b", help="Second inference output directory (e.g. NPU)")
    parser.add_argument("--output_dir", "-o", default=None, help="Directory for plots and report (default: dir_a/cmp_plots)")
    parser.add_argument("--label_a", default="GPU", help="Label for dir_a (default: GPU)")
    parser.add_argument("--label_b", default="NPU", help="Label for dir_b (default: NPU)")
    args = parser.parse_args()

    dir_a, dir_b = args.dir_a, args.dir_b
    label_a, label_b = args.label_a, args.label_b

    files_a = sorted(f for f in os.listdir(dir_a) if f.endswith(".json"))
    files_b = sorted(f for f in os.listdir(dir_b) if f.endswith(".json"))
    common = sorted(set(files_a) & set(files_b))

    print(f"Comparison: {label_a} (n={len(files_a)}) vs {label_b} (n={len(files_b)})")
    print(f"Common samples: {len(common)}")

    out_dir = args.output_dir or os.path.join(dir_a, "cmp_plots")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Collectors ----
    # A↔B
    ab_ade, ab_fde = [], []
    ab_per_step = []
    # A↔GT
    a_gt_ade, a_gt_fde = [], []
    a_gt_per_step = []
    # B↔GT
    b_gt_ade, b_gt_fde = [], []
    b_gt_per_step = []
    # Per-dim MAE: A↔B, A↔GT, B↔GT
    dim_mae_ab = {0: [], 1: [], 2: []}
    dim_mae_a_gt = {0: [], 1: [], 2: []}
    dim_mae_b_gt = {0: [], 1: [], 2: []}
    # Action arrays for scatter
    all_aa, all_ab, all_gt = [], [], []
    # Token-level
    token_match, token_total = 0, 0
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

        # A↔B
        ab_ade.append(compute_ade(aa, ab))
        ab_fde.append(compute_fde(aa, ab))
        ab_per_step.append(compute_per_step_ade(aa, ab))
        for dim in range(3):
            dim_mae_ab[dim].append(np.mean(np.abs(aa[:, dim] - ab[:, dim])))

        all_aa.append(aa)
        all_ab.append(ab)

        # vs GT (only if GT is valid)
        if gt.size > 0 and gt.shape == aa.shape:
            a_gt_ade.append(compute_ade(aa, gt))
            a_gt_fde.append(compute_fde(aa, gt))
            a_gt_per_step.append(compute_per_step_ade(aa, gt))
            b_gt_ade.append(compute_ade(ab, gt))
            b_gt_fde.append(compute_fde(ab, gt))
            b_gt_per_step.append(compute_per_step_ade(ab, gt))
            for dim in range(3):
                dim_mae_a_gt[dim].append(np.mean(np.abs(aa[:, dim] - gt[:, dim])))
                dim_mae_b_gt[dim].append(np.mean(np.abs(ab[:, dim] - gt[:, dim])))
            all_gt.append(gt)

    # ---- Report ----
    n = len(ab_ade)
    print(f"\nValid samples: {n}  (skipped: {skipped})")

    if n == 0:
        print("\nNo valid samples to compare. Model may be too undertrained to decode actions.")
        print(f"Token match rate: {token_match}/{token_total} = {100*token_match/max(token_total,1):.2f}%")
        print(f"Sample match rate: {sample_match}/{len(common)} = {100*sample_match/max(len(common),1):.2f}%")
        return

    ab_ade = np.array(ab_ade)
    ab_fde = np.array(ab_fde)
    has_gt = len(a_gt_ade) > 0
    if has_gt:
        a_gt_ade = np.array(a_gt_ade)
        a_gt_fde = np.array(a_gt_fde)
        b_gt_ade = np.array(b_gt_ade)
        b_gt_fde = np.array(b_gt_fde)

    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE — {label_a} vs {label_b} Inference Comparison")
    print(f"{'='*70}")
    print(f"  {'Metric':<40} {'Value':>25}")
    print(f"  {'-'*65}")

    sections = [
        ("General", [
            ("Samples compared", f"{n}"),
            ("Samples skipped (zero actions)", f"{skipped}"),
        ]),
        (f"Action-space: {label_a} ↔ {label_b}", [
            ("Mean ADE", f"{ab_ade.mean():.6f} m"),
            ("Median ADE", f"{np.median(ab_ade):.6f} m"),
            ("Min ADE", f"{ab_ade.min():.6f} m"),
            ("Max ADE", f"{ab_ade.max():.6f} m"),
            ("Std ADE", f"{ab_ade.std():.6f} m"),
            ("Mean FDE", f"{ab_fde.mean():.6f} m"),
            (f"Samples ADE < 0.01m", f"{(ab_ade < 0.01).sum()} / {n}"),
            (f"Samples ADE < 0.001m", f"{(ab_ade < 0.001).sum()} / {n}"),
        ]),
        (f"Per-dimension MAE: {label_a} ↔ {label_b}", [
            (f"  dx MAE", f"{np.array(dim_mae_ab[0]).mean():.6f}"),
            (f"  dy MAE", f"{np.array(dim_mae_ab[1]).mean():.6f}"),
            (f"  dtheta MAE", f"{np.array(dim_mae_ab[2]).mean():.6f}"),
        ]),
        ("Token-level (diagnostic)", [
            ("Token match rate", f"{token_match}/{token_total} ({100*token_match/max(token_total,1):.2f}%)"),
            ("Sample match rate (bit-exact)", f"{sample_match}/{len(common)} ({100*sample_match/max(len(common),1):.2f}%)"),
        ]),
    ]

    if has_gt:
        gt_gap = abs(a_gt_ade.mean() - b_gt_ade.mean())
        sections.append((f"Action-space: {label_a} ↔ GT", [
            ("Mean ADE", f"{a_gt_ade.mean():.6f} m"),
            ("Median ADE", f"{np.median(a_gt_ade):.6f} m"),
            ("Std ADE", f"{a_gt_ade.std():.6f} m"),
            ("Mean FDE", f"{a_gt_fde.mean():.6f} m"),
        ]))
        sections.append((f"Action-space: {label_b} ↔ GT", [
            ("Mean ADE", f"{b_gt_ade.mean():.6f} m"),
            ("Median ADE", f"{np.median(b_gt_ade):.6f} m"),
            ("Std ADE", f"{b_gt_ade.std():.6f} m"),
            ("Mean FDE", f"{b_gt_fde.mean():.6f} m"),
        ]))
        sections.append((f"Per-dimension MAE: {label_a} ↔ GT", [
            (f"  dx MAE", f"{np.array(dim_mae_a_gt[0]).mean():.6f}"),
            (f"  dy MAE", f"{np.array(dim_mae_a_gt[1]).mean():.6f}"),
            (f"  dtheta MAE", f"{np.array(dim_mae_a_gt[2]).mean():.6f}"),
        ]))
        sections.append((f"Per-dimension MAE: {label_b} ↔ GT", [
            (f"  dx MAE", f"{np.array(dim_mae_b_gt[0]).mean():.6f}"),
            (f"  dy MAE", f"{np.array(dim_mae_b_gt[1]).mean():.6f}"),
            (f"  dtheta MAE", f"{np.array(dim_mae_b_gt[2]).mean():.6f}"),
        ]))
        sections.append(("Cross-reference", [
            (f"{label_a}↔GT vs {label_b}↔GT correlation", f"{np.corrcoef(a_gt_ade, b_gt_ade)[0,1]:.6f}"),
            (f"|{label_a} ADE - {label_b} ADE| vs GT", f"{gt_gap:.6f} m"),
        ]))

    for title, metrics in sections:
        print_metric_section(title, metrics)

    # Verdict
    print(f"\n{'='*70}")
    if sample_match == len(common):
        print(f"✓ VERDICT: Bit-exact match across all samples.")
    elif ab_ade.mean() < 0.01:
        print(f"✓ VERDICT: {label_a} and {label_b} actions nearly identical (mean ADE < 0.01 m).")
    elif ab_ade.mean() < 0.1:
        print(f"⚠ VERDICT: Minor action differences (mean ADE < 0.1 m) — likely FP rounding.")
    else:
        print(f"✗ VERDICT: Significant divergence — check training reproducibility.")

    # Per-timestep breakdowns
    print(f"\n{'='*70}")
    print(f"  PER-TIMESTEP ADE BREAKDOWN")
    print(f"{'='*70}")

    print_per_step_table(f"{label_a} ↔ {label_b}", ab_per_step)
    if has_gt:
        print_per_step_table(f"{label_a} ↔ GT", a_gt_per_step)
        print_per_step_table(f"{label_b} ↔ GT", b_gt_per_step)

    # ---- Plots ----
    if HAS_MPL:
        print(f"\n{'='*70}")
        print(f"  Generating plots → {out_dir}/")
        print(f"{'='*70}")

        plot_ade_histogram(
            ab_ade,
            a_gt_ade if has_gt else np.array([]),
            b_gt_ade if has_gt else np.array([]),
            out_dir,
        )
        plot_per_step_ade(
            ab_per_step,
            a_gt_per_step if has_gt else [],
            b_gt_per_step if has_gt else [],
            out_dir,
        )

        all_aa_arr = np.stack(all_aa)
        all_ab_arr = np.stack(all_ab)
        dim_labels = ["dx", "dy", "dtheta"]

        plot_action_scatter(all_aa_arr, all_ab_arr, label_a, label_b, dim_labels, out_dir, "ab")
        if has_gt:
            all_gt_arr = np.stack(all_gt)
            plot_action_scatter(all_aa_arr, all_gt_arr, label_a, "GT", dim_labels, out_dir, "a_gt")
            plot_action_scatter(all_ab_arr, all_gt_arr, label_b, "GT", dim_labels, out_dir, "b_gt")
            plot_trajectory_samples(all_aa, all_ab, all_gt, label_a, label_b, out_dir)

        print(f"\n  All plots saved to: {out_dir}/")
    else:
        print("\n  (matplotlib not available — skipping plots. pip install matplotlib)")

    # ---- Save report.txt ----
    report_path = os.path.join(out_dir, "report.txt")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    import sys

    # Parse args early to get output_dir for the report file
    _parser = argparse.ArgumentParser()
    _parser.add_argument("dir_a")
    _parser.add_argument("dir_b")
    _parser.add_argument("--output_dir", "-o", default=None)
    _parser.add_argument("--label_a", default="GPU")
    _parser.add_argument("--label_b", default="NPU")
    _args, _ = _parser.parse_known_args()

    _out_dir = _args.output_dir or os.path.join(_args.dir_a, "cmp_plots")
    os.makedirs(_out_dir, exist_ok=True)
    _report_path = os.path.join(_out_dir, "report.txt")

    # Tee stdout to report.txt
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    _report_f = open(_report_path, "w")
    sys.stdout = Tee(sys.stdout, _report_f)

    try:
        main()
    finally:
        _report_f.close()
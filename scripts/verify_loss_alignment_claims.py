#!/usr/bin/env python3
"""
Verify the Sonnet 5 claims about cross-platform loss divergence.

Key testable predictions:
  1. RE peak should align with the grad_norm *transition zone* (high curvature → flat),
     not with a fixed step count.
  2. RE should decay as grad_norm stabilizes — the AR(1) story predicts
     Var(e) ∝ 1/(1-ρ²) where ρ decreases as the landscape flattens.
  3. The RE peak should correlate with grad_norm *variance* (noisy second moment),
     not just magnitude.

If the curvature story is correct, plotting RE against grad_norm (or its derivative)
should show a cleaner relationship than RE against step index.

Usage (on the server):
  python scripts/verify_loss_alignment_claims.py \
      --cuda_json logs/alignment_cuda_fp32_<TS>/tier0_warmup200/trainer_state.json \
      --npu_json  logs/alignment_npu_fp32_<TS>/tier0_warmup200/trainer_state.json \
      --output_dir ./verification_plots
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

# ── Data loading ────────────────────────────────────────────────────────────

def load_trainer_state(filepath):
    with open(filepath) as f:
        data = json.load(f)
    log_history = data.get("log_history", [])
    records = {"step": [], "loss": [], "grad_norm": [], "learning_rate": []}
    for entry in log_history:
        if "loss" in entry:
            for k in records:
                records[k].append(entry.get(k, 0))
    return {k: np.array(v) for k, v in records.items()}


def smooth(data, window=10):
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
    trim = (len(data) - len(smoothed)) // 2
    return smoothed, trim


# ── Metrics ─────────────────────────────────────────────────────────────────

def relative_error(npu, cuda):
    """Relative error with CUDA as baseline: (npu - cuda) / cuda * 100.

    Positive RE means NPU loss is ABOVE the CUDA reference (NPU needs aligning down);
    negative means NPU is below. The essential goal is aligning NPU to CUDA, so CUDA
    is the fixed baseline.
    """
    return (npu - cuda) / cuda * 100


def running_variance(x, window=20):
    """Running variance over a sliding window — proxy for curvature instability."""
    if len(x) < window:
        return np.full_like(x, np.nan)
    rv = np.array([np.var(x[max(0, i - window):i + 1]) for i in range(len(x))])
    return rv


def grad_norm_derivative(gn, window=5):
    """Smoothed first derivative of grad_norm — measures how fast curvature is changing."""
    if len(gn) < window + 1:
        return np.zeros_like(gn)
    smoothed, _ = smooth(gn, window)
    deriv = np.diff(smoothed)
    # Pad to match original length
    return np.concatenate([deriv, np.full(len(gn) - len(deriv), deriv[-1])])


# ── Analysis functions ──────────────────────────────────────────────────────

def analyze_peak_alignment(steps, re, gn_cuda, gn_npu):
    """
    Test: does the RE peak align with the grad_norm transition zone?

    The transition zone is where |d(gn)/dt| is largest — the point where
    the landscape is flattening fastest.  If the curvature story is correct,
    the RE peak should fall within or near this zone.
    """
    gn_avg = (gn_cuda + gn_npu) / 2.0
    gn_deriv = np.abs(grad_norm_derivative(gn_avg))

    re_peak_idx = np.argmax(np.abs(re))
    re_peak_step = steps[re_peak_idx]

    # Find the transition zone: steps where |d(gn)/dt| > 0.5 * max(|d(gn)/dt|)
    threshold = 0.5 * np.max(gn_deriv)
    transition_mask = gn_deriv > threshold
    if not transition_mask.any():
        return {
            "re_peak_step": re_peak_step,
            "re_peak_value": re[re_peak_idx],
            "transition_start": None,
            "transition_end": None,
            "peak_in_transition": False,
            "distance_to_transition": None,
        }

    trans_start = steps[transition_mask][0]
    trans_end = steps[transition_mask][-1]
    peak_in_transition = trans_start <= re_peak_step <= trans_end

    if not peak_in_transition:
        if re_peak_step < trans_start:
            dist = trans_start - re_peak_step
        else:
            dist = re_peak_step - trans_end
    else:
        dist = 0

    return {
        "re_peak_step": re_peak_step,
        "re_peak_value": re[re_peak_idx],
        "transition_start": trans_start,
        "transition_end": trans_end,
        "peak_in_transition": peak_in_transition,
        "distance_to_transition": dist,
    }


def analyze_re_vs_gradnorm(steps, re, gn_cuda, gn_npu):
    """
    Test: correlation between |RE| and grad_norm.

    Under the curvature story, higher |RE| should correlate with higher
    (or more rapidly changing) grad_norm.  Compute Spearman ρ and report.
    """
    gn_avg = (gn_cuda + gn_npu) / 2.0
    abs_re = np.abs(re)

    # Filter out step 0-5 (transient startup)
    mask = steps > 5
    if mask.sum() < 10:
        return {"error": "too few steps after step 5"}

    # Correlation: |RE| vs grad_norm
    r_gn, p_gn = stats.spearmanr(abs_re[mask], gn_avg[mask])

    # Correlation: |RE| vs step (null hypothesis: it's just a step effect)
    r_step, p_step = stats.spearmanr(abs_re[mask], steps[mask])

    # Correlation: |RE| vs grad_norm variance (second moment noise story)
    gn_var = running_variance(gn_avg)
    valid = mask & ~np.isnan(gn_var)
    r_var, p_var = stats.spearmanr(abs_re[valid], gn_var[valid])

    # Partial correlation: |RE| vs grad_norm, controlling for step
    # If curvature matters, this should remain significant
    try:
        from scipy.stats import rankdata
        # Spearman partial correlation via rank residuals
        r_rank = rankdata(abs_re[mask])
        gn_rank = rankdata(gn_avg[mask])
        step_rank = rankdata(steps[mask])

        # Regress out step from both
        r_resid = r_rank - np.polyval(np.polyfit(step_rank, r_rank, 1), step_rank)
        gn_resid = gn_rank - np.polyval(np.polyfit(step_rank, gn_rank, 1), step_rank)
        r_partial, p_partial = stats.spearmanr(r_resid, gn_resid)
    except Exception:
        r_partial, p_partial = np.nan, np.nan

    return {
        "spearman_re_vs_gradnorm": {"r": r_gn, "p": p_gn},
        "spearman_re_vs_step": {"r": r_step, "p": p_step},
        "spearman_re_vs_gradnorm_variance": {"r": r_var, "p": p_var},
        "spearman_re_vs_gradnorm_partial_step": {"r": r_partial, "p": p_partial},
        "conclusion": _conclusion_text(r_gn, p_gn, r_step, p_step, r_var, p_var, r_partial, p_partial),
    }


def _conclusion_text(r_gn, p_gn, r_step, p_step, r_var, p_var, r_partial, p_partial):
    lines = []
    if abs(r_gn) > abs(r_step) and not np.isnan(r_gn):
        lines.append(
            f"|RE| correlates more strongly with grad_norm (ρ={r_gn:.3f}) than with step (ρ={r_step:.3f})"
            f" — supports curvature story."
        )
    else:
        lines.append(
            f"|RE| correlates more strongly with step (ρ={r_step:.3f}) than with grad_norm (ρ={r_gn:.3f})"
            f" — step is a better predictor; curvature story is weaker."
        )

    if not np.isnan(r_partial) and abs(r_partial) > 0.1 and p_partial < 0.05:
        lines.append(
            f"Partial correlation |RE|~grad_norm|step: ρ={r_partial:.3f} (p={p_partial:.4f})"
            f" — grad_norm explains variance in RE beyond what step alone explains."
        )
    elif not np.isnan(r_partial):
        lines.append(
            f"Partial correlation |RE|~grad_norm|step: ρ={r_partial:.3f} (p={p_partial:.4f})"
            f" — grad_norm adds little beyond step."
        )

    if not np.isnan(r_var) and abs(r_var) > 0.2:
        lines.append(
            f"|RE| correlates with grad_norm variance ρ={r_var:.3f} (p={p_var:.4f})"
            f" — supports the noisy-second-moment mechanism."
        )

    return "\n".join(lines)


def analyze_ar1_model(steps, re, gn_cuda, gn_npu):
    """
    Test the AR(1) prediction: Var(e) should converge to a steady-state floor
    as the landscape flattens (grad_norm decreases).

    Split the trajectory into phases:
      - Phase 1: grad_norm > 50% of max (sharp curvature, expected high RE)
      - Phase 2: grad_norm 10-50% of max (transition)
      - Phase 3: grad_norm < 10% of max (flat, expected low RE)
    """
    gn_avg = (gn_cuda + gn_npu) / 2.0
    gn_max = np.max(gn_avg[steps > 5])  # skip initial spike

    p1 = steps > 5
    p1 &= gn_avg > 0.5 * gn_max
    p2 = steps > 5
    p2 &= (gn_avg > 0.1 * gn_max) & (gn_avg <= 0.5 * gn_max)
    p3 = steps > 5
    p3 &= gn_avg <= 0.1 * gn_max

    phases = {}
    for label, mask in [("phase1_high_curvature", p1), ("phase2_transition", p2), ("phase3_flat", p3)]:
        if mask.sum() < 3:
            phases[label] = {"n_steps": int(mask.sum()), "mean_abs_re": None, "std_abs_re": None}
        else:
            phases[label] = {
                "n_steps": int(mask.sum()),
                "step_range": (int(steps[mask].min()), int(steps[mask].max())),
                "mean_abs_re": float(np.mean(np.abs(re[mask]))),
                "std_abs_re": float(np.std(np.abs(re[mask]))),
                "mean_grad_norm": float(np.mean(gn_avg[mask])),
            }

    # The AR(1) prediction: mean_abs_re(phase3) < mean_abs_re(phase2) < mean_abs_re(phase1)
    means = [phases[p].get("mean_abs_re") for p in ["phase1_high_curvature", "phase2_transition", "phase3_flat"]]
    means = [m for m in means if m is not None]
    prediction_holds = len(means) >= 2 and all(
        means[i] > means[i + 1] for i in range(len(means) - 1)
    )

    return {
        "phases": phases,
        "prediction_holds": prediction_holds,
        "verdict": (
            "AR(1) prediction holds: RE decreases as grad_norm (curvature) decreases."
            if prediction_holds
            else "AR(1) prediction does NOT hold: RE does not monotonically decrease with grad_norm."
        ),
    }


def analyze_early_step_sensitivity(steps, re, gn_cuda, gn_npu):
    """
    Test claim (a): at small t, there's no averaging protection.

    Compute the per-step absolute change in RE: |ΔRE_t| = |RE_t - RE_{t-1}|.
    Under the "no averaging" claim, early steps should have larger |ΔRE|.
    """
    abs_dre = np.abs(np.diff(re))
    abs_dre_steps = steps[1:]

    # Split into early (steps 1-30) and late (steps 100+)
    early = abs_dre_steps <= 30
    late = abs_dre_steps >= 100

    early_mean = float(np.mean(abs_dre[early])) if early.any() else None
    late_mean = float(np.mean(abs_dre[late])) if late.any() else None
    ratio = early_mean / late_mean if (early_mean and late_mean and late_mean > 0) else None

    return {
        "mean_abs_delta_re_early": early_mean,
        "mean_abs_delta_re_late": late_mean,
        "ratio_early_late": ratio,
        "verdict": (
            f"Early steps have {ratio:.1f}× larger per-step RE changes than late steps"
            f" — supports the no-averaging-protection claim."
            if ratio and ratio > 1.5
            else "Early steps do NOT show substantially larger RE changes."
        ),
    }


def analyze_magnitude_bursts(steps, re, gn_cuda, gn_npu):
    """
    Test the cross-tier finding: RE spikes are driven by grad_norm MAGNITUDE
    (high-curvature bursts), not by which platform spikes.

    Checks:
      - |RE| vs max(gn) Spearman correlation (magnitude/curvature story)
      - mean|RE| when max(gn) > 3x run median vs when below (amplification)
      - asymmetry: mean|RE| when CUDA has the larger gradient vs NPU — a
        robust result should be roughly symmetric (platform identity irrelevant)
      - burst detection: contiguous runs where |RE| > 3x median|RE| — the
        AR(1) compounding fingerprint
    """
    abs_re = np.abs(re)
    gn_max = np.maximum(gn_cuda, gn_npu)
    med = np.median(np.concatenate([gn_cuda, gn_npu]))

    mask = steps > 5
    r_gn, p_gn = stats.spearmanr(abs_re[mask], gn_max[mask])

    hi = mask & (gn_max > 3 * med)
    lo = mask & (gn_max <= 3 * med)
    hi_mean = float(np.mean(abs_re[hi])) if hi.any() else None
    lo_mean = float(np.mean(abs_re[lo])) if lo.any() else None
    amplification = hi_mean / lo_mean if (hi_mean and lo_mean) else None

    # asymmetry: which platform carries the larger gradient at each step
    cuda_hi = gn_cuda > gn_npu
    npu_hi = gn_npu > gn_cuda
    mean_re_cuda = float(np.mean(abs_re[cuda_hi])) if cuda_hi.any() else None
    mean_re_npu = float(np.mean(abs_re[npu_hi])) if npu_hi.any() else None
    if mean_re_cuda and mean_re_npu:
        asymmetry_ratio = max(mean_re_cuda, mean_re_npu) / min(mean_re_cuda, mean_re_npu)
    else:
        asymmetry_ratio = None

    # burst detection: contiguous runs where |RE| > 3x median|RE|
    re_med = np.median(abs_re[mask])
    hot = abs_re > 3 * re_med
    bursts = []
    in_b = False
    for i in range(len(steps)):
        if hot[i] and not in_b:
            start = i
            in_b = True
        elif not hot[i] and in_b:
            bursts.append((int(steps[start]), int(steps[i - 1])))
            in_b = False
    if in_b:
        bursts.append((int(steps[start]), int(steps[-1])))

    # per-step burst id for CSV: 0 = not hot, else 1-based burst index
    burst_id = np.zeros(len(steps), dtype=int)
    for idx, (b0, b1) in enumerate(bursts, start=1):
        burst_id[(steps >= b0) & (steps <= b1)] = idx

    verdict_lines = []
    if amplification and amplification > 1.5:
        verdict_lines.append(
            f"Amplification CONFIRMED: mean|RE| is {amplification:.1f}× higher when "
            f"max(gn) > 3× median ({hi_mean:.3f}% vs {lo_mean:.3f}%)."
        )
    else:
        verdict_lines.append("Amplification WEAK: high-gradient steps do NOT clearly raise |RE|.")

    if asymmetry_ratio and asymmetry_ratio < 1.5:
        verdict_lines.append(
            f"Platform asymmetry ABSENT: mean|RE| is {asymmetry_ratio:.2f}× when CUDA "
            f"vs {asymmetry_ratio:.2f}×-equivalent when NPU carries the gradient "
            f"({mean_re_cuda:.3f}% vs {mean_re_npu:.3f}%) — identity is not predictive."
        )
    else:
        verdict_lines.append(
            f"Platform asymmetry PRESENT: mean|RE| differs "
            f"({mean_re_cuda:.3f}% CUDA-sided vs {mean_re_npu:.3f}% NPU-sided)."
        )

    if bursts:
        burst_str = "; ".join(f"{b0}-{b1}" for b0, b1 in bursts[:6])
        if len(bursts) > 6:
            burst_str += " ..."
        verdict_lines.append(f"RE bursts (contiguous, AR(1)-style): {burst_str}.")
    else:
        verdict_lines.append("No contiguous RE bursts above 3× median |RE|.")

    return {
        "spearman_re_vs_max_gn": {"r": r_gn, "p": p_gn},
        "mean_abs_re_high_grad": hi_mean,
        "mean_abs_re_low_grad": lo_mean,
        "amplification_ratio": amplification,
        "mean_abs_re_cuda_sided": mean_re_cuda,
        "mean_abs_re_npu_sided": mean_re_npu,
        "asymmetry_ratio": asymmetry_ratio,
        "bursts": bursts,
        "burst_id": burst_id,
        "verdict": "\n".join(verdict_lines),
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def make_plots(steps, re, gn_cuda, gn_npu, lr, peak_info, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    # ── Figure 1: RE vs step + RE vs grad_norm (2×2) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Verification: Curvature-Driven Divergence Model", fontsize=14, fontweight="bold")

    # (0,0): RE vs step
    ax = axes[0, 0]
    ax.plot(steps, re, alpha=0.4, color='#3498db', linewidth=1, marker='o', markersize=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=peak_info["re_peak_step"], color='#e74c3c', linestyle=':', linewidth=1.5,
               label=f"RE peak at step {peak_info['re_peak_step']}")
    if peak_info["transition_start"] is not None:
        ax.axvspan(peak_info["transition_start"], peak_info["transition_end"],
                   alpha=0.15, color='#f39c12', label="Grad norm transition zone")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"Loss RE vs Step (peak={peak_info['re_peak_value']:.2f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1): |RE| vs grad_norm (scatter)
    ax = axes[0, 1]
    gn_avg = (gn_cuda + gn_npu) / 2.0
    mask = steps > 5
    ax.scatter(gn_avg[mask], np.abs(re[mask]), c=steps[mask], cmap='viridis',
               alpha=0.6, s=30, edgecolors='none')
    ax.set_xlabel("Mean Grad Norm (CUDA + NPU) / 2")
    ax.set_ylabel("|Relative Error| (%)")
    ax.set_title("|RE| vs Grad Norm (color = step)")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Step")

    # (1,0): Grad norm over time (both platforms)
    ax = axes[1, 0]
    ax.plot(steps, gn_cuda, alpha=0.5, color='#e74c3c', linewidth=1, label="CUDA")
    ax.plot(steps, gn_npu, alpha=0.5, color='#2ecc71', linewidth=1, label="NPU")
    smoothed_gn_cuda, trim = smooth(gn_cuda, 10)
    smoothed_gn_npu, _ = smooth(gn_npu, 10)
    smoothed_steps = steps[trim:trim + len(smoothed_gn_cuda)]
    ax.plot(smoothed_steps, smoothed_gn_cuda, color='#c0392b', linewidth=2, label="CUDA smoothed")
    ax.plot(smoothed_steps, smoothed_gn_npu, color='#27ae60', linewidth=2, label="NPU smoothed")
    if peak_info["transition_start"] is not None:
        ax.axvspan(peak_info["transition_start"], peak_info["transition_end"],
                   alpha=0.15, color='#f39c12', label="Transition zone")
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad Norm")
    ax.set_title("Gradient Norm Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): |RE| vs |d(gn)/dt| (the key test)
    ax = axes[1, 1]
    gn_deriv = np.abs(grad_norm_derivative(gn_avg))
    mask = steps > 5
    ax.scatter(gn_deriv[mask], np.abs(re[mask]), c=steps[mask], cmap='viridis',
               alpha=0.6, s=30, edgecolors='none')
    ax.set_xlabel("|d(Grad Norm)/dt| (curvature change rate)")
    ax.set_ylabel("|Relative Error| (%)")
    ax.set_title("|RE| vs Curvature Change Rate (color = step)")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Step")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "verification_main.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()

    # ── Figure 2: Phase analysis (AR(1) model) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("AR(1) Model: RE Should Decrease as Curvature Decreases", fontsize=13, fontweight="bold")

    # Left: RE with phase shading
    ax = axes[0]
    gn_max = np.max(gn_avg[steps > 5])
    p1 = gn_avg > 0.5 * gn_max
    p2 = (gn_avg > 0.1 * gn_max) & (gn_avg <= 0.5 * gn_max)
    p3 = gn_avg <= 0.1 * gn_max

    ax.plot(steps, re, alpha=0.4, color='#3498db', linewidth=1)
    for mask, color, label in [(p1, '#e74c3c', 'Phase 1: High curvature'),
                                 (p2, '#f39c12', 'Phase 2: Transition'),
                                 (p3, '#2ecc71', 'Phase 3: Flat')]:
        if mask.any():
            ax.fill_between(steps, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                            alpha=0.1, color=color, label=label)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("RE with Curvature Phases")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: bar chart of mean |RE| per phase
    ax = axes[1]
    # Recompute phases with step>5 filter
    phase_labels = []
    phase_means = []
    phase_stds = []
    phase_colors = []
    for label, mask, color in [("Phase 1\nHigh κ", p1 & (steps > 5), '#e74c3c'),
                                 ("Phase 2\nTransition", p2 & (steps > 5), '#f39c12'),
                                 ("Phase 3\nFlat", p3 & (steps > 5), '#2ecc71')]:
        if mask.sum() >= 3:
            phase_labels.append(label)
            phase_means.append(np.mean(np.abs(re[mask])))
            phase_stds.append(np.std(np.abs(re[mask])))
            phase_colors.append(color)

    x = np.arange(len(phase_labels))
    ax.bar(x, phase_means, yerr=phase_stds, color=phase_colors, alpha=0.7, capsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("Mean |RE| (%)")
    ax.set_title("Mean |RE| by Curvature Phase")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "verification_phases.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()

    # ── Figure 3: Early-step sensitivity ──
    fig, ax = plt.subplots(figsize=(14, 5))
    abs_dre = np.abs(np.diff(re))
    ax.bar(steps[1:], abs_dre, color=['#e74c3c' if s <= 30 else '#95a5a6' for s in steps[1:]],
           alpha=0.6, width=0.8)
    ax.axvline(x=30, color='#e74c3c', linestyle='--', linewidth=1.5, label="Step 30 boundary")
    ax.set_xlabel("Step")
    ax.set_ylabel("|ΔRE| (per-step change)")
    ax.set_title("Per-Step RE Volatility: Early vs Late")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "verification_early_sensitivity.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify mathematical claims about cross-platform loss divergence."
    )
    parser.add_argument("--cuda_json", required=True, help="trainer_state.json from CUDA run")
    parser.add_argument("--npu_json", required=True, help="trainer_state.json from NPU run")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory for plots (default: cwd)")
    parser.add_argument("--no_plots", action="store_true", help="Skip plots, print analysis only")
    args = parser.parse_args()

    out = args.output_dir or os.path.dirname(args.cuda_json) or "."
    os.makedirs(out, exist_ok=True)

    # ── Load ──
    d_cuda = load_trainer_state(args.cuda_json)
    d_npu = load_trainer_state(args.npu_json)

    steps = d_cuda["step"]
    if len(steps) < 10:
        print("ERROR: fewer than 10 steps in trainer_state — not enough data")
        return 1

    loss_cuda = d_cuda["loss"]
    loss_npu = d_npu["loss"]
    gn_cuda = d_cuda["grad_norm"]
    gn_npu = d_npu["grad_norm"]
    lr = d_cuda["learning_rate"]

    # ── Compute ──
    re = relative_error(loss_npu, loss_cuda)  # CUDA as baseline

    print("=" * 72)
    print("VERIFICATION: Cross-Platform Loss Divergence Claims")
    print("=" * 72)
    print(f"Steps: {steps[0]:.0f} – {steps[-1]:.0f} ({len(steps)} total)")
    print(f"Loss CUDA: {loss_cuda[0]:.4f} → {loss_cuda[-1]:.4f}")
    print(f"Loss NPU:  {loss_npu[0]:.4f} → {loss_npu[-1]:.4f}")
    print(f"Max |RE|: {np.max(np.abs(re)):.2f}% at step {steps[np.argmax(np.abs(re))]:.0f}")
    print()

    # ── Test 1: Peak alignment with grad_norm transition ──
    print("─" * 72)
    print("TEST 1: Does RE peak align with grad_norm transition zone?")
    print("─" * 72)
    peak_info = analyze_peak_alignment(steps, re, gn_cuda, gn_npu)
    print(f"  RE peak: step {peak_info['re_peak_step']} ({peak_info['re_peak_value']:.2f}%)")
    if peak_info["transition_start"] is not None:
        print(f"  Transition zone: steps {peak_info['transition_start']} – {peak_info['transition_end']}")
        if peak_info["peak_in_transition"]:
            print(f"  ✓ Peak IS inside the transition zone")
        else:
            print(f"  ✗ Peak is {peak_info['distance_to_transition']} steps outside the transition zone")
    else:
        print(f"  (no clear transition zone detected)")
    print()

    # ── Test 2: Correlation analysis ──
    print("─" * 72)
    print("TEST 2: Does |RE| correlate with grad_norm (curvature proxy)?")
    print("─" * 72)
    corr = analyze_re_vs_gradnorm(steps, re, gn_cuda, gn_npu)
    if "error" in corr:
        print(f"  ERROR: {corr['error']}")
    else:
        for k, v in corr.items():
            if k == "conclusion":
                print(f"  {v}")
            elif isinstance(v, dict):
                print(f"  {k}: r={v['r']:.4f}, p={v['p']:.4f}")
    print()

    # ── Test 3: AR(1) model ──
    print("─" * 72)
    print("TEST 3: AR(1) model — does RE decrease as curvature decreases?")
    print("─" * 72)
    ar1 = analyze_ar1_model(steps, re, gn_cuda, gn_npu)
    for label, info in ar1["phases"].items():
        if info["mean_abs_re"] is not None:
            print(f"  {label}: {info['n_steps']} steps, "
                  f"mean|RE|={info['mean_abs_re']:.3f}%, "
                  f"mean gn={info['mean_grad_norm']:.2f}")
        else:
            print(f"  {label}: {info['n_steps']} steps (insufficient)")
    print(f"  {ar1['verdict']}")
    print()

    # ── Test 4: Early-step sensitivity ──
    print("─" * 72)
    print("TEST 4: Early-step sensitivity — no averaging protection at small t?")
    print("─" * 72)
    early = analyze_early_step_sensitivity(steps, re, gn_cuda, gn_npu)
    print(f"  Mean |ΔRE| early (≤30): {early['mean_abs_delta_re_early']:.3f}%")
    print(f"  Mean |ΔRE| late (≥100): {early['mean_abs_delta_re_late']:.3f}%")
    print(f"  Ratio early/late: {early['ratio_early_late']:.1f}×" if early['ratio_early_late'] else "")
    print(f"  {early['verdict']}")
    print()

    # ── Test 5: Magnitude bursts (cross-tier) ──
    print("─" * 72)
    print("TEST 5: Magnitude bursts — driven by grad_norm magnitude, not platform?")
    print("─" * 72)
    bursts = analyze_magnitude_bursts(steps, re, gn_cuda, gn_npu)
    print(f"  Spearman |RE| vs max(gn): ρ={bursts['spearman_re_vs_max_gn']['r']:.4f} "
          f"(p={bursts['spearman_re_vs_max_gn']['p']:.4f})")
    if bursts["mean_abs_re_high_grad"] is not None:
        print(f"  mean|RE| when max(gn)>3×med: {bursts['mean_abs_re_high_grad']:.3f}% vs "
              f"{bursts['mean_abs_re_low_grad']:.3f}% when below "
              f"(amplification {bursts['amplification_ratio']:.1f}×)")
    if bursts["mean_abs_re_cuda_sided"] is not None:
        print(f"  mean|RE| CUDA-sided: {bursts['mean_abs_re_cuda_sided']:.3f}% vs "
              f"NPU-sided: {bursts['mean_abs_re_npu_sided']:.3f}% "
              f"(asymmetry {bursts['asymmetry_ratio']:.2f}×)")
    print(f"  {bursts['verdict']}")
    print()

    # ── Summary ──
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    tests_pass = 0
    tests_total = 4

    # Test 1 pass/fail
    if peak_info["peak_in_transition"]:
        print("  ✓ Test 1 (peak in transition zone): PASS")
        tests_pass += 1
    elif peak_info["transition_start"] is not None:
        print(f"  ~ Test 1 (peak in transition zone): PARTIAL — {peak_info['distance_to_transition']} steps away")
    else:
        print("  ? Test 1 (peak in transition zone): INCONCLUSIVE (no clear transition)")

    # Test 3 pass/fail
    if ar1["prediction_holds"]:
        print("  ✓ Test 3 (AR(1) model): PASS")
        tests_pass += 1
    else:
        print("  ✗ Test 3 (AR(1) model): FAIL")

    # Test 4 pass/fail
    if early.get("ratio_early_late") and early["ratio_early_late"] > 1.5:
        print("  ✓ Test 4 (early sensitivity): PASS")
        tests_pass += 1
    else:
        print("  ~ Test 4 (early sensitivity): WEAK or INCONCLUSIVE")

    # Test 5 pass/fail: amplification present AND asymmetry absent (identity not predictive)
    if (bursts["amplification_ratio"] and bursts["amplification_ratio"] > 1.5
            and bursts["asymmetry_ratio"] and bursts["asymmetry_ratio"] < 1.5):
        print("  ✓ Test 5 (magnitude bursts): PASS — magnitude drives RE, platform doesn't")
        tests_pass += 1
    else:
        print("  ~ Test 5 (magnitude bursts): PARTIAL — check amplification/asymmetry above")

    print(f"\n  {tests_pass}/{tests_total} tests passed, 1 correlation analysis (see Test 2)")
    print()

    # ── Dump per-step data to CSV (for table-based analysis) ──
    gn_avg = (gn_cuda + gn_npu) / 2.0
    gn_deriv = np.abs(grad_norm_derivative(gn_avg))
    gn_var = running_variance(gn_avg)

    gn_max = np.max(gn_avg[steps > 5])
    phase_arr = np.array(["phase2_transition"] * len(steps), dtype=object)
    phase_arr[gn_avg > 0.5 * gn_max] = "phase1_high_curvature"
    phase_arr[gn_avg <= 0.1 * gn_max] = "phase3_flat"
    phase_arr[steps <= 5] = "startup"

    csv_path = os.path.join(out, "per_step_data.csv")
    with open(csv_path, "w") as f:
        f.write("step,loss_cuda,loss_npu,re_pct,grad_norm_cuda,grad_norm_npu,grad_norm_avg,"
                "grad_norm_deriv_abs,grad_norm_variance,phase,max_grad_norm,cuda_larger,"
                "burst_id\n")
        gn_max = np.maximum(gn_cuda, gn_npu)
        cuda_larger = (gn_cuda > gn_npu).astype(int)
        burst_id = bursts["burst_id"]
        for i in range(len(steps)):
            f.write(f"{steps[i]:.0f},{loss_cuda[i]:.6f},{loss_npu[i]:.6f},{re[i]:.6f},"
                    f"{gn_cuda[i]:.6f},{gn_npu[i]:.6f},{gn_avg[i]:.6f},"
                    f"{gn_deriv[i]:.6f},{gn_var[i]:.6f},{phase_arr[i]},"
                    f"{gn_max[i]:.6f},{cuda_larger[i]},{burst_id[i]}\n")
    print(f"Saved per-step table: {csv_path}")

    # ── Plots ──
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            make_plots(steps, re, gn_cuda, gn_npu, lr, peak_info, out)
        except ImportError:
            print("matplotlib not available — skipping plots. Install with: pip install matplotlib")
            print("Re-run with --no_plots to suppress this message.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
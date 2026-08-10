#!/usr/bin/env python3
"""
Cross-device / cross-impl verification of the AR(1) divergence story.

Part 1 — CUDA V100 vs A800  (both SDPA), fp32 tiers 0/5/7
Part 2 — A800 SDPA vs A800 FA2,           bf16  tiers 0/5/7
Exp 2  — no-clip fp32 (V100 vs 910b, const LR, 100 steps)
         vs const_lr (V100 vs 910b, clipped) — magnitude-injection vs direction-compounding

All metrics replicate verify_loss_alignment_claims.py / make_slide_figures.py
exactly: RE with baseline=CUDA/reference, Spearman rho(|RE|, max grad norm) on
steps>5, amplification (mean|RE| hi/low grad, 3x median, steps>5), platform
asymmetry over all steps, contiguous |RE|>3x-median bursts.

Usage:
  python scripts/analyze_cross_device_verification.py [--repo_root <DriveVLA-W0>]
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

L = None  # set in main


def load_trainer_state(filepath):
    with open(filepath) as f:
        data = json.load(f)
    records = {"step": [], "loss": [], "grad_norm": [], "learning_rate": []}
    for entry in data.get("log_history", []):
        if "loss" in entry:
            for k in records:
                records[k].append(entry.get(k, 0))
    return {k: np.array(v) for k, v in records.items()}


def relative_error(comp, base):
    """Baseline-relative error (%): (comp - base) / base * 100."""
    return (comp - base) / base * 100


def bursts_of(re, steps):
    """Contiguous runs where |RE| > 3x median|RE| (AR(1) compounding fingerprint)."""
    mask = steps > 5
    re_med = np.median(np.abs(re[mask]))
    if re_med == 0:
        return []
    hot = np.abs(re) > 3 * re_med
    out, start, in_b = [], None, False
    for i in range(len(steps)):
        if hot[i] and not in_b:
            start, in_b = i, True
        elif not hot[i] and in_b:
            out.append((int(steps[start]), int(steps[i - 1]))); in_b = False
    if in_b:
        out.append((int(steps[start]), int(steps[-1])))
    return out


def magnitude_stats(gn_base, gn_comp, re, steps):
    """Spearman rho, amplification ratio, asymmetry ratio — identical formulas
    to make_slide_figures.magnitude_stats / verify_loss_alignment_claims."""
    abs_re = np.abs(re)
    gn_max = np.maximum(gn_base, gn_comp)
    med = np.median(np.concatenate([gn_base, gn_comp]))
    mask = steps > 5
    r, p = stats.spearmanr(abs_re[mask], gn_max[mask])
    hi = mask & (gn_max > 3 * med)
    lo = mask & (gn_max <= 3 * med)
    amp = (np.mean(abs_re[hi]) / np.mean(abs_re[lo])
           if hi.any() and lo.any() else np.nan)
    # asymmetry over ALL steps (matches make_slide_figures NOTE)
    base_hi = gn_base > gn_comp
    comp_hi = gn_comp > gn_base
    mb = np.mean(abs_re[base_hi]) if base_hi.any() else np.nan
    mc = np.mean(abs_re[comp_hi]) if comp_hi.any() else np.nan
    asym = (max(mb, mc) / min(mb, mc)
            if not np.isnan(mb) and not np.isnan(mc) and min(mb, mc) > 0 else np.nan)
    return r, p, amp, asym


def report(label, base, comp, with_grad=True):
    """Compute and print the standard metric block for one pair."""
    re = relative_error(comp["loss"], base["loss"])
    steps = base["step"]
    gap = np.abs(base["loss"] - comp["loss"])
    lines = [f"\n### {label}"]
    lines.append(f"  steps {steps[0]:.0f}..{steps[-1]:.0f} ({len(steps)})")
    lines.append(f"  step-1 forward diff (identical weights): "
                 f"loss {base['loss'][0]:.4f} vs {comp['loss'][0]:.4f} -> RE {re[0]:+.2f}%")
    lines.append(f"  mean|RE|={np.mean(np.abs(re)):.3f}%  "
                 f"max|RE|={np.max(np.abs(re)):.2f}% @ step {steps[np.argmax(np.abs(re))]:.0f}  "
                 f"|RE|<=0.5%: {100*np.mean(np.abs(re)<=0.5):.0f}%  "
                 f"max gap={np.max(gap):.3f} @ step {steps[np.argmax(gap)]:.0f}")
    if with_grad:
        r, p, amp, asym = magnitude_stats(base["grad_norm"], comp["grad_norm"], re, steps)
        bursts = bursts_of(re, steps)
        lines.append(f"  Spearman rho(|RE|, max gn)={r:.3f} (p={p:.1e})  "
                     f"amplification={amp:.2f}x  asymmetry={asym:.2f}x")
        lines.append(f"  bursts (>3x median|RE|): "
                     + ("; ".join(f"{b0}-{b1}" for b0, b1 in bursts[:8]) if bursts else "none"))
    else:
        lines.append("  (no grad_norm logged in this run)")
    print("\n".join(lines))
    return re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=(
        "/Users/liumingchang/Desktop/HwWork-temp/autonomous_driving/codespaces/DriveVLA-W0"))
    args = ap.parse_args()
    global L
    L = lambda *p: os.path.join(args.repo_root, "logs", *p)

    print("=" * 74)
    print("Part 1 — CUDA V100 vs A800 (both SDPA), fp32 tiers 0/5/7")
    print("baseline = A800, comparison = V100 — is the platform-agnostic")
    print("steep-regime amplification story reproduced between two CUDA GPUs?")
    print("=" * 74)
    for tier in ["tier0_warmup200", "tier5_warmup300", "tier7_warmup100"]:
        base = load_trainer_state(L(f"fp32/{tier}/trainer_state_a800.json"))
        comp = load_trainer_state(L(f"fp32/{tier}/trainer_state_v100.json"))
        report(f"fp32 {tier}: A800(SDPA) vs V100(SDPA)", base, comp)

    print("\n" + "=" * 74)
    print("Part 2 — A800 SDPA vs A800 FA2, bf16 tiers 0/5/7")
    print("baseline = A800 SDPA, comparison = A800 FA2 — does the attention")
    print("implementation change the divergence pattern (delta_t source)?")
    print("=" * 74)
    for tier in ["tier0_warmup200", "tier5_warmup300", "tier7_warmup100"]:
        base = load_trainer_state(L(f"bf16/{tier}/trainer_state_a800.json"))
        comp = load_trainer_state(L(f"bf16/{tier}/trainer_state_a800_fa2.json"))
        report(f"bf16 {tier}: A800(SDPA) vs A800(FA2)", base, comp)

    print("\n" + "=" * 74)
    print("Exp 2 — no-clip (V100 vs 910b, const LR 1e-5, 100 steps)")
    print("vs const_lr clipped (V100 vs 910b). Hypothesis A: RE explodes beyond")
    print("12% (magnitude injection). Hypothesis B: same ~12% band (direction compounding).")
    print("=" * 74)
    const = load_trainer_state(L("fp32/const_lr/trainer_state_v100.json"))
    const_npu = load_trainer_state(L("fp32/const_lr/trainer_state_910b.json"))
    report("fp32 const_lr CLIPPED (V100 vs 910b), first 100 steps",
           {k: v[:100] for k, v in const.items()},
           {k: v[:100] for k, v in const_npu.items()})
    noclip = load_trainer_state(L("fp32/tier_neg2_noclip/trainer_state_v100.json"))
    noclip_npu = load_trainer_state(L("fp32/tier_neg2_noclip/trainer_state_910b.json"))
    report("fp32 tier_neg2_noclip NO-CLIP (V100 vs 910b), 100 steps",
           noclip, noclip_npu, with_grad=False)

    # direct side-by-side of the Exp 2 test
    re_c = relative_error(const_npu["loss"][:100], const["loss"][:100])
    re_n = relative_error(noclip_npu["loss"], noclip["loss"])
    print("\nExp 2 verdict input:")
    print(f"  clipped : mean|RE|={np.mean(np.abs(re_c)):.3f}%  max|RE|={np.max(np.abs(re_c)):.2f}% @ step {const['step'][np.argmax(np.abs(re_c))]:.0f}")
    print(f"  no-clip : mean|RE|={np.mean(np.abs(re_n)):.3f}%  max|RE|={np.max(np.abs(re_n)):.2f}% @ step {noclip['step'][np.argmax(np.abs(re_n))]:.0f}")
    ratio = np.max(np.abs(re_n)) / np.max(np.abs(re_c))
    print(f"  max|RE| no-clip / clipped = {ratio:.2f}x  "
          + (">> 1  =>  magnitude injection (Hypothesis A) confirmed"
             if ratio > 1.5 else "~1  =>  direction compounding (Hypothesis B) confirmed"))


if __name__ == "__main__":
    main()

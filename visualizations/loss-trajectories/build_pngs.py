from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOG_DIR = Path("/tmp/loss-trajectories-20261127")
OUT_DIR = Path(__file__).resolve().parent
EMA_ALPHA = 0.99
SEED_COLORS = {"42": "tab:blue", "43": "tab:red", "44": "tab:green"}


def read_steps(name: str, limit: int | None = None) -> list[float]:
    text = (LOG_DIR / name).read_text(errors="ignore")
    values = [float(value) for value in re.findall(r"\{'loss':\s*([0-9.eE+-]+),", text)]
    return values if limit is None else values[:limit]


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    smoothed: list[float] = []
    acc = values[0]
    for value in values:
        acc = alpha * acc + (1.0 - alpha) * value
        smoothed.append(acc)
    return smoothed


def arm(label: str, color: str, log_name: str, limit: int | None = None) -> tuple:
    values = read_steps(log_name, limit)
    return label, color, list(range(1, len(values) + 1)), values, ema(values)


def ema_figure(title: str, rows: list[tuple], out_name: str) -> Path:
    """rows: (row_prefix, arms, zoom_range, reference_label)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(19.5, 5.6 * len(rows)), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=15)

    for (row_prefix, arms, zoom, reference), (ax_full, ax_zoom, ax_diff) in zip(rows, axes):
        for label, color, steps, values, smoothed in arms:
            ax_full.plot(steps, values, color=color, linewidth=0.6, alpha=0.25)
            ax_full.plot(steps, smoothed, color=color, linewidth=1.8, label=label)
            start, end = max(zoom[0] - 1, 0), min(zoom[1], len(steps))
            ax_zoom.plot(steps[start:end], values[start:end], color=color, linewidth=0.6, alpha=0.25)
            ax_zoom.plot(steps[start:end], smoothed[start:end], color=color, linewidth=1.8, label=label)

        ax_full.set_yscale("log")
        ax_full.set_title(f"{row_prefix}total loss, full run (thin = per-step, thick = EMA 0.99)", fontsize=12)
        ax_full.legend(loc="upper right", frameon=False)

        ax_zoom.set_xlim(*zoom)
        ax_zoom.set_title(f"{row_prefix}total loss, steps {zoom[0]:,}-{zoom[1]:,} (zoom)", fontsize=12)
        ax_zoom.legend(loc="lower left", frameon=False)

        ref_smoothed = next(a[4] for a in arms if a[0] == reference)
        for label, color, steps, _values, smoothed in arms:
            if label == reference:
                continue
            ax_diff.plot(
                steps[200:],
                [value - ref for value, ref in zip(smoothed[200:], ref_smoothed[200:])],
                color=color,
                linewidth=1.8,
                label=f"{label} - {reference}",
            )
        ax_diff.axhline(0.0, color="black", linewidth=0.8)
        ax_diff.set_title(f"{row_prefix}EMA 0.99 difference vs {reference} (from step 200)", fontsize=12)
        ax_diff.legend(loc="lower left", frameon=False)

        for ax in (ax_full, ax_zoom, ax_diff):
            ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
            ax.set_xlabel("step")

    fig.tight_layout(rect=(0, 0, 1, 0.96 if len(rows) > 1 else 0.94))
    path = OUT_DIR / out_name
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hif8() -> Path:
    seeds = ("42", "43", "44")
    return ema_figure(
        "DTS HiF8 arms · interval 10 · 168 quantized Linear layers · seeds 42 / 43 / 44",
        [
            (
                "pretraining — ",
                [arm(f"seed {s}", SEED_COLORS[s], f"dts_pt_s{s}.log") for s in seeds],
                (6500, 8000),
                "seed 42",
            ),
            (
                "SFT — ",
                [arm(f"seed {s}", SEED_COLORS[s], f"dts_sft_s{s}.log") for s in seeds],
                (2500, 4000),
                "seed 42",
            ),
        ],
        "hif8-loss-trajectories.png",
    )


def plot_bf16() -> Path:
    seeds = ("42", "43")
    return ema_figure(
        "BF16 loss trajectories · NuPlan pair-sampling pretraining · 8,000 steps · seeds 42 / 43\n"
        "pair-level sampling — comparable only within this figure, not against the scene-level HiF8 arms",
        [
            (
                "",
                [arm(f"seed {s}", SEED_COLORS[s], f"bf16_s{s}.log") for s in seeds],
                (6500, 8000),
                "seed 42",
            ),
        ],
        "bf16-loss-trajectories.png",
    )


def plot_seed42_pretrain() -> Path:
    # bf16_scene_gbs96 = pt_ptlr5e5, the only BF16 arm with the same scene-level
    # sampling as the HiF8 run; pt_aligned_bf16_pair_s42 uses pair-level sampling.
    return ema_figure(
        "Seed 42 pretraining: BF16 baseline (pt_ptlr5e5) vs DTS HiF8 · both seed 42, scene-level sampling, gbs 96, lr 5e-5, 8,000 steps",
        [
            (
                "",
                [
                    arm("BF16 baseline", "tab:blue", "bf16_scene_gbs96.log"),
                    arm("DTS HiF8", "tab:red", "dts_pt_s42.log"),
                ],
                (6500, 8000),
                "BF16 baseline",
            ),
        ],
        "seed-42-pretrain-bf16-vs-hif8.png",
    )


def plot_seed43_pretrain() -> Path:
    return ema_figure(
        "Seed 43 pretraining: BF16 (pt_bf16_s43, 11-19 tree) vs DTS HiF8 · both seed 43, scene-level sampling, gbs 96, lr 5e-5, 8,000 steps",
        [
            (
                "",
                [
                    arm("BF16", "tab:blue", "bf16_scene_s43.log"),
                    arm("DTS HiF8", "tab:red", "dts_pt_s43.log"),
                ],
                (6500, 8000),
                "BF16",
            ),
        ],
        "seed-43-pretrain-bf16-vs-hif8.png",
    )


def plot_sft_seeds() -> Path:
    seeds = ("42", "43", "44")
    return ema_figure(
        "Single-stage SFT from pt_ptlr5e5, 4000 steps, DTS-168 i10, zero2, lr 1e-5 — seeds 42 / 43 / 44",
        [
            (
                "",
                [arm(f"seed {s}", SEED_COLORS[s], f"dts_sft_s{s}.log") for s in seeds],
                (2500, 4000),
                "seed 42",
            ),
        ],
        "sft-dts168-i10-seeds-42-43-44.png",
    )


def main() -> None:
    for path in (plot_hif8(), plot_bf16(), plot_seed42_pretrain(), plot_seed43_pretrain(), plot_sft_seeds()):
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

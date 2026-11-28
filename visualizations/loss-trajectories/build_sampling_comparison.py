from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOG_DIR = Path(
    os.environ.get("LOSS_LOG_DIR", "/tmp/loss-trajectories-20261128")
)
OUT_DIR = Path(
    os.environ.get("LOSS_OUT_DIR", str(Path(__file__).resolve().parent))
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMA_ALPHA = 0.99
SEED_COLORS = {"42": "tab:blue", "43": "tab:red", "44": "tab:green"}


def read_losses(name: str) -> list[float]:
    text = (LOG_DIR / name).read_text(errors="ignore")
    values = [
        float(value)
        for value in re.findall(r"\{'loss':\s*([0-9.eE+-]+),", text)
    ]
    if len(values) != 8000:
        raise ValueError(f"{name}: expected 8000 loss points, found {len(values)}")
    return values


def read_validation_losses(name: str) -> list[float]:
    text = (LOG_DIR / name).read_text(errors="ignore")
    values = [
        float(value)
        for value in re.findall(r"'eval_loss':\s*([0-9.eE+-]+)", text)
    ]
    if len(values) != 16:
        raise ValueError(
            f"{name}: expected 16 validation points, found {len(values)}"
        )
    return values


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    smoothed: list[float] = []
    acc = values[0]
    for value in values:
        acc = alpha * acc + (1.0 - alpha) * value
        smoothed.append(acc)
    return smoothed


def load_arm(label: str, color: str, log_name: str) -> tuple:
    values = read_losses(log_name)
    steps = list(range(1, len(values) + 1))
    return label, color, steps, values, ema(values)


def plot_disjoint_with_validation() -> Path:
    arms = [
        load_arm(f"seed {seed}", SEED_COLORS[seed], f"disjoint_s{seed}.log")
        for seed in ("42", "43", "44")
    ]
    validation = [
        (
            f"seed {seed}",
            SEED_COLORS[seed],
            list(range(500, 8001, 500)),
            read_validation_losses(f"disjoint_s{seed}.log"),
        )
        for seed in ("42", "43", "44")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        (
            "BF16 NuPlan loss trajectories · disjoint 2VA matching · "
            "8,000 steps · seeds 42 / 43 / 44\n"
            "Training loss versus 4,096-window holdout validation loss"
        ),
        fontsize=15,
    )
    ax_train, ax_validation, ax_zoom, ax_compare = axes.flat

    for label, color, steps, values, smoothed in arms:
        ax_train.plot(steps, values, color=color, linewidth=0.6, alpha=0.25)
        ax_train.plot(steps, smoothed, color=color, linewidth=1.8, label=label)
        ax_zoom.plot(
            steps[6499:8000],
            values[6499:8000],
            color=color,
            linewidth=0.6,
            alpha=0.25,
        )
        ax_zoom.plot(
            steps[6499:8000],
            smoothed[6499:8000],
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax_compare.plot(
            steps[499:],
            smoothed[499:],
            color=color,
            linewidth=1.8,
            label=f"{label} train EMA",
        )

    for label, color, steps, values in validation:
        ax_validation.plot(
            steps,
            values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4,
            label=label,
        )
        ax_compare.plot(
            steps,
            values,
            color=color,
            linewidth=1.5,
            linestyle="--",
            marker="o",
            markersize=3.5,
            label=f"{label} validation",
        )

    ax_train.set_yscale("log")
    ax_train.set_title(
        "total training loss, full run "
        "(thin = per-step, thick = EMA 0.99)",
        fontsize=12,
    )
    ax_train.legend(loc="upper right", frameon=False)

    ax_validation.set_title("holdout validation loss", fontsize=12)
    ax_validation.legend(loc="upper right", frameon=False)

    ax_zoom.set_xlim(6500, 8000)
    ax_zoom.set_title("training loss, steps 6,500-8,000 (zoom)", fontsize=12)
    ax_zoom.legend(loc="lower left", frameon=False)

    ax_compare.set_title(
        "training EMA 0.99 (solid) vs validation loss (dashed), from step 500",
        fontsize=12,
    )
    ax_compare.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=9,
    )

    for ax in axes.flat:
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT_DIR / "disjoint-2va-loss-trajectories.png"
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_family(title: str, prefix: str, out_name: str) -> Path:
    arms = [
        load_arm(f"seed {seed}", SEED_COLORS[seed], f"{prefix}_s{seed}.log")
        for seed in ("42", "43", "44")
    ]

    fig, (ax_full, ax_zoom, ax_diff) = plt.subplots(
        1, 3, figsize=(19.5, 5.6)
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=15)

    for label, color, steps, values, smoothed in arms:
        ax_full.plot(steps, values, color=color, linewidth=0.6, alpha=0.25)
        ax_full.plot(steps, smoothed, color=color, linewidth=1.8, label=label)
        ax_zoom.plot(steps[6499:8000], values[6499:8000], color=color, linewidth=0.6, alpha=0.25)
        ax_zoom.plot(steps[6499:8000], smoothed[6499:8000], color=color, linewidth=1.8, label=label)

    ax_full.set_yscale("log")
    ax_full.set_title(
        "total loss, full run (thin = per-step, thick = EMA 0.99)",
        fontsize=12,
    )
    ax_full.legend(loc="upper right", frameon=False)

    ax_zoom.set_xlim(6500, 8000)
    ax_zoom.set_title("total loss, steps 6,500-8,000 (zoom)", fontsize=12)
    ax_zoom.legend(loc="lower left", frameon=False)

    reference = arms[0][4]
    for label, color, steps, _values, smoothed in arms[1:]:
        ax_diff.plot(
            steps[200:],
            [value - ref for value, ref in zip(smoothed[200:], reference[200:])],
            color=color,
            linewidth=1.8,
            label=f"{label} - seed 42",
        )
    ax_diff.axhline(0.0, color="black", linewidth=0.8)
    ax_diff.set_title("EMA 0.99 difference vs seed 42 (from step 200)", fontsize=12)
    ax_diff.legend(loc="lower left", frameon=False)

    for ax in (ax_full, ax_zoom, ax_diff):
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
        ax.set_xlabel("step")
    ax_full.set_ylabel("total loss")
    ax_zoom.set_ylabel("total loss")
    ax_diff.set_ylabel("EMA loss difference")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT_DIR / out_name
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    paths = [
        plot_disjoint_with_validation(),
        plot_family(
            (
                "BF16 NuPlan loss trajectories · pair sampling · "
                "8,000 steps · seeds 42 / 43 / 44\n"
                "Legacy anchor-pair sampler with fixed 768k pair set and model-seed ordering"
            ),
            "pair",
            "pair-sampling-loss-trajectories.png",
        ),
    ]
    for path in paths:
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

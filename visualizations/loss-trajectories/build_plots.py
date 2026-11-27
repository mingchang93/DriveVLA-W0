from __future__ import annotations

import json
import re
from pathlib import Path


LOG_DIR = Path("/tmp/loss-trajectories-20261127")
OUT_DIR = Path(__file__).resolve().parent
CHART_BASE_URL = "https://persistent.oaistatic.com/plugins/visualize/charts/1.8.1"
BIN_SIZE = 20


def read_losses(name: str) -> list[float]:
    text = (LOG_DIR / name).read_text(errors="ignore")
    return [float(value) for value in re.findall(r"\{'loss':\s*([0-9.eE+-]+),", text)]


def binned_rows(
    losses: list[float],
    arm: str,
    limit: int | None = None,
) -> list[list[object]]:
    values = losses if limit is None else losses[:limit]
    rows: list[list[object]] = []
    for start in range(0, len(values), BIN_SIZE):
        chunk = values[start : start + BIN_SIZE]
        step = start + len(chunk)
        rows.append([step, round(sum(chunk) / len(chunk), 6), arm])
    return rows


def fragment(
    root_id: str,
    heading: str,
    note: str,
    config: dict,
) -> str:
    payload = json.dumps(config, separators=(",", ":"), ensure_ascii=True)
    return f"""<section id="{root_id}" style="display:grid;gap:calc(var(--font-size-base)*0.75)">
  <div style="display:grid;gap:calc(var(--font-size-base)*0.25)">
    <h1>{heading}</h1>
    <p class="text-small text-muted">{note}</p>
  </div>
  <div data-viz-container></div>
  <script type="application/json" data-viz-chart>{payload}</script>
</section>
<script type="module">
  import {{ mountChart }} from "{CHART_BASE_URL}/element.js";
  const root = document.getElementById("{root_id}");
  const config = JSON.parse(root.querySelector("[data-viz-chart]").textContent);
  mountChart(root.querySelector("[data-viz-container]"), config);
</script>
"""


def line_chart(
    chart_id: str,
    title: str,
    rows: list[list[object]],
    colors: dict[str, str],
    description: str,
) -> dict:
    return {
        "id": chart_id,
        "title": title,
        "description": description,
        "height": 320,
        "columns": ["step", "loss", "arm"],
        "rows": rows,
        "spec": {
            "type": "line",
            "x": "step",
            "y": "loss",
            "series": "arm",
            "xLabel": "Training step",
            "yLabel": f"Total loss ({BIN_SIZE}-step mean)",
            "showLegend": True,
            "startAtZero": False,
            "valueDecimals": 3,
            "colors": colors,
        },
    }


def main() -> None:
    seeds = ("42", "43", "44")
    colors = {
        "seed 42": "var(--chart-1)",
        "seed 43": "var(--chart-2)",
        "seed 44": "var(--chart-3)",
    }

    hif8_pretrain = []
    hif8_sft = []
    for seed in seeds:
        hif8_pretrain.extend(binned_rows(read_losses(f"dts_pt_s{seed}.log"), f"seed {seed}"))
        hif8_sft.extend(binned_rows(read_losses(f"dts_sft_s{seed}.log"), f"seed {seed}"))

    hif8_config = {
        "charts": [
            line_chart(
                "hif8-pretrain-loss",
                "Pretraining total loss",
                hif8_pretrain,
                colors,
                "DTS HiF8 pretraining loss for seeds 42, 43, and 44; each point is a 20-step mean.",
            ),
            line_chart(
                "hif8-sft-loss",
                "SFT total loss",
                hif8_sft,
                colors,
                "DTS HiF8 SFT loss for seeds 42, 43, and 44; each point is a 20-step mean.",
            ),
        ]
    }
    hif8_html = fragment(
        "hif8-loss-trajectories",
        "HiF8 loss trajectories",
        "DTS HiF8, interval 10, 168 quantized layers. Lines show 20-step means of the total objective.",
        hif8_config,
    )
    (OUT_DIR / "hif8-loss-trajectories.html").write_text(hif8_html)

    bf16_rows = []
    for seed in ("42", "43"):
        bf16_rows.extend(
            binned_rows(read_losses(f"bf16_s{seed}.log"), f"seed {seed}", limit=5000)
        )
    bf16_config = line_chart(
        "bf16-pretrain-loss",
        "Pretraining total loss, steps 1–5,000",
        bf16_rows,
        {key: colors[key] for key in ("seed 42", "seed 43")},
        "BF16 pair-sampling pretraining loss for seeds 42 and 43; each point is a 20-step mean.",
    )
    bf16_html = fragment(
        "bf16-loss-trajectories",
        "BF16 loss trajectories",
        "First 5,000 steps, shown as 20-step means. Seed 44 has not started and has no losses yet.",
        bf16_config,
    )
    (OUT_DIR / "bf16-loss-trajectories-first-5000.html").write_text(bf16_html)

    print(f"HiF8 rows: pretrain={len(hif8_pretrain)}, sft={len(hif8_sft)}")
    print(f"BF16 rows: {len(bf16_rows)}")
    for path in sorted(OUT_DIR.glob("*.html")):
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

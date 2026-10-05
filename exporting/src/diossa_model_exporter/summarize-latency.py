#!/usr/bin/env python
"""Summarize an on-device latency CSV into an RST list-table.

The CSV is written by scripts/08-measure-latency.sh. Each row is one image:
accelerator time (``network took``), dequantize time, and ONNX Runtime time.
End-to-end time is the sum of those three. Throughput is ``1000 / mean ms``.

Usage:
    python -m diossa_model_exporter.summarize-latency \
        --csv output/latency/latency.csv \
        --output output/latency/latency.rst \
        --device 192.168.20.6
"""

import argparse
import pathlib
from typing import Optional, Sequence

import pandas as pd

MODEL_ORDER = ("mobilepose", "fcos")
DISPLAY_NAMES = {
    "mobilepose": "MobilePose 224 × 224",
    "fcos": "FCOS 384 × 288",
}
SUMMARY_ROWS = (
    ("Images", "n", "{:.0f}"),
    ("Loops per image", "loops", "{:.0f}"),
    ("Accelerator mean (ms)", "network_mean", "{:.2f}"),
    ("Accelerator std (ms)", "network_std", "{:.2f}"),
    ("Accelerator median (ms)", "network_median", "{:.2f}"),
    ("Accelerator min (ms)", "network_min", "{:.2f}"),
    ("Accelerator max (ms)", "network_max", "{:.2f}"),
    ("Dequantize mean (ms)", "dequantize_mean", "{:.2f}"),
    ("Dequantize std (ms)", "dequantize_std", "{:.2f}"),
    ("ONNX Runtime mean (ms)", "onnxruntime_mean", "{:.2f}"),
    ("ONNX Runtime std (ms)", "onnxruntime_std", "{:.2f}"),
    ("CPU post-processing mean (ms)", "cpu_mean", "{:.2f}"),
    ("CPU post-processing std (ms)", "cpu_std", "{:.2f}"),
    ("End-to-end mean (ms)", "e2e_mean", "{:.2f}"),
    ("End-to-end std (ms)", "e2e_std", "{:.2f}"),
    ("End-to-end median (ms)", "e2e_median", "{:.2f}"),
    ("Accelerator throughput (FPS)", "network_fps", "{:.2f}"),
    ("End-to-end throughput (FPS)", "e2e_fps", "{:.2f}"),
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize run-model latency CSV rows into an RST list-table.",
    )
    parser.add_argument(
        "--csv", type=pathlib.Path, required=True,
        help="Latency CSV from scripts/08-measure-latency.sh",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, required=True,
        help="RST list-table to write",
    )
    parser.add_argument(
        "--device", default=None,
        help="Board address included in the table caption",
    )
    return parser.parse_args()


def load_measurements(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load per-image timings and add CPU and end-to-end columns."""
    measurements = pd.read_csv(csv_path)
    if measurements.empty:
        raise SystemExit(f"No rows in {csv_path}")
    measurements["cpu_ms"] = measurements["dequantize_ms"] + measurements["onnxruntime_ms"]
    measurements["e2e_ms"] = measurements["network_ms"] + measurements["cpu_ms"]
    return measurements


def model_order(names: Sequence[str]) -> list[str]:
    """Keep MobilePose then FCOS, then any other model in CSV order."""
    present = list(dict.fromkeys(names))
    ordered = [name for name in MODEL_ORDER if name in present]
    ordered += [name for name in present if name not in ordered]
    return ordered


def summarize(measurements: pd.DataFrame) -> pd.DataFrame:
    """Aggregate latency per model. Standard deviation is the sample std."""
    grouped = measurements.groupby("model", sort=False)
    summary = pd.DataFrame({
        "n": grouped.size(),
        "network_mean": grouped["network_ms"].mean(),
        "network_std": grouped["network_ms"].std(ddof=1),
        "network_median": grouped["network_ms"].median(),
        "network_min": grouped["network_ms"].min(),
        "network_max": grouped["network_ms"].max(),
        "dequantize_mean": grouped["dequantize_ms"].mean(),
        "dequantize_std": grouped["dequantize_ms"].std(ddof=1),
        "onnxruntime_mean": grouped["onnxruntime_ms"].mean(),
        "onnxruntime_std": grouped["onnxruntime_ms"].std(ddof=1),
        "cpu_mean": grouped["cpu_ms"].mean(),
        "cpu_std": grouped["cpu_ms"].std(ddof=1),
        "e2e_mean": grouped["e2e_ms"].mean(),
        "e2e_std": grouped["e2e_ms"].std(ddof=1),
        "e2e_median": grouped["e2e_ms"].median(),
        "loops": grouped["loops"].median(),
    }).reindex(model_order(measurements["model"]))
    summary["network_fps"] = 1000.0 / summary["network_mean"]
    summary["e2e_fps"] = 1000.0 / summary["e2e_mean"]
    return summary


def format_value(value: float, spec: str) -> str:
    """Format one table cell. A single-image sample has no std."""
    if pd.isna(value):
        return "—"
    return spec.format(value)


def render_rst(summary: pd.DataFrame, device: Optional[str]) -> str:
    """Render the summary as a Sphinx list-table."""
    n_images = int(summary["n"].max())
    where = f" on {device}" if device else ""
    title = f"PolarFire VectorBlox latency{where} ({n_images} images per network)"
    headers = ["Metric"] + [DISPLAY_NAMES.get(name, name) for name in summary.index]
    lines = [
        f".. list-table:: {title}",
        "   :header-rows: 1",
        "   :widths: " + " ".join(["44"] + ["28"] * (len(headers) - 1)),
        "",
        f"   * - {headers[0]}",
    ]
    lines.extend(f"     - {header}" for header in headers[1:])
    for label, column, spec in SUMMARY_ROWS:
        lines.append(f"   * - {label}")
        lines.extend(
            f"     - {format_value(summary.loc[name, column], spec)}"
            for name in summary.index
        )
    lines.append("")
    return "\n".join(lines)


def print_tables(measurements: pd.DataFrame, summary: pd.DataFrame, rst: str) -> None:
    """Print the per-image rows, the summary, and the RST table."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda value: f"{value:.4f}")
    print(measurements.to_string(index=False))
    print()
    printable = summary.T.rename(columns=lambda name: DISPLAY_NAMES.get(name, name))
    print(printable.to_string())
    print()
    print(rst, end="")


def main() -> None:
    """Read a latency CSV and write an RST list-table."""
    args = parse_args()
    measurements = load_measurements(args.csv)
    summary = summarize(measurements)
    rst = render_rst(summary, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rst, encoding="utf-8")
    print_tables(measurements, summary, rst)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

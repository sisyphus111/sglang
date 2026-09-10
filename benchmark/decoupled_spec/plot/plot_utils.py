"""Shared loading and styling helpers for benchmark result plots."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ("#202124", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

UPPER_IQR_OUTLIER_POLICY = (
    "per target/DP series, exclude value > Q3 + 3 * IQR using inclusive "
    "quartiles when at least 8 finite windows are present"
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path: str | Path) -> list[dict[str, str]]:
    # Keep the loader safe for large fixed-schema JSON-array cells.
    csv.field_size_limit(sys.maxsize)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def upper_iqr_outlier_threshold(
    values: list[Any], *, minimum_count: int = 8
) -> float | None:
    """Return the deterministic presentation-only upper-outlier threshold."""
    finite_values = [
        number for value in values if (number := finite_float(value)) is not None
    ]
    if len(finite_values) < minimum_count:
        return None
    q1, _, q3 = statistics.quantiles(
        finite_values, n=4, method="inclusive"
    )
    return q3 + 3 * (q3 - q1)


def style_axis(
    axis: plt.Axes,
    title: str,
    ylabel: str,
    *,
    xlabel: str = "request index",
    grid_axis: str = "y",
) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(
    figure: plt.Figure,
    output_dir: str | Path,
    stem: str,
    source: str,
) -> tuple[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figure.text(
        0.99,
        0.01,
        f"source: {source}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    png_path = output_path / f"{stem}.png"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return (png_path,)

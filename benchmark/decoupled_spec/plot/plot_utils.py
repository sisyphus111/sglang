"""Shared loading and styling helpers for benchmark result plots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ("#202124", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path: str | Path) -> list[dict[str, str]]:
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


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
) -> tuple[Path, Path]:
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
    svg_path = output_path / f"{stem}.svg"
    png_path = output_path / f"{stem}.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return svg_path, png_path


def build_manifest(
    *,
    kind: str,
    run_dir: Path,
    sources: list[Path],
    outputs: list[Path],
) -> dict[str, Any]:
    """Describe one independently reproducible derived artifact."""
    return {
        "schema_version": 1,
        "kind": kind,
        "created_at": time.time(),
        "run_dir": str(run_dir),
        "sources": [
            {"path": str(path.relative_to(run_dir)), "sha256": sha256(path)}
            for path in sources
            if path.is_file()
        ],
        "outputs": [str(path.relative_to(run_dir)) for path in outputs],
    }

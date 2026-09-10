"""Plot fixed request-level speculative metrics for one run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_utils import (
    COLORS,
    finite_float,
    load_csv,
    save_figure,
    style_axis,
)
from run_io import require_run_dir


def render_speculative(run_dir: str | Path) -> dict[str, Any]:
    run_path = require_run_dir(run_dir)
    metrics_path = run_path / "client" / "requests.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    records = load_csv(metrics_path)
    if not records:
        raise ValueError("client/requests.csv contains no request rows")

    metrics = (
        ("valid_draft_len", "Valid draft length", "drafts/verify", COLORS[4]),
        ("acc_len", "Accept length", "tokens/verify", COLORS[2]),
    )
    available = any(
        finite_float(record.get(field)) is not None
        for record in records
        for field, _, _, _ in metrics
    )
    if not available:
        raise ValueError("client/requests.csv lacks speculative metrics")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True)
    for axis, (field, title, ylabel, color) in zip(axes, metrics):
        points = [
            (int(record["batch_row_index"]), value)
            for record in records
            if (value := finite_float(record.get(field))) is not None
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.0,
            )
        style_axis(axis, title, ylabel, xlabel="batch row index")
        axis.set_ylim(bottom=0)
    outputs = list(
        save_figure(
            figure,
            run_path / "plots",
            "request_speculative",
            "client/requests.csv",
        )
    )
    return {"outputs": [str(path) for path in outputs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(render_speculative(args.run_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

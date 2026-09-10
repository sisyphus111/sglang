"""Plot request-level end-to-end latency for one run."""

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


def render_latency(run_dir: str | Path) -> dict[str, Any]:
    run_path = require_run_dir(run_dir)
    metrics_path = run_path / "client" / "requests.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    records = load_csv(metrics_path)
    if not records:
        raise ValueError("client/requests.csv contains no request rows")

    figure, axis = plt.subplots(1, 1, figsize=(6.5, 3.8))
    points = [
        (int(record["batch_row_index"]), value)
        for record in records
        if (value := finite_float(record.get("e2e_latency_s"))) is not None
    ]
    if not points:
        raise ValueError("client/requests.csv lacks E2E latency values")
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=COLORS[2],
        marker="o",
        markersize=3.5,
        linewidth=1.0,
    )
    style_axis(axis, "E2E latency", "seconds", xlabel="batch row index")
    axis.set_ylim(bottom=0)

    output_dir = run_path / "plots"
    outputs = list(
        save_figure(
            figure,
            output_dir,
            "request_latency",
            "client/requests.csv",
        )
    )
    return {"outputs": [str(path) for path in outputs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(render_latency(args.run_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

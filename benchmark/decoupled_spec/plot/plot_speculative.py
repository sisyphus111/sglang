"""Plot request-level speculative acceptance metrics for one run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.artifacts import write_json
from plot_utils import (
    COLORS,
    build_manifest,
    finite_float,
    load_csv,
    save_figure,
    style_axis,
)


def render_speculative(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    metrics_path = run_path / "client" / "request_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    records = load_csv(metrics_path)
    if not records:
        raise ValueError("client/request_metrics.csv contains no request rows")

    metrics = (
        ("spec_accept_rate", "Accept rate", "fraction", COLORS[3]),
        ("spec_accept_length", "Accept length", "tokens/verify", COLORS[2]),
        (
            "spec_proposed_draft_length",
            "Proposed draft length",
            "drafts/verify",
            COLORS[4],
        ),
    )
    available = any(
        finite_float(record.get(field)) is not None
        for record in records
        for field, _, _, _ in metrics
    )
    outputs: list[Path] = []
    if available:
        figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharex=True)
        for axis, (field, title, ylabel, color) in zip(axes, metrics):
            points = [
                (index, value)
                for index, record in enumerate(records)
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
            style_axis(axis, title, ylabel)
            if field == "spec_accept_rate":
                axis.set_ylim(0, 1)
            else:
                axis.set_ylim(bottom=0)
        outputs = list(
            save_figure(
                figure,
                run_path / "plots",
                "request_speculative",
                "client/request_metrics.csv",
            )
        )

    manifest = build_manifest(
        kind="decoupled_spec_request_speculative_plot",
        run_dir=run_path,
        sources=[metrics_path],
        outputs=outputs,
    )
    manifest_path = run_path / "plots" / "request_speculative_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(render_speculative(args.run_dir)["manifest_path"])


if __name__ == "__main__":
    main()

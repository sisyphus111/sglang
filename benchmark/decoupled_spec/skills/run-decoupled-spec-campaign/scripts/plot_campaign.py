#!/usr/bin/env python3
"""Plot cross-case throughput and overlap effects from a campaign summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODE_COLORS = {"nonoverlap": "#202124", "overlap": "#0072B2"}
_MODE_LABELS = {"nonoverlap": "Non-overlap", "overlap": "Overlap"}
_BS_COLORS = {
    4: "#CC79A7",
    8: "#202124",
    16: "#0072B2",
    32: "#D55E00",
    64: "#009E73",
}
_FALLBACK_BS_COLORS = ("#56B4E9", "#E69F00", "#F0E442", "#009E73")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _style_axis(axis: plt.Axes, title: str, ylabel: str, xlabel: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    source_label: str,
    *,
    layout_top: float = 0.94,
) -> list[Path]:
    figure.text(
        0.99,
        0.01,
        f"source: {source_label}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )
    figure.tight_layout(rect=(0, 0.05, 1, layout_top))
    outputs = [output_dir / f"{stem}.svg", output_dir / f"{stem}.png"]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return outputs


def _index_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[bool, str, int, int], dict]:
    indexed = {}
    for row in rows:
        key = (
            bool(row["ignore_eos"]),
            str(row["mode"]),
            int(row["batch_size"]),
            int(row["output_len"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate campaign row: {key}")
        indexed[key] = row
    return indexed


def write_campaign_plots(
    summary_path: str | Path, output_dir: str | Path | None = None
) -> list[Path]:
    summary_file = Path(summary_path).expanduser().resolve()
    summary = _read_json(summary_file)
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("campaign summary contains no rows")
    indexed = _index_rows(rows)
    ignore_eos_values = sorted({bool(row["ignore_eos"]) for row in rows})
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    output_lengths = sorted({int(row["output_len"]) for row in rows})
    batch_colors = {
        batch_size: _BS_COLORS.get(
            batch_size, _FALLBACK_BS_COLORS[index % len(_FALLBACK_BS_COLORS)]
        )
        for index, batch_size in enumerate(batch_sizes)
    }
    expected_keys = {
        (ignore_eos, mode, batch_size, output_len)
        for ignore_eos in ignore_eos_values
        for mode in _MODE_COLORS
        for batch_size in batch_sizes
        for output_len in output_lengths
    }
    if set(indexed) != expected_keys:
        raise ValueError("campaign summary does not contain a complete paired grid")

    plot_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else summary_file.parent / "plots"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    source_label = f"{summary_file.name} sha256={_sha256(summary_file)[:12]}"
    outputs = []
    metric_specs = (
        ("Throughput change", "Overlap vs non-overlap (%)", "throughput"),
        ("Accept-rate change", "Percentage points", "accept"),
        ("Draft-occupancy change", "Percentage points", "occupancy"),
        ("Output/verify change", "Overlap vs non-overlap (%)", "accept_length"),
    )

    for ignore_eos in ignore_eos_values:
        eos_label = "ignore_eos_true" if ignore_eos else "ignore_eos_false"
        legend_title = f"ignore_eos={str(ignore_eos).lower()}"
        column_count = min(2, len(output_lengths))
        row_count = (len(output_lengths) + column_count - 1) // column_count
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(
                5.25 * column_count if column_count > 1 else 6.5,
                3.5 * row_count + (0.8 if row_count == 1 else 0.0),
            ),
            sharex=True,
            squeeze=False,
        )
        plot_axes = axes.ravel()[: len(output_lengths)]
        for axis, output_len in zip(plot_axes, output_lengths, strict=True):
            for mode in ("nonoverlap", "overlap"):
                values = [
                    float(
                        indexed[(ignore_eos, mode, batch_size, output_len)][
                            "output_tokens_per_s"
                        ]
                    )
                    for batch_size in batch_sizes
                ]
                axis.plot(
                    batch_sizes,
                    values,
                    color=_MODE_COLORS[mode],
                    marker="o" if mode == "nonoverlap" else "s",
                    linewidth=2,
                    markersize=5,
                    label=_MODE_LABELS[mode],
                )
            _style_axis(
                axis,
                f"Max output {output_len // 1024}K",
                "Output tokens/s",
                "Batch size",
            )
            axis.set_xticks(batch_sizes)
            axis.set_ylim(bottom=0.0)
        for axis in axes.ravel()[len(output_lengths) :]:
            axis.set_visible(False)
        for index, axis in enumerate(plot_axes):
            if index < column_count * (row_count - 1):
                axis.set_xlabel("")
        handles, labels = plot_axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            title=legend_title,
        )
        outputs.extend(
            _save_figure(
                figure,
                plot_dir,
                f"throughput_by_batch_{eos_label}",
                source_label,
                layout_top=0.78 if row_count == 1 else 0.88,
            )
        )

        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
        for axis, (title, ylabel, metric) in zip(axes.flat, metric_specs, strict=True):
            for batch_size in batch_sizes:
                values = []
                for output_len in output_lengths:
                    nonoverlap = indexed[
                        (ignore_eos, "nonoverlap", batch_size, output_len)
                    ]
                    overlap = indexed[(ignore_eos, "overlap", batch_size, output_len)]
                    if metric == "throughput":
                        value = 100.0 * (
                            float(overlap["output_tokens_per_s"])
                            / float(nonoverlap["output_tokens_per_s"])
                            - 1.0
                        )
                    elif metric == "accept":
                        value = 100.0 * (
                            float(overlap["spec_accept_rate"])
                            - float(nonoverlap["spec_accept_rate"])
                        )
                    elif metric == "occupancy":
                        value = 100.0 * (
                            float(overlap["spec_draft_occupancy_rate"])
                            - float(nonoverlap["spec_draft_occupancy_rate"])
                        )
                    else:
                        value = 100.0 * (
                            float(overlap["spec_accept_length"])
                            / float(nonoverlap["spec_accept_length"])
                            - 1.0
                        )
                    values.append(value)
                axis.plot(
                    output_lengths,
                    values,
                    color=batch_colors[batch_size],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label=f"BS {batch_size}",
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_xticks(
                output_lengths, [f"{value // 1024}K" for value in output_lengths]
            )
            _style_axis(axis, title, ylabel, "Max response length")
        for axis in axes[0]:
            axis.set_xlabel("")
        for axis in axes[1]:
            axis.set_xlabel("Max response length")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(batch_sizes),
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            title=legend_title,
        )
        outputs.extend(
            _save_figure(
                figure,
                plot_dir,
                f"paired_overlap_effects_{eos_label}",
                source_label,
            )
        )

    manifest_path = plot_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "decoupled_spec_campaign_plots",
        "created_at": time.time(),
        "sources": [{"path": str(summary_file), "sha256": _sha256(summary_file)}],
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in outputs],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [*outputs, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "outputs": [
                    str(path)
                    for path in write_campaign_plots(args.summary, args.output_dir)
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Render a compact time-series overview from observability samples."""

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
from common.artifacts import write_json
from plot_utils import COLORS, build_manifest, save_figure, style_axis

ROLE_COLORS = {"verifier": COLORS[0], "drafter": COLORS[1]}
ROLE_COLOR_INDICES = {"verifier": (0, 2, 4), "drafter": (1, 3, 5)}
TARGET_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _load_samples(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spec_metric(item: dict[str, Any], name: str) -> Any:
    speculative = item.get("speculative")
    return speculative.get(name) if isinstance(speculative, dict) else None


def _extract_decode_metric_rows(
    samples: list[dict[str, Any]], origin: float
) -> list[dict[str, Any]]:
    """Deduplicate bounded engine windows by HTTP target, DP rank, and ID."""
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for sample in samples:
        target_id = str(sample.get("target_id", sample["role"]))
        for load in sample.get("payload", {}).get("loads", []):
            dp_rank = int(load.get("dp_rank", 0))
            for window in load.get("decode_metrics_windows") or []:
                if not isinstance(window, dict):
                    raise ValueError("decode_metrics_windows entries must be mappings")
                end_time = float(window["end_time"])
                if end_time < origin:
                    continue
                row = {
                    "target_id": target_id,
                    "role": sample["role"],
                    "rank": int(sample.get("rank", 0)),
                    "dp_rank": dp_rank,
                    "time_s": end_time - origin,
                    **window,
                }
                key = (target_id, dp_rank, int(window["window_id"]))
                previous = rows.get(key)
                if previous is not None and previous != row:
                    raise ValueError(f"conflicting decode metrics window: {key}")
                rows[key] = row
    return sorted(
        rows.values(),
        key=lambda row: (row["time_s"], row["target_id"], row["dp_rank"]),
    )


def render_observability(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    samples_path = run_path / "observability" / "samples.jsonl"
    all_samples = _load_samples(samples_path)
    samples = [
        sample
        for sample in all_samples
        if sample.get("error") is None and sample.get("payload", {}).get("loads")
    ]
    if not samples:
        raise ValueError("no successful /v1/loads samples to plot")
    origin = min(sample["collected_wall_time"] for sample in samples)
    series: dict[str, list[dict[str, Any]]] = {}
    target_roles: dict[str, str] = {}
    target_ranks: dict[str, int] = {}
    for sample in all_samples:
        target_id = str(sample.get("target_id", sample["role"]))
        role = str(sample["role"])
        rank = int(sample.get("rank", 0))
        previous_role = target_roles.setdefault(target_id, role)
        if previous_role != role:
            raise ValueError(f"conflicting roles for target {target_id}")
        previous_rank = target_ranks.setdefault(target_id, rank)
        if previous_rank != rank:
            raise ValueError(f"conflicting ranks for target {target_id}")
        payload = sample.get("payload")
        loads = payload.get("loads") if isinstance(payload, dict) else None
        load = loads[0] if isinstance(loads, list) and loads else {}
        series.setdefault(target_id, []).append(
            {
                "time_s": sample["collected_wall_time"] - origin,
                "role": role,
                "rank": rank,
                **load,
            }
        )
    target_colors = {
        target_id: COLORS[
            ROLE_COLOR_INDICES[target_roles[target_id]][
                target_ranks[target_id]
                % len(ROLE_COLOR_INDICES[target_roles[target_id]])
            ]
        ]
        for target_id in series
    }
    target_markers = {
        target_id: TARGET_MARKERS[target_ranks[target_id] % len(TARGET_MARKERS)]
        for target_id in series
    }

    formal_path = run_path / "client" / "formal_window.json"
    formal_window = (
        json.loads(formal_path.read_text(encoding="utf-8"))
        if formal_path.exists()
        else None
    )
    decode_metric_rows = _extract_decode_metric_rows(samples, origin)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharex=True)
    ax_requests, ax_throughput, ax_usage, ax_spec = axes.flat
    for target_id, target_values in sorted(series.items()):
        values = sorted(target_values, key=lambda item: item["time_s"])
        role = target_roles[target_id]
        times = [item["time_s"] for item in values]
        color = target_colors[target_id]
        marker = target_markers[target_id]
        ax_requests.plot(
            times,
            [item.get("num_running_reqs") for item in values],
            color=color,
            marker=marker,
            markersize=2.5,
            label=f"{target_id} running",
        )
        ax_requests.plot(
            times,
            [item.get("num_waiting_reqs") for item in values],
            color=color,
            linestyle="--",
            marker=marker,
            markersize=2.5,
            alpha=0.7,
            label=f"{target_id} waiting",
        )
        ax_throughput.plot(
            times,
            [item.get("gen_throughput") for item in values],
            color=color,
            marker=marker,
            markersize=2.5,
            label=target_id,
        )
        ax_usage.plot(
            times,
            [item.get("token_usage") for item in values],
            color=color,
            marker=marker,
            markersize=2.5,
            label=target_id,
        )
        if role == "verifier":
            ax_spec.plot(
                times,
                [_spec_metric(item, "accept_length") for item in values],
                color=color,
                marker=marker,
                markersize=2.5,
                label=f"{target_id} accept length",
            )
            ax_spec.plot(
                times,
                [_spec_metric(item, "accept_rate") for item in values],
                color=color,
                linestyle="--",
                marker=marker,
                markersize=2.5,
                label=f"{target_id} accept rate",
            )
            ax_spec.plot(
                times,
                [_spec_metric(item, "proposed_draft_length") for item in values],
                color=color,
                linestyle=":",
                marker=marker,
                markersize=2.5,
                label=f"{target_id} proposed draft length",
            )
            ax_spec.plot(
                times,
                [_spec_metric(item, "draft_occupancy_rate") for item in values],
                color=color,
                linestyle="-.",
                marker=marker,
                markersize=2.5,
                label=f"{target_id} draft occupancy",
            )

    for axis, title, ylabel in (
        (ax_requests, "Running batch size and queue", "requests"),
        (ax_throughput, "Generation throughput", "tokens/s"),
        (ax_usage, "KV token usage", "fraction"),
        (ax_spec, "Speculative decoding", "value"),
    ):
        style_axis(
            axis,
            title,
            ylabel,
            xlabel="time since first sample (s)",
            grid_axis="both",
        )
        axis.legend(frameon=False, fontsize=8)
        if formal_window and formal_window.get("finished_wall_time"):
            axis.axvspan(
                formal_window["started_wall_time"] - origin,
                formal_window["finished_wall_time"] - origin,
                color="#999999",
                alpha=0.10,
                linewidth=0,
            )
    ax_requests.set_ylim(bottom=0)
    output_dir = run_path / "observability" / "plots"
    outputs = list(
        save_figure(
            fig,
            output_dir,
            "overview",
            "observability/samples.jsonl",
        )
    )
    if decode_metric_rows:
        decode_fig, decode_axes = plt.subplots(5, 1, figsize=(10.5, 11.8), sharex=True)
        ax_iter, ax_batch, ax_context, ax_valid_draft, ax_accept = decode_axes
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for row in decode_metric_rows:
            grouped.setdefault(
                (row["target_id"], row["role"], row["dp_rank"]), []
            ).append(row)

        for (target_id, role, dp_rank), values in sorted(grouped.items()):
            values.sort(key=lambda item: item["time_s"])
            label = target_id if dp_rank == 0 else f"{target_id} dp{dp_rank}"
            color = target_colors.get(target_id, ROLE_COLORS.get(role, COLORS[2]))
            marker = target_markers.get(target_id, "o")
            ax_iter.plot(
                [item["time_s"] for item in values],
                [item["iter_latency_ms"] for item in values],
                color=color,
                linewidth=1.0,
                marker=marker,
                markersize=2.5,
                alpha=0.8,
                label=label,
            )
            ax_batch.plot(
                [item["time_s"] for item in values],
                [item["mean_batch_size"] for item in values],
                color=color,
                linewidth=1.0,
                marker=marker,
                markersize=2.5,
                alpha=0.8,
                label=label,
            )
            ax_context.plot(
                [item["time_s"] for item in values],
                [item["mean_context_length"] for item in values],
                color=color,
                linewidth=1.0,
                marker=marker,
                markersize=2.5,
                alpha=0.8,
                label=label,
            )
            if role != "verifier":
                continue
            spec_values = [
                item
                for item in values
                if item.get("proposed_draft_length") is not None
                and item.get("accept_length") is not None
            ]
            ax_valid_draft.plot(
                [item["time_s"] for item in spec_values],
                [item["proposed_draft_length"] for item in spec_values],
                color=color,
                linewidth=1.0,
                marker=marker,
                markersize=2.5,
                alpha=0.8,
                label=label,
            )
            ax_accept.plot(
                [item["time_s"] for item in spec_values],
                [item["accept_length"] for item in spec_values],
                color=color,
                linestyle="--",
                linewidth=1.0,
                marker=marker,
                markersize=2.5,
                alpha=0.8,
                label=label,
            )

        has_spec_metrics = any(
            row["role"] == "verifier"
            and row.get("proposed_draft_length") is not None
            and row.get("accept_length") is not None
            for row in decode_metric_rows
        )
        if has_spec_metrics:
            ax_accept.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
        for axis, title, ylabel in (
            (ax_iter, "Iteration latency", "iteration latency (ms)"),
            (ax_batch, "Mean batch size", "requests / iteration"),
            (ax_context, "Mean context length", "tokens / request"),
            (
                ax_valid_draft,
                "Valid draft length",
                "drafts / verify row",
            ),
            (ax_accept, "Accept length", "tokens / verify row"),
        ):
            style_axis(
                axis,
                title,
                ylabel,
                xlabel="time since first sample (s)",
                grid_axis="both",
            )
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(frameon=False, fontsize=8)
            if formal_window and formal_window.get("finished_wall_time"):
                axis.axvspan(
                    formal_window["started_wall_time"] - origin,
                    formal_window["finished_wall_time"] - origin,
                    color="#999999",
                    alpha=0.10,
                    linewidth=0,
                )
        ax_iter.set_ylim(bottom=0)
        ax_batch.set_ylim(bottom=0)
        ax_context.set_ylim(bottom=0)
        ax_valid_draft.set_ylim(bottom=0)
        ax_accept.set_ylim(bottom=1 if has_spec_metrics else 0)
        outputs.extend(
            save_figure(
                decode_fig,
                output_dir,
                "decode_metrics",
                "observability/samples.jsonl",
            )
        )
    sources = [samples_path]
    if formal_path.is_file():
        sources.append(formal_path)
    manifest = build_manifest(
        kind="decoupled_spec_observability_plot",
        run_dir=run_path,
        sources=sources,
        outputs=outputs,
    )
    manifest["decode_metrics_window_ct"] = len(decode_metric_rows)
    manifest["decode_metrics_window_ct_by_role"] = {
        role: sum(row["role"] == role for row in decode_metric_rows)
        for role in sorted({row["role"] for row in decode_metric_rows})
    }
    manifest["decode_metrics_window_ct_by_target"] = {
        target_id: sum(row["target_id"] == target_id for row in decode_metric_rows)
        for target_id in sorted({row["target_id"] for row in decode_metric_rows})
    }
    manifest["target_ct"] = len(series)
    manifest["target_ct_by_role"] = {
        role: sum(target_role == role for target_role in target_roles.values())
        for role in sorted(set(target_roles.values()))
    }
    manifest["sample_ct_by_target"] = {
        target_id: len(values) for target_id, values in sorted(series.items())
    }
    manifest["missing_sample_plot_policy"] = "gap"
    manifest["zero_based_axes"] = ["running_batch_size", "iteration_latency"]
    manifest_path = output_dir / "plot_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(render_observability(args.run_dir)["manifest_path"])


if __name__ == "__main__":
    main()

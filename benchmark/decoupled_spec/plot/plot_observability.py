"""Render a compact time-series overview from observability samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_io import require_run_dir
from plot_utils import (
    COLORS,
    UPPER_IQR_OUTLIER_POLICY,
    save_figure,
    style_axis,
    upper_iqr_outlier_threshold,
)

ROLE_COLORS = {"verifier": COLORS[0], "drafter": COLORS[1]}
ROLE_COLOR_INDICES = {
    "target": (0, 2, 4),
    "verifier": (0, 2, 4),
    "drafter": (1, 3, 5),
}
TARGET_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

_TRANSPORT_PANELS = (
    ("draft_send_queue_latency_us", "Draft send queue latency"),
    (
        "draft_transport_one_way_latency_us",
        "Drafter to verifier transport latency (record-time calibrated)",
    ),
    (
        "draft_receive_to_gpu_publish_enqueue_latency_us",
        "Receive to GPU publish enqueue latency",
    ),
    ("draft_gpu_publish_completion_latency_us", "GPU publish completion latency"),
)

_TRANSPORT_STAGE_LABELS = {
    "draft_transport_one_way_latency_us": "send to receive",
}


def _load_samples(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _spec_metric(item: dict[str, Any], name: str) -> Any:
    speculative = item.get("speculative")
    return speculative.get(name) if isinstance(speculative, dict) else None


def _integer_histogram_mean(histogram: Any) -> float | None:
    if not isinstance(histogram, dict):
        return None
    counts = histogram.get("counts")
    offset = histogram.get("offset")
    if type(offset) is not int or not isinstance(counts, list):
        return None
    if histogram.get("underflow_count", 0) or histogram.get("overflow_count", 0):
        return None
    if any(type(count) is not int or count < 0 for count in counts):
        return None
    total = sum(counts)
    if total <= 0:
        return None
    return sum((offset + index) * count for index, count in enumerate(counts)) / total


def _latency_histogram_quantile(histogram: Any, quantile: float) -> float | None:
    if not isinstance(histogram, dict):
        return None
    count = histogram.get("count")
    bounds = histogram.get("bucket_upper_bounds_us")
    bucket_counts = histogram.get("bucket_counts")
    if (
        type(count) is not int
        or count <= 0
        or not isinstance(bounds, list)
        or not isinstance(bucket_counts, list)
        or len(bucket_counts) != len(bounds) + 1
        or any(type(item) is not int or item < 0 for item in bucket_counts)
        or count != sum(bucket_counts)
    ):
        return None
    rank = max(1, math.ceil(count * quantile))
    cumulative = 0
    for index, bucket_count in enumerate(bucket_counts):
        cumulative += bucket_count
        if cumulative >= rank:
            # The last bucket is an unbounded overflow bin. Returning null is
            # more honest than plotting its lower bound as a quantile value.
            return float(bounds[index]) if index < len(bounds) else None
    return None


def _latency_histogram_totals(histogram: Any) -> tuple[int, float] | None:
    """Return the exact event count and latency sum for one valid histogram."""
    if not isinstance(histogram, dict):
        return None
    count = histogram.get("count")
    sum_us = histogram.get("sum_us")
    bounds = histogram.get("bucket_upper_bounds_us")
    bucket_counts = histogram.get("bucket_counts")
    if (
        type(count) is not int
        or count <= 0
        or isinstance(sum_us, bool)
        or not isinstance(sum_us, (int, float))
        or not math.isfinite(float(sum_us))
        or float(sum_us) < 0
        or not isinstance(bounds, list)
        or not isinstance(bucket_counts, list)
        or len(bucket_counts) != len(bounds) + 1
        or any(type(item) is not int or item < 0 for item in bucket_counts)
        or count != sum(bucket_counts)
    ):
        return None
    return count, float(sum_us)


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


def _extract_transport_mean_points(
    samples: list[dict[str, Any]],
    started_wall_time: float,
    finished_wall_time: float | None,
) -> list[dict[str, Any]]:
    """Aggregate windows first observed at each poll using their exact sums."""
    dp_ranks_by_target: dict[str, set[int]] = {}
    ordered_samples = []
    for source_index, sample in enumerate(samples):
        target_id = str(sample.get("target_id", sample["role"]))
        payload = sample.get("payload")
        loads = payload.get("loads") if isinstance(payload, dict) else None
        if isinstance(loads, list):
            for load in loads:
                if isinstance(load, dict):
                    dp_ranks_by_target.setdefault(target_id, set()).add(
                        int(load.get("dp_rank", 0))
                    )
        ordered_samples.append(
            (
                int(sample.get("sample_id", source_index)),
                float(sample["collected_wall_time"]),
                source_index,
                sample,
            )
        )

    seen_windows: dict[tuple[str, int, int], dict[str, Any]] = {}
    points: list[dict[str, Any]] = []
    for sample_id, collected_wall_time, _, sample in sorted(ordered_samples):
        target_id = str(sample.get("target_id", sample["role"]))
        payload = sample.get("payload")
        loads = payload.get("loads") if isinstance(payload, dict) else None
        loads_by_dp_rank: dict[int, dict[str, Any]] = {}
        if sample.get("error") is None and isinstance(loads, list):
            for load in loads:
                if not isinstance(load, dict):
                    raise ValueError("loads entries must be mappings")
                dp_rank = int(load.get("dp_rank", 0))
                if dp_rank in loads_by_dp_rank:
                    raise ValueError(
                        f"duplicate loads entry for target {target_id} dp{dp_rank}"
                    )
                loads_by_dp_rank[dp_rank] = load

        for dp_rank in sorted(dp_ranks_by_target.get(target_id, ())):
            totals = {field: [0, 0.0] for field, _ in _TRANSPORT_PANELS}
            has_new_transport_window = False
            load = loads_by_dp_rank.get(dp_rank)
            windows = load.get("decode_metrics_windows") if load else None
            for window in windows or []:
                if not isinstance(window, dict):
                    raise ValueError("decode_metrics_windows entries must be mappings")
                key = (target_id, dp_rank, int(window["window_id"]))
                if key in seen_windows:
                    if seen_windows[key] != window:
                        raise ValueError(f"conflicting decode metrics window: {key}")
                    continue
                seen_windows[key] = window
                end_time = float(window["end_time"])
                if end_time < started_wall_time or (
                    finished_wall_time is not None
                    and end_time > finished_wall_time
                ):
                    continue
                decoupled_spec = window.get("decoupled_spec")
                transport = (
                    decoupled_spec.get("transport")
                    if isinstance(decoupled_spec, dict)
                    else None
                )
                if not isinstance(transport, dict):
                    continue
                has_new_transport_window = True
                for field, _ in _TRANSPORT_PANELS:
                    histogram_totals = _latency_histogram_totals(
                        transport.get(field)
                    )
                    if histogram_totals is None:
                        continue
                    count, sum_us = histogram_totals
                    totals[field][0] += count
                    totals[field][1] += sum_us

            inside_sampling_window = collected_wall_time >= started_wall_time and (
                finished_wall_time is None
                or collected_wall_time <= finished_wall_time
            )
            # A trailing poll can be the first observation of the last engine
            # windows completed before the formal request finished.
            if not inside_sampling_window and not has_new_transport_window:
                continue
            points.append(
                {
                    "target_id": target_id,
                    "role": str(sample["role"]),
                    "rank": int(sample.get("rank", 0)),
                    "dp_rank": dp_rank,
                    "sample_id": sample_id,
                    "collected_wall_time": collected_wall_time,
                    "time_s": collected_wall_time - started_wall_time,
                    "transport_means": {
                        field: sum_us / count if count else None
                        for field, (count, sum_us) in totals.items()
                    },
                    "transport_counts": {
                        field: count for field, (count, _) in totals.items()
                    },
                }
            )
    return points


def render_observability(run_dir: str | Path) -> dict[str, Any]:
    run_path = require_run_dir(run_dir)
    samples_path = run_path / "observer" / "samples.jsonl"
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

    formal_path = run_path / "observer" / "bench_timeline.json"
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
        if formal_window and formal_window.get("client_finished_wall_time"):
            axis.axvspan(
                formal_window["client_started_wall_time"] - origin,
                formal_window["client_finished_wall_time"] - origin,
                color="#999999",
                alpha=0.10,
                linewidth=0,
            )
    ax_requests.set_ylim(bottom=0)
    output_dir = run_path / "plots"
    outputs = list(
        save_figure(
            fig,
            output_dir,
            "overview",
            "observer/samples.jsonl + observer/bench_timeline.json",
        )
    )
    iteration_latency_outliers: list[dict[str, Any]] = []
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
            iter_latency_threshold = upper_iqr_outlier_threshold(
                [item["iter_latency_ms"] for item in values]
            )
            iter_latency_values = []
            for item in values:
                latency_ms = float(item["iter_latency_ms"])
                if (
                    iter_latency_threshold is not None
                    and latency_ms > iter_latency_threshold
                ):
                    iteration_latency_outliers.append(
                        {
                            "target_id": target_id,
                            "role": role,
                            "dp_rank": dp_rank,
                            "window_id": int(item["window_id"]),
                            "time_s": float(item["time_s"]),
                            "end_time": float(item["end_time"]),
                            "iter_latency_ms": latency_ms,
                            "threshold_ms": iter_latency_threshold,
                        }
                    )
                    iter_latency_values.append(None)
                else:
                    iter_latency_values.append(latency_ms)
            ax_iter.plot(
                [item["time_s"] for item in values],
                iter_latency_values,
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
        iteration_title = "Iteration latency"
        if iteration_latency_outliers:
            iteration_title += (
                f" ({len(iteration_latency_outliers)} outlier windows excluded)"
            )
        for axis, title, ylabel in (
            (ax_iter, iteration_title, "iteration latency (ms)"),
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
            if formal_window and formal_window.get("client_finished_wall_time"):
                axis.axvspan(
                    formal_window["client_started_wall_time"] - origin,
                    formal_window["client_finished_wall_time"] - origin,
                    color="#999999",
                    alpha=0.10,
                    linewidth=0,
                )
        ax_iter.set_ylim(bottom=0)
        ax_batch.set_ylim(bottom=0)
        ax_context.set_ylim(bottom=0)
        ax_valid_draft.set_ylim(bottom=0)
        ax_accept.set_ylim(bottom=0)
        outputs.extend(
            save_figure(
                decode_fig,
                output_dir,
                "decode_metrics",
                "observer/samples.jsonl + observer/bench_timeline.json",
            )
        )

    decoupled_rows = [
        row for row in decode_metric_rows if isinstance(row.get("decoupled_spec"), dict)
    ]
    selected_tail_target = None
    selected_tail_dp_rank = None
    selector_series: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    decoupled_time_origin = origin
    decoupled_time_finish = None
    decoupled_time_label = "Server runtime (s)"
    if formal_window and isinstance(
        formal_window.get("client_started_wall_time"), (int, float)
    ):
        decoupled_time_origin = float(formal_window["client_started_wall_time"])
        decoupled_time_label = "Batch runtime (s)"
        formal_finish = formal_window.get("client_finished_wall_time")
        decoupled_time_finish = (
            float(formal_finish) if isinstance(formal_finish, (int, float)) else None
        )
        decoupled_rows = [
            row
            for row in decoupled_rows
            if float(row["end_time"]) >= decoupled_time_origin
            and (
                not isinstance(formal_finish, (int, float))
                or float(row["end_time"]) <= float(formal_finish)
            )
        ]
    transport_mean_points = _extract_transport_mean_points(
        all_samples,
        decoupled_time_origin,
        decoupled_time_finish,
    )

    if decoupled_rows:
        config_path = run_path / "config.json"
        run_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        client_config = run_config.get("client", {})
        selected_tail_target = (
            client_config.get("server", {}).get("target_id")
            if isinstance(client_config.get("server"), dict)
            else None
        )
        tail_rows_by_series: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in decoupled_rows:
            tail_select = row["decoupled_spec"].get("tail_select")
            if row["role"] == "verifier" and isinstance(tail_select, dict):
                tail_rows_by_series.setdefault(
                    (row["target_id"], row["dp_rank"]), []
                ).append(row)
        selector_series = [
            {
                "target_id": target_id,
                "dp_rank": dp_rank,
                "num_select_rows": sum(
                    int(row["decoupled_spec"]["tail_select"].get("num_select_rows", 0))
                    for row in rows
                ),
            }
            for (target_id, dp_rank), rows in sorted(tail_rows_by_series.items())
        ]
        tail_targets = {target_id for target_id, _ in tail_rows_by_series}
        if selected_tail_target not in tail_targets:
            selected_tail_target = max(
                tail_targets,
                key=lambda target_id: sum(
                    int(row["decoupled_spec"]["tail_select"].get("num_select_rows", 0))
                    for (series_target, _), rows in tail_rows_by_series.items()
                    if series_target == target_id
                    for row in rows
                ),
                default=None,
            )
        selected_series = max(
            (
                (target_id, dp_rank)
                for target_id, dp_rank in tail_rows_by_series
                if target_id == selected_tail_target
            ),
            key=lambda series: (
                sum(
                    int(row["decoupled_spec"]["tail_select"].get("num_select_rows", 0))
                    for row in tail_rows_by_series[series]
                ),
                -series[1],
            ),
            default=None,
        )
        if selected_series is not None:
            selected_tail_dp_rank = selected_series[1]
        selected_tail_rows = sorted(
            tail_rows_by_series.get(selected_series, []),
            key=lambda row: row["end_time"],
        )

        decoupled_fig, decoupled_axes = plt.subplots(
            4, 2, figsize=(14.0, 11.0), sharex=True
        )
        (
            ax_reasons,
            ax_tail_lengths,
            ax_delta,
            ax_freshness,
            ax_send_queue,
            ax_one_way,
            ax_receive_publish,
            ax_publish_completion,
        ) = decoupled_axes.flat
        ax_pending_recovery = ax_freshness.twinx()

        reason_names = sorted(
            {
                reason
                for row in selected_tail_rows
                for reason in row["decoupled_spec"]["tail_select"].get(
                    "reason_counts", {}
                )
            }
        )
        valid_reason_rows = [
            row
            for row in selected_tail_rows
            if int(row["decoupled_spec"]["tail_select"].get("num_select_rows", 0)) > 0
        ]
        if valid_reason_rows and reason_names:
            reason_times = [
                float(row["end_time"]) - decoupled_time_origin
                for row in valid_reason_rows
            ]
            reason_values = []
            for reason in reason_names:
                reason_values.append(
                    [
                        100.0
                        * int(
                            row["decoupled_spec"]["tail_select"]
                            .get("reason_counts", {})
                            .get(reason, 0)
                        )
                        / int(row["decoupled_spec"]["tail_select"]["num_select_rows"])
                        for row in valid_reason_rows
                    ]
                )
            reason_colors = [
                plt.get_cmap("tab20")(index % 20) for index in range(len(reason_names))
            ]
            ax_reasons.stackplot(
                reason_times,
                *reason_values,
                labels=reason_names,
                colors=reason_colors,
                alpha=0.85,
            )
            ax_reasons.legend(
                frameon=False,
                fontsize=6,
                ncol=min(3, len(reason_names)),
                loc="upper left",
            )

        tail_time = [
            float(row["end_time"]) - decoupled_time_origin for row in selected_tail_rows
        ]
        for field, label, color in (
            (
                "raw_draft_tail_length_histogram",
                "raw draft tail",
                COLORS[0],
            ),
            (
                "consumable_draft_tail_length_histogram",
                "consumable draft tail",
                COLORS[1],
            ),
            (
                "selected_draft_length_histogram",
                "selected draft length",
                COLORS[2],
            ),
        ):
            ax_tail_lengths.plot(
                tail_time,
                [
                    _integer_histogram_mean(
                        row["decoupled_spec"]["tail_select"].get(field)
                    )
                    for row in selected_tail_rows
                ],
                color=color,
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                label=label,
            )
        ax_delta.plot(
            tail_time,
            [
                _integer_histogram_mean(
                    row["decoupled_spec"]["tail_select"].get("logical_delta_histogram")
                )
                for row in selected_tail_rows
            ],
            color=COLORS[0],
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label="logical delta",
        )
        for field, label, color in (
            ("num_publish_seq_advance", "advance", COLORS[1]),
            ("num_publish_seq_same", "same", COLORS[2]),
            ("num_publish_seq_initial", "initial", COLORS[4]),
            (None, "unavailable", "#777777"),
        ):
            freshness_values = []
            for row in selected_tail_rows:
                tail_select = row["decoupled_spec"]["tail_select"]
                rows = int(tail_select.get("num_select_rows", 0))
                classified = sum(
                    int(tail_select.get(item, 0))
                    for item in (
                        "num_publish_seq_advance",
                        "num_publish_seq_same",
                        "num_publish_seq_initial",
                    )
                )
                freshness_values.append(
                    100.0
                    * (
                        rows - classified
                        if field is None
                        else int(tail_select.get(field, 0))
                    )
                    / rows
                    if rows
                    else None
                )
            ax_freshness.plot(
                tail_time,
                freshness_values,
                color=color,
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                label=label,
            )
        ax_pending_recovery.plot(
            tail_time,
            [
                (
                    (
                        int(
                            row["decoupled_spec"]["tail_select"].get(
                                "num_pending_prefix_fast_forwards", 0
                            )
                        )
                        / int(
                            row["decoupled_spec"]["tail_select"].get(
                                "num_select_rows", 0
                            )
                        )
                    )
                    if int(
                        row["decoupled_spec"]["tail_select"].get("num_select_rows", 0)
                    )
                    else None
                )
                for row in selected_tail_rows
            ],
            color=COLORS[5],
            linestyle="--",
            linewidth=1.2,
            marker="x",
            markersize=3.0,
            label="pending fast-forwards / select row",
        )

        transport_axes = {
            "draft_send_queue_latency_us": ax_send_queue,
            "draft_transport_one_way_latency_us": ax_one_way,
            "draft_receive_to_gpu_publish_enqueue_latency_us": ax_receive_publish,
            "draft_gpu_publish_completion_latency_us": ax_publish_completion,
        }
        populated_transport_axes = set()
        transport_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for point in transport_mean_points:
            transport_groups.setdefault(
                (point["target_id"], point["role"], point["dp_rank"]), []
            ).append(point)
        for (target_id, role, dp_rank), values in sorted(transport_groups.items()):
            values.sort(key=lambda point: point["collected_wall_time"])
            label = target_id if dp_rank == 0 else f"{target_id} dp{dp_rank}"
            color = target_colors.get(target_id, ROLE_COLORS.get(role, COLORS[3]))
            times = [point["time_s"] for point in values]
            for field, _ in _TRANSPORT_PANELS:
                mean_values = [
                    point["transport_means"].get(field) for point in values
                ]
                if not any(value is not None for value in mean_values):
                    continue
                axis = transport_axes[field]
                populated_transport_axes.add(axis)
                stage = _TRANSPORT_STAGE_LABELS.get(field)
                series_label = label if stage is None else f"{label} {stage}"
                axis.plot(
                    times,
                    mean_values,
                    color=color,
                    linestyle="-",
                    linewidth=1.2,
                    marker="o",
                    markersize=2.5,
                    label=f"{series_label} mean",
                )

        selector_title = "Selector reason share"
        if selected_tail_target:
            selector_title += f" ({selected_tail_target} dp{selected_tail_dp_rank})"
        for axis, title, ylabel in (
            (ax_reasons, selector_title, "select rows (%)"),
            (ax_tail_lengths, "Draft tail lengths", "tokens / select row"),
            (ax_delta, "Logical delta", "tokens / select row"),
            (
                ax_freshness,
                "Drafter arrival freshness and pending recovery",
                "select rows (%)",
            ),
            (
                ax_send_queue,
                "Draft send queue latency (per observer poll)",
                "exact mean latency (µs)",
            ),
            (
                ax_one_way,
                "Drafter to verifier send-to-receive latency "
                "(per observer poll, record-time calibrated)",
                "exact mean latency (µs)",
            ),
            (
                ax_receive_publish,
                "Receive to GPU publish enqueue latency (per observer poll)",
                "exact mean latency (µs)",
            ),
            (
                ax_publish_completion,
                "GPU publish completion latency (per observer poll)",
                "exact mean latency (µs)",
            ),
        ):
            if axis in transport_axes.values() and axis not in populated_transport_axes:
                axis.set_visible(False)
                continue
            style_axis(
                axis,
                title,
                ylabel,
                xlabel=decoupled_time_label,
                grid_axis="both",
            )
            handles, _ = axis.get_legend_handles_labels()
            if handles and axis not in (ax_reasons, ax_freshness):
                axis.legend(frameon=False, fontsize=7)
        ax_reasons.set_ylim(0, 100)
        ax_tail_lengths.set_ylim(bottom=0)
        ax_freshness.set_ylim(0, 100)
        ax_pending_recovery.set_ylabel("pending fast-forwards / select row")
        ax_pending_recovery.set_ylim(bottom=0)
        ax_pending_recovery.spines["top"].set_visible(False)
        freshness_handles, freshness_labels = ax_freshness.get_legend_handles_labels()
        recovery_handles, recovery_labels = (
            ax_pending_recovery.get_legend_handles_labels()
        )
        ax_freshness.legend(
            freshness_handles + recovery_handles,
            freshness_labels + recovery_labels,
            frameon=False,
            fontsize=7,
        )
        for axis in transport_axes.values():
            axis.set_ylim(bottom=0)
        outputs.extend(
            save_figure(
                decoupled_fig,
                output_dir,
                "decoupled_spec_metrics",
                "observer/samples.jsonl + observer/bench_timeline.json",
            )
        )
        adaptive_rows = [
            row
            for row in selected_tail_rows
            if isinstance(row["decoupled_spec"].get("adaptive_verify"), dict)
        ]
        if adaptive_rows:
            adaptive_fig, adaptive_axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
            ax_steps, ax_rates, ax_scores, ax_residency = adaptive_axes.flat
            adaptive_times = [
                float(row["end_time"]) - decoupled_time_origin for row in adaptive_rows
            ]
            adaptive_values = [
                row["decoupled_spec"]["adaptive_verify"] for row in adaptive_rows
            ]
            ax_steps.step(
                adaptive_times,
                [int(value["active_verify_steps"]) for value in adaptive_values],
                where="post",
                color=COLORS[0],
                linewidth=1.5,
                label="active K",
            )
            max_steps = max(int(value["max_verify_steps"]) for value in adaptive_values)
            ax_steps.axhline(
                max_steps,
                color="#777777",
                linestyle="--",
                linewidth=1.0,
                label="Kmax",
            )
            for position in range(max_steps):
                color = plt.get_cmap("tab10")(position % 10)
                ax_rates.plot(
                    adaptive_times,
                    [
                        (
                            value.get("supply_ema", [])[position]
                            if position < len(value.get("supply_ema", []))
                            else None
                        )
                        for value in adaptive_values
                    ],
                    color=color,
                    linewidth=1.1,
                    label=f"p{position + 1} supply",
                )
                ax_rates.plot(
                    adaptive_times,
                    [
                        (
                            value.get("conditional_accept_ema", [])[position]
                            if position < len(value.get("conditional_accept_ema", []))
                            else None
                        )
                        for value in adaptive_values
                    ],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    label=f"p{position + 1} accept|supply",
                )
            candidate_steps = sorted(
                {
                    int(score["steps"])
                    for value in adaptive_values
                    for score in value.get("candidate_scores", [])
                }
            )
            for candidate in candidate_steps:
                ax_scores.plot(
                    adaptive_times,
                    [
                        next(
                            (
                                score.get("modeled_tps")
                                for score in value.get("candidate_scores", [])
                                if int(score["steps"]) == candidate
                            ),
                            None,
                        )
                        for value in adaptive_values
                    ],
                    linewidth=1.2,
                    marker="o",
                    markersize=2.5,
                    label=f"K={candidate}",
                )
            final_residency = adaptive_values[-1].get("step_residency", [])
            ax_residency.bar(
                list(range(len(final_residency))),
                final_residency,
                color=COLORS[1],
                alpha=0.85,
            )
            for axis, title, ylabel in (
                (ax_steps, "Active verifier steps", "K"),
                (ax_rates, "Draft supply and conditional acceptance EMA", "EMA"),
                (ax_scores, "Modeled throughput by candidate", "tokens/s"),
                (ax_residency, "Verifier-step residency", "decode rounds"),
            ):
                style_axis(
                    axis,
                    title,
                    ylabel,
                    xlabel=(
                        "K"
                        if axis is ax_residency
                        else (
                            "batch time (s)"
                            if decoupled_time_label == "Batch runtime (s)"
                            else "server time (s)"
                        )
                    ),
                    grid_axis="both",
                )
                handles, _ = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(frameon=False, fontsize=7, ncol=2)
            ax_steps.set_ylim(bottom=0)
            ax_rates.set_ylim(0, 1)
            ax_scores.set_ylim(bottom=0)
            ax_residency.set_ylim(bottom=0)
            outputs.extend(
                save_figure(
                    adaptive_fig,
                    output_dir,
                    "adaptive_verify",
                    "observer/samples.jsonl + observer/bench_timeline.json",
                )
            )
    manifest = {"outputs": [str(path) for path in outputs]}
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
    manifest["decoupled_spec_metrics_window_ct"] = len(decoupled_rows)
    manifest["adaptive_verify_metrics_window_ct"] = len(adaptive_rows)
    manifest["decoupled_spec_time_axis"] = decoupled_time_label
    manifest["decoupled_spec_time_origin"] = decoupled_time_origin
    manifest["selector_target_id"] = selected_tail_target
    manifest["selector_dp_rank"] = selected_tail_dp_rank
    manifest["selector_series"] = selector_series
    manifest["transport_zero_count_policy"] = "gap"
    manifest["transport_observer_point_ct"] = len(transport_mean_points)
    manifest["transport_latency_plot_policy"] = (
        "per-observer-sample weighted exact mean over newly observed engine "
        "windows: sum(sum_us) / sum(count)"
    )
    manifest["transport_point_time_policy"] = (
        "observer collected_wall_time relative to the decoupled-spec time origin"
    )
    manifest["cross_node_latency_peer_policy"] = (
        "plot record-time calibrated samples regardless of drain-time peer state"
    )
    manifest["pending_fast_forward_plot_policy"] = (
        "events per select row on an unbounded secondary axis"
    )
    manifest["integer_histogram_mean_policy"] = (
        "exact indexed-bin mean; null when underflow or overflow is nonzero"
    )
    manifest["iteration_latency_outlier_policy"] = UPPER_IQR_OUTLIER_POLICY
    manifest["iteration_latency_outlier_count"] = len(iteration_latency_outliers)
    manifest["iteration_latency_outliers"] = iteration_latency_outliers
    manifest["zero_based_axes"] = ["running_batch_size", "iteration_latency"]
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(render_observability(args.run_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

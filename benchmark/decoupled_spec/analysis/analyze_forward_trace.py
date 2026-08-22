#!/usr/bin/env python3
"""Summarize decoupled-spec forward-stream structure from a Kineto trace.

This is a deliberately small, read-only companion to the repository's general
LLM torch-profiler analyzer.  It answers the decoupled overlap questions that
kernel aggregation alone cannot answer:

* which numeric stream is the verifier forward stream;
* how many device operations execute between consecutive GPU-tail selects;
* CPU annotation and correlated GPU-work duration distributions; and
* whether verifier-side tail landing overlaps the forward stream.

CPU ``record_function`` spans measure host enqueue work.  Correlated GPU values
below are inclusive per scope and are reported as both summed work and unioned
device-active time; neither is silently treated as scheduler ITL.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from trace_window_gate import build_window_gate

SCOPE_NAMES = (
    "scheduler.recv_requests",
    "scheduler.process_input_requests",
    "scheduler.get_next_batch_to_run",
    "scheduler.run_batch",
    "scheduler.process_batch_result",
    "copy_result_to_cpu",
    "sglang.decoupled_spec.gpu_tail_select",
    "sglang.decoupled_spec.tp_broadcast",
    "sglang.decoupled_spec.verify_input_prepare",
    "sglang.decoupled_spec.target_verify",
    "sglang.speculative.target_verify_forward",
    "sglang.speculative.eagle_sample",
    "sglang.speculative.mamba_commit_after_verify",
    "sglang.decoupled_spec.future_publish",
)

SELECT_KERNEL_FRAGMENT = "select_gpu_draft_tail_kernel"
PUBLISH_KERNEL_FRAGMENT = "publish_gpu_draft_tail_kernel"

_NON_DEVICE_CATEGORIES = {
    "cpu_op",
    "python_function",
    "python tracer",
    "user_annotation",
    "gpu_user_annotation",
    "cuda_runtime",
    "cuda_driver",
    "ac2g",
}


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _interval_union_us(events: Sequence[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in events
        if float(event.get("dur", 0.0)) > 0.0
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


def _intersection_union_us(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> float:
    points: list[tuple[float, int, int]] = []
    for side, events in enumerate((left, right)):
        for event in events:
            start = float(event["ts"])
            end = start + float(event["dur"])
            if end <= start:
                continue
            points.append((start, side, 1))
            points.append((end, side, -1))
    points.sort(key=lambda item: (item[0], item[2]))
    active = [0, 0]
    previous: float | None = None
    overlap = 0.0
    for timestamp, side, delta in points:
        if previous is not None and active[0] > 0 and active[1] > 0:
            overlap += timestamp - previous
        active[side] += delta
        previous = timestamp
    return overlap


def _load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"Trace has no traceEvents list: {path}")
    return [event for event in events if isinstance(event, dict)]


def _resolve_trace(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (
            candidate.name.endswith(".trace.json")
            or candidate.name.endswith(".trace.json.gz")
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No Kineto trace found under {path}")
    tp0 = [
        candidate
        for candidate in candidates
        if any(marker in candidate.name for marker in ("TP-0", "TP_0", "tp-0"))
    ]
    chosen = tp0 if tp0 else candidates
    if len(chosen) != 1:
        rendered = "\n".join(f"  {candidate}" for candidate in chosen)
        raise ValueError(
            "Trace directory is ambiguous; pass one exact TP0 trace file:\n" + rendered
        )
    return chosen[0]


def _complete_event(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and _number(event.get("ts")) is not None
        and _number(event.get("dur")) is not None
    )


def _stream_key(event: dict[str, Any]) -> str | None:
    args = event.get("args") or {}
    stream = args.get("stream", args.get("cuda_stream"))
    if stream is None:
        return None
    device = args.get("device", args.get("Device Id", "?"))
    return f"pid={event.get('pid')} device={device} stream={stream}"


def _is_device_event(event: dict[str, Any]) -> bool:
    if not _complete_event(event) or _stream_key(event) is None:
        return False
    category = str(event.get("cat", "")).lower()
    if category in _NON_DEVICE_CATEGORIES:
        return False
    name = str(event.get("name", "")).lower()
    if name.startswith("cuda") and "memcpy" not in name and "memset" not in name:
        return False
    return True


def _is_cuda_runtime_event(event: dict[str, Any]) -> bool:
    if not _complete_event(event):
        return False
    category = str(event.get("cat", "")).lower()
    return category in {"cuda_runtime", "cuda_driver"}


def _correlation(event: dict[str, Any]) -> int | None:
    args = event.get("args") or {}
    for key in ("correlation", "Correlation id", "correlation_id"):
        value = _coerce_int(args.get(key))
        if value is not None:
            return value
    return None


def _external_id(event: dict[str, Any]) -> int | None:
    args = event.get("args") or {}
    for key in ("External id", "External ID", "external_id"):
        value = _coerce_int(args.get(key))
        if value is not None:
            return value
    return None


def _scope_invocations(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    invocations: list[dict[str, Any]] = []
    for event in events:
        if (
            not _complete_event(event)
            or event.get("cat") != "user_annotation"
            or event.get("name") not in SCOPE_NAMES
        ):
            continue
        invocation = {
            "id": len(invocations),
            "name": str(event["name"]),
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "ts": float(event["ts"]),
            "dur": float(event["dur"]),
            "end": float(event["ts"]) + float(event["dur"]),
            "external_id": _external_id(event),
            "gpu_events": [],
        }
        invocations.append(invocation)
    return invocations


def _attach_gpu_events_to_scopes(
    events: Sequence[dict[str, Any]],
    device_events: Sequence[dict[str, Any]],
    invocations: list[dict[str, Any]],
) -> None:
    scopes_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for scope in invocations:
        scopes_by_thread[(scope["pid"], scope["tid"])].append(scope)

    correlation_scopes: dict[int, set[int]] = defaultdict(set)
    external_scopes: dict[int, set[int]] = defaultdict(set)
    for scope in invocations:
        if scope["external_id"] is not None:
            external_scopes[int(scope["external_id"])].add(int(scope["id"]))

    # CUDA runtime events execute on the same CPU thread as their enclosing
    # record_function annotations.  Keep all enclosing scopes so parent scopes
    # have intentionally inclusive GPU accounting.
    for event in events:
        if not _is_cuda_runtime_event(event):
            continue
        timestamp = float(event["ts"])
        enclosing = [
            scope
            for scope in scopes_by_thread.get((event.get("pid"), event.get("tid")), ())
            if scope["ts"] <= timestamp <= scope["end"]
        ]
        if not enclosing:
            continue
        correlation = _correlation(event)
        external = _external_id(event)
        for scope in enclosing:
            scope_id = int(scope["id"])
            if correlation is not None:
                correlation_scopes[correlation].add(scope_id)
            if external is not None:
                external_scopes[external].add(scope_id)

    for event in device_events:
        scope_ids: set[int] = set()
        correlation = _correlation(event)
        external = _external_id(event)
        if correlation is not None:
            scope_ids.update(correlation_scopes.get(correlation, ()))
        if external is not None:
            scope_ids.update(external_scopes.get(external, ()))
        for scope_id in scope_ids:
            invocations[scope_id]["gpu_events"].append(event)


def _scope_summary(
    invocations: Sequence[dict[str, Any]],
    gpu_projections: Sequence[dict[str, Any]],
    raw_events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for invocation in invocations:
        by_name[invocation["name"]].append(invocation)

    result: dict[str, Any] = {}
    for name in SCOPE_NAMES:
        rows = by_name.get(name, [])
        projections = [row for row in gpu_projections if row.get("name") == name]
        projections_by_external: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for projection_index, projection in enumerate(projections):
            external_id = _external_id(projection)
            key = (
                ("external", external_id)
                if external_id is not None
                else (
                    "projection",
                    projection_index,
                )
            )
            projections_by_external[key].append(projection)
        projection_union = [
            _interval_union_us(group) for group in projections_by_external.values()
        ]
        gpu_work = [
            sum(float(event["dur"]) for event in row["gpu_events"]) for row in rows
        ]
        gpu_union = [_interval_union_us(row["gpu_events"]) for row in rows]
        synchronize_us = []
        synchronize_counts = []
        for row in rows:
            sync_events = [
                event
                for event in raw_events
                if _is_cuda_runtime_event(event)
                and event.get("pid") == row["pid"]
                and event.get("tid") == row["tid"]
                and "cudastreamsynchronize" in str(event.get("name", "")).lower()
                and row["ts"] <= float(event["ts"])
                and float(event["ts"]) + float(event["dur"]) <= row["end"]
            ]
            synchronize_us.append(sum(float(event["dur"]) for event in sync_events))
            synchronize_counts.append(len(sync_events))
        result[name] = {
            "cpu_duration_us": _distribution(row["dur"] for row in rows),
            "gpu_work_us_inclusive": _distribution(gpu_work),
            "gpu_active_union_us_inclusive": _distribution(gpu_union),
            "gpu_device_op_count_inclusive": _distribution(
                len(row["gpu_events"]) for row in rows
            ),
            "gpu_projection_duration_us": _distribution(projection_union),
            "cuda_stream_synchronize_us_inclusive": _distribution(synchronize_us),
            "cuda_stream_synchronize_count_inclusive": _distribution(
                synchronize_counts
            ),
        }
    return result


def _choose_sentinel_stream(
    device_events: Sequence[dict[str, Any]], fragment: str
) -> str | None:
    counts: dict[str, int] = defaultdict(int)
    for event in device_events:
        if fragment not in str(event.get("name", "")):
            continue
        stream = _stream_key(event)
        if stream is not None:
            counts[stream] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _stream_summary(
    device_events: Sequence[dict[str, Any]],
    forward_stream: str | None,
    landing_stream: str | None,
) -> dict[str, Any]:
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in device_events:
        stream = _stream_key(event)
        if stream is not None:
            by_stream[stream].append(event)
    result: dict[str, Any] = {}
    for stream, rows in sorted(
        by_stream.items(), key=lambda item: _interval_union_us(item[1]), reverse=True
    ):
        if stream == forward_stream:
            role = "forward"
        elif stream == landing_stream:
            role = "landing"
        elif any(
            "dtoh" in str(row.get("name", "")).lower()
            or "memcpy d2h" in str(row.get("name", "")).lower()
            for row in rows
        ):
            role = "copy_candidate"
        else:
            role = "other"
        result[stream] = {
            "role": role,
            "device_op_count": len(rows),
            "device_busy_union_us": _interval_union_us(rows),
            "device_op_duration_us": _distribution(row["dur"] for row in rows),
        }
    return result


def _round_summary(
    device_events: Sequence[dict[str, Any]],
    runtime_events: Sequence[dict[str, Any]],
    forward_stream: str | None,
    drop_first_rounds: int,
) -> dict[str, Any]:
    if forward_stream is None:
        return {"complete_round_count": 0, "error": "forward stream not identified"}
    forward_events = sorted(
        (event for event in device_events if _stream_key(event) == forward_stream),
        key=lambda event: float(event["ts"]),
    )
    boundaries = [
        event
        for event in forward_events
        if SELECT_KERNEL_FRAGMENT in str(event.get("name", ""))
    ]
    rows = []
    for index in range(max(0, drop_first_rounds), len(boundaries) - 1):
        start = float(boundaries[index]["ts"])
        end = float(boundaries[index + 1]["ts"])
        operations = [event for event in forward_events if start <= event["ts"] < end]
        busy = _interval_union_us(operations)
        graph_launches = sum(
            1
            for event in runtime_events
            if start <= float(event["ts"]) < end
            and "cudagraphlaunch" in str(event.get("name", "")).lower()
        )
        rows.append(
            {
                "start_us": start,
                "cycle_us": end - start,
                "forward_device_busy_us": busy,
                "forward_device_gap_us": max(0.0, end - start - busy),
                "forward_device_op_count": len(operations),
                "cuda_graph_launch_count": graph_launches,
            }
        )
    return {
        "select_boundary_count": len(boundaries),
        "dropped_first_rounds": max(0, drop_first_rounds),
        "complete_round_count": len(rows),
        "cycle_us": _distribution(row["cycle_us"] for row in rows),
        "forward_device_busy_us": _distribution(
            row["forward_device_busy_us"] for row in rows
        ),
        "forward_device_gap_us": _distribution(
            row["forward_device_gap_us"] for row in rows
        ),
        "forward_device_op_count": _distribution(
            row["forward_device_op_count"] for row in rows
        ),
        "cuda_graph_launch_count": _distribution(
            row["cuda_graph_launch_count"] for row in rows
        ),
    }


def analyze_trace(
    path: Path,
    drop_first_rounds: int = 0,
    *,
    expected_bs: int,
    trigger_path: Path,
    verifier_log_path: Path,
) -> dict[str, Any]:
    events = _load_trace(path)
    device_events = [event for event in events if _is_device_event(event)]
    runtime_events = [event for event in events if _is_cuda_runtime_event(event)]
    gpu_projections = [
        event
        for event in events
        if _complete_event(event)
        and event.get("cat") == "gpu_user_annotation"
        and event.get("name") in SCOPE_NAMES
    ]
    invocations = _scope_invocations(events)
    _attach_gpu_events_to_scopes(events, device_events, invocations)

    forward_stream = _choose_sentinel_stream(device_events, SELECT_KERNEL_FRAGMENT)
    landing_stream = _choose_sentinel_stream(device_events, PUBLISH_KERNEL_FRAGMENT)
    forward_events = [
        event for event in device_events if _stream_key(event) == forward_stream
    ]
    landing_events = [
        event for event in device_events if _stream_key(event) == landing_stream
    ]
    landing_overlap_us = _intersection_union_us(forward_events, landing_events)
    landing_busy_us = _interval_union_us(landing_events)
    window_gate = build_window_gate(
        expected_bs=expected_bs,
        trigger_path=trigger_path,
        verifier_log_path=verifier_log_path,
    )

    return {
        "schema_version": 1,
        "trace_path": str(path.resolve()),
        "window_gate": window_gate,
        "activity_gate": {
            "gpu_activity_present": bool(device_events),
            "cuda_runtime_activity_present": bool(runtime_events),
            "gpu_structure_valid": bool(device_events) and forward_stream is not None,
        },
        "event_counts": {
            "all": len(events),
            "device": len(device_events),
            "cuda_runtime": len(runtime_events),
            "record_function_scopes": len(invocations),
        },
        "sentinels": {
            "gpu_tail_select_kernel_count": sum(
                SELECT_KERNEL_FRAGMENT in str(event.get("name", ""))
                for event in device_events
            ),
            "gpu_tail_publish_kernel_count": sum(
                PUBLISH_KERNEL_FRAGMENT in str(event.get("name", ""))
                for event in device_events
            ),
            "nccl_device_op_count": sum(
                "nccl" in str(event.get("name", "")).lower() for event in device_events
            ),
            "cuda_graph_launch_count": sum(
                "cudagraphlaunch" in str(event.get("name", "")).lower()
                for event in runtime_events
            ),
        },
        "stream_roles": {
            "forward": forward_stream,
            "landing": landing_stream,
        },
        "stream_summary": _stream_summary(
            device_events, forward_stream, landing_stream
        ),
        "forward_rounds": _round_summary(
            device_events,
            runtime_events,
            forward_stream,
            drop_first_rounds,
        ),
        "scope_summary": _scope_summary(invocations, gpu_projections, events),
        "cross_stream": {
            "landing_device_busy_us": landing_busy_us,
            "landing_overlapped_by_forward_device_us": landing_overlap_us,
            "landing_overlap_fraction": (
                landing_overlap_us / landing_busy_us if landing_busy_us else None
            ),
        },
        "interpretation_limits": [
            "CPU scope duration is host enqueue time, not GPU completion time.",
            "Scope GPU values are inclusive and must not be added across nested scopes.",
            "The last select has no following select boundary and is excluded from round statistics.",
            "A torch-profiler trace is structural evidence; use no-profiler runs for production throughput.",
        ],
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_distribution(distribution: dict[str, Any]) -> str:
    return (
        f"n={distribution.get('count', 0)} "
        f"mean={_format_value(distribution.get('mean'))} "
        f"p50={_format_value(distribution.get('p50'))} "
        f"p95={_format_value(distribution.get('p95'))} "
        f"p99={_format_value(distribution.get('p99'))}"
    )


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Trace: {report['trace_path']}",
        (
            "Window gate: "
            f"passed={report['window_gate']['passed']} "
            f"expected_bs={report['window_gate']['expected_bs']} "
            "log_bs="
            f"{(report['window_gate']['log_context'] or {}).get('running_bs')}"
        ),
        (
            "Activity gate: "
            f"gpu={report['activity_gate']['gpu_activity_present']} "
            f"cuda_runtime={report['activity_gate']['cuda_runtime_activity_present']} "
            f"gpu_structure_valid={report['activity_gate']['gpu_structure_valid']}"
        ),
        (
            "Events: all={all} device={device} cuda_runtime={cuda_runtime} "
            "scopes={record_function_scopes}"
        ).format(**report["event_counts"]),
        "",
        "Stream structure:",
        f"  forward: {report['stream_roles']['forward']}",
        f"  landing: {report['stream_roles']['landing']}",
    ]
    for stream, row in report["stream_summary"].items():
        lines.append(
            f"  [{row['role']}] {stream}: ops={row['device_op_count']} "
            f"busy_union_us={row['device_busy_union_us']:.3f}"
        )

    rounds = report["forward_rounds"]
    lines.extend(
        [
            "",
            (
                "Forward rounds: boundaries={select_boundary_count} "
                "complete={complete_round_count} dropped={dropped_first_rounds}"
            ).format(
                select_boundary_count=rounds.get("select_boundary_count", 0),
                complete_round_count=rounds.get("complete_round_count", 0),
                dropped_first_rounds=rounds.get("dropped_first_rounds", 0),
            ),
        ]
    )
    for key in (
        "cycle_us",
        "forward_device_busy_us",
        "forward_device_gap_us",
        "forward_device_op_count",
        "cuda_graph_launch_count",
    ):
        if key in rounds:
            lines.append(f"  {key}: {_format_distribution(rounds[key])}")

    lines.extend(["", "record_function scopes (GPU values are inclusive):"])
    for name in SCOPE_NAMES:
        row = report["scope_summary"][name]
        lines.append(f"  {name}")
        lines.append(
            f"    cpu_duration_us: {_format_distribution(row['cpu_duration_us'])}"
        )
        lines.append(
            "    gpu_active_union_us: "
            + _format_distribution(row["gpu_active_union_us_inclusive"])
        )
        lines.append(
            "    gpu_projection_duration_us: "
            + _format_distribution(row["gpu_projection_duration_us"])
        )
        lines.append(
            "    gpu_device_op_count: "
            + _format_distribution(row["gpu_device_op_count_inclusive"])
        )
        if row["cuda_stream_synchronize_count_inclusive"]["max"]:
            lines.append(
                "    cuda_stream_synchronize_us: "
                + _format_distribution(row["cuda_stream_synchronize_us_inclusive"])
            )

    cross = report["cross_stream"]
    lines.extend(
        [
            "",
            "Cross-stream structure:",
            f"  landing_busy_us={_format_value(cross['landing_device_busy_us'])}",
            (
                "  landing_overlap_with_forward_us="
                f"{_format_value(cross['landing_overlapped_by_forward_device_us'])}"
            ),
            (
                "  landing_overlap_fraction="
                f"{_format_value(cross['landing_overlap_fraction'])}"
            ),
            "",
            "Sentinels: "
            + " ".join(f"{key}={value}" for key, value in report["sentinels"].items()),
            "",
            "Limits:",
            *(f"  - {item}" for item in report["interpretation_limits"]),
        ]
    )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    rounds = report["forward_rounds"]
    gate = report["window_gate"]
    activity = report["activity_gate"]
    lines = [
        "# Decoupled verifier forward trace",
        "",
        f"- Trace: `{report['trace_path']}`",
        (
            f"- Window gate: **{'pass' if gate['passed'] else 'fail'}**; "
            f"expected BS `{gate['expected_bs']}`, "
            f"trigger-context BS `{(gate['log_context'] or {}).get('running_bs')}`"
        ),
        (
            "- Activities: "
            f"GPU `{activity['gpu_activity_present']}`, "
            f"CUDA runtime `{activity['cuda_runtime_activity_present']}`, "
            f"GPU structure valid `{activity['gpu_structure_valid']}`"
        ),
        "",
        "## Forward-stream rounds",
        "",
        (
            f"The trace contains `{rounds.get('select_boundary_count', 0)}` GPU-tail "
            f"select boundaries and `{rounds.get('complete_round_count', 0)}` complete "
            "select-to-select intervals."
        ),
        "",
        "| Metric | n | mean | p50 | p95 | p99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in (
        "cycle_us",
        "forward_device_busy_us",
        "forward_device_gap_us",
        "forward_device_op_count",
        "cuda_graph_launch_count",
    ):
        if key not in rounds:
            continue
        row = rounds[key]
        lines.append(
            f"| `{key}` | {row['count']} | {_format_value(row['mean'])} | "
            f"{_format_value(row['p50'])} | {_format_value(row['p95'])} | "
            f"{_format_value(row['p99'])} |"
        )

    lines.extend(
        [
            "",
            "## CPU scopes and correlated GPU activity",
            "",
            "CPU counts include only `cat=user_annotation`; projected GPU annotations are grouped by External id before percentile calculation.",
            "",
            "| Scope | CPU n | CPU p50 us | CPU p95 us | GPU active p50 us | GPU projection p50 us | cudaStreamSynchronize p50 us | GPU ops p50 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in SCOPE_NAMES:
        row = report["scope_summary"][name]
        cpu = row["cpu_duration_us"]
        gpu = row["gpu_active_union_us_inclusive"]
        projection = row["gpu_projection_duration_us"]
        sync = row["cuda_stream_synchronize_us_inclusive"]
        ops = row["gpu_device_op_count_inclusive"]
        lines.append(
            f"| `{name}` | {cpu['count']} | {_format_value(cpu['p50'])} | "
            f"{_format_value(cpu['p95'])} | {_format_value(gpu['p50'])} | "
            f"{_format_value(projection['p50'])} | {_format_value(sync['p50'])} | "
            f"{_format_value(ops['p50'])} |"
        )

    lines.extend(
        [
            "",
            "## Physical stream structure",
            "",
            "| Role | Stream | Visible ops | Busy union us | Op p50 us |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for stream, row in report["stream_summary"].items():
        lines.append(
            f"| {row['role']} | `{stream}` | {row['device_op_count']} | "
            f"{row['device_busy_union_us']:.3f} | "
            f"{_format_value(row['device_op_duration_us']['p50'])} |"
        )
    cross = report["cross_stream"]
    lines.extend(
        [
            "",
            (
                "Landing device activity totals "
                f"`{_format_value(cross['landing_device_busy_us'])} us`; "
                f"`{_format_value(cross['landing_overlapped_by_forward_device_us'])} us` "
                "overlaps forward-stream activity."
            ),
            "",
            "## Evidence limits",
            "",
            *(f"- {item}" for item in report["interpretation_limits"]),
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-bs", required=True, type=int)
    parser.add_argument("--trigger-file", required=True, type=Path)
    parser.add_argument("--verifier-log", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--drop-first-rounds",
        type=int,
        default=0,
        help="Exclude this many initial select-to-select intervals from percentiles.",
    )
    args = parser.parse_args()
    if args.drop_first_rounds < 0 or args.expected_bs <= 0:
        parser.error(
            "--drop-first-rounds must be non-negative and --expected-bs positive"
        )
    return args


def main() -> None:
    args = _parse_args()
    trace_path = _resolve_trace(args.input)
    report = analyze_trace(
        trace_path,
        args.drop_first_rounds,
        expected_bs=args.expected_bs,
        trigger_path=args.trigger_file,
        verifier_log_path=args.verifier_log,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(render_text(report))
    if not report["window_gate"]["passed"]:
        raise SystemExit(2)
    if not report["activity_gate"]["gpu_structure_valid"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read an Nsight Systems report and summarize decoupled verifier GPU rounds.

The target CUDA graph is intentionally treated as one opaque GPU operation when
the report contains ``CUPTI_ACTIVITY_KIND_GRAPH_TRACE`` but not graph-node
kernels.  Counts are therefore "visible device operations", not CUDA-graph
node counts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from analyze_forward_trace import (
    PUBLISH_KERNEL_FRAGMENT,
    SELECT_KERNEL_FRAGMENT,
    _distribution,
    _format_distribution,
    _format_value,
    _intersection_union_us,
    _interval_union_us,
)
from trace_window_gate import build_window_gate


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _require_cupti_tables(connection: sqlite3.Connection) -> None:
    required = {
        "StringIds",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    }
    missing = sorted(name for name in required if not _table_exists(connection, name))
    if missing:
        raise ValueError(
            "Nsys SQLite is missing required CUPTI tables: " + ", ".join(missing)
        )


def _export_report(report: Path, output: Path) -> None:
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--quiet=true",
            f"--output={output}",
            str(report),
        ],
        check=True,
    )


def _kernel_rows(connection: sqlite3.Connection, device: int) -> list[dict[str, Any]]:
    query = """
        SELECT k.start, k.end, k.deviceId, k.streamId, k.correlationId,
               k.graphId, k.gridX, k.gridY, k.gridZ, strings.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS strings ON strings.id = k.demangledName
        WHERE k.deviceId = ?
        ORDER BY k.start
    """
    return [
        dict(row) | {"kind": "kernel"} for row in connection.execute(query, (device,))
    ]


def _memcpy_labels(connection: sqlite3.Connection) -> dict[int, str]:
    if not _table_exists(connection, "ENUM_CUDA_MEMCPY_OPER"):
        return {}
    return {
        int(row["id"]): str(row["label"])
        for row in connection.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
    }


def _memcpy_rows(connection: sqlite3.Connection, device: int) -> list[dict[str, Any]]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    labels = _memcpy_labels(connection)
    rows = []
    for row in connection.execute(
        """
        SELECT start, end, deviceId, streamId, correlationId, bytes, copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE deviceId = ?
        ORDER BY start
        """,
        (device,),
    ):
        rows.append(
            dict(row)
            | {
                "kind": "memcpy",
                "name": f"memcpy {labels.get(int(row['copyKind']), row['copyKind'])}",
            }
        )
    return rows


def _memset_rows(connection: sqlite3.Connection, device: int) -> list[dict[str, Any]]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMSET"):
        return []
    return [
        dict(row) | {"kind": "memset", "name": "memset"}
        for row in connection.execute(
            """
            SELECT start, end, deviceId, streamId, correlationId, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            WHERE deviceId = ?
            ORDER BY start
            """,
            (device,),
        )
    ]


def _graph_rows(connection: sqlite3.Connection, device: int) -> list[dict[str, Any]]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        return []
    return [
        dict(row) | {"kind": "cuda_graph", "name": "CUDA graph replay"}
        for row in connection.execute(
            """
            SELECT start, end, deviceId, streamId, correlationId, graphId, graphExecId
            FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
            WHERE deviceId = ?
            ORDER BY start
            """,
            (device,),
        )
    ]


def _runtime_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    query = """
        SELECT runtime.start, runtime.end, runtime.correlationId,
               strings.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        ORDER BY runtime.start
    """
    return [dict(row) for row in connection.execute(query)]


def _sentinel_stream(rows: Sequence[dict[str, Any]], fragment: str) -> int | None:
    counts = Counter(
        int(row["streamId"]) for row in rows if fragment in str(row["name"])
    )
    return counts.most_common(1)[0][0] if counts else None


def _stream_rows(
    rows: Sequence[dict[str, Any]], stream: int | None
) -> list[dict[str, Any]]:
    if stream is None:
        return []
    return sorted(
        (row for row in rows if int(row["streamId"]) == stream),
        key=lambda row: int(row["start"]),
    )


def _top_operations(
    rows: Sequence[dict[str, Any]], limit: int = 12
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["name"])].append(row)
    result = []
    for name, values in grouped.items():
        result.append(
            {
                "name": name,
                "count": len(values),
                "total_us": sum(int(row["end"]) - int(row["start"]) for row in values)
                / 1000.0,
                "duration_us": _distribution(
                    (int(row["end"]) - int(row["start"])) / 1000.0 for row in values
                ),
            }
        )
    result.sort(key=lambda row: row["total_us"], reverse=True)
    return result[:limit]


def _ns_to_us_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ts": int(row["start"]) / 1000.0,
            "dur": (int(row["end"]) - int(row["start"])) / 1000.0,
        }
        for row in rows
    ]


def _rounds(
    forward_rows: Sequence[dict[str, Any]],
    landing_rows: Sequence[dict[str, Any]],
    runtime_rows: Sequence[dict[str, Any]],
    drop_first_rounds: int,
) -> dict[str, Any]:
    boundaries = [
        row for row in forward_rows if SELECT_KERNEL_FRAGMENT in str(row["name"])
    ]
    graph_runtime = {
        int(row["correlationId"]): row
        for row in runtime_rows
        if row.get("correlationId") is not None
        and "cudagraphlaunch" in str(row["name"]).lower()
    }
    result_rows = []
    for index in range(max(0, drop_first_rounds), len(boundaries) - 1):
        start = int(boundaries[index]["start"])
        end = int(boundaries[index + 1]["start"])
        operations = [row for row in forward_rows if start <= int(row["start"]) < end]
        landing = [row for row in landing_rows if start <= int(row["start"]) < end]
        graphs = [row for row in operations if row["kind"] == "cuda_graph"]
        graph = graphs[0] if len(graphs) == 1 else None
        publish_rows = [
            row
            for row in landing_rows
            if row["kind"] == "kernel" and PUBLISH_KERNEL_FRAGMENT in str(row["name"])
        ]
        post_graph_publishes = (
            [
                row
                for row in publish_rows
                if int(graph["end"]) <= int(row["start"]) < end
            ]
            if graph is not None
            else []
        )
        completed_publishes = [row for row in publish_rows if int(row["end"]) <= end]
        latest_publish_age_us = (
            (end - int(completed_publishes[-1]["end"])) / 1000.0
            if completed_publishes
            else None
        )
        next_select = boundaries[index + 1]
        publish_select_overlap_count = sum(
            int(row["start"]) < int(next_select["end"])
            and int(row["end"]) > int(next_select["start"])
            for row in publish_rows
        )
        active_us = _interval_union_us(_ns_to_us_rows(operations))
        cycle_us = (end - start) / 1000.0
        graph_runtime_us = []
        for graph_row in graphs:
            runtime = graph_runtime.get(int(graph_row["correlationId"]))
            if runtime is not None:
                graph_runtime_us.append(
                    (int(runtime["end"]) - int(runtime["start"])) / 1000.0
                )
        result_rows.append(
            {
                "cycle_us": cycle_us,
                "forward_visible_device_op_count": len(operations),
                "forward_device_active_union_us": active_us,
                "forward_exposed_gap_us": max(0.0, cycle_us - active_us),
                "cuda_graph_count": len(graphs),
                "cuda_graph_gpu_us": sum(
                    (int(row["end"]) - int(row["start"])) / 1000.0 for row in graphs
                ),
                "cuda_graph_host_api_us": sum(graph_runtime_us),
                "select_to_graph_start_us": (
                    (int(graph["start"]) - start) / 1000.0 if graph else None
                ),
                "graph_end_to_next_select_us": (
                    (end - int(graph["end"])) / 1000.0 if graph else None
                ),
                "landing_visible_device_op_count": len(landing),
                "landing_publish_count": sum(
                    PUBLISH_KERNEL_FRAGMENT in str(row["name"]) for row in landing
                ),
                "landing_device_active_union_us": _interval_union_us(
                    _ns_to_us_rows(landing)
                ),
                "post_graph_landing_publish_count": len(post_graph_publishes),
                "latest_landing_publish_age_at_select_us": latest_publish_age_us,
                "landing_publish_overlap_with_select_count": publish_select_overlap_count,
            }
        )

    def values(name: str) -> Iterable[float]:
        return (row[name] for row in result_rows if row[name] is not None)

    fields = (
        "cycle_us",
        "forward_visible_device_op_count",
        "forward_device_active_union_us",
        "forward_exposed_gap_us",
        "cuda_graph_count",
        "cuda_graph_gpu_us",
        "cuda_graph_host_api_us",
        "select_to_graph_start_us",
        "graph_end_to_next_select_us",
        "landing_visible_device_op_count",
        "landing_publish_count",
        "landing_device_active_union_us",
        "post_graph_landing_publish_count",
        "latest_landing_publish_age_at_select_us",
        "landing_publish_overlap_with_select_count",
    )
    return {
        "select_boundary_count": len(boundaries),
        "complete_round_count": len(result_rows),
        "dropped_first_rounds": max(0, drop_first_rounds),
        "post_graph_landing_publish_counts": [
            row["post_graph_landing_publish_count"] for row in result_rows
        ],
        "latest_landing_publish_age_at_select_us_values": [
            row["latest_landing_publish_age_at_select_us"] for row in result_rows
        ],
        "landing_publish_overlap_with_select_counts": [
            row["landing_publish_overlap_with_select_count"] for row in result_rows
        ],
        "rounds_with_post_graph_landing_publish": sum(
            row["post_graph_landing_publish_count"] > 0 for row in result_rows
        ),
        "rounds_with_publish_select_overlap": sum(
            row["landing_publish_overlap_with_select_count"] > 0 for row in result_rows
        ),
        **{field: _distribution(values(field)) for field in fields},
    }


def analyze_sqlite(
    sqlite_path: Path,
    *,
    source_path: Path,
    device: int,
    drop_first_rounds: int,
    expected_bs: int,
    trigger_path: Path | None,
    verifier_log_path: Path | None,
) -> dict[str, Any]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        _require_cupti_tables(connection)
        kernels = _kernel_rows(connection, device)
        graphs = _graph_rows(connection, device)
        memcopies = _memcpy_rows(connection, device)
        memsets = _memset_rows(connection, device)
        runtime = _runtime_rows(connection)
    finally:
        connection.close()

    visible_rows = sorted(
        [*kernels, *graphs, *memcopies, *memsets], key=lambda row: int(row["start"])
    )
    forward_stream = _sentinel_stream(kernels, SELECT_KERNEL_FRAGMENT)
    landing_stream = _sentinel_stream(kernels, PUBLISH_KERNEL_FRAGMENT)
    forward_rows = _stream_rows(visible_rows, forward_stream)
    landing_rows = _stream_rows(visible_rows, landing_stream)
    forward_us = _ns_to_us_rows(forward_rows)
    landing_us = _ns_to_us_rows(landing_rows)
    landing_overlap_us = _intersection_union_us(forward_us, landing_us)
    landing_busy_us = _interval_union_us(landing_us)

    runtime_by_name: dict[str, list[float]] = defaultdict(list)
    for row in runtime:
        name = str(row["name"])
        if any(
            fragment in name.lower()
            for fragment in (
                "cudagraphlaunch",
                "cudamemcpyasync",
                "cudalaunchkernel",
                "cudastreamsynchronize",
            )
        ):
            runtime_by_name[name].append((int(row["end"]) - int(row["start"])) / 1000.0)

    landing_publish = [
        row for row in kernels if PUBLISH_KERNEL_FRAGMENT in str(row["name"])
    ]
    fold_rows = [
        row for row in kernels if "gdn_replayssm_exact_fold_kernel" in str(row["name"])
    ]
    window_gate = build_window_gate(
        expected_bs=expected_bs,
        fold_grid_y_values=[int(row["gridY"]) for row in fold_rows],
        trigger_path=trigger_path,
        verifier_log_path=verifier_log_path,
    )
    landing_non_publish = [
        row
        for row in landing_rows
        if not (row["kind"] == "kernel" and PUBLISH_KERNEL_FRAGMENT in str(row["name"]))
    ]
    return {
        "schema_version": 1,
        "source_path": str(source_path.resolve()),
        "sqlite_path": str(sqlite_path.resolve()),
        "device": device,
        "window_gate": window_gate,
        "stream_roles": {"forward": forward_stream, "landing": landing_stream},
        "visible_event_counts": {
            "kernels": len(kernels),
            "cuda_graph_replays": len(graphs),
            "memcopies": len(memcopies),
            "memsets": len(memsets),
        },
        "forward_rounds": _rounds(
            forward_rows, landing_rows, runtime, drop_first_rounds
        ),
        "forward_top_operations": _top_operations(forward_rows),
        "landing_top_operations": _top_operations(landing_rows),
        "landing_structure": {
            "publish_kernel_count": len(landing_publish),
            "publish_kernel_duration_us": _distribution(
                (int(row["end"]) - int(row["start"])) / 1000.0
                for row in landing_publish
            ),
            "non_publish_visible_op_count": len(landing_non_publish),
            "busy_union_us": landing_busy_us,
            "overlap_with_forward_us": landing_overlap_us,
            "overlap_fraction": (
                landing_overlap_us / landing_busy_us if landing_busy_us else None
            ),
        },
        "cuda_runtime_duration_us": {
            name: _distribution(values) for name, values in runtime_by_name.items()
        },
        "interpretation_limits": [
            "CUDA graph replay is one opaque visible operation when graph-node tracing is absent.",
            "Nsight Systems collection perturbs timing; use it for structure, not production latency.",
            "A physical landing stream carrying non-publish operations indicates stream aliasing or shared side-stream work, not ownership by the landing daemon alone.",
        ],
    }


def analyze_path(
    path: Path,
    device: int,
    drop_first_rounds: int,
    expected_bs: int,
    trigger_path: Path | None,
    verifier_log_path: Path | None,
) -> dict[str, Any]:
    if path.suffix == ".sqlite":
        return analyze_sqlite(
            path,
            source_path=path,
            device=device,
            drop_first_rounds=drop_first_rounds,
            expected_bs=expected_bs,
            trigger_path=trigger_path,
            verifier_log_path=verifier_log_path,
        )
    if not path.name.endswith(".nsys-rep"):
        raise ValueError("--input must be a .nsys-rep or .sqlite file")
    with tempfile.TemporaryDirectory(prefix="decoupled-nsys-") as directory:
        sqlite_path = Path(directory) / "capture.sqlite"
        _export_report(path, sqlite_path)
        report = analyze_sqlite(
            sqlite_path,
            source_path=path,
            device=device,
            drop_first_rounds=drop_first_rounds,
            expected_bs=expected_bs,
            trigger_path=trigger_path,
            verifier_log_path=verifier_log_path,
        )
        report["sqlite_path"] = None
        return report


def render_text(report: dict[str, Any]) -> str:
    rounds = report["forward_rounds"]
    lines = [
        f"Nsys source: {report['source_path']}",
        f"Device: {report['device']}",
        (
            "Window gate: "
            f"passed={report['window_gate']['passed']} "
            f"expected_bs={report['window_gate']['expected_bs']} "
            "fold_grid_y="
            f"{report['window_gate']['fold_grid_y_values']} "
            "log_bs="
            f"{(report['window_gate']['log_context'] or {}).get('running_bs')}"
        ),
        f"Streams: forward={report['stream_roles']['forward']} landing={report['stream_roles']['landing']}",
        "Visible events: "
        + " ".join(
            f"{key}={value}" for key, value in report["visible_event_counts"].items()
        ),
        "",
        (
            "Forward rounds: boundaries={select_boundary_count} "
            "complete={complete_round_count} dropped={dropped_first_rounds}"
        ).format(**rounds),
    ]
    for field in (
        "cycle_us",
        "forward_visible_device_op_count",
        "forward_device_active_union_us",
        "forward_exposed_gap_us",
        "cuda_graph_count",
        "cuda_graph_gpu_us",
        "cuda_graph_host_api_us",
        "select_to_graph_start_us",
        "graph_end_to_next_select_us",
        "landing_visible_device_op_count",
        "landing_publish_count",
        "landing_device_active_union_us",
        "post_graph_landing_publish_count",
        "latest_landing_publish_age_at_select_us",
        "landing_publish_overlap_with_select_count",
    ):
        lines.append(f"  {field}: {_format_distribution(rounds[field])}")

    lines.extend(["", "Forward stream top visible operations:"])
    for row in report["forward_top_operations"]:
        lines.append(
            f"  {row['count']}x total_us={row['total_us']:.3f} "
            f"p50_us={_format_value(row['duration_us']['p50'])} {row['name']}"
        )
    lines.extend(["", "Landing/shared side stream top visible operations:"])
    for row in report["landing_top_operations"]:
        lines.append(
            f"  {row['count']}x total_us={row['total_us']:.3f} "
            f"p50_us={_format_value(row['duration_us']['p50'])} {row['name']}"
        )
    landing = report["landing_structure"]
    lines.extend(
        [
            "",
            "Landing structure:",
            f"  publish_kernel_count={landing['publish_kernel_count']}",
            "  publish_kernel_duration_us: "
            + _format_distribution(landing["publish_kernel_duration_us"]),
            f"  non_publish_visible_op_count={landing['non_publish_visible_op_count']}",
            f"  busy_union_us={_format_value(landing['busy_union_us'])}",
            f"  overlap_with_forward_us={_format_value(landing['overlap_with_forward_us'])}",
            f"  overlap_fraction={_format_value(landing['overlap_fraction'])}",
            "",
            "CUDA runtime duration:",
        ]
    )
    for name, distribution in report["cuda_runtime_duration_us"].items():
        lines.append(f"  {name}: {_format_distribution(distribution)}")
    lines.extend(
        ["", "Limits:", *(f"  - {item}" for item in report["interpretation_limits"])]
    )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["window_gate"]
    rounds = report["forward_rounds"]
    lines = [
        "# Decoupled verifier Nsys forward structure",
        "",
        f"- Source: `{report['source_path']}`",
        f"- Device: `{report['device']}`",
        (
            f"- Window gate: **{'pass' if gate['passed'] else 'fail'}**; "
            f"expected BS `{gate['expected_bs']}`, fold `gridY` "
            f"`{gate['fold_grid_y_values']}`, trigger-context BS "
            f"`{(gate['log_context'] or {}).get('running_bs')}`"
        ),
        (
            f"- Physical streams: forward `{report['stream_roles']['forward']}`, "
            f"landing `{report['stream_roles']['landing']}`"
        ),
        "",
        "## Select-to-select rounds",
        "",
        "| Metric | n | mean | p50 | p95 | p99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fields = (
        "cycle_us",
        "forward_visible_device_op_count",
        "forward_device_active_union_us",
        "forward_exposed_gap_us",
        "cuda_graph_count",
        "cuda_graph_gpu_us",
        "cuda_graph_host_api_us",
        "select_to_graph_start_us",
        "graph_end_to_next_select_us",
        "landing_visible_device_op_count",
        "landing_publish_count",
        "landing_device_active_union_us",
        "post_graph_landing_publish_count",
        "latest_landing_publish_age_at_select_us",
        "landing_publish_overlap_with_select_count",
    )
    for field in fields:
        row = rounds[field]
        lines.append(
            f"| `{field}` | {row['count']} | {_format_value(row['mean'])} | "
            f"{_format_value(row['p50'])} | {_format_value(row['p95'])} | "
            f"{_format_value(row['p99'])} |"
        )
    landing = report["landing_structure"]
    lines.extend(
        [
            "",
            "## Tail publication phase at select",
            "",
            "- Post-graph to next-select publish counts: `"
            + str(rounds["post_graph_landing_publish_counts"])
            + "`.",
            "- Latest completed publish age at select (us): `"
            + str(rounds["latest_landing_publish_age_at_select_us_values"])
            + "`.",
            "- Publish/select overlap counts: `"
            + str(rounds["landing_publish_overlap_with_select_counts"])
            + "`.",
            "",
            "## Landing stream",
            "",
            f"- Publish kernels: `{landing['publish_kernel_count']}`; "
            f"p50 `{_format_value(landing['publish_kernel_duration_us']['p50'])} us`.",
            f"- Busy union: `{_format_value(landing['busy_union_us'])} us`.",
            f"- Overlap with forward activity: `{_format_value(landing['overlap_with_forward_us'])} us` "
            f"(`{_format_value(landing['overlap_fraction'])}` of landing busy time).",
            "",
            "## Top visible forward operations",
            "",
            "| Count | Total us | p50 us | Operation |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["forward_top_operations"]:
        lines.append(
            f"| {row['count']} | {row['total_us']:.3f} | "
            f"{_format_value(row['duration_us']['p50'])} | `{row['name']}` |"
        )
    lines.extend(
        [
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expected-bs", type=int, required=True)
    parser.add_argument("--drop-first-rounds", type=int, default=0)
    parser.add_argument("--trigger-file", type=Path)
    parser.add_argument("--verifier-log", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    if args.device < 0 or args.drop_first_rounds < 0 or args.expected_bs <= 0:
        parser.error(
            "--device/--drop-first-rounds must be non-negative and --expected-bs positive"
        )
    return args


def main() -> None:
    args = _parse_args()
    report = analyze_path(
        args.input,
        args.device,
        args.drop_first_rounds,
        args.expected_bs,
        args.trigger_file,
        args.verifier_log,
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


if __name__ == "__main__":
    main()

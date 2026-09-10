"""Generate a Markdown summary for one decoupled-spec benchmark run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client.observer import (
    _summarize_decode_metric_windows,
    _update_decode_metric_windows,
)
from plot_utils import (
    UPPER_IQR_OUTLIER_POLICY,
    finite_float,
    load_csv,
    load_json,
    upper_iqr_outlier_threshold,
)
from run_io import require_run_dir


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _format(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _format_peer_range(value: Any, prefix: str) -> str:
    if not isinstance(value, dict):
        return "n/a"
    minimum = value.get(f"{prefix}_min")
    maximum = value.get(f"{prefix}_max")
    if type(minimum) is not int or type(maximum) is not int:
        return "n/a"
    return str(minimum) if minimum == maximum else f"{minimum}-{maximum}"


def generate_report(run_dir: str | Path) -> dict[str, Any]:
    run_path = require_run_dir(run_dir)
    output_dir = run_path / "plots"
    config_path = run_path / "config.json"
    batch_path = run_path / "client" / "batch.json"
    requests_path = run_path / "client" / "requests.csv"
    if not batch_path.is_file():
        raise FileNotFoundError(batch_path)
    if not requests_path.is_file():
        raise FileNotFoundError(requests_path)

    source_paths = [
        config_path,
        batch_path,
        requests_path,
        run_path / "client" / "content.json",
        run_path / "observer" / "samples.jsonl",
        run_path / "observer" / "bench_timeline.json",
    ]
    run_config = load_json(config_path)
    batch = load_json(batch_path)
    request_rows = load_csv(requests_path)
    server = run_config.get("server", {})
    verifier = server.get("verifier", {})
    drafter = server.get("drafter", {})
    client = run_config.get("client", {})
    observability: dict[str, Any] = {}
    samples_path = run_path / "observer" / "samples.jsonl"
    if samples_path.is_file():
        records = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        targets_by_id = {}
        for record in records:
            target_id = str(record.get("target_id", record.get("role")))
            targets_by_id[target_id] = {
                "target_id": target_id,
                "role": str(record["role"]),
                "rank": int(record.get("rank", 0)),
                "base_url": str(record.get("base_url", "")),
            }
        windows = {}
        min_end_time = min(
            (
                float(record["collected_wall_time"])
                for record in records
                if record.get("error") is None
                and record.get("payload", {}).get("loads")
            ),
            default=0.0,
        )
        for record in records:
            _update_decode_metric_windows(record, windows, min_end_time)
        by_target, by_role = _summarize_decode_metric_windows(
            windows, list(targets_by_id.values())
        )
        observability = {
            "decode_metrics_by_target": by_target,
            "decode_metrics": by_role,
        }
        entries_by_series: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for entry in windows.values():
            entries_by_series.setdefault(
                (entry["target_id"], entry["dp_rank"]), []
            ).append(entry)
        iteration_latency_outliers = []
        for (target_id, dp_rank), entries in sorted(entries_by_series.items()):
            threshold = upper_iqr_outlier_threshold(
                [entry["window"]["iter_latency_ms"] for entry in entries]
            )
            if threshold is None:
                continue
            for entry in entries:
                window = entry["window"]
                latency_ms = float(window["iter_latency_ms"])
                if latency_ms > threshold:
                    iteration_latency_outliers.append(
                        {
                            "target_id": target_id,
                            "dp_rank": dp_rank,
                            "window_id": int(window["window_id"]),
                            "time_s": float(window["end_time"]) - min_end_time,
                            "iter_latency_ms": latency_ms,
                            "threshold_ms": threshold,
                        }
                    )
        iteration_latency_outliers.sort(
            key=lambda item: (
                item["target_id"],
                item["dp_rank"],
                item["time_s"],
                item["window_id"],
            )
        )
    else:
        iteration_latency_outliers = []
    verifier_args = verifier.get("server_args", {})
    drafter_args = drafter.get("server_args", {})
    dataset = client.get("dataset", {})
    generation = client.get("generation", {})
    decode_metric_lines: list[str] = []
    decoupled_metric_lines: list[str] = []
    decode_metrics = observability.get("decode_metrics_by_target")
    if not isinstance(decode_metrics, dict) or not decode_metrics:
        decode_metrics = observability.get("decode_metrics")
    if isinstance(decode_metrics, dict) and decode_metrics:
        decode_metric_lines = [
            "### Decode-window telemetry",
            "",
            "| Engine | Windows | Iteration latency mean | p50 | p95 | Mean BS | Mean context | Valid draft len | Accept len |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        ordered_metrics = sorted(
            decode_metrics.items(),
            key=lambda item: (
                str(item[1].get("role", item[0])) if isinstance(item[1], dict) else "",
                int(item[1].get("rank", 0)) if isinstance(item[1], dict) else 0,
            ),
        )
        for target_id, metrics in ordered_metrics:
            if not isinstance(metrics, dict):
                continue
            decode_metric_lines.append(
                "| {role} | {windows} | {mean} ms | {p50} ms | {p95} ms | "
                "{mean_bs} | {mean_context} | {valid_draft} | {accept} |".format(
                    role=target_id,
                    windows=metrics.get("window_count", 0),
                    mean=_format(_nested(metrics, "scheduler_cycle_ms", "mean")),
                    p50=_format(_nested(metrics, "scheduler_cycle_ms", "p50")),
                    p95=_format(_nested(metrics, "scheduler_cycle_ms", "p95")),
                    mean_bs=_format(metrics.get("mean_batch_size")),
                    mean_context=_format(metrics.get("mean_context_length")),
                    valid_draft=_format(metrics.get("valid_draft_length")),
                    accept=_format(metrics.get("accept_length")),
                )
            )
        decode_metric_lines.append("")
        decoupled_rows = [
            (target_id, metrics)
            for target_id, metrics in ordered_metrics
            if isinstance(metrics, dict)
            and isinstance(metrics.get("decoupled_spec"), dict)
        ]
        if decoupled_rows:
            decoupled_metric_lines = [
                "### Decoupled-spec observability",
                "",
                "Transport quantiles below are recomputed from merged raw histogram "
                "buckets; they are not averages of per-window quantiles.",
                "Drafter-to-verifier transport quantiles include record-time "
                "calibrated samples; "
                "the valid/invalid ranges show drain-time peer state.",
                "",
                "| Engine | Select rows | Valid row rate | Pending fast-forwards | "
                "Seqlock retry rows/retries/max | Frames | Tokens | "
                "Clock peers current valid/invalid | Send queue p50/p95 | "
                "Ready to receive p50/p95 (record-time calibrated) | "
                "Send to receive p50/p95 (record-time calibrated) | "
                "Receive to publish enqueue p50/p95 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for target_id, metrics in decoupled_rows:
                tail = _nested(metrics, "decoupled_spec", "tail_select") or {}
                transport = _nested(metrics, "decoupled_spec", "transport") or {}
                clock_peers = transport.get("clock_sync_peers")
                decoupled_metric_lines.append(
                    "| {engine} | {rows} | {valid} | {fast_forwards} | {seqlock} | "
                    "{frames} | {tokens} | "
                    "{valid_peers}/{invalid_peers} | {send_p50}/{send_p95} us | "
                    "{ready_p50}/{ready_p95} us | {one_way_p50}/{one_way_p95} us | "
                    "{receive_p50}/{receive_p95} us |".format(
                        engine=target_id,
                        rows=tail.get("num_select_rows", "n/a"),
                        valid=_format(tail.get("valid_row_rate")),
                        fast_forwards=tail.get(
                            "num_pending_prefix_fast_forwards", "n/a"
                        ),
                        seqlock=(
                            f"{tail.get('num_seqlock_retry_rows', 'n/a')}/"
                            f"{tail.get('num_seqlock_retries', 'n/a')}/"
                            f"{tail.get('max_seqlock_retries', 'n/a')}"
                            if tail
                            else "n/a"
                        ),
                        frames=transport.get("num_draft_result_frames", "n/a"),
                        tokens=transport.get("num_draft_result_tokens", "n/a"),
                        valid_peers=_format_peer_range(clock_peers, "valid"),
                        invalid_peers=_format_peer_range(clock_peers, "invalid"),
                        send_p50=_format(
                            _nested(
                                transport,
                                "draft_send_queue_latency_us",
                                "p50_us",
                            )
                        ),
                        send_p95=_format(
                            _nested(
                                transport,
                                "draft_send_queue_latency_us",
                                "p95_us",
                            )
                        ),
                        ready_p50=_format(
                            _nested(
                                transport,
                                "draft_result_ready_to_receive_latency_us",
                                "p50_us",
                            )
                        ),
                        ready_p95=_format(
                            _nested(
                                transport,
                                "draft_result_ready_to_receive_latency_us",
                                "p95_us",
                            )
                        ),
                        one_way_p50=_format(
                            _nested(
                                transport,
                                "draft_transport_one_way_latency_us",
                                "p50_us",
                            )
                        ),
                        one_way_p95=_format(
                            _nested(
                                transport,
                                "draft_transport_one_way_latency_us",
                                "p95_us",
                            )
                        ),
                        receive_p50=_format(
                            _nested(
                                transport,
                                "draft_receive_to_gpu_publish_enqueue_latency_us",
                                "p50_us",
                            )
                        ),
                        receive_p95=_format(
                            _nested(
                                transport,
                                "draft_receive_to_gpu_publish_enqueue_latency_us",
                                "p95_us",
                            )
                        ),
                    )
                )
            decoupled_metric_lines.append("")
    iteration_outlier_lines: list[str] = []
    if iteration_latency_outliers:
        iteration_outlier_lines = [
            "### Iteration-latency plot exclusions",
            "",
            f"The figure excludes {len(iteration_latency_outliers)} presentation "
            "outlier windows. Raw observer windows and the aggregate telemetry "
            "table above are unchanged.",
            f"Rule: `{UPPER_IQR_OUTLIER_POLICY}`.",
            "",
            "| Engine | DP rank | Threshold (ms) | Window ID | Time since first sample (s) | Value (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *(
                "| {target_id} | {dp_rank} | {threshold_ms:.3f} | "
                "{window_id} | {time_s:.3f} | {iter_latency_ms:.3f} |".format(
                    **item
                )
                for item in iteration_latency_outliers
            ),
            "",
        ]
    figures = [
        path
        for path in (
            run_path / "plots" / "request_latency.png",
            run_path / "plots" / "request_speculative.png",
            run_path / "plots" / "overview.png",
            run_path / "plots" / "decode_metrics.png",
            run_path / "plots" / "decoupled_spec_metrics.png",
            run_path / "plots" / "adaptive_verify.png",
        )
        if path.is_file()
    ]
    lines = [
        "# Decoupled Speculation Run Summary",
        "",
        f"- Run: `{run_path}`",
        f"- Run directory: `{run_path}`",
        "",
        "## Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Target model | `{verifier_args.get('model_path', 'n/a')}` |",
        f"| Target TP | `{verifier_args.get('tp_size', 'n/a')}` |",
        f"| Draft model | `{drafter_args.get('model_path', 'n/a')}` |",
        f"| Draft TP | `{drafter_args.get('tp_size', 'n/a')}` |",
        f"| Spec steps | `{verifier_args.get('speculative_num_steps', 'n/a')}` |",
        f"| Draft fanout F | `{verifier_args.get('speculative_eagle_topk', 'n/a')}` |",
        "| Overlap schedule | "
        f"`{not verifier_args.get('disable_overlap_schedule', False)}` |",
        f"| Batch size | `{len(request_rows)}` |",
        f"| Dataset | `{dataset.get('format', 'n/a')}` |",
        f"| Output length | `{generation.get('output_len', 'n/a')}` |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        "| Batch elapsed latency | "
        f"{_format(batch.get('batch_elapsed_latency_s'))} s |",
        f"| Completion tokens | {_format(batch.get('output_tokens'), 0)} |",
        "| Output throughput | " f"{_format(batch.get('batch_thpt'))} tokens/s |",
        "| Mean valid draft length | "
        f"{_format(batch.get('mean_valid_draft_len'))} drafts/verify |",
        f"| Mean accept length | {_format(batch.get('acclen'))} tokens/verify |",
        "",
        *decode_metric_lines,
        *iteration_outlier_lines,
        *decoupled_metric_lines,
        "### Request-level metrics",
        "",
        "| Batch row | Dataset row | Verifier rank | Prompt len | Response len | "
        "Verify count | Valid draft len | Accept len | E2E latency (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(
            "| {batch_row_index} | {dataset_idx} | {verifier_rank} | "
            "{prompt_len} | {resp_len} | {spec_verify_ct} | {valid_draft_len} | "
            "{acc_len} | {e2e_latency_s} |".format(**row)
            for row in request_rows
        ),
        "",
        "## Figures",
        "",
    ]
    lines.extend(
        f"- [{path.name}]({os.path.relpath(path, output_dir)})" for path in figures
    )
    lines.extend(
        [
            "",
            "## Sources",
            "",
            *(
                f"- `{path.relative_to(run_path)}`"
                for path in source_paths
                if path.is_file()
            ),
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(generate_report(args.run_dir)["report_path"])


if __name__ == "__main__":
    main()

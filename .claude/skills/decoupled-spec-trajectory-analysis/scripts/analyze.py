#!/usr/bin/env python3
"""Analyze decoupled-spec scheduler logs against a throughput profile cache."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from analysis_common import (
    Case,
    ProfileTable,
    discover_cases,
    moving_average,
    parse_decode_points,
    parse_switches,
    write_csv,
)
from analysis_render import plot_dynamic, plot_static, plot_trajectories, write_report


SMOOTH_METRICS = (
    "observed_itl_ms",
    "modeled_itl_ms",
    "observed_throughput_tok_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse scheduler INFO, reproduce controller profile lookup, and plot gaps."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-cpu-overhead-ms", type=float, default=2.0)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--latency-cutoff-ms", type=float, default=100.0)
    parser.add_argument(
        "--include-partial-batches",
        action="store_true",
        help="Keep tail points below the maximum runtime batch size in profile-fit plots.",
    )
    args = parser.parse_args()
    if args.runtime_cpu_overhead_ms < 0:
        parser.error("--runtime-cpu-overhead-ms must be non-negative")
    if args.smooth_window < 1:
        parser.error("--smooth-window must be positive")
    if args.latency_cutoff_ms <= 0:
        parser.error("--latency-cutoff-ms must be positive")
    return args


def filter_points(
    rows: list[dict[str, Any]], cutoff_ms: float, include_partial_batches: bool
) -> list[dict[str, Any]]:
    """Remove startup outliers and, by default, per-case tail batches."""
    max_bs_by_case: dict[str, int] = defaultdict(int)
    for row in rows:
        max_bs_by_case[row["label"]] = max(
            max_bs_by_case[row["label"]], int(row["batch_size"])
        )
    return [
        row
        for row in rows
        if float(row["observed_itl_ms"]) < cutoff_ms
        and (
            include_partial_batches
            or int(row["batch_size"]) == max_bs_by_case[row["label"]]
        )
    ]


def filter_trajectory_points(
    rows: list[dict[str, Any]], cutoff_ms: float
) -> list[dict[str, Any]]:
    """Keep the full batch-size trajectory while removing latency outliers."""
    return [row for row in rows if float(row["observed_itl_ms"]) < cutoff_ms]


def smooth_points(rows: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = sorted({str(row["label"]) for row in rows})
    for label in labels:
        case_rows = [row for row in rows if row["label"] == label]
        smoothed = {
            metric: moving_average(
                [float(row[metric]) for row in case_rows], window
            )
            for metric in SMOOTH_METRICS
        }
        accept_values = [row["accept_len"] for row in case_rows]
        accept_smooth = (
            moving_average([float(value) for value in accept_values], window)
            if all(value != "" for value in accept_values)
            else None
        )
        for index, row in enumerate(case_rows):
            current = dict(row)
            for metric in SMOOTH_METRICS:
                current[f"{metric}_smooth"] = smoothed[metric][index]
            current["accept_len_smooth"] = (
                accept_smooth[index] if accept_smooth is not None else ""
            )
            current["observed_minus_modeled_itl_ms_smooth"] = (
                current["observed_itl_ms_smooth"]
                - current["modeled_itl_ms_smooth"]
            )
            output.append(current)
    return output


def summarize(rows: list[dict[str, Any]], switches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for label in sorted({str(row["label"]) for row in rows}):
        case_rows = [row for row in rows if row["label"] == label]
        first = case_rows[0]
        modeled_throughputs = [
            float(row["modeled_throughput_tok_s"])
            for row in case_rows
            if row["modeled_throughput_tok_s"] != ""
        ]
        throughput_ratios = [
            float(row["observed_over_modeled_throughput"])
            for row in case_rows
            if row["observed_over_modeled_throughput"] != ""
        ]
        summaries.append(
            {
                "label": label,
                "allow_partial": first["allow_partial"],
                "dynamic": first["dynamic"],
                "max_step": first["max_step"],
                "points": len(case_rows),
                "duration_s": max(float(row["elapsed_s"]) for row in case_rows),
                "batch_size_min": min(int(row["batch_size"]) for row in case_rows),
                "batch_size_max": max(int(row["batch_size"]) for row in case_rows),
                "queue_req_max": max(int(row["queue_req"]) for row in case_rows),
                "switch_count": sum(1 for row in switches if row["label"] == label),
                "observed_itl_mean_ms": mean(
                    float(row["observed_itl_ms"]) for row in case_rows
                ),
                "modeled_itl_mean_ms": mean(
                    float(row["modeled_itl_ms"]) for row in case_rows
                ),
                "observed_minus_modeled_itl_mean_ms": mean(
                    float(row["observed_minus_modeled_itl_ms"]) for row in case_rows
                ),
                "observed_throughput_mean_tok_s": mean(
                    float(row["observed_throughput_tok_s"]) for row in case_rows
                ),
                "modeled_throughput_mean_tok_s": (
                    mean(modeled_throughputs) if modeled_throughputs else ""
                ),
                "observed_over_modeled_throughput_mean": (
                    mean(throughput_ratios) if throughput_ratios else ""
                ),
                "observed_itl_median_ms": median(
                    float(row["observed_itl_ms"]) for row in case_rows
                ),
                "modeled_itl_median_ms": median(
                    float(row["modeled_itl_ms"]) for row in case_rows
                ),
            }
        )
    return summaries


def summarize_step_occupancy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure controller residency by iterations, reconstructed time, and output."""
    output: list[dict[str, Any]] = []
    for label in sorted({str(row["label"]) for row in rows}):
        case_rows = [row for row in rows if row["label"] == label]
        total_points = len(case_rows)
        total_time_ms = sum(float(row["observed_itl_ms"]) for row in case_rows)
        total_tokens = sum(float(row["output_tokens"]) for row in case_rows)
        for step in sorted({int(row["active_step"]) for row in case_rows}):
            step_rows = [row for row in case_rows if int(row["active_step"]) == step]
            step_time_ms = sum(float(row["observed_itl_ms"]) for row in step_rows)
            step_tokens = sum(float(row["output_tokens"]) for row in step_rows)
            output.append(
                {
                    "label": label,
                    "active_step": step,
                    "points": len(step_rows),
                    "point_share": len(step_rows) / total_points,
                    "reconstructed_time_s": step_time_ms / 1000.0,
                    "time_share": step_time_ms / total_time_ms,
                    "output_tokens": step_tokens,
                    "output_token_share": step_tokens / total_tokens,
                    "batch_size_mean": mean(
                        float(row["batch_size"]) for row in step_rows
                    ),
                    "ctx_per_req_mean": mean(
                        float(row["ctx_per_req"]) for row in step_rows
                    ),
                }
            )
    return output


def summarize_e2e(run_dir: Path, cases: list[Case]) -> list[dict[str, Any]]:
    """Extract benchmark-level metrics without mixing them with scheduler-point means."""
    rows: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: item.label):
        label = case.label
        path = run_dir / "runs" / label / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        metrics = payload.get("decoupled_spec")
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "label": label,
                "allow_partial": case.allow_partial,
                "dynamic": case.dynamic,
                "max_step": case.max_step,
                "summary_path": str(path),
                "output_throughput_tok_per_s": metrics.get(
                    "output_throughput_tok_per_s"
                ),
                "generation_time_s": metrics.get("generation_time_s"),
                "total_generated_tokens": metrics.get("total_generated_tokens"),
                "avg_spec_accept_length": metrics.get("avg_spec_accept_length"),
                "avg_spec_accept_rate": metrics.get("avg_spec_accept_rate"),
                "avg_spec_valid_accept_rate": metrics.get(
                    "avg_spec_valid_accept_rate"
                ),
                "avg_spec_valid_accept_rate_by_position_json": json.dumps(
                    metrics.get("avg_spec_valid_accept_rate_by_position"),
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def summarize_speedups(e2e_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare dynamic E2E throughput with same-step and best static baselines."""
    rows: list[dict[str, Any]] = []
    for allow_partial in (False, True):
        static_rows = [
            row
            for row in e2e_rows
            if not row["dynamic"] and bool(row["allow_partial"]) == allow_partial
        ]
        if not static_rows:
            continue
        best_static = max(
            static_rows, key=lambda row: float(row["output_throughput_tok_per_s"])
        )
        static_by_step = {int(row["max_step"]): row for row in static_rows}
        for dynamic in [
            row
            for row in e2e_rows
            if row["dynamic"] and bool(row["allow_partial"]) == allow_partial
        ]:
            throughput = float(dynamic["output_throughput_tok_per_s"])
            same_step = static_by_step.get(int(dynamic["max_step"]))
            rows.append(
                {
                    "label": dynamic["label"],
                    "allow_partial": allow_partial,
                    "max_step": dynamic["max_step"],
                    "dynamic_throughput_tok_s": throughput,
                    "same_step_static_label": same_step["label"] if same_step else "",
                    "same_step_static_throughput_tok_s": (
                        same_step["output_throughput_tok_per_s"] if same_step else ""
                    ),
                    "speedup_vs_same_step": (
                        throughput / float(same_step["output_throughput_tok_per_s"])
                        if same_step
                        else ""
                    ),
                    "best_static_label": best_static["label"],
                    "best_static_throughput_tok_s": best_static[
                        "output_throughput_tok_per_s"
                    ],
                    "speedup_vs_best_static": throughput
                    / float(best_static["output_throughput_tok_per_s"]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    logs_dir = args.run_dir / "logs"
    output_dir = args.output_dir or args.run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(logs_dir)
    if not cases:
        raise RuntimeError(f"no ap{{0,1}}_{{static,dynamic}}_dl* logs found in {logs_dir}")
    profile = ProfileTable.load(args.profile)

    raw_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    for case in cases:
        raw_rows.extend(
            parse_decode_points(case, profile, args.runtime_cpu_overhead_ms)
        )
        switch_rows.extend(parse_switches(case))
    if not raw_rows:
        raise RuntimeError("case logs contain no parseable scheduler Decode batch INFO")

    filtered_rows = filter_points(
        raw_rows, args.latency_cutoff_ms, args.include_partial_batches
    )
    smooth_rows = smooth_points(filtered_rows, args.smooth_window)
    trajectory_rows = filter_trajectory_points(raw_rows, args.latency_cutoff_ms)
    trajectory_smooth_rows = smooth_points(trajectory_rows, args.smooth_window)
    summaries = summarize(filtered_rows, switch_rows)
    step_occupancy = summarize_step_occupancy(raw_rows)
    e2e_rows = summarize_e2e(args.run_dir, cases)
    speedups = summarize_speedups(e2e_rows)

    write_csv(output_dir / "decode_points.csv", raw_rows)
    write_csv(output_dir / "decode_points_filtered.csv", filtered_rows)
    write_csv(output_dir / "decode_points_smooth.csv", smooth_rows)
    write_csv(output_dir / "decode_points_trajectory.csv", trajectory_rows)
    write_csv(
        output_dir / "decode_points_trajectory_smooth.csv", trajectory_smooth_rows
    )
    write_csv(output_dir / "controller_switches.csv", switch_rows)
    write_csv(output_dir / "profile_costs.csv", list(profile.rows()))
    write_csv(output_dir / "case_summary.csv", summaries)
    write_csv(output_dir / "step_occupancy.csv", step_occupancy)
    write_csv(output_dir / "e2e_summary.csv", e2e_rows)
    write_csv(output_dir / "speedup_summary.csv", speedups)
    (output_dir / "case_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "e2e_summary.json").write_text(
        json.dumps(e2e_rows, indent=2, sort_keys=True) + "\n"
    )

    figures = []
    figures.extend(
        plot_trajectories(
            trajectory_rows,
            trajectory_smooth_rows,
            output_dir,
            args.smooth_window,
        )
    )
    figures.extend(plot_static(filtered_rows, output_dir, smooth=False))
    figures.extend(plot_static(smooth_rows, output_dir, smooth=True))
    figures.extend(plot_dynamic(filtered_rows, switch_rows, output_dir, smooth=False))
    figures.extend(plot_dynamic(smooth_rows, switch_rows, output_dir, smooth=True))
    write_report(output_dir, summaries, e2e_rows, speedups, figures)
    metadata = {
        "run_dir": str(args.run_dir.resolve()),
        "profile": str(args.profile.resolve()),
        "runtime_cpu_overhead_ms": args.runtime_cpu_overhead_ms,
        "lookup": "batch_size ceil/clamp, then ctx_len nearest with lower tie",
        "latency_cutoff_ms": args.latency_cutoff_ms,
        "include_partial_batches": args.include_partial_batches,
        "smooth_window_points": args.smooth_window,
        "case_count": len(cases),
        "raw_points": len(raw_rows),
        "filtered_points": len(filtered_rows),
        "trajectory_points": len(trajectory_rows),
        "switches": len(switch_rows),
        "step_occupancy_rows": len(step_occupancy),
        "e2e_summaries": len(e2e_rows),
        "speedup_comparisons": len(speedups),
        "figures": [str(path) for path in figures],
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

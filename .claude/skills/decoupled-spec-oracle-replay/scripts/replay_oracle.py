#!/usr/bin/env python3
"""Estimate a fluid optimal DL schedule from normalized decoupled-spec logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _state_key(row: dict[str, str]) -> tuple[bool, int, int]:
    return (
        _bool(row["allow_partial"]),
        int(row["matched_profile_batch_size"]),
        int(row["matched_profile_ctx_len"]),
    )


def _select_reference(
    e2e_rows: list[dict[str, str]], reference_label: str | None
) -> dict[str, str]:
    static_rows = [row for row in e2e_rows if not _bool(row["dynamic"])]
    if not static_rows:
        raise ValueError("e2e_summary.csv contains no static cases")
    if reference_label:
        matches = [row for row in static_rows if row["label"] == reference_label]
        if len(matches) != 1:
            raise ValueError(f"reference static case not found: {reference_label}")
        return matches[0]
    return max(static_rows, key=lambda row: float(row["output_throughput_tok_per_s"]))


def replay(
    decode_rows: list[dict[str, str]],
    e2e_rows: list[dict[str, str]],
    *,
    reference_label: str | None,
    target_capture: float,
    min_points: int,
    latency_cutoff_ms: float,
    static_step_filter: set[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reference_candidates = e2e_rows
    if static_step_filter is not None:
        reference_candidates = [
            row
            for row in e2e_rows
            if _bool(row["dynamic"]) or int(row["max_step"]) in static_step_filter
        ]
    reference = _select_reference(reference_candidates, reference_label)
    reference_label = reference["label"]
    allow_partial = _bool(reference["allow_partial"])
    available_static_steps = {
        int(row["max_step"])
        for row in e2e_rows
        if not _bool(row["dynamic"]) and _bool(row["allow_partial"]) == allow_partial
    }
    if static_step_filter is not None:
        missing_steps = static_step_filter - available_static_steps
        if missing_steps:
            raise ValueError(
                f"requested static steps are absent: {sorted(missing_steps)}"
            )
        if int(reference["max_step"]) not in static_step_filter:
            raise ValueError("reference static step is excluded by static_step_filter")
    static_labels = {
        row["label"]
        for row in e2e_rows
        if not _bool(row["dynamic"])
        and _bool(row["allow_partial"]) == allow_partial
        and (static_step_filter is None or int(row["max_step"]) in static_step_filter)
    }
    static_steps = sorted(
        {int(row["max_step"]) for row in e2e_rows if row["label"] in static_labels}
    )

    static_queue_points = [
        {
            "label": row["label"],
            "point_index": int(row["point_index"]),
            "queue_req": int(row.get("queue_req", 0) or 0),
        }
        for row in decode_rows
        if row["label"] in static_labels and int(row.get("queue_req", 0) or 0) > 0
    ]
    static_max_queue_req = max(
        (item["queue_req"] for item in static_queue_points), default=0
    )

    eligible_rows = [
        row
        for row in decode_rows
        if row["label"] in static_labels
        and float(row["observed_itl_ms"]) < latency_cutoff_ms
        and float(row["output_tokens"]) > 0
    ]
    aggregates: dict[tuple[tuple[bool, int, int], int], dict[str, float]] = defaultdict(
        lambda: {"points": 0.0, "tokens": 0.0, "time_ms": 0.0}
    )
    for row in eligible_rows:
        key = (_state_key(row), int(row["active_step"]))
        item = aggregates[key]
        item["points"] += 1
        item["tokens"] += float(row["output_tokens"])
        item["time_ms"] += float(row["observed_itl_ms"])

    policy_rows: list[dict[str, Any]] = []
    rates: dict[tuple[tuple[bool, int, int], int], float] = {}
    for (state, step), item in sorted(aggregates.items()):
        if item["points"] < min_points or item["time_ms"] <= 0:
            continue
        rate = item["tokens"] * 1000.0 / item["time_ms"]
        rates[(state, step)] = rate
        policy_rows.append(
            {
                "allow_partial": state[0],
                "batch_slot": state[1],
                "ctx_bucket": state[2],
                "step": step,
                "points": int(item["points"]),
                "output_tokens": item["tokens"],
                "time_ms": item["time_ms"],
                "throughput_tok_s": rate,
            }
        )
    best_step_by_state: dict[tuple[bool, int, int], int] = {}
    for state, _ in {key for key in rates}:
        candidates = [
            (rate, step)
            for (candidate_state, step), rate in rates.items()
            if candidate_state == state
        ]
        if candidates:
            best_step_by_state[state] = max(candidates)[1]
    for row in policy_rows:
        state = (
            bool(row["allow_partial"]),
            int(row["batch_slot"]),
            int(row["ctx_bucket"]),
        )
        row["selected_by_oracle"] = int(row["step"]) == best_step_by_state.get(state)

    reference_points = [
        row
        for row in decode_rows
        if row["label"] == reference_label
        and float(row["observed_itl_ms"]) < latency_cutoff_ms
        and float(row["output_tokens"]) > 0
    ]
    if not reference_points:
        raise ValueError(
            f"reference trace contains no eligible points: {reference_label}"
        )

    replay_rows: list[dict[str, Any]] = []
    reference_time_ms = 0.0
    oracle_time_ms = 0.0
    total_tokens = 0.0
    multi_candidate_tokens = 0.0
    all_tier_tokens = 0.0
    fallback_tokens = 0.0
    queued_tokens = 0.0
    max_queue_req = 0
    state_coverage: dict[tuple[bool, int, int], dict[str, Any]] = {}
    oracle_tokens_by_step: dict[int, float] = defaultdict(float)
    oracle_time_by_step_ms: dict[int, float] = defaultdict(float)
    for row in reference_points:
        state = _state_key(row)
        candidates = [
            (rates[(state, step)], step)
            for step in static_steps
            if (state, step) in rates
        ]
        output_tokens = float(row["output_tokens"])
        observed_time_ms = float(row["observed_itl_ms"])
        queue_req = int(row.get("queue_req", 0) or 0)
        if candidates:
            oracle_rate, oracle_step = max(candidates)
            point_oracle_ms = output_tokens * 1000.0 / oracle_rate
        else:
            oracle_rate = output_tokens * 1000.0 / observed_time_ms
            oracle_step = int(row["active_step"])
            point_oracle_ms = observed_time_ms
        candidate_count = len(candidates)
        total_tokens += output_tokens
        reference_time_ms += observed_time_ms
        oracle_time_ms += point_oracle_ms
        if candidate_count >= 2:
            multi_candidate_tokens += output_tokens
        if candidate_count == len(static_steps):
            all_tier_tokens += output_tokens
        if candidate_count == 0:
            fallback_tokens += output_tokens
        if queue_req > 0:
            queued_tokens += output_tokens
        max_queue_req = max(max_queue_req, queue_req)
        coverage = state_coverage.setdefault(
            state,
            {
                "allow_partial": state[0],
                "batch_slot": state[1],
                "ctx_bucket": state[2],
                "reference_points": 0,
                "reference_tokens": 0.0,
                "available_steps": [
                    step for step in static_steps if (state, step) in rates
                ],
                "missing_steps": [
                    step for step in static_steps if (state, step) not in rates
                ],
            },
        )
        coverage["reference_points"] += 1
        coverage["reference_tokens"] += output_tokens
        oracle_tokens_by_step[oracle_step] += output_tokens
        oracle_time_by_step_ms[oracle_step] += point_oracle_ms
        replay_rows.append(
            {
                "point_index": int(row["point_index"]),
                "batch_size": int(row["batch_size"]),
                "batch_slot": state[1],
                "ctx_per_req": float(row["ctx_per_req"]),
                "ctx_bucket": state[2],
                "reference_step": int(row["active_step"]),
                "oracle_step": oracle_step,
                "candidate_count": candidate_count,
                "queue_req": queue_req,
                "output_tokens": output_tokens,
                "reference_time_ms": observed_time_ms,
                "oracle_time_ms": point_oracle_ms,
                "oracle_throughput_tok_s": oracle_rate,
                "regret_ms": observed_time_ms - point_oracle_ms,
            }
        )

    if oracle_time_ms <= 0 or reference_time_ms <= 0:
        raise ValueError("replay produced non-positive time")
    oracle_speedup = reference_time_ms / oracle_time_ms
    reference_e2e_time = float(reference["generation_time_s"])
    reference_total_tokens = float(reference["total_generated_tokens"])
    reference_throughput = float(reference["output_throughput_tok_per_s"])
    oracle_e2e_time = reference_e2e_time / oracle_speedup
    target_time = reference_e2e_time - target_capture * (
        reference_e2e_time - oracle_e2e_time
    )

    dynamic_rows = [
        row
        for row in e2e_rows
        if _bool(row["dynamic"]) and _bool(row["allow_partial"]) == allow_partial
    ]
    dynamic = (
        max(dynamic_rows, key=lambda row: float(row["output_throughput_tok_per_s"]))
        if dynamic_rows
        else None
    )
    dynamic_normalized_time = None
    dynamic_time_capture = None
    speedup_gain_capture = None
    if dynamic is not None:
        dynamic_throughput = float(dynamic["output_throughput_tok_per_s"])
        dynamic_normalized_time = reference_total_tokens / dynamic_throughput
        oracle_saved = reference_e2e_time - oracle_e2e_time
        if oracle_saved > 0:
            dynamic_time_capture = (
                reference_e2e_time - dynamic_normalized_time
            ) / oracle_saved
        oracle_gain = oracle_speedup - 1.0
        if oracle_gain > 0:
            speedup_gain_capture = (
                reference_e2e_time / dynamic_normalized_time - 1.0
            ) / oracle_gain

    static_token_totals = [
        float(row["total_generated_tokens"])
        for row in e2e_rows
        if row["label"] in static_labels
    ]
    token_relative_range = (
        (max(static_token_totals) - min(static_token_totals))
        / (sum(static_token_totals) / len(static_token_totals))
        if static_token_totals
        else math.nan
    )
    candidate_coverage = multi_candidate_tokens / total_tokens
    all_tier_coverage = all_tier_tokens / total_tokens
    fallback_share = fallback_tokens / total_tokens
    queued_share = queued_tokens / total_tokens
    state_coverage_rows = sorted(
        state_coverage.values(),
        key=lambda item: (
            item["allow_partial"],
            item["batch_slot"],
            item["ctx_bucket"],
        ),
    )
    incomplete_states = [item for item in state_coverage_rows if item["missing_steps"]]
    fallback_states = [
        item for item in state_coverage_rows if not item["available_steps"]
    ]
    decision_grade_reasons = []
    if candidate_coverage < 0.95:
        decision_grade_reasons.append("multi-candidate token coverage is below 95%")
    if all_tier_coverage < 0.75:
        decision_grade_reasons.append("all-tier token coverage is below 75%")
    if token_relative_range > 0.01:
        decision_grade_reasons.append("static generated-token drift exceeds 1%")
    if static_queue_points:
        decision_grade_reasons.append("static matrix contains queued requests")
    oracle_step_token_share = {
        str(step): tokens / total_tokens
        for step, tokens in sorted(oracle_tokens_by_step.items())
    }
    oracle_step_time_share = {
        str(step): time_ms / oracle_time_ms
        for step, time_ms in sorted(oracle_time_by_step_ms.items())
    }
    summary: dict[str, Any] = {
        "method": "fluid_token_progress_replay",
        "reference_label": reference_label,
        "allow_partial": allow_partial,
        "static_steps": static_steps,
        "candidate_set_scope": (
            "full_matrix" if static_step_filter is None else "filtered_sensitivity"
        ),
        "min_points_per_state_tier": min_points,
        "latency_cutoff_ms": latency_cutoff_ms,
        "reference_e2e_time_s": reference_e2e_time,
        "reference_total_tokens": reference_total_tokens,
        "reference_throughput_tok_s": reference_throughput,
        "reference_reconstructed_time_s": reference_time_ms / 1000.0,
        "oracle_reconstructed_time_s": oracle_time_ms / 1000.0,
        "oracle_speedup": oracle_speedup,
        "oracle_gain": oracle_speedup - 1.0,
        "oracle_e2e_time_s": oracle_e2e_time,
        "oracle_throughput_tok_s": reference_total_tokens / oracle_e2e_time,
        "target_capture": target_capture,
        "target_capture_time_s": target_time,
        "target_capture_throughput_tok_s": reference_total_tokens / target_time,
        "candidate_coverage_token_share": candidate_coverage,
        "all_tier_coverage_token_share": all_tier_coverage,
        "fallback_token_share": fallback_share,
        "queued_token_share": queued_share,
        "max_queue_req": max_queue_req,
        "static_queued_point_count": len(static_queue_points),
        "static_max_queue_req": static_max_queue_req,
        "static_queued_points": static_queue_points,
        "state_coverage": state_coverage_rows,
        "incomplete_states": incomplete_states,
        "fallback_states": fallback_states,
        "oracle_step_token_share": oracle_step_token_share,
        "oracle_step_time_share": oracle_step_time_share,
        "static_total_token_relative_range": token_relative_range,
        "decision_grade": not decision_grade_reasons,
        "decision_grade_reasons": decision_grade_reasons,
        "dynamic_label": dynamic["label"] if dynamic else None,
        "dynamic_normalized_time_s": dynamic_normalized_time,
        "dynamic_time_saving_capture": dynamic_time_capture,
        "speedup_gain_capture": speedup_gain_capture,
        "assumption": (
            "reference batch/context state path is invariant in generated-token space; "
            "static bucket rates transfer across the replay"
        ),
    }
    return summary, policy_rows, replay_rows


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Decoupled Spec Fluid Oracle Replay",
        "",
        f"- Decision-grade: {summary['decision_grade']}",
        f"- Decision-grade reasons: {summary['decision_grade_reasons']}",
        f"- Reference: `{summary['reference_label']}`",
        f"- Candidate set scope: {summary['candidate_set_scope']}",
        f"- Static E2E time: {summary['reference_e2e_time_s']:.4f} s",
        f"- Oracle E2E time: {summary['oracle_e2e_time_s']:.4f} s",
        f"- Oracle speedup: {summary['oracle_speedup']:.6f}x",
        f"- Target ({summary['target_capture']:.0%} time-saving capture): "
        f"{summary['target_capture_time_s']:.4f} s",
        f"- Multi-candidate token coverage: "
        f"{summary['candidate_coverage_token_share']:.2%}",
        f"- All-tier token coverage: {summary['all_tier_coverage_token_share']:.2%}",
        f"- Fallback token share: {summary['fallback_token_share']:.2%}",
        f"- Queued token share: {summary['queued_token_share']:.2%}",
        f"- Maximum queued requests: {summary['max_queue_req']}",
        f"- Static queued point count: {summary['static_queued_point_count']}",
        f"- Static maximum queued requests: {summary['static_max_queue_req']}",
        f"- Static queued point identities: {summary['static_queued_points']}",
        f"- Incomplete state identities: {summary['incomplete_states']}",
        f"- Fallback state identities: {summary['fallback_states']}",
        f"- Oracle step token share: {summary['oracle_step_token_share']}",
    ]
    if summary["dynamic_label"] is not None:
        lines.extend(
            [
                f"- Dynamic: `{summary['dynamic_label']}`",
                f"- Dynamic oracle time-saving capture: "
                f"{summary['dynamic_time_saving_capture']:.4f}",
            ]
        )
    lines.extend(["", f"> Assumption: {summary['assumption']}", ""])
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-label")
    parser.add_argument(
        "--static-steps",
        help="Comma-separated candidate steps for an explicitly labeled sensitivity replay",
    )
    parser.add_argument("--target-capture", type=float, default=0.7)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--latency-cutoff-ms", type=float, default=100.0)
    args = parser.parse_args()
    if not 0.0 <= args.target_capture <= 1.0:
        parser.error("--target-capture must be in [0, 1]")
    if args.min_points <= 0:
        parser.error("--min-points must be positive")
    if args.latency_cutoff_ms <= 0:
        parser.error("--latency-cutoff-ms must be positive")
    if args.static_steps:
        try:
            args.static_steps = {int(value) for value in args.static_steps.split(",")}
        except ValueError:
            parser.error("--static-steps must be comma-separated integers")
    else:
        args.static_steps = None
    if args.static_steps is not None and args.output_dir is None:
        parser.error("--static-steps requires an explicit --output-dir")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.analysis_dir / "oracle_replay"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, policy_rows, replay_rows = replay(
        _read_csv(args.analysis_dir / "decode_points.csv"),
        _read_csv(args.analysis_dir / "e2e_summary.csv"),
        reference_label=args.reference_label,
        target_capture=args.target_capture,
        min_points=args.min_points,
        latency_cutoff_ms=args.latency_cutoff_ms,
        static_step_filter=args.static_steps,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "bucket_policy.csv", policy_rows)
    _write_csv(output_dir / "replay_points.csv", replay_rows)
    _write_summary(output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

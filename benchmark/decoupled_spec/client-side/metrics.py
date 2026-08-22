"""Request-level metrics and aggregate summaries."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Iterable

from common.artifacts import write_json


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latency(values: Iterable[float | None]) -> dict[str, float | None]:
    present = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.fmean(present) if present else None,
        "p50": percentile(present, 0.50),
        "p95": percentile(present, 0.95),
        "p99": percentile(present, 0.99),
    }


def _sum_position_counts(records: list[dict[str, Any]], field: str) -> list[int]:
    metric_len = max(
        (len(value) for item in records if (value := item.get(field))),
        default=0,
    )
    totals = [0] * metric_len
    for item in records:
        for position, value in enumerate(item.get(field) or []):
            totals[position] += int(value)
    return totals


def summarize(records: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    completed = [item for item in records if item.get("error") is None]
    total_output = sum(
        int(item.get("resp_len", item.get("completion_tokens", 0)))
        for item in completed
    )
    total_prompt = sum(int(item.get("prompt_len", 0)) for item in completed)
    proposed = sum(
        int(item.get("spec_num_proposed_drafts", 0) or 0) for item in completed
    )
    correct = sum(
        int(item.get("spec_num_correct_drafts", 0) or 0) for item in completed
    )
    verify_ct = sum(int(item.get("spec_verify_ct", 0) or 0) for item in completed)
    proposed_by_position = _sum_position_counts(
        completed, "spec_num_proposed_drafts_by_position"
    )
    correct_by_position = _sum_position_counts(
        completed, "spec_num_correct_drafts_by_position"
    )
    proposed_histogram = _sum_position_counts(
        completed, "spec_proposed_drafts_histogram"
    )
    metric_len = max(len(proposed_by_position), len(correct_by_position))
    proposed_by_position.extend([0] * (metric_len - len(proposed_by_position)))
    correct_by_position.extend([0] * (metric_len - len(correct_by_position)))
    occupancy_weight = sum(
        int(item.get("spec_verify_ct", 0) or 0)
        for item in completed
        if item.get("spec_draft_occupancy_rate") is not None
    )
    weighted_occupancy = sum(
        float(item["spec_draft_occupancy_rate"])
        * int(item.get("spec_verify_ct", 0) or 0)
        for item in completed
        if item.get("spec_draft_occupancy_rate") is not None
    )
    return {
        "batch_size": len(records),
        "request_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(records) - len(completed),
        "batch_elapsed_s": elapsed_s,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_output,
        "output_tokens_per_s": total_output / elapsed_s if elapsed_s > 0 else None,
        "ttft_ms": summarize_latency(item.get("ttft_ms") for item in completed),
        "tpot_ms": summarize_latency(item.get("tpot_ms") for item in completed),
        "e2e_latency_ms": summarize_latency(
            item.get("e2e_latency_ms") for item in completed
        ),
        "spec_verify_ct": verify_ct,
        "spec_num_proposed_drafts": proposed,
        "spec_num_correct_drafts": correct,
        "spec_accept_rate": correct / proposed if proposed else None,
        "spec_accept_length": total_output / verify_ct if verify_ct else None,
        "spec_proposed_draft_length": proposed / verify_ct if verify_ct else None,
        "spec_proposed_drafts_histogram": proposed_histogram,
        "spec_num_proposed_drafts_by_position": proposed_by_position,
        "spec_num_correct_drafts_by_position": correct_by_position,
        "spec_accept_rate_by_position": [
            (
                correct_at_position / proposed_at_position
                if proposed_at_position > 0
                else None
            )
            for proposed_at_position, correct_at_position in zip(
                proposed_by_position, correct_by_position
            )
        ],
        "spec_draft_occupancy_rate": (
            weighted_occupancy / occupancy_weight if occupancy_weight else None
        ),
    }


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = sorted({key for record in records for key in record})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    write_json(Path(path), summary)

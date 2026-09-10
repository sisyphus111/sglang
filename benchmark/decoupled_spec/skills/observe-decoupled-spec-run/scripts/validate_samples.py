#!/usr/bin/env python3
"""Validate observer sample quality and formal-window coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

_TAIL_HISTOGRAM_FIELDS = (
    "selected_draft_length_histogram",
    "raw_draft_tail_length_histogram",
    "consumable_draft_tail_length_histogram",
    "logical_delta_histogram",
    "pending_prefix_length_histogram",
)
_DRAFTER_TRANSPORT_FIELDS = {
    "draft_send_queue_latency_us",
    "draft_send_queue_depth_max",
}
_VERIFIER_TRANSPORT_FIELDS = {
    "draft_result_ready_to_receive_latency_us",
    "draft_receive_to_gpu_publish_enqueue_latency_us",
    "draft_gpu_publish_completion_latency_us",
    "draft_transport_one_way_latency_us",
    "gpu_publish_staging_slots_max",
    "clock_sync_valid",
    "clock_error_bound_us",
    "num_clock_sync_valid_peers",
    "num_clock_sync_invalid_peers",
}
_LATENCY_HISTOGRAM_FIELDS = {
    "draft_send_queue_latency_us",
    "draft_result_ready_to_receive_latency_us",
    "draft_receive_to_gpu_publish_enqueue_latency_us",
    "draft_gpu_publish_completion_latency_us",
    "draft_transport_one_way_latency_us",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number}: expected a JSON object")
            continue
        records.append(value)
    return records


def _is_success(record: dict[str, Any]) -> bool:
    status = record.get("status_code")
    payload = record.get("payload")
    return (
        record.get("error") is None
        and isinstance(status, int)
        and 200 <= status < 300
        and isinstance(payload, dict)
        and isinstance(payload.get("loads"), list)
        and bool(payload["loads"])
    )


def _max_gap(times: list[float]) -> float | None:
    return (
        max(right - left for left, right in zip(times, times[1:]))
        if len(times) > 1
        else None
    )


def _validate_nonnegative_int(value: Any, field: str, errors: list[str]) -> bool:
    if type(value) is not int or value < 0:
        errors.append(f"{field} must be a non-negative integer, got {value!r}")
        return False
    return True


def _validate_integer_histogram(
    value: Any,
    field: str,
    expected_count: int | None,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an IntegerHistogram mapping")
        return
    offset = value.get("offset")
    counts = value.get("counts")
    if type(offset) is not int:
        errors.append(f"{field}.offset must be an integer")
    if not isinstance(counts, list):
        errors.append(f"{field}.counts must be a list")
        return
    counts_valid = all(
        _validate_nonnegative_int(count, f"{field}.counts[{index}]", errors)
        for index, count in enumerate(counts)
    )
    underflow = value.get("underflow_count")
    overflow = value.get("overflow_count")
    underflow_valid = _validate_nonnegative_int(
        underflow, f"{field}.underflow_count", errors
    )
    overflow_valid = _validate_nonnegative_int(
        overflow, f"{field}.overflow_count", errors
    )
    if (
        counts_valid
        and underflow_valid
        and overflow_valid
        and expected_count is not None
    ):
        observed_count = sum(counts) + underflow + overflow
        if observed_count != expected_count:
            errors.append(
                f"{field} coverage mismatch: observed={observed_count}, "
                f"num_select_rows={expected_count}"
            )


def _validate_latency_histogram(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be a LatencyHistogram mapping")
        return
    count = value.get("count")
    sum_us = value.get("sum_us")
    bounds = value.get("bucket_upper_bounds_us")
    bucket_counts = value.get("bucket_counts")
    count_valid = _validate_nonnegative_int(count, f"{field}.count", errors)
    if (
        not isinstance(sum_us, (int, float))
        or isinstance(sum_us, bool)
        or not math.isfinite(float(sum_us))
        or sum_us < 0
    ):
        errors.append(f"{field}.sum_us must be finite and non-negative")
    if not isinstance(bounds, list) or any(
        not isinstance(bound, (int, float))
        or isinstance(bound, bool)
        or not math.isfinite(float(bound))
        or bound < 0
        for bound in (bounds or [])
    ):
        errors.append(f"{field}.bucket_upper_bounds_us must contain finite bounds")
        return
    if any(right <= left for left, right in zip(bounds, bounds[1:])):
        errors.append(f"{field}.bucket_upper_bounds_us must be strictly increasing")
    if not isinstance(bucket_counts, list):
        errors.append(f"{field}.bucket_counts must be a list")
        return
    bucket_counts_valid = all(
        _validate_nonnegative_int(
            bucket_count, f"{field}.bucket_counts[{index}]", errors
        )
        for index, bucket_count in enumerate(bucket_counts)
    )
    if len(bucket_counts) != len(bounds) + 1:
        errors.append(
            f"{field}.bucket_counts must contain one non-cumulative overflow bucket"
        )
    if count_valid and bucket_counts_valid and count != sum(bucket_counts):
        errors.append(f"{field}.count must equal sum(bucket_counts)")


def _validate_decoupled_spec_window(
    role: str,
    target_id: str,
    dp_rank: int,
    window_id: int,
    window: dict[str, Any],
    errors: list[str],
) -> None:
    decoupled_spec = window.get("decoupled_spec")
    if decoupled_spec is None:
        return
    prefix = f"{target_id}/dp{dp_rank}/window{window_id}.decoupled_spec"
    if not isinstance(decoupled_spec, dict):
        errors.append(f"{prefix} must be a mapping")
        return

    tail_select = decoupled_spec.get("tail_select")
    if tail_select is not None:
        if role != "verifier":
            errors.append(f"{prefix}.tail_select is verifier-only")
        if not isinstance(tail_select, dict):
            errors.append(f"{prefix}.tail_select must be a mapping")
        else:
            rows = tail_select.get("num_select_rows")
            valid_rows = tail_select.get("num_select_valid_rows")
            rows_valid = _validate_nonnegative_int(
                rows, f"{prefix}.tail_select.num_select_rows", errors
            )
            valid_rows_valid = _validate_nonnegative_int(
                valid_rows,
                f"{prefix}.tail_select.num_select_valid_rows",
                errors,
            )
            if rows_valid and valid_rows_valid and valid_rows > rows:
                errors.append(
                    f"{prefix}.tail_select.num_select_valid_rows exceeds rows"
                )
            reason_counts = tail_select.get("reason_counts")
            if not isinstance(reason_counts, dict):
                errors.append(f"{prefix}.tail_select.reason_counts must be a mapping")
            else:
                reasons_valid = all(
                    isinstance(reason, str)
                    and bool(reason)
                    and _validate_nonnegative_int(
                        count,
                        f"{prefix}.tail_select.reason_counts.{reason}",
                        errors,
                    )
                    for reason, count in reason_counts.items()
                )
                if rows_valid and reasons_valid and sum(reason_counts.values()) != rows:
                    errors.append(
                        f"{prefix}.tail_select.reason_counts must cover every row"
                    )
            for field in _TAIL_HISTOGRAM_FIELDS:
                if field not in tail_select:
                    errors.append(f"{prefix}.tail_select.{field} is missing")
                    continue
                _validate_integer_histogram(
                    tail_select[field],
                    f"{prefix}.tail_select.{field}",
                    rows if rows_valid else None,
                    errors,
                )
            selected_histogram = tail_select.get("selected_draft_length_histogram")
            if isinstance(selected_histogram, dict) and isinstance(
                selected_histogram.get("counts"), list
            ):
                num_steps = len(selected_histogram["counts"]) - 1
                if num_steps < 0 or selected_histogram.get("offset") != 0:
                    errors.append(
                        f"{prefix}.tail_select.selected_draft_length_histogram "
                        "must start at zero"
                    )
                elif num_steps >= 0:
                    expected_layouts = {
                        "raw_draft_tail_length_histogram": (-1, 2 * num_steps + 3),
                        "consumable_draft_tail_length_histogram": (
                            -1,
                            2 * num_steps + 3,
                        ),
                        "pending_prefix_length_histogram": (-1, 2 * num_steps + 3),
                        "logical_delta_histogram": (
                            -(2 * num_steps + 1),
                            4 * num_steps + 3,
                        ),
                    }
                    for field, (
                        expected_offset,
                        expected_length,
                    ) in expected_layouts.items():
                        histogram = tail_select.get(field)
                        if not isinstance(histogram, dict):
                            continue
                        if (
                            histogram.get("offset") != expected_offset
                            or len(histogram.get("counts", [])) != expected_length
                        ):
                            errors.append(
                                f"{prefix}.tail_select.{field} has an invalid "
                                f"K={num_steps} bucket layout"
                            )
            freshness_fields = (
                "num_publish_seq_initial",
                "num_publish_seq_same",
                "num_publish_seq_advance",
            )
            freshness_valid = True
            for field in (*freshness_fields, "num_protocol_errors"):
                freshness_valid &= _validate_nonnegative_int(
                    tail_select.get(field), f"{prefix}.tail_select.{field}", errors
                )
            if (
                rows_valid
                and freshness_valid
                and sum(tail_select[field] for field in freshness_fields) > rows
            ):
                errors.append(
                    f"{prefix}.tail_select publish-sequence counts exceed rows"
                )
            if (
                rows_valid
                and freshness_valid
                and tail_select["num_protocol_errors"] > rows
            ):
                errors.append(f"{prefix}.tail_select.num_protocol_errors exceeds rows")
            fast_forwards = tail_select.get("num_pending_prefix_fast_forwards")
            _validate_nonnegative_int(
                fast_forwards,
                f"{prefix}.tail_select.num_pending_prefix_fast_forwards",
                errors,
            )
            retry_fields = (
                "num_seqlock_retry_rows",
                "num_seqlock_retries",
                "max_seqlock_retries",
            )
            if any(field in tail_select for field in retry_fields):
                retries_valid = True
                for field in retry_fields:
                    retries_valid &= _validate_nonnegative_int(
                        tail_select.get(field),
                        f"{prefix}.tail_select.{field}",
                        errors,
                    )
                if retries_valid:
                    retry_rows = tail_select["num_seqlock_retry_rows"]
                    retries = tail_select["num_seqlock_retries"]
                    max_retries = tail_select["max_seqlock_retries"]
                    if rows_valid and retry_rows > rows:
                        errors.append(
                            f"{prefix}.tail_select seqlock retry rows exceed rows"
                        )
                    if retries < retry_rows or max_retries > retries:
                        errors.append(
                            f"{prefix}.tail_select seqlock retry counters disagree"
                        )
                    if (
                        bool(retry_rows) != bool(retries)
                        or bool(retry_rows) != bool(max_retries)
                    ):
                        errors.append(
                            f"{prefix}.tail_select seqlock retry max disagrees "
                            "with retry rows"
                        )

    transport = decoupled_spec.get("transport")
    if transport is None:
        return
    if not isinstance(transport, dict):
        errors.append(f"{prefix}.transport must be a mapping")
        return
    for field in ("num_draft_result_frames", "num_draft_result_tokens"):
        _validate_nonnegative_int(
            transport.get(field), f"{prefix}.transport.{field}", errors
        )
    forbidden = (
        _VERIFIER_TRANSPORT_FIELDS if role == "drafter" else _DRAFTER_TRANSPORT_FIELDS
    )
    for field in sorted(forbidden & transport.keys()):
        errors.append(f"{prefix}.transport.{field} is not owned by role={role}")
    for field in sorted(_LATENCY_HISTOGRAM_FIELDS & transport.keys()):
        _validate_latency_histogram(
            transport[field], f"{prefix}.transport.{field}", errors
        )
    for field in ("draft_send_queue_depth_max", "gpu_publish_staging_slots_max"):
        if field in transport:
            _validate_nonnegative_int(
                transport[field], f"{prefix}.transport.{field}", errors
            )
    if (
        "clock_sync_valid" in transport
        and type(transport["clock_sync_valid"]) is not bool
    ):
        errors.append(f"{prefix}.transport.clock_sync_valid must be boolean")
    valid_peers = transport.get("num_clock_sync_valid_peers")
    invalid_peers = transport.get("num_clock_sync_invalid_peers")
    peer_counts_reported = valid_peers is not None or invalid_peers is not None
    peer_counts_valid = False
    if peer_counts_reported:
        valid_peers_valid = _validate_nonnegative_int(
            valid_peers,
            f"{prefix}.transport.num_clock_sync_valid_peers",
            errors,
        )
        invalid_peers_valid = _validate_nonnegative_int(
            invalid_peers,
            f"{prefix}.transport.num_clock_sync_invalid_peers",
            errors,
        )
        peer_counts_valid = valid_peers_valid and invalid_peers_valid
        if peer_counts_valid and "clock_sync_valid" in transport:
            expected_all_valid = valid_peers > 0 and invalid_peers == 0
            if transport["clock_sync_valid"] != expected_all_valid:
                errors.append(
                    f"{prefix}.transport.clock_sync_valid disagrees with peer counts"
                )
    calibrated_histogram_has_samples = any(
        isinstance(transport.get(field), dict)
        and type(transport[field].get("count")) is int
        and transport[field]["count"] > 0
        for field in (
            "draft_transport_one_way_latency_us",
            "draft_result_ready_to_receive_latency_us",
        )
    )
    if (
        transport.get("clock_sync_valid") is True
        or (peer_counts_valid and valid_peers > 0)
        or calibrated_histogram_has_samples
    ):
        error_bound = transport.get("clock_error_bound_us")
        if (
            not isinstance(error_bound, (int, float))
            or isinstance(error_bound, bool)
            or not math.isfinite(float(error_bound))
            or error_bound < 0
        ):
            errors.append(
                f"{prefix}.transport.clock_error_bound_us must be finite when "
                "clock-synchronized samples or peers are reported"
            )


def _observed_targets(
    records: list[dict[str, Any]], required_roles: list[str], errors: list[str]
) -> dict[str, dict[str, Any]]:
    normalized = {}
    role_ranks = set()
    for record in records:
        role = record.get("role")
        target_id = record.get("target_id", role)
        rank = record.get("rank", 0)
        base_url = record.get("base_url")
        if not isinstance(target_id, str) or role not in {
            "target",
            "verifier",
            "drafter",
        }:
            errors.append(f"invalid observer target identity: {record!r}")
            continue
        if type(rank) is not int or rank < 0:
            errors.append(f"{target_id}: rank must be a non-negative integer")
            continue
        if not isinstance(base_url, str) or not base_url:
            errors.append(f"{target_id}: base_url must be a non-empty string")
            continue
        target = {
            "target_id": target_id,
            "role": role,
            "rank": rank,
            "base_url": base_url.rstrip("/"),
        }
        previous = normalized.get(target_id)
        if previous is not None:
            if previous != target:
                errors.append(f"conflicting observer target identity: {target_id}")
            continue
        if (role, rank) in role_ranks:
            errors.append(f"duplicate observer target role/rank: {role}/{rank}")
            continue
        role_ranks.add((role, rank))
        normalized[target_id] = target
    missing_roles = set(required_roles) - {
        target["role"] for target in normalized.values()
    }
    if missing_roles:
        errors.append(f"observer samples lack required roles: {sorted(missing_roles)}")
    return normalized


def _fixed_batch_decode_window(
    run_dir: Path,
    records: list[dict[str, Any]],
    start: float,
    finish: float,
    errors: list[str],
) -> tuple[float, float] | None:
    """Bound queue checks by full-BS verifier decode, excluding batch fill/drain."""
    try:
        client = _read_json(run_dir / "config.json")["client"]
        batch_size = int(client["batch"]["size"])
        with (run_dir / "client" / "requests.csv").open(newline="") as handle:
            requests = list(csv.DictReader(handle))
        if batch_size <= 0 or len(requests) != batch_size:
            raise ValueError("request count does not match the configured batch size")
        # The Client launch precedes individual request starts. This bound is
        # conservative; it must not extend into the first request's drain phase.
        latencies = [float(r["e2e_latency_s"]) for r in requests]
        if not all(math.isfinite(value) and value > 0 for value in latencies):
            raise ValueError("request latencies must be finite and positive")
        finish = min(finish, start + min(latencies))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot determine full-BS decode window: {exc}")
        return None

    windows: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    selected_target = client.get("server", {}).get("target_id")
    for record in records:
        if record.get("role") not in {"target", "verifier"} or not _is_success(record):
            continue
        if selected_target is not None and record["target_id"] != selected_target:
            continue
        for load in record["payload"]["loads"]:
            if not isinstance(load, dict):
                continue
            key = (record["target_id"], int(load.get("dp_rank", 0)))
            for window in load.get("decode_metrics_windows") or []:
                if (
                    not isinstance(window, dict)
                    or type(window.get("window_id")) is not int
                ):
                    continue
                end = window.get("end_time")
                if isinstance(end, (int, float)) and start <= end < finish:
                    windows.setdefault(key, {})[window["window_id"]] = window
    bounds = []
    for engine_windows in windows.values():
        full_ends = []
        for window in sorted(engine_windows.values(), key=lambda w: w["window_id"]):
            iters = window.get("num_decode_iters", 0)
            full = (
                type(iters) is int
                and iters > 0
                and window.get("num_decode_rows") == batch_size * iters
            )
            if full:
                full_ends.append(float(window["end_time"]))
            elif full_ends:
                break
        # The first full window has no trustworthy start boundary. Use its end
        # as the start and retain only subsequent complete full-batch windows.
        if len(full_ends) >= 2:
            bounds.append((full_ends[0], full_ends[-1]))
    if len(bounds) != 1:
        errors.append(
            "cannot determine one full-BS decode window before the first exit: "
            f"found {len(bounds)} eligible engines"
        )
        return None
    return bounds[0]


def validate_samples(
    run_dir: Path,
    required_roles: list[str],
    require_formal_window: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    samples_path = run_dir / "observer" / "samples.jsonl"
    formal_path = run_dir / "observer" / "bench_timeline.json"

    for path in (samples_path,):
        if not path.is_file():
            errors.append(f"missing required file: {path}")
    if errors:
        return {
            "ok": False,
            "run_dir": str(run_dir),
            "errors": errors,
            "warnings": warnings,
            "roles": {},
        }

    records = _read_jsonl(samples_path, errors)
    expected_targets = _observed_targets(records, required_roles, errors)
    for record in records:
        role = record.get("role")
        target_id = str(record.get("target_id", role))
        expected = expected_targets.get(target_id)
        if expected is None:
            errors.append(f"sample references unknown target_id={target_id!r}")
            continue
        if role != expected["role"]:
            errors.append(
                f"{target_id}: sample role mismatch: "
                f"observed={role!r}, expected={expected['role']!r}"
            )
        for field in ("rank", "base_url"):
            observed = record.get(field)
            if field == "base_url" and isinstance(observed, str):
                observed = observed.rstrip("/")
            if observed != expected[field]:
                errors.append(
                    f"{target_id}: sample {field} mismatch: "
                    f"observed={observed!r}, expected={expected[field]!r}"
                )
    interval_values = {
        float(record["interval_s"])
        for record in records
        if isinstance(record.get("interval_s"), (int, float))
    }
    if len(interval_values) != 1:
        errors.append(
            f"observer samples have inconsistent interval_s: {interval_values}"
        )
    interval_s = next(iter(interval_values), 0.0)
    if interval_s <= 0:
        errors.append("observer interval_s must be positive")
    started_values = {
        float(record["observer_started_wall_time"])
        for record in records
        if isinstance(record.get("observer_started_wall_time"), (int, float))
    }
    if len(started_values) != 1:
        errors.append(
            "observer samples have inconsistent observer_started_wall_time: "
            f"{started_values}"
        )
    observer_started_at = next(iter(started_values), None)

    formal: dict[str, Any] | None = None
    if formal_path.is_file():
        try:
            formal = _read_json(formal_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    elif require_formal_window:
        errors.append(f"missing required file: {formal_path}")

    formal_start = formal.get("client_started_wall_time") if formal else None
    formal_finish = formal.get("client_finished_wall_time") if formal else None
    if formal is not None:
        timeline_fields = {
            "observer_started_wall_time",
            "client_started_wall_time",
            "client_finished_wall_time",
            "observer_finished_wall_time",
            "observer_elapsed_s",
        }
        if set(formal) != timeline_fields:
            errors.append(
                "benchmark timeline fields do not match the fixed contract: "
                f"expected={sorted(timeline_fields)}, observed={sorted(formal)}"
            )
        if not isinstance(formal_start, (int, float)) or not isinstance(
            formal_finish, (int, float)
        ):
            errors.append("benchmark timeline lacks numeric client start/finish times")
            formal_start = formal_finish = None
        elif formal_finish < formal_start:
            errors.append("benchmark client window finishes before it starts")
            formal_start = formal_finish = None
        boundary_fields = (
            "observer_started_wall_time",
            "client_started_wall_time",
            "client_finished_wall_time",
            "observer_finished_wall_time",
        )
        boundaries = [formal.get(field) for field in boundary_fields]
        if not all(isinstance(value, (int, float)) for value in boundaries):
            errors.append("benchmark timeline lacks numeric wall-time boundaries")
        elif boundaries != sorted(boundaries):
            errors.append("benchmark timeline boundaries are not correctly ordered")
        observer_elapsed_s = formal.get("observer_elapsed_s")
        if not isinstance(observer_elapsed_s, (int, float)) or observer_elapsed_s < 0:
            errors.append("benchmark timeline has invalid observer_elapsed_s")
        elif all(
            isinstance(value, (int, float)) for value in boundaries
        ) and not math.isclose(
            observer_elapsed_s,
            boundaries[-1] - boundaries[0],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            errors.append("benchmark timeline observer_elapsed_s is inconsistent")

    decode_queue_window = None
    if formal_start is not None and formal_finish is not None:
        decode_queue_window = _fixed_batch_decode_window(
            run_dir, records, formal_start, formal_finish, errors
        )
    role_reports: dict[str, Any] = {}
    for role in required_roles:
        role_records = [record for record in records if record.get("role") == role]
        successful = [record for record in role_records if _is_success(record)]
        times = sorted(
            float(record["collected_wall_time"])
            for record in successful
            if isinstance(record.get("collected_wall_time"), (int, float))
        )
        report: dict[str, Any] = {
            "record_count": len(role_records),
            "success_count": len(successful),
            "error_count": len(role_records) - len(successful),
            "first_success_wall_time": times[0] if times else None,
            "last_success_wall_time": times[-1] if times else None,
            "max_success_gap_s": _max_gap(times),
        }
        formal_waiting_samples = []
        decode_queue_sample_count = 0
        decode_queue_target_counts: dict[str, int] = {}
        if decode_queue_window is not None:
            for record in successful:
                collected_at = record.get("collected_wall_time")
                if not isinstance(collected_at, (int, float)):
                    continue
                for load in record["payload"]["loads"]:
                    if not isinstance(load, dict):
                        continue
                    sampled_at = load.get("timestamp", collected_at)
                    if not isinstance(sampled_at, (int, float)) or not (
                        decode_queue_window[0] < sampled_at <= decode_queue_window[1]
                    ):
                        continue
                    decode_queue_sample_count += 1
                    target_id = str(record.get("target_id", role))
                    decode_queue_target_counts[target_id] = (
                        decode_queue_target_counts.get(target_id, 0) + 1
                    )
                    waiting = load.get("num_waiting_reqs")
                    if type(waiting) is not int or waiting < 0:
                        errors.append(
                            f"{role}: full-BS decode load sample lacks a valid "
                            f"num_waiting_reqs: sample_id={record.get('sample_id')!r} "
                            f"value={waiting!r}"
                        )
                        continue
                    if waiting > 0:
                        formal_waiting_samples.append(
                            {
                                "target_id": str(record.get("target_id", role)),
                                "sample_id": record.get("sample_id"),
                                "dp_rank": int(load.get("dp_rank", 0)),
                                "collected_wall_time": float(collected_at),
                                "sampled_wall_time": float(sampled_at),
                                "num_waiting_reqs": waiting,
                            }
                        )
        report["decode_queue_sample_count"] = decode_queue_sample_count
        report["formal_waiting_sample_count"] = len(formal_waiting_samples)
        report["max_waiting_reqs_in_formal_window"] = max(
            (sample["num_waiting_reqs"] for sample in formal_waiting_samples),
            default=0,
        )
        report["formal_waiting_samples"] = formal_waiting_samples
        if formal_waiting_samples:
            errors.append(
                f"{role}: waiting requests observed inside the full-BS decode window: "
                f"max={report['max_waiting_reqs_in_formal_window']} "
                f"sample_count={len(formal_waiting_samples)}"
            )
        decode_windows: dict[tuple[str, int, int], dict[str, Any]] = {}
        for record in successful:
            target_id = str(record.get("target_id", role))
            for load in record["payload"]["loads"]:
                if not isinstance(load, dict):
                    continue
                dp_rank = int(load.get("dp_rank", 0))
                for window in load.get("decode_metrics_windows") or []:
                    if not isinstance(window, dict) or not isinstance(
                        window.get("window_id"), int
                    ):
                        errors.append(
                            f"{role}: invalid decode_metrics_windows entry: {window!r}"
                        )
                        continue
                    if isinstance(observer_started_at, (int, float)) and float(
                        window.get("end_time", 0)
                    ) < float(observer_started_at):
                        continue
                    key = (target_id, dp_rank, int(window["window_id"]))
                    previous = decode_windows.get(key)
                    if previous is not None and previous != window:
                        errors.append(
                            f"{role}: conflicting decode metrics window {key}"
                        )
                    decode_windows[key] = window
        for (target_id, dp_rank, window_id), window in decode_windows.items():
            _validate_decoupled_spec_window(
                role,
                target_id,
                dp_rank,
                window_id,
                window,
                errors,
            )
        report["decode_metrics_window_count"] = len(decode_windows)
        report["decoupled_spec_window_count"] = sum(
            isinstance(window.get("decoupled_spec"), dict)
            for window in decode_windows.values()
        )
        report["tail_select_window_count"] = sum(
            isinstance(window.get("decoupled_spec", {}).get("tail_select"), dict)
            for window in decode_windows.values()
            if isinstance(window.get("decoupled_spec"), dict)
        )
        report["transport_window_count"] = sum(
            isinstance(window.get("decoupled_spec", {}).get("transport"), dict)
            for window in decode_windows.values()
            if isinstance(window.get("decoupled_spec"), dict)
        )
        decode_window_gaps: dict[str, dict[int, list[int]]] = {}
        for target_id in sorted({key[0] for key in decode_windows}):
            target_gaps = {}
            for dp_rank in sorted(
                {key[1] for key in decode_windows if key[0] == target_id}
            ):
                ids = sorted(
                    key[2]
                    for key in decode_windows
                    if key[0] == target_id and key[1] == dp_rank
                )
                missing = [
                    window_id
                    for left, right in zip(ids, ids[1:])
                    for window_id in range(left + 1, right)
                ]
                if missing:
                    target_gaps[dp_rank] = missing
                    errors.append(
                        f"{target_id}: missing decode metrics windows for "
                        f"dp_rank={dp_rank}: {missing}"
                    )
            if target_gaps:
                decode_window_gaps[target_id] = target_gaps
        report["decode_metrics_window_gaps"] = decode_window_gaps

        target_reports = {}
        target_ids = sorted(
            target_id
            for target_id, target in expected_targets.items()
            if target["role"] == role
        )
        for target_id in target_ids:
            target_successful = [
                record
                for record in successful
                if str(record.get("target_id", role)) == target_id
            ]
            target_times = sorted(
                float(record["collected_wall_time"])
                for record in target_successful
                if isinstance(record.get("collected_wall_time"), (int, float))
            )
            target_window_count = sum(key[0] == target_id for key in decode_windows)
            queue_sample_count = decode_queue_target_counts.get(target_id, 0)
            if decode_queue_window is not None and queue_sample_count == 0:
                errors.append(f"{target_id}: no queue sample inside the full-BS decode window")
            target_report = {
                "success_count": len(target_successful),
                "error_count": sum(
                    str(record.get("target_id", role)) == target_id
                    and not _is_success(record)
                    for record in role_records
                ),
                "first_success_wall_time": target_times[0] if target_times else None,
                "last_success_wall_time": target_times[-1] if target_times else None,
                "max_success_gap_s": _max_gap(target_times),
                "decode_metrics_window_count": target_window_count,
                "decode_queue_sample_count": queue_sample_count,
            }
            if not target_successful:
                errors.append(f"{target_id}: no successful /v1/loads samples")
            if formal_start is not None and formal_finish is not None and target_times:
                before = sum(value <= formal_start for value in target_times)
                inside = sum(
                    formal_start <= value <= formal_finish for value in target_times
                )
                after = sum(value >= formal_finish for value in target_times)
                target_report.update(
                    {
                        "before_or_at_formal_count": before,
                        "inside_formal_count": inside,
                        "after_or_at_formal_count": after,
                    }
                )
                if before == 0:
                    errors.append(f"{target_id}: no baseline sample before formal")
                if after == 0:
                    errors.append(f"{target_id}: no trailing sample after formal")
                if formal_finish - formal_start >= interval_s and inside == 0:
                    errors.append(f"{target_id}: no sample inside formal window")
            target_gap = target_report["max_success_gap_s"]
            if (
                target_gap is not None
                and interval_s > 0
                and target_gap > 2.5 * interval_s
            ):
                warnings.append(
                    f"{target_id}: maximum successful-sample gap "
                    f"{target_gap:.3f}s exceeds 2.5x interval_s"
                )
            target_reports[target_id] = target_report
        report["targets"] = target_reports
        if not successful:
            errors.append(f"{role}: no successful /v1/loads samples")
        if formal_start is not None and formal_finish is not None and times:
            before = sum(value <= formal_start for value in times)
            inside = sum(formal_start <= value <= formal_finish for value in times)
            after = sum(value >= formal_finish for value in times)
            report.update(
                {
                    "before_or_at_formal_count": before,
                    "inside_formal_count": inside,
                    "after_or_at_formal_count": after,
                }
            )
            if before == 0:
                errors.append(
                    f"{role}: no successful baseline sample before the formal window"
                )
            if after == 0:
                errors.append(
                    f"{role}: no successful trailing sample after the formal window"
                )
            if formal_finish - formal_start >= interval_s and inside == 0:
                errors.append(f"{role}: no successful sample inside the formal window")
        gap = report["max_success_gap_s"]
        if gap is not None and interval_s > 0 and gap > 2.5 * interval_s:
            warnings.append(
                f"{role}: maximum successful-sample gap {gap:.3f}s exceeds 2.5x interval_s"
            )
        role_reports[role] = report

    unknown_roles = sorted(
        {
            str(record.get("role"))
            for record in records
            if record.get("role") not in required_roles
        }
    )
    if unknown_roles:
        warnings.append(f"samples contain unrequested roles: {unknown_roles}")

    targets_by_sample = {}
    for record in records:
        targets_by_sample.setdefault(record.get("sample_id"), set()).add(
            str(record.get("target_id", record.get("role")))
        )
    expected_target_ids = set(expected_targets)
    for sample_id, target_ids in sorted(targets_by_sample.items()):
        if target_ids != expected_target_ids:
            errors.append(
                f"observer sample round {sample_id!r} has targets "
                f"{sorted(target_ids)}, expected {sorted(expected_target_ids)}"
            )

    return {
        "ok": not errors,
        "run_dir": str(run_dir),
        "interval_s": interval_s,
        "formal_window": formal,
        "decode_queue_window": decode_queue_window,
        "record_count": len(records),
        "roles": role_reports,
        "errors": errors,
        "warnings": warnings,
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _default_roles(run_dir: Path) -> list[str]:
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = _read_json(config_path)
        server = config.get("server")
        if isinstance(server, dict) and server.get("deployment") == "coupled_spec":
            return ["target"]
    return ["verifier", "drafter"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        choices=("target", "verifier", "drafter"),
        help="Required role; repeat to override the default pair.",
    )
    parser.add_argument(
        "--require-formal-window",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    report = validate_samples(
        run_dir,
        args.roles or _default_roles(run_dir),
        args.require_formal_window,
    )
    if args.output:
        _write_json(Path(args.output).expanduser(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

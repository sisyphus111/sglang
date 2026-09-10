"""Periodically collect read-only HTTP state from verifier and drafter servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_io import require_run_dir


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


_TAIL_HISTOGRAM_FIELDS = (
    "selected_draft_length_histogram",
    "raw_draft_tail_length_histogram",
    "consumable_draft_tail_length_histogram",
    "logical_delta_histogram",
    "pending_prefix_length_histogram",
)
_TRANSPORT_LATENCY_FIELDS = (
    "draft_send_queue_latency_us",
    "draft_result_ready_to_receive_latency_us",
    "draft_receive_to_gpu_publish_enqueue_latency_us",
    "draft_gpu_publish_completion_latency_us",
    "draft_transport_one_way_latency_us",
)


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _merge_integer_histograms(
    values: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    """Merge exact integer-bin counts; bucket positions must agree."""
    offset: int | None = None
    merged_counts: list[int] | None = None
    underflow_count = 0
    overflow_count = 0
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a mapping")
        current_offset = value.get("offset")
        counts = value.get("counts")
        if type(current_offset) is not int or not isinstance(counts, list):
            raise ValueError(f"{field} must contain integer offset and list counts")
        current_counts = [
            _nonnegative_int(item, f"{field}.counts[{index}]")
            for index, item in enumerate(counts)
        ]
        if offset is None:
            offset = current_offset
            merged_counts = [0] * len(current_counts)
        elif current_offset != offset or len(current_counts) != len(merged_counts):
            raise ValueError(f"{field} bucket layout changed across decode windows")
        assert merged_counts is not None
        for index, count in enumerate(current_counts):
            merged_counts[index] += count
        underflow_count += _nonnegative_int(
            value.get("underflow_count"), f"{field}.underflow_count"
        )
        overflow_count += _nonnegative_int(
            value.get("overflow_count"), f"{field}.overflow_count"
        )
    assert offset is not None and merged_counts is not None
    return {
        "offset": offset,
        "counts": merged_counts,
        "underflow_count": underflow_count,
        "overflow_count": overflow_count,
    }


def _latency_histogram_quantile(
    histogram: dict[str, Any], quantile: float
) -> float | None:
    """Return the finite bucket upper bound, or None for an overflow quantile."""
    count = int(histogram["count"])
    if count <= 0:
        return None
    rank = max(1, math.ceil(count * quantile))
    cumulative = 0
    bounds = histogram["bucket_upper_bounds_us"]
    for index, bucket_count in enumerate(histogram["bucket_counts"]):
        cumulative += int(bucket_count)
        if cumulative >= rank:
            return float(bounds[index]) if index < len(bounds) else None
    raise ValueError("latency histogram count disagrees with bucket counts")


def _merge_latency_histograms(
    values: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    """Merge non-cumulative latency buckets and derive aggregate quantiles."""
    bounds: list[float] | None = None
    bucket_counts: list[int] | None = None
    count = 0
    sum_us = 0.0
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a mapping")
        current_bounds_raw = value.get("bucket_upper_bounds_us")
        current_counts_raw = value.get("bucket_counts")
        if not isinstance(current_bounds_raw, list) or not isinstance(
            current_counts_raw, list
        ):
            raise ValueError(
                f"{field} must contain bucket_upper_bounds_us and bucket_counts"
            )
        current_bounds = [float(item) for item in current_bounds_raw]
        if any(not math.isfinite(item) or item < 0 for item in current_bounds) or any(
            right <= left for left, right in zip(current_bounds, current_bounds[1:])
        ):
            raise ValueError(f"{field} bounds must be finite and increasing")
        current_counts = [
            _nonnegative_int(item, f"{field}.bucket_counts[{index}]")
            for index, item in enumerate(current_counts_raw)
        ]
        if len(current_counts) != len(current_bounds) + 1:
            raise ValueError(f"{field} must have one non-cumulative overflow bucket")
        current_count = _nonnegative_int(value.get("count"), f"{field}.count")
        if current_count != sum(current_counts):
            raise ValueError(f"{field}.count must equal sum(bucket_counts)")
        current_sum_us = float(value.get("sum_us"))
        if not math.isfinite(current_sum_us) or current_sum_us < 0:
            raise ValueError(f"{field}.sum_us must be finite and non-negative")
        if bounds is None:
            bounds = current_bounds
            bucket_counts = [0] * len(current_counts)
        elif current_bounds != bounds:
            raise ValueError(f"{field} bucket bounds changed across decode windows")
        assert bucket_counts is not None
        for index, bucket_count in enumerate(current_counts):
            bucket_counts[index] += bucket_count
        count += current_count
        sum_us += current_sum_us
    assert bounds is not None and bucket_counts is not None
    merged = {
        "count": count,
        "sum_us": sum_us,
        "bucket_upper_bounds_us": bounds,
        "bucket_counts": bucket_counts,
    }
    merged.update(
        {
            "mean_us": sum_us / count if count else None,
            "p50_us": _latency_histogram_quantile(merged, 0.50),
            "p95_us": _latency_histogram_quantile(merged, 0.95),
        }
    )
    return merged


def _summarize_decoupled_spec_entries(
    entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    sections = [
        entry["window"].get("decoupled_spec")
        for entry in entries
        if isinstance(entry["window"].get("decoupled_spec"), dict)
    ]
    if not sections:
        return None

    tail_values = [
        section["tail_select"]
        for section in sections
        if isinstance(section.get("tail_select"), dict)
    ]
    tail_summary = None
    if tail_values:
        tail_summary = {
            field: sum(
                _nonnegative_int(value.get(field, 0), f"tail_select.{field}")
                for value in tail_values
            )
            for field in (
                "num_select_rows",
                "num_select_valid_rows",
                "num_publish_seq_initial",
                "num_publish_seq_same",
                "num_publish_seq_advance",
                "num_pending_prefix_fast_forwards",
                "num_protocol_errors",
                "num_seqlock_retry_rows",
                "num_seqlock_retries",
            )
        }
        tail_summary["max_seqlock_retries"] = max(
            (
                _nonnegative_int(
                    value.get("max_seqlock_retries", 0),
                    "tail_select.max_seqlock_retries",
                )
                for value in tail_values
            ),
            default=0,
        )
        reason_counts: dict[str, int] = {}
        for value in tail_values:
            current = value.get("reason_counts")
            if not isinstance(current, dict):
                raise ValueError("tail_select.reason_counts must be a mapping")
            for reason, count in current.items():
                if not isinstance(reason, str) or not reason:
                    raise ValueError(
                        "tail_select reason names must be non-empty strings"
                    )
                reason_counts[reason] = reason_counts.get(reason, 0) + _nonnegative_int(
                    count, f"tail_select.reason_counts.{reason}"
                )
        tail_summary["reason_counts"] = dict(sorted(reason_counts.items()))
        for field in _TAIL_HISTOGRAM_FIELDS:
            histograms = [value[field] for value in tail_values if field in value]
            if histograms:
                tail_summary[field] = _merge_integer_histograms(histograms, field)
        rows = tail_summary["num_select_rows"]
        tail_summary["valid_row_rate"] = (
            tail_summary["num_select_valid_rows"] / rows if rows else None
        )

    transport_values = [
        section["transport"]
        for section in sections
        if isinstance(section.get("transport"), dict)
    ]
    transport_summary = None
    if transport_values:
        transport_summary = {
            field: sum(
                _nonnegative_int(value.get(field, 0), f"transport.{field}")
                for value in transport_values
            )
            for field in ("num_draft_result_frames", "num_draft_result_tokens")
        }
        for field in ("draft_send_queue_depth_max", "gpu_publish_staging_slots_max"):
            values = [
                _nonnegative_int(value[field], f"transport.{field}")
                for value in transport_values
                if field in value
            ]
            if values:
                transport_summary[field] = max(values)
        for field in _TRANSPORT_LATENCY_FIELDS:
            histograms = [
                value[field] for value in transport_values if value.get(field)
            ]
            if histograms:
                transport_summary[field] = _merge_latency_histograms(histograms, field)
        peer_counts = []
        for value in transport_values:
            valid_peers = value.get("num_clock_sync_valid_peers")
            invalid_peers = value.get("num_clock_sync_invalid_peers")
            if valid_peers is None and invalid_peers is None:
                continue
            peer_counts.append(
                (
                    _nonnegative_int(
                        valid_peers, "transport.num_clock_sync_valid_peers"
                    ),
                    _nonnegative_int(
                        invalid_peers, "transport.num_clock_sync_invalid_peers"
                    ),
                )
            )
        if peer_counts:
            valid_counts = [value[0] for value in peer_counts]
            invalid_counts = [value[1] for value in peer_counts]
            transport_summary["clock_sync_peers"] = {
                "window_count": len(peer_counts),
                "valid_min": min(valid_counts),
                "valid_max": max(valid_counts),
                "invalid_min": min(invalid_counts),
                "invalid_max": max(invalid_counts),
                "all_valid_window_count": sum(
                    valid > 0 and invalid == 0 for valid, invalid in peer_counts
                ),
            }
        sync_values = [
            value["clock_sync_valid"]
            for value in transport_values
            if "clock_sync_valid" in value
        ]
        if sync_values:
            if any(type(item) is not bool for item in sync_values):
                raise ValueError("transport.clock_sync_valid must be boolean")
            transport_summary["clock_sync_valid"] = all(sync_values)
            transport_summary["clock_sync_valid_window_count"] = sum(sync_values)
            transport_summary["clock_sync_invalid_window_count"] = len(
                sync_values
            ) - sum(sync_values)
        error_bounds = [
            float(value["clock_error_bound_us"])
            for value in transport_values
            if value.get("clock_error_bound_us") is not None
        ]
        if error_bounds:
            if any(not math.isfinite(item) or item < 0 for item in error_bounds):
                raise ValueError(
                    "transport.clock_error_bound_us must be finite and non-negative"
                )
            transport_summary["clock_error_bound_us"] = max(error_bounds)

    adaptive_values = [
        section["adaptive_verify"]
        for section in sections
        if isinstance(section.get("adaptive_verify"), dict)
    ]
    adaptive_summary = None
    if adaptive_values:
        latest = max(
            adaptive_values,
            key=lambda value: (
                int(value.get("round_count", 0)),
                int(value.get("decision_seq", 0)),
            ),
        )
        adaptive_summary = dict(latest)
        adaptive_summary["window_reevaluation_count"] = sum(
            _nonnegative_int(
                value.get("window_reevaluation_count", 0),
                "adaptive_verify.window_reevaluation_count",
            )
            for value in adaptive_values
        )
        adaptive_summary["window_switch_count"] = sum(
            _nonnegative_int(
                value.get("window_switch_count", 0),
                "adaptive_verify.window_switch_count",
            )
            for value in adaptive_values
        )
        residency_width = max(
            len(value.get("window_step_residency", [])) for value in adaptive_values
        )
        adaptive_summary["window_step_residency"] = [
            sum(
                int(value.get("window_step_residency", [])[step])
                for value in adaptive_values
                if step < len(value.get("window_step_residency", []))
            )
            for step in range(residency_width)
        ]

    return {
        "tail_select": tail_summary,
        "transport": transport_summary,
        "adaptive_verify": adaptive_summary,
    }


def _update_decode_metric_windows(
    record: dict[str, Any],
    windows: dict[tuple[str, int, int], dict[str, Any]],
    min_end_time: float,
) -> None:
    """Deduplicate one HTTP sample's bounded history without projecting it."""
    if record.get("error") is not None:
        return
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return
    target_id = str(record["target_id"])
    for load in payload.get("loads", []):
        if not isinstance(load, dict):
            continue
        dp_rank = int(load.get("dp_rank", 0))
        for window in load.get("decode_metrics_windows") or []:
            if not isinstance(window, dict) or not isinstance(
                window.get("window_id"), int
            ):
                raise ValueError(
                    f"{target_id}: invalid decode_metrics_windows entry: {window!r}"
                )
            if float(window["end_time"]) < min_end_time:
                continue
            key = (target_id, dp_rank, int(window["window_id"]))
            entry = {
                "target_id": target_id,
                "role": str(record["role"]),
                "rank": int(record["rank"]),
                "base_url": str(record["base_url"]),
                "dp_rank": dp_rank,
                "window": window,
            }
            previous = windows.get(key)
            if previous is not None and previous != entry:
                raise ValueError(f"conflicting decode metrics window: {key}")
            windows[key] = entry


def _summarize_decode_metric_entries(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not entries:
        return {
            "window_count": 0,
            "dp_ranks": [],
            "first_window_end_time": None,
            "last_window_end_time": None,
            "num_decode_iters": 0,
            "scheduler_cycle_ms": {
                "mean": None,
                "min": None,
                "p50": None,
                "p95": None,
                "max": None,
            },
            "mean_batch_size": None,
            "mean_context_length": None,
            "valid_draft_length": None,
            "accept_length": None,
            "decoupled_spec": None,
        }
    num_decode_iters = sum(
        int(entry["window"]["num_decode_iters"]) for entry in entries
    )
    num_decode_rows = sum(int(entry["window"]["num_decode_rows"]) for entry in entries)
    sum_context_lens = sum(
        int(entry["window"]["sum_context_lens"]) for entry in entries
    )
    num_verify_rows = sum(int(entry["window"]["num_verify_rows"]) for entry in entries)
    num_accept_tokens = sum(
        int(entry["window"]["num_accept_tokens"]) for entry in entries
    )
    num_proposed_drafts = sum(
        int(entry["window"]["num_proposed_drafts"]) for entry in entries
    )
    cycle_values = [float(entry["window"]["iter_latency_ms"]) for entry in entries]
    weighted_cycle_ms = (
        sum(
            float(entry["window"]["iter_latency_ms"])
            * int(entry["window"]["num_decode_iters"])
            for entry in entries
        )
        / num_decode_iters
        if num_decode_iters > 0
        else None
    )
    return {
        "window_count": len(entries),
        "dp_ranks": sorted({int(entry["dp_rank"]) for entry in entries}),
        "first_window_end_time": min(
            float(entry["window"]["end_time"]) for entry in entries
        ),
        "last_window_end_time": max(
            float(entry["window"]["end_time"]) for entry in entries
        ),
        "num_decode_iters": num_decode_iters,
        "scheduler_cycle_ms": {
            "mean": weighted_cycle_ms,
            "min": min(cycle_values),
            "p50": _nearest_rank(cycle_values, 0.50),
            "p95": _nearest_rank(cycle_values, 0.95),
            "max": max(cycle_values),
        },
        "mean_batch_size": (
            num_decode_rows / num_decode_iters if num_decode_iters > 0 else None
        ),
        "mean_context_length": (
            sum_context_lens / num_decode_rows if num_decode_rows > 0 else None
        ),
        "valid_draft_length": (
            num_proposed_drafts / num_verify_rows if num_verify_rows > 0 else None
        ),
        "accept_length": (
            num_accept_tokens / num_verify_rows if num_verify_rows > 0 else None
        ),
        "decoupled_spec": _summarize_decoupled_spec_entries(entries),
    }


def _summarize_decode_metric_windows(
    windows: dict[tuple[str, int, int], dict[str, Any]],
    targets: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_target: dict[str, list[dict[str, Any]]] = {
        target["target_id"]: [] for target in targets
    }
    by_role: dict[str, list[dict[str, Any]]] = {
        target["role"]: [] for target in targets
    }
    target_metadata = {target["target_id"]: target for target in targets}
    for entry in windows.values():
        by_target.setdefault(entry["target_id"], []).append(entry)
        by_role.setdefault(entry["role"], []).append(entry)

    target_summaries = {}
    for target_id, entries in sorted(by_target.items()):
        summary = _summarize_decode_metric_entries(entries)
        metadata = target_metadata[target_id]
        summary.update(
            {
                "target_id": target_id,
                "role": metadata["role"],
                "rank": metadata["rank"],
                "base_url": metadata["base_url"],
            }
        )
        target_summaries[target_id] = summary

    role_summaries = {}
    for role, entries in sorted(by_role.items()):
        summary = _summarize_decode_metric_entries(entries)
        summary.update(
            {
                "target_ids": sorted(
                    target["target_id"] for target in targets if target["role"] == role
                ),
                "engine_ranks": sorted(
                    target["rank"] for target in targets if target["role"] == role
                ),
            }
        )
        role_summaries[role] = summary
    return target_summaries, role_summaries


def _load_ready_manifest_targets(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("server manifest must contain a JSON object")
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("server manifest schema_version must be 1")
    if manifest.get("state") != "ready":
        raise ValueError(
            f"server manifest is not ready: state={manifest.get('state')!r}"
        )
    engines = manifest.get("engines")
    if not isinstance(engines, list) or not engines:
        raise ValueError("ready server manifest must contain non-empty engines")

    targets = {}
    role_ranks = set()
    for index, engine in enumerate(engines):
        if not isinstance(engine, dict):
            raise ValueError(f"server manifest engines[{index}] must be a mapping")
        target_id = engine.get("engine_id")
        role = engine.get("role")
        rank = engine.get("rank")
        base_url = engine.get("http_url")
        if not isinstance(target_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", target_id
        ):
            raise ValueError(f"server manifest engines[{index}].engine_id is invalid")
        if target_id in targets:
            raise ValueError(f"duplicate server manifest engine_id: {target_id}")
        if role not in {"target", "verifier", "drafter"}:
            raise ValueError(f"{target_id}: role must be target, verifier, or drafter")
        if type(rank) is not int or rank < 0:
            raise ValueError(f"{target_id}: rank must be a non-negative integer")
        if (role, rank) in role_ranks:
            raise ValueError(f"duplicate server manifest role/rank: {role}/{rank}")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(f"{target_id}: http_url must be a non-empty string")
        role_ranks.add((role, rank))
        targets[target_id] = {
            "target_id": target_id,
            "role": role,
            "rank": rank,
            "base_url": base_url.rstrip("/"),
        }
    return targets


def _normalize_targets(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("targets")
    if not isinstance(configured, dict) or not configured:
        raise ValueError("targets must be a non-empty mapping")
    targets = []
    role_ranks = set()
    for key, target in configured.items():
        if not isinstance(key, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", key
        ):
            raise ValueError(f"invalid observer target id: {key!r}")
        if not isinstance(target, dict):
            raise ValueError(f"targets.{key} must be a mapping")
        target_id = target.get("target_id", key)
        role = target.get(
            "role", key if key in {"target", "verifier", "drafter"} else None
        )
        rank = target.get("rank", 0)
        base_url = target.get("base_url")
        if target_id != key:
            raise ValueError(f"targets.{key}.target_id must equal its mapping key")
        if role not in {"target", "verifier", "drafter"}:
            raise ValueError(
                f"targets.{key}.role must be target, verifier, or drafter"
            )
        if type(rank) is not int or rank < 0:
            raise ValueError(f"targets.{key}.rank must be a non-negative integer")
        if (role, rank) in role_ranks:
            raise ValueError(f"duplicate observer target role/rank: {role}/{rank}")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError(f"targets.{key}.base_url is required")
        role_ranks.add((role, rank))
        targets.append(
            {
                "target_id": target_id,
                "role": role,
                "rank": rank,
                "base_url": base_url.rstrip("/"),
            }
        )
    return sorted(targets, key=lambda target: (target["role"], target["rank"]))


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose the observer's operational controls as typed overrides."""
    parser.add_argument("--target-url")
    parser.add_argument("--verifier-url")
    parser.add_argument("--drafter-url")
    parser.add_argument("--server-manifest")
    parser.add_argument("--interval-s", type=float)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--loads-include", nargs="+")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply only explicitly supplied CLI values to the loaded YAML."""
    server_manifest = getattr(args, "server_manifest", None)
    explicit_urls = {
        role: getattr(args, f"{role}_url", None)
        for role in ("target", "verifier", "drafter")
    }
    if server_manifest is not None and any(explicit_urls.values()):
        raise ValueError("--server-manifest conflicts with explicit target URLs")

    for argument in ("interval_s", "request_timeout_s"):
        value = getattr(args, argument)
        if value is not None:
            config[argument] = value

    if args.loads_include is not None:
        loads = config.setdefault("loads", {})
        if not isinstance(loads, dict):
            raise ValueError("loads must be a mapping")
        loads["include"] = args.loads_include

    if any(explicit_urls.values()):
        targets = config.setdefault("targets", {})
        if not isinstance(targets, dict):
            raise ValueError("targets must be a mapping")
        for role, value in explicit_urls.items():
            if value is None:
                continue
            target = targets.setdefault(role, {})
            if not isinstance(target, dict):
                raise ValueError(f"targets.{role} must be a mapping")
            target["base_url"] = value

    if server_manifest is not None:
        manifest_path = Path(server_manifest).expanduser().resolve()
        config["targets"] = _load_ready_manifest_targets(manifest_path)
        config["server_manifest"] = {
            "path": str(manifest_path),
            "state": "ready",
        }


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 1)) != 1:
        raise ValueError("unsupported schema_version; expected 1")
    if float(config.get("interval_s", 0)) <= 0:
        raise ValueError("interval_s must be positive")
    _normalize_targets(config)


async def _get_json(
    session: aiohttp.ClientSession,
    target_id: str,
    role: str,
    rank: int,
    base_url: str,
    path: str,
    sample_id: int,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    collected_wall_time = time.time()
    try:
        async with session.get(base_url.rstrip("/") + path) as response:
            payload = await response.json()
            status_code = response.status
            response.raise_for_status()
        error = None
    except Exception as exc:
        payload = None
        status_code = None
        error = repr(exc)
    finished_ns = time.monotonic_ns()
    return {
        "sample_id": sample_id,
        "target_id": target_id,
        "role": role,
        "rank": rank,
        "base_url": base_url,
        "path": path,
        "collected_wall_time": collected_wall_time,
        "latency_ms": (finished_ns - started_ns) / 1e6,
        "status_code": status_code,
        "error": error,
        "payload": payload,
    }


async def collect(
    config: dict[str, Any], run_dir: Path, duration_s: float | None = None
) -> dict[str, Any]:
    interval_s = float(config["interval_s"])
    include = ",".join(
        config.get("loads", {}).get("include", ["core", "spec", "queues"])
    )
    loads_path = f"/v1/loads?include={include}"
    timeout = aiohttp.ClientTimeout(total=float(config.get("request_timeout_s", 0.8)))
    targets = _normalize_targets(config)
    output_dir = run_dir / "observer"
    output_dir.mkdir(parents=True, exist_ok=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    started_wall_time = time.time()
    started_ns = time.monotonic_ns()
    sample_ct = error_ct = 0
    decode_metric_windows: dict[tuple[str, int, int], dict[str, Any]] = {}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        deadline = time.monotonic()
        with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
            while not stop.is_set():
                if (
                    duration_s is not None
                    and time.monotonic_ns() - started_ns >= duration_s * 1e9
                ):
                    break
                records = await asyncio.gather(
                    *(
                        _get_json(
                            session,
                            target["target_id"],
                            target["role"],
                            target["rank"],
                            target["base_url"],
                            loads_path,
                            sample_ct,
                        )
                        for target in targets
                    )
                )
                for record in records:
                    record["interval_s"] = interval_s
                    record["observer_started_wall_time"] = started_wall_time
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    error_ct += int(record["error"] is not None)
                    _update_decode_metric_windows(
                        record, decode_metric_windows, started_wall_time
                    )
                stream.flush()
                sample_ct += 1
                deadline += interval_s
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=remaining)
                    except TimeoutError:
                        pass

    target_decode_metrics, role_decode_metrics = _summarize_decode_metric_windows(
        decode_metric_windows, targets
    )
    summary = {
        "started_wall_time": started_wall_time,
        "finished_wall_time": time.time(),
        "interval_s": interval_s,
        "sample_ct": sample_ct,
        "target_ct": len(targets),
        "targets": {
            target["target_id"]: target
            for target in sorted(targets, key=lambda item: item["target_id"])
        },
        "error_ct": error_ct,
        "decode_metrics_by_target": target_decode_metrics,
        "decode_metrics": role_decode_metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--duration-s", type=float)
    add_cli_args(parser)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = (
        yaml.safe_load(Path(args.config).expanduser().read_text(encoding="utf-8")) or {}
    )
    apply_cli_overrides(config, args)
    validate_config(config)
    if args.check:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    run_dir = require_run_dir(args.run_dir)
    summary = asyncio.run(collect(config, run_dir, args.duration_s))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

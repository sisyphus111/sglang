"""Periodically collect read-only HTTP state from verifier and drafter servers."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.artifacts import update_status, write_json


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


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
    role = str(record["role"])
    for load in payload.get("loads", []):
        if not isinstance(load, dict):
            continue
        dp_rank = int(load.get("dp_rank", 0))
        for window in load.get("decode_metrics_windows") or []:
            if not isinstance(window, dict) or not isinstance(
                window.get("window_id"), int
            ):
                raise ValueError(
                    f"{role}: invalid decode_metrics_windows entry: {window!r}"
                )
            if float(window["end_time"]) < min_end_time:
                continue
            key = (role, dp_rank, int(window["window_id"]))
            previous = windows.get(key)
            if previous is not None and previous != window:
                raise ValueError(f"conflicting decode metrics window: {key}")
            windows[key] = window


def _summarize_decode_metric_windows(
    windows: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_role: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for (role, dp_rank, _), window in windows.items():
        by_role.setdefault(role, []).append((dp_rank, window))

    summaries: dict[str, dict[str, Any]] = {}
    for role, entries in sorted(by_role.items()):
        num_decode_iters = sum(int(window["num_decode_iters"]) for _, window in entries)
        num_decode_rows = sum(int(window["num_decode_rows"]) for _, window in entries)
        sum_context_lens = sum(int(window["sum_context_lens"]) for _, window in entries)
        num_verify_rows = sum(int(window["num_verify_rows"]) for _, window in entries)
        num_accept_tokens = sum(
            int(window["num_accept_tokens"]) for _, window in entries
        )
        num_proposed_drafts = sum(
            int(window["num_proposed_drafts"]) for _, window in entries
        )
        cycle_values = [float(window["iter_latency_ms"]) for _, window in entries]
        weighted_cycle_ms = (
            sum(
                float(window["iter_latency_ms"]) * int(window["num_decode_iters"])
                for _, window in entries
            )
            / num_decode_iters
            if num_decode_iters > 0
            else None
        )
        summaries[role] = {
            "window_count": len(entries),
            "dp_ranks": sorted({dp_rank for dp_rank, _ in entries}),
            "first_window_end_time": min(
                float(window["end_time"]) for _, window in entries
            ),
            "last_window_end_time": max(
                float(window["end_time"]) for _, window in entries
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
        }
    return summaries


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose the collector's operational controls as typed overrides."""
    parser.add_argument("--verifier-url")
    parser.add_argument("--drafter-url")
    parser.add_argument("--interval-s", type=float)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--loads-include", nargs="+")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply only explicitly supplied CLI values to the loaded YAML."""
    for argument in ("interval_s", "request_timeout_s"):
        value = getattr(args, argument)
        if value is not None:
            config[argument] = value

    if args.loads_include is not None:
        loads = config.setdefault("loads", {})
        if not isinstance(loads, dict):
            raise ValueError("loads must be a mapping")
        loads["include"] = args.loads_include

    if args.verifier_url is not None or args.drafter_url is not None:
        targets = config.setdefault("targets", {})
        if not isinstance(targets, dict):
            raise ValueError("targets must be a mapping")
        for role, value in (
            ("verifier", args.verifier_url),
            ("drafter", args.drafter_url),
        ):
            if value is None:
                continue
            target = targets.setdefault(role, {})
            if not isinstance(target, dict):
                raise ValueError(f"targets.{role} must be a mapping")
            target["base_url"] = value


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 1)) != 1:
        raise ValueError("unsupported schema_version; expected 1")
    if float(config.get("interval_s", 0)) <= 0:
        raise ValueError("interval_s must be positive")
    targets = config.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("targets must be a non-empty mapping")
    for role, target in targets.items():
        if not isinstance(target, dict) or not target.get("base_url"):
            raise ValueError(f"targets.{role}.base_url is required")


async def _get_json(
    session: aiohttp.ClientSession,
    role: str,
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
        "role": role,
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
    output_dir = run_dir / "observability"
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
        for role, target in config["targets"].items():
            base_url = target["base_url"]
            for endpoint in ("/model_info", "/server_info"):
                snapshot = await _get_json(session, role, base_url, endpoint, -1)
                write_json(
                    output_dir / "startup" / role / f"{endpoint[1:]}.json",
                    snapshot,
                )

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
                            role,
                            target["base_url"],
                            loads_path,
                            sample_ct,
                        )
                        for role, target in config["targets"].items()
                    )
                )
                for record in records:
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

    summary = {
        "started_wall_time": started_wall_time,
        "finished_wall_time": time.time(),
        "interval_s": interval_s,
        "sample_ct": sample_ct,
        "target_ct": len(config["targets"]),
        "error_ct": error_ct,
        "decode_metrics": _summarize_decode_metric_windows(decode_metric_windows),
    }
    write_json(output_dir / "summary.json", summary)
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
    run_dir = Path(args.run_dir).expanduser()
    write_json(run_dir / "observability" / "resolved_config.json", config)
    update_status(run_dir, "observability", "collecting")
    try:
        summary = asyncio.run(collect(config, run_dir, args.duration_s))
        update_status(
            run_dir,
            "observability",
            "completed",
            sample_ct=summary["sample_ct"],
            error_ct=summary["error_ct"],
            decode_metric_window_counts={
                role: metrics["window_count"]
                for role, metrics in summary["decode_metrics"].items()
            },
        )
    except BaseException as exc:
        update_status(run_dir, "observability", "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

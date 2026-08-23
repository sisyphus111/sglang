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
        if role not in {"verifier", "drafter"}:
            raise ValueError(f"{target_id}: role must be verifier or drafter")
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
            raise ValueError(f"invalid collector target id: {key!r}")
        if not isinstance(target, dict):
            raise ValueError(f"targets.{key} must be a mapping")
        target_id = target.get("target_id", key)
        role = target.get("role", key if key in {"verifier", "drafter"} else None)
        rank = target.get("rank", 0)
        base_url = target.get("base_url")
        if target_id != key:
            raise ValueError(f"targets.{key}.target_id must equal its mapping key")
        if role not in {"verifier", "drafter"}:
            raise ValueError(f"targets.{key}.role must be verifier or drafter")
        if type(rank) is not int or rank < 0:
            raise ValueError(f"targets.{key}.rank must be a non-negative integer")
        if (role, rank) in role_ranks:
            raise ValueError(f"duplicate collector target role/rank: {role}/{rank}")
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
    """Expose the collector's operational controls as typed overrides."""
    parser.add_argument("--verifier-url")
    parser.add_argument("--drafter-url")
    parser.add_argument("--server-manifest")
    parser.add_argument("--interval-s", type=float)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--loads-include", nargs="+")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply only explicitly supplied CLI values to the loaded YAML."""
    server_manifest = getattr(args, "server_manifest", None)
    if server_manifest is not None and (
        args.verifier_url is not None or args.drafter_url is not None
    ):
        raise ValueError(
            "--server-manifest conflicts with --verifier-url/--drafter-url"
        )

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
        for target in targets:
            for endpoint in ("/model_info", "/server_info"):
                snapshot = await _get_json(
                    session,
                    target["target_id"],
                    target["role"],
                    target["rank"],
                    target["base_url"],
                    endpoint,
                    -1,
                )
                write_json(
                    output_dir
                    / "startup"
                    / target["target_id"]
                    / f"{endpoint[1:]}.json",
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
            decode_metric_window_counts_by_target={
                target_id: metrics["window_count"]
                for target_id, metrics in summary["decode_metrics_by_target"].items()
            },
        )
    except BaseException as exc:
        update_status(run_dir, "observability", "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

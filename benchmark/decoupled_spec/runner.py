"""Coordinate one benchmark Client with an isolated Observer subprocess."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client.client import (
    add_cli_args as add_client_cli_args,
    apply_cli_overrides as apply_client_cli_overrides,
    execute_client,
    validate_config as validate_client_config,
)
from client.observer import (
    _normalize_targets,
    apply_cli_overrides as apply_observer_cli_overrides,
    validate_config as validate_observer_config,
)
from run_io import require_run_dir, write_json


def _load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a YAML mapping: {path}")
    return value


def _read_samples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                break
            raise ValueError(f"invalid observer sample at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"observer sample line {line_number} is not an object")
        records.append(record)
    return records


def _is_successful_sample(record: dict[str, Any]) -> bool:
    return (
        record.get("error") is None
        and record.get("status_code") == 200
        and isinstance(record.get("payload"), dict)
        and isinstance(record["payload"].get("loads"), list)
        and bool(record["payload"]["loads"])
    )


def _has_zero_waiting(record: dict[str, Any]) -> bool:
    return all(
        type(load.get("num_waiting_reqs")) is int
        and load["num_waiting_reqs"] == 0
        for load in record["payload"]["loads"]
        if isinstance(load, dict)
    ) and all(isinstance(load, dict) for load in record["payload"]["loads"])


def _find_complete_sample_round(
    records: list[dict[str, Any]],
    expected_target_ids: set[str],
    *,
    collected_at_or_after: float | None = None,
    require_zero_waiting: bool = False,
) -> tuple[int, float] | None:
    by_sample_id: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        target_id = record.get("target_id")
        collected_at = record.get("collected_wall_time")
        if (
            type(sample_id) is not int
            or target_id not in expected_target_ids
            or not isinstance(collected_at, (int, float))
            or not _is_successful_sample(record)
            or (
                collected_at_or_after is not None
                and float(collected_at) < collected_at_or_after
            )
            or (require_zero_waiting and not _has_zero_waiting(record))
        ):
            continue
        by_sample_id.setdefault(sample_id, {})[str(target_id)] = record

    for sample_id in sorted(by_sample_id):
        sample = by_sample_id[sample_id]
        if set(sample) == expected_target_ids:
            completed_at = max(
                float(record["collected_wall_time"])
                + float(record.get("latency_ms", 0.0)) / 1000.0
                for record in sample.values()
            )
            return sample_id, completed_at
    return None


def _observer_failure(process: subprocess.Popen[str]) -> RuntimeError:
    stdout, stderr = process.communicate()
    detail = stderr.strip() or stdout.strip() or "no subprocess output"
    return RuntimeError(f"Observer exited before the sampling barrier: {detail}")


def _wait_for_sample_round(
    samples_path: Path,
    process: subprocess.Popen[str],
    expected_target_ids: set[str],
    timeout_s: float,
    *,
    collected_at_or_after: float | None = None,
    require_zero_waiting: bool = False,
) -> tuple[int, float]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        match = _find_complete_sample_round(
            _read_samples(samples_path),
            expected_target_ids,
            collected_at_or_after=collected_at_or_after,
            require_zero_waiting=require_zero_waiting,
        )
        if match is not None:
            return match
        if process.poll() is not None:
            raise _observer_failure(process)
        time.sleep(0.05)
    boundary = (
        "zero-waiting baseline"
        if collected_at_or_after is None
        else "post-client trailing sample"
    )
    raise TimeoutError(f"timed out waiting for Observer {boundary}")


def _stop_observer(
    process: subprocess.Popen[str], timeout_s: float
) -> tuple[str, str]:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise TimeoutError("Observer did not stop after SIGTERM")
    if process.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "no subprocess output"
        raise RuntimeError(f"Observer exited with code {process.returncode}: {detail}")
    return stdout, stderr


def _observer_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "client" / "observer.py"),
        "--config",
        str(Path(args.observer_config).expanduser().resolve()),
        "--server-manifest",
        str(Path(args.server_manifest).expanduser().resolve()),
        "--run-dir",
        str(require_run_dir(args.run_dir)),
    ]
    for argument, option in (
        (args.observer_interval_s, "--interval-s"),
        (args.observer_request_timeout_s, "--request-timeout-s"),
    ):
        if argument is not None:
            command.extend((option, str(argument)))
    if args.observer_loads_include is not None:
        command.append("--loads-include")
        command.extend(args.observer_loads_include)
    return command


def _resolved_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    client_config = _load_yaml(args.client_config)
    apply_client_cli_overrides(client_config, args)
    validate_client_config(client_config)

    observer_config = _load_yaml(args.observer_config)
    observer_args = argparse.Namespace(
        verifier_url=None,
        drafter_url=None,
        server_manifest=args.server_manifest,
        interval_s=args.observer_interval_s,
        request_timeout_s=args.observer_request_timeout_s,
        loads_include=args.observer_loads_include,
    )
    apply_observer_cli_overrides(observer_config, observer_args)
    validate_observer_config(observer_config)
    return client_config, observer_config


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = require_run_dir(args.run_dir)
    timeline_path = run_dir / "observer" / "bench_timeline.json"
    if timeline_path.exists():
        raise FileExistsError(f"benchmark timeline already exists: {timeline_path}")
    client_config, observer_config = _resolved_configs(args)
    expected_target_ids = {
        target["target_id"] for target in _normalize_targets(observer_config)
    }
    observer = subprocess.Popen(
        _observer_command(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_sample_round(
            run_dir / "observer" / "samples.jsonl",
            observer,
            expected_target_ids,
            args.observer_barrier_timeout_s,
            require_zero_waiting=True,
        )

        batch_result, execution = execute_client(client_config, run_dir)

        _wait_for_sample_round(
            run_dir / "observer" / "samples.jsonl",
            observer,
            expected_target_ids,
            args.observer_barrier_timeout_s,
            collected_at_or_after=execution.finished_wall_time,
        )
        observer_stdout, _ = _stop_observer(
            observer, args.observer_stop_timeout_s
        )
        observer_summary = json.loads(observer_stdout)
        observer_started_at = observer_summary.get("started_wall_time")
        observer_finished_at = observer_summary.get("finished_wall_time")
        if not isinstance(observer_started_at, (int, float)) or not isinstance(
            observer_finished_at, (int, float)
        ):
            raise ValueError("Observer summary lacks numeric start/finish times")
        write_json(
            timeline_path,
            {
                "observer_started_wall_time": observer_started_at,
                "client_started_wall_time": execution.started_wall_time,
                "client_finished_wall_time": execution.finished_wall_time,
                "observer_finished_wall_time": observer_finished_at,
                "observer_elapsed_s": observer_finished_at - observer_started_at,
            },
        )
        return batch_result
    except BaseException:
        if observer.poll() is None:
            try:
                _stop_observer(observer, args.observer_stop_timeout_s)
            except Exception:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-config", required=True)
    parser.add_argument("--observer-config", required=True)
    parser.add_argument("--run-dir")
    add_client_cli_args(parser)
    parser.add_argument("--observer-interval-s", type=float)
    parser.add_argument("--observer-request-timeout-s", type=float)
    parser.add_argument("--observer-loads-include", nargs="+")
    parser.add_argument("--observer-barrier-timeout-s", type=float, default=30.0)
    parser.add_argument("--observer-stop-timeout-s", type=float, default=10.0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.server_manifest is None:
        parser.error("--server-manifest is required")
    if args.observer_barrier_timeout_s <= 0 or args.observer_stop_timeout_s <= 0:
        parser.error("Observer timeouts must be positive")
    if args.check:
        client_config, observer_config = _resolved_configs(args)
        print(
            json.dumps(
                {"client": client_config, "observer": observer_config},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --check is used")
    batch_result = run(args)
    print(json.dumps(batch_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

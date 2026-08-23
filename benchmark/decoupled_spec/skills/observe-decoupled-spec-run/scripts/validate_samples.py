#!/usr/bin/env python3
"""Validate observability sample quality and formal-window coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def _expected_targets(
    config: dict[str, Any], required_roles: list[str], errors: list[str]
) -> dict[str, dict[str, Any]]:
    targets = config.get("targets")
    if not isinstance(targets, dict) or not targets:
        if isinstance(config.get("server_manifest"), dict):
            errors.append("observability config has no targets")
            return {}
        return {
            role: {
                "target_id": role,
                "role": role,
                "rank": 0,
                "base_url": None,
            }
            for role in required_roles
        }

    normalized = {}
    role_ranks = set()
    for key, target in targets.items():
        if not isinstance(key, str) or not isinstance(target, dict):
            errors.append(f"invalid observability target: {key!r}={target!r}")
            continue
        target_id = target.get("target_id", key)
        role = target.get("role", key if key in {"verifier", "drafter"} else None)
        rank = target.get("rank", 0)
        base_url = target.get("base_url")
        if target_id != key or role not in {"verifier", "drafter"}:
            errors.append(f"invalid observability target identity: {key!r}={target!r}")
            continue
        if type(rank) is not int or rank < 0:
            errors.append(f"{target_id}: rank must be a non-negative integer")
            continue
        if (role, rank) in role_ranks:
            errors.append(f"duplicate observability target role/rank: {role}/{rank}")
            continue
        if not isinstance(base_url, str) or not base_url:
            errors.append(f"{target_id}: base_url must be a non-empty string")
            continue
        role_ranks.add((role, rank))
        normalized[target_id] = {
            "target_id": target_id,
            "role": role,
            "rank": rank,
            "base_url": base_url.rstrip("/"),
        }
    return normalized


def validate_samples(
    run_dir: Path,
    required_roles: list[str],
    require_formal_window: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    samples_path = run_dir / "observability" / "samples.jsonl"
    config_path = run_dir / "observability" / "resolved_config.json"
    summary_path = run_dir / "observability" / "summary.json"
    formal_path = run_dir / "client" / "formal_window.json"

    for path in (samples_path, config_path, summary_path):
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

    try:
        config = _read_json(config_path)
        summary = _read_json(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(str(exc))
        return {
            "ok": False,
            "run_dir": str(run_dir),
            "errors": errors,
            "warnings": warnings,
            "roles": {},
        }

    records = _read_jsonl(samples_path, errors)
    expected_targets = _expected_targets(config, required_roles, errors)
    manifest_mode = isinstance(config.get("server_manifest"), dict)
    observed_target_ids = set()
    for record in records:
        role = record.get("role")
        target_id = str(record.get("target_id", role))
        observed_target_ids.add(target_id)
        expected = expected_targets.get(target_id)
        if expected is None:
            errors.append(f"sample references unknown target_id={target_id!r}")
            continue
        if role != expected["role"]:
            errors.append(
                f"{target_id}: sample role mismatch: "
                f"observed={role!r}, expected={expected['role']!r}"
            )
        if manifest_mode:
            for field in ("rank", "base_url"):
                observed = record.get(field)
                if field == "base_url" and isinstance(observed, str):
                    observed = observed.rstrip("/")
                if observed != expected[field]:
                    errors.append(
                        f"{target_id}: sample {field} mismatch: "
                        f"observed={observed!r}, expected={expected[field]!r}"
                    )
    if manifest_mode and observed_target_ids != set(expected_targets):
        errors.append(
            "observability samples do not cover the configured engine set: "
            f"observed={sorted(observed_target_ids)}, "
            f"expected={sorted(expected_targets)}"
        )
    interval_s = float(config.get("interval_s", 0) or 0)
    if interval_s <= 0:
        errors.append("observability interval_s must be positive")

    formal: dict[str, Any] | None = None
    if formal_path.is_file():
        try:
            formal = _read_json(formal_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
    elif require_formal_window:
        errors.append(f"missing required file: {formal_path}")

    formal_start = formal.get("started_wall_time") if formal else None
    formal_finish = formal.get("finished_wall_time") if formal else None
    if formal is not None:
        if formal.get("state") != "completed":
            errors.append("client formal window is not completed")
        if not isinstance(formal_start, (int, float)) or not isinstance(
            formal_finish, (int, float)
        ):
            errors.append("client formal window lacks numeric start/finish times")
            formal_start = formal_finish = None
        elif formal_finish < formal_start:
            errors.append("client formal window finishes before it starts")
            formal_start = formal_finish = None

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
        if formal_start is not None and formal_finish is not None:
            for record in successful:
                collected_at = record.get("collected_wall_time")
                if not isinstance(collected_at, (int, float)) or not (
                    formal_start <= collected_at <= formal_finish
                ):
                    continue
                for load in record["payload"]["loads"]:
                    if not isinstance(load, dict):
                        continue
                    waiting = load.get("num_waiting_reqs")
                    if type(waiting) is not int or waiting < 0:
                        errors.append(
                            f"{role}: formal-window load sample lacks a valid "
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
                                "num_waiting_reqs": waiting,
                            }
                        )
        report["formal_waiting_sample_count"] = len(formal_waiting_samples)
        report["max_waiting_reqs_in_formal_window"] = max(
            (sample["num_waiting_reqs"] for sample in formal_waiting_samples),
            default=0,
        )
        report["formal_waiting_samples"] = formal_waiting_samples
        if formal_waiting_samples:
            errors.append(
                f"{role}: waiting requests observed inside the formal window: "
                f"max={report['max_waiting_reqs_in_formal_window']} "
                f"sample_count={len(formal_waiting_samples)}"
            )
        decode_windows: dict[tuple[str, int, int], dict[str, Any]] = {}
        collector_started_at = summary.get("started_wall_time")
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
                    if isinstance(collector_started_at, (int, float)) and float(
                        window.get("end_time", 0)
                    ) < float(collector_started_at):
                        continue
                    key = (target_id, dp_rank, int(window["window_id"]))
                    previous = decode_windows.get(key)
                    if previous is not None and previous != window:
                        errors.append(
                            f"{role}: conflicting decode metrics window {key}"
                        )
                    decode_windows[key] = window
        report["decode_metrics_window_count"] = len(decode_windows)
        summary_decode_metrics = summary.get("decode_metrics")
        summary_role_metrics = (
            summary_decode_metrics.get(role)
            if isinstance(summary_decode_metrics, dict)
            else None
        )
        if decode_windows:
            if not isinstance(summary_role_metrics, dict):
                errors.append(f"{role}: collector summary lacks decode metrics")
            elif int(summary_role_metrics.get("window_count", -1)) != len(
                decode_windows
            ):
                errors.append(
                    f"{role}: collector decode window count does not match samples: "
                    f"summary={summary_role_metrics.get('window_count')!r}, "
                    f"observed={len(decode_windows)}"
                )
            elif not isinstance(summary_role_metrics.get("scheduler_cycle_ms"), dict):
                errors.append(f"{role}: collector summary lacks scheduler cycle stats")
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
        target_summary_metrics = summary.get("decode_metrics_by_target")
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
            if target_window_count or manifest_mode:
                summarized = (
                    target_summary_metrics.get(target_id)
                    if isinstance(target_summary_metrics, dict)
                    else None
                )
                # Legacy single-engine runs only persisted the role aggregate.
                # Their implicit target_id equals the role, so that aggregate
                # is also the exact per-target summary.
                if not isinstance(summarized, dict) and target_id == role:
                    summarized = summary_role_metrics
                if not isinstance(summarized, dict):
                    errors.append(
                        f"{target_id}: collector summary lacks decode metrics"
                    )
                elif int(summarized.get("window_count", -1)) != target_window_count:
                    errors.append(
                        f"{target_id}: decode window count mismatch: "
                        f"summary={summarized.get('window_count')!r} "
                        f"observed={target_window_count}"
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

    observed_error_ct = sum(not _is_success(record) for record in records)
    if int(summary.get("error_ct", -1)) != observed_error_ct:
        errors.append(
            "observability summary error_ct does not match samples.jsonl: "
            f"summary={summary.get('error_ct')!r}, observed={observed_error_ct}"
        )
    target_ct = int(summary.get("target_ct", 0) or 0)
    sample_ct = int(summary.get("sample_ct", 0) or 0)
    if manifest_mode and target_ct != len(expected_targets):
        errors.append(
            "observability summary target_ct does not match resolved targets: "
            f"summary={target_ct}, expected={len(expected_targets)}"
        )
    summary_targets = summary.get("targets")
    if manifest_mode and (
        not isinstance(summary_targets, dict)
        or set(summary_targets) != set(expected_targets)
    ):
        errors.append(
            "observability summary target identities do not match resolved targets"
        )
    if target_ct > 0 and sample_ct * target_ct != len(records):
        errors.append(
            "observability summary sample_ct * target_ct does not match JSONL records: "
            f"{sample_ct} * {target_ct} != {len(records)}"
        )

    return {
        "ok": not errors,
        "run_dir": str(run_dir),
        "interval_s": interval_s,
        "formal_window": formal,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--role",
        action="append",
        dest="roles",
        choices=("verifier", "drafter"),
        help="Required role; repeat to override the default pair.",
    )
    parser.add_argument(
        "--require-formal-window",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate_samples(
        Path(args.run_dir).expanduser().resolve(),
        args.roles or ["verifier", "drafter"],
        args.require_formal_window,
    )
    if args.output:
        _write_json(Path(args.output).expanduser(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

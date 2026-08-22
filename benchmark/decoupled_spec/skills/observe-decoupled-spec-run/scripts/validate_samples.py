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
        decode_windows: dict[tuple[int, int], dict[str, Any]] = {}
        for record in successful:
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
                    key = (dp_rank, int(window["window_id"]))
                    previous = decode_windows.get(key)
                    if previous is not None and previous != window:
                        errors.append(
                            f"{role}: conflicting decode metrics window {key}"
                        )
                    decode_windows[key] = window
        report["decode_metrics_window_count"] = len(decode_windows)
        decode_window_gaps: dict[int, list[int]] = {}
        for dp_rank in sorted({key[0] for key in decode_windows}):
            ids = sorted(key[1] for key in decode_windows if key[0] == dp_rank)
            missing = [
                window_id
                for left, right in zip(ids, ids[1:])
                for window_id in range(left + 1, right)
            ]
            if missing:
                decode_window_gaps[dp_rank] = missing
                errors.append(
                    f"{role}: missing decode metrics windows for dp_rank={dp_rank}: "
                    f"{missing}"
                )
        report["decode_metrics_window_gaps"] = decode_window_gaps
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

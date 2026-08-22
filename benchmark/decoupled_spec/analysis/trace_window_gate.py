"""Shared capture-window provenance checks for decoupled profiler traces."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

_DECODE_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^]]*\] "
    r"Decode batch, #running-req: (?P<running>\d+)"
)


def _parse_trigger(path: Path) -> datetime:
    value = path.read_text(encoding="utf-8").strip()
    # The capture helper writes ISO-8601 with a comma fractional separator.
    if "," in value:
        head, tail = value.split(",", 1)
        value = head + "." + tail
    return datetime.fromisoformat(value)


def infer_running_bs_from_log(
    *, trigger_path: Path, verifier_log_path: Path
) -> dict[str, Any]:
    trigger = _parse_trigger(trigger_path)
    candidates = []
    for line in verifier_log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = _DECODE_RE.match(line)
        if match is None:
            continue
        timestamp = datetime.strptime(
            match.group("timestamp"), "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=trigger.tzinfo)
        candidates.append(
            (
                abs((timestamp - trigger).total_seconds()),
                timestamp,
                int(match.group("running")),
                line,
            )
        )
    if not candidates:
        return {
            "trigger_time": trigger.isoformat(),
            "running_bs": None,
            "nearest_log_delta_s": None,
            "nearest_log_line": None,
        }
    delta, timestamp, running_bs, line = min(candidates, key=lambda item: item[0])
    return {
        "trigger_time": trigger.isoformat(),
        "nearest_log_time": timestamp.isoformat(),
        "running_bs": running_bs,
        "nearest_log_delta_s": delta,
        "nearest_log_line": line,
    }


def build_window_gate(
    *,
    expected_bs: int,
    fold_grid_y_values: list[int] | None = None,
    trigger_path: Path | None = None,
    verifier_log_path: Path | None = None,
    max_log_delta_s: float = 2.0,
) -> dict[str, Any]:
    unique_fold_values = sorted(set(fold_grid_y_values or []))
    inferred_fold_bs = unique_fold_values[0] if len(unique_fold_values) == 1 else None
    log_context = None
    if (trigger_path is None) != (verifier_log_path is None):
        raise ValueError("--trigger-file and --verifier-log must be supplied together")
    if trigger_path is not None and verifier_log_path is not None:
        log_context = infer_running_bs_from_log(
            trigger_path=trigger_path, verifier_log_path=verifier_log_path
        )

    errors = []
    if fold_grid_y_values is not None:
        if inferred_fold_bs is None:
            errors.append(
                "ReplaySSM fold gridY does not identify one stable raw batch size: "
                f"values={unique_fold_values}"
            )
        elif inferred_fold_bs != expected_bs:
            errors.append(
                "ReplaySSM fold gridY batch mismatch: "
                f"expected={expected_bs} observed={inferred_fold_bs}"
            )
    if log_context is not None:
        if log_context["nearest_log_delta_s"] is None:
            errors.append("Verifier log contains no Decode batch line")
        elif float(log_context["nearest_log_delta_s"]) > max_log_delta_s:
            errors.append(
                "Nearest verifier Decode batch line is too far from the trigger: "
                f"delta_s={log_context['nearest_log_delta_s']:.3f}"
            )
        elif int(log_context["running_bs"]) != expected_bs:
            errors.append(
                "Verifier trigger-context batch mismatch: "
                f"expected={expected_bs} observed={log_context['running_bs']}"
            )

    return {
        "passed": not errors,
        "expected_bs": expected_bs,
        "fold_grid_y_values": unique_fold_values,
        "inferred_bs_from_fold_grid_y": inferred_fold_bs,
        "log_context": log_context,
        "errors": errors,
    }

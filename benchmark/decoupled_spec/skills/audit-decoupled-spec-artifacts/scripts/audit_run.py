#!/usr/bin/env python3
"""Audit a decoupled-spec run before sealing or verify a sealed run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

_CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_QUEUE_RE = re.compile(r"#queue-req:\s*(\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing required file: {path}")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read JSON {path}: {exc}")
    return None


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read JSONL {path}: {exc}")
        return []
    records = []
    for line_number, line in enumerate(lines, start=1):
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


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    try:
        # Long generations can put more than Python's 128 KiB default in one
        # generated-text field. These are owned local artifacts, so preserve
        # the full provenance instead of rejecting an otherwise valid run.
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        errors.append(f"cannot read CSV {path}: {exc}")
        return []


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _process_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_observability_validator() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[2]
        / "observe-decoupled-spec-run"
        / "scripts"
        / "validate_samples.py"
    )
    spec = importlib.util.spec_from_file_location(
        "decoupled_spec_validate_samples", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load observability validator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _has_decode_metrics_windows(run_dir: Path) -> bool:
    path = run_dir / "observability" / "samples.jsonl"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = record.get("payload") if isinstance(record, dict) else None
        for load in payload.get("loads", []) if isinstance(payload, dict) else []:
            if isinstance(load, dict) and load.get("decode_metrics_windows"):
                return True
    return False


def _unified_server_engines(run_dir: Path) -> list[dict[str, Any]] | None:
    path = run_dir / "server" / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    engines = manifest.get("engines") if isinstance(manifest, dict) else None
    if not isinstance(engines, list):
        return []
    return [engine for engine in engines if isinstance(engine, dict)]


def _check_required_files(run_dir: Path, errors: list[str]) -> None:
    required = [
        "provenance/run_start.json",
        "roles/client/status.json",
        "roles/observability/status.json",
        "client/resolved_config.json",
        "client/verifier_model_info.json",
        "client/sampled_requests.jsonl",
        "client/formal_window.json",
        "client/stream_timing_events.jsonl",
        "client/raw_batch_response.json",
        "client/responses.jsonl",
        "client/request_metrics.csv",
        "client/summary.json",
        "observability/resolved_config.json",
        "observability/samples.jsonl",
        "observability/summary.json",
        "plots/request_latency.svg",
        "plots/request_latency.png",
        "plots/request_latency_manifest.json",
        "plots/request_speculative_manifest.json",
        "plots/run_report.md",
        "plots/run_report_manifest.json",
        "observability/plots/overview.svg",
        "observability/plots/overview.png",
        "observability/plots/plot_manifest.json",
    ]
    engines = _unified_server_engines(run_dir)
    if engines is None:
        required.extend(
            (
                "roles/verifier/resolved_config.json",
                "roles/drafter/resolved_config.json",
                "roles/verifier/status.json",
                "roles/drafter/status.json",
                "observability/startup/verifier/model_info.json",
                "observability/startup/verifier/server_info.json",
                "observability/startup/drafter/model_info.json",
                "observability/startup/drafter/server_info.json",
            )
        )
    else:
        required.extend(
            (
                "server/manifest.json",
                "server/resolved_config.json",
                "roles/server/status.json",
            )
        )
        if not engines:
            errors.append("unified server manifest has no engine entries")
        for engine in engines:
            engine_id = engine.get("engine_id")
            if not isinstance(engine_id, str) or not engine_id:
                errors.append(f"invalid unified server engine identity: {engine!r}")
                continue
            for field in ("resolved_config_path", "status_path", "log_path"):
                relative = engine.get(field)
                if not isinstance(relative, str) or not relative:
                    errors.append(f"{engine_id}: manifest lacks {field}")
                else:
                    required.append(relative)
            required.extend(
                (
                    f"observability/startup/{engine_id}/model_info.json",
                    f"observability/startup/{engine_id}/server_info.json",
                )
            )
    if _has_decode_metrics_windows(run_dir):
        required.extend(
            (
                "observability/plots/decode_metrics.svg",
                "observability/plots/decode_metrics.png",
            )
        )
    for relative in required:
        candidate = run_dir / relative
        if Path(relative).is_absolute() or not _is_within(candidate, run_dir):
            errors.append(f"required artifact path escapes RUN_DIR: {relative!r}")
        elif not candidate.is_file():
            errors.append(f"missing required file: {candidate}")


def _check_unified_consumer_identity(
    run_dir: Path, errors: list[str]
) -> dict[str, Any]:
    manifest_path = run_dir / "server" / "manifest.json"
    if not manifest_path.is_file():
        return {"mode": "legacy"}
    manifest = _read_json(manifest_path, errors)
    client = _read_json(run_dir / "client" / "resolved_config.json", errors)
    collector = _read_json(run_dir / "observability" / "resolved_config.json", errors)
    if not all(isinstance(value, dict) for value in (manifest, client, collector)):
        return {"mode": "unified"}
    if manifest.get("schema_version") != 1:
        errors.append(
            "unified server manifest schema_version must be 1, got "
            f"{manifest.get('schema_version')!r}"
        )
    raw_engines = manifest.get("engines")
    if not isinstance(raw_engines, list) or not raw_engines:
        errors.append("unified server manifest must contain non-empty engines")
        return {"mode": "unified", "engine_count": 0}

    engines = {}
    role_ranks = set()
    for index, engine in enumerate(raw_engines):
        if not isinstance(engine, dict):
            errors.append(f"server manifest engines[{index}] must be a JSON object")
            continue
        engine_id = engine.get("engine_id")
        role = engine.get("role")
        rank = engine.get("rank")
        http_url = engine.get("http_url")
        if not isinstance(engine_id, str) or not engine_id:
            errors.append(f"server manifest engines[{index}] has invalid engine_id")
            continue
        if engine_id in engines:
            errors.append(f"duplicate server manifest engine_id: {engine_id}")
            continue
        if role not in {"verifier", "drafter"} or type(rank) is not int or rank < 0:
            errors.append(f"{engine_id}: invalid server manifest role/rank")
            continue
        if (role, rank) in role_ranks:
            errors.append(f"duplicate server manifest role/rank: {role}/{rank}")
        if not isinstance(http_url, str) or not http_url:
            errors.append(f"{engine_id}: invalid server manifest http_url")
            continue
        role_ranks.add((role, rank))
        engines[engine_id] = {
            "target_id": engine_id,
            "role": role,
            "rank": rank,
            "base_url": http_url.rstrip("/"),
        }

    topology = manifest.get("topology")
    if isinstance(topology, dict):
        for role, count_field in (
            ("verifier", "num_verifiers"),
            ("drafter", "num_drafters"),
        ):
            actual = sum(engine["role"] == role for engine in engines.values())
            if topology.get(count_field) != actual:
                errors.append(
                    f"server manifest topology {count_field} mismatch: "
                    f"recorded={topology.get(count_field)!r}, actual={actual}"
                )
    else:
        errors.append("unified server manifest lacks topology")

    selected = client.get("server")
    if not isinstance(selected, dict):
        errors.append("unified client config lacks selected server identity")
    else:
        target_id = selected.get("target_id")
        expected = engines.get(target_id)
        observed = {
            "target_id": target_id,
            "role": selected.get("role"),
            "rank": selected.get("rank"),
            "base_url": (
                selected.get("base_url", "").rstrip("/")
                if isinstance(selected.get("base_url"), str)
                else selected.get("base_url")
            ),
        }
        if expected is None or expected["role"] != "verifier":
            errors.append(
                f"client selected target is not a manifest verifier: {target_id!r}"
            )
        elif observed != expected:
            errors.append(
                "client selected verifier identity disagrees with server manifest: "
                f"observed={observed!r}, expected={expected!r}"
            )

    raw_targets = collector.get("targets")
    collector_targets = {}
    if not isinstance(raw_targets, dict):
        errors.append("unified observability config lacks target mapping")
    else:
        for target_id, target in raw_targets.items():
            if not isinstance(target_id, str) or not isinstance(target, dict):
                errors.append(f"invalid observability target: {target_id!r}")
                continue
            base_url = target.get("base_url")
            collector_targets[target_id] = {
                "target_id": target.get("target_id", target_id),
                "role": target.get("role"),
                "rank": target.get("rank"),
                "base_url": (
                    base_url.rstrip("/") if isinstance(base_url, str) else base_url
                ),
            }
    if collector_targets != engines:
        errors.append(
            "observability targets disagree with server manifest: "
            f"observed={collector_targets!r}, expected={engines!r}"
        )
    return {
        "mode": "unified",
        "engine_count": len(engines),
        "client_target_id": (
            selected.get("target_id") if isinstance(selected, dict) else None
        ),
        "collector_target_count": len(collector_targets),
    }


def _check_statuses(
    run_dir: Path, errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    engines = _unified_server_engines(run_dir)
    local_roles = ["client", "observability"]
    if engines is None:
        local_roles = ["verifier", "drafter", *local_roles]
    else:
        local_roles = ["server", *local_roles]
    for role in local_roles:
        status = _read_json(run_dir / "roles" / role / "status.json", errors)
        if not isinstance(status, dict):
            continue
        state = status.get("state")
        pid = status.get("pid")
        alive = _process_alive(pid)
        reports[role] = {"state": state, "pid": pid, "process_alive": alive}
        if state == "failed":
            errors.append(f"{role} status is failed: {status.get('error')!r}")
        if role in {"client", "observability"} and state != "completed":
            errors.append(f"{role} status must be 'completed', got {state!r}")
        if role in {"client", "observability"} and alive:
            errors.append(f"{role} process PID {pid} is still alive")
        if role in {"verifier", "drafter", "server"}:
            if alive:
                errors.append(f"{role} process PID {pid} is still alive")
            allowed_states = (
                {"exited", "http_ready"} if role != "server" else {"exited"}
            )
            if state not in allowed_states:
                errors.append(
                    f"{role} final state must be one of {sorted(allowed_states)}, "
                    f"got {state!r}"
                )
            if state == "http_ready" and not alive:
                warnings.append(
                    f"{role} process is dead but status remains 'http_ready'; inspect its log"
                )
    if engines is None:
        for role in ("verifier", "drafter"):
            log_path = run_dir / "logs" / f"{role}.log"
            if not log_path.is_file():
                warnings.append(f"missing captured server log: {log_path}")
    else:
        manifest = _read_json(run_dir / "server" / "manifest.json", errors)
        if isinstance(manifest, dict):
            if manifest.get("state") == "failed":
                errors.append(
                    f"unified server manifest failed: {manifest.get('error')!r}"
                )
            elif manifest.get("state") != "stopped":
                errors.append(
                    "unified server manifest final state must be 'stopped', got "
                    f"{manifest.get('state')!r}"
                )
        engine_reports = {}
        for engine in engines:
            engine_id = str(engine.get("engine_id", ""))
            status_path = engine.get("status_path")
            if not engine_id or not isinstance(status_path, str):
                continue
            status = _read_json(run_dir / status_path, errors)
            state = status.get("state") if isinstance(status, dict) else None
            engine_reports[engine_id] = {
                "state": state,
                "remote_pid": status.get("pid") if isinstance(status, dict) else None,
            }
            if state != "exited":
                errors.append(
                    f"{engine_id} final state must be 'exited', got {state!r}"
                )
        reports["engines"] = engine_reports
    return reports


def _check_zero_waiting_queues(
    run_dir: Path,
    errors: list[str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    if warnings is None:
        warnings = []
    reports = {}
    engines = _unified_server_engines(run_dir)
    log_entries = (
        [(role, run_dir / "logs" / f"{role}.log") for role in ("verifier", "drafter")]
        if engines is None
        else [
            (
                str(engine.get("engine_id")),
                run_dir / str(engine.get("log_path")),
            )
            for engine in engines
        ]
    )
    for role, path in log_entries:
        if not path.is_file():
            message = f"{role}: queue validation requires captured log {path}"
            if engines is None:
                warnings.append(message)
            else:
                errors.append(message)
            reports[role] = {"queue_sample_count": 0, "max_waiting_reqs": None}
            continue
        queue_samples = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            match = _QUEUE_RE.search(line)
            if match is None:
                continue
            waiting = int(match.group(1))
            queue_samples.append((line_number, waiting, line))
        if not queue_samples:
            errors.append(f"{role}: role log contains no queue telemetry")
            reports[role] = {"queue_sample_count": 0, "max_waiting_reqs": None}
            continue
        max_waiting = max(waiting for _, waiting, _ in queue_samples)
        offending = [sample for sample in queue_samples if sample[1] > 0]
        reports[role] = {
            "queue_sample_count": len(queue_samples),
            "max_waiting_reqs": max_waiting,
            "positive_queue_sample_count": len(offending),
            "first_positive_queue_sample": (
                {
                    "line_number": offending[0][0],
                    "num_waiting_reqs": offending[0][1],
                    "line": offending[0][2],
                }
                if offending
                else None
            ),
        }
        if offending:
            errors.append(
                f"{role}: role log observed waiting requests: "
                f"max={max_waiting} sample_count={len(offending)} "
                f"first_line={offending[0][0]}"
            )
    return reports


def _check_client(run_dir: Path, errors: list[str]) -> dict[str, Any]:
    config = _read_json(run_dir / "client" / "resolved_config.json", errors)
    formal = _read_json(run_dir / "client" / "formal_window.json", errors)
    summary = _read_json(run_dir / "client" / "summary.json", errors)
    raw = _read_json(run_dir / "client" / "raw_batch_response.json", errors)
    sampled = _read_jsonl(run_dir / "client" / "sampled_requests.jsonl", errors)
    responses = _read_jsonl(run_dir / "client" / "responses.jsonl", errors)
    timing_events = _read_jsonl(
        run_dir / "client" / "stream_timing_events.jsonl", errors
    )
    metrics = _read_csv(run_dir / "client" / "request_metrics.csv", errors)

    if not all(
        isinstance(value, dict) for value in (config, formal, summary)
    ) or not isinstance(raw, list):
        return {}
    try:
        batch_size = int(config["batch"]["size"])
    except (KeyError, TypeError, ValueError):
        errors.append("client resolved config lacks a valid batch.size")
        return {}
    if batch_size <= 0:
        errors.append(f"client batch.size must be positive, got {batch_size}")

    cardinalities = {
        "resolved_batch_size": batch_size,
        "formal_batch_size": formal.get("batch_size"),
        "summary_batch_size": summary.get("batch_size"),
        "summary_request_count": summary.get("request_count"),
        "summary_completed_count": summary.get("completed_count"),
        "sampled_request_count": len(sampled),
        "response_count": len(responses),
        "metrics_row_count": len(metrics),
        "raw_response_count": len(raw),
    }
    for name, value in cardinalities.items():
        try:
            matches = int(value) == batch_size
        except (TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(
                f"client cardinality mismatch: {name}={value!r}, expected {batch_size}"
            )
    if int(summary.get("failed_count", -1)) != 0:
        errors.append(
            f"client summary failed_count must be 0, got {summary.get('failed_count')!r}"
        )

    if formal.get("state") != "completed":
        errors.append(
            f"client formal window state is {formal.get('state')!r}, expected 'completed'"
        )
    start = formal.get("started_wall_time")
    finish = formal.get("finished_wall_time")
    elapsed = formal.get("elapsed_s")
    if not isinstance(start, (int, float)) or not isinstance(finish, (int, float)):
        errors.append("client formal window lacks numeric wall-clock bounds")
    elif finish <= start:
        errors.append("client formal window must finish after it starts")
    if not isinstance(elapsed, (int, float)) or elapsed <= 0:
        errors.append("client formal window elapsed_s must be positive")

    id_sets: dict[str, set[str]] = {}
    for name, records in (
        ("sampled_requests", sampled),
        ("responses", responses),
        ("request_metrics", metrics),
    ):
        values = [str(record.get("request_id")) for record in records]
        if any(value == "None" for value in values):
            errors.append(f"{name} contains a missing request_id")
        if len(set(values)) != len(values):
            errors.append(f"{name} contains duplicate request_id values")
        id_sets[name] = set(values)
    if len({frozenset(values) for values in id_sets.values()}) > 1:
        errors.append(f"request IDs disagree across client artifacts: {id_sets}")

    indices = {
        event.get("index")
        for event in timing_events
        if isinstance(event.get("index"), int)
    }
    expected_indices = set(range(batch_size))
    if indices != expected_indices:
        errors.append(
            f"stream timing indices {sorted(indices)} do not cover {sorted(expected_indices)}"
        )
    return {
        "cardinalities": cardinalities,
        "request_ids": sorted(id_sets.get("responses", set())),
    }


def _check_observability(
    run_dir: Path, errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    try:
        validator = _load_observability_validator()
        report = validator.validate_samples(
            run_dir, ["verifier", "drafter"], require_formal_window=True
        )
    except Exception as exc:
        errors.append(f"cannot run observability validator: {exc}")
        return {}
    errors.extend(f"observability: {message}" for message in report.get("errors", []))
    warnings.extend(
        f"observability: {message}" for message in report.get("warnings", [])
    )

    return report


def _safe_manifest_path(run_dir: Path, value: Any, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"invalid manifest path: {value!r}")
        return None
    path = run_dir / value
    if not _is_within(path, run_dir):
        errors.append(f"manifest path escapes RUN_DIR: {value!r}")
        return None
    return path


def _check_derived_manifest(
    run_dir: Path,
    relative_path: str,
    expected_kind: str,
    allow_empty_outputs: bool,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    path = run_dir / relative_path
    manifest = _read_json(path, errors)
    if not isinstance(manifest, dict):
        return {}
    if manifest.get("kind") != expected_kind:
        errors.append(f"unexpected manifest kind in {path}: {manifest.get('kind')!r}")
    recorded_run = manifest.get("run_dir")
    if (
        isinstance(recorded_run, str)
        and Path(recorded_run).resolve() != run_dir.resolve()
    ):
        warnings.append(
            f"manifest {path} recorded original run path {recorded_run!r}; "
            f"current path is {str(run_dir)!r}"
        )
    checked_sources = 0
    for source in manifest.get("sources", []):
        if not isinstance(source, dict):
            errors.append(f"invalid source entry in {path}: {source!r}")
            continue
        source_path = _safe_manifest_path(run_dir, source.get("path"), errors)
        if source_path is None:
            continue
        if not source_path.is_file():
            errors.append(f"missing derived-artifact source: {source_path}")
            continue
        actual = _sha256(source_path)
        if actual != source.get("sha256"):
            errors.append(
                f"source hash mismatch for {path}: {source_path}; "
                f"expected={source.get('sha256')}, actual={actual}"
            )
        checked_sources += 1
    outputs = manifest.get("outputs", [])
    if not isinstance(outputs, list):
        errors.append(f"manifest outputs must be a list: {path}")
        outputs = []
    elif not outputs and not allow_empty_outputs:
        errors.append(f"manifest has no outputs: {path}")
    for output in outputs:
        output_path = _safe_manifest_path(run_dir, output, errors)
        if output_path is not None and not output_path.is_file():
            errors.append(f"missing derived-artifact output: {output_path}")
    return {"source_count": checked_sources, "output_count": len(outputs)}


def _check_derived_manifests(
    run_dir: Path, errors: list[str], warnings: list[str]
) -> dict[str, Any]:
    specifications = (
        (
            "plots/request_latency_manifest.json",
            "decoupled_spec_request_latency_plot",
            False,
        ),
        (
            "plots/request_speculative_manifest.json",
            "decoupled_spec_request_speculative_plot",
            True,
        ),
        (
            "observability/plots/plot_manifest.json",
            "decoupled_spec_observability_plot",
            False,
        ),
        ("plots/run_report_manifest.json", "decoupled_spec_run_report", False),
    )
    reports = {
        relative_path: _check_derived_manifest(
            run_dir,
            relative_path,
            expected_kind,
            allow_empty_outputs,
            errors,
            warnings,
        )
        for relative_path, expected_kind, allow_empty_outputs in specifications
    }
    if _has_decode_metrics_windows(run_dir):
        path = run_dir / "observability" / "plots" / "plot_manifest.json"
        manifest = _read_json(path, errors)
        outputs = (
            set(manifest.get("outputs", [])) if isinstance(manifest, dict) else set()
        )
        for required in (
            "observability/plots/decode_metrics.svg",
            "observability/plots/decode_metrics.png",
        ):
            if required not in outputs:
                errors.append(f"observability plot manifest omits {required}")
    return reports


def _check_seal(run_dir: Path, errors: list[str]) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json", errors)
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("sealed_at"), (int, float)
    ):
        errors.append("run_manifest.json lacks a numeric sealed_at")

    checksum_path = run_dir / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"cannot read {checksum_path}: {exc}")
        return {}
    listed: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_RE.fullmatch(line)
        if match is None:
            errors.append(f"{checksum_path}:{line_number}: invalid checksum line")
            continue
        digest, relative = match.groups()
        if relative in listed:
            errors.append(f"duplicate checksum path: {relative}")
            continue
        candidate = _safe_manifest_path(run_dir, relative, errors)
        if candidate is None:
            continue
        listed[relative] = digest
        if not candidate.is_file():
            errors.append(f"checksum file is missing: {candidate}")
            continue
        actual = _sha256(candidate)
        if actual != digest:
            errors.append(
                f"checksum mismatch: {candidate}; expected={digest}, actual={actual}"
            )

    actual_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    listed_files = set(listed)
    for relative in sorted(actual_files - listed_files):
        errors.append(f"file added or omitted after sealing: {run_dir / relative}")
    for relative in sorted(listed_files - actual_files):
        errors.append(f"checksum lists a missing file: {run_dir / relative}")
    return {
        "listed_file_count": len(listed_files),
        "actual_file_count": len(actual_files),
    }


def audit_run(run_dir: Path, phase: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not run_dir.is_dir():
        return {
            "schema_version": 1,
            "ok": False,
            "phase": phase,
            "run_dir": str(run_dir),
            "errors": [f"RUN_DIR does not exist: {run_dir}"],
            "warnings": [],
            "checks": {},
        }
    if phase == "pre-seal" and (
        (run_dir / "run_manifest.json").exists() or (run_dir / "SHA256SUMS").exists()
    ):
        errors.append("pre-seal audit cannot run after seal artifacts already exist")

    _check_required_files(run_dir, errors)
    checks = {
        "server_consumer_identity": _check_unified_consumer_identity(run_dir, errors),
        "statuses": _check_statuses(run_dir, errors, warnings),
        "zero_waiting_queues": _check_zero_waiting_queues(run_dir, errors, warnings),
        "client": _check_client(run_dir, errors),
        "observability": _check_observability(run_dir, errors, warnings),
        "derived_artifacts": _check_derived_manifests(run_dir, errors, warnings),
    }
    if phase == "sealed":
        checks["seal"] = _check_seal(run_dir, errors)
    return {
        "schema_version": 1,
        "ok": not errors,
        "phase": phase,
        "checked_at": time.time(),
        "run_dir": str(run_dir),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
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
    parser.add_argument("--phase", choices=("pre-seal", "sealed"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    if args.phase == "sealed" and output is not None and _is_within(output, run_dir):
        parser.error("sealed-mode --output must be outside RUN_DIR")
    report = audit_run(run_dir, args.phase)
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

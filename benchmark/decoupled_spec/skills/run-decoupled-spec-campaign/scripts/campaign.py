#!/usr/bin/env python3
"""Materialize and maintain an Agent-operated decoupled-spec campaign.

This module never launches a server, observer, or client. It expands a saved
campaign specification into exact case contracts and records the lifecycle of
the independent component commands run by an Agent.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT.parent))

from decoupled_spec.server.orchestrator import (  # noqa: E402
    config_to_dict,
    load_config,
)

SCHEMA_VERSION = 1
ATTEMPT_STATES = (
    "initialized",
    "servers_ready",
    "observer_active",
    "client_complete",
    "processes_stopped",
    "completed",
)
TERMINAL_ATTEMPT_STATES = {"completed", "incomplete"}
_CAMPAIGN_KEYS = {
    "schema_version",
    "campaign",
    "configs",
    "axes",
    "expected",
    "execution",
    "stop_gates",
}


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _read_client_result_metrics(
    run_dir: Path, *, max_proposed_drafts: int
) -> dict[str, Any]:
    """Derive campaign metrics from the fixed three-file client contract."""

    requests_path = run_dir / "client" / "requests.csv"
    with requests_path.open(encoding="utf-8", newline="") as stream:
        request_rows = list(csv.DictReader(stream))
    batch = _read_json(run_dir / "client" / "batch.json")
    content_path = run_dir / "client" / "content.json"
    content = json.loads(content_path.read_text(encoding="utf-8"))
    if not isinstance(content, list) or len(content) != len(request_rows):
        raise ValueError("fixed client result cardinality mismatch")

    verify = sum(int(row["spec_verify_ct"]) for row in request_rows)
    proposed_by_position = [
        json.loads(row["spec_num_proposed_drafts_by_position"]) for row in request_rows
    ]
    correct_by_position = [
        json.loads(row["spec_num_correct_drafts_by_position"]) for row in request_rows
    ]
    proposed = sum(sum(values) for values in proposed_by_position)
    correct = sum(sum(values) for values in correct_by_position)
    completion_tokens = int(batch["output_tokens"])
    if not 0 <= correct <= proposed <= max_proposed_drafts * verify:
        raise ValueError(
            "aggregate speculative counts violate "
            "0 <= correct <= proposed <= K * verify"
        )
    return {
        "batch_size": len(request_rows),
        "request_count": len(request_rows),
        "completed_count": len(request_rows),
        "failed_count": 0,
        "batch_elapsed_s": float(batch["batch_elapsed_latency_s"]),
        "prompt_tokens": sum(int(row["prompt_len"]) for row in request_rows),
        "completion_tokens": completion_tokens,
        "output_tokens_per_s": float(batch["batch_thpt"]),
        "e2e_mean_s": statistics.fmean(
            float(row["e2e_latency_s"]) for row in request_rows
        ),
        "spec_verify_ct": verify,
        "spec_num_proposed_drafts": proposed,
        "spec_num_correct_drafts": correct,
        "spec_accept_rate": correct / proposed if proposed else None,
        "spec_accept_length": float(batch["acclen"]),
        "spec_proposed_draft_length": float(batch["mean_valid_draft_len"]),
        "spec_draft_occupancy_rate": (
            proposed / (max_proposed_drafts * verify)
            if max_proposed_drafts and verify
            else 0.0
        ),
    }


def _repo_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"referenced config does not exist: {path}")
    return path


def _truthy(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return False
    raise ValueError(f"invalid boolean-like value: {value!r}")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_at_least(actual: Any, minimum: int, label: str) -> None:
    if type(actual) is not int or actual < minimum:
        raise ValueError(f"{label}: expected at least {minimum}, got {actual!r}")


def load_campaign_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    config = _read_mapping(config_path)
    unknown = set(config) - _CAMPAIGN_KEYS
    missing = _CAMPAIGN_KEYS - set(config)
    if unknown or missing:
        raise ValueError(
            f"invalid campaign keys: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"campaign schema_version must be {SCHEMA_VERSION}")
    for key in ("campaign", "configs", "axes", "expected", "execution"):
        if not isinstance(config[key], dict):
            raise ValueError(f"campaign.{key} must be a mapping")
    if not isinstance(config["stop_gates"], dict):
        raise ValueError("campaign.stop_gates must be a mapping")
    concurrency = config["execution"].get("max_concurrent_deployments")
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("execution.max_concurrent_deployments must be positive")
    return config_path, config


def _validate_axes(
    config: dict[str, Any],
) -> tuple[str, list[str], list[bool], list[int], list[int]]:
    axes = config["axes"]
    mode_target = axes.get("mode_target", "verifier")
    if not isinstance(mode_target, str) or mode_target not in {
        "verifier",
        "drafter",
    }:
        raise ValueError("axes.mode_target must be verifier or drafter")
    modes = axes.get("modes")
    ignore_eos_values = axes.get("ignore_eos")
    batch_sizes = axes.get("batch_sizes")
    output_lengths = axes.get("output_lengths")
    if modes != ["nonoverlap", "overlap"]:
        raise ValueError("axes.modes must be [nonoverlap, overlap]")
    if (
        not isinstance(ignore_eos_values, list)
        or not ignore_eos_values
        or any(type(value) is not bool for value in ignore_eos_values)
        or len(set(ignore_eos_values)) != len(ignore_eos_values)
    ):
        raise ValueError("axes.ignore_eos must be a non-empty list of unique booleans")
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise ValueError("axes.batch_sizes must be a non-empty list")
    if not isinstance(output_lengths, list) or not output_lengths:
        raise ValueError("axes.output_lengths must be a non-empty list")
    batch_sizes = [int(value) for value in batch_sizes]
    output_lengths = [int(value) for value in output_lengths]
    if any(value <= 0 for value in batch_sizes + output_lengths):
        raise ValueError("batch sizes and output lengths must be positive")
    if len(set(batch_sizes)) != len(batch_sizes) or len(set(output_lengths)) != len(
        output_lengths
    ):
        raise ValueError("campaign axes must not contain duplicate values")
    if batch_sizes != sorted(batch_sizes) or output_lengths != sorted(output_lengths):
        raise ValueError("batch and output axes must be in ascending capacity order")
    return mode_target, modes, ignore_eos_values, batch_sizes, output_lengths


def _config_sources(config: dict[str, Any]) -> dict[str, Path]:
    configs = config["configs"]
    server_by_mode = configs.get("server_by_mode")
    if not isinstance(server_by_mode, dict) or set(server_by_mode) != {
        "nonoverlap",
        "overlap",
    }:
        raise ValueError("configs.server_by_mode must define both campaign modes")
    return {
        "server_nonoverlap": _repo_file(server_by_mode["nonoverlap"]),
        "server_overlap": _repo_file(server_by_mode["overlap"]),
        "client": _repo_file(configs["client"]),
        "observer": _repo_file(configs["observer"]),
        "workload_inspection": _repo_file(configs["workload_inspection"]),
    }


def _validate_static_contract(
    config: dict[str, Any],
    sources: dict[str, Path],
    mode_target: str,
    modes: list[str],
    ignore_eos_values: list[bool],
    batch_sizes: list[int],
) -> dict[str, dict[str, Any]]:
    expected = config["expected"]
    max_batch_size = max(batch_sizes)
    client = _read_mapping(sources["client"])
    observer = _read_mapping(sources["observer"])
    workload_inspection = _read_mapping(sources["workload_inspection"])
    resolved: dict[str, dict[str, Any]] = {
        "client": client,
        "observer": observer,
        "workload_inspection": workload_inspection,
    }

    _require_equal(
        client.get("target_tokenizer", {}).get("model_path"),
        expected["target_model_path"],
        "client target tokenizer",
    )
    _require_equal(
        client.get("dataset", {}).get("format"),
        expected["dataset_format"],
        "client dataset format",
    )
    _require_equal(
        client.get("dataset", {}).get("path"),
        expected["dataset_path"],
        "client dataset path",
    )
    _require_equal(
        client.get("chat_template", {}).get("mode"),
        expected["chat_template_mode"],
        "client chat template",
    )
    _require_equal(
        client.get("chat_template", {}).get("enable_thinking"),
        expected["enable_thinking"],
        "client thinking mode",
    )
    _require_equal(
        client.get("generation", {}).get("temperature"),
        expected["temperature"],
        "client temperature",
    )
    _require_equal(
        ignore_eos_values,
        expected["ignore_eos_values"],
        "campaign ignore_eos values",
    )
    if client.get("generation", {}).get("ignore_eos") not in ignore_eos_values:
        raise ValueError("base client ignore_eos must be covered by the campaign axis")

    targets = observer.get("targets", {})
    if set(targets) != {"verifier", "drafter"}:
        raise ValueError("observer config must contain verifier and drafter")

    capacity = expected["capacity_reference"]
    _require_equal(
        _sha256_file(sources["workload_inspection"]),
        capacity["inspection_sha256"],
        "workload inspection sha256",
    )
    _require_equal(
        workload_inspection.get("batch_size"),
        max_batch_size,
        "workload inspection batch size",
    )
    prompt_lengths = workload_inspection.get("prompt_lengths", {})
    _require_equal(
        prompt_lengths.get("min"),
        capacity["prompt_len_min"],
        "workload inspection prompt min",
    )
    _require_equal(
        prompt_lengths.get("max"),
        capacity["prompt_len_max"],
        "workload inspection prompt max",
    )
    _require_equal(
        sum(int(value) for value in prompt_lengths.get("values", [])),
        capacity["prompt_len_sum"],
        "workload inspection prompt sum",
    )
    _require_equal(
        set(workload_inspection.get("requested_output_lengths", [])),
        {max(config["axes"]["output_lengths"])},
        "workload inspection output lengths",
    )
    required_tokens = capacity["prompt_len_sum"] + max_batch_size * (
        max(config["axes"]["output_lengths"]) + capacity["verify_reserve_per_request"]
    )
    _require_equal(
        required_tokens,
        capacity["required_tokens"],
        "capacity reference token requirement",
    )

    for mode in modes:
        alias = f"server_{mode}"
        server = config_to_dict(load_config(sources[alias]))
        resolved[alias] = server
        verifier = server["verifier"]
        drafter = server["drafter"]
        verifier_args = verifier["server_args"]
        drafter_args = drafter["server_args"]
        _require_equal(
            verifier_args.get("model_path"),
            expected["target_model_path"],
            f"{mode} verifier model_path",
        )
        _require_equal(
            verifier_args.get("tp_size"),
            expected["target_tp_size"],
            f"{mode} verifier tp_size",
        )
        _require_equal(
            verifier_args.get("random_seed"),
            expected["random_seed"],
            f"{mode} verifier random_seed",
        )
        _require_equal(
            verifier_args.get("disable_overlap_schedule"),
            mode == "nonoverlap" if mode_target == "verifier" else False,
            f"{mode} verifier schedule",
        )
        for field in (
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
        ):
            _require_equal(
                verifier_args.get(field), expected[field], f"{mode} verifier {field}"
            )
        replay_flag = expected["verifier_replayssm_flag"]
        if verifier_args.get(replay_flag) is not True:
            raise ValueError(f"{mode} verifier must set {replay_flag}=true")
        if bool(verifier_args.get("enable_linear_replayssm", False)):
            raise ValueError(
                f"{mode} verifier cannot combine decode ReplaySSM with spec ReplaySSM"
            )
        _require_equal(
            verifier_args.get("mamba_radix_cache_strategy"),
            "extra_buffer",
            f"{mode} verifier mamba_radix_cache_strategy",
        )
        _require_at_least(
            verifier_args.get("max_running_requests"),
            max_batch_size,
            f"{mode} verifier max_running_requests",
        )
        _require_equal(
            verifier_args.get("max_total_tokens"),
            expected["max_total_tokens"],
            f"{mode} verifier max_total_tokens",
        )
        _require_equal(
            verifier_args.get("cuda_graph_bs_decode"),
            expected["cuda_graph_bs_decode"],
            f"{mode} verifier decode CUDA Graph buckets",
        )
        required_verifier_slots = max_batch_size * int(
            expected["verifier_mamba_slots_per_request"]
        )
        if int(verifier_args.get("max_mamba_cache_size", 0)) < required_verifier_slots:
            raise ValueError(
                f"{mode} verifier max_mamba_cache_size cannot cover overlap slots: "
                f"required={required_verifier_slots}"
            )
        env = verifier.get("runtime", {}).get("env", {})
        _require_equal(
            env.get("SGLANG_RAGGED_VERIFY_MODE"),
            "static",
            f"{mode} verifier ragged verify mode",
        )
        if "SGLANG_DECOUPLED_SPEC_SNAPSHOT_WAIT_MS" in env:
            raise ValueError(f"{mode} verifier must not use a fixed snapshot wait")
        _require_equal(
            drafter_args.get("model_path"),
            expected["drafter_model_path"],
            f"{mode} drafter model_path",
        )
        _require_equal(
            drafter_args.get("tp_size"),
            expected["drafter_tp_size"],
            f"{mode} drafter tp_size",
        )
        _require_equal(
            drafter_args.get("random_seed"),
            expected["random_seed"],
            f"{mode} drafter random_seed",
        )
        if mode_target == "drafter":
            _require_equal(
                drafter_args.get("disable_overlap_schedule"),
                mode == "nonoverlap",
                f"{mode} drafter schedule",
            )
        _require_at_least(
            drafter_args.get("max_running_requests"),
            max_batch_size,
            f"{mode} drafter max_running_requests",
        )
        _require_equal(
            drafter_args.get("max_total_tokens"),
            expected["max_total_tokens"],
            f"{mode} drafter max_total_tokens",
        )
        _require_equal(
            drafter_args.get("cuda_graph_bs_decode"),
            expected["cuda_graph_bs_decode"],
            f"{mode} drafter decode CUDA Graph buckets",
        )
        for field in (
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
        ):
            _require_equal(
                drafter_args.get(field), expected[field], f"{mode} drafter {field}"
            )
        required_drafter_slots = max_batch_size * int(
            expected["drafter_mamba_slots_per_request"]
        )
        if int(drafter_args.get("max_mamba_cache_size", 0)) < required_drafter_slots:
            raise ValueError(
                f"{mode} drafter max_mamba_cache_size cannot cover rollback slots: "
                f"required={required_drafter_slots}"
            )
        if bool(drafter_args.get("enable_linear_replayssm", False)):
            raise ValueError(f"{mode} drafter ReplaySSM must remain disabled")
    return resolved


def _snapshot_sources(
    sources: dict[str, Path], resolved: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    snapshots = {}
    for alias, path in sources.items():
        try:
            relative = str(path.relative_to(REPO_ROOT))
        except ValueError:
            relative = str(path)
        snapshots[alias] = {
            "path": relative,
            "sha256": _sha256_file(path),
            "content": resolved[alias],
        }
    return snapshots


def _case_commands(case: dict[str, Any], snapshots: dict[str, Any]) -> dict[str, Any]:
    server_path = snapshots[case["config_aliases"]["server"]]["path"]
    client_path = snapshots["client"]["path"]
    observer_path = snapshots["observer"]["path"]
    return {
        "environment": {"PYTHONPATH": "python"},
        "server": [
            "python",
            "benchmark/decoupled_spec/server/server.py",
            "--config",
            server_path,
            "--run-dir",
            "<RUN_DIR>",
            "--runtime-dir",
            "<RUNTIME_DIR>",
        ],
        "runner": [
            "python",
            "benchmark/decoupled_spec/runner.py",
            "--client-config",
            client_path,
            "--observer-config",
            observer_path,
            "--run-dir",
            "<RUN_DIR>",
            "--server-manifest",
            "<RUNTIME_DIR>/server/manifest.json",
            "--verifier-rank",
            "0",
            "--batch-size",
            str(case["batch_size"]),
            "--output-len",
            str(case["output_len"]),
            "--ignore-eos" if case["ignore_eos"] else "--no-ignore-eos",
        ],
    }


def build_campaign_manifest(path: str | Path) -> dict[str, Any]:
    config_path, config = load_campaign_config(path)
    mode_target, modes, ignore_eos_values, batch_sizes, output_lengths = (
        _validate_axes(config)
    )
    sources = _config_sources(config)
    resolved = _validate_static_contract(
        config, sources, mode_target, modes, ignore_eos_values, batch_sizes
    )
    snapshots = _snapshot_sources(sources, resolved)
    cases = []
    run_name_prefix = config["campaign"].get(
        "run_name_prefix", config["campaign"].get("name")
    )
    if not isinstance(run_name_prefix, str) or not run_name_prefix:
        raise ValueError("campaign.run_name_prefix or campaign.name is required")
    for ignore_eos in ignore_eos_values:
        eos_label = "ignore-eos" if ignore_eos else "eos"
        for batch_size in batch_sizes:
            for output_len in output_lengths:
                for mode in modes:
                    case = {
                        "case_id": (
                            f"{eos_label}-{mode}-bs{batch_size}-out{output_len}"
                        ),
                        "run_name": (
                            f"{run_name_prefix}-bs{batch_size}-out{output_len}-"
                            f"{eos_label}-{mode}"
                        ),
                        "mode": mode,
                        "mode_target": mode_target,
                        "ignore_eos": ignore_eos,
                        "batch_size": batch_size,
                        "output_len": output_len,
                        "config_aliases": {
                            "server": f"server_{mode}",
                            "client": "client",
                            "observer": "observer",
                        },
                        "client_overrides": {
                            "batch_size": batch_size,
                            "output_len": output_len,
                            "ignore_eos": ignore_eos,
                        },
                    }
                    case["commands"] = _case_commands(case, snapshots)
                    case["case_sha256"] = _canonical_sha256(case)
                    cases.append(case)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign": copy.deepcopy(config["campaign"]),
        "source": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "content": config,
        },
        "case_count": len(cases),
        "mode_target": mode_target,
        "axes": copy.deepcopy(config["axes"]),
        "expected": copy.deepcopy(config["expected"]),
        "execution": copy.deepcopy(config["execution"]),
        "stop_gates": copy.deepcopy(config["stop_gates"]),
        "config_snapshots": snapshots,
        "cases": cases,
    }
    manifest["contract_sha256"] = _canonical_sha256(manifest)
    return manifest


def materialize_campaign(config_path: str | Path, campaign_dir: str | Path) -> Path:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    manifest = build_campaign_manifest(config_path)
    manifest_path = campaign_path / "campaign_manifest.json"
    ledger_path = campaign_path / "ledger.json"
    if manifest_path.exists() or ledger_path.exists():
        current = _read_json(manifest_path)
        if current.get("contract_sha256") != manifest["contract_sha256"]:
            raise ValueError(
                "campaign directory already belongs to a different campaign contract"
            )
        if not ledger_path.is_file():
            raise ValueError("campaign manifest exists without ledger.json")
        return campaign_path
    if campaign_path.exists() and any(campaign_path.iterdir()):
        raise ValueError(f"campaign directory is not empty: {campaign_path}")
    campaign_path.mkdir(parents=True, exist_ok=True)
    now = time.time()
    prerequisites = manifest["execution"].get("prerequisites", [])
    if not isinstance(prerequisites, list) or any(
        not isinstance(item, dict) or not item.get("id") for item in prerequisites
    ):
        raise ValueError("execution.prerequisites must be a list of identified gates")
    prerequisite_ids = [item["id"] for item in prerequisites]
    if len(set(prerequisite_ids)) != len(prerequisite_ids):
        raise ValueError("campaign prerequisite IDs must be unique")
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "campaign_contract_sha256": manifest["contract_sha256"],
        "created_at": now,
        "updated_at": now,
        "prerequisites": {
            item["id"]: {
                "state": item.get("initial_state", "pending"),
                "description": item.get("description"),
                "evidence": None,
                "updated_at": now,
            }
            for item in prerequisites
        },
        "cases": {
            case["case_id"]: {
                "case_sha256": case["case_sha256"],
                "blocked": None,
                "attempts": [],
            }
            for case in manifest["cases"]
        },
    }
    _write_json(manifest_path, manifest)
    _write_json(ledger_path, ledger)
    return campaign_path


@contextmanager
def _ledger_lock(campaign_dir: Path) -> Iterator[None]:
    lock_path = campaign_dir / ".ledger.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_campaign(
    campaign_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    manifest = _read_json(campaign_path / "campaign_manifest.json")
    ledger = _read_json(campaign_path / "ledger.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported campaign manifest schema")
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported campaign ledger schema")
    if ledger.get("campaign_contract_sha256") != manifest.get("contract_sha256"):
        raise ValueError("campaign ledger does not match campaign manifest")
    expected_prerequisites = {
        item["id"] for item in manifest["execution"].get("prerequisites", [])
    }
    if set(ledger.get("prerequisites", {})) != expected_prerequisites:
        raise ValueError("campaign prerequisite set does not match manifest")
    expected_cases = {case["case_id"]: case for case in manifest["cases"]}
    if set(ledger.get("cases", {})) != set(expected_cases):
        raise ValueError("campaign ledger case set does not match manifest")
    for case_id, case_state in ledger["cases"].items():
        if case_state.get("case_sha256") != expected_cases[case_id]["case_sha256"]:
            raise ValueError(f"ledger case hash mismatch: {case_id}")
    return campaign_path, manifest, ledger


def _find_case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in manifest["cases"]:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"unknown campaign case: {case_id}")


def _validate_live_config_sources(manifest: dict[str, Any]) -> None:
    for alias, snapshot in manifest["config_snapshots"].items():
        path = _repo_file(snapshot["path"])
        actual = _sha256_file(path)
        if actual != snapshot["sha256"]:
            raise ValueError(
                f"campaign source drift for {alias}: expected "
                f"{snapshot['sha256']}, got {actual}; materialize a new campaign"
            )


def register_attempt(
    campaign_dir: str | Path,
    case_id: str,
    run_dir: str | Path,
    runtime_dir: str | Path,
    note: str | None,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise ValueError(
            f"RUN_DIR does not exist; create it before registering the attempt: {run_path}"
        )
    runtime_path = Path(runtime_dir).expanduser().resolve()
    if not runtime_path.is_dir():
        raise ValueError(
            "RUNTIME_DIR does not exist; create it before registering the attempt: "
            f"{runtime_path}"
        )
    with _ledger_lock(campaign_path):
        _, manifest, ledger = load_campaign(campaign_path)
        case = _find_case(manifest, case_id)
        _validate_live_config_sources(manifest)
        pending_gates = [
            gate_id
            for gate_id, gate in ledger.get("prerequisites", {}).items()
            if gate.get("state") != "passed"
            or not gate.get("evidence")
            or not Path(gate["evidence"]).exists()
        ]
        if pending_gates:
            raise ValueError(
                "campaign prerequisites are not passed: " + ", ".join(pending_gates)
            )
        case_state = ledger["cases"][case_id]
        if case_state.get("blocked") is not None:
            raise ValueError(f"case is blocked: {case_state['blocked']}")
        for state in ledger["cases"].values():
            for attempt in state["attempts"]:
                if Path(attempt["run_dir"]).resolve() == run_path:
                    raise ValueError(f"RUN_DIR is already registered: {run_path}")
                if Path(attempt["runtime_dir"]).resolve() == runtime_path:
                    raise ValueError(
                        f"RUNTIME_DIR is already registered: {runtime_path}"
                    )
        if (
            case_state["attempts"]
            and case_state["attempts"][-1]["state"] not in TERMINAL_ATTEMPT_STATES
        ):
            raise ValueError("case already has an active attempt")
        now = time.time()
        attempt = {
            "attempt_id": f"attempt-{len(case_state['attempts']) + 1:03d}",
            "run_dir": str(run_path),
            "runtime_dir": str(runtime_path),
            "state": "initialized",
            "created_at": now,
            "updated_at": now,
            "history": [{"state": "initialized", "at": now, "note": note}],
        }
        case_state["attempts"].append(attempt)
        ledger["updated_at"] = now
        _write_json(campaign_path / "ledger.json", ledger)
        return copy.deepcopy(attempt)


def validate_server_gate(
    manifest: dict[str, Any],
    case: dict[str, Any],
    run_dir: str | Path,
    runtime_dir: str | Path,
) -> dict[str, Any]:
    """Validate the unified ready inventory and effective per-engine configs."""
    run_path = Path(run_dir).expanduser().resolve()
    runtime_path = Path(runtime_dir).expanduser().resolve()
    hard = manifest["stop_gates"].get("hard", {})
    errors = []
    expected_server = manifest["config_snapshots"][case["config_aliases"]["server"]][
        "content"
    ]
    config_path = run_path / "config.json"
    inventory_path = runtime_path / "server" / "manifest.json"
    if not config_path.is_file():
        errors.append(f"missing run config: {config_path}")
    elif _read_json(config_path).get("server") != expected_server:
        errors.append("config.json server section does not match campaign case")
    if not inventory_path.is_file():
        errors.append(f"missing unified server manifest: {inventory_path}")
        return {"ok": False, "errors": errors, "engines": []}

    inventory = _read_json(inventory_path)
    if inventory.get("state") not in {"ready", "stopped"}:
        errors.append(
            "server manifest must preserve a ready inventory, got "
            f"state={inventory.get('state')!r}"
        )
    engines = inventory.get("engines")
    if not isinstance(engines, list) or not engines:
        errors.append("server manifest must contain a non-empty engines list")
        engines = []
    role_counts = {"verifier": 0, "drafter": 0}
    seen_ids = set()
    engine_reports = []
    for engine in engines:
        if not isinstance(engine, dict):
            errors.append(f"invalid server engine entry: {engine!r}")
            continue
        engine_id = engine.get("engine_id")
        role = engine.get("role")
        if not isinstance(engine_id, str) or engine_id in seen_ids:
            errors.append(f"invalid or duplicate engine_id: {engine_id!r}")
            continue
        seen_ids.add(engine_id)
        if role not in role_counts:
            errors.append(f"{engine_id}: invalid role {role!r}")
            continue
        role_counts[role] += 1
        relative = engine.get("resolved_config_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            errors.append(f"{engine_id}: invalid resolved_config_path {relative!r}")
            continue
        engine_config_path = (runtime_path / relative).resolve()
        try:
            engine_config_path.relative_to(runtime_path)
        except ValueError:
            errors.append(f"{engine_id}: resolved_config_path escapes RUNTIME_DIR")
            continue
        if not engine_config_path.is_file():
            errors.append(f"{engine_id}: missing resolved config")
            continue
        engine_config = _read_json(engine_config_path)
        expected_role = expected_server[role]
        actual_args = engine_config.get("server_args")
        if not isinstance(actual_args, dict):
            errors.append(f"{engine_id}: server_args is not a mapping")
            continue
        for field, expected_value in expected_role["server_args"].items():
            if actual_args.get(field) != expected_value:
                errors.append(
                    f"{engine_id}: server_args.{field} differs from fleet template"
                )
        if engine_config.get("runtime") != expected_role["runtime"]:
            errors.append(f"{engine_id}: runtime differs from fleet template")
        for field, gate_name in (
            ("max_total_tokens", f"min_{role}_max_total_num_tokens"),
            ("max_mamba_cache_size", f"min_{role}_mamba_cache_size"),
        ):
            minimum = int(hard.get(gate_name, 0))
            if int(actual_args.get(field, 0) or 0) < minimum:
                errors.append(
                    f"{engine_id}: {field} below {minimum}: "
                    f"observed={actual_args.get(field)!r}"
                )
        placements = engine.get("rank_placements")
        expected_tp = int(expected_role["server_args"].get("tp_size", 1))
        if not isinstance(placements, list) or len(placements) != expected_tp:
            errors.append(
                f"{engine_id}: rank placement count does not equal tp_size={expected_tp}"
            )
        engine_reports.append(
            {
                "engine_id": engine_id,
                "role": role,
                "rank": engine.get("rank"),
                "tp_size": expected_tp,
            }
        )
    for role, count in role_counts.items():
        expected_count = int(expected_server[role]["replicas"])
        if count != expected_count:
            errors.append(
                f"server manifest {role} count mismatch: "
                f"expected={expected_count}, observed={count}"
            )
    return {
        "ok": not errors,
        "errors": errors,
        "manifest_state": inventory.get("state"),
        "engines": engine_reports,
    }


def transition_attempt(
    campaign_dir: str | Path,
    case_id: str,
    state: str,
    note: str | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if state not in ATTEMPT_STATES and state != "incomplete":
        raise ValueError(f"unknown attempt state: {state}")
    campaign_path, manifest, ledger = load_campaign(campaign_dir)
    case = _find_case(manifest, case_id)
    attempts = ledger["cases"][case_id]["attempts"]
    if not attempts:
        raise ValueError("case has no registered attempt")
    attempt = attempts[-1]
    if attempt["state"] in TERMINAL_ATTEMPT_STATES:
        raise ValueError(f"attempt is already terminal: {attempt['state']}")
    if state == "incomplete":
        if not failed_stage or not error:
            raise ValueError(
                "incomplete transition requires --failed-stage and --error"
            )
        verification = None
    else:
        current_index = ATTEMPT_STATES.index(attempt["state"])
        requested_index = ATTEMPT_STATES.index(state)
        if requested_index != current_index + 1:
            raise ValueError(
                "attempt transitions cannot skip lifecycle gates: "
                f"{attempt['state']} -> {state}"
            )
        if state == "servers_ready":
            verification = validate_server_gate(
                manifest,
                case,
                attempt["run_dir"],
                attempt["runtime_dir"],
            )
            if not verification["ok"]:
                raise ValueError(
                    "unified server inventory failed the campaign gate: "
                    + "; ".join(verification["errors"])
                )
        else:
            verification = None

    with _ledger_lock(campaign_path):
        _, _, current_ledger = load_campaign(campaign_path)
        current = current_ledger["cases"][case_id]["attempts"][-1]
        if (
            current["attempt_id"] != attempt["attempt_id"]
            or current["state"] != attempt["state"]
        ):
            raise ValueError("attempt changed while transition was being validated")
        now = time.time()
        current["state"] = state
        current["updated_at"] = now
        history = {"state": state, "at": now, "note": note}
        if state == "incomplete":
            current["failed_stage"] = failed_stage
            current["error"] = error
            history.update({"failed_stage": failed_stage, "error": error})
        if verification is not None:
            current["server_gate"] = verification
        current["history"].append(history)
        current_ledger["updated_at"] = now
        _write_json(campaign_path / "ledger.json", current_ledger)
        return copy.deepcopy(current)


def set_case_block(
    campaign_dir: str | Path,
    case_id: str,
    blocked: bool,
    reason: str | None,
    evidence_case_id: str | None,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    with _ledger_lock(campaign_path):
        _, manifest, ledger = load_campaign(campaign_path)
        _find_case(manifest, case_id)
        state = ledger["cases"][case_id]
        if any(attempt["state"] == "completed" for attempt in state["attempts"]):
            raise ValueError("cannot block an already completed case")
        if (
            state["attempts"]
            and state["attempts"][-1]["state"] not in TERMINAL_ATTEMPT_STATES
        ):
            raise ValueError("cannot block a case with an active attempt")
        if blocked:
            if not reason:
                raise ValueError("blocking a case requires --reason")
            state["blocked"] = {
                "reason": reason,
                "evidence_case_id": evidence_case_id,
                "at": time.time(),
            }
        else:
            state["blocked"] = None
        ledger["updated_at"] = time.time()
        _write_json(campaign_path / "ledger.json", ledger)
        return copy.deepcopy(state)


def set_prerequisite(
    campaign_dir: str | Path,
    gate_id: str,
    state: str,
    evidence: str | None,
    note: str | None,
) -> dict[str, Any]:
    if state not in {"pending", "passed", "failed"}:
        raise ValueError(f"invalid prerequisite state: {state}")
    if state == "passed" and not evidence:
        raise ValueError("passing a prerequisite requires --evidence")
    if evidence:
        evidence_path = Path(evidence).expanduser().resolve()
        if not evidence_path.exists():
            raise ValueError(f"prerequisite evidence does not exist: {evidence_path}")
        evidence = str(evidence_path)
    campaign_path = Path(campaign_dir).expanduser().resolve()
    with _ledger_lock(campaign_path):
        _, _, ledger = load_campaign(campaign_path)
        if gate_id not in ledger.get("prerequisites", {}):
            raise ValueError(f"unknown campaign prerequisite: {gate_id}")
        gate = ledger["prerequisites"][gate_id]
        gate.update(
            {
                "state": state,
                "evidence": evidence,
                "note": note,
                "updated_at": time.time(),
            }
        )
        ledger["updated_at"] = time.time()
        _write_json(campaign_path / "ledger.json", ledger)
        return copy.deepcopy(gate)


def campaign_status(campaign_dir: str | Path) -> dict[str, Any]:
    _, manifest, ledger = load_campaign(campaign_dir)
    counts: dict[str, int] = {}
    cases = []
    for case in manifest["cases"]:
        state = ledger["cases"][case["case_id"]]
        attempts = state["attempts"]
        if any(attempt["state"] == "completed" for attempt in attempts):
            label = "completed"
        elif state.get("blocked") is not None:
            label = "blocked"
        elif attempts and attempts[-1]["state"] not in TERMINAL_ATTEMPT_STATES:
            label = attempts[-1]["state"]
        elif attempts:
            label = "retryable_incomplete"
        else:
            label = "pending"
        counts[label] = counts.get(label, 0) + 1
        cases.append(
            {
                "case_id": case["case_id"],
                "status": label,
                "attempt_count": len(attempts),
                "last_run_dir": attempts[-1]["run_dir"] if attempts else None,
            }
        )
    return {
        "campaign": manifest["campaign"]["name"],
        "case_count": manifest["case_count"],
        "counts": counts,
        "prerequisites": ledger.get("prerequisites", {}),
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", required=True)
    materialize.add_argument("--campaign-dir", required=True)

    show = subparsers.add_parser("show-case")
    show.add_argument("--campaign-dir", required=True)
    show.add_argument("--case-id", required=True)

    register = subparsers.add_parser("register-attempt")
    register.add_argument("--campaign-dir", required=True)
    register.add_argument("--case-id", required=True)
    register.add_argument("--run-dir", required=True)
    register.add_argument("--runtime-dir", required=True)
    register.add_argument("--note")

    transition = subparsers.add_parser("transition")
    transition.add_argument("--campaign-dir", required=True)
    transition.add_argument("--case-id", required=True)
    transition.add_argument(
        "--state", choices=ATTEMPT_STATES[1:] + ("incomplete",), required=True
    )
    transition.add_argument("--note")
    transition.add_argument("--failed-stage")
    transition.add_argument("--error")

    block = subparsers.add_parser("block-case")
    block.add_argument("--campaign-dir", required=True)
    block.add_argument("--case-id", required=True)
    block.add_argument("--reason", required=True)
    block.add_argument("--evidence-case-id")

    unblock = subparsers.add_parser("unblock-case")
    unblock.add_argument("--campaign-dir", required=True)
    unblock.add_argument("--case-id", required=True)

    prerequisite = subparsers.add_parser("set-prerequisite")
    prerequisite.add_argument("--campaign-dir", required=True)
    prerequisite.add_argument("--gate-id", required=True)
    prerequisite.add_argument(
        "--state", choices=("pending", "passed", "failed"), required=True
    )
    prerequisite.add_argument("--evidence")
    prerequisite.add_argument("--note")

    status = subparsers.add_parser("status")
    status.add_argument("--campaign-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate":
        manifest = build_campaign_manifest(args.config)
        output = {
            "valid": True,
            "campaign": manifest["campaign"]["name"],
            "case_count": manifest["case_count"],
            "contract_sha256": manifest["contract_sha256"],
        }
    elif args.command == "materialize":
        output = {
            "campaign_dir": str(materialize_campaign(args.config, args.campaign_dir))
        }
    elif args.command == "show-case":
        _, manifest, _ = load_campaign(args.campaign_dir)
        output = _find_case(manifest, args.case_id)
    elif args.command == "register-attempt":
        output = register_attempt(
            args.campaign_dir,
            args.case_id,
            args.run_dir,
            args.runtime_dir,
            args.note,
        )
    elif args.command == "transition":
        output = transition_attempt(
            args.campaign_dir,
            args.case_id,
            args.state,
            args.note,
            args.failed_stage,
            args.error,
        )
    elif args.command == "block-case":
        output = set_case_block(
            args.campaign_dir,
            args.case_id,
            True,
            args.reason,
            args.evidence_case_id,
        )
    elif args.command == "unblock-case":
        output = set_case_block(args.campaign_dir, args.case_id, False, None, None)
    elif args.command == "set-prerequisite":
        output = set_prerequisite(
            args.campaign_dir,
            args.gate_id,
            args.state,
            args.evidence,
            args.note,
        )
    else:
        output = campaign_status(args.campaign_dir)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

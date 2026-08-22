#!/usr/bin/env python3
"""Materialize and maintain an Agent-operated decoupled-spec campaign.

This module never launches a server, collector, or client.  It expands a
static matrix into exact case contracts and records the lifecycle of the
independent role commands run by an Agent.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
SERVER_ROOT = BENCHMARK_ROOT / "server-side"
sys.path.insert(0, str(SERVER_ROOT))

from config import resolve_role_config, validate_role_pair  # noqa: E402

SCHEMA_VERSION = 1
ATTEMPT_STATES = (
    "initialized",
    "servers_ready",
    "collector_active",
    "client_complete",
    "processes_stopped",
    "derived",
    "pre_seal_audited",
    "sealed",
    "verified",
)
TERMINAL_ATTEMPT_STATES = {"verified", "incomplete"}
_MATRIX_KEYS = {
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


def spec_metric_invariant_errors(
    summary: dict[str, Any],
    raw_responses: list[dict[str, Any]],
    *,
    batch_size: int,
    output_len: int,
    max_proposed_drafts: int,
) -> list[str]:
    """Validate lossless accounting identities within one measured run."""

    errors = []
    verify = int(summary.get("spec_verify_ct", 0) or 0)
    proposed = int(summary.get("spec_num_proposed_drafts", 0) or 0)
    correct = int(summary.get("spec_num_correct_drafts", 0) or 0)
    completion_tokens = int(summary.get("completion_tokens", 0) or 0)
    if not 0 <= correct <= proposed <= max_proposed_drafts * verify:
        errors.append(
            "aggregate speculative counts violate "
            "0 <= correct <= proposed <= K * verify"
        )
    expected_rates = (
        ("spec_accept_rate", correct / proposed if proposed else None),
        (
            "spec_draft_occupancy_rate",
            (
                proposed / (max_proposed_drafts * verify)
                if max_proposed_drafts and verify
                else 0.0
            ),
        ),
        ("spec_accept_length", completion_tokens / verify if verify else 0.0),
    )
    for field, expected in expected_rates:
        actual = summary.get(field)
        if expected is None:
            matches = actual is None
        else:
            matches = actual is not None and math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
            )
        if not matches:
            errors.append(f"{field} does not match its count-derived value")

    if len(raw_responses) != batch_size:
        errors.append("raw response count does not equal batch_size")
        return errors
    seen_indices = set()
    raw_verify = raw_proposed = raw_correct = raw_completion = 0
    for response in raw_responses:
        index = int(response.get("index", -1))
        if index in seen_indices or not 0 <= index < batch_size:
            errors.append(f"invalid or duplicate raw response index {index}")
            continue
        seen_indices.add(index)
        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list) or len(output_ids) != output_len:
            errors.append(
                f"response {index} output_ids length does not equal output_len"
            )
        meta = response.get("meta_info")
        if not isinstance(meta, dict):
            errors.append(f"response {index} has no meta_info mapping")
            continue
        req_verify = int(meta.get("spec_verify_ct", 0) or 0)
        req_proposed = int(meta.get("spec_num_proposed_drafts", 0) or 0)
        req_correct = int(meta.get("spec_num_correct_drafts", 0) or 0)
        req_completion = int(meta.get("completion_tokens", 0) or 0)
        if req_completion != output_len:
            errors.append(
                f"response {index} completion_tokens does not equal output_len"
            )
        if not 0 <= req_correct <= req_proposed <= max_proposed_drafts * req_verify:
            errors.append(f"response {index} speculative counts are inconsistent")
        histogram = meta.get("spec_correct_drafts_histogram")
        # Runtime histograms omit trailing zero-count bins. Bin i still means
        # exactly i accepted drafts, so lengths 1..K+1 are equivalent after
        # right-padding with zeros.
        if (
            not isinstance(histogram, list)
            or not histogram
            or len(histogram) > max_proposed_drafts + 1
        ):
            errors.append(f"response {index} has an invalid acceptance histogram")
        else:
            histogram = [int(value) for value in histogram]
            if sum(histogram) != req_verify:
                errors.append(f"response {index} histogram count does not equal verify")
            if sum(i * count for i, count in enumerate(histogram)) != req_correct:
                errors.append(
                    f"response {index} histogram weighted sum does not equal correct"
                )
        raw_verify += req_verify
        raw_proposed += req_proposed
        raw_correct += req_correct
        raw_completion += req_completion
    if (raw_verify, raw_proposed, raw_correct, raw_completion) != (
        verify,
        proposed,
        correct,
        completion_tokens,
    ):
        errors.append("aggregate speculative counts do not equal raw response sums")
    return errors


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


def load_matrix_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    config = _read_mapping(config_path)
    unknown = set(config) - _MATRIX_KEYS
    missing = _MATRIX_KEYS - set(config)
    if unknown or missing:
        raise ValueError(
            f"invalid matrix keys: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"matrix schema_version must be {SCHEMA_VERSION}")
    for key in ("campaign", "configs", "axes", "expected", "execution"):
        if not isinstance(config[key], dict):
            raise ValueError(f"matrix.{key} must be a mapping")
    if not isinstance(config["stop_gates"], dict):
        raise ValueError("matrix.stop_gates must be a mapping")
    return config_path, config


def _validate_axes(config: dict[str, Any]) -> tuple[list[str], list[int], list[int]]:
    axes = config["axes"]
    modes = axes.get("modes")
    batch_sizes = axes.get("batch_sizes")
    output_lengths = axes.get("output_lengths")
    if modes != ["nonoverlap", "overlap"]:
        raise ValueError("axes.modes must be [nonoverlap, overlap]")
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
        raise ValueError("matrix axes must not contain duplicate values")
    if batch_sizes != sorted(batch_sizes) or output_lengths != sorted(output_lengths):
        raise ValueError("batch and output axes must be in ascending capacity order")
    return modes, batch_sizes, output_lengths


def _config_sources(config: dict[str, Any]) -> dict[str, Path]:
    configs = config["configs"]
    verifier_by_mode = configs.get("verifier_by_mode")
    if not isinstance(verifier_by_mode, dict) or set(verifier_by_mode) != {
        "nonoverlap",
        "overlap",
    }:
        raise ValueError("configs.verifier_by_mode must define both matrix modes")
    return {
        "verifier_nonoverlap": _repo_file(verifier_by_mode["nonoverlap"]),
        "verifier_overlap": _repo_file(verifier_by_mode["overlap"]),
        "drafter": _repo_file(configs["drafter"]),
        "client": _repo_file(configs["client"]),
        "observability": _repo_file(configs["observability"]),
        "workload_inspection": _repo_file(configs["workload_inspection"]),
    }


def _validate_static_contract(
    config: dict[str, Any],
    sources: dict[str, Path],
    modes: list[str],
    batch_sizes: list[int],
) -> dict[str, dict[str, Any]]:
    expected = config["expected"]
    max_batch_size = max(batch_sizes)
    drafter = resolve_role_config(sources["drafter"], "drafter")
    client = _read_mapping(sources["client"])
    observability = _read_mapping(sources["observability"])
    workload_inspection = _read_mapping(sources["workload_inspection"])
    resolved: dict[str, dict[str, Any]] = {
        "drafter": drafter,
        "client": client,
        "observability": observability,
        "workload_inspection": workload_inspection,
    }

    drafter_args = drafter["server_args"]
    _require_equal(
        drafter_args.get("model_path"),
        expected["drafter_model_path"],
        "drafter model_path",
    )
    _require_equal(
        drafter_args.get("tp_size"), expected["drafter_tp_size"], "drafter tp_size"
    )
    _require_equal(
        drafter_args.get("random_seed"),
        expected["random_seed"],
        "drafter random_seed",
    )
    _require_equal(
        drafter["runtime"].get("cuda_visible_devices"),
        expected["drafter_cuda_visible_devices"],
        "drafter CUDA_VISIBLE_DEVICES",
    )
    _require_equal(
        drafter_args.get("max_running_requests"),
        max_batch_size,
        "drafter max_running_requests",
    )
    _require_equal(
        drafter_args.get("max_total_tokens"),
        expected["max_total_tokens"],
        "drafter max_total_tokens",
    )
    _require_equal(
        drafter_args.get("cuda_graph_bs_decode"),
        expected["cuda_graph_bs_decode"],
        "drafter decode CUDA Graph buckets",
    )
    required_drafter_slots = max_batch_size * int(
        expected["drafter_mamba_slots_per_request"]
    )
    if int(drafter_args.get("max_mamba_cache_size", 0)) < required_drafter_slots:
        raise ValueError(
            "drafter max_mamba_cache_size cannot cover active + rollback slots: "
            f"required={required_drafter_slots}"
        )
    if bool(drafter_args.get("enable_linear_replayssm", False)):
        raise ValueError("drafter ReplaySSM must remain disabled")

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
        client.get("generation", {}).get("ignore_eos"),
        expected["ignore_eos"],
        "client ignore_eos",
    )

    targets = observability.get("targets", {})
    if set(targets) != {"verifier", "drafter"}:
        raise ValueError("observability config must contain verifier and drafter")

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
        capacity["bs64_out32768_required_tokens"],
        "capacity reference token requirement",
    )

    for mode in modes:
        alias = f"verifier_{mode}"
        verifier = resolve_role_config(sources[alias], "verifier")
        validate_role_pair(verifier, drafter)
        resolved[alias] = verifier
        verifier_args = verifier["server_args"]
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
            verifier["runtime"].get("cuda_visible_devices"),
            expected["target_cuda_visible_devices"],
            f"{mode} verifier CUDA_VISIBLE_DEVICES",
        )
        _require_equal(
            verifier_args.get("disable_overlap_schedule"),
            mode == "nonoverlap",
            f"{mode} schedule",
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
        _require_equal(
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
        if not _truthy(env.get("SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND", "0")):
            raise ValueError(f"{mode} verifier must use the C++ data plane")
        _require_equal(
            env.get("SGLANG_RAGGED_VERIFY_MODE"),
            "static",
            f"{mode} verifier ragged verify mode",
        )
        if "SGLANG_DECOUPLED_SPEC_SNAPSHOT_WAIT_MS" in env:
            raise ValueError(f"{mode} verifier must not use a fixed snapshot wait")

    if not _truthy(
        drafter.get("runtime", {})
        .get("env", {})
        .get("SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND", "0")
    ):
        raise ValueError("drafter must use the C++ data plane")
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
    verifier_path = snapshots[case["config_aliases"]["verifier"]]["path"]
    drafter_path = snapshots["drafter"]["path"]
    client_path = snapshots["client"]["path"]
    observability_path = snapshots["observability"]["path"]
    return {
        "environment": {"PYTHONPATH": "python"},
        "verifier": [
            "python",
            "benchmark/decoupled_spec/server-side/verifier_server.py",
            "--config",
            verifier_path,
            "--run-dir",
            "<RUN_DIR>",
        ],
        "drafter": [
            "python",
            "benchmark/decoupled_spec/server-side/drafter_server.py",
            "--config",
            drafter_path,
            "--run-dir",
            "<RUN_DIR>",
        ],
        "collector": [
            "python",
            "benchmark/decoupled_spec/common/collector.py",
            "--config",
            observability_path,
            "--run-dir",
            "<RUN_DIR>",
        ],
        "client": [
            "python",
            "benchmark/decoupled_spec/client-side/client.py",
            "--config",
            client_path,
            "--run-dir",
            "<RUN_DIR>",
            "--batch-size",
            str(case["batch_size"]),
            "--output-len",
            str(case["output_len"]),
        ],
    }


def build_campaign_manifest(path: str | Path) -> dict[str, Any]:
    config_path, config = load_matrix_config(path)
    modes, batch_sizes, output_lengths = _validate_axes(config)
    sources = _config_sources(config)
    resolved = _validate_static_contract(config, sources, modes, batch_sizes)
    snapshots = _snapshot_sources(sources, resolved)
    cases = []
    for mode in modes:
        for batch_size in batch_sizes:
            for output_len in output_lengths:
                case = {
                    "case_id": f"{mode}-bs{batch_size}-out{output_len // 1024}k",
                    "run_name": (
                        "qwen35-27b-tp4-draft-0.8b-tp1-k3-f1-"
                        f"bs{batch_size}-dapo-thinking-out{output_len}-{mode}-"
                        "cpp-gpu-tail-replayssm"
                    ),
                    "mode": mode,
                    "batch_size": batch_size,
                    "output_len": output_len,
                    "config_aliases": {
                        "verifier": f"verifier_{mode}",
                        "drafter": "drafter",
                        "client": "client",
                        "observability": "observability",
                    },
                    "client_overrides": {
                        "batch_size": batch_size,
                        "output_len": output_len,
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
                "campaign directory already belongs to a different matrix contract"
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
    raise ValueError(f"unknown matrix case: {case_id}")


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
    campaign_dir: str | Path, case_id: str, run_dir: str | Path, note: str | None
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    if not (run_path / "provenance" / "run_start.json").is_file():
        raise ValueError("RUN_DIR must already contain provenance/run_start.json")
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
        if (
            case_state["attempts"]
            and case_state["attempts"][-1]["state"] not in TERMINAL_ATTEMPT_STATES
        ):
            raise ValueError("case already has an active attempt")
        provenance = _read_json(run_path / "provenance" / "run_start.json")
        _require_equal(provenance.get("name"), case["run_name"], "RUN_DIR name")
        now = time.time()
        attempt = {
            "attempt_id": f"attempt-{len(case_state['attempts']) + 1:03d}",
            "run_dir": str(run_path),
            "state": "initialized",
            "created_at": now,
            "updated_at": now,
            "history": [{"state": "initialized", "at": now, "note": note}],
        }
        case_state["attempts"].append(attempt)
        ledger["updated_at"] = now
        _write_json(campaign_path / "ledger.json", ledger)
        return copy.deepcopy(attempt)


def _load_audit_function():
    path = (
        BENCHMARK_ROOT
        / "skills"
        / "audit-decoupled-spec-artifacts"
        / "scripts"
        / "audit_run.py"
    )
    spec = importlib.util.spec_from_file_location("decoupled_spec_audit_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audit helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_run


def _expected_client_config(
    manifest: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    client = copy.deepcopy(manifest["config_snapshots"]["client"]["content"])
    client.setdefault("batch", {})["size"] = case["batch_size"]
    client.setdefault("generation", {})["output_len"] = case["output_len"]
    return client


def validate_boot_gate(manifest: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Validate measured pool capacity and C++ selection from role boot logs."""
    run_path = Path(run_dir).expanduser().resolve()
    hard = manifest["stop_gates"].get("hard", {})
    verifier_log = run_path / "logs" / "verifier.log"
    drafter_log = run_path / "logs" / "drafter.log"
    if not verifier_log.is_file() or not drafter_log.is_file():
        return {
            "ok": False,
            "errors": ["boot gate requires logs/verifier.log and logs/drafter.log"],
        }
    verifier_text = verifier_log.read_text(encoding="utf-8", errors="replace")
    drafter_text = drafter_log.read_text(encoding="utf-8", errors="replace")

    def values(pattern: str, text: str) -> list[int]:
        return [int(value) for value in re.findall(pattern, text)]

    verifier_tokens = values(r"max_total_num_tokens=(\d+)", verifier_text)
    drafter_tokens = values(r"max_total_num_tokens=(\d+)", drafter_text)
    verifier_mamba = values(r"max_mamba_cache_size:\s*(\d+)", verifier_text)
    drafter_mamba = values(r"max_mamba_cache_size:\s*(\d+)", drafter_text)
    errors = []
    for label, observed, gate_name in (
        (
            "verifier max_total_num_tokens",
            verifier_tokens,
            "min_verifier_boot_max_total_num_tokens",
        ),
        (
            "drafter max_total_num_tokens",
            drafter_tokens,
            "min_drafter_boot_max_total_num_tokens",
        ),
        (
            "verifier max_mamba_cache_size",
            verifier_mamba,
            "min_verifier_boot_mamba_cache_size",
        ),
        (
            "drafter max_mamba_cache_size",
            drafter_mamba,
            "min_drafter_boot_mamba_cache_size",
        ),
    ):
        minimum = int(hard.get(gate_name, 0))
        if not observed:
            errors.append(f"{label} is absent from role log")
        elif min(observed) < minimum:
            errors.append(f"{label} below {minimum}: observed={observed}")
    cpp_marker = "Loading the decoupled-spec C++ data plane"
    if hard.get("require_cpp_dataplane_selection_log", False):
        if cpp_marker not in verifier_text:
            errors.append("verifier C++ data-plane selection log is absent")
        if cpp_marker not in drafter_text:
            errors.append("drafter C++ data-plane selection log is absent")
    expected = manifest["expected"]
    if len(verifier_mamba) < int(expected["target_tp_size"]):
        errors.append(
            "verifier Mamba allocation log does not cover every TP rank: "
            f"observed={len(verifier_mamba)}"
        )
    if len(drafter_mamba) < int(expected["drafter_tp_size"]):
        errors.append(
            "drafter Mamba allocation log does not cover every TP rank: "
            f"observed={len(drafter_mamba)}"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "verifier_max_total_num_tokens": verifier_tokens,
        "drafter_max_total_num_tokens": drafter_tokens,
        "verifier_max_mamba_cache_size": verifier_mamba,
        "drafter_max_mamba_cache_size": drafter_mamba,
    }


def validate_verified_run(
    manifest: dict[str, Any], case: dict[str, Any], run_dir: str | Path
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    audit = _load_audit_function()(run_path, "sealed")
    errors = list(audit.get("errors", []))
    boot_gate = validate_boot_gate(manifest, run_path)
    errors.extend(boot_gate["errors"])
    snapshots = manifest["config_snapshots"]
    expected_configs = {
        "roles/verifier/resolved_config.json": snapshots[
            case["config_aliases"]["verifier"]
        ]["content"],
        "roles/drafter/resolved_config.json": snapshots["drafter"]["content"],
        "client/resolved_config.json": _expected_client_config(manifest, case),
        "observability/resolved_config.json": snapshots["observability"]["content"],
    }
    for relative, expected in expected_configs.items():
        path = run_path / relative
        if not path.is_file():
            errors.append(f"missing exact-config artifact: {path}")
            continue
        actual = _read_json(path)
        if actual != expected:
            errors.append(f"resolved config does not match matrix case: {path}")

    summary_path = run_path / "client" / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    hard = manifest["stop_gates"].get("hard", {})
    if hard.get("require_completed_count_equals_batch_size", False) and int(
        summary.get("completed_count", -1)
    ) != int(case["batch_size"]):
        errors.append("client completed_count does not equal case batch_size")
    if hard.get("require_completion_tokens_equals_batch_times_output_len", False):
        expected_tokens = int(case["batch_size"]) * int(case["output_len"])
        if int(summary.get("completion_tokens", -1)) != expected_tokens:
            errors.append(
                "client completion_tokens does not equal batch_size * output_len: "
                f"expected={expected_tokens}, got={summary.get('completion_tokens')!r}"
            )
    for field, gate in (
        ("spec_verify_ct", "min_spec_verify_ct"),
        ("spec_num_proposed_drafts", "min_spec_num_proposed_drafts"),
    ):
        minimum = int(hard.get(gate, 0))
        if int(summary.get(field, 0) or 0) < minimum:
            errors.append(f"{field} is below matrix hard gate {minimum}")
    proposed_drafts = int(summary.get("spec_num_proposed_drafts", 0) or 0)
    correct_drafts = int(summary.get("spec_num_correct_drafts", 0) or 0)
    accept_rate = summary.get("spec_accept_rate")
    occupancy = summary.get("spec_draft_occupancy_rate")
    for field, value in (
        ("spec_accept_rate", accept_rate),
        ("spec_draft_occupancy_rate", occupancy),
    ):
        if (
            field == "spec_accept_rate"
            and value is None
            and proposed_drafts == 0
            and correct_drafts == 0
        ):
            continue
        if (
            value is None
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            errors.append(f"{field} must be finite and in [0, 1]")
    raw_path = run_path / "client" / "raw_batch_response.json"
    if raw_path.is_file():
        raw_responses = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(raw_responses, list) or any(
            not isinstance(response, dict) for response in raw_responses
        ):
            errors.append("client raw_batch_response.json must be a list of objects")
        else:
            errors.extend(
                spec_metric_invariant_errors(
                    summary,
                    raw_responses,
                    batch_size=int(case["batch_size"]),
                    output_len=int(case["output_len"]),
                    max_proposed_drafts=int(
                        manifest["expected"]["speculative_num_steps"]
                    ),
                )
            )
    else:
        errors.append("client raw_batch_response.json is missing")
    return {
        "ok": not errors,
        "case_id": case["case_id"],
        "run_dir": str(run_path),
        "audit": audit,
        "boot_gate": boot_gate,
        "errors": errors,
        "summary": summary,
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
            verification = validate_boot_gate(manifest, attempt["run_dir"])
            if not verification["ok"]:
                raise ValueError(
                    "role boot logs failed the capacity/data-plane gate: "
                    + "; ".join(verification["errors"])
                )
        elif state == "verified":
            verification = validate_verified_run(manifest, case, attempt["run_dir"])
            if not verification["ok"]:
                raise ValueError(
                    "sealed RUN_DIR failed matrix verification: "
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
            if state == "servers_ready":
                current["boot_gate"] = verification
            else:
                current["sealed_verification"] = {
                    "checked_at": verification["audit"].get("checked_at"),
                    "warning_count": len(verification["audit"].get("warnings", [])),
                    "sha256sums_sha256": _sha256_file(
                        Path(current["run_dir"]) / "SHA256SUMS"
                    ),
                }
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
        if any(attempt["state"] == "verified" for attempt in state["attempts"]):
            raise ValueError("cannot block an already verified case")
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
        if any(attempt["state"] == "verified" for attempt in attempts):
            label = "verified"
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
            args.campaign_dir, args.case_id, args.run_dir, args.note
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

"""Single-file scheduler-cycle profile contract for decoupled verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION = 1
DECOUPLED_VERIFY_PROFILE_ABI_VERSION = 1
DECOUPLED_VERIFY_PROFILE_KIND = "sglang_decoupled_verify_scheduler_cycle_profile"
DECOUPLED_VERIFY_PROFILE_JOB_KIND = (
    "sglang_decoupled_verify_scheduler_cycle_profile_job"
)
DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK = "scheduler_verify_commit_gap"
DECOUPLED_VERIFY_PROFILE_COST_SCOPE = "decoupled_verifier_scheduler_cycle"
DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR = "trimmed_mean_10pct"
DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER = "forward_stream_mock_gpu_tail_selector"
DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE = "daemon_drop"
DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE = "measurement_midpoint"
DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE = "full"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically replace one local JSON artifact and fsync its contents."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping in {resolved}.")
    return value


def estimate_scheduler_cycle_cost(costs_ms: list[float]) -> float:
    """Return a deterministic 10%-trimmed mean over positive finite samples."""

    values = sorted(
        float(value)
        for value in costs_ms
        if math.isfinite(float(value)) and float(value) > 0
    )
    if len(values) != len(costs_ms) or not values:
        raise ValueError("Scheduler-cycle samples must be positive and finite.")
    trim = int(len(values) * 0.10)
    if trim and len(values) - 2 * trim > 0:
        values = values[trim:-trim]
    return sum(values) / len(values)


def resolve_profile_input_context_len(
    *,
    target_context_len: int,
    step: int,
    warmup_iters: int,
    measure_iters: int,
) -> int:
    """Center full-accept measurements around ``target_context_len``."""

    target_context_len = int(target_context_len)
    step = int(step)
    warmup_iters = int(warmup_iters)
    measure_iters = int(measure_iters)
    if target_context_len <= 0 or step < 0 or warmup_iters < 0 or measure_iters <= 0:
        raise ValueError("Invalid context, step, warmup, or measurement count.")
    midpoint_offset = 1.0 + (step + 1) * (warmup_iters + (measure_iters + 1) / 2.0)
    input_context_len = int(math.floor(target_context_len - midpoint_offset + 0.5))
    if input_context_len <= 0:
        raise ValueError(
            "Profile context is too short to center its measurement window: "
            f"target_context_len={target_context_len}, step={step}, "
            f"warmup_iters={warmup_iters}, measure_iters={measure_iters}."
        )
    return input_context_len


def build_decoupled_verify_profile_fingerprint(
    *, server_args: Any, model_config: Any | None = None
) -> dict[str, Any]:
    """Build the runtime fields that make scheduler-cycle costs comparable."""

    device_name = None
    device_capability = None
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        device_capability = list(torch.cuda.get_device_capability(device_index))
    dtype = getattr(model_config, "dtype", None)
    return {
        "profile_abi_version": DECOUPLED_VERIFY_PROFILE_ABI_VERSION,
        "model_path": str(Path(server_args.model_path).expanduser()),
        "model_revision": getattr(server_args, "revision", None),
        "dtype": str(
            dtype if dtype is not None else getattr(server_args, "dtype", None)
        ),
        "kv_cache_dtype": str(getattr(server_args, "kv_cache_dtype", None)),
        "tp_size": int(server_args.tp_size),
        "pp_size": int(getattr(server_args, "pp_size", 1)),
        "nnodes": int(getattr(server_args, "nnodes", 1)),
        "attention_backend": str(getattr(server_args, "attention_backend", None)),
        "cuda_graph_backend_decode": str(
            getattr(server_args, "cuda_graph_backend_decode", None)
        ),
        "cuda_graph_bs_decode": [
            int(value)
            for value in (getattr(server_args, "cuda_graph_bs_decode", None) or [])
        ],
        "page_size": int(getattr(server_args, "page_size", 1)),
        "disable_overlap_schedule": bool(server_args.disable_overlap_schedule),
        "enable_linear_replayssm_spec": bool(
            getattr(server_args, "enable_linear_replayssm_spec", False)
        ),
        "mamba_radix_cache_strategy": str(
            getattr(server_args, "mamba_radix_cache_strategy", None)
        ),
        "max_verify_steps": int(server_args.speculative_num_steps),
        "device_name": device_name,
        "device_capability": device_capability,
    }


def validate_profile_fingerprint(
    profile_fingerprint: dict[str, Any], runtime_fingerprint: dict[str, Any]
) -> None:
    if not isinstance(profile_fingerprint, dict):
        raise ValueError("Decoupled verify profile fingerprint must be a mapping.")
    mismatches = {
        key: {"profile": profile_fingerprint.get(key), "runtime": runtime_value}
        for key, runtime_value in runtime_fingerprint.items()
        if profile_fingerprint.get(key) != runtime_value
    }
    if mismatches:
        raise ValueError(
            "Decoupled verify throughput profile is incompatible with this runtime: "
            f"{mismatches}"
        )


def validate_decoupled_verify_profile(
    profile: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    if profile.get("schema_version") != DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported decoupled verify profile schema_version: "
            f"{profile.get('schema_version')!r}."
        )
    if profile.get("kind") != DECOUPLED_VERIFY_PROFILE_KIND:
        raise ValueError(
            f"Unexpected decoupled verify profile kind: {profile.get('kind')!r}."
        )
    if profile.get("measurement_clock") != DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK:
        raise ValueError(
            "Decoupled verify profile must use scheduler verify-commit gaps."
        )
    if profile.get("cost_scope") != DECOUPLED_VERIFY_PROFILE_COST_SCOPE:
        raise ValueError("Decoupled verify profile has the wrong cost scope.")
    if profile.get("cost_estimator") != DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR:
        raise ValueError("Decoupled verify profile has the wrong cost estimator.")
    if profile.get("draft_provider") != DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER:
        raise ValueError(
            "Decoupled verify profile must use the forward-stream mock GPU-tail "
            "selector."
        )
    if profile.get("profile_control_plane") != DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE:
        raise ValueError(
            "Decoupled verify profile must leave synthetic tail production out of "
            "the verifier daemon."
        )
    if (
        profile.get("context_anchor_mode")
        != DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE
    ):
        raise ValueError(
            "Decoupled verify profile must use midpoint context anchoring."
        )
    if (
        profile.get("trajectory_acceptance")
        != DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE
    ):
        raise ValueError("Decoupled verify profile must use full-accept trajectories.")
    if require_complete and profile.get("status") != "complete":
        raise ValueError(
            "Production decoupled verification requires a complete profile, got "
            f"status={profile.get('status')!r}."
        )
    points = profile.get("points")
    if not isinstance(points, list):
        raise ValueError("Decoupled verify profile points must be a list.")
    seen: set[tuple[int, int, int]] = set()
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Each decoupled verify profile point must be a mapping.")
        key = (
            int(point.get("step", -1)),
            int(point.get("batch_size", 0)),
            int(point.get("context_len", 0)),
        )
        cost_ms = float(point.get("cost_ms", 0))
        if (
            key[0] < 0
            or key[1] <= 0
            or key[2] <= 0
            or cost_ms <= 0
            or not math.isfinite(cost_ms)
        ):
            raise ValueError(f"Invalid decoupled verify profile point: {point!r}.")
        if key in seen:
            raise ValueError(f"Duplicate decoupled verify profile point: {key}.")
        if float(point.get("cuda_graph_fraction", 0)) != 1.0:
            raise ValueError(
                "Decoupled verify profile points must contain only CUDA Graph "
                f"rounds: {point!r}."
            )
        if not math.isclose(
            float(point.get("mean_selected_draft_length", -1)),
            float(key[0]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Decoupled verify profile point did not verify the requested K: "
                f"{point!r}."
            )
        if not math.isclose(
            float(point.get("mean_accept_length", -1)),
            float(key[0] + 1),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Decoupled verify profile point is not a full-accept trajectory: "
                f"{point!r}."
            )
        seen.add(key)
    return profile


def load_decoupled_verify_profile(
    path: str | os.PathLike[str], *, require_complete: bool = True
) -> dict[str, Any]:
    return validate_decoupled_verify_profile(
        read_json(path), require_complete=require_complete
    )


def load_decoupled_verify_profile_job(path: str | os.PathLike[str]) -> dict[str, Any]:
    job = read_json(path)
    if job.get("schema_version") != DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported offline profile job schema: {job!r}.")
    if job.get("kind") != DECOUPLED_VERIFY_PROFILE_JOB_KIND:
        raise ValueError(f"Unexpected offline profile job kind: {job.get('kind')!r}.")
    if not isinstance(job.get("points"), list) or not job["points"]:
        raise ValueError("Offline profile job must contain at least one point.")
    if not job.get("output_path") or not job.get("request_prefix"):
        raise ValueError("Offline profile job requires output_path and request_prefix.")
    return job


__all__ = [
    "DECOUPLED_VERIFY_PROFILE_ABI_VERSION",
    "DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR",
    "DECOUPLED_VERIFY_PROFILE_COST_SCOPE",
    "DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE",
    "DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE",
    "DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER",
    "DECOUPLED_VERIFY_PROFILE_JOB_KIND",
    "DECOUPLED_VERIFY_PROFILE_KIND",
    "DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK",
    "DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION",
    "DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE",
    "atomic_write_json",
    "build_decoupled_verify_profile_fingerprint",
    "canonical_json_bytes",
    "estimate_scheduler_cycle_cost",
    "file_sha256",
    "json_sha256",
    "load_decoupled_verify_profile",
    "load_decoupled_verify_profile_job",
    "read_json",
    "resolve_profile_input_context_len",
    "validate_decoupled_verify_profile",
    "validate_profile_fingerprint",
]

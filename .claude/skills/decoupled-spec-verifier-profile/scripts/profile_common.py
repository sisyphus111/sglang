#!/usr/bin/env python3
"""Configuration, engine arguments, and cache validation for verifier profiling."""

from __future__ import annotations

import json
import math
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileSpec:
    config_path: Path
    repo: Path
    target_model: str
    output_dir: Path
    gpus: tuple[str, ...]
    tp_size: int
    dp_size: int
    enable_dp_attention: bool
    dtype: str
    trust_remote_code: bool
    mem_fraction_static: float | None
    ep_size: int | None
    moe_a2a_backend: str | None
    attention_backend: str | None
    batch_sizes: tuple[int, ...]
    ctx_lens: tuple[int, ...]
    steps: tuple[int, ...]
    timeout_s: float

    @property
    def engine_tp_size(self) -> int:
        return self.tp_size * self.dp_size if self.enable_dp_attention else self.tp_size

    @property
    def profile_path(self) -> Path:
        return self.output_dir / "profile.json"

    @property
    def required_points(self) -> set[tuple[int, int, int]]:
        return {
            (bs, step, ctx)
            for step in self.steps
            for bs in self.batch_sizes
            for ctx in self.ctx_lens
        }


def _positive_unique(raw: Any, name: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    if any(type(value) is not int for value in raw):
        raise ValueError(f"{name} values must be integers")
    values = tuple(raw)
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be unique")
    return tuple(sorted(values))


def load_spec(path: Path) -> ProfileSpec:
    path = path.expanduser().resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)
    for section in ("paths", "target", "profile", "run"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config missing [{section}] section")
    paths, target, profile, run = (
        config["paths"],
        config["target"],
        config["profile"],
        config["run"],
    )
    base = path.parent

    for key in ("repo", "target_model", "output_dir"):
        if not Path(str(paths[key])).expanduser().is_absolute():
            raise ValueError(f"paths.{key} must be absolute")

    def resolve(raw: Any) -> Path:
        value = Path(str(raw)).expanduser()
        return (base / value).resolve() if not value.is_absolute() else value.resolve()

    batch_sizes = _positive_unique(profile.get("batch_sizes"), "batch_sizes")
    ctx_lens = _positive_unique(profile.get("ctx_lens"), "ctx_lens")
    raw_steps = profile.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a non-empty list")
    if any(type(value) is not int for value in raw_steps):
        raise ValueError("steps values must be integers")
    steps = tuple(raw_steps)
    if steps != tuple(range(max(steps) + 1)):
        raise ValueError("steps must be the ordered contiguous set 0..max_step")

    gpus = target.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ValueError("target.gpus must be a non-empty list")
    gpu_ids = tuple(str(value) for value in gpus)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("target.gpus must be unique")
    if (
        type(target.get("tp_size")) is not int
        or type(target.get("dp_size", 1)) is not int
    ):
        raise ValueError("target tp_size and dp_size must be integers")
    tp_size = target["tp_size"]
    dp_size = target.get("dp_size", 1)
    if type(target.get("enable_dp_attention", False)) is not bool:
        raise ValueError("target.enable_dp_attention must be a boolean")
    enable_dp_attention = target.get("enable_dp_attention", False)
    if tp_size <= 0 or dp_size <= 0:
        raise ValueError("target tp_size and dp_size must be positive")
    if not enable_dp_attention and dp_size != 1:
        raise ValueError("dp_size must be 1 when DP attention is disabled")
    engine_tp_size = tp_size * dp_size if enable_dp_attention else tp_size
    if len(gpu_ids) != engine_tp_size:
        raise ValueError(
            f"target.gpus has {len(gpu_ids)} entries, expected {engine_tp_size}"
        )
    timeout_value = run.get("timeout_s", 7200)
    if isinstance(timeout_value, bool) or not isinstance(timeout_value, (int, float)):
        raise ValueError("run.timeout_s must be a number")
    timeout_s = float(timeout_value)
    if timeout_s <= 0:
        raise ValueError("run.timeout_s must be positive")
    mem_fraction = target.get("mem_fraction_static")
    if mem_fraction is not None and not 0 < float(mem_fraction) < 1:
        raise ValueError("target.mem_fraction_static must be in (0, 1)")

    if type(target.get("trust_remote_code", False)) is not bool:
        raise ValueError("target.trust_remote_code must be a boolean")
    if "deterministic" in target:
        raise ValueError(
            "target.deterministic is no longer supported by this profiler"
        )

    return ProfileSpec(
        config_path=path,
        repo=resolve(paths["repo"]),
        target_model=str(resolve(paths["target_model"])),
        output_dir=resolve(paths["output_dir"]),
        gpus=gpu_ids,
        tp_size=tp_size,
        dp_size=dp_size,
        enable_dp_attention=enable_dp_attention,
        dtype=str(target.get("dtype", "auto")),
        trust_remote_code=bool(target.get("trust_remote_code", False)),
        mem_fraction_static=(float(mem_fraction) if mem_fraction is not None else None),
        ep_size=(int(target["ep_size"]) if "ep_size" in target else None),
        moe_a2a_backend=(
            str(target["moe_a2a_backend"]) if "moe_a2a_backend" in target else None
        ),
        attention_backend=(
            str(target["attention_backend"]) if "attention_backend" in target else None
        ),
        batch_sizes=batch_sizes,
        ctx_lens=ctx_lens,
        steps=steps,
        timeout_s=timeout_s,
    )


def build_engine_kwargs(spec: ProfileSpec, dist_init_addr: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_path": spec.target_model,
        "tp_size": spec.engine_tp_size,
        "dp_size": spec.dp_size,
        "dist_init_addr": dist_init_addr,
        "speculative_algorithm": "DECOUPLED_VERIFY",
        "speculative_num_steps": max(spec.steps),
        "speculative_num_draft_tokens": max(spec.steps) + 1,
        "speculative_adaptive": True,
        "speculative_adaptive_strategy": "throughput_aware",
        "decoupled_verify_throughput_profile_path": str(spec.profile_path),
        "decoupled_verify_throughput_profile_ctx_lens": ",".join(
            str(value) for value in spec.ctx_lens
        ),
        "cuda_graph_bs_decode": list(spec.batch_sizes),
        "max_running_requests": max(spec.batch_sizes),
        "disable_radix_cache": True,
        "decoupled_spec_rank_base": 0,
        "dtype": spec.dtype,
        "trust_remote_code": spec.trust_remote_code,
        "log_level": "info",
    }
    if spec.enable_dp_attention:
        kwargs["enable_dp_attention"] = True
    if spec.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = spec.mem_fraction_static
    if spec.ep_size is not None:
        kwargs["ep_size"] = spec.ep_size
    if spec.moe_a2a_backend is not None:
        kwargs["moe_a2a_backend"] = spec.moe_a2a_backend
    if spec.attention_backend is not None:
        kwargs["attention_backend"] = spec.attention_backend
    return kwargs


def validate_profile(
    spec: ProfileSpec, *, expected_gpu_name: str | None = None
) -> dict[str, Any]:
    payload = json.loads(spec.profile_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("profile payload must be an object")
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("profile fingerprint must be an object")
    expected_fingerprint = {
        "target_model_path": spec.target_model,
        "target_tp_size": spec.tp_size,
        "target_dp_size": spec.dp_size,
        "enable_dp_attention": spec.enable_dp_attention,
    }
    if expected_gpu_name is not None:
        expected_fingerprint["gpu_name"] = expected_gpu_name
    mismatches = {
        key: {"expected": expected, "actual": fingerprint.get(key)}
        for key, expected in expected_fingerprint.items()
        if fingerprint.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"profile fingerprint mismatch: {mismatches}")
    rows = payload.get("costs")
    if not isinstance(rows, list):
        raise ValueError("profile costs must be a list")
    seen: set[tuple[int, int, int]] = set()
    duplicates: list[tuple[int, int, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"profile cost row {index} must be an object")
        point = (int(row["batch_size"]), int(row["steps"]), int(row["ctx_len"]))
        cost_ms = float(row["cost_ms"])
        if not math.isfinite(cost_ms) or cost_ms <= 0:
            raise ValueError(f"profile cost must be finite and positive: {row}")
        if point in seen:
            duplicates.append(point)
        seen.add(point)
    missing = sorted(spec.required_points - seen)
    extra = sorted(seen - spec.required_points)
    if duplicates or missing or extra:
        raise ValueError(
            "profile grid mismatch: "
            f"duplicates={duplicates[:3]}, missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        "requested_points": len(spec.required_points),
        "actual_points": len(rows),
        "missing_points": [],
        "extra_points": [],
        "duplicate_points": 0,
        "fingerprint": fingerprint,
    }


def detect_gpu_name(gpus: tuple[str, ...]) -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            ",".join(gpus),
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    names = {line.strip() for line in output.splitlines() if line.strip()}
    if len(names) != 1:
        raise ValueError(
            f"target GPUs must have one homogeneous model name: {sorted(names)}"
        )
    return names.pop()

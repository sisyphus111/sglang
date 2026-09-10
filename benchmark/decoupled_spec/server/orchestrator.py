"""Ray orchestration for native SGLang Engine actors in a decoupled fleet."""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import msgspec
import yaml

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

from .placement import (
    get_alive_gpu_nodes,
    plan_decoupled_spec_placement,
    plan_spec_engine_placement,
)

SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 2
MANIFEST_RELATIVE_PATH = Path("server") / "manifest.json"
_DYNAMIC_SERVER_ARGS = {
    "host",
    "port",
    "nccl_port",
    "dist_init_addr",
    "nnodes",
    "node_rank",
    "base_gpu_id",
    "gpu_id_step",
    "use_ray",
    "decoupled_spec_role",
    "decoupled_spec_rank",
    "decoupled_spec_bind_endpoint",
    "decoupled_spec_connect_endpoints",
    "decoupled_spec_peer_configs",
}
_CPP_DATA_PLANE_ENV = "SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND"
_VERIFY_PROFILE_ENV = "SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH"


class RaySettings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Ray connection and lifecycle timeouts owned by the unified launcher."""

    address: str = "auto"
    namespace: str = "decoupled-spec-benchmark"
    placement_timeout_s: float = 120.0
    startup_timeout_s: float = 900.0
    shutdown_grace_s: float = 30.0


class RuntimeSettings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Environment inherited by every HTTP process for one role."""

    env: dict[str, Any] = msgspec.field(default_factory=dict)


class RoleSettings(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One role's replica count and SGLang ServerArgs template."""

    server_args: dict[str, Any]
    replicas: int = 1
    runtime: RuntimeSettings = msgspec.field(default_factory=RuntimeSettings)


class UnifiedServerConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Ray-managed coupled or decoupled speculative-decoding fleet."""

    target: RoleSettings | None = None
    verifier: RoleSettings | None = None
    drafter: RoleSettings | None = None
    ray: RaySettings = msgspec.field(default_factory=RaySettings)
    schema_version: int = SCHEMA_VERSION
    deployment: str | None = None

    @property
    def deployment_kind(self) -> str:
        if self.schema_version == SCHEMA_VERSION:
            return "decoupled_spec"
        assert self.deployment is not None
        return self.deployment

    @property
    def primary_role(self) -> str:
        return "target" if self.deployment_kind == "coupled_spec" else "verifier"

    @property
    def primary(self) -> RoleSettings:
        settings = self.target if self.primary_role == "target" else self.verifier
        assert settings is not None
        return settings


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _normalize_bool_env(value: Any, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid {name} value: {value!r}")


def _validate_role(role: str, settings: RoleSettings) -> dict[str, Any]:
    if settings.replicas <= 0:
        raise ValueError(f"{role}.replicas must be positive")
    if not isinstance(settings.server_args, dict):
        raise ValueError(f"{role}.server_args must be a mapping")
    dynamic = set(settings.server_args) & _DYNAMIC_SERVER_ARGS
    if dynamic:
        raise ValueError(
            f"{role}.server_args contains launcher-owned fields: {sorted(dynamic)}"
        )
    if "CUDA_VISIBLE_DEVICES" in settings.runtime.env:
        raise ValueError(
            f"{role}.runtime.env must not set CUDA_VISIBLE_DEVICES; Ray owns GPUs"
        )

    server_arg_fields = {field.name for field in dataclasses.fields(ServerArgs)}
    unknown_args = set(settings.server_args) - server_arg_fields
    if unknown_args:
        raise ValueError(f"unknown {role} ServerArgs fields: {sorted(unknown_args)}")
    args = settings.server_args
    if not args.get("model_path"):
        raise ValueError(f"{role}.server_args.model_path is required")
    if int(args.get("tp_size", 1)) <= 0:
        raise ValueError(f"{role}.server_args.tp_size must be positive")
    if int(args.get("dp_size", 1)) != 1 or int(args.get("pp_size", 1)) != 1:
        raise ValueError(f"Ray-managed {role} requires dp_size=1 and pp_size=1")
    if role == "verifier":
        algorithm = args.get("speculative_algorithm")
        if not isinstance(algorithm, str) or algorithm.upper() != "DECOUPLED_VERIFY":
            raise ValueError(
                "verifier.server_args.speculative_algorithm must be " "DECOUPLED_VERIFY"
            )
    elif role == "drafter":
        if args.get("speculative_algorithm") is not None:
            raise ValueError("drafter.server_args.speculative_algorithm must be unset")
        if int(args.get("tp_size", 1)) != 1:
            raise ValueError("the current decoupled drafter requires tp_size=1")
        if int(args.get("page_size", 1)) != 1:
            raise ValueError("the current decoupled drafter requires page_size=1")
        if (
            not args.get("disable_overlap_schedule", False)
            and not args.get("disable_radix_cache", False)
        ):
            raise ValueError(
                "drafter overlap requires disable_radix_cache=true for "
                "GPU-owned recurrent-state checkpoints"
            )
    else:
        algorithm = args.get("speculative_algorithm")
        if not isinstance(algorithm, str) or not algorithm:
            raise ValueError(
                "target.server_args.speculative_algorithm must select an ordinary "
                "speculative decoding method"
            )
        if algorithm.upper() == "DECOUPLED_VERIFY":
            raise ValueError(
                "target.server_args.speculative_algorithm must not be DECOUPLED_VERIFY"
            )
    return args


def validate_config(config: UnifiedServerConfig) -> None:
    if config.schema_version not in {SCHEMA_VERSION, CONFIG_SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported schema_version={config.schema_version}; "
            f"expected {SCHEMA_VERSION} or {CONFIG_SCHEMA_VERSION}"
        )
    if config.schema_version == SCHEMA_VERSION:
        if config.deployment is not None or config.target is not None:
            raise ValueError(
                "schema_version 1 is the legacy decoupled verifier/drafter shape"
            )
        if config.verifier is None or config.drafter is None:
            raise ValueError("schema_version 1 requires verifier and drafter")
    elif config.deployment == "coupled_spec":
        if (
            config.target is None
            or config.verifier is not None
            or config.drafter is not None
        ):
            raise ValueError(
                "coupled_spec requires target and forbids verifier/drafter"
            )
    elif config.deployment == "decoupled_spec":
        if (
            config.target is not None
            or config.verifier is None
            or config.drafter is None
        ):
            raise ValueError(
                "decoupled_spec requires verifier/drafter and forbids target"
            )
    else:
        raise ValueError(
            "schema_version 2 deployment must be coupled_spec or decoupled_spec"
        )
    for name, value in (
        ("ray.placement_timeout_s", config.ray.placement_timeout_s),
        ("ray.startup_timeout_s", config.ray.startup_timeout_s),
        ("ray.shutdown_grace_s", config.ray.shutdown_grace_s),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if config.deployment_kind == "coupled_spec":
        _validate_role("target", config.primary)
        return

    assert config.verifier is not None and config.drafter is not None
    verifier_args = _validate_role("verifier", config.verifier)
    drafter_args = _validate_role("drafter", config.drafter)
    for field in (
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    ):
        verifier_value = verifier_args.get(field)
        drafter_value = drafter_args.get(field)
        if verifier_value != drafter_value:
            raise ValueError(
                f"role configs disagree on {field}: "
                f"verifier={verifier_value!r}, drafter={drafter_value!r}"
            )
    num_steps = verifier_args.get("speculative_num_steps")
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("speculative_num_steps must be a positive integer")
    if verifier_args.get("speculative_eagle_topk") != 1:
        raise ValueError("decoupled speculation currently requires topk=1")
    if verifier_args.get("speculative_num_draft_tokens") != num_steps + 1:
        raise ValueError(
            "speculative_num_draft_tokens must equal speculative_num_steps + 1"
        )

    _normalize_bool_env(
        config.verifier.runtime.env.get(_CPP_DATA_PLANE_ENV, "0"),
        _CPP_DATA_PLANE_ENV,
    )
    _normalize_bool_env(
        config.drafter.runtime.env.get(_CPP_DATA_PLANE_ENV, "0"),
        _CPP_DATA_PLANE_ENV,
    )


def load_config(path: str | Path) -> UnifiedServerConfig:
    raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"unified server config must be a YAML mapping: {path}")
    try:
        config = msgspec.convert(raw, type=UnifiedServerConfig, strict=True)
    except msgspec.ValidationError as exc:
        raise ValueError(f"invalid unified server config {path}: {exc}") from exc
    validate_config(config)
    return config


def config_to_dict(config: UnifiedServerConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in msgspec.to_builtins(config).items()
        if value is not None
    }


def validate_server_args_templates(config: UnifiedServerConfig) -> None:
    """Resolve each role's real ServerArgs with a synthetic local topology."""

    dynamic = {
        "target": {
            "host": "127.0.0.1",
            "port": 30000,
            "nccl_port": 29000,
            "nnodes": 1,
            "node_rank": 0,
            "dist_init_addr": "127.0.0.1:29000",
        },
        "verifier": {
            "host": "127.0.0.1",
            "port": 30000,
            "nccl_port": 29000,
            "decoupled_spec_role": "verifier",
            "decoupled_spec_rank": 0,
            "decoupled_spec_bind_endpoint": "tcp://127.0.0.1:31000",
            "decoupled_spec_peer_configs": [
                {
                    "rank": 0,
                    "endpoint": "tcp://127.0.0.1:31001",
                    "quota": 1,
                }
            ],
        },
        "drafter": {
            "host": "127.0.0.1",
            "port": 30001,
            "nccl_port": 29001,
            "decoupled_spec_role": "drafter",
            "decoupled_spec_rank": 0,
            "decoupled_spec_bind_endpoint": "tcp://127.0.0.1:31001",
            "decoupled_spec_peer_configs": [
                {
                    "rank": 0,
                    "endpoint": "tcp://127.0.0.1:31000",
                    "quota": 1,
                }
            ],
        },
    }
    roles = (
        ("target",)
        if config.deployment_kind == "coupled_spec"
        else ("verifier", "drafter")
    )
    for role in roles:
        settings = getattr(config, role)
        assert settings is not None
        original_env = {key: os.environ.get(key) for key in settings.runtime.env}
        try:
            os.environ.update(
                {str(key): str(value) for key, value in settings.runtime.env.items()}
            )
            server_args = dict(settings.server_args)
            server_args.update(dynamic[role])
            ServerArgs(**server_args)
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def plan_decoupled_spec_quota_edges(
    num_verifiers: int, num_drafters: int
) -> list[dict[str, int]]:
    """Build the v0.5.14-dev equal-share sparse quota graph.

    Each role partitions the same integer interval. Intersections are the
    smooth-weighted-round-robin quotas, so the graph stays sparse even when the
    verifier and drafter replica counts differ.
    """
    if num_verifiers <= 0 or num_drafters <= 0:
        raise ValueError("num_verifiers and num_drafters must be positive")

    edges: list[dict[str, int]] = []
    verifier_rank = drafter_rank = 0
    while verifier_rank < num_verifiers and drafter_rank < num_drafters:
        verifier_end = (verifier_rank + 1) * num_drafters
        drafter_end = (drafter_rank + 1) * num_verifiers
        overlap_start = max(verifier_rank * num_drafters, drafter_rank * num_verifiers)
        overlap_end = min(verifier_end, drafter_end)
        quota = overlap_end - overlap_start
        if quota > 0:
            edges.append(
                {
                    "verifier_rank": verifier_rank,
                    "drafter_rank": drafter_rank,
                    "quota": quota,
                }
            )
        if verifier_end <= drafter_end:
            verifier_rank += 1
        if drafter_end <= verifier_end:
            drafter_rank += 1
    return edges


def _is_ipv6(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).version == 6
    except ValueError:
        return False


def _host_port(host: str, port: int) -> str:
    return f"[{host}]:{port}" if _is_ipv6(host) else f"{host}:{port}"


def _numbered_env_ports() -> list[int]:
    """Return local PORT{n} values in numeric-suffix order."""

    raw_values = [
        value
        for _, value in sorted(
            (int(name[4:]), value)
            for name, value in os.environ.items()
            if name.startswith("PORT") and name[4:].isdigit() and int(name[4:]) > 0
        )
    ]
    ports = []
    for raw_port in " ".join(raw_values).replace(",", " ").split():
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(f"invalid reserved port: {raw_port!r}") from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"reserved port out of range: {port}")
        ports.append(port)
    if len(set(ports)) != len(ports):
        raise ValueError(f"reserved ports must be unique: {ports}")
    return ports


def _reserve_tcp_socket(host: str) -> socket.socket:
    family = socket.AF_INET6 if _is_ipv6(host) else socket.AF_INET

    def bind(port: int) -> socket.socket:
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                try:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                except OSError:
                    pass
            if family == socket.AF_INET6:
                sock.bind((host, port, 0, 0))
            else:
                sock.bind((host, port))
            sock.listen(1)
            return sock
        except BaseException:
            sock.close()
            raise

    for port in _numbered_env_ports():
        try:
            return bind(port)
        except OSError:
            continue
    return bind(0)


def build_peer_configs(
    engine_infos: list[dict[str, Any]],
    quota_edges: list[dict[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    """Resolve sparse quota edges to rank-addressed runtime peer configs."""
    by_key = {(info["role"], int(info["rank"])): info for info in engine_infos}
    peer_configs: dict[str, list[dict[str, Any]]] = {
        str(info["engine_id"]): [] for info in engine_infos
    }
    for edge in quota_edges:
        verifier_rank = int(edge["verifier_rank"])
        drafter_rank = int(edge["drafter_rank"])
        quota = int(edge["quota"])
        verifier = by_key[("verifier", verifier_rank)]
        drafter = by_key[("drafter", drafter_rank)]
        peer_configs[str(verifier["engine_id"])].append(
            {
                "rank": drafter_rank,
                "endpoint": str(drafter["transport_endpoint"]),
                "quota": quota,
            }
        )
        peer_configs[str(drafter["engine_id"])].append(
            {
                "rank": verifier_rank,
                "endpoint": str(verifier["transport_endpoint"]),
                "quota": quota,
            }
        )
    for peers in peer_configs.values():
        peers.sort(key=lambda item: int(item["rank"]))
    if any(not peers for peers in peer_configs.values()):
        raise RuntimeError("quota graph left an engine without a peer")
    return peer_configs


class EngineActor:
    """Own one node shard of a native multi-node SGLang Engine replica."""

    def __init__(
        self,
        *,
        engine_id: str,
        role: str,
        rank: int,
        tp_size: int,
        node_rank: int,
        nnodes: int,
        expected_num_gpus: int,
        runtime_env: dict[str, Any],
        staged_files: dict[str, dict[str, Any]] | None = None,
        node_info_override: dict[str, str] | None = None,
        gpu_ids_override: list[str] | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.role = role
        self.rank = rank
        self.tp_size = tp_size
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.expected_num_gpus = expected_num_gpus
        # Actor-local files must not assume a shared filesystem across Ray nodes.
        self.local_dir = Path(tempfile.mkdtemp(prefix=f"sglang-{engine_id}-"))
        self.runtime_env = {str(key): str(value) for key, value in runtime_env.items()}
        self._staged_server_arg_paths: dict[str, str] = {}
        self.staged_artifacts: dict[str, dict[str, Any]] = {}
        for name, artifact in (staged_files or {}).items():
            local_path = self.local_dir / "artifacts" / str(artifact["filename"])
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(bytes(artifact["bytes"]))
            if artifact.get("env"):
                self.runtime_env[str(artifact["env"])] = str(local_path)
            if artifact.get("server_arg"):
                self._staged_server_arg_paths[str(artifact["server_arg"])] = str(
                    local_path
                )
            self.staged_artifacts[str(name)] = {
                "source_path": str(artifact["source_path"]),
                "sha256": str(artifact["sha256"]),
                "resolved_path": str(local_path),
            }
        self.node_info_override = node_info_override
        self.gpu_ids_override = gpu_ids_override
        self.node_id: str | None = None
        self.node_ip: str | None = None
        self.gpu_ids: list[str] = []
        self._leases: dict[str, socket.socket] = {}
        self._requires_transport = True
        self._process: subprocess.Popen[bytes] | None = None
        self._engine: Any = None
        self._log_stream: Any = None
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._stopped = False
        self._resolved_config: dict[str, Any] | None = None
        self._stop_result: dict[str, Any] | None = None

    def _runtime_identity(self) -> tuple[str, str, list[str]]:
        if self.node_info_override is not None:
            return (
                str(self.node_info_override["node_id"]),
                NetworkAddress(str(self.node_info_override["node_ip"]), 0).host,
                list(self.gpu_ids_override or []),
            )
        import ray

        context = ray.get_runtime_context()
        gpu_ids = sorted(
            (str(value) for value in context.get_accelerator_ids().get("GPU", [])),
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
        node_ip = NetworkAddress(str(ray.util.get_node_ip_address()), 0).host
        return (
            str(context.get_node_id()),
            node_ip,
            gpu_ids,
        )

    def allocation(self) -> dict[str, Any]:
        self.node_id, self.node_ip, self.gpu_ids = self._runtime_identity()
        if len(self.gpu_ids) != self.expected_num_gpus:
            raise RuntimeError(
                f"{self.engine_id} node_rank={self.node_rank} expected "
                f"{self.expected_num_gpus} Ray GPUs, got {self.gpu_ids}"
            )
        if self.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_ids)
        return {
            "engine_id": self.engine_id,
            "role": self.role,
            "rank": self.rank,
            "node_rank": self.node_rank,
            "nnodes": self.nnodes,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "gpu_ids": self.gpu_ids,
            "expected_num_gpus": self.expected_num_gpus,
            "staged_artifacts": self.staged_artifacts,
        }

    def reserve_ports(self, include_transport: bool = True) -> dict[str, Any]:
        import sglang

        if self.node_rank != 0:
            raise RuntimeError("only Engine node_rank 0 may reserve service ports")
        if self._leases:
            raise RuntimeError(f"{self.engine_id} already owns port leases")
        if self.node_id is None:
            self.allocation()
        self._requires_transport = include_transport
        port_names = (
            ("http", "transport", "nccl")
            if include_transport
            else ("http", "nccl")
        )
        try:
            self._leases = {
                name: _reserve_tcp_socket(self.node_ip)
                for name in port_names
            }
        except BaseException:
            self._release_port_leases()
            raise
        ports = {
            name: int(sock.getsockname()[1]) for name, sock in self._leases.items()
        }
        info = {
            "engine_id": self.engine_id,
            "role": self.role,
            "rank": self.rank,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "http_url": f"http://{_host_port(self.node_ip, ports['http'])}",
            "nccl_port": ports["nccl"],
            "http_port": ports["http"],
            "tp_size": self.tp_size,
            "gpu_ids": self.gpu_ids,
            "nnodes": self.nnodes,
            "sglang_path": str(Path(sglang.__file__).resolve()),
        }
        if include_transport:
            info["transport_endpoint"] = (
                f"tcp://{_host_port(self.node_ip, ports['transport'])}"
            )
        return info

    def _release_port_leases(self) -> None:
        for sock in self._leases.values():
            sock.close()
        self._leases.clear()

    def start(self, resolved_config: dict[str, Any]) -> dict[str, Any]:
        if self._process is not None or self._engine is not None:
            raise RuntimeError(f"{self.engine_id} was already started")
        required_leases = (
            {"http", "transport", "nccl"}
            if self._requires_transport
            else {"http", "nccl"}
        )
        if self.node_rank == 0 and set(self._leases) != required_leases:
            raise RuntimeError(f"{self.engine_id} does not own all required ports")
        if self.node_rank > 0 and self._leases:
            raise RuntimeError(
                f"{self.engine_id} nonzero node rank unexpectedly owns service ports"
            )
        if self.node_id is None:
            self.allocation()

        engine_dir = self.local_dir / "engine"
        log_path = self.local_dir / f"{self.engine_id}-node{self.node_rank}.log"
        config_path = engine_dir / "resolved_config.json"
        status_path = engine_dir / "status.json"
        if status_path.exists():
            raise RuntimeError(
                f"refusing to reuse existing engine state: {status_path}"
            )
        resolved_config = json.loads(json.dumps(resolved_config))
        resolved_config.setdefault("runtime", {}).setdefault("env", {}).update(
            self.runtime_env
        )
        for key, local_path in self._staged_server_arg_paths.items():
            resolved_config["server_args"][key] = local_path
        if self.staged_artifacts:
            resolved_config["staged_artifacts"] = self.staged_artifacts
        self._resolved_config = resolved_config
        _write_json(config_path, resolved_config)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = log_path.open("ab", buffering=0)
        if self.node_rank > 0:
            os.environ.update(self.runtime_env)
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            _write_json(
                status_path,
                {
                    "engine_id": self.engine_id,
                    "role": self.role,
                    "rank": self.rank,
                    "node_rank": self.node_rank,
                    "state": "starting",
                    "pid": os.getpid(),
                    "updated_at": time.time(),
                },
            )
            import sglang as sgl

            try:
                self._engine = sgl.Engine(**resolved_config["server_args"])
            except BaseException as exc:
                _write_json(
                    status_path,
                    {
                        "engine_id": self.engine_id,
                        "role": self.role,
                        "rank": self.rank,
                        "node_rank": self.node_rank,
                        "state": "failed",
                        "pid": os.getpid(),
                        "updated_at": time.time(),
                        "error": repr(exc),
                    },
                )
                self._restore_actor_output()
                raise
            _write_json(
                status_path,
                {
                    "engine_id": self.engine_id,
                    "role": self.role,
                    "rank": self.rank,
                    "node_rank": self.node_rank,
                    "state": "engine_ready",
                    "pid": os.getpid(),
                    "updated_at": time.time(),
                },
            )
            return {"pid": os.getpid(), "state": "engine_ready"}

        command = [
            sys.executable,
            str(Path(__file__).with_name("role_server.py")),
            "--resolved-config",
            str(config_path),
            "--status-file",
            str(status_path),
        ]
        env = os.environ.copy()
        env.update(self.runtime_env)
        import sglang

        current_python_root = str(Path(sglang.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                current_python_root,
                *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []),
            ]
        )
        env["PYTHONUNBUFFERED"] = "1"

        # Transport and NCCL bind during engine initialization. The HTTP socket
        # is inherited by role_server and handed directly to uvicorn, so its
        # port stays reserved throughout slow weight loading and CUDA capture.
        http_socket = self._leases.pop("http")
        http_socket.listen(2048)
        http_socket_fd = http_socket.fileno()
        command.extend(["--http-socket-fd", str(http_socket_fd)])
        self._release_port_leases()
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self.local_dir,
                env=env,
                stdout=None,
                stderr=None,
                start_new_session=True,
                pass_fds=(http_socket_fd,),
            )
        except BaseException:
            self._close_log()
            raise
        finally:
            http_socket.close()
        return {
            "pid": self._process.pid,
            "state": "starting",
        }

    def _tail_log(self, limit: int = 80) -> str:
        path = self.local_dir / f"{self.engine_id}-node{self.node_rank}.log"
        try:
            return "\n".join(path.read_text(encoding="utf-8").splitlines()[-limit:])
        except (OSError, UnicodeDecodeError):
            return "<log unavailable>"

    def readiness(self) -> dict[str, Any] | None:
        """Return ready metadata after one bounded probe, or None while starting."""
        if self.node_rank == 0 and self._process is None:
            raise RuntimeError(f"{self.engine_id} has not started")
        if self.node_rank > 0 and self._engine is None:
            raise RuntimeError(f"{self.engine_id} node_rank={self.node_rank} not ready")
        status_path = self.local_dir / "engine" / "status.json"
        if self.node_rank > 0:
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                return None
            if status.get("state") == "failed":
                raise RuntimeError(
                    f"{self.engine_id} node_rank={self.node_rank} failed: "
                    f"{status.get('error')}\n{self._tail_log()}"
                )
            return status if status.get("state") == "engine_ready" else None

        endpoint_path = (
            "/health" if self.role in {"target", "verifier"} else "/model_info"
        )
        return_code = self._process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"{self.engine_id} exited with code {return_code} before ready\n"
                f"{self._tail_log()}"
            )
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if status.get("state") == "failed":
            raise RuntimeError(
                f"{self.engine_id} failed during startup: {status.get('error')}\n"
                f"{self._tail_log()}"
            )
        if status.get("state") != "http_ready":
            return None
        try:
            with urllib.request.urlopen(
                str(status["http_url"]).rstrip("/") + endpoint_path,
                timeout=1.0,
            ) as response:
                return status if response.status == 200 else None
        except BaseException:
            return None

    def poll(self) -> int | None:
        if self.node_rank > 0:
            return None if self._engine is not None else 0
        return None if self._process is None else self._process.poll()

    def _restore_actor_output(self) -> None:
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)
            self._saved_stdout_fd = None
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
            self._saved_stderr_fd = None

    def _close_log(self) -> None:
        self._restore_actor_output()
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def stop(self, grace_s: float) -> dict[str, Any]:
        if self._stopped:
            return dict(self._stop_result or {"engine_id": self.engine_id})
        self._stopped = True
        self._release_port_leases()
        escalated = False
        process_group_alive = False
        shutdown_error = None
        if self._engine is not None:
            engine, self._engine = self._engine, None
            try:
                engine.shutdown()
            except BaseException as exc:
                shutdown_error = repr(exc)
        if self._process is not None:
            process_group_id = self._process.pid
            if self._process_group_exists(process_group_id):
                try:
                    # The SGLang HTTP leader owns graceful request draining and
                    # child cleanup. Signaling the whole group here would kill
                    # schedulers first and turn a normal stop into crash cleanup.
                    self._process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + grace_s
                while self._process_group_exists(process_group_id):
                    # Reap the leader as soon as it exits. A zombie leader still
                    # belongs to the process group and would otherwise look like
                    # a live grandchild until the grace deadline expires.
                    self._process.poll()
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                if self._process_group_exists(process_group_id):
                    escalated = True
                    try:
                        os.killpg(process_group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process_group_alive = True
            else:
                kill_deadline = time.monotonic() + 5.0
                while (
                    self._process_group_exists(process_group_id)
                    and time.monotonic() < kill_deadline
                ):
                    time.sleep(0.05)
                process_group_alive = self._process_group_exists(process_group_id)
        if self.node_rank > 0:
            return_code = 0
        else:
            return_code = None if self._process is None else self._process.poll()
        self._close_log()
        log_path = self.local_dir / f"{self.engine_id}-node{self.node_rank}.log"
        status_path = self.local_dir / "engine" / "status.json"
        try:
            log_bytes = log_path.read_bytes()
        except OSError:
            log_bytes = b""
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {
                "engine_id": self.engine_id,
                "role": self.role,
                "rank": self.rank,
                "node_rank": self.node_rank,
            }
        status.update(
            {
                "node_rank": self.node_rank,
                "node_id": self.node_id,
                "node_ip": self.node_ip,
                "gpu_ids": self.gpu_ids,
            }
        )
        if shutdown_error is not None:
            status.update(
                {
                    "state": "failed",
                    "error": f"Engine shutdown failed: {shutdown_error}",
                    "return_code": return_code,
                    "updated_at": time.time(),
                }
            )
        elif process_group_alive:
            status.update(
                {
                    "state": "failed",
                    "error": "owned process group remained alive after SIGKILL",
                    "return_code": return_code,
                    "updated_at": time.time(),
                }
            )
        elif status.get("state") != "failed":
            status.update(
                {
                    "state": "exited",
                    "return_code": return_code,
                    "updated_at": time.time(),
                }
            )
        self._stop_result = {
            "engine_id": self.engine_id,
            "node_rank": self.node_rank,
            "return_code": return_code,
            "sigkill_escalated": escalated,
            "process_group_alive": process_group_alive,
            "artifacts": {
                "resolved_config": self._resolved_config,
                "status": status,
                "log_bytes": log_bytes,
            },
        }
        shutil.rmtree(self.local_dir, ignore_errors=True)
        return dict(self._stop_result)


class RayServerOrchestrator:
    """Jointly place node-level Engine actors and manage the fleet as one unit."""

    def __init__(
        self,
        config: UnifiedServerConfig,
        *,
        config_path: str | Path,
        run_dir: str | Path,
        ray_address: str | None = None,
        ray_namespace: str | None = None,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.ray_address = ray_address or config.ray.address
        self.ray_namespace = ray_namespace or config.ray.namespace
        self.manifest_path = self.run_dir / MANIFEST_RELATIVE_PATH
        self.ray: Any = None
        self.primary_pgs: list[Any] = []
        self.verifier_pgs = self.primary_pgs
        self.actors: list[Any] = []
        self.actor_records: list[dict[str, Any]] = []
        self.leader_actors: list[Any] = []
        self.engine_infos: list[dict[str, Any]] = []
        self.placement_plan: Any = None
        self._shutdown = False
        self.verifier_staged_files = self._load_verifier_staged_files()

    def _load_verifier_staged_files(self) -> dict[str, dict[str, Any]]:
        staged: dict[str, dict[str, Any]] = {}
        if self.config.deployment_kind != "decoupled_spec":
            return staged
        assert self.config.verifier is not None
        runtime_env = self.config.verifier.runtime.env
        profile_path_value = runtime_env.get(_VERIFY_PROFILE_ENV)
        if profile_path_value:
            profile_path = Path(str(profile_path_value)).expanduser().resolve()
            profile_bytes = profile_path.read_bytes()
            profile = json.loads(profile_bytes)
            from sglang.srt.speculative.decoupled_verify_profile import (
                validate_decoupled_verify_profile,
            )

            try:
                if not isinstance(profile, dict):
                    raise ValueError("Profile root must be a JSON mapping.")
                validate_decoupled_verify_profile(profile, require_complete=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Verifier throughput profile must be a complete decoupled "
                    "scheduler-cycle JSON produced by the forward-stream mock "
                    f"GPU-tail selector: {profile_path}."
                ) from exc
            staged["throughput_profile"] = {
                "bytes": profile_bytes,
                "filename": "decoupled-verify-throughput-profile.json",
                "source_path": str(profile_path),
                "sha256": hashlib.sha256(profile_bytes).hexdigest(),
                "env": _VERIFY_PROFILE_ENV,
            }
        adaptive_config_value = self.config.verifier.server_args.get(
            "speculative_adaptive_config"
        )
        if adaptive_config_value:
            adaptive_path = Path(str(adaptive_config_value)).expanduser().resolve()
            adaptive_bytes = adaptive_path.read_bytes()
            adaptive_value = json.loads(adaptive_bytes)
            if not isinstance(adaptive_value, dict):
                raise ValueError(
                    f"Verifier adaptive config must be a JSON mapping: {adaptive_path}."
                )
            staged["adaptive_config"] = {
                "bytes": adaptive_bytes,
                "filename": "decoupled-verify-adaptive-config.json",
                "source_path": str(adaptive_path),
                "sha256": hashlib.sha256(adaptive_bytes).hexdigest(),
                "server_arg": "speculative_adaptive_config",
            }
        return staged

    def _write_manifest(self, state: str, **extra: Any) -> dict[str, Any]:
        decoupled = self.config.deployment_kind == "decoupled_spec"
        num_drafters = self.config.drafter.replicas if self.config.drafter else 0
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "deployment": self.config.deployment_kind,
            "state": state,
            "updated_at": time.time(),
            "config_path": str(self.config_path),
            "ray_address": self.ray_address,
            "ray_namespace": self.ray_namespace,
            "topology": {
                "primary_role": self.config.primary_role,
                "num_primary_replicas": self.config.primary.replicas,
                "num_verifiers": (
                    self.config.verifier.replicas if self.config.verifier else 0
                ),
                "num_drafters": num_drafters,
                "placement": (
                    None
                    if self.placement_plan is None
                    else self.placement_plan.to_dict()
                ),
                "quota_edges": (
                    plan_decoupled_spec_quota_edges(
                        self.config.verifier.replicas, num_drafters
                    )
                    if decoupled and self.config.verifier is not None
                    else []
                ),
            },
            "engines": self.engine_infos,
            "verifier_staged_artifacts": {
                name: {
                    key: value
                    for key, value in artifact.items()
                    if key not in {"bytes", "env", "server_arg", "filename"}
                }
                for name, artifact in self.verifier_staged_files.items()
            },
            **extra,
        }
        _write_json(self.manifest_path, manifest)
        return manifest

    def _connect_ray(self) -> Any:
        import ray

        repository_root = Path(
            os.environ.get(
                "SGLANG_BENCHMARK_REPOSITORY_ROOT",
                Path(__file__).resolve().parents[3],
            )
        ).resolve()
        benchmark_package_root = Path(__file__).resolve().parents[1]
        ray.init(
            address=self.ray_address,
            namespace=self.ray_namespace,
            ignore_reinit_error=True,
            # Role processes inherit their EngineActor stdout/stderr. Keep Ray
            # forwarding enabled so the owning task shows model loading, CUDA
            # Graph capture, readiness, and runtime INFO logs by default.
            log_to_driver=True,
            runtime_env={
                "py_modules": [
                    # Upload the benchmark package root, not the ``server``
                    # directory itself. Ray preserves the py_module basename,
                    # so workers deserialize the actor through the canonical
                    # ``decoupled_spec.server.orchestrator`` package identity.
                    str(benchmark_package_root),
                    str(repository_root / "python" / "sglang"),
                ],
                "excludes": [
                    "**/__pycache__",
                    "**/*.pyc",
                    "analysis/**",
                    "client/**",
                    "configs/**",
                    "plot/**",
                    "results/**",
                    "skills/**",
                ],
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                },
            },
        )
        return ray

    def _get_refs(
        self,
        refs: list[Any],
        *,
        timeout_s: float,
        stop_event: Any = None,
    ) -> list[Any]:
        """Resolve Ray refs in input order while remaining SIGTERM-interruptible."""
        deadline = time.monotonic() + timeout_s
        pending = set(refs)
        values: dict[Any, Any] = {}
        while pending:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("server startup interrupted")
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(
                    f"timed out waiting for {len(pending)} Ray operations"
                )
            ready, _ = self.ray.wait(
                list(pending),
                num_returns=len(pending),
                timeout=min(0.5, remaining_s),
            )
            if ready:
                ready_values = self.ray.get(ready)
                values.update(zip(ready, ready_values, strict=True))
                pending.difference_update(ready)
        return [values[ref] for ref in refs]

    def _wait_http_ready(self, stop_event: Any = None) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.config.ray.startup_timeout_s
        ready_by_index: dict[int, dict[str, Any]] = {}
        while len(ready_by_index) != len(self.leader_actors):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("server startup interrupted")
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                missing = [
                    self.engine_infos[index]["engine_id"]
                    for index in range(len(self.leader_actors))
                    if index not in ready_by_index
                ]
                raise TimeoutError(
                    "servers were not HTTP-ready after "
                    f"{self.config.ray.startup_timeout_s:.1f}s: {missing}"
                )
            indices = [
                index
                for index in range(len(self.leader_actors))
                if index not in ready_by_index
            ]
            refs = [self.leader_actors[index].readiness.remote() for index in indices]
            results = self._get_refs(
                refs,
                timeout_s=min(5.0, remaining_s),
                stop_event=stop_event,
            )
            for index, result in zip(indices, results, strict=True):
                if result is not None:
                    ready_by_index[index] = result
            if len(ready_by_index) != len(self.leader_actors):
                if stop_event is None:
                    time.sleep(0.2)
                elif stop_event.wait(0.2):
                    raise InterruptedError("server startup interrupted")
        return [ready_by_index[index] for index in range(len(self.leader_actors))]

    def start(self, stop_event: Any = None) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            raise RuntimeError(
                f"refusing to reuse server manifest: {self.manifest_path}"
            )
        self._write_manifest("starting")
        try:
            self.ray = self._connect_ray()
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                NodeAffinitySchedulingStrategy,
                PlacementGroupSchedulingStrategy,
            )

            candidate_nodes = get_alive_gpu_nodes()
            primary_role = self.config.primary_role
            primary_settings = self.config.primary
            primary_tp_size = int(primary_settings.server_args.get("tp_size", 1))
            if self.config.deployment_kind == "decoupled_spec":
                assert self.config.drafter is not None
                self.placement_plan = plan_decoupled_spec_placement(
                    candidate_nodes,
                    num_verifiers=primary_settings.replicas,
                    target_tp_size=primary_tp_size,
                    num_drafters=self.config.drafter.replicas,
                    draft_tp_size=int(
                        self.config.drafter.server_args.get("tp_size", 1)
                    ),
                )
            else:
                self.placement_plan = plan_spec_engine_placement(
                    candidate_nodes,
                    num_replicas=primary_settings.replicas,
                    tp_size=primary_tp_size,
                )
            node_ip_by_id = {node.node_id: node.node_ip for node in candidate_nodes}

            for primary_rank, node_ids in enumerate(
                self.placement_plan.verifier_bundle_node_ids
            ):
                bundles = [
                    {
                        "CPU": 1,
                        "GPU": self.placement_plan.verifier_gpus_per_node,
                        f"node:{node_ip_by_id[node_id]}": 0.001,
                    }
                    for node_id in node_ids
                ]
                pg = placement_group(
                    bundles,
                    strategy="PACK" if len(bundles) == 1 else "STRICT_SPREAD",
                    name=(
                        f"spec-benchmark-{primary_role}{primary_rank}-"
                        f"{os.getpid()}-{time.monotonic_ns()}"
                    ),
                )
                self.primary_pgs.append(pg)
            self._get_refs(
                [pg.ready() for pg in self.primary_pgs],
                timeout_s=self.config.ray.placement_timeout_s,
                stop_event=stop_event,
            )

            remote_actor = self.ray.remote(EngineActor)
            engine_actor_groups: dict[str, list[Any]] = {}
            for primary_rank, (pg, node_ids) in enumerate(
                zip(
                    self.primary_pgs,
                    self.placement_plan.verifier_bundle_node_ids,
                    strict=True,
                )
            ):
                engine_id = f"{primary_role}-{primary_rank}"
                group = []
                engine_actor_groups[engine_id] = group
                nnodes = len(node_ids)
                for node_rank, node_id in enumerate(node_ids):
                    actor = remote_actor.options(
                        num_cpus=1,
                        num_gpus=self.placement_plan.verifier_gpus_per_node,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=node_rank,
                            placement_group_capture_child_tasks=False,
                        ),
                    ).remote(
                        engine_id=engine_id,
                        role=primary_role,
                        rank=primary_rank,
                        tp_size=primary_tp_size,
                        node_rank=node_rank,
                        nnodes=nnodes,
                        expected_num_gpus=(self.placement_plan.verifier_gpus_per_node),
                        runtime_env=primary_settings.runtime.env,
                        staged_files=self.verifier_staged_files,
                    )
                    group.append(actor)
                    self.actors.append(actor)
                    self.actor_records.append(
                        {
                            "engine_id": engine_id,
                            "role": primary_role,
                            "rank": primary_rank,
                            "node_rank": node_rank,
                            "planned_node_id": node_id,
                            "actor": actor,
                        }
                    )

            if self.config.drafter is not None:
                draft_tp_size = int(
                    self.config.drafter.server_args.get("tp_size", 1)
                )
                for drafter_rank, node_id in enumerate(
                    self.placement_plan.drafter_node_ids
                ):
                    engine_id = f"drafter-{drafter_rank}"
                    actor = remote_actor.options(
                        num_cpus=1,
                        num_gpus=draft_tp_size,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=node_id,
                            soft=False,
                        ),
                    ).remote(
                        engine_id=engine_id,
                        role="drafter",
                        rank=drafter_rank,
                        tp_size=draft_tp_size,
                        node_rank=0,
                        nnodes=1,
                        expected_num_gpus=draft_tp_size,
                        runtime_env=self.config.drafter.runtime.env,
                    )
                    engine_actor_groups[engine_id] = [actor]
                    self.actors.append(actor)
                    self.actor_records.append(
                        {
                            "engine_id": engine_id,
                            "role": "drafter",
                            "rank": drafter_rank,
                            "node_rank": 0,
                            "planned_node_id": node_id,
                            "actor": actor,
                        }
                    )

            allocations = self._get_refs(
                [record["actor"].allocation.remote() for record in self.actor_records],
                timeout_s=self.config.ray.startup_timeout_s,
                stop_event=stop_event,
            )
            for record, allocation in zip(self.actor_records, allocations, strict=True):
                if allocation["node_id"] != record["planned_node_id"]:
                    raise RuntimeError(
                        "Engine actor placement differs from joint plan: "
                        f"record={record} allocation={allocation}"
                    )
                record.update(allocation)

            leader_records = [
                record for record in self.actor_records if record["node_rank"] == 0
            ]
            self.leader_actors = [record["actor"] for record in leader_records]
            self.engine_infos = self._get_refs(
                [
                    actor.reserve_ports.remote(
                        self.config.deployment_kind == "decoupled_spec"
                    )
                    for actor in self.leader_actors
                ],
                timeout_s=self.config.ray.startup_timeout_s,
                stop_event=stop_event,
            )
            engine_info_by_id = {
                str(info["engine_id"]): info for info in self.engine_infos
            }
            for engine_id, group in engine_actor_groups.items():
                records = sorted(
                    (
                        record
                        for record in self.actor_records
                        if record["engine_id"] == engine_id
                    ),
                    key=lambda record: record["node_rank"],
                )
                engine_info = engine_info_by_id[engine_id]
                rank_placements = []
                tp_rank = 0
                node_actors = []
                for record in records:
                    if record["node_rank"] == 0:
                        resolved_path = (
                            Path("server")
                            / "engines"
                            / engine_id
                            / "resolved_config.json"
                        )
                        status_path = (
                            Path("server") / "engines" / engine_id / "status.json"
                        )
                        log_path = Path("logs") / "server" / f"{engine_id}.log"
                    else:
                        resolved_path = (
                            Path("server")
                            / "engines"
                            / engine_id
                            / "nodes"
                            / str(record["node_rank"])
                            / "resolved_config.json"
                        )
                        status_path = (
                            Path("server")
                            / "engines"
                            / engine_id
                            / "nodes"
                            / str(record["node_rank"])
                            / "status.json"
                        )
                        log_path = (
                            Path("logs")
                            / "server"
                            / f"{engine_id}-node{record['node_rank']}.log"
                        )
                    record.update(
                        {
                            "resolved_config_path": str(resolved_path),
                            "status_path": str(status_path),
                            "log_path": str(log_path),
                        }
                    )
                    node_actor = {
                        key: record[key]
                        for key in (
                            "node_rank",
                            "node_id",
                            "node_ip",
                            "gpu_ids",
                            "resolved_config_path",
                            "status_path",
                            "log_path",
                        )
                    }
                    if record.get("staged_artifacts"):
                        node_actor["staged_artifacts"] = record["staged_artifacts"]
                    node_actors.append(node_actor)
                    for local_gpu_index, gpu_id in enumerate(record["gpu_ids"]):
                        rank_placements.append(
                            {
                                "tp_rank": tp_rank,
                                "node_rank": record["node_rank"],
                                "local_gpu_index": local_gpu_index,
                                "node_id": record["node_id"],
                                "node_ip": record["node_ip"],
                                "gpu_id": gpu_id,
                            }
                        )
                        tp_rank += 1
                if tp_rank != int(engine_info["tp_size"]):
                    raise RuntimeError(
                        f"{engine_id} TP placement has {tp_rank} ranks, "
                        f"expected {engine_info['tp_size']}"
                    )
                engine_info.update(
                    {
                        "rank_placements": rank_placements,
                        "node_actors": node_actors,
                        "resolved_config_path": node_actors[0]["resolved_config_path"],
                        "status_path": node_actors[0]["status_path"],
                        "log_path": node_actors[0]["log_path"],
                    }
                )

            if self.config.deployment_kind == "decoupled_spec":
                assert self.config.verifier is not None
                assert self.config.drafter is not None
                quota_edges = plan_decoupled_spec_quota_edges(
                    self.config.verifier.replicas,
                    self.config.drafter.replicas,
                )
                peers_by_engine = build_peer_configs(self.engine_infos, quota_edges)
            else:
                peers_by_engine = {
                    str(info["engine_id"]): [] for info in self.engine_infos
                }
            starts = []
            for record in self.actor_records:
                engine_id = record["engine_id"]
                engine_info = engine_info_by_id[engine_id]
                role = record["role"]
                settings = getattr(self.config, role)
                assert settings is not None
                dist_init_addr = NetworkAddress(
                    str(engine_info["node_ip"]),
                    int(engine_info["nccl_port"]),
                ).to_host_port_str()
                server_args = dict(settings.server_args)
                server_args.setdefault("log_level", "info")
                server_args.update(
                    {
                        "host": str(record["node_ip"]),
                        "port": int(engine_info["http_port"]),
                        "nccl_port": int(engine_info["nccl_port"]),
                        "nnodes": len(engine_actor_groups[engine_id]),
                        "node_rank": int(record["node_rank"]),
                        "dist_init_addr": dist_init_addr,
                    }
                )
                if self.config.deployment_kind == "decoupled_spec":
                    server_args.update(
                        {
                            "decoupled_spec_role": role,
                            "decoupled_spec_rank": int(engine_info["rank"]),
                            "decoupled_spec_bind_endpoint": str(
                                engine_info["transport_endpoint"]
                            ),
                            "decoupled_spec_peer_configs": peers_by_engine[engine_id],
                        }
                    )
                resolved = {
                    "schema_version": SCHEMA_VERSION,
                    "engine_id": engine_id,
                    "role": role,
                    "rank": engine_info["rank"],
                    "node_rank": record["node_rank"],
                    "runtime": msgspec.to_builtins(settings.runtime),
                    "server_args": server_args,
                }
                _write_json(
                    self.run_dir / record["resolved_config_path"],
                    resolved,
                )
                starts.append(record["actor"].start.remote(resolved))
            start_infos = self._get_refs(
                starts,
                timeout_s=self.config.ray.startup_timeout_s,
                stop_event=stop_event,
            )
            for record, start_info in zip(self.actor_records, start_infos, strict=True):
                record["start_info"] = start_info
                _write_json(
                    self.run_dir / record["status_path"],
                    {
                        "engine_id": record["engine_id"],
                        "role": record["role"],
                        "rank": record["rank"],
                        "node_rank": record["node_rank"],
                        "state": start_info["state"],
                        "pid": start_info["pid"],
                        "node_id": record["node_id"],
                        "node_ip": record["node_ip"],
                        "gpu_ids": record["gpu_ids"],
                        "updated_at": time.time(),
                    },
                )

            ready_infos = self._wait_http_ready(stop_event)
            for engine_info, ready_info in zip(
                self.engine_infos, ready_infos, strict=True
            ):
                if ready_info.get("engine_id") != engine_info["engine_id"]:
                    raise RuntimeError(
                        "engine readiness identity mismatch: "
                        f"{engine_info['engine_id']} vs {ready_info!r}"
                    )
                engine_info["peer_configs"] = peers_by_engine[
                    str(engine_info["engine_id"])
                ]
                engine_info["ready_at"] = ready_info["updated_at"]
                _write_json(
                    self.run_dir / str(engine_info["status_path"]),
                    ready_info,
                )
            return self._write_manifest("ready")
        except InterruptedError:
            self._write_manifest("stopping")
            self.shutdown(final_state="stopped")
            raise
        except BaseException as exc:
            self._write_manifest("failed", error=repr(exc))
            self.shutdown(final_state="failed")
            raise

    def wait(self, stop_event: Any) -> None:
        while not stop_event.wait(1.0):
            return_codes = self.ray.get(
                [actor.poll.remote() for actor in self.actors], timeout=10.0
            )
            exited = [
                (record["engine_id"], record["node_rank"], code)
                for record, code in zip(self.actor_records, return_codes, strict=True)
                if code is not None
            ]
            if exited:
                raise RuntimeError(f"server engine exited unexpectedly: {exited}")

    def shutdown(self, *, final_state: str = "stopped") -> dict[str, Any]:
        if self._shutdown:
            if self.manifest_path.exists():
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return {"state": final_state}
        self._shutdown = True
        stop_results: list[dict[str, Any]] = []
        if self.ray is not None and self.actors:
            refs = [
                actor.stop.remote(self.config.ray.shutdown_grace_s)
                for actor in self.actors
            ]
            try:
                stop_results = self.ray.get(
                    refs,
                    timeout=self.config.ray.shutdown_grace_s + 15.0,
                )
            except BaseException as exc:
                stop_results = [{"error": repr(exc)}]
            manifest_stop_results = []
            record_by_key = {
                (str(record["engine_id"]), int(record["node_rank"])): record
                for record in self.actor_records
            }
            for result in stop_results:
                result = dict(result)
                artifacts = result.pop("artifacts", None)
                engine_id = str(result.get("engine_id", ""))
                node_rank = int(result.get("node_rank", -1))
                record = record_by_key.get((engine_id, node_rank))
                if isinstance(artifacts, dict) and record is not None:
                    resolved_config = artifacts.get("resolved_config")
                    if isinstance(resolved_config, dict):
                        _write_json(
                            self.run_dir / str(record["resolved_config_path"]),
                            resolved_config,
                        )
                    status = artifacts.get("status")
                    if isinstance(status, dict):
                        _write_json(self.run_dir / str(record["status_path"]), status)
                    log_bytes = artifacts.get("log_bytes")
                    if isinstance(log_bytes, bytes):
                        log_path = self.run_dir / str(record["log_path"])
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_bytes(log_bytes)
                manifest_stop_results.append(result)
            stop_results = manifest_stop_results
            for actor in self.actors:
                try:
                    self.ray.kill(actor, no_restart=True)
                except BaseException:
                    pass
        if self.ray is not None and self.primary_pgs:
            from ray.util.placement_group import remove_placement_group

            for pg in self.primary_pgs:
                try:
                    remove_placement_group(pg)
                except BaseException:
                    pass
        manifest = self._write_manifest(final_state, shutdown=stop_results)
        if self.ray is not None:
            self.ray.shutdown()
        return manifest


def print_ready_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    print(f"\n{manifest.get('deployment', 'decoupled_spec')} server fleet is ready:")
    print(
        f"{'ROLE':<9} {'RANK':>4} {'ENGINE':<16} {'NODE/IP':<30} "
        f"{'HTTP':<32} {'TRANSPORT':<32} {'NCCL':>5} NODES"
    )
    for engine in manifest["engines"]:
        node = f"{engine['node_id'][:8]}/{engine['node_ip']}"
        print(
            f"{engine['role']:<9} {engine['rank']:>4} "
            f"{engine['engine_id']:<16} {node:<30} "
            f"{engine['http_url']:<32} "
            f"{engine.get('transport_endpoint', '-'):<32} "
            f"{engine['nccl_port']:>5} "
            f"{len(engine.get('node_actors', []))}"
        )
    print("\nTP rank placements:")
    for engine in manifest["engines"]:
        placements = " ".join(
            f"TP{placement['tp_rank']}={placement['node_ip']}/GPU{placement['gpu_id']}"
            for placement in engine.get("rank_placements", [])
        )
        print(f"{engine['engine_id']}: {placements}")
    quota_edges = manifest["topology"]["quota_edges"]
    if quota_edges:
        print("\nSparse quota routes:")
        print(f"{'VERIFIER':>8} {'DRAFTER':>7} {'QUOTA':>5}")
        for edge in quota_edges:
            print(
                f"{edge['verifier_rank']:>8} {edge['drafter_rank']:>7} "
                f"{edge['quota']:>5}"
            )
    print(
        "DSPEC_SERVER_MANIFEST "
        + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    )
    print(f"SERVER_MANIFEST={manifest_path}", flush=True)


__all__ = [
    "EngineActor",
    "MANIFEST_RELATIVE_PATH",
    "RayServerOrchestrator",
    "UnifiedServerConfig",
    "build_peer_configs",
    "config_to_dict",
    "load_config",
    "plan_decoupled_spec_quota_edges",
    "print_ready_manifest",
    "validate_config",
    "validate_server_args_templates",
]

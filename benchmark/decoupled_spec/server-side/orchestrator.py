"""Ray orchestration for a jointly placed decoupled-spec server fleet."""

from __future__ import annotations

import dataclasses
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

SCHEMA_VERSION = 1
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
    """Complete verifier and drafter fleet configuration."""

    verifier: RoleSettings
    drafter: RoleSettings
    ray: RaySettings = msgspec.field(default_factory=RaySettings)
    schema_version: int = SCHEMA_VERSION


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
        raise ValueError(f"decoupled {role} requires dp_size=1 and pp_size=1")
    if role == "verifier":
        algorithm = args.get("speculative_algorithm")
        if not isinstance(algorithm, str) or algorithm.upper() != "DECOUPLED_VERIFY":
            raise ValueError(
                "verifier.server_args.speculative_algorithm must be " "DECOUPLED_VERIFY"
            )
    else:
        if args.get("speculative_algorithm") is not None:
            raise ValueError("drafter.server_args.speculative_algorithm must be unset")
        if not args.get("disable_overlap_schedule", False):
            raise ValueError(
                "drafter.server_args.disable_overlap_schedule must be true"
            )
        if int(args.get("tp_size", 1)) != 1:
            raise ValueError("the current decoupled drafter requires tp_size=1")
        if int(args.get("page_size", 1)) != 1:
            raise ValueError("the current decoupled drafter requires page_size=1")
    return args


def validate_config(config: UnifiedServerConfig) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={config.schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    for name, value in (
        ("ray.placement_timeout_s", config.ray.placement_timeout_s),
        ("ray.startup_timeout_s", config.ray.startup_timeout_s),
        ("ray.shutdown_grace_s", config.ray.shutdown_grace_s),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

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

    verifier_cpp = _normalize_bool_env(
        config.verifier.runtime.env.get(_CPP_DATA_PLANE_ENV, "0"),
        _CPP_DATA_PLANE_ENV,
    )
    drafter_cpp = _normalize_bool_env(
        config.drafter.runtime.env.get(_CPP_DATA_PLANE_ENV, "0"),
        _CPP_DATA_PLANE_ENV,
    )
    if verifier_cpp != drafter_cpp:
        raise ValueError(
            "verifier and drafter must use the same Python/C++ data-plane backend"
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
    return msgspec.to_builtins(config)


def validate_server_args_templates(config: UnifiedServerConfig) -> None:
    """Resolve each role's real ServerArgs with a synthetic local topology."""

    dynamic = {
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
    for role in ("verifier", "drafter"):
        settings = getattr(config, role)
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


def _reserve_tcp_socket(host: str) -> socket.socket:
    family = socket.AF_INET6 if _is_ipv6(host) else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if family == socket.AF_INET6:
            sock.bind((host, 0, 0, 0))
        else:
            sock.bind((host, 0))
        sock.listen(1)
        return sock
    except BaseException:
        sock.close()
        raise


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
    """Own one single-node TP replica, its leases, subprocess, and log."""

    def __init__(
        self,
        *,
        engine_id: str,
        role: str,
        rank: int,
        tp_size: int,
        runtime_env: dict[str, Any],
        node_info_override: dict[str, str] | None = None,
        gpu_ids_override: list[str] | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.role = role
        self.rank = rank
        self.tp_size = tp_size
        # Actor-local files must not assume a shared filesystem across Ray nodes.
        self.local_dir = Path(tempfile.mkdtemp(prefix=f"sglang-{engine_id}-"))
        self.runtime_env = {str(key): str(value) for key, value in runtime_env.items()}
        self.node_info_override = node_info_override
        self.gpu_ids_override = gpu_ids_override
        self.node_id: str | None = None
        self.node_ip: str | None = None
        self.gpu_ids: list[str] = []
        self._leases: dict[str, socket.socket] = {}
        self._process: subprocess.Popen[bytes] | None = None
        self._log_stream: Any = None
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
        gpu_ids = [str(value) for value in context.get_accelerator_ids().get("GPU", [])]
        node_ip = NetworkAddress(str(ray.util.get_node_ip_address()), 0).host
        return (
            str(context.get_node_id()),
            node_ip,
            gpu_ids,
        )

    def reserve_ports(self) -> dict[str, Any]:
        import sglang

        if self._leases:
            raise RuntimeError(f"{self.engine_id} already owns port leases")
        self.node_id, self.node_ip, self.gpu_ids = self._runtime_identity()
        if self.gpu_ids_override is None and len(self.gpu_ids) != self.tp_size:
            raise RuntimeError(
                f"{self.engine_id} expected {self.tp_size} Ray GPUs, "
                f"got {self.gpu_ids}"
            )
        try:
            self._leases = {
                name: _reserve_tcp_socket(self.node_ip)
                for name in ("http", "transport", "nccl")
            }
        except BaseException:
            self._release_port_leases()
            raise
        ports = {
            name: int(sock.getsockname()[1]) for name, sock in self._leases.items()
        }
        return {
            "engine_id": self.engine_id,
            "role": self.role,
            "rank": self.rank,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "http_url": f"http://{_host_port(self.node_ip, ports['http'])}",
            "transport_endpoint": (
                f"tcp://{_host_port(self.node_ip, ports['transport'])}"
            ),
            "nccl_port": ports["nccl"],
            "http_port": ports["http"],
            "tp_size": self.tp_size,
            "gpu_ids": self.gpu_ids,
            "sglang_path": str(Path(sglang.__file__).resolve()),
        }

    def _release_port_leases(self) -> None:
        for sock in self._leases.values():
            sock.close()
        self._leases.clear()

    def start(self, resolved_config: dict[str, Any]) -> dict[str, Any]:
        if self._process is not None:
            raise RuntimeError(f"{self.engine_id} was already started")
        if set(self._leases) != {"http", "transport", "nccl"}:
            raise RuntimeError(f"{self.engine_id} does not own all required ports")

        engine_dir = self.local_dir / "engine"
        log_path = self.local_dir / f"{self.engine_id}.log"
        config_path = engine_dir / "resolved_config.json"
        status_path = engine_dir / "status.json"
        if status_path.exists():
            raise RuntimeError(
                f"refusing to reuse existing engine state: {status_path}"
            )
        self._resolved_config = resolved_config
        _write_json(config_path, resolved_config)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = log_path.open("ab", buffering=0)
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

        # Every endpoint was socket-backed until this point. Release immediately
        # before Popen so no other orchestrated engine can receive the same port.
        self._release_port_leases()
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self.local_dir,
                env=env,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self._close_log()
            raise
        return {
            "pid": self._process.pid,
        }

    def _tail_log(self, limit: int = 80) -> str:
        path = self.local_dir / f"{self.engine_id}.log"
        try:
            return "\n".join(path.read_text(encoding="utf-8").splitlines()[-limit:])
        except (OSError, UnicodeDecodeError):
            return "<log unavailable>"

    def readiness(self) -> dict[str, Any] | None:
        """Return ready metadata after one bounded probe, or None while starting."""
        if self._process is None:
            raise RuntimeError(f"{self.engine_id} has not started")
        status_path = self.local_dir / "engine" / "status.json"
        endpoint_path = "/health" if self.role == "verifier" else "/model_info"
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
        return None if self._process is None else self._process.poll()

    def _close_log(self) -> None:
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
        return_code = None if self._process is None else self._process.poll()
        self._close_log()
        log_path = self.local_dir / f"{self.engine_id}.log"
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
            }
        if process_group_alive:
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
    """Create one joint PG and manage all role HTTP processes as a unit."""

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
        self.placement_group: Any = None
        self.actors: list[Any] = []
        self.engine_infos: list[dict[str, Any]] = []
        self._shutdown = False

    def _write_manifest(self, state: str, **extra: Any) -> dict[str, Any]:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": time.time(),
            "config_path": str(self.config_path),
            "ray_address": self.ray_address,
            "ray_namespace": self.ray_namespace,
            "topology": {
                "num_verifiers": self.config.verifier.replicas,
                "num_drafters": self.config.drafter.replicas,
                "quota_edges": plan_decoupled_spec_quota_edges(
                    self.config.verifier.replicas, self.config.drafter.replicas
                ),
            },
            "engines": self.engine_infos,
            **extra,
        }
        _write_json(self.manifest_path, manifest)
        return manifest

    def _connect_ray(self) -> Any:
        import ray

        repository_root = Path(__file__).resolve().parents[3]
        ray.init(
            address=self.ray_address,
            namespace=self.ray_namespace,
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env={
                "py_modules": [
                    str(Path(__file__).resolve().parent),
                    str(repository_root / "python" / "sglang"),
                ],
                "excludes": ["**/__pycache__", "**/*.pyc"],
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
        while len(ready_by_index) != len(self.actors):
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("server startup interrupted")
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                missing = [
                    self.engine_infos[index]["engine_id"]
                    for index in range(len(self.actors))
                    if index not in ready_by_index
                ]
                raise TimeoutError(
                    "servers were not HTTP-ready after "
                    f"{self.config.ray.startup_timeout_s:.1f}s: {missing}"
                )
            indices = [
                index
                for index in range(len(self.actors))
                if index not in ready_by_index
            ]
            refs = [self.actors[index].readiness.remote() for index in indices]
            results = self._get_refs(
                refs,
                timeout_s=min(5.0, remaining_s),
                stop_event=stop_event,
            )
            for index, result in zip(indices, results, strict=True):
                if result is not None:
                    ready_by_index[index] = result
            if len(ready_by_index) != len(self.actors):
                if stop_event is None:
                    time.sleep(0.2)
                elif stop_event.wait(0.2):
                    raise InterruptedError("server startup interrupted")
        return [ready_by_index[index] for index in range(len(self.actors))]

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
                PlacementGroupSchedulingStrategy,
            )

            identities: list[tuple[str, int, RoleSettings]] = []
            for role, settings in (
                ("verifier", self.config.verifier),
                ("drafter", self.config.drafter),
            ):
                identities.extend(
                    (role, rank, settings) for rank in range(settings.replicas)
                )
            bundles = [
                {
                    "CPU": 1,
                    "GPU": int(settings.server_args.get("tp_size", 1)),
                }
                for _, _, settings in identities
            ]
            self.placement_group = placement_group(
                bundles,
                strategy="PACK",
                name=f"decoupled-spec-{os.getpid()}-{time.monotonic_ns()}",
            )
            self._get_refs(
                [self.placement_group.ready()],
                timeout_s=self.config.ray.placement_timeout_s,
                stop_event=stop_event,
            )

            remote_actor = self.ray.remote(EngineActor)
            for bundle_index, (role, rank, settings) in enumerate(identities):
                tp_size = int(settings.server_args.get("tp_size", 1))
                engine_id = f"{role}-{rank}"
                actor = remote_actor.options(
                    num_cpus=1,
                    num_gpus=tp_size,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_index,
                        placement_group_capture_child_tasks=False,
                    ),
                ).remote(
                    engine_id=engine_id,
                    role=role,
                    rank=rank,
                    tp_size=tp_size,
                    runtime_env=settings.runtime.env,
                )
                self.actors.append(actor)

            self.engine_infos = self._get_refs(
                [actor.reserve_ports.remote() for actor in self.actors],
                timeout_s=self.config.ray.startup_timeout_s,
                stop_event=stop_event,
            )
            quota_edges = plan_decoupled_spec_quota_edges(
                self.config.verifier.replicas, self.config.drafter.replicas
            )
            peers_by_engine = build_peer_configs(self.engine_infos, quota_edges)
            starts = []
            for actor, engine_info in zip(self.actors, self.engine_infos, strict=True):
                role = str(engine_info["role"])
                settings = getattr(self.config, role)
                server_args = dict(settings.server_args)
                server_args.update(
                    {
                        "host": str(engine_info["node_ip"]),
                        "port": int(engine_info["http_port"]),
                        "nccl_port": int(engine_info["nccl_port"]),
                        "decoupled_spec_role": role,
                        "decoupled_spec_rank": int(engine_info["rank"]),
                        "decoupled_spec_bind_endpoint": str(
                            engine_info["transport_endpoint"]
                        ),
                        "decoupled_spec_peer_configs": peers_by_engine[
                            str(engine_info["engine_id"])
                        ],
                    }
                )
                resolved = {
                    "schema_version": SCHEMA_VERSION,
                    "engine_id": engine_info["engine_id"],
                    "role": role,
                    "rank": engine_info["rank"],
                    "runtime": msgspec.to_builtins(settings.runtime),
                    "server_args": server_args,
                }
                engine_dir = (
                    self.run_dir / "server" / "engines" / str(engine_info["engine_id"])
                )
                _write_json(engine_dir / "resolved_config.json", resolved)
                starts.append(actor.start.remote(resolved))
                engine_info["peer_configs"] = peers_by_engine[
                    str(engine_info["engine_id"])
                ]
                engine_info.update(
                    {
                        "log_path": str(
                            Path("logs") / "server" / f"{engine_info['engine_id']}.log"
                        ),
                        "resolved_config_path": str(
                            Path("server")
                            / "engines"
                            / str(engine_info["engine_id"])
                            / "resolved_config.json"
                        ),
                        "status_path": str(
                            Path("server")
                            / "engines"
                            / str(engine_info["engine_id"])
                            / "status.json"
                        ),
                    }
                )
            start_infos = self._get_refs(
                starts,
                timeout_s=self.config.ray.startup_timeout_s,
                stop_event=stop_event,
            )
            for engine_info, start_info in zip(
                self.engine_infos, start_infos, strict=True
            ):
                engine_info.update(start_info)

            ready_infos = self._wait_http_ready(stop_event)
            for engine_info, ready_info in zip(
                self.engine_infos, ready_infos, strict=True
            ):
                if ready_info.get("engine_id") != engine_info["engine_id"]:
                    raise RuntimeError(
                        "engine readiness identity mismatch: "
                        f"{engine_info['engine_id']} vs {ready_info!r}"
                    )
                engine_info["ready_at"] = ready_info["updated_at"]
                _write_json(self.run_dir / str(engine_info["status_path"]), ready_info)
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
                (info["engine_id"], code)
                for info, code in zip(self.engine_infos, return_codes, strict=True)
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
            engine_by_id = {str(info["engine_id"]): info for info in self.engine_infos}
            for result in stop_results:
                result = dict(result)
                artifacts = result.pop("artifacts", None)
                engine_id = str(result.get("engine_id", ""))
                engine_info = engine_by_id.get(engine_id)
                if isinstance(artifacts, dict) and engine_info is not None:
                    resolved_config = artifacts.get("resolved_config")
                    if isinstance(resolved_config, dict):
                        _write_json(
                            self.run_dir / str(engine_info["resolved_config_path"]),
                            resolved_config,
                        )
                    status = artifacts.get("status")
                    if isinstance(status, dict):
                        _write_json(
                            self.run_dir / str(engine_info["status_path"]), status
                        )
                    log_bytes = artifacts.get("log_bytes")
                    if isinstance(log_bytes, bytes):
                        log_path = self.run_dir / str(engine_info["log_path"])
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_bytes(log_bytes)
                manifest_stop_results.append(result)
            stop_results = manifest_stop_results
            for actor in self.actors:
                try:
                    self.ray.kill(actor, no_restart=True)
                except BaseException:
                    pass
        if self.ray is not None and self.placement_group is not None:
            try:
                from ray.util.placement_group import remove_placement_group

                remove_placement_group(self.placement_group)
            except BaseException:
                pass
        manifest = self._write_manifest(final_state, shutdown=stop_results)
        if self.ray is not None:
            self.ray.shutdown()
        return manifest


def print_ready_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    print("\nDecoupled-spec server fleet is ready:")
    print(
        f"{'ROLE':<9} {'RANK':>4} {'ENGINE':<16} {'NODE/IP':<30} "
        f"{'HTTP':<32} {'TRANSPORT':<32} {'NCCL':>5} GPUS"
    )
    for engine in manifest["engines"]:
        node = f"{engine['node_id'][:8]}/{engine['node_ip']}"
        print(
            f"{engine['role']:<9} {engine['rank']:>4} "
            f"{engine['engine_id']:<16} {node:<30} "
            f"{engine['http_url']:<32} "
            f"{engine['transport_endpoint']:<32} "
            f"{engine['nccl_port']:>5} "
            f"{','.join(engine['gpu_ids'])}"
        )
    print("\nSparse quota routes:")
    print(f"{'VERIFIER':>8} {'DRAFTER':>7} {'QUOTA':>5}")
    for edge in manifest["topology"]["quota_edges"]:
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

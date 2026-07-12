from __future__ import annotations

import argparse
import ipaddress
import os
import socket
from typing import Any

try:
    import ray
    from ray.util.scheduling_strategies import (
        NodeAffinitySchedulingStrategy,
    )
except ImportError:

    class _MissingRay:
        def remote(self, obj=None, **_kwargs):
            if obj is None:
                return lambda inner: inner
            return obj

        def __getattr__(self, name: str):
            raise ImportError(
                "ray is required for Ray-based decoupled speculation helpers"
            ) from None

    ray = _MissingRay()
    NodeAffinitySchedulingStrategy = None

try:
    import sglang as sgl
    from sglang.srt.utils.network import is_valid_ipv6_address
except ImportError:

    class _MissingSGLang:
        def __getattr__(self, name: str):
            raise ImportError(
                "sglang is required for engine-based decoupled speculation helpers"
            ) from None

    sgl = _MissingSGLang()

    def is_valid_ipv6_address(ip: str) -> bool:
        try:
            return ipaddress.ip_address(ip).version == 6
        except ValueError:
            return False

from .types import DecoupledSpecEndpointInfo, DecoupledSpecTopology

DPA_DIST_INIT_PORT_BLOCK_SIZE = 6


def parse_reserved_ports(raw_ports: str | None) -> list[int]:
    if raw_ports is None:
        return []
    ports: list[int] = []
    for raw_port in raw_ports.replace(",", " ").split():
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError(f"invalid reserved port: {raw_port!r}") from exc
        if port <= 0 or port > 65535:
            raise ValueError(f"reserved port out of range: {port}")
        ports.append(port)
    if len(set(ports)) != len(ports):
        raise ValueError(f"reserved ports must be unique: {ports}")
    return ports


def target_uses_env_available_ports(args: argparse.Namespace) -> bool:
    return args.target_use_env_ports


def target_dp_attention_uses_dist_init_derived_ports(
    args: argparse.Namespace,
) -> bool:
    return args.target_enable_dp_attention and not args.target_use_env_ports


def dist_init_port_stride(args: argparse.Namespace) -> int:
    return (
        DPA_DIST_INIT_PORT_BLOCK_SIZE
        if target_dp_attention_uses_dist_init_derived_ports(args)
        else 1
    )


def reserved_port_block_bases(
    reserved_ports: list[int], *, num_blocks: int, block_size: int
) -> list[int]:
    required_ports = num_blocks * block_size
    if len(reserved_ports) < required_ports:
        raise ValueError(
            f"--reserved-ports provides {len(reserved_ports)} ports, but this "
            f"run needs {required_ports}"
        )
    bases = []
    for block_index in range(num_blocks):
        start = block_index * block_size
        block = reserved_ports[start : start + block_size]
        expected = list(range(block[0], block[0] + block_size))
        if block != expected:
            raise ValueError(
                "DP attention requires each reserved dist-init allocation to be "
                f"a contiguous {block_size}-port block; block {block_index} is "
                f"{block}, expected {expected}"
            )
        bases.append(block[0])
    return bases


def get_target_engine_tp_size(args: argparse.Namespace) -> int:
    if args.target_enable_dp_attention:
        return args.target_tp_size * args.target_dp_size
    return args.target_tp_size


def format_host_port(ip: str, port: int | str) -> str:
    """Return a host:port string, preserving IPv6 bracket formatting."""
    host = f"[{ip}]" if is_valid_ipv6_address(ip) else ip
    return f"{host}:{port}"


def format_tcp_address(ip: str, port: int | str) -> str:
    """Return a ZMQ TCP endpoint, preserving IPv6 bracket formatting."""
    return f"tcp://{format_host_port(ip, port)}"


def get_decoupled_spec_actor_env_vars() -> dict[str, str]:
    """Collect decoupled-spec environment variables for Ray actors."""
    env_vars: dict[str, str] = {
        "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL": os.environ.get(
            "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL", "1"
        )
    }
    for env_name in (
        "CUDA_LAUNCH_BLOCKING",
        "SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            env_vars[env_name] = env_value
    return env_vars


def _sort_gpu_ids(gpu_ids: list[Any]) -> list[str]:
    """Sort Ray/CUDA GPU ids numerically when possible and return strings."""

    def sort_key(value: Any) -> tuple[int, Any]:
        """Build a stable sort key for numeric and non-numeric GPU ids."""
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    return [str(value) for value in sorted(gpu_ids, key=sort_key)]


def _get_assigned_gpu_ids_from_ray() -> list[str]:
    """Read the GPU ids assigned to the current Ray actor."""
    context = ray.get_runtime_context()
    accelerator_ids = getattr(context, "get_accelerator_ids", lambda: {})()
    gpu_ids = accelerator_ids.get("GPU", [])
    if gpu_ids:
        return _sort_gpu_ids(gpu_ids)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        return _sort_gpu_ids(
            [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        )
    return []


def pin_actor_to_assigned_gpus(expected_num_gpus: int) -> list[str]:
    """Set CUDA_VISIBLE_DEVICES to the GPUs Ray assigned to this actor."""
    gpu_ids = _get_assigned_gpu_ids_from_ray()
    if expected_num_gpus > 0 and len(gpu_ids) < expected_num_gpus:
        raise RuntimeError(
            f"Ray assigned {len(gpu_ids)} GPUs to actor, expected at least "
            f"{expected_num_gpus}: {gpu_ids}"
        )
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return gpu_ids


def is_tcp_port_available(port: int) -> bool:
    """Return whether a TCP port can be bound on this host."""
    try:
        infos = socket.getaddrinfo(
            None,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
            0,
            socket.AI_ADDRCONFIG | socket.AI_PASSIVE,
        )
    except socket.gaierror:
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", port))]

    seen_families: set[int] = set()
    for family, socktype, proto, _, sockaddr in infos:
        if family in seen_families:
            continue
        seen_families.add(family)
        sock = socket.socket(family, socktype, proto)
        try:
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
        except (OSError, OverflowError):
            return False
        finally:
            sock.close()
    return True


def reserve_tcp_port(
    preferred_port: int | None = None,
    avoid_ports: set[int] | None = None,
    bind_host: str | None = None,
) -> tuple[int, socket.socket]:
    """Bind and hold a TCP port, returning both the port and lock socket."""
    avoid_ports = set(avoid_ports or ())
    bind_host = (bind_host or "0.0.0.0").strip()
    if bind_host.startswith("[") and bind_host.endswith("]"):
        bind_host = bind_host[1:-1]
    is_ipv6 = is_valid_ipv6_address(bind_host)
    socket_family = socket.AF_INET6 if is_ipv6 else socket.AF_INET

    def bind_port(port: int) -> socket.socket:
        """Bind a listening socket to the same address family as the endpoint."""
        sock = socket.socket(socket_family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if is_ipv6 and hasattr(socket, "IPV6_V6ONLY"):
            try:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except OSError:
                pass
        if is_ipv6:
            sock.bind((bind_host, port, 0, 0))
        else:
            sock.bind((bind_host, port))
        sock.listen(1)
        return sock

    if preferred_port is not None:
        if preferred_port in avoid_ports:
            raise RuntimeError(
                f"preferred port {preferred_port} conflicts with avoided ports "
                f"{sorted(avoid_ports)}"
            )
        return preferred_port, bind_port(preferred_port)

    for _ in range(256):
        probe = bind_port(0)
        candidate_port = int(probe.getsockname()[1])
        if candidate_port in avoid_ports:
            probe.close()
            continue
        return candidate_port, probe

    raise RuntimeError("failed to reserve a TCP port")


def reserve_tcp_port_block(
    preferred_base_port: int | None = None,
    *,
    num_ports: int,
    avoid_ports: set[int] | None = None,
    bind_host: str | None = None,
) -> tuple[int, list[socket.socket]]:
    """Bind and hold a contiguous TCP port block."""
    if num_ports <= 0:
        raise ValueError("num_ports must be positive")
    avoid_ports = set(avoid_ports or ())

    def reserve_from_base(base_port: int) -> list[socket.socket]:
        if base_port <= 0 or base_port + num_ports - 1 > 65535:
            raise RuntimeError(
                f"port block out of range: {base_port}.."
                f"{base_port + num_ports - 1}"
            )
        block_ports = {base_port + offset for offset in range(num_ports)}
        if block_ports & avoid_ports:
            raise RuntimeError(
                f"port block {base_port}..{base_port + num_ports - 1} "
                f"conflicts with avoided ports {sorted(block_ports & avoid_ports)}"
            )
        sockets = []
        try:
            for offset in range(num_ports):
                _, sock = reserve_tcp_port(
                    base_port + offset,
                    bind_host=bind_host,
                )
                sockets.append(sock)
        except Exception:
            for sock in sockets:
                sock.close()
            raise
        return sockets

    if preferred_base_port is not None:
        return preferred_base_port, reserve_from_base(preferred_base_port)

    for _ in range(256):
        base_port, first_socket = reserve_tcp_port(bind_host=bind_host)
        if base_port + num_ports - 1 > 65535:
            first_socket.close()
            continue
        block_ports = {base_port + offset for offset in range(num_ports)}
        if block_ports & avoid_ports:
            first_socket.close()
            continue
        sockets = [first_socket]
        try:
            for offset in range(1, num_ports):
                _, sock = reserve_tcp_port(
                    base_port + offset,
                    bind_host=bind_host,
                )
                sockets.append(sock)
            return base_port, sockets
        except Exception:
            for sock in sockets:
                sock.close()

    raise RuntimeError(f"failed to reserve a TCP block of {num_ports} ports")


def _get_alive_gpu_nodes() -> list[dict[str, Any]]:
    """Return alive Ray nodes with currently available accelerator capacity."""
    available_resources = ray._private.state.available_resources_per_node()
    node_infos = {node["NodeID"]: node for node in ray.nodes() if node.get("Alive")}

    candidates = []
    for node_id, node in node_infos.items():
        node_resources = available_resources.get(node_id, {})
        available_gpus = int(node_resources.get("GPU", node_resources.get("NPU", 0)))
        if available_gpus <= 0:
            continue
        candidates.append(
            {
                "node_id": node_id,
                "node_ip": node["NodeManagerAddress"],
                "available_gpus": available_gpus,
            }
        )

    candidates.sort(key=lambda item: (-item["available_gpus"], item["node_ip"]))
    return candidates


def plan_draft_placement(args: argparse.Namespace) -> list[str]:
    """Plan drafter actor node placement from currently available Ray GPUs."""
    if args.draft_tp_size <= 0:
        raise ValueError("draft-tp-size must be positive")
    if args.num_draft_replicas is None or args.num_draft_replicas <= 0:
        raise ValueError("num-draft-replicas must be positive")

    candidate_nodes = _get_alive_gpu_nodes()
    total_capacity = sum(
        node["available_gpus"] // args.draft_tp_size for node in candidate_nodes
    )
    if total_capacity < args.num_draft_replicas:
        raise ValueError(
            "Not enough free GPUs for drafters after reserving verifier resources: "
            f"need {args.num_draft_replicas} replicas with "
            f"tp_size={args.draft_tp_size}, "
            f"capacity is {total_capacity}"
        )

    remaining = args.num_draft_replicas
    node_assignments: list[str] = []
    for node in candidate_nodes:
        capacity = node["available_gpus"] // args.draft_tp_size
        take = min(capacity, remaining)
        node_assignments.extend([node["node_id"]] * take)
        remaining -= take
        if remaining == 0:
            break

    if remaining != 0:
        raise ValueError(
            f"Unable to place {args.num_draft_replicas} drafters with "
            f"tp_size={args.draft_tp_size}"
        )
    return node_assignments


@ray.remote
class PortActor:
    """Ray actor used to reserve bootstrap ports on a specific placement node."""

    def __init__(self):
        """Initialize without a reservation; callers reserve ports explicitly."""
        self._reserved_sockets: list[socket.socket] = []

    def reserve_port(
        self,
        preferred_port: int | None = None,
        avoid_ports: list[int] | None = None,
    ) -> dict[str, Any]:
        """Reserve a TCP port on this actor's node and report host and port."""
        self.release_port()
        bind_host = ray.util.get_node_ip_address()
        port, sock = reserve_tcp_port(
            preferred_port,
            avoid_ports=set(avoid_ports or ()),
            bind_host=bind_host,
        )
        self._reserved_sockets = [sock]
        return {
            "host": bind_host,
            "port": port,
            "size": 1,
        }

    def reserve_port_block(
        self,
        preferred_base_port: int | None = None,
        num_ports: int = 1,
        avoid_ports: list[int] | None = None,
    ) -> dict[str, Any]:
        """Reserve a contiguous TCP port block on this actor's node."""
        self.release_port()
        bind_host = ray.util.get_node_ip_address()
        base_port, sockets = reserve_tcp_port_block(
            preferred_base_port,
            num_ports=num_ports,
            avoid_ports=set(avoid_ports or ()),
            bind_host=bind_host,
        )
        self._reserved_sockets = sockets
        return {
            "host": bind_host,
            "port": base_port,
            "size": num_ports,
        }

    def release_port(self) -> bool:
        """Release any port reservation currently held by this actor."""
        while self._reserved_sockets:
            self._reserved_sockets.pop().close()
        return True

    def get_node_info(self) -> dict[str, Any]:
        """Return the Ray node identity for this actor."""
        return {
            "host": ray.util.get_node_ip_address(),
        }

    def get_numbered_env_ports(
        self,
        *,
        prefix: str = "PORT",
        max_count: int = 30,
    ) -> list[int]:
        """Return numbered platform ports from this actor's node environment."""
        ports: list[int] = []
        for index in range(1, max_count + 1):
            env_name = f"{prefix}{index}"
            env_value = os.environ.get(env_name)
            if not env_value:
                continue
            try:
                port = int(env_value)
            except ValueError as exc:
                raise ValueError(f"invalid {env_name}: {env_value!r}") from exc
            if port <= 0 or port > 65535:
                raise ValueError(f"{env_name} out of range: {port}")
            ports.append(port)
        if len(set(ports)) != len(ports):
            raise ValueError(f"{prefix} environment ports must be unique: {ports}")
        return ports

    def get_available_numbered_env_port_info(
        self,
        *,
        prefix: str = "PORT",
        max_count: int = 30,
        avoid_ports: list[int] | None = None,
    ) -> dict[str, Any]:
        """Return available numbered platform ports from this actor's node."""
        env_ports = self.get_numbered_env_ports(prefix=prefix, max_count=max_count)
        avoid_port_set = set(avoid_ports or ())
        skipped_avoided_ports = [port for port in env_ports if port in avoid_port_set]
        candidate_ports = [port for port in env_ports if port not in avoid_port_set]
        available_ports = []
        skipped_unavailable_ports = []
        for port in candidate_ports:
            if is_tcp_port_available(port):
                available_ports.append(port)
            else:
                skipped_unavailable_ports.append(port)
        return {
            "host": ray.util.get_node_ip_address(),
            "env_ports": env_ports,
            "available_ports": available_ports,
            "skipped_avoided_ports": skipped_avoided_ports,
            "skipped_unavailable_ports": skipped_unavailable_ports,
        }


@ray.remote
class DraftActor:
    """Ray actor that hosts a draft engine for Ray/multi-node benchmark runs."""

    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        speculative_num_steps: int,
        rank_base: int,
        max_running_requests: int | None = None,
    ):
        """Pin GPUs and create the draft engine."""
        self.assigned_gpu_ids = pin_actor_to_assigned_gpus(tp_size)
        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            decoupled_spec_rank_base=rank_base,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
        )
        if max_running_requests is not None:
            engine_kwargs["max_running_requests"] = max_running_requests
        self.engine = sgl.Engine(**engine_kwargs)
        self.endpoint_infos = self.engine.get_decoupled_spec_endpoint_infos()

    def ready(self) -> dict[str, Any]:
        """Return actor metadata once the remote draft engine is ready."""
        return {
            "node_host": ray.util.get_node_ip_address(),
            "assigned_gpu_ids": self.assigned_gpu_ids,
            "endpoint_infos": self.endpoint_infos,
        }

    def configure_peers(self, connect_endpoints: list[str]) -> tuple[bool, str]:
        return self.engine.configure_decoupled_spec_peers(connect_endpoints)

    def shutdown(self) -> bool:
        """Shutdown the remote draft engine owned by this actor."""
        self.engine.shutdown()
        return True


def launch_draft_actors(
    args: argparse.Namespace,
    node_assignments: list[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Launch draft actors at planned nodes and collect their published endpoints."""
    if len(node_assignments) != args.num_draft_replicas:
        raise ValueError(
            f"node_assignments has {len(node_assignments)} entries, expected "
            f"{args.num_draft_replicas}"
        )

    actors = []
    actor_env_vars = get_decoupled_spec_actor_env_vars()
    for rank, node_id in enumerate(node_assignments):
        actor_options: dict[str, Any] = dict(
            num_gpus=args.draft_tp_size,
            num_cpus=1,
            max_concurrency=128,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        )
        if actor_env_vars:
            actor_options["runtime_env"] = {"env_vars": actor_env_vars}
        actor = DraftActor.options(**actor_options).remote(
            model_path=args.draft_model_path,
            tp_size=args.draft_tp_size,
            speculative_num_steps=args.num_speculative_steps,
            rank_base=rank,
            max_running_requests=getattr(args, "max_running_requests", None),
        )
        actors.append(actor)
    ready_infos = ray.get([actor.ready.remote() for actor in actors])
    endpoint_infos = []
    for ready_info in ready_infos:
        node_host = ready_info.get("node_host")
        for endpoint_info in ready_info.get("endpoint_infos", []):
            endpoint_info = dict(endpoint_info)
            if node_host is not None:
                endpoint_info["node_host"] = str(node_host)
            endpoint_infos.append(endpoint_info)
    return actors, endpoint_infos


def to_endpoint_infos(
    raw_infos: list[dict[str, Any]]
) -> list[DecoupledSpecEndpointInfo]:
    return [
        DecoupledSpecEndpointInfo(
            role=str(info["role"]),
            rank=int(info["rank"]),
            local_dp_rank=int(info.get("local_dp_rank", 0)),
            bind_endpoint=str(info["bind_endpoint"]),
            node_host=(
                str(info["node_host"]) if info.get("node_host") is not None else None
            ),
        )
        for info in raw_infos
    ]


def create_remote_decoupled_spec_topology(
    args: argparse.Namespace,
    verifier_pgs,
) -> DecoupledSpecTopology:
    """Launch Ray/multi-node draft actors and collect their published endpoints."""
    if not isinstance(verifier_pgs, list):
        verifier_pgs = [verifier_pgs]
    if not verifier_pgs:
        raise ValueError("at least one verifier placement group is required")
    node_assignments = plan_draft_placement(args)
    draft_actors, drafter_endpoint_infos = launch_draft_actors(
        args, node_assignments
    )
    return DecoupledSpecTopology(
        drafter_endpoint_infos=to_endpoint_infos(drafter_endpoint_infos),
        verifier_endpoint_infos=[],
        draft_actors=draft_actors,
    )


@ray.remote
class TargetActor:
    """Ray actor that hosts a decoupled verifier, decode, or MTP engine."""

    def __init__(
        self,
        *,
        mode: str,
        model_path: str,
        tp_size: int,
        dp_size: int = 1,
        enable_dp_attention: bool = False,
        ep_size: int | None = None,
        moe_a2a_backend: str | None = None,
        nnodes: int,
        node_rank: int,
        dist_init_addr: str | None,
        speculative_num_steps: int | None = None,
        speculative_adaptive: bool = False,
        speculative_adaptive_strategy: str = "throughput_aware",
        speculative_adaptive_config: str | None = None,
        decoupled_verify_throughput_profile_path: str | None = None,
        cuda_graph_bs_decode: list[int] | None = None,
        rank_base: int = 0,
        log_level: str | None = None,
        available_ports: list[int] | None = None,
        max_running_requests: int | None = None,
        dist_init_port_reservation_actor: Any | None = None,
    ):
        """Pin GPUs and initialize the target engine for one node rank."""
        self.mode = mode
        self.node_rank = node_rank
        self.assigned_gpu_ids = pin_actor_to_assigned_gpus(max(tp_size // nnodes, 1))
        if node_rank >= 1:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            nnodes=nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
        )
        if log_level is not None:
            engine_kwargs["log_level"] = log_level
        if cuda_graph_bs_decode is not None:
            engine_kwargs["cuda_graph_bs_decode"] = cuda_graph_bs_decode
        if max_running_requests is not None:
            engine_kwargs["max_running_requests"] = max_running_requests
        if ep_size is not None:
            engine_kwargs["ep_size"] = ep_size
        if moe_a2a_backend is not None:
            engine_kwargs["moe_a2a_backend"] = moe_a2a_backend
        if enable_dp_attention:
            engine_kwargs["enable_dp_attention"] = True
            if available_ports is not None:
                engine_kwargs["available_ports"] = available_ports
        if mode == "decoupled_spec":
            engine_kwargs.update(
                speculative_algorithm="DECOUPLED_VERIFY",
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_steps + 1,
                decoupled_spec_rank_base=rank_base,
                disable_radix_cache=True,
            )
            if speculative_adaptive:
                engine_kwargs["speculative_adaptive"] = True
                engine_kwargs["speculative_adaptive_strategy"] = (
                    speculative_adaptive_strategy
                )
                if speculative_adaptive_config is not None:
                    engine_kwargs["speculative_adaptive_config"] = (
                        speculative_adaptive_config
                    )
                if decoupled_verify_throughput_profile_path is not None:
                    engine_kwargs["decoupled_verify_throughput_profile_path"] = (
                        decoupled_verify_throughput_profile_path
                    )
        elif mode == "decode":
            engine_kwargs["disable_overlap_schedule"] = True
        elif mode == "mtp":
            engine_kwargs.update(
                speculative_algorithm="EAGLE",
                speculative_num_steps=speculative_num_steps,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=speculative_num_steps + 1,
                disable_radix_cache=True,
                disable_overlap_schedule=True,
                mamba_scheduler_strategy="no_buffer",
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if dist_init_port_reservation_actor is not None:
            ray.get(dist_init_port_reservation_actor.release_port.remote())
        self.engine = sgl.Engine(**engine_kwargs)
        self.endpoint_infos = (
            self.engine.get_decoupled_spec_endpoint_infos()
            if mode == "decoupled_spec"
            else []
        )

    def ready(self) -> dict[str, Any]:
        """Return actor metadata once the target engine has initialized."""
        return {
            "mode": self.mode,
            "node_rank": self.node_rank,
            "assigned_gpu_ids": self.assigned_gpu_ids,
            "endpoint_infos": self.endpoint_infos,
        }

    def configure_peers(self, connect_endpoints: list[str]) -> tuple[bool, str]:
        return self.engine.configure_decoupled_spec_peers(connect_endpoints)

    def generate_batch(
        self,
        prompt_input_ids: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run generation on rank 0 and return raw outputs."""
        if self.node_rank != 0:
            raise RuntimeError("generate_batch must be called on node rank 0")

        outputs = self.engine.generate(
            input_ids=prompt_input_ids,
            sampling_params=sampling_params,
        )
        if not isinstance(outputs, list):
            outputs = [outputs]
        return {"outputs": outputs}

    def shutdown(self) -> bool:
        """Shutdown the target engine owned by this actor."""
        self.engine.shutdown()
        return True

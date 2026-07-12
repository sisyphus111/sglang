#!/usr/bin/env python3
"""
Run decoupled speculative decoding for either an input prompt or a prompt dataset.

By default, this compares decoupled speculative decoding against normal decode.
Use `--baseline none` to run decoupled speculation only, `--baseline mtp` to
compare against SGLang's builtin colocated MTP baseline, or `--baseline all` to
run decode and MTP baselines in sequence. Use `--show-responses` to print full
response text. When `--output-dir` is set, JSON output records full prompt and
response text.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from typing import Any

logger = logging.getLogger(__name__)

import common

DEFAULT_RAY_NAMESPACE = "dspec"
DEFAULT_PROMPT_COLUMN_CANDIDATES = common.DEFAULT_PROMPT_COLUMN_CANDIDATES
DPA_ENV_FIXED_PORT_COUNT = 6
_RUNTIME_IMPORTS_READY = False


def _ensure_runtime_imports() -> None:
    """Import Ray and decoupled-spec helpers after argparse handles --help."""
    global _RUNTIME_IMPORTS_READY
    global create_remote_decoupled_spec_topology
    global PlacementGroupSchedulingStrategy
    global PortActor
    global TargetActor
    global get_decoupled_spec_actor_env_vars
    global placement_group
    global ray
    global remove_placement_group

    if _RUNTIME_IMPORTS_READY:
        return

    import ray as ray_module
    from ray.util.placement_group import placement_group as ray_placement_group
    from ray.util.placement_group import (
        remove_placement_group as ray_remove_placement_group,
    )
    from ray.util.scheduling_strategies import (
        PlacementGroupSchedulingStrategy as RayPlacementGroupSchedulingStrategy,
    )

    ray = ray_module
    placement_group = ray_placement_group
    remove_placement_group = ray_remove_placement_group
    PlacementGroupSchedulingStrategy = RayPlacementGroupSchedulingStrategy
    create_remote_decoupled_spec_topology = common.create_remote_decoupled_spec_topology
    PortActor = common.PortActor
    TargetActor = common.TargetActor
    get_decoupled_spec_actor_env_vars = common.get_decoupled_spec_actor_env_vars
    _RUNTIME_IMPORTS_READY = True


PromptSample = common.PromptSample
ModeMetrics = common.ModeMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation on one prompt or a parquet prompt batch, "
            "optionally comparing against a decode or MTP baseline."
        )
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Single prompt to generate from. When --batch-size is greater than 1, "
            "the prompt is repeated to fill the batch. Mutually exclusive with "
            "--dataset-path."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        "--parquet-path",
        dest="dataset_path",
        default=None,
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help=(
            "Prompt column in the parquet file. If omitted, common names are "
            f"searched in order: {DEFAULT_PROMPT_COLUMN_CANDIDATES}."
        ),
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "codeforces_raw", "dapo_math_17k"],
        default="auto",
        help=(
            "How to interpret the parquet rows. "
            "'auto' reads one prompt-like column. "
            "'codeforces_raw' builds a prompt from Codeforces problem fields and, "
            "when enabled, renders it through the model tokenizer's chat template. "
            "'dapo_math_17k' reads the DAPO-Math-17k structured prompt messages "
            "and renders them through the target model chat template."
        ),
    )
    parser.add_argument(
        "--code-language",
        choices=["python", "py", "cpp", "c++"],
        default="python",
        help=(
            "Target language used when --dataset-format=codeforces_raw. "
            "Ignored for normal prompt-column datasets."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        "--bs",
        dest="batch_size",
        type=int,
        default=1,
        help=(
            "Number of valid prompts to run in one generate call. When using "
            "multiple verifier replicas, prompts are distributed as evenly as "
            "possible across verifier replicas."
        ),
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template for chat-style prompt objects.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking-style generation when building chat prompts for "
            "models such as Qwen3/Qwen3.5. Disabled by default."
        ),
    )
    parser.add_argument(
        "--context-length",
        "--max-new-tokens",
        dest="context_length",
        type=int,
        required=True,
        help="Generation length. This is passed as max_new_tokens.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Optional prompt token upper bound. Prompts over this limit are skipped.",
    )
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Target/verifier model path.",
    )
    parser.add_argument(
        "--draft-model-path",
        required=True,
        help="Draft model path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path used for prompt length filtering. Defaults to target model.",
    )
    parser.add_argument(
        "--target-tp-size",
        type=int,
        required=True,
        help=(
            "Target/verifier TP size. When --target-enable-dp-attention is set, "
            "this is the attention TP size per DP lane; the SGLang engine "
            "tp_size is target_tp_size * target_dp_size."
        ),
    )
    parser.add_argument(
        "--target-dp-size",
        type=int,
        default=1,
        help="Target/verifier DP size.",
    )
    parser.add_argument(
        "--target-enable-dp-attention",
        "--enable-dp-attention",
        dest="target_enable_dp_attention",
        action="store_true",
        help=(
            "Enable DP attention for the target/verifier engine. With this set, "
            "--target-tp-size is interpreted per DP lane."
        ),
    )
    parser.add_argument(
        "--target-use-env-ports",
        action="store_true",
        help=(
            "Use PORT1..PORT30 from each verifier replica's rank-0 node as "
            "the target engine's available port pool. If unset, the legacy "
            "dist-init-derived ports are used."
        ),
    )
    parser.add_argument(
        "--target-ep-size",
        type=int,
        default=None,
        help="Expert parallel size for the target/verifier engine.",
    )
    parser.add_argument(
        "--target-moe-a2a-backend",
        default=None,
        help="MoE A2A backend for the target/verifier engine, e.g. deepep.",
    )
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument(
        "--max-running-requests",
        "--max-running-reqs",
        dest="max_running_requests",
        type=int,
        default=None,
        help=(
            "Override SGLang max_running_requests for the decoupled verifier, "
            "drafters, and decode/MTP baseline engines."
        ),
    )
    parser.add_argument(
        "--num-speculative-steps",
        type=int,
        default=3,
        help=(
            "Static speculative steps. With decoupled verifier "
            "--speculative-adaptive-strategy=throughput_aware, this is the "
            "maximum candidate verify step."
        ),
    )
    parser.add_argument(
        "--speculative-adaptive",
        action="store_true",
        help=(
            "Enable adaptive decoupled verifier dynamic verify length on the "
            "target engine."
        ),
    )
    parser.add_argument(
        "--speculative-adaptive-strategy",
        choices=["ema", "throughput_aware"],
        default="throughput_aware",
        help=(
            "Adaptive decoupled verifier strategy. throughput_aware profiles "
            "verifier-only costs at startup; ema requires "
            "--speculative-adaptive-config."
        ),
    )
    parser.add_argument(
        "--speculative-adaptive-config",
        default=None,
        help=(
            "Optional adaptive speculative config JSON path for the target "
            "engine when --speculative-adaptive is enabled."
        ),
    )
    parser.add_argument(
        "--decoupled-verify-throughput-profile-path",
        default=None,
        help=(
            "Optional JSON cost-table cache file for decoupled verifier "
            "throughput-aware startup profiling."
        ),
    )
    parser.add_argument(
        "--cuda-graph-bs-decode",
        default=None,
        help=(
            "Comma-separated decode CUDA Graph batch sizes for target/verifier "
            "and baseline target engines, e.g. 32,64,128."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Set sampling_params.ignore_eos=True for both decoupled speculative "
            "decoding and normal decoding. Disabled by default."
        ),
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address. Use 'auto' for an existing cluster or local fallback on nnodes=1.",
    )
    parser.add_argument("--ray-namespace", default=DEFAULT_RAY_NAMESPACE)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument(
        "--n-gpu-per-node",
        type=int,
        default=None,
        help=(
            "GPU count available on each Ray node. Used with --nnodes to bound "
            "the total verifier and drafter GPU budgets."
        ),
    )
    parser.add_argument(
        "--verify-ngpus",
        dest="verify_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for verifier replicas. If omitted, all GPUs "
            "not reserved by --draft-ngpus are used for verifier replicas."
        ),
    )
    parser.add_argument(
        "--draft-ngpus",
        dest="draft_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for all drafter replicas. The number of "
            "drafters is derived as draft_ngpus / draft_tp_size."
        ),
    )
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help=(
            "Optional SGLang distributed init address override. If omitted for "
            "multi-node runs, the script uses each verifier placement group's "
            "rank-0 host."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=None,
        help=(
            "Base port for this run. With V verifier replicas, spec dist-init "
            "uses the first V ports and baseline dist-init ports follow. "
            "Decoupled result/control endpoints are bound dynamically by SGLang."
        ),
    )
    parser.add_argument(
        "--reserved-ports",
        default=None,
        help=(
            "Comma- or whitespace-separated list of pre-reserved ports, e.g. "
            "Merlin PORT1..PORT15. Ports do not need to be contiguous. They are "
            "consumed in order: spec dist-init ports, optional baseline dist-init "
            "ports. Decoupled result/control endpoints are bound dynamically by "
            "SGLang. This cannot be combined with --dist-init-addr or "
            "--dist-init-port. With --target-enable-dp-attention, each dist-init "
            "allocation consumes a contiguous 6-port block in the legacy "
            "dist-init-derived port mode; with --target-use-env-ports, only "
            "the dist-init ports "
            "are consumed here and the fixed DP-attention ports come from "
            "PORT1..PORT30 on the verifier rank-0 node."
        ),
    )
    parser.add_argument(
        "--num-draft-replicas",
        dest="num_draft_replicas",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory to write per-mode CSV/JSON outputs. "
            "A baseline comparison run writes decoupled-spec.csv/json plus "
            "<baseline>.csv/json."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=common.BASELINE_CHOICES,
        default="decode",
        help=(
            "Baseline to run after decoupled speculation. Use 'all' to run "
            "decode and MTP. 'mtp' uses SGLang's builtin colocated, serial "
            "draft-verify MTP/EAGLE path."
        ),
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help=(
            "Print full response text in the terminal. When --output-dir is set, "
            "full prompt and response text is always included in per-mode JSON."
        ),
    )
    return parser.parse_args()


def load_prompt_samples(
    args: argparse.Namespace,
) -> tuple[str, list[PromptSample], int]:
    return common.load_prompt_samples(args)


def _parse_host_port(addr: str) -> tuple[str, int | None]:
    if addr.startswith("["):
        end = addr.find("]")
        if end != -1:
            host = addr[1:end]
            rest = addr[end + 1 :]
            if rest.startswith(":") and len(rest) > 1:
                return host, int(rest[1:])
        return addr, None
    if addr.count(":") == 1:
        host, raw_port = addr.rsplit(":", 1)
        if raw_port:
            return host, int(raw_port)
    return addr, None


def _normalize_layout_host(host: str) -> str:
    """Normalize host strings used only for layout display/grouping."""
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _host_from_endpoint(endpoint: str) -> str:
    """Extract the host from a tcp://host:port endpoint."""
    endpoint = endpoint.removeprefix("tcp://")
    if endpoint.startswith("["):
        end = endpoint.find("]")
        if end != -1:
            return _normalize_layout_host(endpoint[1:end])
    host, _ = _parse_host_port(endpoint)
    return _normalize_layout_host(host)


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _parse_reserved_ports(raw_ports: str | None) -> list[int]:
    return common.parse_reserved_ports(raw_ports)


def _target_uses_env_available_ports(args: argparse.Namespace) -> bool:
    return common.target_uses_env_available_ports(args)


def _target_dp_attention_uses_dist_init_derived_ports(
    args: argparse.Namespace,
) -> bool:
    return common.target_dp_attention_uses_dist_init_derived_ports(args)


def _dist_init_port_stride(args: argparse.Namespace) -> int:
    return common.dist_init_port_stride(args)


def _reserved_port_block_bases(
    reserved_ports: list[int],
    *,
    num_blocks: int,
    block_size: int,
) -> list[int]:
    return common.reserved_port_block_bases(
        reserved_ports, num_blocks=num_blocks, block_size=block_size
    )


def _split_reserved_ports(
    args: argparse.Namespace,
) -> tuple[
    list[int],
    dict[str, list[int]],
    list[int] | None,
]:
    reserved_ports = _parse_reserved_ports(args.reserved_ports)
    if not reserved_ports:
        return [], {}, None

    if args.dist_init_addr is not None or args.dist_init_port is not None:
        raise ValueError(
            "--reserved-ports cannot be combined with --dist-init-addr or "
            "--dist-init-port"
        )

    num_verifiers = args.num_verifier_replicas
    baseline_modes = common.resolve_baseline_modes(args.baseline)
    port_stride = _dist_init_port_stride(args)
    required_blocks = num_verifiers + len(baseline_modes) * num_verifiers
    required_ports = required_blocks * port_stride
    if len(reserved_ports) < required_ports:
        port_usage = f"{num_verifiers} spec dist-init"
        if baseline_modes:
            port_usage += (
                f" and {len(baseline_modes) * num_verifiers} baseline dist-init"
            )
        if port_stride > 1:
            port_usage += f" blocks of {port_stride} contiguous ports"
        raise ValueError(
            f"--reserved-ports provides {len(reserved_ports)} ports, but this "
            f"run needs {required_ports}: {port_usage}"
        )

    if port_stride > 1:
        block_bases = _reserved_port_block_bases(
            reserved_ports,
            num_blocks=required_blocks,
            block_size=port_stride,
        )
        cursor = 0
        spec_ports = block_bases[cursor : cursor + num_verifiers]
        cursor += num_verifiers
        baseline_ports_by_mode = {}
        for baseline_mode in baseline_modes:
            baseline_ports_by_mode[baseline_mode] = block_bases[
                cursor : cursor + num_verifiers
            ]
            cursor += num_verifiers
        return (
            spec_ports,
            baseline_ports_by_mode,
            reserved_ports,
        )

    cursor = 0
    spec_ports = reserved_ports[cursor : cursor + num_verifiers]
    cursor += num_verifiers
    baseline_ports_by_mode = {}
    for baseline_mode in baseline_modes:
        baseline_ports_by_mode[baseline_mode] = reserved_ports[
            cursor : cursor + num_verifiers
        ]
        cursor += num_verifiers
    return (
        spec_ports,
        baseline_ports_by_mode,
        reserved_ports,
    )


def derive_dist_init_addr(
    args: argparse.Namespace,
    *,
    port_offset: int = 0,
    preferred_port: int | None = None,
) -> str | None:
    port_stride = _dist_init_port_stride(args)
    if args.nnodes == 1 and args.dist_init_addr is None:
        if preferred_port is not None:
            return f"127.0.0.1:{preferred_port}"
        if args.dist_init_port is not None:
            return f"127.0.0.1:{args.dist_init_port + port_offset * port_stride}"
        return f"127.0.0.1:{_pick_free_local_port()}"

    if args.dist_init_addr is None:
        raise ValueError("dist-init-addr is required when nnodes > 1")

    host, parsed_port = _parse_host_port(args.dist_init_addr)
    base_port = args.dist_init_port if args.dist_init_port is not None else parsed_port
    if base_port is None:
        raise ValueError(
            "dist-init-addr must include a port or dist-init-port must be set"
        )

    return common.format_host_port(host, base_port + port_offset * port_stride)


def reserve_dist_init_addr_from_pg(
    args: argparse.Namespace,
    pg,
    *,
    port_offset: int = 0,
    preferred_port: int | None = None,
) -> tuple[str | None, Any | None]:
    if args.dist_init_addr is not None:
        return (
            derive_dist_init_addr(
                args, port_offset=port_offset, preferred_port=preferred_port
            ),
            None,
        )

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    actor = PortActor.options(
        num_cpus=0,
        scheduling_strategy=scheduling_strategy,
    ).remote()
    port_stride = _dist_init_port_stride(args)
    try:
        if preferred_port is not None:
            selected_port = preferred_port
        elif args.dist_init_port is not None:
            selected_port = args.dist_init_port + port_offset * port_stride
        else:
            selected_port = None
        if port_stride > 1:
            reservation = ray.get(
                actor.reserve_port_block.remote(selected_port, port_stride)
            )
        else:
            reservation = ray.get(actor.reserve_port.remote(selected_port))
        host = reservation["host"]
        port = int(reservation["port"])
    except Exception:
        ray.kill(actor, no_restart=True)
        raise

    return common.format_host_port(host, port), actor


def shutdown_port_reservation_actors(actors: list[Any]) -> None:
    for actor in actors:
        try:
            ray.get(actor.release_port.remote(), timeout=10)
        except Exception:
            pass
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass


def init_ray(address: str, namespace: str, nnodes: int) -> None:
    init_kwargs = dict(
        address=address,
        namespace=namespace,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_level=logging.ERROR,
    )
    try:
        ray.init(**init_kwargs)
    except Exception:
        if address != "auto" or nnodes != 1:
            raise
        ray.init(
            namespace=namespace,
            ignore_reinit_error=True,
            log_to_driver=True,
            logging_level=logging.ERROR,
        )


def get_target_engine_tp_size(args: argparse.Namespace) -> int:
    return common.get_target_engine_tp_size(args)


def derive_target_layout(args: argparse.Namespace) -> tuple[int, int]:
    target_engine_tp_size = get_target_engine_tp_size(args)
    for candidate_nnodes in range(1, args.nnodes + 1):
        if target_engine_tp_size % candidate_nnodes != 0:
            continue
        target_gpus_per_node = target_engine_tp_size // candidate_nnodes
        if target_gpus_per_node <= args.n_gpu_per_node:
            return candidate_nnodes, target_gpus_per_node

    raise ValueError(
        f"target verifier world size ({target_engine_tp_size}) cannot be packed "
        f"evenly across up to {args.nnodes} nodes with "
        f"{args.n_gpu_per_node} GPUs per node"
    )


def validate_resources(args: argparse.Namespace) -> tuple[int, int]:
    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.target_tp_size <= 0:
        raise ValueError("target-tp-size must be positive")
    if args.target_dp_size <= 0:
        raise ValueError("target-dp-size must be positive")
    if args.target_enable_dp_attention and args.target_dp_size <= 1:
        raise ValueError(
            "target-dp-size must be greater than 1 when DP attention is enabled"
        )
    if args.target_dp_size > 1 and not args.target_enable_dp_attention:
        raise ValueError(
            "target-dp-size > 1 is only supported with --target-enable-dp-attention "
            "in this multi-node example"
        )
    if args.target_ep_size is not None and args.target_ep_size <= 0:
        raise ValueError("target-ep-size must be positive when set")
    if args.draft_tp_size <= 0:
        raise ValueError("draft-tp-size must be positive")
    if args.max_running_requests is not None and args.max_running_requests <= 0:
        raise ValueError("max-running-requests must be positive when set")
    if args.cuda_graph_bs_decode is not None:
        args.cuda_graph_bs_decode = [
            int(item)
            for item in str(args.cuda_graph_bs_decode).replace(",", " ").split()
        ]
        if not args.cuda_graph_bs_decode or any(
            bs <= 0 for bs in args.cuda_graph_bs_decode
        ):
            raise ValueError("cuda-graph-bs-decode must contain positive integers")
    if (
        args.speculative_adaptive
        and args.speculative_adaptive_strategy == "ema"
        and args.speculative_adaptive_config is None
    ):
        raise ValueError(
            "--speculative-adaptive-config is required when "
            "--speculative-adaptive-strategy=ema is used with "
            "--speculative-adaptive"
        )
    if args.verify_ngpus is not None and args.verify_ngpus <= 0:
        raise ValueError("verify-ngpus must be positive when set")
    if args.draft_ngpus is not None and args.draft_ngpus <= 0:
        raise ValueError("draft-ngpus must be positive when set")
    if args.num_draft_replicas is not None and args.num_draft_replicas <= 0:
        raise ValueError("num-draft-replicas must be positive when set")
    if args.draft_ngpus is None:
        args.draft_ngpus = args.draft_tp_size * (args.num_draft_replicas or 1)
    if args.draft_ngpus % args.draft_tp_size != 0:
        raise ValueError(
            f"draft-ngpus ({args.draft_ngpus}) must be divisible by "
            f"draft-tp-size ({args.draft_tp_size})"
        )
    derived_num_draft_replicas = args.draft_ngpus // args.draft_tp_size
    if (
        args.num_draft_replicas is not None
        and args.num_draft_replicas != derived_num_draft_replicas
    ):
        raise ValueError(
            "num-draft-replicas must match draft-ngpus / draft-tp-size "
            f"({derived_num_draft_replicas}) when --draft-ngpus is set"
        )
    args.num_draft_replicas = derived_num_draft_replicas
    target_engine_tp_size = get_target_engine_tp_size(args)
    if (
        args.verify_ngpus is not None
        and args.verify_ngpus % target_engine_tp_size != 0
    ):
        raise ValueError(
            f"verify-ngpus ({args.verify_ngpus}) must be divisible by "
            f"target verifier world size ({target_engine_tp_size})"
        )

    if args.n_gpu_per_node is None:
        if args.nnodes != 1:
            raise ValueError("n-gpu-per-node is required when nnodes > 1")
        args.n_gpu_per_node = (
            args.verify_ngpus or target_engine_tp_size
        ) + args.draft_ngpus
    if args.n_gpu_per_node <= 0:
        raise ValueError("n-gpu-per-node must be positive")

    total_cluster_gpus = args.n_gpu_per_node * args.nnodes
    if args.verify_ngpus is None:
        args.verify_ngpus = total_cluster_gpus - args.draft_ngpus
    if args.draft_ngpus + args.verify_ngpus > total_cluster_gpus:
        raise ValueError(
            f"verify-ngpus + draft-ngpus ({args.verify_ngpus} + "
            f"{args.draft_ngpus}) exceeds nnodes*n-gpu-per-node "
            f"({total_cluster_gpus})"
        )
    if args.verify_ngpus <= 0:
        raise ValueError(
            f"draft-ngpus ({args.draft_ngpus}) must leave GPUs for at least "
            "one verifier replica"
        )
    if args.verify_ngpus % target_engine_tp_size != 0:
        raise ValueError(
            f"verify-ngpus ({args.verify_ngpus}) must be divisible by "
            f"target verifier world size ({target_engine_tp_size})"
        )
    args.num_verifier_replicas = args.verify_ngpus // target_engine_tp_size

    target_nnodes, target_gpus_per_node = derive_target_layout(args)

    if args.draft_tp_size > args.n_gpu_per_node:
        raise ValueError(
            f"each draft actor needs {args.draft_tp_size} GPUs on one node, "
            f"but n-gpu-per-node is only {args.n_gpu_per_node}"
        )

    ray_gpus = int(ray.cluster_resources().get("GPU", 0))
    if ray_gpus and total_cluster_gpus > ray_gpus:
        raise ValueError(
            f"Ray cluster reports {ray_gpus} GPUs, but this run requires "
            f"{total_cluster_gpus}"
        )

    alive_target_nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float(node.get("Resources", {}).get("GPU", 0)) >= target_gpus_per_node
    ]
    if len(alive_target_nodes) < target_nnodes:
        raise ValueError(
            f"Ray cluster has {len(alive_target_nodes)} alive GPU nodes with at "
            f"least {target_gpus_per_node} GPUs, but target needs {target_nnodes} nodes"
        )

    return target_nnodes, target_gpus_per_node


def create_target_placement_group(target_nnodes: int, target_gpus_per_node: int):
    bundles = [{"CPU": 1, "GPU": target_gpus_per_node} for _ in range(target_nnodes)]
    strategy = "PACK" if target_nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg


def _get_node_resource_key(node: dict[str, Any]) -> str | None:
    for resource_name in node.get("Resources", {}):
        if resource_name.startswith("node:"):
            return resource_name
    return None


def _plan_stable_target_node_groups(
    num_replicas: int,
    target_nnodes: int,
    target_gpus_per_node: int,
) -> list[list[dict[str, Any]]] | None:
    candidate_nodes = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        if float(node.get("Resources", {}).get("GPU", 0)) < target_gpus_per_node:
            continue
        node_resource_key = _get_node_resource_key(node)
        if node_resource_key is None:
            return None
        candidate_nodes.append(
            {
                "node_ip": node["NodeManagerAddress"],
                "node_resource_key": node_resource_key,
                "remaining_bundles": int(
                    float(node.get("Resources", {}).get("GPU", 0))
                    // target_gpus_per_node
                ),
            }
        )

    candidate_nodes.sort(key=lambda item: item["node_ip"])

    node_groups = []
    for _ in range(num_replicas):
        node_group = []
        for node in candidate_nodes:
            if node["remaining_bundles"] <= 0:
                continue
            node_group.append(node)
            node["remaining_bundles"] -= 1
            if len(node_group) == target_nnodes:
                break
        if len(node_group) != target_nnodes:
            return None
        node_groups.append(node_group)

    return node_groups


def create_target_placement_group_on_nodes(
    target_nnodes: int,
    target_gpus_per_node: int,
    node_group: list[dict[str, Any]] | None,
):
    if node_group is None:
        return create_target_placement_group(target_nnodes, target_gpus_per_node)

    bundles = []
    for node in node_group:
        bundles.append(
            {
                "CPU": 1,
                "GPU": target_gpus_per_node,
                node["node_resource_key"]: 0.001,
            }
        )
    strategy = "PACK" if target_nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg


def create_target_placement_groups(
    num_replicas: int,
    target_nnodes: int,
    target_gpus_per_node: int,
):
    node_groups = _plan_stable_target_node_groups(
        num_replicas,
        target_nnodes,
        target_gpus_per_node,
    )
    return [
        create_target_placement_group_on_nodes(
            target_nnodes,
            target_gpus_per_node,
            node_groups[replica_index] if node_groups is not None else None,
        )
        for replica_index in range(num_replicas)
    ]


def _get_pg_bundle_hosts(pg, num_bundles: int) -> list[str]:
    hosts = []
    for bundle_index in range(num_bundles):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
        )
        actor = PortActor.options(
            num_cpus=0,
            scheduling_strategy=scheduling_strategy,
        ).remote()
        try:
            info = ray.get(actor.get_node_info.remote())
            hosts.append(info["host"])
        finally:
            ray.kill(actor, no_restart=True)
    return hosts


def _ports_from_dist_init_addrs(addrs: list[str | None]) -> set[int]:
    ports: set[int] = set()
    for addr in addrs:
        if addr is None:
            continue
        _, port = _parse_host_port(addr)
        if port is not None:
            ports.add(port)
    return ports


def _known_dist_init_ports(
    args: argparse.Namespace,
    *,
    spec_reserved_ports: list[int],
    baseline_reserved_ports: dict[str, list[int]],
) -> set[int]:
    ports = set(spec_reserved_ports)
    for mode_ports in baseline_reserved_ports.values():
        ports.update(mode_ports)
    if ports:
        return ports

    host, parsed_port = (
        _parse_host_port(args.dist_init_addr)
        if args.dist_init_addr is not None
        else ("", None)
    )
    del host
    base_port = args.dist_init_port if args.dist_init_port is not None else parsed_port
    if base_port is None:
        return ports

    baseline_modes = common.resolve_baseline_modes(args.baseline)
    num_slots = args.num_verifier_replicas * (1 + len(baseline_modes))
    port_stride = _dist_init_port_stride(args)
    return {base_port + slot * port_stride for slot in range(num_slots)}


def _get_target_available_ports_from_pg(
    args: argparse.Namespace,
    pg,
    *,
    engine_slot: int,
    avoid_ports: set[int],
) -> list[int] | None:
    if not (
        args.target_enable_dp_attention and _target_uses_env_available_ports(args)
    ):
        return None

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    actor = PortActor.options(
        num_cpus=0,
        scheduling_strategy=scheduling_strategy,
    ).remote()
    try:
        port_info = ray.get(
            actor.get_available_numbered_env_port_info.remote(
                prefix="PORT",
                max_count=30,
                avoid_ports=sorted(avoid_ports),
            )
        )
    finally:
        ray.kill(actor, no_restart=True)

    env_ports = port_info["env_ports"]
    usable_ports = port_info["available_ports"]
    skipped_unavailable_ports = port_info["skipped_unavailable_ports"]
    skipped_avoided_ports = port_info["skipped_avoided_ports"]
    if skipped_unavailable_ports or skipped_avoided_ports:
        print(
            "target_env_ports_skipped: "
            f"node={port_info['host']} "
            f"unavailable={skipped_unavailable_ports} "
            f"reserved={skipped_avoided_ports}",
            flush=True,
        )
    start = engine_slot * DPA_ENV_FIXED_PORT_COUNT
    selected_ports = usable_ports[start : start + DPA_ENV_FIXED_PORT_COUNT]
    if len(selected_ports) != DPA_ENV_FIXED_PORT_COUNT:
        raise ValueError(
            "--target-use-env-ports needs "
            f"{DPA_ENV_FIXED_PORT_COUNT} PORT env ports for engine slot "
            f"{engine_slot} on rank-0 node {port_info['host']}; "
            f"env_ports={env_ports} available_ports={usable_ports} "
            f"unavailable={skipped_unavailable_ports} "
            f"reserved={skipped_avoided_ports}"
        )
    return selected_ports


def print_decoupled_spec_layout(
    *,
    args: argparse.Namespace,
    target_nnodes: int,
    target_gpus_per_node: int,
    verifier_pgs: list[Any],
    topology: Any,
) -> None:
    node_hosts = sorted(
        {
            _normalize_layout_host(node["NodeManagerAddress"])
            for node in ray.nodes()
            if node.get("Alive") and node.get("NodeManagerAddress")
        }
    )
    node_layout: dict[str, list[str]] = {host: [] for host in node_hosts}
    target_engine_tp_size = get_target_engine_tp_size(args)

    for verifier_rank, pg in enumerate(verifier_pgs):
        bundle_hosts = _get_pg_bundle_hosts(pg, target_nnodes)
        for bundle_index, raw_host in enumerate(bundle_hosts):
            host = _normalize_layout_host(raw_host)
            node_layout.setdefault(host, [])
            label = f"verifier{verifier_rank}(tp={args.target_tp_size}"
            if args.target_enable_dp_attention:
                label += (
                    f", dp={args.target_dp_size}, "
                    f"engine_tp={target_engine_tp_size}"
                )
            if target_nnodes > 1:
                label += f", bundle={bundle_index}"
            label += f", gpus={target_gpus_per_node})"
            node_layout[host].append(label)

    for drafter_rank, endpoint_info in enumerate(topology.drafter_endpoint_infos):
        host = endpoint_info.node_host or _host_from_endpoint(
            endpoint_info.bind_endpoint
        )
        host = _normalize_layout_host(host)
        node_layout.setdefault(host, [])
        node_layout[host].append(f"drafter{drafter_rank}(tp={args.draft_tp_size})")

    print("=== decoupled_spec_layout ===")
    print(f"nnodes: {args.nnodes}")
    print(
        f"nverifier={args.num_verifier_replicas}, "
        f"tp_size={args.target_tp_size}, "
        f"dp_size={args.target_dp_size}, "
        f"enable_dp_attention={args.target_enable_dp_attention}, "
        f"engine_tp_size={target_engine_tp_size}, "
        f"target_nnodes={target_nnodes}, "
        f"target_gpus_per_node={target_gpus_per_node}"
    )
    print(f"ndrafter={args.num_draft_replicas}, tp_size={args.draft_tp_size}")
    for node_index, host in enumerate(sorted(node_layout)):
        items = ", ".join(node_layout[host]) if node_layout[host] else "idle"
        print(f"node{node_index} ({host}): {items}")


def launch_target_actors(
    *,
    args: argparse.Namespace,
    mode: str,
    dist_init_addr: str | None,
    dist_init_port_reservation_actor: Any | None,
    target_nnodes: int,
    target_gpus_per_node: int,
    pg,
    rank_base: int = 0,
    available_ports: list[int] | None = None,
) -> list[Any]:
    actor_env_vars = get_decoupled_spec_actor_env_vars()
    actors = []
    target_engine_tp_size = get_target_engine_tp_size(args)
    for node_rank in range(target_nnodes):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=node_rank,
        )
        actor_options: dict[str, Any] = dict(
            num_gpus=target_gpus_per_node,
            num_cpus=1,
            scheduling_strategy=scheduling_strategy,
        )
        if actor_env_vars:
            actor_options["runtime_env"] = {"env_vars": actor_env_vars}
        actor = TargetActor.options(**actor_options).remote(
            mode=mode,
            model_path=args.target_model_path,
            tp_size=target_engine_tp_size,
            dp_size=args.target_dp_size,
            enable_dp_attention=args.target_enable_dp_attention,
            ep_size=args.target_ep_size,
            moe_a2a_backend=args.target_moe_a2a_backend,
            nnodes=target_nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            speculative_num_steps=args.num_speculative_steps,
            speculative_adaptive=args.speculative_adaptive,
            speculative_adaptive_strategy=args.speculative_adaptive_strategy,
            speculative_adaptive_config=args.speculative_adaptive_config,
            decoupled_verify_throughput_profile_path=(
                args.decoupled_verify_throughput_profile_path
            ),
            cuda_graph_bs_decode=args.cuda_graph_bs_decode,
            rank_base=rank_base,
            log_level="info",
            available_ports=available_ports,
            max_running_requests=args.max_running_requests,
            dist_init_port_reservation_actor=(
                dist_init_port_reservation_actor if node_rank == 0 else None
            ),
        )
        actors.append(actor)

    ray.get([actor.ready.remote() for actor in actors])
    return actors


def shutdown_actors(actors: list[Any]) -> None:
    if not actors:
        return
    try:
        ray.get([actor.shutdown.remote() for actor in actors], timeout=60)
    except Exception as exc:
        logger.warning("actor shutdown failed: %s", exc)
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

collect_mode_metrics = common.collect_mode_metrics


def _split_indices(num_items: int, num_shards: int) -> list[list[int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    base_shard_size, remainder = divmod(num_items, num_shards)
    shards: list[list[int]] = []
    start = 0
    for shard_index in range(num_shards):
        shard_size = base_shard_size + (1 if shard_index < remainder else 0)
        end = start + shard_size
        shards.append(list(range(start, end)))
        start = end
    return shards


def _endpoint_field(info: Any, field: str) -> Any:
    if isinstance(info, dict):
        return info[field]
    return getattr(info, field)


def _sorted_bind_endpoints(endpoint_infos: list[Any], *, role: str) -> list[str]:
    role_infos = [
        info for info in endpoint_infos if _endpoint_field(info, "role") == role
    ]
    if not role_infos:
        raise RuntimeError(f"no decoupled-spec {role} endpoints were published")
    role_infos.sort(key=lambda info: int(_endpoint_field(info, "rank")))
    ranks = [int(_endpoint_field(info, "rank")) for info in role_infos]
    expected = list(range(len(ranks)))
    if ranks != expected:
        raise RuntimeError(
            f"decoupled-spec {role} ranks must be zero-based and contiguous: "
            f"got {ranks}"
        )
    return [str(_endpoint_field(info, "bind_endpoint")) for info in role_infos]


def run_mode(
    *,
    args: argparse.Namespace,
    mode: str,
    prompt_input_ids: list[list[int]],
    sampling_params: dict[str, Any],
    prompt_samples: list[PromptSample],
    dist_init_addrs: list[str | None],
    target_nnodes: int,
    target_gpus_per_node: int,
    dist_init_port_reservation_actors: list[Any | None] | None = None,
    pgs: list[Any] | None = None,
    topology: Any | None = None,
    include_output_text: bool = True,
    available_port_slot_base: int = 0,
    available_port_avoid_ports: set[int] | None = None,
) -> ModeMetrics:
    target_actor_groups: list[list[Any]] = []
    owns_pgs = pgs is None
    num_replicas = len(dist_init_addrs)
    if num_replicas <= 0:
        raise ValueError("run_mode requires at least one target replica")
    if len(dist_init_addrs) != num_replicas:
        raise ValueError(
            f"dist_init_addrs has {len(dist_init_addrs)} entries, expected {num_replicas}"
        )
    if pgs is not None and len(pgs) != num_replicas:
        raise ValueError(f"pgs has {len(pgs)} entries, expected {num_replicas}")
    if dist_init_port_reservation_actors is None:
        dist_init_port_reservation_actors = [None] * num_replicas
    if len(dist_init_port_reservation_actors) != num_replicas:
        raise ValueError(
            "dist_init_port_reservation_actors has "
            f"{len(dist_init_port_reservation_actors)} entries, "
            f"expected {num_replicas}"
        )

    replica_indices = _split_indices(len(prompt_samples), num_replicas)
    verifier_assignments = [0] * len(prompt_samples)
    for replica_index, indices in enumerate(replica_indices):
        for index in indices:
            verifier_assignments[index] = replica_index
    outputs_by_index: list[dict[str, Any] | None] = [None] * len(prompt_samples)
    try:
        if pgs is None:
            pgs = create_target_placement_groups(
                num_replicas,
                target_nnodes,
                target_gpus_per_node,
            )

        available_port_avoid_ports = set(available_port_avoid_ports or ())
        available_port_avoid_ports.update(
            _ports_from_dist_init_addrs(dist_init_addrs)
        )
        available_ports_by_replica = []
        for replica_index in range(num_replicas):
            available_ports_by_replica.append(
                _get_target_available_ports_from_pg(
                    args,
                    pgs[replica_index],
                    engine_slot=available_port_slot_base + replica_index,
                    avoid_ports=available_port_avoid_ports,
                )
            )

        for replica_index in range(num_replicas):
            available_ports = available_ports_by_replica[replica_index]
            if available_ports is not None:
                print(
                    "target_available_ports: "
                    f"mode={mode} replica={replica_index} "
                    f"ports={available_ports}",
                    flush=True,
                )
            actors = launch_target_actors(
                args=args,
                mode=mode,
                dist_init_addr=dist_init_addrs[replica_index],
                dist_init_port_reservation_actor=(
                    dist_init_port_reservation_actors[replica_index]
                ),
                target_nnodes=target_nnodes,
                target_gpus_per_node=target_gpus_per_node,
                pg=pgs[replica_index],
                rank_base=replica_index * args.target_dp_size,
                available_ports=available_ports,
            )
            target_actor_groups.append(actors)

        if mode == "decoupled_spec":
            if topology is None or topology.draft_actors is None:
                raise RuntimeError("decoupled_spec run requires draft topology")
            verifier_raw_infos = []
            for actors in target_actor_groups:
                ready_infos = ray.get([actor.ready.remote() for actor in actors])
                for ready_info in ready_infos:
                    verifier_raw_infos.extend(ready_info.get("endpoint_infos", []))
            topology.verifier_endpoint_infos = common.to_endpoint_infos(
                verifier_raw_infos
            )
            draft_control_endpoints = _sorted_bind_endpoints(
                topology.drafter_endpoint_infos, role="drafter"
            )
            verifier_result_endpoints = _sorted_bind_endpoints(
                topology.verifier_endpoint_infos, role="verifier"
            )
            verifier_results = ray.get(
                [
                    actors[0].configure_peers.remote(draft_control_endpoints)
                    for actors in target_actor_groups
                ]
            )
            draft_results = ray.get(
                [
                    actor.configure_peers.remote(verifier_result_endpoints)
                    for actor in topology.draft_actors
                ]
            )
            failures = [
                message
                for success, message in [*verifier_results, *draft_results]
                if not success
            ]
            if failures:
                raise RuntimeError(
                    "failed to configure decoupled-spec peers: "
                    + " | ".join(failures)
                )

        result_refs = []
        for replica_index, indices in enumerate(replica_indices):
            if not indices:
                continue
            shard_input_ids = [prompt_input_ids[index] for index in indices]
            result_refs.append(
                (
                    indices,
                    target_actor_groups[replica_index][0].generate_batch.remote(
                        shard_input_ids,
                        sampling_params,
                    ),
                )
            )

        for indices, result_ref in result_refs:
            result = ray.get(result_ref)
            shard_outputs = result["outputs"]
            if len(shard_outputs) != len(indices):
                raise RuntimeError(
                    f"{mode} returned {len(shard_outputs)} outputs for "
                    f"{len(indices)} prompts on one replica"
                )
            for index, output in zip(indices, shard_outputs, strict=True):
                outputs_by_index[index] = output
    finally:
        for actors in target_actor_groups:
            shutdown_actors(actors)
        if owns_pgs and pgs is not None:
            for pg in pgs:
                remove_placement_group(pg)

    if any(output is None for output in outputs_by_index):
        missing = [
            index for index, output in enumerate(outputs_by_index) if output is None
        ]
        raise RuntimeError(f"{mode} did not return outputs for indices {missing}")

    return collect_mode_metrics(
        mode=mode,
        outputs=[output for output in outputs_by_index if output is not None],
        prompt_samples=prompt_samples,
        verifier_assignments=verifier_assignments,
        include_output_text=include_output_text,
    )

build_result = common.build_result
write_output_files = common.write_output_files
print_summary = common.print_summary


def main() -> None:
    args = parse_args()
    _ensure_runtime_imports()

    prompt_column, prompt_samples, total_rows = load_prompt_samples(args)
    prompt_input_ids = [list(sample.prompt_input_ids) for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    draft_actors: list[Any] = []
    dist_init_port_reservation_actors: list[Any] = []
    spec_pgs = []
    try:
        init_ray(args.ray_address, args.ray_namespace, args.nnodes)
        target_nnodes, target_gpus_per_node = validate_resources(args)
        baseline_modes = common.resolve_baseline_modes(args.baseline)

        spec_pgs = create_target_placement_groups(
            args.num_verifier_replicas,
            target_nnodes,
            target_gpus_per_node,
        )
        (
            spec_reserved_ports,
            baseline_reserved_ports,
            reserved_ports,
        ) = _split_reserved_ports(args)
        if reserved_ports is not None:
            print(f"reserved_ports: {reserved_ports}", flush=True)
        available_port_avoid_ports = _known_dist_init_ports(
            args,
            spec_reserved_ports=spec_reserved_ports,
            baseline_reserved_ports=baseline_reserved_ports,
        )
        if args.target_enable_dp_attention and _target_uses_env_available_ports(args):
            print(
                "target_use_env_ports: true "
                f"avoid_dist_init_ports={sorted(available_port_avoid_ports)}",
                flush=True,
            )
        spec_dist_init_reservations = [
            reserve_dist_init_addr_from_pg(
                args,
                pg,
                port_offset=replica_index,
                preferred_port=(
                    spec_reserved_ports[replica_index]
                    if spec_reserved_ports
                    else None
                ),
            )
            for replica_index, pg in enumerate(spec_pgs)
        ]
        spec_dist_init_addrs = [
            reservation[0] for reservation in spec_dist_init_reservations
        ]
        spec_dist_init_port_reservation_actors = [
            reservation[1] for reservation in spec_dist_init_reservations
        ]
        dist_init_port_reservation_actors.extend(
            actor
            for actor in spec_dist_init_port_reservation_actors
            if actor is not None
        )
        topology = create_remote_decoupled_spec_topology(
            args,
            spec_pgs,
        )
        print_decoupled_spec_layout(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            verifier_pgs=spec_pgs,
            topology=topology,
        )
        draft_actors = topology.draft_actors or []
        spec_metrics = run_mode(
            args=args,
            mode="decoupled_spec",
            prompt_input_ids=prompt_input_ids,
            sampling_params=sampling_params,
            prompt_samples=prompt_samples,
            dist_init_addrs=spec_dist_init_addrs,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            dist_init_port_reservation_actors=spec_dist_init_port_reservation_actors,
            pgs=spec_pgs,
            topology=topology,
            include_output_text=True,
            available_port_slot_base=0,
            available_port_avoid_ports=available_port_avoid_ports,
        )
        shutdown_actors(draft_actors)
        draft_actors = []

        baseline_metrics = []
        baseline_dist_init_addrs_by_mode = {}
        for baseline_index, baseline_mode in enumerate(baseline_modes):
            baseline_offset = args.num_verifier_replicas * (1 + baseline_index)
            baseline_dist_init_reservations = [
                reserve_dist_init_addr_from_pg(
                    args,
                    pg,
                    port_offset=baseline_offset + replica_index,
                    preferred_port=(
                        baseline_reserved_ports[baseline_mode][replica_index]
                        if baseline_mode in baseline_reserved_ports
                        else None
                    ),
                )
                for replica_index, pg in enumerate(spec_pgs)
            ]
            baseline_dist_init_addrs = [
                reservation[0] for reservation in baseline_dist_init_reservations
            ]
            baseline_dist_init_port_reservation_actors = [
                reservation[1] for reservation in baseline_dist_init_reservations
            ]
            dist_init_port_reservation_actors.extend(
                actor
                for actor in baseline_dist_init_port_reservation_actors
                if actor is not None
            )
            baseline_dist_init_addrs_by_mode[baseline_mode] = baseline_dist_init_addrs
            baseline_metrics.append(
                run_mode(
                    args=args,
                    mode=baseline_mode,
                    prompt_input_ids=prompt_input_ids,
                    sampling_params=sampling_params,
                    prompt_samples=prompt_samples,
                    dist_init_addrs=baseline_dist_init_addrs,
                    target_nnodes=target_nnodes,
                    target_gpus_per_node=target_gpus_per_node,
                    dist_init_port_reservation_actors=(
                        baseline_dist_init_port_reservation_actors
                    ),
                    pgs=spec_pgs,
                    include_output_text=True,
                    available_port_slot_base=baseline_offset,
                    available_port_avoid_ports=available_port_avoid_ports,
                )
            )

        result = build_result(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            baseline_metrics=baseline_metrics,
        )
        if reserved_ports is not None:
            result["config"]["reserved_ports"] = reserved_ports
            result["config"]["spec_dist_init_addrs"] = spec_dist_init_addrs
        result["config"]["verifier_result_endpoints"] = _sorted_bind_endpoints(
            topology.verifier_endpoint_infos, role="verifier"
        )
        result["config"]["draft_control_endpoints"] = _sorted_bind_endpoints(
            topology.drafter_endpoint_infos, role="drafter"
        )
        if len(baseline_dist_init_addrs_by_mode) == 1:
            result["config"]["baseline_dist_init_addrs"] = next(
                iter(baseline_dist_init_addrs_by_mode.values())
            )
        elif baseline_dist_init_addrs_by_mode:
            result["config"]["baseline_dist_init_addrs_by_mode"] = (
                baseline_dist_init_addrs_by_mode
            )
        print_summary(result)
        if args.output_dir:
            print("output_files:")
            for output_path in write_output_files(result, args.output_dir):
                print(f"  {output_path}")
    finally:
        shutdown_actors(draft_actors)
        shutdown_port_reservation_actors(dist_init_port_reservation_actors)
        for pg in spec_pgs:
            remove_placement_group(pg)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()

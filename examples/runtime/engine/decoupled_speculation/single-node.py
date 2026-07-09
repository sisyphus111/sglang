#!/usr/bin/env python3
"""
Run decoupled speculative decoding on a single node without Ray.

This keeps the same user-facing workload arguments as
multi-node.py, but removes Ray/multi-node launch. The drafter
engines run in local child processes and the verifier engine runs in this
process, which makes it convenient to launch under nsys. Use --baseline all to
run decode and MTP baselines in sequence.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from pathlib import Path
from typing import Any

from sglang.srt.speculative.decoupled_speculation import common

LOCAL_HOST = "127.0.0.1"
DPA_DIST_INIT_PORT_BLOCK_SIZE = 6
DPA_ENV_FIXED_PORT_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation on a single node without Ray, "
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
            f"searched in order: {common.DEFAULT_PROMPT_COLUMN_CANDIDATES}."
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
            "Number of valid prompts to run in one generate call. Single-node "
            "mode supports one verifier replica."
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
            "Use PORT1..PORT30 as the target engine's available port pool. "
            "If unset, the legacy dist-init-derived ports are used."
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
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic inference for both decoupled drafter and "
            "verifier engines."
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Set sampling_params.ignore_eos=True for both decoupled speculative "
            "decoding and normal decoding. Disabled by default."
        ),
    )
    parser.add_argument(
        "--n-gpu-per-node",
        type=int,
        default=None,
        help=(
            "GPU count available on this node. If omitted, it is derived from "
            "--verify-ngpus and --draft-ngpus."
        ),
    )
    parser.add_argument(
        "--verify-ngpus",
        dest="verify_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for the verifier. Single-node mode supports "
            "one verifier replica, so this must equal the target verifier world "
            "size when set."
        ),
    )
    parser.add_argument(
        "--draft-ngpus",
        dest="draft_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for all drafters. The number of drafters is "
            "derived as draft_ngpus / draft_tp_size."
        ),
    )
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help=(
            "Optional verifier distributed init address override. If omitted, "
            "the script derives one from --dist-init-port or a free local port."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=None,
        help=(
            "Base port for this run. Spec dist-init uses base; baseline "
            "dist-init ports follow. Decoupled result/control endpoints are "
            "bound dynamically by SGLang."
        ),
    )
    parser.add_argument(
        "--reserved-ports",
        default=None,
        help=(
            "Comma- or whitespace-separated list of pre-reserved local ports. "
            "When set, ports do not need to be contiguous. They are consumed in "
            "order: spec dist-init, then optional baseline dist-init. Decoupled "
            "result/control endpoints are bound dynamically by SGLang. This "
            "cannot be combined with --dist-init-addr or --dist-init-port. "
            "With --target-enable-dp-attention, each dist-init allocation "
            "consumes a contiguous 6-port block in the legacy "
            "dist-init-derived port mode; with --target-use-env-ports, only "
            "the dist-init ports are consumed here and the fixed DP-attention "
            "ports come from PORT1..PORT30."
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
    parser.add_argument(
        "--spec-trace-dir",
        default=None,
        help="Directory for speculative decoding CSV trace files.",
    )
    parser.add_argument(
        "--draft-ready-timeout-s",
        type=float,
        default=900.0,
        help="Seconds to wait for all local draft engines to finish loading.",
    )
    return parser.parse_args()


def _visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    return [str(index) for index in range(256)]


def get_decoupled_spec_actor_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {
        "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL": os.environ.get(
            "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL", "1"
        )
    }
    for env_name in (
        "CUDA_LAUNCH_BLOCKING",
        "SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND",
        "SGLANG_DECOUPLED_SPEC_TRACE_DIR",
        "SGLANG_DECOUPLED_SPEC_SUMMARY_INTERVAL",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            env_vars[env_name] = env_value
    return env_vars


def _child_env_for_gpus(gpus: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    for name, value in get_decoupled_spec_actor_env_vars().items():
        env[name] = value
    return env


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _pick_free_port_block(
    num_ports: int,
    *,
    avoid_ports: set[int] | None = None,
) -> int:
    avoid_ports = avoid_ports or set()
    for _ in range(256):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((LOCAL_HOST, 0))
            base_port = int(sock.getsockname()[1])
        candidate_ports = {base_port + offset for offset in range(num_ports)}
        if candidate_ports & avoid_ports:
            continue
        if all(_port_available(port) for port in candidate_ports):
            return base_port
    raise RuntimeError(f"failed to find a free local block of {num_ports} ports")


def _port_from_dist_init_addr(addr: str | None) -> int | None:
    if addr is None:
        return None
    try:
        return int(addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _parse_reserved_ports(raw_ports: str | None) -> list[int]:
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


def _target_uses_env_available_ports(args: argparse.Namespace) -> bool:
    return args.target_use_env_ports


def _target_dp_attention_uses_dist_init_derived_ports(
    args: argparse.Namespace,
) -> bool:
    return (
        args.target_enable_dp_attention
        and not args.target_use_env_ports
    )


def _dist_init_port_stride(args: argparse.Namespace) -> int:
    return (
        DPA_DIST_INIT_PORT_BLOCK_SIZE
        if _target_dp_attention_uses_dist_init_derived_ports(args)
        else 1
    )


def _reserved_port_block_bases(
    reserved_ports: list[int],
    *,
    num_blocks: int,
    block_size: int,
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


def _allocate_local_ports(
    args: argparse.Namespace,
    *,
    num_verifiers: int,
) -> tuple[str, dict[str, str], list[int] | None]:
    baseline_modes = common.resolve_baseline_modes(args.baseline)
    port_stride = _dist_init_port_stride(args)
    reserved_ports = _parse_reserved_ports(args.reserved_ports)
    if reserved_ports:
        if args.dist_init_addr is not None or args.dist_init_port is not None:
            raise ValueError(
                "--reserved-ports cannot be combined with --dist-init-addr or "
                "--dist-init-port"
            )
        required_blocks = 1 + len(baseline_modes)
        required_ports = required_blocks * port_stride
        if len(reserved_ports) < required_ports:
            port_usage = "spec dist-init"
            if baseline_modes:
                port_usage += " and baseline dist-init"
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
            unavailable_ports = [
                port
                for base_port in block_bases
                for port in range(base_port, base_port + port_stride)
                if not _port_available(port)
            ]
            if unavailable_ports:
                raise ValueError(
                    f"--reserved-ports contains unavailable DPA block ports: "
                    f"{unavailable_ports}"
                )
            port_iter = iter(block_bases)
            spec_dist_init_addr = f"{LOCAL_HOST}:{next(port_iter)}"
            baseline_dist_init_addrs = {
                mode: f"{LOCAL_HOST}:{next(port_iter)}" for mode in baseline_modes
            }
            return (
                spec_dist_init_addr,
                baseline_dist_init_addrs,
                reserved_ports,
            )
        usable_ports = [port for port in reserved_ports if _port_available(port)]
        skipped_ports = [port for port in reserved_ports if port not in usable_ports]
        if len(usable_ports) < required_ports:
            raise ValueError(
                f"--reserved-ports has only {len(usable_ports)} currently "
                f"available ports after skipping in-use ports {skipped_ports}, "
                f"but this run needs {required_ports}"
            )
        if skipped_ports:
            print(
                f"skipping_in_use_reserved_ports: {skipped_ports}",
                flush=True,
            )
        port_iter = iter(usable_ports)
        spec_dist_init_addr = f"{LOCAL_HOST}:{next(port_iter)}"
        baseline_dist_init_addrs = {
            mode: f"{LOCAL_HOST}:{next(port_iter)}" for mode in baseline_modes
        }
        return (
            spec_dist_init_addr,
            baseline_dist_init_addrs,
            reserved_ports,
        )

    explicit_dist_init_port = _port_from_dist_init_addr(args.dist_init_addr)
    baseline_slot_count = len(baseline_modes)
    base_port = args.dist_init_port or _pick_free_port_block(
        (1 + baseline_slot_count) * port_stride,
        avoid_ports=(
            {explicit_dist_init_port} if explicit_dist_init_port is not None else None
        ),
    )
    spec_dist_init_addr = args.dist_init_addr or f"{LOCAL_HOST}:{base_port}"
    baseline_dist_init_addrs = {
        mode: (
            f"{LOCAL_HOST}:"
            f"{base_port + num_verifiers * (1 + mode_index) * port_stride}"
        )
        for mode_index, mode in enumerate(baseline_modes)
    }
    return (
        spec_dist_init_addr,
        baseline_dist_init_addrs,
        None,
    )


def _ports_from_dist_init_addrs(addrs: list[str]) -> set[int]:
    ports: set[int] = set()
    for addr in addrs:
        port = _port_from_dist_init_addr(addr)
        if port is not None:
            ports.add(port)
    return ports


def _known_dist_init_ports(
    args: argparse.Namespace,
    *,
    spec_dist_init_addr: str,
    baseline_dist_init_addrs: dict[str, str],
) -> set[int]:
    ports = _ports_from_dist_init_addrs(
        [spec_dist_init_addr, *baseline_dist_init_addrs.values()]
    )
    if ports:
        return ports

    base_port = args.dist_init_port
    if base_port is None:
        return ports

    baseline_modes = common.resolve_baseline_modes(args.baseline)
    port_stride = _dist_init_port_stride(args)
    return {
        base_port + slot * port_stride
        for slot in range(1 + len(baseline_modes))
    }


def _numbered_env_ports(
    *,
    prefix: str = "PORT",
    max_count: int = 30,
) -> list[int]:
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


def _select_target_available_ports(
    args: argparse.Namespace,
    *,
    engine_slot: int,
    avoid_ports: set[int],
) -> list[int] | None:
    if not (
        args.target_enable_dp_attention and _target_uses_env_available_ports(args)
    ):
        return None
    env_ports = _numbered_env_ports(prefix="PORT", max_count=30)
    skipped_avoided_ports = [port for port in env_ports if port in avoid_ports]
    candidate_ports = [port for port in env_ports if port not in avoid_ports]
    usable_ports = [
        port for port in candidate_ports if common.is_tcp_port_available(port)
    ]
    skipped_unavailable_ports = [
        port for port in candidate_ports if port not in usable_ports
    ]
    if skipped_unavailable_ports or skipped_avoided_ports:
        print(
            "target_env_ports_skipped: "
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
            f"{engine_slot}; env_ports={env_ports} "
            f"available_ports={usable_ports} "
            f"unavailable={skipped_unavailable_ports} "
            f"reserved={skipped_avoided_ports}"
        )
    return selected_ports


def get_target_engine_tp_size(args: argparse.Namespace) -> int:
    if args.target_enable_dp_attention:
        return args.target_tp_size * args.target_dp_size
    return args.target_tp_size


def validate_single_node_args(args: argparse.Namespace) -> None:
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
            "in this single-node example"
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
    if args.verify_ngpus is None:
        args.verify_ngpus = target_engine_tp_size
    if args.verify_ngpus != target_engine_tp_size:
        raise ValueError(
            "single-node.py currently supports one verifier replica, so "
            f"verify-ngpus must equal target verifier world size "
            f"({target_engine_tp_size})"
        )

    if args.n_gpu_per_node is None:
        args.n_gpu_per_node = args.verify_ngpus + args.draft_ngpus
    if args.n_gpu_per_node <= 0:
        raise ValueError("n-gpu-per-node must be positive")
    if args.verify_ngpus + args.draft_ngpus > args.n_gpu_per_node:
        raise ValueError(
            f"verify-ngpus + draft-ngpus ({args.verify_ngpus} + "
            f"{args.draft_ngpus}) exceeds n-gpu-per-node ({args.n_gpu_per_node})"
        )

    visible_gpus = _visible_gpu_ids()
    if len(visible_gpus) < args.n_gpu_per_node:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES exposes {len(visible_gpus)} GPUs, but this run "
            f"requires {args.n_gpu_per_node}"
        )

    args.num_verifier_replicas = 1
    args.nnodes = 1
    args.ray_address = None
    args.ray_namespace = None
    args.target_gpus = visible_gpus[: args.verify_ngpus]
    draft_start = args.verify_ngpus
    draft_end = draft_start + args.draft_ngpus
    args.draft_gpus = visible_gpus[draft_start:draft_end]


def run_draft_engine_process(
    *,
    rank: int,
    gpu_ids: list[str],
    model_path: str,
    tp_size: int,
    speculative_num_steps: int,
    max_running_requests: int | None,
    deterministic: bool,
    spec_trace_dir: str | None,
    ready_queue,
    control_reader,
) -> None:
    os.environ.update(_child_env_for_gpus(gpu_ids))
    try:
        import sglang as sgl

        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            disable_radix_cache=True,
            enable_deterministic_inference=deterministic,
            spec_trace_dir=spec_trace_dir,
            decoupled_spec_rank_base=rank,
        )
        if max_running_requests is not None:
            engine_kwargs["max_running_requests"] = max_running_requests
        engine = sgl.Engine(**engine_kwargs)
        endpoint_infos = engine.get_decoupled_spec_endpoint_infos()
        ready_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "gpus": gpu_ids,
                "endpoint_infos": endpoint_infos,
            }
        )
        while True:
            try:
                command = control_reader.recv()
            except EOFError:
                break
            if command == "stop":
                break
            if isinstance(command, dict) and command.get("type") == "configure":
                success, message = engine.configure_decoupled_spec_peers(
                    command["connect_endpoints"]
                )
                ready_queue.put(
                    {
                        "rank": rank,
                        "pid": os.getpid(),
                        "configured": bool(success),
                        "message": message,
                    }
                )
                if not success:
                    raise RuntimeError(message)
                continue
            raise RuntimeError(f"unknown draft control command: {command!r}")
    except Exception:
        ready_queue.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "gpus": gpu_ids,
                "error": traceback.format_exc(),
            }
        )
        raise
    finally:
        try:
            control_reader.close()
        except Exception:
            pass
        if "engine" in locals():
            engine.shutdown()


def wait_for_drafts(
    processes: list[mp.Process],
    ready_queue,
    timeout_s: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    pending = {index for index in range(len(processes))}
    ready: list[dict[str, Any]] = []
    while pending:
        for index in list(pending):
            process = processes[index]
            if not process.is_alive() and process.exitcode is not None:
                raise RuntimeError(
                    f"draft process index={index} exited early with "
                    f"exitcode={process.exitcode}"
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for draft engines: {pending}")
        try:
            message = ready_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        rank = int(message.get("rank", -1))
        if "error" in message:
            raise RuntimeError(
                f"draft rank {rank} failed to start:\n{message['error']}"
            )
        if rank in pending:
            pending.remove(rank)
            ready.append(message)
            endpoints = ",".join(
                info["bind_endpoint"] for info in message["endpoint_infos"]
            )
            print(
                "draft_ready: "
                f"rank={rank} pid={message['pid']} "
                f"gpus={','.join(message['gpus'])} "
                f"endpoints={endpoints}",
                flush=True,
            )
    ready.sort(key=lambda item: int(item["rank"]))
    return ready


def wait_for_draft_configure(
    processes: list[mp.Process],
    ready_queue,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    pending = {index for index in range(len(processes))}
    while pending:
        for index in list(pending):
            process = processes[index]
            if not process.is_alive() and process.exitcode is not None:
                raise RuntimeError(
                    f"draft process index={index} exited during configure with "
                    f"exitcode={process.exitcode}"
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out configuring draft engines: {pending}")
        try:
            message = ready_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        rank = int(message.get("rank", -1))
        if "error" in message:
            raise RuntimeError(
                f"draft rank {rank} failed during configure:\n{message['error']}"
            )
        if "configured" not in message:
            continue
        if not message["configured"]:
            raise RuntimeError(
                f"draft rank {rank} failed to configure: {message.get('message', '')}"
            )
        if rank in pending:
            pending.remove(rank)
            print(f"draft_configured: rank={rank}", flush=True)


def start_draft_engines(
    args: argparse.Namespace,
) -> tuple[list[mp.Process], list[Any], Any, list[dict[str, Any]]]:
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    stop_senders = []
    processes: list[mp.Process] = []
    try:
        for rank in range(args.num_draft_replicas):
            start = rank * args.draft_tp_size
            end = start + args.draft_tp_size
            gpu_ids = args.draft_gpus[start:end]
            control_reader, control_sender = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=run_draft_engine_process,
                kwargs=dict(
                    rank=rank,
                    gpu_ids=gpu_ids,
                    model_path=args.draft_model_path,
                    tp_size=args.draft_tp_size,
                    speculative_num_steps=args.num_speculative_steps,
                    max_running_requests=args.max_running_requests,
                    deterministic=args.deterministic,
                    spec_trace_dir=args.spec_trace_dir,
                    ready_queue=ready_queue,
                    control_reader=control_reader,
                ),
                name=f"decoupled-draft-rank-{rank}",
            )
            process.start()
            control_reader.close()
            stop_senders.append(control_sender)
            processes.append(process)
        draft_infos = wait_for_drafts(
            processes, ready_queue, args.draft_ready_timeout_s
        )
        return processes, stop_senders, ready_queue, draft_infos
    except Exception:
        shutdown_draft_engines(processes, stop_senders)
        raise


def _flatten_endpoint_infos(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoint_infos = []
    for message in messages:
        endpoint_infos.extend(message.get("endpoint_infos", []))
    return endpoint_infos


def _sorted_bind_endpoints(
    endpoint_infos: list[dict[str, Any]], *, role: str
) -> list[str]:
    role_infos = [info for info in endpoint_infos if info.get("role") == role]
    if not role_infos:
        raise RuntimeError(f"no decoupled-spec {role} endpoints were published")
    role_infos.sort(key=lambda info: int(info["rank"]))
    ranks = [int(info["rank"]) for info in role_infos]
    expected = list(range(len(ranks)))
    if ranks != expected:
        raise RuntimeError(
            f"decoupled-spec {role} ranks must be zero-based and contiguous: "
            f"got {ranks}"
        )
    return [str(info["bind_endpoint"]) for info in role_infos]


def configure_draft_engines(
    control_senders,
    ready_queue,
    verifier_result_endpoints: list[str],
    timeout_s: float,
    processes: list[mp.Process],
) -> None:
    for control_sender in control_senders:
        control_sender.send(
            {
                "type": "configure",
                "connect_endpoints": verifier_result_endpoints,
            }
        )
    wait_for_draft_configure(processes, ready_queue, timeout_s)


def shutdown_draft_engines(processes: list[mp.Process], stop_senders) -> None:
    if not processes:
        return
    print("stopping_draft_engines...", flush=True)
    for stop_sender in stop_senders or []:
        try:
            stop_sender.send("stop")
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            try:
                stop_sender.close()
            except Exception:
                pass
    for process in processes:
        process.join(timeout=60)
    for process in processes:
        if process.is_alive():
            print(
                f"terminating_draft_engine: name={process.name} pid={process.pid}",
                flush=True,
            )
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            print(
                f"killing_draft_engine: name={process.name} pid={process.pid}",
                flush=True,
            )
            process.kill()
    for process in processes:
        process.join(timeout=10)


def create_verifier_engine(
    args: argparse.Namespace,
    *,
    dist_init_addr: str,
    available_ports: list[int] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.target_gpus)
    for name, value in get_decoupled_spec_actor_env_vars().items():
        os.environ[name] = value

    import sglang as sgl

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tp_size=get_target_engine_tp_size(args),
        dp_size=args.target_dp_size,
        dist_init_addr=dist_init_addr,
        speculative_algorithm="DECOUPLED_VERIFY",
        speculative_num_steps=args.num_speculative_steps,
        speculative_num_draft_tokens=args.num_speculative_steps + 1,
        disable_radix_cache=True,
        enable_deterministic_inference=args.deterministic,
        spec_trace_dir=args.spec_trace_dir,
        log_level="info",
        decoupled_spec_rank_base=0,
    )
    if args.cuda_graph_bs_decode is not None:
        engine_kwargs["cuda_graph_bs_decode"] = args.cuda_graph_bs_decode
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.speculative_adaptive:
        engine_kwargs["speculative_adaptive"] = True
        engine_kwargs["speculative_adaptive_strategy"] = (
            args.speculative_adaptive_strategy
        )
        if args.speculative_adaptive_config is not None:
            engine_kwargs["speculative_adaptive_config"] = (
                args.speculative_adaptive_config
            )
        if args.decoupled_verify_throughput_profile_path is not None:
            engine_kwargs["decoupled_verify_throughput_profile_path"] = (
                args.decoupled_verify_throughput_profile_path
            )
    if args.target_enable_dp_attention:
        engine_kwargs["enable_dp_attention"] = True
        if available_ports is not None:
            engine_kwargs["available_ports"] = available_ports
    if args.target_ep_size is not None:
        engine_kwargs["ep_size"] = args.target_ep_size
    if args.target_moe_a2a_backend is not None:
        engine_kwargs["moe_a2a_backend"] = args.target_moe_a2a_backend
    return sgl.Engine(**engine_kwargs)


def create_decode_engine(
    args: argparse.Namespace,
    *,
    dist_init_addr: str,
    available_ports: list[int] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.target_gpus)

    import sglang as sgl

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tp_size=get_target_engine_tp_size(args),
        dp_size=args.target_dp_size,
        dist_init_addr=dist_init_addr,
        enable_deterministic_inference=args.deterministic,
        disable_overlap_schedule=True,
        spec_trace_dir=args.spec_trace_dir,
        log_level="info",
    )
    if args.cuda_graph_bs_decode is not None:
        engine_kwargs["cuda_graph_bs_decode"] = args.cuda_graph_bs_decode
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.target_enable_dp_attention:
        engine_kwargs["enable_dp_attention"] = True
        if available_ports is not None:
            engine_kwargs["available_ports"] = available_ports
    if args.target_ep_size is not None:
        engine_kwargs["ep_size"] = args.target_ep_size
    if args.target_moe_a2a_backend is not None:
        engine_kwargs["moe_a2a_backend"] = args.target_moe_a2a_backend
    return sgl.Engine(**engine_kwargs)


def create_mtp_engine(
    args: argparse.Namespace,
    *,
    dist_init_addr: str,
    available_ports: list[int] | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.target_gpus)

    import sglang as sgl

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tp_size=get_target_engine_tp_size(args),
        dp_size=args.target_dp_size,
        dist_init_addr=dist_init_addr,
        speculative_algorithm="EAGLE",
        speculative_num_steps=args.num_speculative_steps,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=args.num_speculative_steps + 1,
        enable_deterministic_inference=args.deterministic,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        mamba_scheduler_strategy="no_buffer",
        spec_trace_dir=args.spec_trace_dir,
        log_level="info",
    )
    if args.cuda_graph_bs_decode is not None:
        engine_kwargs["cuda_graph_bs_decode"] = args.cuda_graph_bs_decode
    if args.max_running_requests is not None:
        engine_kwargs["max_running_requests"] = args.max_running_requests
    if args.target_enable_dp_attention:
        engine_kwargs["enable_dp_attention"] = True
        if available_ports is not None:
            engine_kwargs["available_ports"] = available_ports
    if args.target_ep_size is not None:
        engine_kwargs["ep_size"] = args.target_ep_size
    if args.target_moe_a2a_backend is not None:
        engine_kwargs["moe_a2a_backend"] = args.target_moe_a2a_backend
    return sgl.Engine(**engine_kwargs)


def run_engine_generate(
    engine,
    *,
    input_ids: list[list[int]],
    sampling_params: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    if not isinstance(outputs, list):
        outputs = [outputs]
    return outputs


def _collect_mode_metrics(
    *,
    mode: str,
    outputs: list[dict[str, Any]],
    prompt_samples,
):
    return common.collect_mode_metrics(
        mode=mode,
        outputs=outputs,
        prompt_samples=prompt_samples,
        verifier_assignments=[0] * len(prompt_samples),
        include_output_text=True,
    )


def _write_summary_json(result: dict[str, Any], output_dir: str) -> Path:
    summary_path = Path(output_dir).expanduser() / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    args = parse_args()
    validate_single_node_args(args)
    target_engine_tp_size = get_target_engine_tp_size(args)

    prompt_column, prompt_samples, total_rows = common.load_prompt_samples(args)
    prompt_input_ids = [list(sample.prompt_input_ids) for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    num_verifiers = args.num_verifier_replicas
    (
        spec_dist_init_addr,
        baseline_dist_init_addrs,
        reserved_ports,
    ) = _allocate_local_ports(
        args,
        num_verifiers=num_verifiers,
    )
    available_port_avoid_ports = _known_dist_init_ports(
        args,
        spec_dist_init_addr=spec_dist_init_addr,
        baseline_dist_init_addrs=baseline_dist_init_addrs,
    )
    spec_available_ports = _select_target_available_ports(
        args,
        engine_slot=0,
        avoid_ports=available_port_avoid_ports,
    )
    baseline_available_ports = {
        mode: _select_target_available_ports(
            args,
            engine_slot=1 + mode_index,
            avoid_ports=available_port_avoid_ports,
        )
        for mode_index, mode in enumerate(common.resolve_baseline_modes(args.baseline))
    }

    print("local_decoupled_spec_topology:", flush=True)
    print(f"  target_gpus: {args.target_gpus}", flush=True)
    print(f"  target_tp_size: {args.target_tp_size}", flush=True)
    print(f"  target_dp_size: {args.target_dp_size}", flush=True)
    print(
        f"  target_enable_dp_attention: {args.target_enable_dp_attention}",
        flush=True,
    )
    print(f"  target_engine_tp_size: {target_engine_tp_size}", flush=True)
    print(
        f"  speculative_adaptive: {args.speculative_adaptive}",
        flush=True,
    )
    if args.speculative_adaptive:
        print(
            "  speculative_adaptive_strategy: "
            f"{args.speculative_adaptive_strategy}",
            flush=True,
        )
        if args.speculative_adaptive_config is not None:
            print(
                "  speculative_adaptive_config: "
                f"{args.speculative_adaptive_config}",
                flush=True,
            )
    print(f"  draft_gpus: {args.draft_gpus}", flush=True)
    if reserved_ports is not None:
        print(f"  reserved_ports: {reserved_ports}", flush=True)
    print(f"  verifier_dist_init_addr: {spec_dist_init_addr}", flush=True)
    if baseline_dist_init_addrs:
        print(f"  baseline_dist_init_addrs: {baseline_dist_init_addrs}", flush=True)
    if spec_available_ports is not None:
        print(
            f"  verifier_available_ports: {spec_available_ports}",
            flush=True,
        )
    if any(ports is not None for ports in baseline_available_ports.values()):
        print(
            f"  baseline_available_ports: {baseline_available_ports}",
            flush=True,
        )

    draft_processes: list[mp.Process] = []
    draft_stop_senders = None
    draft_ready_queue = None
    verifier_engine = None
    baseline_engine = None
    try:
        draft_processes, draft_stop_senders, draft_ready_queue, draft_infos = (
            start_draft_engines(args)
        )
        draft_control_endpoints = _sorted_bind_endpoints(
            _flatten_endpoint_infos(draft_infos), role="drafter"
        )
        print("creating_verifier_engine...", flush=True)
        verifier_engine = create_verifier_engine(
            args,
            dist_init_addr=spec_dist_init_addr,
            available_ports=spec_available_ports,
        )
        verifier_result_endpoints = _sorted_bind_endpoints(
            verifier_engine.get_decoupled_spec_endpoint_infos(), role="verifier"
        )
        print(f"  verifier_result_endpoints: {verifier_result_endpoints}", flush=True)
        print(f"  draft_control_endpoints: {draft_control_endpoints}", flush=True)
        success, message = verifier_engine.configure_decoupled_spec_peers(
            draft_control_endpoints
        )
        if not success:
            raise RuntimeError(f"failed to configure verifier peers: {message}")
        configure_draft_engines(
            draft_stop_senders,
            draft_ready_queue,
            verifier_result_endpoints,
            args.draft_ready_timeout_s,
            draft_processes,
        )
        print("running_decoupled_spec_generate...", flush=True)
        spec_outputs = run_engine_generate(
            verifier_engine,
            input_ids=prompt_input_ids,
            sampling_params=sampling_params,
        )
        print("decoupled_spec_generate_done", flush=True)
        spec_metrics = _collect_mode_metrics(
            mode="decoupled_spec",
            outputs=spec_outputs,
            prompt_samples=prompt_samples,
        )
        print("shutting_down_verifier_engine...", flush=True)
        verifier_engine.shutdown()
        verifier_engine = None
        shutdown_draft_engines(draft_processes, draft_stop_senders)
        draft_processes = []
        draft_stop_senders = None

        baseline_metrics = []
        for baseline_mode in common.resolve_baseline_modes(args.baseline):
            print(f"creating_{baseline_mode}_engine...", flush=True)
            baseline_dist_init_addr = baseline_dist_init_addrs[baseline_mode]
            if baseline_mode == "decode":
                baseline_engine = create_decode_engine(
                    args,
                    dist_init_addr=baseline_dist_init_addr,
                    available_ports=baseline_available_ports[baseline_mode],
                )
            elif baseline_mode == "mtp":
                baseline_engine = create_mtp_engine(
                    args,
                    dist_init_addr=baseline_dist_init_addr,
                    available_ports=baseline_available_ports[baseline_mode],
                )
            else:
                raise ValueError(f"Unsupported baseline: {baseline_mode}")
            print(f"running_{baseline_mode}_generate...", flush=True)
            baseline_outputs = run_engine_generate(
                baseline_engine,
                input_ids=prompt_input_ids,
                sampling_params=sampling_params,
            )
            print(f"{baseline_mode}_generate_done", flush=True)
            baseline_metrics.append(
                _collect_mode_metrics(
                    mode=baseline_mode,
                    outputs=baseline_outputs,
                    prompt_samples=prompt_samples,
                )
            )
            baseline_engine.shutdown()
            baseline_engine = None

        result = common.build_result(
            args=args,
            target_nnodes=1,
            target_gpus_per_node=target_engine_tp_size,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            baseline_metrics=baseline_metrics,
        )
        result["config"]["runner"] = "single_node_no_ray"
        result["config"]["target_gpus"] = args.target_gpus
        result["config"]["draft_gpus"] = args.draft_gpus
        result["config"]["verifier_result_endpoints"] = verifier_result_endpoints
        result["config"]["draft_control_endpoints"] = draft_control_endpoints
        result["config"]["verifier_dist_init_addr"] = spec_dist_init_addr
        if baseline_dist_init_addrs:
            result["config"]["baseline_dist_init_addrs"] = baseline_dist_init_addrs
        if reserved_ports is not None:
            result["config"]["reserved_ports"] = reserved_ports

        common.print_summary(result)
        if args.output_dir:
            print("output_files:")
            for output_path in common.write_output_files(result, args.output_dir):
                print(f"  {output_path}")
            print(f"  {_write_summary_json(result, args.output_dir)}")
    finally:
        if baseline_engine is not None:
            baseline_engine.shutdown()
        if verifier_engine is not None:
            verifier_engine.shutdown()
        shutdown_draft_engines(draft_processes, draft_stop_senders)


if __name__ == "__main__":
    main()

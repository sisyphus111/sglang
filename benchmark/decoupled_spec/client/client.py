"""Submit one pre-tokenized batch to a ready speculative-decoding engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client.metrics import build_result_artifacts, write_result_artifacts
from client.request_loader import RequestSpec, load_requests
from run_io import require_run_dir, update_run_config

DATASET_FORMATS = (
    "gsm8k",
    "parquet",
    "generic_parquet",
    "dapo_math_17k",
    "jsonl",
    "generic_jsonl",
    "codeforces_raw",
    "sharegpt",
    "synthetic_ids",
)


@dataclass(frozen=True)
class BatchExecution:
    """Completed streaming batch plus the client-side request boundaries."""

    outputs: list[dict[str, Any]]
    timings: list[dict[str, Any]]
    started_wall_time: float
    finished_wall_time: float
    started_monotonic_ns: int
    finished_monotonic_ns: int

    @property
    def elapsed_s(self) -> float:
        return (self.finished_monotonic_ns - self.started_monotonic_ns) / 1e9


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose common benchmark axes as typed YAML overrides."""
    parser.add_argument("--base-url")
    parser.add_argument("--server-manifest")
    parser.add_argument("--engine-rank", type=int)
    parser.add_argument("--verifier-rank", type=int)
    parser.add_argument("--target-tokenizer-path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataset-format", choices=DATASET_FORMATS)
    parser.add_argument("--dataset-path")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prompt-column")
    parser.add_argument("--reference-column")
    parser.add_argument("--prompt-len", type=int)
    parser.add_argument("--token-id", type=int)
    parser.add_argument("--chat-template-mode", choices=("tokenizer", "none"))
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply only explicitly supplied CLI values to the loaded YAML."""
    server_manifest = getattr(args, "server_manifest", None)
    engine_rank = getattr(args, "engine_rank", None)
    verifier_rank = getattr(args, "verifier_rank", None)
    if server_manifest is not None and args.base_url is not None:
        raise ValueError("--server-manifest conflicts with --base-url")
    if (
        engine_rank is not None or verifier_rank is not None
    ) and server_manifest is None:
        raise ValueError("--engine-rank/--verifier-rank requires --server-manifest")
    if engine_rank is not None and verifier_rank is not None:
        raise ValueError("--engine-rank conflicts with --verifier-rank")

    mappings = (
        ("base_url", "server", "base_url"),
        ("target_tokenizer_path", "target_tokenizer", "model_path"),
        ("batch_size", "batch", "size"),
        ("dataset_format", "dataset", "format"),
        ("dataset_path", "dataset", "path"),
        ("seed", "dataset", "seed"),
        ("shuffle", "dataset", "shuffle"),
        ("prompt_column", "dataset", "prompt_column"),
        ("reference_column", "dataset", "reference_column"),
        ("prompt_len", "dataset", "prompt_len"),
        ("token_id", "dataset", "token_id"),
        ("chat_template_mode", "chat_template", "mode"),
        ("enable_thinking", "chat_template", "enable_thinking"),
        ("output_len", "generation", "output_len"),
        ("temperature", "generation", "temperature"),
        ("ignore_eos", "generation", "ignore_eos"),
    )
    for argument, section_name, field_name in mappings:
        value = getattr(args, argument)
        if value is None:
            continue
        section = config.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")
        section[field_name] = value

    if server_manifest is not None:
        manifest_path = Path(server_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("server manifest must contain a JSON object")
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError("server manifest schema_version must be 1")
        if manifest.get("state") != "ready":
            raise ValueError(
                f"server manifest is not ready: state={manifest.get('state')!r}"
            )
        engines = manifest.get("engines")
        if not isinstance(engines, list) or not engines:
            raise ValueError("ready server manifest must contain non-empty engines")

        target_engines = [
            engine
            for engine in engines
            if isinstance(engine, dict) and engine.get("role") == "target"
        ]
        selected_role = "target" if target_engines else "verifier"
        candidate_engines = (
            target_engines
            if target_engines
            else [
                engine
                for engine in engines
                if isinstance(engine, dict) and engine.get("role") == "verifier"
            ]
        )
        requested_rank = engine_rank if engine_rank is not None else verifier_rank
        if requested_rank is None:
            if len(candidate_engines) != 1:
                rank_option = (
                    "--engine-rank" if selected_role == "target" else "--verifier-rank"
                )
                raise ValueError(
                    f"{rank_option} is required when the server manifest contains "
                    f"{len(candidate_engines)} {selected_role} engines"
                )
            rank = candidate_engines[0].get("rank")
        else:
            rank = requested_rank
        if type(rank) is not int:
            raise ValueError("selected engine rank must be an integer")
        if rank < 0:
            raise ValueError("selected engine rank must be non-negative")
        matching = [
            engine
            for engine in candidate_engines
            if engine.get("role") == selected_role
            and type(engine.get("rank")) is int
            and engine.get("rank") == rank
        ]
        if len(matching) != 1:
            raise ValueError(
                "server manifest must contain exactly one "
                f"{selected_role} rank {rank}; "
                f"found {len(matching)}"
            )
        engine = matching[0]
        target_id = engine.get("engine_id")
        base_url = engine.get("http_url")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("selected engine_id must be a non-empty string")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("selected engine http_url must be a non-empty string")
        server = config.setdefault("server", {})
        if not isinstance(server, dict):
            raise ValueError("server must be a mapping")
        server.update(
            {
                "base_url": base_url.rstrip("/"),
                "target_id": target_id,
                "role": selected_role,
                "rank": rank,
            }
        )
        config["server_manifest"] = {
            "path": str(manifest_path),
            "state": "ready",
        }


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 1)) != 1:
        raise ValueError("unsupported schema_version; expected 1")
    if not config.get("server", {}).get("base_url"):
        raise ValueError("server.base_url is required")
    if not config.get("target_tokenizer", {}).get("model_path"):
        raise ValueError("target_tokenizer.model_path is required")
    if int(config.get("batch", {}).get("size", 0)) <= 0:
        raise ValueError("batch.size must be positive")


def build_batch_payload(
    requests: list[RequestSpec], generation: dict[str, Any]
) -> dict[str, Any]:
    if not requests:
        raise ValueError("cannot submit an empty batch")
    if any(request.input_ids is None for request in requests):
        raise ValueError("every request must be tokenized before batch submission")
    base_sampling_params = {"temperature": 0, "ignore_eos": True}
    base_sampling_params.update(
        {key: value for key, value in generation.items() if key != "output_len"}
    )
    sampling_params = []
    for request in requests:
        params = dict(base_sampling_params)
        params["max_new_tokens"] = request.requested_output_len
        sampling_params.append(params)
    return {
        "rid": [request.request_id for request in requests],
        "input_ids": [request.input_ids for request in requests],
        "sampling_params": sampling_params,
        "stream": True,
    }


def extract_sse_events(buffer: bytearray) -> tuple[list[dict[str, Any]], bool]:
    """Consume complete SSE frames while preserving an incomplete tail."""
    buffer[:] = buffer.replace(b"\r\n", b"\n")
    events = []
    done = False
    while b"\n\n" in buffer:
        raw, _, rest = buffer.partition(b"\n\n")
        buffer[:] = rest
        data = b"\n".join(
            line[5:].strip() for line in raw.splitlines() if line.startswith(b"data:")
        )
        if not data:
            continue
        if data == b"[DONE]":
            done = True
            continue
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"verifier streaming error: {event['error']}")
        events.append(event)
    return events, done


async def run_batch(
    config: dict[str, Any], requests: list[RequestSpec]
) -> BatchExecution:
    base_url = config["server"]["base_url"].rstrip("/")
    timeout = aiohttp.ClientTimeout(
        total=float(config["server"].get("timeout_s", 7200))
    )
    payload = build_batch_payload(requests, config.get("generation", {}))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        started_wall_time = time.time()
        started_ns = time.monotonic_ns()
        outputs: list[dict[str, Any] | None] = [None] * len(requests)
        timing_states = [
            {
                "last_token_ns": None,
                "last_completion_tokens": 0,
            }
            for _ in requests
        ]
        done_seen = False
        async with session.post(base_url + "/generate", json=payload) as response:
            response.raise_for_status()
            buffer = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                buffer.extend(chunk)
                events, done = extract_sse_events(buffer)
                done_seen = done_seen or done
                for event in events:
                    received_ns = time.monotonic_ns()
                    index = event.get("index")
                    if not isinstance(index, int) or not 0 <= index < len(requests):
                        raise ValueError(f"invalid batch stream index: {index!r}")
                    state = timing_states[index]
                    completion_tokens = int(
                        event.get("meta_info", {}).get("completion_tokens", 0) or 0
                    )
                    if completion_tokens > state["last_completion_tokens"]:
                        state["last_token_ns"] = received_ns
                        state["last_completion_tokens"] = completion_tokens
                    outputs[index] = event
            trailing_events, done = extract_sse_events(buffer)
            done_seen = done_seen or done
            if trailing_events or buffer.strip():
                raise ValueError("incomplete SSE response at end of batch stream")
        finished_ns = time.monotonic_ns()
        finished_wall_time = time.time()
    if not done_seen:
        raise ValueError("batch stream ended without [DONE]")
    if any(output is None for output in outputs):
        missing = [index for index, output in enumerate(outputs) if output is None]
        raise ValueError(f"batch stream returned no output for indices {missing}")
    timings = []
    for state in timing_states:
        last_ns = state["last_token_ns"]
        timings.append(
            {
                "e2e_latency_ms": ((last_ns - started_ns) / 1e6 if last_ns else None),
            }
        )
    return BatchExecution(
        outputs=[output for output in outputs if output is not None],
        timings=timings,
        started_wall_time=started_wall_time,
        finished_wall_time=finished_wall_time,
        started_monotonic_ns=started_ns,
        finished_monotonic_ns=finished_ns,
    )


def execute_client(
    config: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], BatchExecution]:
    """Prepare one workload, execute it, and persist only Client-owned artifacts."""

    update_run_config(run_dir, "client", config)

    from sglang.benchmark.utils import get_tokenizer

    tokenizer = get_tokenizer(config["target_tokenizer"]["model_path"])
    requests = load_requests(config, tokenizer)
    if len(requests) != int(config["batch"]["size"]):
        raise ValueError("prepared request count does not equal batch.size")
    execution = asyncio.run(run_batch(config, requests))
    verifier_rank = int(config.get("server", {}).get("rank", 0))
    request_rows, batch_result, content_rows = build_result_artifacts(
        requests,
        execution.outputs,
        execution.timings,
        verifier_rank,
    )
    write_result_artifacts(
        run_dir / "client",
        request_rows,
        batch_result,
        content_rows,
    )
    return batch_result, execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    add_cli_args(parser)
    parser.add_argument("--check", action="store_true")
    cli = parser.parse_args()
    config = (
        yaml.safe_load(Path(cli.config).expanduser().read_text(encoding="utf-8")) or {}
    )
    apply_cli_overrides(config, cli)
    validate_config(config)
    if cli.check:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if cli.run_dir is None:
        parser.error("--run-dir is required unless --check is used")

    run_dir = require_run_dir(cli.run_dir)
    batch_result, _ = execute_client(config, run_dir)

    print(json.dumps(batch_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

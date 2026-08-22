"""Submit one pre-tokenized batch to an already-running verifier server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.artifacts import update_status, write_json
from metrics import summarize, write_records, write_summary
from request_loader import RequestSpec, load_requests

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


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose common benchmark axes as typed YAML overrides."""
    parser.add_argument("--base-url")
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


def build_records(
    requests: list[RequestSpec],
    outputs: list[dict[str, Any]],
    timings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if len(outputs) != len(requests):
        raise ValueError(
            f"verifier returned {len(outputs)} outputs for batch size {len(requests)}"
        )
    records = []
    if timings is not None and len(timings) != len(requests):
        raise ValueError("timing count does not match batch size")
    for index, (request, output) in enumerate(zip(requests, outputs)):
        meta_info = output.get("meta_info", {})
        completion_tokens = int(meta_info.get("completion_tokens", 0) or 0)
        timing = timings[index] if timings is not None else {}
        records.append(
            {
                "request_id": request.request_id,
                "row_index": request.row_index,
                "prompt_len": request.prompt_len,
                "resp_len": completion_tokens,
                "completion_tokens": completion_tokens,
                "generated_text": output.get("text", ""),
                "reference_response": request.reference_response,
                "ttft_ms": timing.get("ttft_ms"),
                "tpot_ms": timing.get("tpot_ms"),
                "e2e_latency_ms": timing.get("e2e_latency_ms"),
                "stream_event_ct": timing.get("stream_event_ct"),
                **{
                    key: meta_info.get(key)
                    for key in (
                        "spec_verify_ct",
                        "spec_num_correct_drafts",
                        "spec_num_proposed_drafts",
                        "spec_accept_length",
                        "spec_accept_rate",
                        "spec_draft_occupancy_rate",
                        "spec_proposed_draft_length",
                        "spec_proposed_drafts_histogram",
                        "spec_num_proposed_drafts_by_position",
                        "spec_num_correct_drafts_by_position",
                        "spec_accept_rate_by_position",
                    )
                },
            }
        )
    return records


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
    config: dict[str, Any], requests: list[RequestSpec], run_dir: Path | None
) -> tuple[list[dict[str, Any]], float]:
    base_url = config["server"]["base_url"].rstrip("/")
    timeout = aiohttp.ClientTimeout(
        total=float(config["server"].get("timeout_s", 7200))
    )
    payload = build_batch_payload(requests, config.get("generation", {}))
    client_dir = run_dir / "client" if run_dir else None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(base_url + "/model_info") as response:
            response.raise_for_status()
            model_info = await response.json()
        if client_dir:
            write_json(client_dir / "verifier_model_info.json", model_info)
        started_wall_time = time.time()
        started_ns = time.monotonic_ns()
        if client_dir:
            write_json(
                client_dir / "formal_window.json",
                {
                    "state": "running",
                    "batch_size": len(requests),
                    "started_wall_time": started_wall_time,
                    "started_monotonic_ns": started_ns,
                },
            )
        outputs: list[dict[str, Any] | None] = [None] * len(requests)
        timing_states = [
            {
                "first_token_ns": None,
                "last_token_ns": None,
                "last_completion_tokens": 0,
                "stream_event_ct": 0,
            }
            for _ in requests
        ]
        raw_event_records = []
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
                    received_wall_time = time.time()
                    index = event.get("index")
                    if not isinstance(index, int) or not 0 <= index < len(requests):
                        raise ValueError(f"invalid batch stream index: {index!r}")
                    state = timing_states[index]
                    state["stream_event_ct"] += 1
                    completion_tokens = int(
                        event.get("meta_info", {}).get("completion_tokens", 0) or 0
                    )
                    if completion_tokens > state["last_completion_tokens"]:
                        if state["first_token_ns"] is None:
                            state["first_token_ns"] = received_ns
                        state["last_token_ns"] = received_ns
                        state["last_completion_tokens"] = completion_tokens
                    outputs[index] = event
                    if client_dir:
                        meta_info = event.get("meta_info", {})
                        raw_event_records.append(
                            {
                                "received_wall_time": received_wall_time,
                                "received_monotonic_ns": received_ns,
                                "index": index,
                                "request_id": requests[index].request_id,
                                "completion_tokens": completion_tokens,
                                "finish_reason": meta_info.get("finish_reason"),
                                # Cumulative request counters let offline analysis
                                # recover each verify round's accepted/proposed
                                # increments without adding runtime hot-path tracing.
                                "spec_verify_ct": meta_info.get("spec_verify_ct"),
                                "spec_num_proposed_drafts": meta_info.get(
                                    "spec_num_proposed_drafts"
                                ),
                                "spec_num_correct_drafts": meta_info.get(
                                    "spec_num_correct_drafts"
                                ),
                            }
                        )
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
    elapsed_s = (finished_ns - started_ns) / 1e9
    timings = []
    for state in timing_states:
        first_ns = state["first_token_ns"]
        last_ns = state["last_token_ns"]
        completion_tokens = state["last_completion_tokens"]
        timings.append(
            {
                "ttft_ms": (first_ns - started_ns) / 1e6 if first_ns else None,
                "tpot_ms": (
                    (last_ns - first_ns) / (completion_tokens - 1) / 1e6
                    if first_ns and last_ns and completion_tokens > 1
                    else None
                ),
                "e2e_latency_ms": ((last_ns - started_ns) / 1e6 if last_ns else None),
                "stream_event_ct": state["stream_event_ct"],
            }
        )
    if client_dir:
        with (client_dir / "stream_timing_events.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for record in raw_event_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(client_dir / "raw_batch_response.json", outputs)
        write_json(
            client_dir / "formal_window.json",
            {
                "state": "completed",
                "batch_size": len(requests),
                "started_wall_time": started_wall_time,
                "finished_wall_time": finished_wall_time,
                "started_monotonic_ns": started_ns,
                "finished_monotonic_ns": finished_ns,
                "elapsed_s": elapsed_s,
            },
        )
    return build_records(requests, outputs, timings), elapsed_s


def msgspec_to_dict(item: RequestSpec) -> dict[str, Any]:
    return {field: getattr(item, field) for field in item.__struct_fields__}


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

    run_dir = Path(cli.run_dir).expanduser() if cli.run_dir else None
    if run_dir:
        write_json(run_dir / "client" / "resolved_config.json", config)
        update_status(
            run_dir,
            "client",
            "starting",
            config=str(Path(cli.config).resolve()),
            batch_size=int(config["batch"]["size"]),
        )

    try:
        from sglang.benchmark.utils import get_tokenizer

        tokenizer = get_tokenizer(config["target_tokenizer"]["model_path"])
        requests = load_requests(config, tokenizer)
        if len(requests) != int(config["batch"]["size"]):
            raise ValueError("prepared request count does not equal batch.size")
        if run_dir:
            with (run_dir / "client" / "sampled_requests.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for item in requests:
                    stream.write(
                        json.dumps(msgspec_to_dict(item), ensure_ascii=False) + "\n"
                    )
        records, elapsed_s = asyncio.run(run_batch(config, requests, run_dir))
    except BaseException as exc:
        if run_dir:
            update_status(run_dir, "client", "failed", error=repr(exc))
        raise

    summary = summarize(records, elapsed_s)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if run_dir:
        client_dir = run_dir / "client"
        with (client_dir / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_records(client_dir / "request_metrics.csv", records)
        write_summary(client_dir / "summary.json", summary)
        update_status(
            run_dir,
            "client",
            "completed",
            batch_size=len(records),
            elapsed_s=elapsed_s,
        )


if __name__ == "__main__":
    main()

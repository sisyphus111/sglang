"""Generate one decoupled-verifier scheduler-cycle profile JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

from sglang.srt.speculative.decoupled_verify_profile import (
    DECOUPLED_VERIFY_PROFILE_JOB_KIND,
    DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
    load_decoupled_verify_profile,
    resolve_profile_input_context_len,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_config(path: str) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Offline profile config must be a YAML mapping.")
    unknown = set(value) - {
        "schema_version",
        "profile",
        "runtime",
        "server_args",
    }
    if unknown:
        raise ValueError(f"Unknown offline profile config keys: {sorted(unknown)}.")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("Offline profile config schema_version must be 1.")
    for key in ("profile", "runtime", "server_args"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"Offline profile config requires a {key} mapping.")
    return value


def _normalize_grid(
    config: dict[str, Any],
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    profile = config["profile"]
    steps = sorted({int(value) for value in profile["steps"]})
    batch_sizes = sorted({int(value) for value in profile["batch_sizes"]})
    context_lens = sorted({int(value) for value in profile["context_lens"]})
    warmup_iters = int(profile.get("warmup_iters", 10))
    measure_iters = int(profile.get("measure_iters", 50))
    max_steps = int(config["server_args"]["speculative_num_steps"])
    if (
        not steps
        or not batch_sizes
        or not context_lens
        or steps[0] < 0
        or steps[-1] > max_steps
        or batch_sizes[0] <= 0
        or context_lens[0] <= 0
        or warmup_iters < 0
        or measure_iters <= 0
    ):
        raise ValueError("Offline profile grid contains invalid values.")
    points = [
        {
            "step": step,
            "batch_size": batch_size,
            "context_len": context_len,
            "input_context_len": resolve_profile_input_context_len(
                target_context_len=context_len,
                step=step,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
            ),
        }
        for step in steps
        for batch_size in batch_sizes
        for context_len in context_lens
    ]
    return points, {
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "random_seed": int(profile.get("random_seed", 42)),
    }


def _point_key(point: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(point["step"]),
        int(point["batch_size"]),
        int(point["context_len"]),
    )


def _build_engine_kwargs(
    config: dict[str, Any], *, profile_batch_sizes: list[int]
) -> dict[str, Any]:
    kwargs = dict(config["server_args"])
    max_steps = int(kwargs["speculative_num_steps"])
    kwargs.update(
        speculative_algorithm="DECOUPLED_VERIFY",
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=max_steps + 1,
        speculative_adaptive=False,
        decoupled_spec_role="verifier",
        decoupled_spec_rank=0,
        decoupled_spec_bind_endpoint=f"tcp://127.0.0.1:{_free_port()}",
        decoupled_spec_connect_endpoints=[f"tcp://127.0.0.1:{_free_port()}"],
        cuda_graph_bs_decode=profile_batch_sizes,
        max_running_requests=max(
            int(kwargs.get("max_running_requests") or 0),
            max(profile_batch_sizes) + 4,
        ),
        skip_server_warmup=True,
        disable_prefill_cuda_graph=True,
        disable_radix_cache=True,
    )
    return kwargs


def _response_errors(response: Any) -> list[str]:
    rows = response if isinstance(response, list) else [response]
    errors = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("error", "message", "detail"):
            if row.get(key):
                errors.append(str(row[key]))
    return errors


def _run_workload(engine: Any, job: dict[str, Any]) -> None:
    model_vocab_size = int(engine.tokenizer_manager.model_config.vocab_size)
    tokenizer_vocab_size = int(
        getattr(engine.tokenizer_manager.tokenizer, "vocab_size", model_vocab_size)
    )
    vocab_size = min(model_vocab_size, tokenizer_vocab_size)
    rng = random.Random(int(job["random_seed"]))
    prefill_microbatch_tokens = int(
        engine.server_args.chunked_prefill_size or engine.server_args.max_prefill_tokens
    )
    for point_index, point in enumerate(job["points"]):
        step = int(point["step"])
        batch_size = int(point["batch_size"])
        input_context_len = int(point["input_context_len"])
        num_prefill_microbatches = math.ceil(
            batch_size * input_context_len / prefill_microbatch_tokens
        )
        rounds = (
            int(job["warmup_iters"])
            + int(job["measure_iters"])
            + max(12, 6 * num_prefill_microbatches)
            + 5
        )
        max_new_tokens = rounds * (step + 1)
        input_ids = [
            [rng.randrange(1, vocab_size) for _ in range(input_context_len)]
            for _ in range(batch_size)
        ]
        rids = [
            f"{job['request_prefix']}p{point_index}-k{step}-b{batch_size}-"
            f"c{point['context_len']}-r{row}"
            for row in range(batch_size)
        ]
        started = time.perf_counter()
        response = engine.generate(
            input_ids=input_ids,
            rid=rids,
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "ignore_eos": True,
                "max_new_tokens": max_new_tokens,
            },
        )
        errors = _response_errors(response)
        if errors:
            raise RuntimeError(
                f"Offline profile point {_point_key(point)} failed: {errors}."
            )
        print(
            "offline-profile point completed: "
            f"point={point_index + 1}/{len(job['points'])} "
            f"step={step} bs={batch_size} ctx={point['context_len']} "
            f"elapsed_s={time.perf_counter() - started:.2f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    requested_points, settings = _normalize_grid(config)
    output_path = Path(args.output).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve() if args.input else None
    in_place = input_path is not None and input_path == output_path
    if output_path.exists() and not (args.overwrite or in_place):
        raise FileExistsError(
            f"Output exists; pass --overwrite or use --input in place: {output_path}."
        )
    if output_path.exists() and args.overwrite and not in_place:
        output_path.unlink()
    input_profile = (
        load_decoupled_verify_profile(input_path, require_complete=False)
        if input_path is not None
        else None
    )
    existing_keys = (
        set()
        if input_profile is None
        else {_point_key(point) for point in input_profile["points"]}
    )
    missing_points = [
        point for point in requested_points if _point_key(point) not in existing_keys
    ]
    summary = {
        "requested_points": len(requested_points),
        "reused_points": len(requested_points) - len(missing_points),
        "points_to_profile": len(missing_points),
        "output": str(output_path),
    }
    if args.check:
        print(json.dumps({"status": "checked", **summary}, indent=2, sort_keys=True))
        return
    if input_profile is not None and not in_place:
        atomic_write_json(output_path, input_profile)
    if not missing_points:
        if input_profile is None:
            raise RuntimeError("No profile input or missing point was provided.")
        print(
            json.dumps(
                {
                    "status": input_profile["status"],
                    "sha256": file_sha256(output_path),
                    **summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    request_prefix = f"__sglang_decoupled_verify_profile__{time.time_ns()}-"
    job = {
        "schema_version": DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION,
        "kind": DECOUPLED_VERIFY_PROFILE_JOB_KIND,
        "output_path": str(output_path),
        "request_prefix": request_prefix,
        "points": missing_points,
        "requested_points": requested_points,
        "reused_point_count": summary["reused_points"],
        **settings,
        "provenance": {
            "config_path": str(Path(args.config).expanduser().resolve()),
            "profile_cli": "benchmark.decoupled_spec.server.profile",
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="sglang-decoupled-profile-", delete=False
    ) as file:
        json.dump(job, file, ensure_ascii=False, indent=2)
        spec_path = Path(file.name)

    runtime_env = {
        str(key): str(value) for key, value in config["runtime"].get("env", {}).items()
    }
    runtime_env.update(
        {
            "SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND": "1",
            "SGLANG_DECOUPLED_VERIFY_OFFLINE_PROFILE_SPEC_PATH": str(spec_path),
            "SGLANG_SIMULATE_ACC_LEN": str(
                int(config["server_args"]["speculative_num_steps"]) + 1
            ),
            "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed",
            "SGLANG_RAGGED_VERIFY_MODE": "static",
        }
    )
    old_env = {key: os.environ.get(key) for key in runtime_env}
    os.environ.update(runtime_env)
    engine = None
    try:
        import sglang as sgl

        engine = sgl.Engine(
            **_build_engine_kwargs(
                config,
                profile_batch_sizes=sorted(
                    {int(point["batch_size"]) for point in missing_points}
                ),
            )
        )
        _run_workload(engine, job)
    finally:
        if engine is not None:
            engine.shutdown()
        spec_path.unlink(missing_ok=True)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    profile = load_decoupled_verify_profile(output_path, require_complete=True)
    print(
        json.dumps(
            {
                "status": profile["status"],
                "sha256": file_sha256(output_path),
                "actual_points": len(profile["points"]),
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

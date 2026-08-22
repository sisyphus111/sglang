#!/usr/bin/env python3
"""Resolve and materialize a decoupled-spec client workload without sending it."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
CLIENT_ROOT = BENCHMARK_ROOT / "client-side"
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(CLIENT_ROOT))

from client import add_cli_args, apply_cli_overrides, validate_config  # noqa: E402
from common.artifacts import write_json  # noqa: E402
from request_loader import RequestSpec, load_requests  # noqa: E402


def _preview(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"


def _ids_sha256(input_ids: list[int] | None) -> str | None:
    if input_ids is None:
        return None
    payload = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_report(
    config: dict[str, Any],
    requests: list[RequestSpec],
    config_path: Path,
    preview_chars: int,
    include_input_ids: bool,
) -> dict[str, Any]:
    if preview_chars < 0:
        raise ValueError("preview_chars must be non-negative")
    expected = int(config["batch"]["size"])
    if len(requests) != expected:
        raise ValueError(
            f"prepared request count {len(requests)} does not equal batch.size={expected}"
        )
    prompt_lengths = [request.prompt_len for request in requests]
    samples = []
    for request in requests:
        sample = {
            "request_id": request.request_id,
            "row_index": request.row_index,
            "source": request.source,
            "prompt_len": request.prompt_len,
            "requested_output_len": request.requested_output_len,
            "raw_prompt_preview": _preview(request.raw_prompt, preview_chars),
            "rendered_prompt_preview": _preview(request.rendered_prompt, preview_chars),
            "input_ids_sha256": _ids_sha256(request.input_ids),
        }
        if include_input_ids:
            sample["input_ids"] = request.input_ids
        samples.append(sample)
    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "resolved_config": config,
        "dataset_format": config.get("dataset", {}).get("format"),
        "dataset_path": config.get("dataset", {}).get("path"),
        "target_tokenizer_path": config["target_tokenizer"]["model_path"],
        "batch_size": len(requests),
        "prompt_lengths": {
            "values": prompt_lengths,
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "mean": statistics.fmean(prompt_lengths),
        },
        "requested_output_lengths": [
            request.requested_output_len for request in requests
        ],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    add_cli_args(parser)
    parser.add_argument("--output")
    parser.add_argument("--preview-chars", type=int, default=160)
    parser.add_argument(
        "--include-input-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    apply_cli_overrides(config, args)
    validate_config(config)

    from sglang.benchmark.utils import get_tokenizer

    tokenizer = get_tokenizer(config["target_tokenizer"]["model_path"])
    requests = load_requests(config, tokenizer)
    report = build_report(
        config,
        requests,
        config_path,
        args.preview_chars,
        args.include_input_ids,
    )
    if args.output:
        write_json(Path(args.output).expanduser(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

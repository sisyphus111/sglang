"""Build the fixed human-facing client result contract."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from client.request_loader import RequestSpec
from run_io import write_json

# User-frozen external schema. Change only on an explicit contract request; see
# skills/send-decoupled-spec-workload/references/client-artifact-contract.md.
REQUESTS_CSV_FIELDS = (
    "batch_row_index",
    "dataset_idx",
    "verifier_rank",
    "prompt_len",
    "resp_len",
    "spec_verify_ct",
    "valid_draft_len",
    "acc_len",
    "e2e_latency_s",
    "spec_num_proposed_drafts_by_position",
    "spec_num_correct_drafts_by_position",
    "spec_accept_rate_by_position",
)
BATCH_JSON_FIELDS = (
    "output_tokens",
    "batch_elapsed_latency_s",
    "batch_thpt",
    "mean_valid_draft_len",
    "acclen",
)
CONTENT_JSON_FIELDS = (
    "batch_row_idx",
    "dataset_idx",
    "input_len",
    "output_len",
    "input_ids",
    "input_text",
    "output_ids",
    "output_text",
)
_ARRAY_CSV_FIELDS = (
    "spec_num_proposed_drafts_by_position",
    "spec_num_correct_drafts_by_position",
    "spec_accept_rate_by_position",
)


def _normalize_position_metrics(
    meta_info: dict[str, Any],
    *,
    spec_verify_ct: int,
    proposed_draft_length: int | float,
) -> tuple[list[int], list[int], list[float | None]]:
    """Return one per-position contract for decoupled and coupled MTP results."""

    proposed = meta_info.get("spec_num_proposed_drafts_by_position")
    correct = meta_info.get("spec_num_correct_drafts_by_position")
    rates = meta_info.get("spec_accept_rate_by_position")
    if proposed is not None and correct is not None and rates is not None:
        return proposed, correct, rates
    if proposed is not None or correct is not None or rates is not None:
        raise ValueError("speculative response has incomplete per-position metrics")

    histogram = meta_info.get("spec_correct_drafts_histogram")
    if not isinstance(histogram, list) or any(
        type(value) is not int or value < 0 for value in histogram
    ):
        raise ValueError(
            "speculative response has neither per-position metrics nor a valid "
            "spec_correct_drafts_histogram"
        )
    if sum(histogram) != spec_verify_ct:
        raise ValueError(
            "spec_correct_drafts_histogram does not match spec_verify_ct: "
            f"histogram={histogram} spec_verify_ct={spec_verify_ct}"
        )
    rounded_length = round(float(proposed_draft_length))
    if not math.isclose(
        float(proposed_draft_length), float(rounded_length), rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError(
            "fixed-K speculative response has a non-integral proposed draft length: "
            f"{proposed_draft_length!r}"
        )
    num_positions = int(rounded_length)
    if num_positions <= 0:
        raise ValueError("fixed-K speculative response must propose at least one draft")
    if len(histogram) > num_positions + 1 and any(histogram[num_positions + 1 :]):
        raise ValueError(
            "spec_correct_drafts_histogram accepts more drafts than were proposed"
        )
    histogram = histogram[: num_positions + 1]
    proposed = [spec_verify_ct] * num_positions
    correct = [sum(histogram[position + 1 :]) for position in range(num_positions)]
    rates = [value / spec_verify_ct for value in correct]
    return proposed, correct, rates


def build_result_artifacts(
    requests: list[RequestSpec],
    outputs: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    verifier_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Create the exact requests.csv, batch.json, and content.json payloads."""

    if not requests:
        raise ValueError("client result artifacts require a non-empty batch")
    if len(outputs) != len(requests) or len(timings) != len(requests):
        raise ValueError(
            "client result cardinality mismatch: "
            f"requests={len(requests)} outputs={len(outputs)} timings={len(timings)}"
        )
    if type(verifier_rank) is not int or verifier_rank < 0:
        raise ValueError(
            f"verifier_rank must be a non-negative integer: {verifier_rank!r}"
        )

    request_rows: list[dict[str, Any]] = []
    content_rows: list[dict[str, Any]] = []
    for request, output, timing in zip(requests, outputs, timings, strict=True):
        if type(request.batch_row_index) is not int or request.batch_row_index < 0:
            raise ValueError("batch_row_index must be a non-negative integer")
        if type(request.dataset_idx) is not int or request.dataset_idx < 0:
            raise ValueError("dataset_idx must be a non-negative integer")
        input_ids = request.input_ids
        if not isinstance(input_ids, list) or any(
            type(token) is not int for token in input_ids
        ):
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid input_ids"
            )
        output_ids = output.get("output_ids")
        if not isinstance(output_ids, list) or any(
            type(token) is not int for token in output_ids
        ):
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid output_ids"
            )
        output_text = output.get("text")
        if not isinstance(output_text, str):
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid output text"
            )

        meta_info = output.get("meta_info")
        if not isinstance(meta_info, dict):
            raise ValueError(
                f"batch row {request.batch_row_index} has no final meta_info"
            )
        completion_tokens = meta_info.get("completion_tokens")
        if type(completion_tokens) is not int or completion_tokens < 0:
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid completion_tokens"
            )
        if completion_tokens != len(output_ids):
            raise ValueError(
                "output_ids length does not match completion_tokens: "
                f"batch_row_index={request.batch_row_index} "
                f"output_ids={len(output_ids)} completion_tokens={completion_tokens}"
            )
        if request.prompt_len != len(input_ids):
            raise ValueError(
                "input_ids length does not match prompt_len: "
                f"batch_row_index={request.batch_row_index} "
                f"input_ids={len(input_ids)} prompt_len={request.prompt_len}"
            )

        spec_verify_ct = meta_info.get("spec_verify_ct")
        valid_draft_len = meta_info.get("spec_proposed_draft_length")
        acc_len = meta_info.get("spec_accept_length")
        if type(spec_verify_ct) is not int or spec_verify_ct <= 0:
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid spec_verify_ct"
            )
        for name, value in (
            ("valid_draft_len", valid_draft_len),
            ("acc_len", acc_len),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(
                    f"batch row {request.batch_row_index} has invalid {name}: {value!r}"
                )
        (
            proposed_by_position,
            correct_by_position,
            accept_rate_by_position,
        ) = _normalize_position_metrics(
            meta_info,
            spec_verify_ct=spec_verify_ct,
            proposed_draft_length=valid_draft_len,
        )
        for name, values in (
            (
                "spec_num_proposed_drafts_by_position",
                proposed_by_position,
            ),
            (
                "spec_num_correct_drafts_by_position",
                correct_by_position,
            ),
        ):
            if not isinstance(values, list) or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(
                    f"batch row {request.batch_row_index} has invalid {name}"
                )
        if not isinstance(accept_rate_by_position, list) or any(
            value is not None
            and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            )
            for value in accept_rate_by_position
        ):
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid "
                "spec_accept_rate_by_position"
            )
        position_count = len(proposed_by_position)
        if (
            len(correct_by_position) != position_count
            or len(accept_rate_by_position) != position_count
        ):
            raise ValueError(
                "per-position speculative metrics disagree on length: "
                f"batch_row_index={request.batch_row_index}"
            )
        for position, (num_proposed, num_correct, accept_rate) in enumerate(
            zip(
                proposed_by_position,
                correct_by_position,
                accept_rate_by_position,
                strict=True,
            )
        ):
            if num_correct > num_proposed:
                raise ValueError(
                    "correct drafts exceed proposed drafts: "
                    f"batch_row_index={request.batch_row_index} position={position}"
                )
            expected_rate = num_correct / num_proposed if num_proposed > 0 else None
            if expected_rate is None:
                if accept_rate is not None:
                    raise ValueError(
                        "accept rate must be null when no draft was proposed: "
                        f"batch_row_index={request.batch_row_index} position={position}"
                    )
            elif not math.isclose(float(accept_rate), expected_rate, rel_tol=1e-9):
                raise ValueError(
                    "per-position accept rate is inconsistent: "
                    f"batch_row_index={request.batch_row_index} position={position}"
                )
        expected_valid_draft_len = sum(proposed_by_position) / spec_verify_ct
        if not math.isclose(
            float(valid_draft_len), expected_valid_draft_len, rel_tol=1e-9
        ):
            raise ValueError(
                f"batch row {request.batch_row_index} has inconsistent valid_draft_len"
            )
        expected_acc_len = completion_tokens / spec_verify_ct
        if not math.isclose(float(acc_len), expected_acc_len, rel_tol=1e-9):
            raise ValueError(
                f"batch row {request.batch_row_index} has inconsistent acc_len"
            )

        e2e_latency_ms = timing.get("e2e_latency_ms")
        if (
            not isinstance(e2e_latency_ms, (int, float))
            or isinstance(e2e_latency_ms, bool)
            or not math.isfinite(float(e2e_latency_ms))
            or e2e_latency_ms <= 0
        ):
            raise ValueError(
                f"batch row {request.batch_row_index} has invalid e2e latency"
            )
        e2e_latency_s = float(e2e_latency_ms) / 1000.0

        request_rows.append(
            {
                "batch_row_index": request.batch_row_index,
                "dataset_idx": request.dataset_idx,
                "verifier_rank": verifier_rank,
                "prompt_len": request.prompt_len,
                "resp_len": completion_tokens,
                "spec_verify_ct": spec_verify_ct,
                "valid_draft_len": float(valid_draft_len),
                "acc_len": float(acc_len),
                "e2e_latency_s": e2e_latency_s,
                "spec_num_proposed_drafts_by_position": list(proposed_by_position),
                "spec_num_correct_drafts_by_position": list(correct_by_position),
                "spec_accept_rate_by_position": list(accept_rate_by_position),
            }
        )
        content_rows.append(
            {
                "batch_row_idx": request.batch_row_index,
                "dataset_idx": request.dataset_idx,
                "input_len": request.prompt_len,
                "output_len": completion_tokens,
                "input_ids": list(input_ids),
                "input_text": request.rendered_prompt,
                "output_ids": list(output_ids),
                "output_text": output_text,
            }
        )

    request_rows.sort(key=lambda row: row["batch_row_index"])
    content_rows.sort(key=lambda row: row["batch_row_idx"])
    expected_indices = list(range(len(requests)))
    if [row["batch_row_index"] for row in request_rows] != expected_indices:
        raise ValueError("requests.csv batch_row_index must be contiguous from zero")
    if [row["batch_row_idx"] for row in content_rows] != expected_indices:
        raise ValueError("content.json batch_row_idx must be contiguous from zero")

    output_tokens = sum(row["resp_len"] for row in request_rows)
    batch_elapsed_latency_s = max(row["e2e_latency_s"] for row in request_rows)
    batch_result = {
        "output_tokens": output_tokens,
        "batch_elapsed_latency_s": batch_elapsed_latency_s,
        "batch_thpt": output_tokens / batch_elapsed_latency_s,
        "mean_valid_draft_len": statistics.fmean(
            row["valid_draft_len"] for row in request_rows
        ),
        "acclen": statistics.fmean(row["acc_len"] for row in request_rows),
    }
    return request_rows, batch_result, content_rows


def write_result_artifacts(
    client_dir: str | Path,
    request_rows: list[dict[str, Any]],
    batch_result: dict[str, Any],
    content_rows: list[dict[str, Any]],
) -> Path:
    """Write exactly three human-facing files directly under client/."""

    if any(tuple(row) != REQUESTS_CSV_FIELDS for row in request_rows):
        raise ValueError("requests.csv rows do not match the fixed field contract")
    if tuple(batch_result) != BATCH_JSON_FIELDS:
        raise ValueError("batch.json does not match the fixed field contract")
    if any(tuple(row) != CONTENT_JSON_FIELDS for row in content_rows):
        raise ValueError("content.json rows do not match the fixed field contract")

    output_dir = Path(client_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    requests_path = output_dir / "requests.csv"
    temporary = requests_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUESTS_CSV_FIELDS)
        writer.writeheader()
        for row in request_rows:
            serialized = dict(row)
            for field in _ARRAY_CSV_FIELDS:
                serialized[field] = json.dumps(
                    row[field], ensure_ascii=False, separators=(",", ":")
                )
            writer.writerow(serialized)
    temporary.replace(requests_path)
    write_json(output_dir / "batch.json", batch_result)
    write_json(output_dir / "content.json", content_rows)
    return output_dir

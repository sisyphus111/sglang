#!/usr/bin/env python3
"""Build a cross-run report from verified, sealed matrix attempts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from campaign import (
    _find_case,
    campaign_status,
    load_campaign,
    validate_verified_run,
)

_CONTENT_SNIPPET_CHARS = 96
_MAX_REPLACEMENT_CHARACTER_RATIO = 0.01
_BINDING_ISSUES = {
    "missing_sampled_request",
    "missing_raw_response",
    "missing_response_record",
    "invalid_sampled_request_id",
    "invalid_sampled_row_index",
    "invalid_input_ids",
    "missing_meta_info",
    "raw_request_id_mismatch",
    "source_row_index_mismatch",
    "response_row_index_mismatch",
    "response_prompt_len_mismatch",
    "generated_text_mismatch",
    "input_binding_mismatch_across_runs",
}
_LENGTH_ISSUES = {
    "requested_output_len_mismatch",
    "invalid_output_ids",
    "output_ids_length_mismatch",
    "completion_tokens_mismatch",
    "finish_reason_not_length",
    "response_completion_tokens_mismatch",
    "response_length_mismatch",
}
_TEXT_ISSUES = {
    "missing_detokenized_text",
    "invalid_utf8_text",
    "empty_detokenized_text",
    "nul_in_detokenized_text",
    "excessive_replacement_characters",
    "generated_text_mismatch",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _text_snippet(text: str, *, tail: bool = False) -> str:
    fragment = text[-_CONTENT_SNIPPET_CHARS:] if tail else text[:_CONTENT_SNIPPET_CHARS]
    return " ".join(fragment.split())


def _finish_reason_is_length(value: Any, output_len: int) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "length"
        and type(value.get("length")) is int
        and value["length"] == output_len
    )


def _matches_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _refresh_content_sanity(result: dict[str, Any]) -> None:
    requests = result["requests"]
    result["healthy_request_count"] = sum(not request["issues"] for request in requests)
    result["issue_count"] = len(result["issues"])
    result["status"] = "pass" if not result["issues"] else "review"


def assess_run_content(
    run_dir: str | Path, batch_size: int, output_len: int
) -> dict[str, Any]:
    """Check saved request/output binding and coarse text health without judging it."""

    run_path = Path(run_dir).expanduser().resolve()
    sampled = _read_jsonl(run_path / "client" / "sampled_requests.jsonl")
    raw_payload = json.loads(
        (run_path / "client" / "raw_batch_response.json").read_text(encoding="utf-8")
    )
    responses = _read_jsonl(run_path / "client" / "responses.jsonl")
    if not isinstance(raw_payload, list):
        raise ValueError("client/raw_batch_response.json must contain a list")

    issues = []
    if len(sampled) != batch_size:
        issues.append("sampled_request_count_mismatch")
    if len(raw_payload) != batch_size:
        issues.append("raw_response_count_mismatch")
    if len(responses) != batch_size:
        issues.append("response_record_count_mismatch")

    raw_by_index: dict[int, dict[str, Any]] = {}
    for raw in raw_payload:
        if not isinstance(raw, dict) or type(raw.get("index")) is not int:
            issues.append("invalid_raw_response_index")
            continue
        index = raw["index"]
        if not 0 <= index < batch_size:
            issues.append("raw_response_index_out_of_range")
            continue
        if index in raw_by_index:
            issues.append("duplicate_raw_response_index")
            continue
        raw_by_index[index] = raw

    response_by_request_id: dict[str, dict[str, Any]] = {}
    for response in responses:
        request_id = response.get("request_id")
        if not isinstance(request_id, str):
            issues.append("invalid_response_request_id")
            continue
        if request_id in response_by_request_id:
            issues.append("duplicate_response_request_id")
            continue
        response_by_request_id[request_id] = response

    sampled_request_ids = [row.get("request_id") for row in sampled]
    sampled_row_indices = [row.get("row_index") for row in sampled]
    if len(set(map(str, sampled_request_ids))) != len(sampled_request_ids):
        issues.append("duplicate_sampled_request_id")
    if len(set(map(str, sampled_row_indices))) != len(sampled_row_indices):
        issues.append("duplicate_sampled_row_index")

    request_results = []
    for index in range(batch_size):
        request_issues = []
        sample = sampled[index] if index < len(sampled) else None
        raw = raw_by_index.get(index)

        if sample is None:
            request_issues.append("missing_sampled_request")
            sample = {}
        if raw is None:
            request_issues.append("missing_raw_response")
            raw = {}

        request_id = sample.get("request_id")
        row_index = sample.get("row_index")
        response = (
            response_by_request_id.get(request_id)
            if isinstance(request_id, str)
            else None
        )
        if not isinstance(request_id, str):
            request_issues.append("invalid_sampled_request_id")
        if type(row_index) is not int:
            request_issues.append("invalid_sampled_row_index")
        if response is None:
            request_issues.append("missing_response_record")
            response = {}

        input_ids = sample.get("input_ids")
        if not isinstance(input_ids, list) or any(
            type(token_id) is not int for token_id in input_ids
        ):
            request_issues.append("invalid_input_ids")
            input_ids = None
        output_ids = raw.get("output_ids")
        if not isinstance(output_ids, list) or any(
            type(token_id) is not int for token_id in output_ids
        ):
            request_issues.append("invalid_output_ids")
            output_ids = None

        if not _matches_int(sample.get("requested_output_len"), output_len):
            request_issues.append("requested_output_len_mismatch")
        if output_ids is None or len(output_ids) != output_len:
            request_issues.append("output_ids_length_mismatch")

        meta_info = raw.get("meta_info")
        if not isinstance(meta_info, dict):
            request_issues.append("missing_meta_info")
            meta_info = {}
        if meta_info.get("id") != request_id:
            request_issues.append("raw_request_id_mismatch")
        if not _matches_int(meta_info.get("completion_tokens"), output_len):
            request_issues.append("completion_tokens_mismatch")
        if not _finish_reason_is_length(meta_info.get("finish_reason"), output_len):
            request_issues.append("finish_reason_not_length")

        source = sample.get("source")
        if isinstance(source, dict) and source.get("row_index") != row_index:
            request_issues.append("source_row_index_mismatch")
        if response.get("row_index") != row_index:
            request_issues.append("response_row_index_mismatch")
        if response.get("prompt_len") != sample.get("prompt_len"):
            request_issues.append("response_prompt_len_mismatch")
        if not _matches_int(response.get("completion_tokens"), output_len):
            request_issues.append("response_completion_tokens_mismatch")
        if not _matches_int(response.get("resp_len"), output_len):
            request_issues.append("response_length_mismatch")

        text = raw.get("text")
        text_bytes = None
        nul_count = 0
        replacement_count = 0
        replacement_ratio = 0.0
        if not isinstance(text, str):
            request_issues.append("missing_detokenized_text")
            text = ""
        else:
            try:
                text_bytes = text.encode("utf-8")
            except UnicodeEncodeError:
                request_issues.append("invalid_utf8_text")
            if not text.strip():
                request_issues.append("empty_detokenized_text")
            nul_count = text.count("\x00")
            replacement_count = text.count("\ufffd")
            replacement_ratio = replacement_count / len(text) if text else 0.0
            if nul_count:
                request_issues.append("nul_in_detokenized_text")
            if replacement_ratio > _MAX_REPLACEMENT_CHARACTER_RATIO:
                request_issues.append("excessive_replacement_characters")
        if response.get("generated_text") != text:
            request_issues.append("generated_text_mismatch")

        binding_value = {
            "request_id": request_id,
            "row_index": row_index,
            "prompt_len": sample.get("prompt_len"),
            "source": source,
            "input_ids_sha256": (
                _json_sha256(input_ids) if input_ids is not None else None
            ),
        }
        request_result = {
            "request_index": index,
            "request_id": request_id,
            "row_index": row_index,
            "input_ids_len": len(input_ids) if input_ids is not None else None,
            "input_ids_sha256": binding_value["input_ids_sha256"],
            "input_binding_sha256": _json_sha256(binding_value),
            "output_ids_len": len(output_ids) if output_ids is not None else None,
            "output_ids_sha256": (
                _json_sha256(output_ids) if output_ids is not None else None
            ),
            "text_chars": len(text),
            "text_utf8_bytes": len(text_bytes) if text_bytes is not None else None,
            "text_sha256": (
                hashlib.sha256(text_bytes).hexdigest()
                if text_bytes is not None
                else None
            ),
            "text_head": _text_snippet(text),
            "text_tail": _text_snippet(text, tail=True),
            "nul_count": nul_count,
            "replacement_character_count": replacement_count,
            "replacement_character_ratio": replacement_ratio,
            "binding_ok": not bool(set(request_issues) & _BINDING_ISSUES),
            "length_ok": not bool(set(request_issues) & _LENGTH_ISSUES),
            "text_ok": not bool(set(request_issues) & _TEXT_ISSUES),
            "issues": request_issues,
        }
        request_results.append(request_result)
        issues.extend(f"request[{index}]:{issue}" for issue in request_issues)

    text_hashes = [request["text_sha256"] for request in request_results]
    binding_hashes = [request["input_binding_sha256"] for request in request_results]
    result = {
        "schema_version": 1,
        "replacement_character_ratio_threshold": _MAX_REPLACEMENT_CHARACTER_RATIO,
        "request_count": len(request_results),
        "healthy_request_count": 0,
        "text_sha256": _json_sha256(text_hashes),
        "input_binding_sha256": _json_sha256(binding_hashes),
        "requests": request_results,
        "issues": issues,
    }
    _refresh_content_sanity(result)
    return result


def _prompt_lengths(run_dir: Path) -> list[int]:
    lengths = []
    with (run_dir / "client" / "sampled_requests.jsonl").open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            if line.strip():
                lengths.append(int(json.loads(line)["prompt_len"]))
    return lengths


def _output_ids_by_index(run_dir: Path) -> dict[int, list[int]]:
    path = run_dir / "client" / "raw_batch_response.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a response list: {path}")
    rows: dict[int, list[int]] = {}
    for response in payload:
        if not isinstance(response, dict):
            raise ValueError(f"invalid response row: {path}")
        index = int(response["index"])
        if index in rows:
            raise ValueError(f"duplicate response index {index}: {path}")
        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list):
            raise ValueError(f"response {index} has no output_ids list: {path}")
        rows[index] = [int(token_id) for token_id in output_ids]
    return rows


def _verified_attempts(ledger_case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        attempt
        for attempt in ledger_case.get("attempts", [])
        if attempt.get("state") == "verified"
    ]


def collect_verified_rows(
    campaign_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, manifest, ledger = load_campaign(campaign_dir)
    rows = []
    sources = []
    seen_run_dirs: set[Path] = set()
    for case in manifest["cases"]:
        attempts = _verified_attempts(ledger["cases"][case["case_id"]])
        if not attempts:
            continue
        attempt = attempts[-1]
        run_dir = Path(attempt["run_dir"]).expanduser().resolve()
        if run_dir in seen_run_dirs:
            raise ValueError(f"multiple cases reference the same RUN_DIR: {run_dir}")
        seen_run_dirs.add(run_dir)
        verification = validate_verified_run(manifest, case, run_dir)
        if not verification["ok"]:
            raise ValueError(
                f"verified case {case['case_id']} no longer passes sealed audit: "
                + "; ".join(verification["errors"])
            )
        summary = verification["summary"]
        prompt_lengths = _prompt_lengths(run_dir)
        if len(prompt_lengths) != case["batch_size"]:
            raise ValueError(
                f"{case['case_id']}: prompt count does not match batch size"
            )
        content_sanity = assess_run_content(
            run_dir, int(case["batch_size"]), int(case["output_len"])
        )
        first_content = content_sanity["requests"][0]
        last_content = content_sanity["requests"][-1]
        diagnostics = []
        if content_sanity["status"] != "pass":
            diagnostics.append("content_sanity_failed")
        row = {
            "case_id": case["case_id"],
            "mode": case["mode"],
            "batch_size": case["batch_size"],
            "output_len": case["output_len"],
            "run_dir": str(run_dir),
            "prompt_len_min": min(prompt_lengths),
            "prompt_len_max": max(prompt_lengths),
            "prompt_len_mean": statistics.fmean(prompt_lengths),
            "prompt_len_sum": sum(prompt_lengths),
            "batch_elapsed_s": summary.get("batch_elapsed_s"),
            "completion_tokens": summary.get("completion_tokens"),
            "output_tokens_per_s": summary.get("output_tokens_per_s"),
            "ttft_mean_ms": summary.get("ttft_ms", {}).get("mean"),
            "tpot_mean_ms": summary.get("tpot_ms", {}).get("mean"),
            "e2e_mean_ms": summary.get("e2e_latency_ms", {}).get("mean"),
            "spec_verify_ct": summary.get("spec_verify_ct"),
            "spec_num_proposed_drafts": summary.get("spec_num_proposed_drafts"),
            "spec_num_correct_drafts": summary.get("spec_num_correct_drafts"),
            "spec_accept_rate": summary.get("spec_accept_rate"),
            "spec_accept_length": summary.get("spec_accept_length"),
            "spec_proposed_draft_length": summary.get("spec_proposed_draft_length"),
            "spec_draft_occupancy_rate": summary.get("spec_draft_occupancy_rate"),
            "content_sanity_status": content_sanity["status"],
            "content_sanity_issue_count": content_sanity["issue_count"],
            "content_sanity_healthy_request_count": content_sanity[
                "healthy_request_count"
            ],
            "content_text_sha256": content_sanity["text_sha256"],
            "content_input_binding_sha256": content_sanity["input_binding_sha256"],
            "content_first_text_head": first_content["text_head"],
            "content_first_text_tail": first_content["text_tail"],
            "content_last_text_head": last_content["text_head"],
            "content_last_text_tail": last_content["text_tail"],
            "content_sanity": content_sanity,
            "quality_status": "pass",
            "diagnostics": diagnostics,
        }
        rows.append(row)
        sources.append(
            {
                "case_id": case["case_id"],
                "run_dir": str(run_dir),
                "sha256sums_sha256": _sha256(run_dir / "SHA256SUMS"),
                "client_summary_sha256": _sha256(run_dir / "client" / "summary.json"),
                "client_raw_batch_response_sha256": _sha256(
                    run_dir / "client" / "raw_batch_response.json"
                ),
                "client_sampled_requests_sha256": _sha256(
                    run_dir / "client" / "sampled_requests.jsonl"
                ),
                "client_responses_sha256": _sha256(
                    run_dir / "client" / "responses.jsonl"
                ),
                "case_sha256": case["case_sha256"],
            }
        )
    return manifest, rows, sources


def apply_output_correctness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record fixed-seed output drift without treating it as deterministic parity."""

    by_key = {(row["mode"], row["batch_size"], row["output_len"]): row for row in rows}
    mismatches = []
    for row in rows:
        if row["mode"] != "overlap":
            continue
        baseline = by_key.get(("nonoverlap", row["batch_size"], row["output_len"]))
        if baseline is None:
            continue
        baseline_outputs = _output_ids_by_index(Path(baseline["run_dir"]))
        overlap_outputs = _output_ids_by_index(Path(row["run_dir"]))
        first_diffs = []
        for request_index in sorted(set(baseline_outputs) | set(overlap_outputs)):
            left = baseline_outputs.get(request_index)
            right = overlap_outputs.get(request_index)
            if left is None or right is None:
                first_diffs.append(
                    {
                        "request_index": request_index,
                        "position": 0,
                        "nonoverlap_token": None if left is None else left[0],
                        "overlap_token": None if right is None else right[0],
                    }
                )
                continue
            common = min(len(left), len(right))
            position = next(
                (index for index in range(common) if left[index] != right[index]),
                common if len(left) != len(right) else None,
            )
            if position is not None:
                first_diffs.append(
                    {
                        "request_index": request_index,
                        "position": position,
                        "nonoverlap_token": (
                            left[position] if position < len(left) else None
                        ),
                        "overlap_token": (
                            right[position] if position < len(right) else None
                        ),
                    }
                )
        exact = not first_diffs
        row["output_ids_exact_vs_nonoverlap"] = exact
        row["output_ids_first_diffs"] = first_diffs
        if not exact:
            mismatches.append(
                {
                    "batch_size": row["batch_size"],
                    "output_len": row["output_len"],
                    "first_diffs": first_diffs,
                }
            )
    return mismatches


def apply_input_binding_consistency(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require the same sampled request index to bind to the same input in every run."""

    by_request_index: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row in rows:
        for request in row["content_sanity"]["requests"]:
            by_request_index.setdefault(request["request_index"], []).append(
                (row, request)
            )

    mismatches = []
    for request_index, observations in sorted(by_request_index.items()):
        counts = Counter(request["input_binding_sha256"] for _, request in observations)
        expected_hash = sorted(
            counts.items(), key=lambda item: (-item[1], str(item[0]))
        )[0][0]
        expected_case = next(
            row["case_id"]
            for row, request in observations
            if request["input_binding_sha256"] == expected_hash
        )
        for row, request in observations:
            observed_hash = request["input_binding_sha256"]
            if observed_hash == expected_hash:
                continue
            issue = "input_binding_mismatch_across_runs"
            if issue not in request["issues"]:
                request["issues"].append(issue)
                request["binding_ok"] = False
                row["content_sanity"]["issues"].append(
                    f"request[{request_index}]:{issue}"
                )
                _refresh_content_sanity(row["content_sanity"])
                row["content_sanity_status"] = row["content_sanity"]["status"]
                row["content_sanity_issue_count"] = row["content_sanity"]["issue_count"]
                row["content_sanity_healthy_request_count"] = row["content_sanity"][
                    "healthy_request_count"
                ]
            if "content_sanity_failed" not in row["diagnostics"]:
                row["diagnostics"].append("content_sanity_failed")
            mismatches.append(
                {
                    "case_id": row["case_id"],
                    "request_index": request_index,
                    "expected_case_id": expected_case,
                    "expected_input_binding_sha256": expected_hash,
                    "observed_input_binding_sha256": observed_hash,
                }
            )
    return mismatches


def apply_diagnostics(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    diagnostic = manifest["stop_gates"].get("diagnostic", {})
    accept_floor = float(diagnostic.get("min_spec_accept_rate", 0))
    occupancy_floor = float(diagnostic.get("min_spec_draft_occupancy_rate", 0))
    max_overlap_drop = float(
        diagnostic.get("max_overlap_accept_rate_drop_vs_nonoverlap", 1)
    )
    require_proposals = bool(diagnostic.get("require_nonzero_spec_proposals", False))
    by_key = {(row["mode"], row["batch_size"], row["output_len"]): row for row in rows}
    for row in rows:
        proposed_drafts = row.get("spec_num_proposed_drafts")
        if (
            require_proposals
            and proposed_drafts is not None
            and int(proposed_drafts) <= 0
        ):
            row["diagnostics"].append("no_draft_proposals")
        if (
            row["spec_accept_rate"] is not None
            and float(row["spec_accept_rate"]) < accept_floor
        ):
            row["diagnostics"].append(f"accept_rate<{accept_floor:.3f}")
        if float(row["spec_draft_occupancy_rate"]) < occupancy_floor:
            row["diagnostics"].append(f"draft_occupancy<{occupancy_floor:.3f}")
        if row["mode"] == "overlap":
            baseline = by_key.get(("nonoverlap", row["batch_size"], row["output_len"]))
            if baseline is not None:
                row["throughput_ratio_vs_nonoverlap"] = float(
                    row["output_tokens_per_s"]
                ) / float(baseline["output_tokens_per_s"])
                if (
                    row["spec_accept_rate"] is not None
                    and baseline["spec_accept_rate"] is not None
                ):
                    row["accept_rate_delta_vs_nonoverlap"] = float(
                        row["spec_accept_rate"]
                    ) - float(baseline["spec_accept_rate"])
                row["occupancy_delta_vs_nonoverlap"] = float(
                    row["spec_draft_occupancy_rate"]
                ) - float(baseline["spec_draft_occupancy_rate"])
                if (
                    "accept_rate_delta_vs_nonoverlap" in row
                    and row["accept_rate_delta_vs_nonoverlap"] < -max_overlap_drop
                ):
                    row["diagnostics"].append(
                        f"overlap_accept_rate_drop>{max_overlap_drop:.3f}"
                    )
        if row["diagnostics"]:
            row["quality_status"] = "review"


def build_summary(campaign_dir: str | Path, require_complete: bool) -> dict[str, Any]:
    campaign_path, _, _ = load_campaign(campaign_dir)
    manifest, rows, sources = collect_verified_rows(campaign_path)
    input_binding_mismatches = apply_input_binding_consistency(rows)
    output_mismatches = apply_output_correctness(rows)
    apply_diagnostics(manifest, rows)
    status = campaign_status(campaign_path)
    if require_complete and len(rows) != int(manifest["case_count"]):
        raise ValueError(
            f"matrix is incomplete: verified={len(rows)}/{manifest['case_count']}"
        )
    return {
        "schema_version": 2,
        "generated_at": time.time(),
        "campaign": manifest["campaign"],
        "campaign_contract_sha256": manifest["contract_sha256"],
        "case_count": manifest["case_count"],
        "verified_case_count": len(rows),
        "complete": len(rows) == int(manifest["case_count"]),
        "content_sanity_pass_count": sum(
            row["content_sanity_status"] == "pass" for row in rows
        ),
        "content_sanity_review_count": sum(
            row["content_sanity_status"] != "pass" for row in rows
        ),
        "input_binding_mismatch_count": len(input_binding_mismatches),
        "input_binding_mismatches": input_binding_mismatches,
        "paired_output_comparison_is_informational": True,
        "paired_output_mismatch_count": len(output_mismatches),
        "paired_output_mismatches": output_mismatches,
        "status_counts": status["counts"],
        "stop_gates": manifest["stop_gates"],
        "rows": rows,
        "sources": sources,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "mode",
        "batch_size",
        "output_len",
        "prompt_len_min",
        "prompt_len_max",
        "prompt_len_mean",
        "prompt_len_sum",
        "batch_elapsed_s",
        "completion_tokens",
        "output_tokens_per_s",
        "ttft_mean_ms",
        "tpot_mean_ms",
        "e2e_mean_ms",
        "spec_verify_ct",
        "spec_num_proposed_drafts",
        "spec_num_correct_drafts",
        "spec_accept_rate",
        "spec_accept_length",
        "spec_proposed_draft_length",
        "spec_draft_occupancy_rate",
        "content_sanity_status",
        "content_sanity_issue_count",
        "content_sanity_healthy_request_count",
        "content_text_sha256",
        "content_input_binding_sha256",
        "content_first_text_head",
        "content_first_text_tail",
        "content_last_text_head",
        "content_last_text_tail",
        "throughput_ratio_vs_nonoverlap",
        "accept_rate_delta_vs_nonoverlap",
        "occupancy_delta_vs_nonoverlap",
        "output_ids_exact_vs_nonoverlap",
        "quality_status",
        "diagnostics",
        "run_dir",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {field: row.get(field) for field in fields}
            value["diagnostics"] = ";".join(row["diagnostics"])
            writer.writerow(value)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_cell(value: Any) -> str:
    return html.escape(str(value), quote=False).replace("|", "&#124;")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    review_count = sum(row["quality_status"] == "review" for row in rows)
    lines = [
        "# Decoupled-Spec ReplaySSM 扩展矩阵结果",
        "",
        (
            f"当前读取了 **{summary['verified_case_count']}/{summary['case_count']}** "
            f"个 sealed + verified case；其中 {review_count} 个触发诊断阈值。"
        ),
        "只有通过 sealed audit、checksum 和 exact-config gate 的 RUN_DIR 会进入本报告；"
        "失败或未封存 attempt 只保留在 ledger，不参与聚合。",
        "",
        "## 关键图表",
        "",
        "![Throughput by batch](plots/throughput_by_batch.png)",
        "",
        "![Paired overlap effects](plots/paired_overlap_effects.png)",
        "",
        "## Case 结果",
        "",
        "| Mode | BS | Output | tok/s | Accept rate | Draft occupancy | Proposed draft length | Accept length | Content | Quality | RUN_DIR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["mode"],
                    str(row["batch_size"]),
                    str(row["output_len"]),
                    _fmt(row["output_tokens_per_s"]),
                    _fmt(row["spec_accept_rate"]),
                    _fmt(row["spec_draft_occupancy_rate"]),
                    _fmt(row["spec_proposed_draft_length"]),
                    _fmt(row["spec_accept_length"]),
                    row["content_sanity_status"],
                    row["quality_status"],
                    f"`{row['run_dir']}`",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired overlap 对照",
            "",
            "| BS | Output | Output IDs exact (info) | Throughput ratio | Accept-rate delta | Occupancy delta | Diagnostics |",
            "| ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if row["mode"] != "overlap" or "throughput_ratio_vs_nonoverlap" not in row:
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["batch_size"]),
                    str(row["output_len"]),
                    str(row.get("output_ids_exact_vs_nonoverlap", "-")).lower(),
                    _fmt(row["throughput_ratio_vs_nonoverlap"]),
                    _fmt(row.get("accept_rate_delta_vs_nonoverlap")),
                    _fmt(row["occupancy_delta_vs_nonoverlap"]),
                    ", ".join(row["diagnostics"]) or "-",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Content sanity",
            "",
            "每个 case 的完整逐请求 hash、首尾 snippet 和 issue code 位于 "
            "`matrix_summary.json` 的 `content_sanity.requests`。这里展示 case 级 "
            "SHA-256 前缀、首请求开头和末请求结尾，便于快速人工抽查。",
            "",
            "| Mode | BS | Output | Status | Healthy | Text SHA-256 | First request head | Last request tail |",
            "| --- | ---: | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["mode"],
                    str(row["batch_size"]),
                    str(row["output_len"]),
                    row["content_sanity_status"],
                    (
                        f"{row['content_sanity_healthy_request_count']}/"
                        f"{row['batch_size']}"
                    ),
                    row["content_text_sha256"][:16],
                    _md_cell(row["content_first_text_head"]),
                    _md_cell(row["content_last_text_tail"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "`spec_accept_rate` 只衡量实际提出 token 的正确率；"
            "`spec_draft_occupancy_rate` 衡量 draft 供给占 K-step nominal capacity 的比例。"
            "两者必须分开解释。HTTP streaming latency 和服务级 observability 不能分解"
            "单轮 CUDA stream、ZMQ 或 H2D 时延。性能矩阵没有开启 deterministic inference；"
            "因此 paired output-ID exactness 只作为 informational drift 记录，不改变 quality。"
            "Content sanity 只检查保存产物的 request/input 绑定、固定长度和粗粒度文本健康，"
            "不重新加载 tokenizer 解码，也不判断数学答案或生成语义是否正确。"
            "所有诊断阈值都不删除或改写有效测量。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    campaign_dir: str | Path, require_complete: bool = False
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).expanduser().resolve()
    summary = build_summary(campaign_path, require_complete)
    output_dir = campaign_path / "summary"
    json_path = output_dir / "matrix_summary.json"
    csv_path = output_dir / "matrix_summary.csv"
    markdown_path = output_dir / "matrix_report.md"
    _write_json(json_path, summary)
    _write_csv(csv_path, summary["rows"])
    from plot_summary import write_matrix_plots

    plot_outputs = write_matrix_plots(json_path)
    _write_markdown(markdown_path, summary)
    manifest_path = output_dir / "manifest.json"
    source_paths = [
        campaign_path / "campaign_manifest.json",
        campaign_path / "ledger.json",
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": time.time(),
            "sources": [
                {"path": str(path), "sha256": _sha256(path)} for path in source_paths
            ]
            + summary["sources"],
            "outputs": [
                str(json_path),
                str(csv_path),
                str(markdown_path),
                *[str(path) for path in plot_outputs],
            ],
            "sealed_runs_only": True,
        },
    )
    return {
        "summary_dir": str(output_dir),
        "verified_case_count": summary["verified_case_count"],
        "case_count": summary["case_count"],
        "complete": summary["complete"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            write_summary(args.campaign_dir, args.require_complete),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

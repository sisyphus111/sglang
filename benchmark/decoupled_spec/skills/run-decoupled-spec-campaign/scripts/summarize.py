#!/usr/bin/env python3
"""Build a cross-run report from completed campaign attempts."""

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
    _read_client_result_metrics,
    _find_case,
    campaign_status,
    load_campaign,
)

_CONTENT_SNIPPET_CHARS = 96
_MAX_REPLACEMENT_CHARACTER_RATIO = 0.01
_BINDING_ISSUES = {
    "missing_requests_csv_row",
    "missing_content_row",
    "batch_index_mismatch",
    "dataset_idx_mismatch",
    "invalid_input_ids",
    "invalid_input_text",
    "input_len_mismatch",
    "input_binding_mismatch_across_runs",
}
_LENGTH_ISSUES = {
    "invalid_output_ids",
    "output_len_mismatch",
    "output_length_out_of_range",
}
_TEXT_ISSUES = {
    "missing_detokenized_text",
    "invalid_utf8_text",
    "empty_detokenized_text",
    "nul_in_detokenized_text",
    "excessive_replacement_characters",
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


def _refresh_content_sanity(result: dict[str, Any]) -> None:
    requests = result["requests"]
    result["healthy_request_count"] = sum(not request["issues"] for request in requests)
    result["issue_count"] = len(result["issues"])
    result["status"] = "pass" if not result["issues"] else "review"


def assess_run_content(
    run_dir: str | Path, batch_size: int, output_len: int, ignore_eos: bool = True
) -> dict[str, Any]:
    """Check fixed client content binding and coarse text health without judging it."""

    run_path = Path(run_dir).expanduser().resolve()
    with (run_path / "client" / "requests.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        request_rows = list(csv.DictReader(stream))
    content = json.loads(
        (run_path / "client" / "content.json").read_text(encoding="utf-8")
    )
    if not isinstance(content, list):
        raise ValueError("client/content.json must contain a list")

    issues = []
    if len(request_rows) != batch_size:
        issues.append("requests_csv_count_mismatch")
    if len(content) != batch_size:
        issues.append("content_count_mismatch")

    request_results = []
    for index in range(batch_size):
        request_issues = []
        metrics = request_rows[index] if index < len(request_rows) else None
        item = content[index] if index < len(content) else None
        if metrics is None:
            request_issues.append("missing_requests_csv_row")
            metrics = {}
        if not isinstance(item, dict):
            request_issues.append("missing_content_row")
            item = {}

        try:
            metrics_index = int(metrics.get("batch_row_index", -1))
            metrics_dataset_idx = int(metrics.get("dataset_idx", -1))
            prompt_len = int(metrics.get("prompt_len", -1))
            response_len = int(metrics.get("resp_len", -1))
        except (TypeError, ValueError):
            metrics_index = metrics_dataset_idx = prompt_len = response_len = -1
            request_issues.append("invalid_requests_csv_row")
        content_index = item.get("batch_row_idx")
        dataset_idx = item.get("dataset_idx")
        if metrics_index != index or content_index != index:
            request_issues.append("batch_index_mismatch")
        if metrics_dataset_idx != dataset_idx:
            request_issues.append("dataset_idx_mismatch")

        input_ids = item.get("input_ids")
        if not isinstance(input_ids, list) or any(
            type(token_id) is not int for token_id in input_ids
        ):
            request_issues.append("invalid_input_ids")
            input_ids = None
        output_ids = item.get("output_ids")
        if not isinstance(output_ids, list) or any(
            type(token_id) is not int for token_id in output_ids
        ):
            request_issues.append("invalid_output_ids")
            output_ids = None
        if (
            input_ids is None
            or item.get("input_len") != prompt_len
            or len(input_ids) != prompt_len
        ):
            request_issues.append("input_len_mismatch")
        if (
            output_ids is None
            or item.get("output_len") != response_len
            or len(output_ids) != response_len
        ):
            request_issues.append("output_len_mismatch")
        if ignore_eos and response_len != output_len:
            request_issues.append("output_length_out_of_range")
        if not ignore_eos and not 0 < response_len <= output_len:
            request_issues.append("output_length_out_of_range")

        input_text = item.get("input_text")
        if not isinstance(input_text, str):
            request_issues.append("invalid_input_text")
            input_text = ""
        text = item.get("output_text")
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

        binding_value = {
            "batch_row_idx": index,
            "dataset_idx": dataset_idx,
            "input_len": item.get("input_len"),
            "input_ids_sha256": (
                _json_sha256(input_ids) if input_ids is not None else None
            ),
            "input_text_sha256": _json_sha256(input_text),
        }
        request_result = {
            "request_index": index,
            "dataset_idx": dataset_idx,
            "input_ids_len": len(input_ids) if input_ids is not None else None,
            "input_ids_sha256": binding_value["input_ids_sha256"],
            "input_binding_sha256": _json_sha256(binding_value),
            "output_ids_len": len(output_ids) if output_ids is not None else None,
            "output_ids_sha256": (
                _json_sha256(output_ids) if output_ids is not None else None
            ),
            "completion_tokens": response_len,
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
        "schema_version": 2,
        "replacement_character_ratio_threshold": _MAX_REPLACEMENT_CHARACTER_RATIO,
        "request_count": len(request_results),
        "healthy_request_count": 0,
        "text_sha256": _json_sha256(text_hashes),
        "input_binding_sha256": _json_sha256(binding_hashes),
        "ignore_eos": ignore_eos,
        "completion_tokens_min": min(
            request["completion_tokens"] for request in request_results
        ),
        "completion_tokens_max": max(
            request["completion_tokens"] for request in request_results
        ),
        "completion_tokens_mean": statistics.fmean(
            request["completion_tokens"] for request in request_results
        ),
        "requests": request_results,
        "issues": issues,
    }
    _refresh_content_sanity(result)
    return result


def _prompt_lengths(run_dir: Path) -> list[int]:
    with (run_dir / "client" / "requests.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        return [int(row["prompt_len"]) for row in csv.DictReader(stream)]


def _output_ids_by_index(run_dir: Path) -> dict[int, list[int]]:
    path = run_dir / "client" / "content.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a content list: {path}")
    rows: dict[int, list[int]] = {}
    for content in payload:
        if not isinstance(content, dict):
            raise ValueError(f"invalid content row: {path}")
        index = int(content["batch_row_idx"])
        if index in rows:
            raise ValueError(f"duplicate content index {index}: {path}")
        output_ids = content.get("output_ids")
        if not isinstance(output_ids, list):
            raise ValueError(f"content {index} has no output_ids list: {path}")
        rows[index] = [int(token_id) for token_id in output_ids]
    return rows


def _completed_attempts(ledger_case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        attempt
        for attempt in ledger_case.get("attempts", [])
        if attempt.get("state") == "completed"
    ]


def collect_completed_rows(
    campaign_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _, manifest, ledger = load_campaign(campaign_dir)
    rows = []
    sources = []
    seen_run_dirs: set[Path] = set()
    for case in manifest["cases"]:
        attempts = _completed_attempts(ledger["cases"][case["case_id"]])
        if not attempts:
            continue
        attempt = attempts[-1]
        run_dir = Path(attempt["run_dir"]).expanduser().resolve()
        if run_dir in seen_run_dirs:
            raise ValueError(f"multiple cases reference the same RUN_DIR: {run_dir}")
        seen_run_dirs.add(run_dir)
        summary = _read_client_result_metrics(
            run_dir,
            max_proposed_drafts=int(manifest["expected"]["speculative_num_steps"]),
        )
        prompt_lengths = _prompt_lengths(run_dir)
        if len(prompt_lengths) != case["batch_size"]:
            raise ValueError(
                f"{case['case_id']}: prompt count does not match batch size"
            )
        content_sanity = assess_run_content(
            run_dir,
            int(case["batch_size"]),
            int(case["output_len"]),
            bool(case["ignore_eos"]),
        )
        first_content = content_sanity["requests"][0]
        last_content = content_sanity["requests"][-1]
        diagnostics = []
        if content_sanity["status"] != "pass":
            diagnostics.append("content_sanity_failed")
        row = {
            "case_id": case["case_id"],
            "mode": case["mode"],
            "ignore_eos": case["ignore_eos"],
            "batch_size": case["batch_size"],
            "output_len": case["output_len"],
            "run_dir": str(run_dir),
            "prompt_len_min": min(prompt_lengths),
            "prompt_len_max": max(prompt_lengths),
            "prompt_len_mean": statistics.fmean(prompt_lengths),
            "prompt_len_sum": sum(prompt_lengths),
            "batch_elapsed_s": summary.get("batch_elapsed_s"),
            "completion_tokens": summary.get("completion_tokens"),
            "completion_tokens_per_request": (
                float(summary.get("completion_tokens", 0)) / case["batch_size"]
            ),
            "completion_tokens_min": content_sanity["completion_tokens_min"],
            "completion_tokens_max": content_sanity["completion_tokens_max"],
            "output_tokens_per_s": summary.get("output_tokens_per_s"),
            "e2e_mean_s": summary.get("e2e_mean_s"),
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
                "client_batch_sha256": _sha256(run_dir / "client" / "batch.json"),
                "client_requests_sha256": _sha256(run_dir / "client" / "requests.csv"),
                "client_content_sha256": _sha256(run_dir / "client" / "content.json"),
                "case_sha256": case["case_sha256"],
            }
        )
    return manifest, rows, sources


def apply_output_correctness(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record fixed-seed output drift without treating it as deterministic parity."""

    by_key = {
        (
            bool(row.get("ignore_eos", True)),
            row["mode"],
            row["batch_size"],
            row["output_len"],
        ): row
        for row in rows
    }
    mismatches = []
    for row in rows:
        if row["mode"] != "overlap":
            continue
        baseline = by_key.get(
            (
                bool(row.get("ignore_eos", True)),
                "nonoverlap",
                row["batch_size"],
                row["output_len"],
            )
        )
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
                    "ignore_eos": bool(row.get("ignore_eos", True)),
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
    by_key = {
        (
            bool(row.get("ignore_eos", True)),
            row["mode"],
            row["batch_size"],
            row["output_len"],
        ): row
        for row in rows
    }
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
            baseline = by_key.get(
                (
                    bool(row.get("ignore_eos", True)),
                    "nonoverlap",
                    row["batch_size"],
                    row["output_len"],
                )
            )
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
    manifest, rows, sources = collect_completed_rows(campaign_path)
    input_binding_mismatches = apply_input_binding_consistency(rows)
    output_mismatches = apply_output_correctness(rows)
    apply_diagnostics(manifest, rows)
    status = campaign_status(campaign_path)
    if require_complete and len(rows) != int(manifest["case_count"]):
        raise ValueError(
            f"campaign is incomplete: completed={len(rows)}/{manifest['case_count']}"
        )
    return {
        "schema_version": 2,
        "generated_at": time.time(),
        "campaign": manifest["campaign"],
        "campaign_contract_sha256": manifest["contract_sha256"],
        "case_count": manifest["case_count"],
        "completed_case_count": len(rows),
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
        "ignore_eos",
        "batch_size",
        "output_len",
        "prompt_len_min",
        "prompt_len_max",
        "prompt_len_mean",
        "prompt_len_sum",
        "batch_elapsed_s",
        "completion_tokens",
        "completion_tokens_per_request",
        "completion_tokens_min",
        "completion_tokens_max",
        "output_tokens_per_s",
        "e2e_mean_s",
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
    campaign_name = str(summary.get("campaign", {}).get("name", "campaign"))
    lines = [
        f"# Decoupled-Spec Campaign 结果：{campaign_name}",
        "",
        (
            f"当前读取了 **{summary['completed_case_count']}/{summary['case_count']}** "
            f"个 completed case；其中 {review_count} 个触发诊断阈值。"
        ),
        "只有在 campaign ledger 中标记为 completed 的 RUN_DIR 会进入本报告；"
        "失败或未完成 attempt 只保留在 ledger，不参与聚合。",
        "",
        "## 关键图表",
        "",
    ]
    for ignore_eos in sorted({bool(row["ignore_eos"]) for row in rows}):
        eos_label = "ignore_eos_true" if ignore_eos else "ignore_eos_false"
        natural_label = "ignore EOS" if ignore_eos else "natural EOS"
        lines.extend(
            [
                f"### ignore_eos={str(ignore_eos).lower()}",
                "",
                f"![Throughput by batch, {natural_label}](plots/throughput_by_batch_{eos_label}.png)",
                "",
                f"![Paired overlap effects, {natural_label}](plots/paired_overlap_effects_{eos_label}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Case 结果",
            "",
            "| Mode | Ignore EOS | BS | Max output | Actual output/req | tok/s | Accept rate | Draft occupancy | Proposed draft length | Accept length | Content | Quality | RUN_DIR |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["mode"],
                    str(row["ignore_eos"]).lower(),
                    str(row["batch_size"]),
                    str(row["output_len"]),
                    _fmt(row["completion_tokens_per_request"]),
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
            "| Ignore EOS | BS | Max output | Output IDs exact (info) | Throughput ratio | Accept-rate delta | Occupancy delta | Diagnostics |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        if row["mode"] != "overlap" or "throughput_ratio_vs_nonoverlap" not in row:
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["ignore_eos"]).lower(),
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
            "`campaign_summary.json` 的 `content_sanity.requests`。这里展示 case 级 "
            "SHA-256 前缀、首请求开头和末请求结尾，便于快速人工抽查。",
            "",
            "| Mode | Ignore EOS | BS | Max output | Status | Healthy | Text SHA-256 | First request head | Last request tail |",
            "| --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["mode"],
                    str(row["ignore_eos"]).lower(),
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
            "Content sanity 检查 request/input 绑定、配置要求的长度/自然结束范围和粗粒度文本健康，"
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
    json_path = output_dir / "campaign_summary.json"
    csv_path = output_dir / "campaign_summary.csv"
    markdown_path = output_dir / "campaign_report.md"
    _write_json(json_path, summary)
    _write_csv(csv_path, summary["rows"])
    from plot_campaign import write_campaign_plots

    plot_outputs = write_campaign_plots(json_path)
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
            "completed_runs_only": True,
        },
    )
    return {
        "summary_dir": str(output_dir),
        "completed_case_count": summary["completed_case_count"],
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

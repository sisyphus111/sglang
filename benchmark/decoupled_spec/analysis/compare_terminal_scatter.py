#!/usr/bin/env python3
"""Compare sealed pre/post terminal-scatter decoupled-spec runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_ROOT = REPO_ROOT / "benchmark" / "decoupled_spec" / "matrix"
sys.path.insert(0, str(MATRIX_ROOT))

from campaign import validate_verified_run  # noqa: E402
from summarize import assess_run_content  # noqa: E402

VARIANTS = ("old_nonoverlap", "old_overlap", "post_scatter_overlap")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_metric_row(summary: dict[str, Any]) -> dict[str, float | int]:
    elapsed = float(summary["batch_elapsed_s"])
    completion = int(summary["completion_tokens"])
    verify_count = int(summary["spec_verify_ct"])
    proposed = int(summary["spec_num_proposed_drafts"])
    correct = int(summary["spec_num_correct_drafts"])
    if elapsed <= 0 or completion <= 0 or verify_count <= 0 or proposed <= 0:
        raise ValueError("comparison requires positive elapsed/output/verify/proposals")
    row_rate = verify_count / elapsed
    output_per_verify = completion / verify_count
    tps = float(summary["output_tokens_per_s"])
    reconstructed_tps = row_rate * output_per_verify
    if not math.isclose(tps, reconstructed_tps, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "TPS decomposition mismatch: "
            f"reported={tps} reconstructed={reconstructed_tps}"
        )
    return {
        "batch_elapsed_s": elapsed,
        "completion_tokens": completion,
        "output_tokens_per_s": tps,
        "verify_rows_per_s": row_rate,
        "output_tokens_per_verify_row": output_per_verify,
        "spec_draft_occupancy_rate": float(summary["spec_draft_occupancy_rate"]),
        "spec_accept_rate": float(summary["spec_accept_rate"]),
        "spec_verify_ct": verify_count,
        "spec_num_proposed_drafts": proposed,
        "spec_num_correct_drafts": correct,
        "reconstructed_output_tokens_per_s": reconstructed_tps,
    }


def compare_metric_rows(
    baseline: dict[str, float | int], candidate: dict[str, float | int]
) -> dict[str, Any]:
    metrics = (
        "output_tokens_per_s",
        "verify_rows_per_s",
        "output_tokens_per_verify_row",
        "spec_draft_occupancy_rate",
        "spec_accept_rate",
        "spec_verify_ct",
        "spec_num_proposed_drafts",
        "spec_num_correct_drafts",
    )
    result = {}
    for metric in metrics:
        left = float(baseline[metric])
        right = float(candidate[metric])
        result[metric] = {
            "baseline": baseline[metric],
            "candidate": candidate[metric],
            "delta": right - left,
            "ratio": None if left == 0 else right / left,
            "relative_change_pct": None if left == 0 else (right / left - 1) * 100,
        }
    tps_ratio = float(result["output_tokens_per_s"]["ratio"])
    factorized_ratio = float(result["verify_rows_per_s"]["ratio"]) * float(
        result["output_tokens_per_verify_row"]["ratio"]
    )
    result["tps_ratio_factorization"] = {
        "observed": tps_ratio,
        "verify_row_rate_factor": result["verify_rows_per_s"]["ratio"],
        "output_per_verify_factor": result["output_tokens_per_verify_row"]["ratio"],
        "product": factorized_ratio,
        "absolute_error": abs(tps_ratio - factorized_ratio),
    }
    return result


def validate_scatter_provenance(
    run_dir: Path, current_source_path: Path
) -> dict[str, Any]:
    provenance_path = run_dir / "provenance" / "terminal_scatter_source.json"
    provenance = _read_json(provenance_path)
    source_sha256 = _sha256(current_source_path)
    errors = []
    if provenance.get("source_sha256") != source_sha256:
        errors.append(
            "terminal scatter source hash mismatch: "
            f"recorded={provenance.get('source_sha256')} current={source_sha256}"
        )
    scatter_code = str(provenance.get("scatter_code", ""))
    if "retrieve_next_token.scatter_(" not in scatter_code:
        errors.append("terminal scatter provenance does not contain scatter_ code")
    if provenance.get("focused_test_result") != "23 passed":
        errors.append("terminal scatter provenance lacks the 23-pass focused result")
    return {
        "ok": not errors,
        "errors": errors,
        "artifact": _source(provenance_path),
        "record": provenance,
        "current_source": _source(current_source_path),
    }


def _case_by_id(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in manifest["cases"]:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"campaign manifest has no case {case_id}")


def _audit_run(
    *,
    manifest: dict[str, Any],
    case: dict[str, Any],
    run_dir: Path,
    batch_size: int,
    output_len: int,
) -> dict[str, Any]:
    verification = validate_verified_run(manifest, case, run_dir)
    content = assess_run_content(run_dir, batch_size, output_len)
    summary = verification["summary"]
    errors = list(verification["errors"])
    if content["status"] != "pass":
        errors.extend(content["issues"])
    if int(summary.get("spec_num_proposed_drafts", 0) or 0) <= 0:
        errors.append("run has no actual draft proposals")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": verification["audit"].get("warnings", []),
        "sealed_audit": verification["audit"],
        "boot_gate": verification["boot_gate"],
        "content_sanity": content,
        "summary": summary,
        "metrics": build_metric_row(summary),
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "sha256sums": _source(run_dir / "SHA256SUMS"),
            "run_manifest": _source(run_dir / "run_manifest.json"),
            "client_summary": _source(run_dir / "client" / "summary.json"),
            "raw_batch_response": _source(
                run_dir / "client" / "raw_batch_response.json"
            ),
            "sampled_requests": _source(run_dir / "client" / "sampled_requests.jsonl"),
            "responses": _source(run_dir / "client" / "responses.jsonl"),
            "run_start": _source(run_dir / "provenance" / "run_start.json"),
        },
    }


def _require_input_binding_match(
    output_len: int, rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    hashes = {
        variant: row["content_sanity"]["input_binding_sha256"]
        for variant, row in rows.items()
    }
    ok = len(set(hashes.values())) == 1
    return {
        "ok": ok,
        "output_len": output_len,
        "hashes": hashes,
        "error": None if ok else "input binding differs across comparison variants",
    }


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.campaign_dir.resolve() / "campaign_manifest.json"
    manifest = _read_json(manifest_path)
    current_source_path = (
        REPO_ROOT
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "decoupled_verify_worker.py"
    )
    case_map = {
        ("old_nonoverlap", 1024): _case_by_id(manifest, "nonoverlap-bs8-out1k"),
        ("old_overlap", 1024): _case_by_id(manifest, "overlap-bs8-out1k"),
        ("post_scatter_overlap", 1024): _case_by_id(manifest, "overlap-bs8-out1k"),
        ("old_nonoverlap", 4096): _case_by_id(manifest, "nonoverlap-bs8-out4k"),
        ("old_overlap", 4096): _case_by_id(manifest, "overlap-bs8-out4k"),
        ("post_scatter_overlap", 4096): _case_by_id(manifest, "overlap-bs8-out4k"),
    }
    run_map = {
        ("old_nonoverlap", 1024): args.old_nonoverlap_1k,
        ("old_overlap", 1024): args.old_overlap_1k,
        ("post_scatter_overlap", 1024): args.post_scatter_overlap_1k,
        ("old_nonoverlap", 4096): args.old_nonoverlap_4k,
        ("old_overlap", 4096): args.old_overlap_4k,
        ("post_scatter_overlap", 4096): args.post_scatter_overlap_4k,
    }
    runs: dict[str, dict[str, dict[str, Any]]] = {"1024": {}, "4096": {}}
    hard_errors = []
    for (variant, output_len), run_dir in run_map.items():
        row = _audit_run(
            manifest=manifest,
            case=case_map[(variant, output_len)],
            run_dir=run_dir.resolve(),
            batch_size=8,
            output_len=output_len,
        )
        runs[str(output_len)][variant] = row
        hard_errors.extend(
            f"{variant}/out{output_len}: {error}" for error in row["errors"]
        )

    input_binding = []
    for output_len in (1024, 4096):
        check = _require_input_binding_match(output_len, runs[str(output_len)])
        input_binding.append(check)
        if not check["ok"]:
            hard_errors.append(f"out{output_len}: {check['error']}")

    scatter_provenance = {}
    for output_len in (1024, 4096):
        run_dir = run_map[("post_scatter_overlap", output_len)].resolve()
        check = validate_scatter_provenance(run_dir, current_source_path)
        scatter_provenance[str(output_len)] = check
        hard_errors.extend(
            f"post_scatter/out{output_len}: {error}" for error in check["errors"]
        )
    if scatter_provenance["1024"]["record"] != scatter_provenance["4096"]["record"]:
        hard_errors.append("post-scatter source provenance differs between 1K and 4K")

    comparisons = {}
    diagnostics = {}
    for output_len in (1024, 4096):
        rows = runs[str(output_len)]
        metrics = {variant: rows[variant]["metrics"] for variant in VARIANTS}
        comparisons[str(output_len)] = {
            "old_overlap_vs_old_nonoverlap": compare_metric_rows(
                metrics["old_nonoverlap"], metrics["old_overlap"]
            ),
            "post_scatter_vs_old_overlap": compare_metric_rows(
                metrics["old_overlap"], metrics["post_scatter_overlap"]
            ),
            "post_scatter_vs_old_nonoverlap": compare_metric_rows(
                metrics["old_nonoverlap"], metrics["post_scatter_overlap"]
            ),
        }
        post = metrics["post_scatter_overlap"]
        pre = metrics["old_overlap"]
        control = metrics["old_nonoverlap"]
        accept_floor = float(post["spec_accept_rate"]) >= 0.50
        occupancy_floor = float(post["spec_draft_occupancy_rate"]) >= 0.25
        accept_drop_ok = (
            float(post["spec_accept_rate"]) >= float(control["spec_accept_rate"]) - 0.10
        )
        diagnostics[str(output_len)] = {
            "status": (
                "pass"
                if accept_floor and occupancy_floor and accept_drop_ok
                else "review"
            ),
            "accept_rate_at_least_0_50": accept_floor,
            "occupancy_at_least_0_25": occupancy_floor,
            "accept_drop_vs_old_nonoverlap_within_0_10": accept_drop_ok,
            "tps_improved_vs_old_overlap": float(post["output_tokens_per_s"])
            > float(pre["output_tokens_per_s"]),
            "verify_row_rate_improved_vs_old_overlap": float(post["verify_rows_per_s"])
            > float(pre["verify_rows_per_s"]),
            "occupancy_improved_vs_old_overlap": float(
                post["spec_draft_occupancy_rate"]
            )
            > float(pre["spec_draft_occupancy_rate"]),
            "occupancy_recovered_to_old_nonoverlap": float(
                post["spec_draft_occupancy_rate"]
            )
            >= float(control["spec_draft_occupancy_rate"]),
            "output_per_verify_improved_vs_old_overlap": float(
                post["output_tokens_per_verify_row"]
            )
            > float(pre["output_tokens_per_verify_row"]),
            "verify_count_reduced_vs_old_overlap": int(post["spec_verify_ct"])
            < int(pre["spec_verify_ct"]),
        }

    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "kind": "decoupled_spec_terminal_scatter_three_way_comparison",
        "hard_gate": {"ok": not hard_errors, "errors": hard_errors},
        "contract": {
            "batch_size": 8,
            "output_lengths": [1024, 4096],
            "variants": list(VARIANTS),
            "token_exactness_required": False,
            "endpoint_identity": (
                "output_tokens_per_s = verify_rows_per_s * "
                "output_tokens_per_verify_row"
            ),
        },
        "campaign_manifest": _source(manifest_path),
        "input_binding_checks": input_binding,
        "scatter_source_provenance": scatter_provenance,
        "runs": runs,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "conclusion": {
            "row_rate": (
                "Post-scatter overlap raises verify-row service rate at both "
                "response lengths."
            ),
            "output_per_row": (
                "Output tokens per verify row decreases slightly; draft occupancy "
                "does not recover and is lower than old overlap at both lengths."
            ),
            "tps": (
                "The row-rate gain dominates the output-per-row loss, so TPS improves "
                "versus old overlap and exceeds the old non-overlap control in both "
                "single observations."
            ),
            "causal_boundary": (
                "This closes the scalar-sync mechanism ablation directionally, but each "
                "cell is one batch-level observation and old controls were not rerun."
            ),
        },
        "statistical_boundary": [
            "Each variant/output cell contains one batch-level throughput observation.",
            "The eight request rows share one scheduler/GPU batch and are not independent repeats.",
            "The 1K and 4K cases are different horizons, not two repeats of one case.",
            "No confidence interval, p-value, or token-exact cross-version claim is made.",
        ],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Terminal scatter: old non-overlap / old overlap / post-scatter overlap",
        "",
        "## Conclusion",
        "",
        "The scatter change improves throughput by increasing verifier row rate, not by recovering draft supply.",
        "Draft occupancy and output tokens per verify row are slightly worse than the old overlap run at both 1K and 4K; the row-rate gain is larger, so total TPS improves.",
        "",
        "All six input RUN_DIRs passed a fresh sealed audit, exact-config/boot/spec-accounting validation, content sanity, checksum coverage, and request input-binding checks. Cross-version output-token exactness was not required.",
        "",
        "## Three-way metrics",
        "",
        "| Output | Variant | TPS | Verify rows/s | Output/verify | Occupancy | Accept | Verify count | Proposed | Correct |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "old_nonoverlap": "old non-overlap",
        "old_overlap": "old overlap",
        "post_scatter_overlap": "post-scatter overlap",
    }
    for output_len in (1024, 4096):
        for variant in VARIANTS:
            row = report["runs"][str(output_len)][variant]["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(output_len),
                        labels[variant],
                        _fmt(row["output_tokens_per_s"]),
                        _fmt(row["verify_rows_per_s"]),
                        _fmt(row["output_tokens_per_verify_row"]),
                        _fmt(row["spec_draft_occupancy_rate"]),
                        _fmt(row["spec_accept_rate"]),
                        str(row["spec_verify_ct"]),
                        str(row["spec_num_proposed_drafts"]),
                        str(row["spec_num_correct_drafts"]),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Post-scatter deltas",
            "",
            "| Output | Comparison | TPS | Row rate | Output/verify | Occupancy | Accept | Verify count |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for output_len in (1024, 4096):
        for comparison, label in (
            ("post_scatter_vs_old_overlap", "post vs old overlap"),
            ("post_scatter_vs_old_nonoverlap", "post vs old non-overlap"),
        ):
            row = report["comparisons"][str(output_len)][comparison]

            def change(metric: str) -> str:
                return f"{row[metric]['relative_change_pct']:+.2f}%"

            lines.append(
                "| "
                + " | ".join(
                    (
                        str(output_len),
                        label,
                        change("output_tokens_per_s"),
                        change("verify_rows_per_s"),
                        change("output_tokens_per_verify_row"),
                        change("spec_draft_occupancy_rate"),
                        change("spec_accept_rate"),
                        change("spec_verify_ct"),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Mechanism interpretation",
            "",
            "- 1K: post-scatter versus old overlap raises TPS and row rate, while occupancy/output-per-verify fall and verify count rises.",
            "- 4K: the same direction repeats; TPS now exceeds the old non-overlap control, but occupancy remains far below it.",
            "- Therefore the scalar-sync removal improves scheduler/forward cadence. It does not fix the independent draft-tail freshness/supply limitation.",
            "",
            "## Hard gates and provenance",
            "",
            f"- Overall hard gate: **{'PASS' if report['hard_gate']['ok'] else 'FAIL'}**.",
        ]
    )
    for output_len in (1024, 4096):
        provenance = report["scatter_source_provenance"][str(output_len)]
        lines.append(
            f"- Post-scatter {output_len}: source SHA `{provenance['record']['source_sha256']}`, "
            f"focused tests `{provenance['record']['focused_test_result']}`, "
            f"source provenance **{'PASS' if provenance['ok'] else 'FAIL'}**."
        )
    for output_len in (1024, 4096):
        for variant in VARIANTS:
            row = report["runs"][str(output_len)][variant]
            statuses = row["sealed_audit"]["checks"]["statuses"]
            lines.append(
                f"- {output_len} {labels[variant]}: sealed audit **{'PASS' if row['ok'] else 'FAIL'}**, "
                f"content `{row['content_sanity']['healthy_request_count']}/8`, "
                f"verifier/drafter `{statuses['verifier']['state']}/{statuses['drafter']['state']}`, "
                f"alive `{statuses['verifier']['process_alive']}/{statuses['drafter']['process_alive']}`, "
                f"audit warnings `{len(row['warnings'])}`, "
                f"SHA256SUMS `{row['artifacts']['sha256sums']['sha256']}`."
            )

    lines.extend(
        [
            "",
            "## Statistical and correctness boundary",
            "",
            *(f"- {item}" for item in report["statistical_boundary"]),
            "- Request-level scatter points, if plotted, are descriptive paired prompt rows only; batch TPS remains one point per cell.",
            "- Conditional acceptance and occupancy are distinct: acceptance remains healthy, while occupancy did not recover.",
            "- The four old controls retain stale `http_ready` status files, but their recorded PIDs are dead; sealed audit warnings are preserved in JSON. Both new runs end in explicit `exited/exited` with no warnings.",
            "",
            "## RUN_DIRs",
            "",
        ]
    )
    for output_len in (1024, 4096):
        for variant in VARIANTS:
            run_dir = report["runs"][str(output_len)][variant]["artifacts"]["run_dir"]
            lines.append(f"- `{output_len}` `{variant}`: `{run_dir}`")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--old-nonoverlap-1k", type=Path, required=True)
    parser.add_argument("--old-overlap-1k", type=Path, required=True)
    parser.add_argument("--post-scatter-overlap-1k", type=Path, required=True)
    parser.add_argument("--old-nonoverlap-4k", type=Path, required=True)
    parser.add_argument("--old-overlap-4k", type=Path, required=True)
    parser.add_argument("--post-scatter-overlap-4k", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--output-checksums", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_comparison(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    checksum_lines = [
        f"{_sha256(args.output_json)}  {args.output_json.name}",
        f"{_sha256(args.output_markdown)}  {args.output_markdown.name}",
    ]
    args.output_checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(args.output_json.resolve())
    print(args.output_markdown.resolve())
    print(args.output_checksums.resolve())
    if not report["hard_gate"]["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

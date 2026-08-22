#!/usr/bin/env python3
"""Build one provenance-rich overlap/non-overlap forward-trace report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _pair(nonoverlap: float, overlap: float) -> dict[str, float | None]:
    ratio = None if float(nonoverlap) == 0.0 else float(overlap) / float(nonoverlap)
    return {
        "nonoverlap": float(nonoverlap),
        "overlap": float(overlap),
        "delta_overlap_minus_nonoverlap": float(overlap) - float(nonoverlap),
        "ratio_overlap_over_nonoverlap": ratio,
        "relative_change_pct": None if ratio is None else (ratio - 1.0) * 100.0,
    }


def _p50(report: dict[str, Any], section: str, name: str) -> float:
    return float(report[section][name]["p50"])


def _scope_p50(report: dict[str, Any], scope: str, metric: str) -> float | None:
    value = report["scope_summary"][scope][metric]["p50"]
    return None if value is None else float(value)


def _nullable_pair(
    nonoverlap: float | None, overlap: float | None
) -> dict[str, float | None]:
    if nonoverlap is None or overlap is None:
        return {
            "nonoverlap": nonoverlap,
            "overlap": overlap,
            "delta_overlap_minus_nonoverlap": None,
            "ratio_overlap_over_nonoverlap": None,
            "relative_change_pct": None,
        }
    return _pair(nonoverlap, overlap)


def _validate_inputs(
    non_torch: dict[str, Any],
    overlap_torch: dict[str, Any],
    non_nsys: dict[str, Any],
    overlap_nsys: dict[str, Any],
    expected_bs: int,
) -> None:
    for label, report in (
        ("nonoverlap torch", non_torch),
        ("overlap torch", overlap_torch),
        ("nonoverlap nsys", non_nsys),
        ("overlap nsys", overlap_nsys),
    ):
        gate = report["window_gate"]
        if not gate["passed"] or int(gate["expected_bs"]) != expected_bs:
            raise ValueError(f"{label} failed its BS{expected_bs} window gate: {gate}")
    for label, report in (
        ("nonoverlap torch", non_torch),
        ("overlap torch", overlap_torch),
    ):
        if not report["activity_gate"]["gpu_structure_valid"]:
            raise ValueError(f"{label} has no valid CUPTI GPU structure")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    non_torch = _load(args.nonoverlap_torch_summary)
    overlap_torch = _load(args.overlap_torch_summary)
    non_nsys = _load(args.nonoverlap_nsys_summary)
    overlap_nsys = _load(args.overlap_nsys_summary)
    non_endpoint = _load(args.nonoverlap_endpoint_summary)
    overlap_endpoint = _load(args.overlap_endpoint_summary)
    _validate_inputs(non_torch, overlap_torch, non_nsys, overlap_nsys, args.expected_bs)

    endpoint_metrics = {
        "output_tokens_per_s": _pair(
            non_endpoint["output_tokens_per_s"],
            overlap_endpoint["output_tokens_per_s"],
        ),
        "verify_rows_per_s": _pair(
            non_endpoint["spec_verify_ct"] / non_endpoint["batch_elapsed_s"],
            overlap_endpoint["spec_verify_ct"] / overlap_endpoint["batch_elapsed_s"],
        ),
        "output_tokens_per_verify_row": _pair(
            non_endpoint["completion_tokens"] / non_endpoint["spec_verify_ct"],
            overlap_endpoint["completion_tokens"] / overlap_endpoint["spec_verify_ct"],
        ),
        "draft_occupancy_rate": _pair(
            non_endpoint["spec_draft_occupancy_rate"],
            overlap_endpoint["spec_draft_occupancy_rate"],
        ),
        "conditional_accept_rate": _pair(
            non_endpoint["spec_accept_rate"], overlap_endpoint["spec_accept_rate"]
        ),
        "verify_row_count": _pair(
            non_endpoint["spec_verify_ct"], overlap_endpoint["spec_verify_ct"]
        ),
    }

    nsys_fields = (
        "cycle_us",
        "forward_visible_device_op_count",
        "forward_device_active_union_us",
        "forward_exposed_gap_us",
        "cuda_graph_gpu_us",
        "select_to_graph_start_us",
        "graph_end_to_next_select_us",
        "landing_publish_count",
        "landing_device_active_union_us",
        "post_graph_landing_publish_count",
        "latest_landing_publish_age_at_select_us",
        "landing_publish_overlap_with_select_count",
    )
    nsys_p50 = {
        field: _pair(
            _p50(non_nsys, "forward_rounds", field),
            _p50(overlap_nsys, "forward_rounds", field),
        )
        for field in nsys_fields
    }
    nsys_sync = {}
    for mode, report in (("nonoverlap", non_nsys), ("overlap", overlap_nsys)):
        matching = [
            value
            for name, value in report["cuda_runtime_duration_us"].items()
            if "cudaStreamSynchronize" in name
        ]
        if len(matching) != 1:
            raise ValueError(f"{mode} Nsys sync distribution is ambiguous: {matching}")
        nsys_sync[mode] = matching[0]

    scopes = (
        "scheduler.get_next_batch_to_run",
        "scheduler.run_batch",
        "scheduler.process_batch_result",
        "copy_result_to_cpu",
        "sglang.decoupled_spec.gpu_tail_select",
        "sglang.decoupled_spec.tp_broadcast",
        "sglang.decoupled_spec.verify_input_prepare",
        "sglang.decoupled_spec.target_verify",
        "sglang.speculative.target_verify_forward",
        "sglang.speculative.eagle_sample",
        "sglang.speculative.mamba_commit_after_verify",
        "sglang.decoupled_spec.future_publish",
    )
    torch_scope_p50 = {}
    for scope in scopes:
        torch_scope_p50[scope] = {}
        for metric in (
            "cpu_duration_us",
            "gpu_active_union_us_inclusive",
            "cuda_stream_synchronize_us_inclusive",
            "gpu_device_op_count_inclusive",
        ):
            non_value = _scope_p50(non_torch, scope, metric)
            overlap_value = _scope_p50(overlap_torch, scope, metric)
            torch_scope_p50[scope][metric] = _nullable_pair(non_value, overlap_value)

    source_paths = {
        "nonoverlap_torch_summary": args.nonoverlap_torch_summary,
        "overlap_torch_summary": args.overlap_torch_summary,
        "nonoverlap_nsys_summary": args.nonoverlap_nsys_summary,
        "overlap_nsys_summary": args.overlap_nsys_summary,
        "nonoverlap_endpoint_summary": args.nonoverlap_endpoint_summary,
        "overlap_endpoint_summary": args.overlap_endpoint_summary,
        "nonoverlap_torch_trace": Path(non_torch["trace_path"]),
        "overlap_torch_trace": Path(overlap_torch["trace_path"]),
        "nonoverlap_nsys_report": Path(non_nsys["source_path"]),
        "overlap_nsys_report": Path(overlap_nsys["source_path"]),
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "batch_size": args.expected_bs,
            "output_len": 4096,
            "target": "Qwen3.5-27B TP4",
            "drafter": "Qwen3.5-0.8B TP1",
            "spec_shape": "K3/F1",
            "verifier_state": "ReplaySSM spec",
            "workload": "DAPO-Math-17k first 8, thinking, temperature 0, ignore EOS",
            "server_seed": 42,
        },
        "source_files": {name: _source(path) for name, path in source_paths.items()},
        "window_gates": {
            "nonoverlap_torch": non_torch["window_gate"],
            "overlap_torch": overlap_torch["window_gate"],
            "nonoverlap_nsys": non_nsys["window_gate"],
            "overlap_nsys": overlap_nsys["window_gate"],
        },
        "endpoint_decomposition": {
            "identity": "output_tokens_per_s = verify_rows_per_s * output_tokens_per_verify_row",
            "metrics": endpoint_metrics,
        },
        "nsys_structure_p50": nsys_p50,
        "nsys_cuda_stream_synchronize_us": nsys_sync,
        "tail_publication_phase": {
            "nonoverlap_post_graph_publish_counts": non_nsys["forward_rounds"][
                "post_graph_landing_publish_counts"
            ],
            "overlap_post_graph_publish_counts": overlap_nsys["forward_rounds"][
                "post_graph_landing_publish_counts"
            ],
            "nonoverlap_latest_publish_age_at_select_us": non_nsys["forward_rounds"][
                "latest_landing_publish_age_at_select_us_values"
            ],
            "overlap_latest_publish_age_at_select_us": overlap_nsys["forward_rounds"][
                "latest_landing_publish_age_at_select_us_values"
            ],
            "nonoverlap_publish_select_overlap_counts": non_nsys["forward_rounds"][
                "landing_publish_overlap_with_select_counts"
            ],
            "overlap_publish_select_overlap_counts": overlap_nsys["forward_rounds"][
                "landing_publish_overlap_with_select_counts"
            ],
        },
        "torch_scope_p50": torch_scope_p50,
        "microprobe": {
            "provenance": "parent-provided single-run GPU7 mechanism probe",
            "samples": 1,
            "queued_sleep_cycles": 100_000_000,
            "advanced_index_scalar_host_return_us": args.advanced_index_host_us,
            "scatter_scalar_host_return_us": args.scatter_host_us,
            "elementwise_equal_after_completion": True,
            "boundary": (
                "Host-return timing after queuing torch.cuda._sleep; this is a "
                "mechanism probe, not a production latency benchmark."
            ),
            "reproduction": [
                "CUDA_VISIBLE_DEVICES=7 python3 - <<'PY'",
                "import time, torch",
                "rows, width = 8, 4",
                "row_ids = torch.arange(rows, device='cuda')",
                "cols = torch.arange(rows, device='cuda') % width",
                "advanced = torch.zeros((rows, width), dtype=torch.int64, device='cuda')",
                "torch.cuda._sleep(100_000_000)",
                "t0 = time.perf_counter_ns(); advanced[row_ids, cols] = -1",
                "print('advanced_index_scalar host_return_us=', (time.perf_counter_ns()-t0)/1e3)",
                "torch.cuda.synchronize()",
                "scatter = torch.zeros_like(advanced)",
                "torch.cuda._sleep(100_000_000)",
                "t0 = time.perf_counter_ns(); scatter.scatter_(1, cols[:, None], -1)",
                "print('scatter_scalar host_return_us=', (time.perf_counter_ns()-t0)/1e3)",
                "torch.cuda.synchronize(); torch.testing.assert_close(advanced, scatter)",
                "PY",
            ],
        },
        "excluded_captures": [
            {
                "path": str(args.old_nonoverlap_nsys.resolve()),
                "reason": "window gate failed: ReplaySSM fold gridY=1 and trigger-context running BS=4",
                "allowed_use": "BS1-tail structural evidence only",
            },
            {
                "path": str(args.old_nonoverlap_torch.resolve()),
                "reason": "CUPTI GPU/runtime activity absent",
                "allowed_use": "host-scope evidence only; excluded from paired GPU analysis",
            },
        ],
        "evidence_status": {
            "observed": [
                "Overlap raises verifier-row service rate but lowers output tokens per verify row.",
                "Overlap has a shorter forward exposed gap while target graph GPU duration is similar.",
                "Overlap verify-input preparation contains a multi-millisecond cudaStreamSynchronize; non-overlap does not.",
                "Post-graph tail publications are usually present before non-overlap select and almost absent before overlap select.",
                "No publish kernel directly overlaps a select kernel in either 9-round window.",
            ],
            "inference": [
                "The scalar advanced-index assignment blocks the scheduler CPU behind prior forward-stream work.",
                "Because overlap processes result r only after run_batch(r+1), this synchronization delays commit publication and makes the next snapshot older.",
                "Landing H2D/publish GPU work is too small to explain the endpoint slowdown as raw landing compute overhead.",
            ],
            "unverified_causal_step": (
                "Replace the scalar advanced-index update with the equivalent asynchronous scatter, "
                "then rerun the unprofiled BS8 endpoint and the matched trace."
            ),
        },
        "evidence_limits": [
            "Each profiler mode contributes only 9 complete select-to-select intervals from one capture.",
            "Nsys perturbs timing and treats each CUDA graph replay as one opaque operation.",
            "Torch profiler exposes graph nodes but perturbs CPU enqueue timing; profiler throughput is not used as the endpoint response variable.",
            "Nested GPU scope values are inclusive and cannot be added together.",
            "Kernel timestamp phase evidence does not expose semantic publish sequence numbers.",
            "The microprobe is one mechanism sample, not a distribution or production benchmark.",
        ],
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    endpoint = report["endpoint_decomposition"]["metrics"]
    nsys = report["nsys_structure_p50"]
    scopes = report["torch_scope_p50"]
    phase = report["tail_publication_phase"]
    sync = report["nsys_cuda_stream_synchronize_us"]
    micro = report["microprobe"]
    lines = [
        "# Decoupled-spec overlap vs non-overlap: matched BS8 trace analysis",
        "",
        "## Conclusion",
        "",
        (
            "Overlap is not slower at executing verifier rounds. In the unprofiled BS8×4K endpoint it processes "
            f"verify rows `{endpoint['verify_rows_per_s']['relative_change_pct']:+.2f}%` faster, but each verify row "
            f"produces `{endpoint['output_tokens_per_verify_row']['relative_change_pct']:+.2f}%` fewer output tokens. "
            f"The product is `{endpoint['output_tokens_per_s']['relative_change_pct']:+.2f}%` output throughput."
        ),
        "",
        (
            "The matched traces identify one concrete timing mechanism: overlap's GPU-tail snapshot preparation "
            f"contains a `cudaStreamSynchronize` with Torch p50 `{_fmt(scopes['sglang.decoupled_spec.verify_input_prepare']['cuda_stream_synchronize_us_inclusive']['overlap'])} us` "
            f"and Nsys p50 `{_fmt(sync['overlap']['p50'])} us`; the matched non-overlap values are "
            f"`{_fmt(scopes['sglang.decoupled_spec.verify_input_prepare']['cuda_stream_synchronize_us_inclusive']['nonoverlap'])} us` and "
            f"`{_fmt(sync['nonoverlap']['p50'])} us`. The source operation is the scalar advanced-index update of "
            "`retrieve_next_token`."
        ),
        "",
        (
            "This synchronization is mostly hidden under prior GPU work, so it is not valid to add ~6 ms directly "
            "to endpoint latency. Its important effect is ordering: it delays `process_batch_result(r)` and the commit "
            "publication while `select(r+1)` is already queued."
        ),
        "",
        "## Contract and hard gates",
        "",
        "All four included captures passed BS8 gates: Torch used the nearest trigger-context `#running-req=8`; "
        "Nsys additionally required ReplaySSM exact-fold `gridY=8`. The tuple is Qwen3.5-27B TP4 + "
        "Qwen3.5-0.8B TP1, K3/F1, ReplaySSM, DAPO first 8, thinking, output 4096, seed 42.",
        "",
        "## Unprofiled endpoint decomposition",
        "",
        f"`{report['endpoint_decomposition']['identity']}`",
        "",
        "| Metric | Non-overlap | Overlap | Relative change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Output tokens/s", "output_tokens_per_s"),
        ("Verify rows/s", "verify_rows_per_s"),
        ("Output tokens/verify row", "output_tokens_per_verify_row"),
        ("Draft occupancy", "draft_occupancy_rate"),
        ("Conditional accept rate", "conditional_accept_rate"),
        ("Verify row count", "verify_row_count"),
    ):
        row = endpoint[key]
        lines.append(
            f"| {label} | {_fmt(row['nonoverlap'], 6)} | {_fmt(row['overlap'], 6)} | "
            f"{row['relative_change_pct']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Nsys forward-stream structure",
            "",
            "Each value below is the p50 of 9 complete GPU `select → next select` intervals. CUDA graph replay is opaque in Nsys.",
            "",
            "| Metric | Non-overlap | Overlap | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, key in (
        ("Cycle (us)", "cycle_us"),
        ("Forward active union (us)", "forward_device_active_union_us"),
        ("Exposed forward gap (us)", "forward_exposed_gap_us"),
        ("CUDA graph GPU (us)", "cuda_graph_gpu_us"),
        ("Select → graph start (us)", "select_to_graph_start_us"),
        ("Graph end → next select (us)", "graph_end_to_next_select_us"),
        ("Visible forward ops", "forward_visible_device_op_count"),
        ("Landing publishes/round", "landing_publish_count"),
        ("Landing active/round (us)", "landing_device_active_union_us"),
        ("Post-graph publishes", "post_graph_landing_publish_count"),
        (
            "Latest publish age at select (us)",
            "latest_landing_publish_age_at_select_us",
        ),
    ):
        row = nsys[key]
        lines.append(
            f"| {label} | {_fmt(row['nonoverlap'])} | {_fmt(row['overlap'])} | "
            f"{row['delta_overlap_minus_nonoverlap']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "The landing kernels themselves are not the dominant compute cost: p50 landing activity is only "
            f"`{_fmt(nsys['landing_device_active_union_us']['nonoverlap'])} us` non-overlap and "
            f"`{_fmt(nsys['landing_device_active_union_us']['overlap'])} us` overlap per round.",
            "",
            "## Tail freshness at the next select",
            "",
            f"- Non-overlap post-graph publish counts: `{phase['nonoverlap_post_graph_publish_counts']}`; 7/9 intervals contain at least one.",
            f"- Overlap post-graph publish counts: `{phase['overlap_post_graph_publish_counts']}`; only 1/9 contains one.",
            f"- Latest completed publish age p50: non-overlap `{_fmt(nsys['latest_landing_publish_age_at_select_us']['nonoverlap'])} us`, overlap `{_fmt(nsys['latest_landing_publish_age_at_select_us']['overlap'])} us`.",
            f"- Publish/select direct overlap counts are zero in every interval for both modes: non-overlap `{phase['nonoverlap_publish_select_overlap_counts']}`, overlap `{phase['overlap_publish_select_overlap_counts']}`.",
            "",
            "Therefore this window supports stale/older snapshots, but does not support publish/select seqlock collision as the primary mechanism.",
            "",
            "## Torch scope p50",
            "",
            "CPU counts use only `cat=user_annotation`; GPU projections are separate. GPU values are inclusive.",
            "",
            "| Scope | CPU non (us) | CPU overlap (us) | GPU non (us) | GPU overlap (us) | Sync non (us) | Sync overlap (us) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope in (
        "scheduler.get_next_batch_to_run",
        "scheduler.run_batch",
        "scheduler.process_batch_result",
        "sglang.decoupled_spec.gpu_tail_select",
        "sglang.decoupled_spec.verify_input_prepare",
        "sglang.decoupled_spec.target_verify",
        "sglang.speculative.target_verify_forward",
        "sglang.speculative.eagle_sample",
        "sglang.speculative.mamba_commit_after_verify",
        "sglang.decoupled_spec.future_publish",
    ):
        row = scopes[scope]
        cpu = row["cpu_duration_us"]
        gpu = row["gpu_active_union_us_inclusive"]
        scope_sync = row["cuda_stream_synchronize_us_inclusive"]
        lines.append(
            f"| `{scope}` | {_fmt(cpu['nonoverlap'])} | "
            f"{_fmt(cpu['overlap'])} | "
            f"{_fmt(gpu['nonoverlap'])} | "
            f"{_fmt(gpu['overlap'])} | "
            f"{_fmt(scope_sync['nonoverlap'])} | "
            f"{_fmt(scope_sync['overlap'])} |"
        )

    lines.extend(
        [
            "",
            "## Scalar-index mechanism microprobe",
            "",
            f"A single GPU7 mechanism probe queued `torch.cuda._sleep({micro['queued_sleep_cycles']:_})`, then measured host return. "
            f"Advanced-index scalar assignment returned in `{micro['advanced_index_scalar_host_return_us']:.1f} us`; "
            f"the equivalent scalar `scatter_` returned in `{micro['scatter_scalar_host_return_us']:.1f} us`. "
            "After stream completion the tensors were elementwise identical.",
            "",
            "This is a one-sample mechanism falsification, not a production benchmark. Reproduction:",
            "",
            "```bash",
            *micro["reproduction"],
            "```",
            "",
            "## Causal status",
            "",
            "Observed:",
            "",
            *(f"- {item}" for item in report["evidence_status"]["observed"]),
            "",
            "Inference:",
            "",
            *(f"- {item}" for item in report["evidence_status"]["inference"]),
            "",
            "Still required for a causal conclusion:",
            "",
            f"- {report['evidence_status']['unverified_causal_step']}",
            "",
            "A versioned sequence probe is not required before this one-line ablation; it remains the next diagnostic if occupancy does not recover.",
            "",
            "## Excluded captures",
            "",
            *(
                f"- `{item['path']}`: {item['reason']}; {item['allowed_use']}."
                for item in report["excluded_captures"]
            ),
            "",
            "## Evidence limits",
            "",
            *(f"- {item}" for item in report["evidence_limits"]),
            "",
            "## Source artifacts",
            "",
            *(
                f"- `{name}`: `{item['path']}` (`sha256={item['sha256']}`)"
                for name, item in report["source_files"].items()
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-bs", type=int, required=True)
    parser.add_argument("--nonoverlap-torch-summary", type=Path, required=True)
    parser.add_argument("--overlap-torch-summary", type=Path, required=True)
    parser.add_argument("--nonoverlap-nsys-summary", type=Path, required=True)
    parser.add_argument("--overlap-nsys-summary", type=Path, required=True)
    parser.add_argument("--nonoverlap-endpoint-summary", type=Path, required=True)
    parser.add_argument("--overlap-endpoint-summary", type=Path, required=True)
    parser.add_argument("--old-nonoverlap-nsys", type=Path, required=True)
    parser.add_argument("--old-nonoverlap-torch", type=Path, required=True)
    parser.add_argument("--advanced-index-host-us", type=float, required=True)
    parser.add_argument("--scatter-host-us", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(args.output_json.resolve())
    print(args.output_markdown.resolve())


if __name__ == "__main__":
    main()

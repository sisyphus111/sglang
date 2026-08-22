"""Generate a Markdown summary for one decoupled-spec benchmark run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.artifacts import write_json
from plot_utils import build_manifest, finite_float, load_json


def _optional_json(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _format(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def generate_report(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    output_dir = run_path / "plots"
    summary_path = run_path / "client" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    source_paths = [
        summary_path,
        run_path / "client" / "request_metrics.csv",
        run_path / "provenance" / "run_start.json",
        run_path / "roles" / "verifier" / "resolved_config.json",
        run_path / "roles" / "drafter" / "resolved_config.json",
        run_path / "client" / "resolved_config.json",
        run_path / "plots" / "request_latency_manifest.json",
        run_path / "plots" / "request_speculative_manifest.json",
        run_path / "observability" / "plots" / "plot_manifest.json",
    ]
    summary = load_json(summary_path)
    provenance = _optional_json(run_path / "provenance" / "run_start.json")
    verifier = _optional_json(run_path / "roles" / "verifier" / "resolved_config.json")
    drafter = _optional_json(run_path / "roles" / "drafter" / "resolved_config.json")
    client = _optional_json(run_path / "client" / "resolved_config.json")
    verifier_args = verifier.get("server_args", {})
    drafter_args = drafter.get("server_args", {})
    dataset = client.get("dataset", {})
    generation = client.get("generation", {})
    figures = [
        path
        for path in (
            run_path / "plots" / "request_latency.svg",
            run_path / "plots" / "request_speculative.svg",
            run_path / "observability" / "plots" / "overview.svg",
            run_path / "observability" / "plots" / "decode_metrics.svg",
        )
        if path.is_file()
    ]
    lines = [
        "# Decoupled Speculation Run Summary",
        "",
        f"- Run: `{run_path}`",
        f"- Name: `{provenance.get('name', run_path.name)}`",
        f"- Git commit: `{provenance.get('git_commit', 'n/a')}`",
        f"- Git branch: `{provenance.get('git_branch', 'n/a')}`",
        "",
        "## Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Target model | `{verifier_args.get('model_path', 'n/a')}` |",
        f"| Target TP | `{verifier_args.get('tp_size', 'n/a')}` |",
        f"| Draft model | `{drafter_args.get('model_path', 'n/a')}` |",
        f"| Draft TP | `{drafter_args.get('tp_size', 'n/a')}` |",
        f"| Spec steps | `{verifier_args.get('speculative_num_steps', 'n/a')}` |",
        f"| Draft fanout F | `{verifier_args.get('speculative_eagle_topk', 'n/a')}` |",
        "| Overlap schedule | "
        f"`{not verifier_args.get('disable_overlap_schedule', False)}` |",
        f"| Batch size | `{summary.get('batch_size', 'n/a')}` |",
        f"| Dataset | `{dataset.get('format', 'n/a')}` |",
        f"| Output length | `{generation.get('output_len', 'n/a')}` |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Batch elapsed | {_format(summary.get('batch_elapsed_s'))} s |",
        f"| Prompt tokens | {_format(summary.get('prompt_tokens'), 0)} |",
        f"| Completion tokens | {_format(summary.get('completion_tokens'), 0)} |",
        "| Output throughput | "
        f"{_format(summary.get('output_tokens_per_s'))} tokens/s |",
        f"| Verify count | {_format(summary.get('spec_verify_ct'), 0)} |",
        f"| Proposed drafts | {_format(summary.get('spec_num_proposed_drafts'), 0)} |",
        f"| Correct drafts | {_format(summary.get('spec_num_correct_drafts'), 0)} |",
        f"| Accept rate | {_format(summary.get('spec_accept_rate'))} |",
        f"| Draft occupancy | {_format(summary.get('spec_draft_occupancy_rate'))} |",
        "| Proposed draft length | "
        f"{_format(summary.get('spec_proposed_draft_length'))} drafts/verify |",
        "| Accept length | "
        f"{_format(summary.get('spec_accept_length'))} tokens/verify |",
        "",
        "### Latency distribution",
        "",
        "| Metric | Mean | p50 | p95 | p99 | Unit |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        "| TTFT | {mean} | {p50} | {p95} | {p99} | ms |".format(
            **{
                name: _format(_nested(summary, "ttft_ms", name))
                for name in ("mean", "p50", "p95", "p99")
            }
        ),
        "| TPOT | {mean} | {p50} | {p95} | {p99} | ms/token |".format(
            **{
                name: _format(_nested(summary, "tpot_ms", name))
                for name in ("mean", "p50", "p95", "p99")
            }
        ),
        "| E2E latency | {mean} | {p50} | {p95} | {p99} | ms |".format(
            **{
                name: _format(_nested(summary, "e2e_latency_ms", name))
                for name in ("mean", "p50", "p95", "p99")
            }
        ),
        "",
        "## Figures",
        "",
    ]
    lines.extend(
        f"- [{path.name}]({os.path.relpath(path, output_dir)})" for path in figures
    )
    lines.extend(
        [
            "",
            "## Sources",
            "",
            *(
                f"- `{path.relative_to(run_path)}`"
                for path in source_paths
                if path.is_file()
            ),
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = build_manifest(
        kind="decoupled_spec_run_report",
        run_dir=run_path,
        sources=source_paths,
        outputs=[report_path],
    )
    manifest_path = output_dir / "run_report_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(generate_report(args.run_dir)["manifest_path"])


if __name__ == "__main__":
    main()

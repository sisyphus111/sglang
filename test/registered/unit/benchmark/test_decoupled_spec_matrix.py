"""CPU tests for the Agent-operated decoupled-spec matrix contract."""

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MATRIX_ROOT = _REPO_ROOT / "benchmark" / "decoupled_spec" / "matrix"
_MATRIX_CONFIG = (
    _REPO_ROOT
    / "benchmark"
    / "decoupled_spec"
    / "configs"
    / "matrix"
    / "qwen35_dapo_thinking_replayssm.yaml"
)
sys.path.insert(0, str(_MATRIX_ROOT))

from campaign import (  # noqa: E402
    build_campaign_manifest,
    campaign_status,
    materialize_campaign,
    register_attempt,
    set_prerequisite,
    spec_metric_invariant_errors,
    transition_attempt,
    validate_boot_gate,
)
from summarize import (  # noqa: E402
    apply_diagnostics,
    apply_input_binding_consistency,
    apply_output_correctness,
    assess_run_content,
)
from plot_summary import write_matrix_plots  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_content_fixture(root: Path) -> None:
    sampled = [
        {
            "request_id": f"req-{index:06d}",
            "row_index": index,
            "input_ids": [100 + index, 200 + index],
            "prompt_len": 2,
            "requested_output_len": 3,
            "source": {
                "format": "dapo_math_17k",
                "row_index": index,
                "dataset_index": f"dataset-{index}",
            },
        }
        for index in range(2)
    ]
    raw = [
        {
            "index": index,
            "text": f"request {index} produced a normal mathematical explanation",
            "output_ids": [10 + index, 20 + index, 30 + index],
            "meta_info": {
                "id": f"req-{index:06d}",
                "completion_tokens": 3,
                "finish_reason": {"type": "length", "length": 3},
            },
        }
        for index in range(2)
    ]
    responses = [
        {
            "request_id": f"req-{index:06d}",
            "row_index": index,
            "prompt_len": 2,
            "resp_len": 3,
            "completion_tokens": 3,
            "generated_text": raw[index]["text"],
        }
        for index in range(2)
    ]
    _write_jsonl(root / "client" / "sampled_requests.jsonl", sampled)
    _write_json(root / "client" / "raw_batch_response.json", raw)
    _write_jsonl(root / "client" / "responses.jsonl", responses)


class TestDecoupledSpecMatrix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_campaign_manifest(_MATRIX_CONFIG)

    def test_matrix_expands_exact_32_case_replayssm_contract(self):
        manifest = self.manifest
        self.assertEqual(manifest["case_count"], 32)
        self.assertEqual(
            {case["batch_size"] for case in manifest["cases"]}, {8, 16, 32, 64}
        )
        self.assertEqual(
            {case["output_len"] for case in manifest["cases"]},
            {1024, 4096, 16384, 32768},
        )
        self.assertEqual(
            {case["mode"] for case in manifest["cases"]},
            {"nonoverlap", "overlap"},
        )
        self.assertEqual(len({case["case_id"] for case in manifest["cases"]}), 32)
        for alias in ("verifier_nonoverlap", "verifier_overlap"):
            args = manifest["config_snapshots"][alias]["content"]["server_args"]
            self.assertTrue(args["enable_linear_replayssm_spec"])
            self.assertFalse(args["enable_linear_replayssm"])
            self.assertEqual(args["mamba_radix_cache_strategy"], "extra_buffer")
            self.assertEqual(args["max_mamba_cache_size"], 320)
            self.assertEqual(args["max_total_tokens"], 2_200_000)
            self.assertEqual(args["cuda_graph_bs_decode"], [8, 16, 32, 64])
            self.assertEqual(args["random_seed"], 42)
        drafter_args = manifest["config_snapshots"]["drafter"]["content"]["server_args"]
        self.assertEqual(drafter_args["cuda_graph_bs_decode"], [8, 16, 32, 64])
        self.assertEqual(drafter_args["random_seed"], 42)

    def test_matrix_plot_stage_writes_vector_raster_and_manifest(self):
        rows = []
        for mode in ("nonoverlap", "overlap"):
            for batch_size in (8, 16, 32, 64):
                for output_len in (1024, 4096, 16384, 32768):
                    mode_scale = 1.1 if mode == "overlap" else 1.0
                    rows.append(
                        {
                            "mode": mode,
                            "batch_size": batch_size,
                            "output_len": output_len,
                            "output_tokens_per_s": batch_size * mode_scale,
                            "spec_accept_rate": 0.8 + 0.01 * mode_scale,
                            "spec_draft_occupancy_rate": 0.4 / mode_scale,
                            "spec_accept_length": 2.0 / mode_scale,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "matrix_summary.json"
            _write_json(summary_path, {"rows": rows})
            outputs = write_matrix_plots(summary_path)

            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.is_file() for path in outputs))
            manifest = json.loads(outputs[-1].read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "decoupled_spec_matrix_plots")
            self.assertEqual(len(manifest["outputs"]), 4)
            self.assertTrue(
                all(len(output["sha256"]) == 64 for output in manifest["outputs"])
            )

    def test_prerequisite_blocks_attempt_and_incomplete_retry_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_dir = materialize_campaign(_MATRIX_CONFIG, root / "campaign")
            case = self.manifest["cases"][0]
            first_run = root / "run-1"
            _write_json(
                first_run / "provenance" / "run_start.json",
                {"name": case["run_name"]},
            )
            with self.assertRaisesRegex(ValueError, "prerequisites are not passed"):
                register_attempt(campaign_dir, case["case_id"], first_run, None)

            evidence = root / "bs8-probe"
            evidence.mkdir()
            set_prerequisite(
                campaign_dir,
                "draft_inbox_segment_bs8_correctness",
                "passed",
                str(evidence),
                "unit-test evidence",
            )
            first_attempt = register_attempt(
                campaign_dir, case["case_id"], first_run, None
            )
            self.assertEqual(first_attempt["attempt_id"], "attempt-001")
            transition_attempt(
                campaign_dir,
                case["case_id"],
                "incomplete",
                failed_stage="client",
                error="synthetic failure",
            )

            second_run = root / "run-2"
            _write_json(
                second_run / "provenance" / "run_start.json",
                {"name": case["run_name"]},
            )
            second_attempt = register_attempt(
                campaign_dir, case["case_id"], second_run, "retry"
            )
            self.assertEqual(second_attempt["attempt_id"], "attempt-002")
            with self.assertRaisesRegex(ValueError, "cannot skip lifecycle gates"):
                transition_attempt(
                    campaign_dir,
                    case["case_id"],
                    "client_complete",
                )
            status = campaign_status(campaign_dir)
            self.assertEqual(status["counts"]["initialized"], 1)
            ledger = json.loads((campaign_dir / "ledger.json").read_text())
            attempts = ledger["cases"][case["case_id"]]["attempts"]
            self.assertEqual(attempts[0]["state"], "incomplete")
            self.assertEqual(attempts[0]["run_dir"], str(first_run.resolve()))

    def test_boot_gate_reads_measured_capacity_and_cpp_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            logs = run_dir / "logs"
            logs.mkdir()
            cpp = "Loading the decoupled-spec C++ data plane"
            (logs / "verifier.log").write_text(
                "\n".join(
                    [
                        "max_total_num_tokens=2200000",
                        *[
                            "Mamba Cache is allocated. max_mamba_cache_size: 320"
                            for _ in range(4)
                        ],
                        cpp,
                    ]
                ),
                encoding="utf-8",
            )
            (logs / "drafter.log").write_text(
                "\n".join(
                    [
                        "max_total_num_tokens=2200000",
                        "Mamba Cache is allocated. max_mamba_cache_size: 512",
                        cpp,
                    ]
                ),
                encoding="utf-8",
            )
            report = validate_boot_gate(self.manifest, run_dir)
            self.assertTrue(report["ok"], report["errors"])

            (logs / "verifier.log").write_text(
                "max_total_num_tokens=2000000\n"
                "Mamba Cache is allocated. max_mamba_cache_size: 320\n" + cpp,
                encoding="utf-8",
            )
            report = validate_boot_gate(self.manifest, run_dir)
            self.assertFalse(report["ok"])
            self.assertTrue(
                any(
                    "verifier max_total_num_tokens below" in item
                    for item in report["errors"]
                )
            )

    def test_diagnostics_separate_acceptance_from_occupancy(self):
        rows = [
            {
                "mode": "nonoverlap",
                "batch_size": 8,
                "output_len": 1024,
                "output_tokens_per_s": 100.0,
                "spec_accept_rate": 0.75,
                "spec_draft_occupancy_rate": 0.60,
                "quality_status": "pass",
                "diagnostics": [],
            },
            {
                "mode": "overlap",
                "batch_size": 8,
                "output_len": 1024,
                "output_tokens_per_s": 120.0,
                "spec_accept_rate": 0.70,
                "spec_draft_occupancy_rate": 0.20,
                "quality_status": "pass",
                "diagnostics": [],
            },
        ]
        apply_diagnostics(self.manifest, rows)
        overlap = rows[1]
        self.assertEqual(overlap["quality_status"], "review")
        self.assertIn("draft_occupancy<0.250", overlap["diagnostics"])
        self.assertNotIn("accept_rate<0.500", overlap["diagnostics"])
        self.assertAlmostEqual(overlap["throughput_ratio_vs_nonoverlap"], 1.2)
        self.assertAlmostEqual(overlap["accept_rate_delta_vs_nonoverlap"], -0.05)

    def test_fixed_seed_pair_compares_output_ids_by_request_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nonoverlap = root / "nonoverlap" / "client"
            overlap = root / "overlap" / "client"
            nonoverlap.mkdir(parents=True)
            overlap.mkdir(parents=True)
            (nonoverlap / "raw_batch_response.json").write_text(
                json.dumps(
                    [
                        {"index": 1, "output_ids": [20, 21]},
                        {"index": 0, "output_ids": [10, 11, 12]},
                    ]
                ),
                encoding="utf-8",
            )
            (overlap / "raw_batch_response.json").write_text(
                json.dumps(
                    [
                        {"index": 0, "output_ids": [10, 99, 12]},
                        {"index": 1, "output_ids": [20, 21]},
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "mode": "nonoverlap",
                    "batch_size": 8,
                    "output_len": 1024,
                    "run_dir": str(root / "nonoverlap"),
                    "diagnostics": [],
                },
                {
                    "mode": "overlap",
                    "batch_size": 8,
                    "output_len": 1024,
                    "run_dir": str(root / "overlap"),
                    "diagnostics": [],
                },
            ]
            mismatches = apply_output_correctness(rows)
            self.assertFalse(rows[1]["output_ids_exact_vs_nonoverlap"])
            self.assertEqual(mismatches[0]["first_diffs"][0]["position"], 1)
            self.assertEqual(rows[1]["diagnostics"], [])

    def test_content_sanity_binds_saved_requests_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_content_fixture(run_dir)
            result = assess_run_content(run_dir, batch_size=2, output_len=3)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["healthy_request_count"], 2)
        self.assertEqual(result["issues"], [])
        self.assertEqual(len(result["text_sha256"]), 64)
        self.assertEqual(len(result["input_binding_sha256"]), 64)
        for index, request in enumerate(result["requests"]):
            self.assertEqual(request["request_index"], index)
            self.assertTrue(request["binding_ok"])
            self.assertTrue(request["length_ok"])
            self.assertTrue(request["text_ok"])
            self.assertEqual(len(request["input_ids_sha256"]), 64)
            self.assertEqual(len(request["output_ids_sha256"]), 64)
            self.assertEqual(len(request["text_sha256"]), 64)
            self.assertTrue(request["text_head"])
            self.assertTrue(request["text_tail"])

    def test_content_sanity_reports_binding_length_and_text_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_content_fixture(run_dir)
            raw_path = run_dir / "client" / "raw_batch_response.json"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw[1]["meta_info"]["id"] = "req-000000"
            raw[1]["meta_info"]["completion_tokens"] = 2
            raw[1]["meta_info"]["finish_reason"] = {
                "type": "length",
                "length": 2,
            }
            raw[1]["output_ids"] = [11, 21]
            raw[1]["text"] = "\x00" + "\ufffd" * 10
            _write_json(raw_path, raw)
            result = assess_run_content(run_dir, batch_size=2, output_len=3)

        request = result["requests"][1]
        self.assertEqual(result["status"], "review")
        self.assertFalse(request["binding_ok"])
        self.assertFalse(request["length_ok"])
        self.assertFalse(request["text_ok"])
        self.assertIn("raw_request_id_mismatch", request["issues"])
        self.assertIn("output_ids_length_mismatch", request["issues"])
        self.assertIn("nul_in_detokenized_text", request["issues"])
        self.assertIn("excessive_replacement_characters", request["issues"])
        self.assertIn("generated_text_mismatch", request["issues"])

    def test_cross_run_input_binding_uses_majority_and_marks_only_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_content_fixture(run_dir)
            content = assess_run_content(run_dir, batch_size=2, output_len=3)

        rows = []
        for case_id in ("baseline-a", "baseline-b", "drift"):
            rows.append(
                {
                    "case_id": case_id,
                    "content_sanity": deepcopy(content),
                    "content_sanity_status": "pass",
                    "content_sanity_issue_count": 0,
                    "content_sanity_healthy_request_count": 2,
                    "diagnostics": [],
                }
            )
        rows[2]["content_sanity"]["requests"][1]["input_binding_sha256"] = "0" * 64
        mismatches = apply_input_binding_consistency(rows)

        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["case_id"], "drift")
        self.assertEqual(rows[0]["content_sanity_status"], "pass")
        self.assertEqual(rows[1]["content_sanity_status"], "pass")
        self.assertEqual(rows[2]["content_sanity_status"], "review")
        self.assertIn("content_sanity_failed", rows[2]["diagnostics"])

    def test_spec_metric_invariants_close_over_raw_responses(self):
        responses = [
            {
                "index": 0,
                "output_ids": [10, 11, 12, 13],
                "meta_info": {
                    "completion_tokens": 4,
                    "spec_verify_ct": 2,
                    "spec_num_proposed_drafts": 3,
                    "spec_num_correct_drafts": 2,
                    "spec_correct_drafts_histogram": [1, 0, 1, 0],
                },
            }
        ]
        summary = {
            "completion_tokens": 4,
            "spec_verify_ct": 2,
            "spec_num_proposed_drafts": 3,
            "spec_num_correct_drafts": 2,
            "spec_accept_rate": 2 / 3,
            "spec_draft_occupancy_rate": 1 / 2,
            "spec_accept_length": 2.0,
        }
        self.assertEqual(
            spec_metric_invariant_errors(
                summary,
                responses,
                batch_size=1,
                output_len=4,
                max_proposed_drafts=3,
            ),
            [],
        )
        responses[0]["meta_info"]["spec_correct_drafts_histogram"] = [1, 0, 1]
        self.assertEqual(
            spec_metric_invariant_errors(
                summary,
                responses,
                batch_size=1,
                output_len=4,
                max_proposed_drafts=3,
            ),
            [],
        )
        responses[0]["meta_info"]["spec_correct_drafts_histogram"] = [2, 0, 0, 0]
        self.assertTrue(
            spec_metric_invariant_errors(
                summary,
                responses,
                batch_size=1,
                output_len=4,
                max_proposed_drafts=3,
            )
        )

    def test_target_only_startup_is_valid_but_diagnostic(self):
        responses = [
            {
                "index": 0,
                "output_ids": [10, 11],
                "meta_info": {
                    "completion_tokens": 2,
                    "spec_verify_ct": 2,
                    "spec_num_proposed_drafts": 0,
                    "spec_num_correct_drafts": 0,
                    "spec_correct_drafts_histogram": [2, 0, 0, 0],
                },
            }
        ]
        summary = {
            "completion_tokens": 2,
            "spec_verify_ct": 2,
            "spec_num_proposed_drafts": 0,
            "spec_num_correct_drafts": 0,
            "spec_accept_rate": None,
            "spec_draft_occupancy_rate": 0.0,
            "spec_accept_length": 1.0,
        }
        self.assertEqual(
            spec_metric_invariant_errors(
                summary,
                responses,
                batch_size=1,
                output_len=2,
                max_proposed_drafts=3,
            ),
            [],
        )

        rows = [
            {
                "mode": "overlap",
                "batch_size": 8,
                "output_len": 1024,
                "output_tokens_per_s": 100.0,
                "spec_num_proposed_drafts": 0,
                "spec_accept_rate": None,
                "spec_draft_occupancy_rate": 0.0,
                "quality_status": "pass",
                "diagnostics": [],
            }
        ]
        apply_diagnostics(self.manifest, rows)
        self.assertEqual(rows[0]["quality_status"], "review")
        self.assertIn("no_draft_proposals", rows[0]["diagnostics"])


if __name__ == "__main__":
    unittest.main()

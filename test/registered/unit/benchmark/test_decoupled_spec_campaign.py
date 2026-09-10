"""CPU tests for the Agent-operated decoupled-spec campaign contract."""

import hashlib
import csv
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
_CAMPAIGN_SCRIPTS = (
    _REPO_ROOT
    / "benchmark"
    / "decoupled_spec"
    / "skills"
    / "run-decoupled-spec-campaign"
    / "scripts"
)
sys.path.insert(0, str(_CAMPAIGN_SCRIPTS))

from campaign import (  # noqa: E402
    ATTEMPT_STATES,
    TERMINAL_ATTEMPT_STATES,
    _read_client_result_metrics,
    build_campaign_manifest,
    campaign_status,
    materialize_campaign,
    register_attempt,
    set_prerequisite,
    transition_attempt,
    validate_server_gate,
)
from summarize import (  # noqa: E402
    _write_markdown,
    apply_diagnostics,
    apply_input_binding_consistency,
    apply_output_correctness,
    assess_run_content,
)
from plot_campaign import write_campaign_plots  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_campaign_fixture(root: Path) -> Path:
    inspection = {
        "batch_size": 8,
        "prompt_lengths": {"min": 1, "max": 1, "values": [1] * 8},
        "requested_output_lengths": [4096],
    }
    inspection_path = root / "workload.inspect.json"
    _write_json(inspection_path, inspection)
    inspection_sha256 = hashlib.sha256(inspection_path.read_bytes()).hexdigest()
    config = {
        "schema_version": 1,
        "campaign": {
            "name": "unit-decoupled-spec-campaign",
            "run_name_prefix": "unit-dspec",
        },
        "configs": {
            "server_by_mode": {
                "nonoverlap": str(
                    _REPO_ROOT / "benchmark/decoupled_spec/configs/server/"
                    "qwen35_27b_tp4_0_8b_tp1_k3_nonoverlap_cpp_replayssm_bs64_out4k.yaml"
                ),
                "overlap": str(
                    _REPO_ROOT / "benchmark/decoupled_spec/configs/server/"
                    "qwen35_27b_tp4_0_8b_tp1_k3_overlap_cpp_replayssm_bs64_out4k.yaml"
                ),
            },
            "client": str(
                _REPO_ROOT / "benchmark/decoupled_spec/configs/client/"
                "dapo_math_17k_row0_thinking_bs1_out1024_temp1.yaml"
            ),
            "observer": str(
                _REPO_ROOT / "benchmark/decoupled_spec/configs/observer/default.yaml"
            ),
            "workload_inspection": str(inspection_path),
        },
        "axes": {
            "modes": ["nonoverlap", "overlap"],
            "ignore_eos": [False, True],
            "batch_sizes": [1, 8],
            "output_lengths": [1024, 4096],
        },
        "expected": {
            "target_model_path": "/path/to/ssd_dtgl/models/Qwen/Qwen3.5-27B",
            "target_tp_size": 4,
            "drafter_model_path": "/path/to/ssd_dtgl/models/Qwen/Qwen3.5-0.8B",
            "drafter_tp_size": 1,
            "random_seed": 42,
            "dataset_format": "dapo_math_17k",
            "dataset_path": (
                "/path/to/ssd_dtgl/datasets/DAPO-Math-17k/data/"
                "dapo-math-17k.parquet"
            ),
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "verifier_replayssm_flag": "enable_linear_replayssm_spec",
            "verifier_mamba_slots_per_request": 5,
            "drafter_mamba_slots_per_request": 8,
            "max_total_tokens": 300000,
            "cuda_graph_bs_decode": [1, 8, 16, 32, 64],
            "capacity_reference": {
                "prompt_len_min": 1,
                "prompt_len_max": 1,
                "prompt_len_sum": 8,
                "verify_reserve_per_request": 4,
                "required_tokens": 32808,
                "inspection_sha256": inspection_sha256,
            },
            "chat_template_mode": "tokenizer",
            "enable_thinking": True,
            "temperature": 1,
            "ignore_eos_values": [False, True],
        },
        "execution": {
            "ordering": ["ignore_eos", "batch_size", "output_len", "mode"],
            "results_root": str(root / "results"),
            "max_concurrent_deployments": 1,
            "prerequisites": [
                {
                    "id": "draft-path-smoke",
                    "initial_state": "pending",
                    "description": "unit-test prerequisite",
                }
            ],
        },
        "stop_gates": {
            "hard": {
                "require_completed_count_equals_batch_size": True,
                "require_completion_tokens_respect_ignore_eos": True,
                "require_zero_waiting_reqs": True,
                "min_verifier_max_total_num_tokens": 32808,
                "min_drafter_max_total_num_tokens": 32808,
                "min_verifier_mamba_cache_size": 40,
                "min_drafter_mamba_cache_size": 64,
                "min_spec_verify_ct": 1,
                "min_spec_num_proposed_drafts": 0,
            },
            "diagnostic": {
                "require_nonzero_spec_proposals": True,
                "min_spec_accept_rate": 0.5,
                "min_spec_draft_occupancy_rate": 0.25,
                "max_overlap_accept_rate_drop_vs_nonoverlap": 0.1,
            },
        },
    }
    config_path = root / "campaign.yaml"
    _write_json(config_path, config)
    return config_path


def _write_content_fixture(root: Path) -> None:
    requests = [
        {
            "batch_row_index": index,
            "dataset_idx": index,
            "verifier_rank": 0,
            "prompt_len": 2,
            "resp_len": 3,
            "spec_verify_ct": 2,
            "valid_draft_len": 1.5,
            "acc_len": 1.5,
            "e2e_latency_s": 1.0 + 0.1 * index,
            "spec_num_proposed_drafts_by_position": "[2,1]",
            "spec_num_correct_drafts_by_position": "[1,1]",
            "spec_accept_rate_by_position": "[0.5,1.0]",
        }
        for index in range(2)
    ]
    content = [
        {
            "batch_row_idx": index,
            "dataset_idx": index,
            "input_len": 2,
            "output_len": 3,
            "input_ids": [100 + index, 200 + index],
            "input_text": f"rendered input {index}",
            "output_ids": [10 + index, 20 + index, 30 + index],
            "output_text": (
                f"request {index} produced a normal mathematical explanation"
            ),
        }
        for index in range(2)
    ]
    results_dir = root / "client"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "requests.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(requests[0]))
        writer.writeheader()
        writer.writerows(requests)
    _write_json(
        results_dir / "batch.json",
        {
            "output_tokens": 6,
            "batch_elapsed_latency_s": 1.1,
            "batch_thpt": 6 / 1.1,
            "mean_valid_draft_len": 1.5,
            "acclen": 1.5,
        },
    )
    _write_json(results_dir / "content.json", content)


class TestDecoupledSpecCampaign(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.campaign_config = _write_campaign_fixture(Path(cls._temporary.name))
        cls.manifest = build_campaign_manifest(cls.campaign_config)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_temporary"):
            cls._temporary.cleanup()

    def test_campaign_lifecycle_ends_at_completed_without_seal_states(self):
        self.assertEqual(
            ATTEMPT_STATES,
            (
                "initialized",
                "servers_ready",
                "observer_active",
                "client_complete",
                "processes_stopped",
                "completed",
            ),
        )
        self.assertEqual(TERMINAL_ATTEMPT_STATES, {"completed", "incomplete"})

    def test_campaign_requires_explicit_positive_deployment_concurrency(self):
        config = json.loads(self.campaign_config.read_text())
        config["execution"]["max_concurrent_deployments"] = 0
        invalid = Path(self._temporary.name) / "invalid-concurrency.yaml"
        _write_json(invalid, config)
        with self.assertRaisesRegex(ValueError, "max_concurrent_deployments"):
            build_campaign_manifest(invalid)

    def test_campaign_expands_unified_server_cases(self):
        manifest = self.manifest
        self.assertEqual(manifest["mode_target"], "verifier")
        self.assertEqual(manifest["case_count"], 16)
        self.assertEqual({case["batch_size"] for case in manifest["cases"]}, {1, 8})
        self.assertEqual(
            {case["output_len"] for case in manifest["cases"]},
            {1024, 4096},
        )
        self.assertEqual(
            {case["mode"] for case in manifest["cases"]},
            {"nonoverlap", "overlap"},
        )
        self.assertEqual(
            {case["ignore_eos"] for case in manifest["cases"]}, {False, True}
        )
        self.assertEqual(len({case["case_id"] for case in manifest["cases"]}), 16)
        for case in manifest["cases"]:
            self.assertEqual(case["mode_target"], "verifier")
            expected_flag = "--ignore-eos" if case["ignore_eos"] else "--no-ignore-eos"
            runner_command = case["commands"]["runner"]
            self.assertIn(expected_flag, runner_command)
            manifest_index = runner_command.index("--server-manifest")
            self.assertEqual(
                runner_command[manifest_index + 1],
                "<RUNTIME_DIR>/server/manifest.json",
            )
            verifier_rank_index = runner_command.index("--verifier-rank")
            self.assertEqual(runner_command[verifier_rank_index + 1], "0")
            self.assertEqual(
                set(case["commands"]), {"environment", "server", "runner"}
            )
            self.assertIn(
                "benchmark/decoupled_spec/server/server.py", case["commands"]["server"]
            )
            runtime_index = case["commands"]["server"].index("--runtime-dir")
            self.assertEqual(
                case["commands"]["server"][runtime_index + 1], "<RUNTIME_DIR>"
            )
            self.assertIn(
                "benchmark/decoupled_spec/runner.py",
                case["commands"]["runner"],
            )
        self.assertTrue(manifest["stop_gates"]["hard"]["require_zero_waiting_reqs"])
        for alias in ("server_nonoverlap", "server_overlap"):
            server = manifest["config_snapshots"][alias]["content"]
            args = server["verifier"]["server_args"]
            self.assertTrue(args["enable_linear_replayssm_spec"])
            self.assertFalse(args["enable_linear_replayssm"])
            self.assertEqual(args["mamba_radix_cache_strategy"], "extra_buffer")
            self.assertEqual(args["max_mamba_cache_size"], 320)
            self.assertEqual(args["max_total_tokens"], 300_000)
            self.assertEqual(args["chunked_prefill_size"], 16_384)
            self.assertEqual(args["max_prefill_tokens"], 16_384)
            self.assertEqual(args["cuda_graph_bs_decode"], [1, 8, 16, 32, 64])
            self.assertEqual(args["random_seed"], 42)
        drafter_args = manifest["config_snapshots"]["server_overlap"]["content"][
            "drafter"
        ]["server_args"]
        self.assertEqual(drafter_args["chunked_prefill_size"], 16_384)
        self.assertEqual(drafter_args["max_prefill_tokens"], 16_384)
        self.assertEqual(drafter_args["cuda_graph_bs_decode"], [1, 8, 16, 32, 64])
        self.assertEqual(drafter_args["random_seed"], 42)

    def test_campaign_can_target_drafter_overlap_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(self.campaign_config.read_text())
            config["axes"]["mode_target"] = "drafter"

            verifier_overlap_server = self.manifest["config_snapshots"][
                "server_overlap"
            ]["content"]
            for mode in ("nonoverlap", "overlap"):
                server = deepcopy(verifier_overlap_server)
                server["drafter"]["server_args"]["disable_overlap_schedule"] = (
                    mode == "nonoverlap"
                )
                server_path = root / f"server-{mode}.json"
                _write_json(server_path, server)
                config["configs"]["server_by_mode"][mode] = str(server_path)

            config_path = root / "campaign.json"
            _write_json(config_path, config)
            manifest = build_campaign_manifest(config_path)

            self.assertEqual(manifest["mode_target"], "drafter")
            for case in manifest["cases"]:
                self.assertEqual(case["mode_target"], "drafter")
            for mode in ("nonoverlap", "overlap"):
                server = manifest["config_snapshots"][f"server_{mode}"]["content"]
                self.assertFalse(
                    server["verifier"]["server_args"]["disable_overlap_schedule"]
                )
                self.assertEqual(
                    server["drafter"]["server_args"]["disable_overlap_schedule"],
                    mode == "nonoverlap",
                )

            invalid_server = deepcopy(verifier_overlap_server)
            invalid_server["verifier"]["server_args"][
                "disable_overlap_schedule"
            ] = True
            invalid_server["drafter"]["server_args"][
                "disable_overlap_schedule"
            ] = True
            invalid_server_path = root / "server-invalid-verifier.json"
            _write_json(invalid_server_path, invalid_server)
            config["configs"]["server_by_mode"]["nonoverlap"] = str(
                invalid_server_path
            )
            _write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "nonoverlap verifier schedule"):
                build_campaign_manifest(config_path)

            invalid_server["verifier"]["server_args"][
                "disable_overlap_schedule"
            ] = False
            invalid_server["drafter"]["server_args"][
                "disable_overlap_schedule"
            ] = False
            _write_json(invalid_server_path, invalid_server)
            with self.assertRaisesRegex(ValueError, "nonoverlap drafter schedule"):
                build_campaign_manifest(config_path)

    def test_campaign_rejects_unknown_mode_target(self):
        config = json.loads(self.campaign_config.read_text())
        config["axes"]["mode_target"] = "scheduler"
        invalid = Path(self._temporary.name) / "invalid-mode-target.yaml"
        _write_json(invalid, config)
        with self.assertRaisesRegex(ValueError, "axes.mode_target"):
            build_campaign_manifest(invalid)

    def test_campaign_plot_stage_writes_vector_raster_and_manifest(self):
        rows = []
        for ignore_eos in (False, True):
            for mode in ("nonoverlap", "overlap"):
                for batch_size in (8, 16, 32, 64):
                    for output_len in (1024, 4096, 16384, 32768):
                        mode_scale = 1.1 if mode == "overlap" else 1.0
                        rows.append(
                            {
                                "mode": mode,
                                "ignore_eos": ignore_eos,
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
            summary_path = root / "campaign_summary.json"
            _write_json(summary_path, {"rows": rows})
            outputs = write_campaign_plots(summary_path)

            self.assertEqual(len(outputs), 9)
            self.assertTrue(all(path.is_file() for path in outputs))
            manifest = json.loads(outputs[-1].read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "decoupled_spec_campaign_plots")
            self.assertEqual(len(manifest["outputs"]), 8)
            self.assertTrue(
                all(len(output["sha256"]) == 64 for output in manifest["outputs"])
            )

    def test_campaign_plot_stage_supports_bs4_and_one_output_length(self):
        rows = []
        for mode in ("nonoverlap", "overlap"):
            rows.append(
                {
                    "mode": mode,
                    "ignore_eos": True,
                    "batch_size": 4,
                    "output_len": 4096,
                    "output_tokens_per_s": 100.0 + (mode == "overlap"),
                    "spec_accept_rate": 0.8,
                    "spec_draft_occupancy_rate": 0.5,
                    "spec_accept_length": 2.0,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "campaign_summary.json"
            _write_json(summary_path, {"rows": rows})
            outputs = write_campaign_plots(summary_path)

            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.is_file() for path in outputs))
            manifest = json.loads(outputs[-1].read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["outputs"]), 4)

    def test_campaign_report_only_links_observed_ignore_eos_figures(self):
        rows = []
        for mode in ("nonoverlap", "overlap"):
            rows.append(
                {
                    "mode": mode,
                    "ignore_eos": True,
                    "batch_size": 4,
                    "output_len": 4096,
                    "completion_tokens_per_request": 4096,
                    "output_tokens_per_s": 100.0,
                    "spec_accept_rate": 0.8,
                    "spec_draft_occupancy_rate": 0.5,
                    "spec_proposed_draft_length": 1.5,
                    "spec_accept_length": 2.0,
                    "content_sanity_status": "pass",
                    "content_sanity_healthy_request_count": 4,
                    "content_text_sha256": "a" * 64,
                    "content_first_text_head": "head",
                    "content_last_text_tail": "tail",
                    "quality_status": "pass",
                    "diagnostics": [],
                    "run_dir": f"/tmp/{mode}",
                    **(
                        {
                            "throughput_ratio_vs_nonoverlap": 1.0,
                            "accept_rate_delta_vs_nonoverlap": 0.0,
                            "occupancy_delta_vs_nonoverlap": 0.0,
                            "output_ids_exact_vs_nonoverlap": True,
                        }
                        if mode == "overlap"
                        else {}
                    ),
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "campaign_report.md"
            _write_markdown(
                report_path,
                {
                    "campaign": {"name": "drafter-overlap-ab"},
                    "completed_case_count": 2,
                    "case_count": 2,
                    "rows": rows,
                },
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("drafter-overlap-ab", report)
        self.assertIn("throughput_by_batch_ignore_eos_true.png", report)
        self.assertNotIn("ignore_eos=false", report)
        self.assertNotIn("throughput_by_batch_ignore_eos_false.png", report)

    def test_prerequisite_blocks_attempt_and_incomplete_retry_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_dir = materialize_campaign(self.campaign_config, root / "campaign")
            case = self.manifest["cases"][0]
            first_run = root / "run-1"
            first_runtime = root / "runtime-1"
            first_run.mkdir()
            first_runtime.mkdir()
            with self.assertRaisesRegex(ValueError, "prerequisites are not passed"):
                register_attempt(
                    campaign_dir, case["case_id"], first_run, first_runtime, None
                )

            evidence = root / "bs8-probe"
            evidence.mkdir()
            set_prerequisite(
                campaign_dir,
                "draft-path-smoke",
                "passed",
                str(evidence),
                "unit-test evidence",
            )
            first_attempt = register_attempt(
                campaign_dir, case["case_id"], first_run, first_runtime, None
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
            second_runtime = root / "runtime-2"
            second_run.mkdir()
            second_runtime.mkdir()
            second_attempt = register_attempt(
                campaign_dir, case["case_id"], second_run, second_runtime, "retry"
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

    def test_server_gate_reads_unified_ready_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            runtime_dir = root / "runtime"
            run_dir.mkdir()
            runtime_dir.mkdir()
            case = next(
                case for case in self.manifest["cases"] if case["mode"] == "overlap"
            )
            server = self.manifest["config_snapshots"]["server_overlap"]["content"]
            _write_json(run_dir / "config.json", {"server": server})
            engines = []
            for role in ("verifier", "drafter"):
                role_config = server[role]
                engine_id = f"{role}-0"
                relative = f"server/engines/{engine_id}/resolved_config.json"
                _write_json(
                    runtime_dir / relative,
                    {
                        "schema_version": 1,
                        "engine_id": engine_id,
                        "role": role,
                        "rank": 0,
                        "node_rank": 0,
                        "runtime": role_config["runtime"],
                        "server_args": role_config["server_args"],
                    },
                )
                tp_size = role_config["server_args"]["tp_size"]
                engines.append(
                    {
                        "engine_id": engine_id,
                        "role": role,
                        "rank": 0,
                        "resolved_config_path": relative,
                        "rank_placements": [
                            {"tp_rank": rank} for rank in range(tp_size)
                        ],
                    }
                )
            _write_json(
                runtime_dir / "server" / "manifest.json",
                {"schema_version": 1, "state": "ready", "engines": engines},
            )

            report = validate_server_gate(self.manifest, case, run_dir, runtime_dir)
            self.assertTrue(report["ok"], report["errors"])

            verifier_path = runtime_dir / engines[0]["resolved_config_path"]
            verifier = json.loads(verifier_path.read_text())
            verifier["server_args"]["max_total_tokens"] = 1
            _write_json(verifier_path, verifier)
            report = validate_server_gate(self.manifest, case, run_dir, runtime_dir)
            self.assertFalse(report["ok"])
            self.assertTrue(
                any("max_total_tokens below" in item for item in report["errors"])
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
            (nonoverlap / "content.json").write_text(
                json.dumps(
                    [
                        {"batch_row_idx": 1, "output_ids": [20, 21]},
                        {"batch_row_idx": 0, "output_ids": [10, 11, 12]},
                    ]
                ),
                encoding="utf-8",
            )
            (overlap / "content.json").write_text(
                json.dumps(
                    [
                        {"batch_row_idx": 0, "output_ids": [10, 99, 12]},
                        {"batch_row_idx": 1, "output_ids": [20, 21]},
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

    def test_natural_eos_accepts_short_consistent_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_content_fixture(run_dir)
            results_dir = run_dir / "client"
            with (results_dir / "requests.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                requests = list(csv.DictReader(stream))
            for row in requests:
                row["resp_len"] = "2"
                row["acc_len"] = "1.0"
            with (results_dir / "requests.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(requests[0]))
                writer.writeheader()
                writer.writerows(requests)
            content_rows = json.loads(
                (results_dir / "content.json").read_text(encoding="utf-8")
            )
            for row in content_rows:
                row["output_ids"] = row["output_ids"][:2]
                row["output_len"] = 2
            _write_json(results_dir / "content.json", content_rows)
            _write_json(
                results_dir / "batch.json",
                {
                    "output_tokens": 4,
                    "batch_elapsed_latency_s": 1.1,
                    "batch_thpt": 4 / 1.1,
                    "mean_valid_draft_len": 1.5,
                    "acclen": 1.0,
                },
            )

            content = assess_run_content(
                run_dir, batch_size=2, output_len=3, ignore_eos=False
            )
            metrics = _read_client_result_metrics(run_dir, max_proposed_drafts=3)

        self.assertEqual(content["status"], "pass", content["issues"])
        self.assertEqual(content["completion_tokens_min"], 2)
        self.assertEqual(content["completion_tokens_max"], 2)
        self.assertEqual(metrics["completion_tokens"], 4)
        self.assertEqual(metrics["spec_accept_length"], 1.0)

    def test_content_sanity_reports_binding_length_and_text_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_content_fixture(run_dir)
            content_path = run_dir / "client" / "content.json"
            content = json.loads(content_path.read_text(encoding="utf-8"))
            content[1]["dataset_idx"] = 0
            content[1]["output_len"] = 2
            content[1]["output_ids"] = [11, 21]
            content[1]["output_text"] = "\x00" + "\ufffd" * 10
            _write_json(content_path, content)
            result = assess_run_content(run_dir, batch_size=2, output_len=3)

        request = result["requests"][1]
        self.assertEqual(result["status"], "review")
        self.assertFalse(request["binding_ok"])
        self.assertFalse(request["length_ok"])
        self.assertFalse(request["text_ok"])
        self.assertIn("dataset_idx_mismatch", request["issues"])
        self.assertIn("output_len_mismatch", request["issues"])
        self.assertIn("nul_in_detokenized_text", request["issues"])
        self.assertIn("excessive_replacement_characters", request["issues"])

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

    def test_fixed_client_metrics_derive_spec_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_content_fixture(run_dir)
            metrics = _read_client_result_metrics(run_dir, max_proposed_drafts=3)

        self.assertEqual(metrics["batch_size"], 2)
        self.assertEqual(metrics["completion_tokens"], 6)
        self.assertEqual(metrics["spec_verify_ct"], 4)
        self.assertEqual(metrics["spec_num_proposed_drafts"], 6)
        self.assertEqual(metrics["spec_num_correct_drafts"], 4)
        self.assertAlmostEqual(metrics["spec_accept_rate"], 4 / 6)
        self.assertAlmostEqual(metrics["spec_draft_occupancy_rate"], 0.5)
        self.assertEqual(metrics["spec_accept_length"], 1.5)

    def test_target_only_startup_is_valid_but_diagnostic(self):
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

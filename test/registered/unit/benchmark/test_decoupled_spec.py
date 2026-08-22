"""CPU unit tests for the decoupled-spec benchmark client contract."""

import argparse
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aiohttp import web
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4] / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "client-side"))
sys.path.insert(0, str(_ROOT / "plot"))
sys.path.insert(0, str(_ROOT / "server-side"))

from client import (
    add_cli_args,
    apply_cli_overrides,
    build_batch_payload,
    build_records,
    run_batch,
)
from common.collector import collect as collect_observability
from metrics import summarize, write_records
from plot_observability import render_observability
from plot_speculative import render_speculative
from request_loader import load_requests
from config import load_yaml, validate_role_pair


class TestDecoupledSpecBenchmark(CustomTestCase):
    def test_collector_preserves_decode_metrics_http_payload(self):
        async def exercise():
            async def metadata(_request):
                return web.json_response({"model_path": "mock"})

            async def loads(_request):
                return web.json_response(
                    {
                        "loads": [
                            {
                                "dp_rank": 0,
                                "decode_metrics_windows": [
                                    {
                                        "window_id": 9,
                                        "end_time": 100.0,
                                        "num_decode_iters": 40,
                                        "iter_latency_ms": 10.0,
                                        "num_decode_rows": 320,
                                        "sum_context_lens": 3_200_000,
                                        "mean_batch_size": 8.0,
                                        "mean_context_length": 10_000.0,
                                        "num_verify_rows": 320,
                                        "num_accept_tokens": 640,
                                        "num_proposed_drafts": 480,
                                        "accept_length": 2.0,
                                        "proposed_draft_length": 1.5,
                                    }
                                ],
                            }
                        ]
                    }
                )

            app = web.Application()
            app.router.add_get("/model_info", metadata)
            app.router.add_get("/server_info", metadata)
            app.router.add_get("/v1/loads", loads)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = Path(directory)
                    summary = await collect_observability(
                        {
                            "interval_s": 0.01,
                            "request_timeout_s": 0.5,
                            "targets": {
                                "verifier": {"base_url": f"http://127.0.0.1:{port}"},
                                "drafter": {"base_url": f"http://127.0.0.1:{port}"},
                            },
                            "loads": {"include": ["core", "spec", "queues"]},
                        },
                        run_dir,
                        duration_s=0.025,
                    )
                    samples = [
                        json.loads(line)
                        for line in (run_dir / "observability" / "samples.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
            finally:
                await runner.cleanup()

            self.assertGreaterEqual(summary["sample_ct"], 2)
            self.assertTrue(samples)
            for sample in samples:
                window = sample["payload"]["loads"][0]["decode_metrics_windows"][0]
                self.assertEqual(window["window_id"], 9)
                self.assertEqual(window["mean_batch_size"], 8.0)
                self.assertEqual(window["mean_context_length"], 10_000.0)
                self.assertEqual(window["proposed_draft_length"], 1.5)
                self.assertEqual(window["accept_length"], 2.0)

        asyncio.run(exercise())

    def test_observability_plots_fixed_decode_windows(self):
        def record(sample_id, role, collected_wall_time, windows, speculative=None):
            return {
                "sample_id": sample_id,
                "role": role,
                "path": "/v1/loads?include=core,spec,queues",
                "collected_wall_time": collected_wall_time,
                "latency_ms": 1.0,
                "status_code": 200,
                "error": None,
                "payload": {
                    "loads": [
                        {
                            "dp_rank": 0,
                            "num_running_reqs": 8,
                            "num_waiting_reqs": 0,
                            "gen_throughput": 100.0,
                            "token_usage": 0.2,
                            "decode_metrics_windows": windows,
                            "speculative": speculative,
                        }
                    ]
                },
            }

        verifier_windows = [
            {
                "window_id": 1,
                "end_time": 100.2,
                "num_decode_iters": 40,
                "iter_latency_ms": 10.0,
                "num_decode_rows": 320,
                "sum_context_lens": 3_200_000,
                "mean_batch_size": 8.0,
                "mean_context_length": 10_000.0,
                "num_verify_rows": 320,
                "num_accept_tokens": 640,
                "num_proposed_drafts": 480,
                "accept_length": 2.0,
                "proposed_draft_length": 1.5,
            },
            {
                "window_id": 2,
                "end_time": 100.6,
                "num_decode_iters": 40,
                "iter_latency_ms": 11.0,
                "num_decode_rows": 320,
                "sum_context_lens": 3_520_000,
                "mean_batch_size": 8.0,
                "mean_context_length": 11_000.0,
                "num_verify_rows": 320,
                "num_accept_tokens": 608,
                "num_proposed_drafts": 400,
                "accept_length": 1.9,
                "proposed_draft_length": 1.25,
            },
        ]
        drafter_windows = [
            {
                "window_id": 1,
                "end_time": 100.25,
                "num_decode_iters": 40,
                "iter_latency_ms": 8.0,
                "num_decode_rows": 320,
                "sum_context_lens": 3_200_000,
                "mean_batch_size": 8.0,
                "mean_context_length": 10_000.0,
                "num_verify_rows": 0,
                "num_accept_tokens": 0,
                "num_proposed_drafts": 0,
                "accept_length": None,
                "proposed_draft_length": None,
            }
        ]
        speculative = {
            "accept_length": 2.0,
            "accept_rate": 0.8,
            "draft_occupancy_rate": 0.5,
            "proposed_draft_length": 1.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            observability_dir = run_dir / "observability"
            observability_dir.mkdir(parents=True)
            records = [
                record(0, "verifier", 100.0, verifier_windows[:1], speculative),
                record(0, "drafter", 100.0, drafter_windows),
                record(1, "verifier", 101.0, verifier_windows, speculative),
                record(1, "drafter", 101.0, drafter_windows),
            ]
            (observability_dir / "samples.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            (run_dir / "client").mkdir()
            (run_dir / "client" / "formal_window.json").write_text(
                json.dumps(
                    {
                        "started_wall_time": 100.1,
                        "finished_wall_time": 100.9,
                    }
                ),
                encoding="utf-8",
            )

            manifest = render_observability(run_dir)

            self.assertEqual(manifest["decode_metrics_window_ct"], 3)
            self.assertEqual(len(manifest["outputs"]), 4)
            self.assertTrue(
                (observability_dir / "plots" / "decode_metrics.svg").is_file()
            )
            self.assertTrue(
                (observability_dir / "plots" / "decode_metrics.png").is_file()
            )

    def test_formal_verifiers_require_cpp_without_a_fixed_snapshot_wait(self):
        for name, overlap_disabled in (
            ("qwen35_27b_tp4_k3_f1_non_overlap_cpp.yaml", True),
            ("qwen35_27b_tp4_k3_f1_overlap_cpp.yaml", False),
        ):
            with self.subTest(name=name):
                config = load_yaml(_ROOT / "configs" / "verifier" / name)
                env = config["runtime"]["env"]
                self.assertEqual(env["SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND"], "1")
                self.assertNotIn("SGLANG_DECOUPLED_SPEC_SNAPSHOT_WAIT_MS", env)
                self.assertEqual(
                    config["server_args"]["disable_overlap_schedule"],
                    overlap_disabled,
                )

    def test_v0517_role_pair_and_cpp_backend_contract(self):
        verifier = {
            "runtime": {
                "cuda_visible_devices": ["0", "1", "2", "3"],
                "env": {"SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND": "1"},
            },
            "server_args": {
                "speculative_algorithm": "DECOUPLED_VERIFY",
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 4,
                "decoupled_spec_bind_endpoint": "tcp://127.0.0.1:31100",
                "decoupled_spec_connect_endpoints": ["tcp://127.0.0.1:31101"],
            },
        }
        drafter = {
            "runtime": {
                "cuda_visible_devices": ["4"],
                "env": {"SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND": "true"},
            },
            "server_args": {
                "speculative_algorithm": None,
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 4,
                "disable_overlap_schedule": True,
                "decoupled_spec_bind_endpoint": "tcp://127.0.0.1:31101",
                "decoupled_spec_connect_endpoints": ["tcp://127.0.0.1:31100"],
            },
        }

        validate_role_pair(verifier, drafter)

        mixed_backend = copy.deepcopy(drafter)
        mixed_backend["runtime"]["env"]["SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND"] = "0"
        with self.assertRaisesRegex(ValueError, "same Python/C\\+\\+ data-plane"):
            validate_role_pair(verifier, mixed_backend)

    def test_named_cli_overrides_preserve_unspecified_yaml_values(self):
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args(
            [
                "--batch-size",
                "4",
                "--output-len",
                "1024",
                "--ignore-eos",
                "--no-enable-thinking",
            ]
        )
        config = {
            "server": {"base_url": "http://127.0.0.1:30000"},
            "batch": {"size": 1},
            "chat_template": {"mode": "tokenizer", "enable_thinking": True},
            "generation": {"output_len": 256, "ignore_eos": False},
        }
        apply_cli_overrides(config, args)
        self.assertEqual(config["server"]["base_url"], "http://127.0.0.1:30000")
        self.assertEqual(config["batch"]["size"], 4)
        self.assertEqual(config["generation"]["output_len"], 1024)
        self.assertTrue(config["generation"]["ignore_eos"])
        self.assertFalse(config["chat_template"]["enable_thinking"])

    def test_text_is_templated_and_tokenized_on_client(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<chat>question</chat>"
        tokenizer.encode.return_value = [7, 8, 9]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"prompt": "question", "answer": "answer"}\n',
                encoding="utf-8",
            )
            requests = load_requests(
                {
                    "batch": {"size": 1},
                    "dataset": {
                        "format": "generic_jsonl",
                        "path": str(path),
                        "prompt_column": "prompt",
                        "reference_column": "answer",
                    },
                    "chat_template": {"mode": "tokenizer"},
                    "generation": {"output_len": 4},
                },
                tokenizer,
            )
        self.assertEqual(requests[0].rendered_prompt, "<chat>question</chat>")
        self.assertEqual(requests[0].input_ids, [7, 8, 9])
        self.assertEqual(requests[0].prompt_len, 3)

    @patch("request_loader._read_rows")
    def test_dapo_messages_use_native_chat_template(self, read_rows):
        messages = [{"role": "user", "content": "Solve this problem."}]
        read_rows.return_value = [
            {
                "prompt": messages,
                "reward_model": {"ground_truth": "42"},
                "extra_info": {"index": "dataset-row-id"},
            }
        ]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<chat>Solve this problem.</chat>"
        tokenizer.encode.return_value = [7, 8, 9]

        requests = load_requests(
            {
                "batch": {"size": 1},
                "dataset": {
                    "format": "dapo_math_17k",
                    "path": "/unused/in/mocked/test",
                    "prompt_column": "prompt",
                    "reference_column": "reward_model.ground_truth",
                },
                "chat_template": {"mode": "tokenizer", "enable_thinking": True},
                "generation": {"output_len": 4},
            },
            tokenizer,
        )

        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        self.assertEqual(requests[0].raw_prompt, "Solve this problem.")
        self.assertEqual(
            requests[0].rendered_prompt, "<chat>Solve this problem.</chat>"
        )
        self.assertEqual(requests[0].reference_response, "42")
        self.assertEqual(requests[0].source["dataset_index"], "dataset-row-id")

    def test_builds_one_batch_generate_payload(self):
        requests = load_requests(
            {
                "batch": {"size": 2},
                "dataset": {
                    "format": "synthetic_ids",
                    "prompt_len": 3,
                    "output_len": 4,
                },
            }
        )
        payload = build_batch_payload(requests, {"temperature": 0, "ignore_eos": True})
        self.assertEqual(payload["rid"], ["req-000000", "req-000001"])
        self.assertEqual(payload["input_ids"], [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(len(payload["sampling_params"]), 2)
        self.assertTrue(payload["stream"])

    def test_batch_response_and_speculative_summary(self):
        requests = load_requests(
            {
                "batch": {"size": 1},
                "dataset": {
                    "format": "synthetic_ids",
                    "prompt_len": 2,
                    "output_len": 4,
                },
            }
        )
        records = build_records(
            requests,
            [
                {
                    "text": "answer",
                    "meta_info": {
                        "completion_tokens": 4,
                        "spec_verify_ct": 2,
                        "spec_num_proposed_drafts": 6,
                        "spec_num_correct_drafts": 4,
                        "spec_draft_occupancy_rate": 0.75,
                        "spec_proposed_draft_length": 3.0,
                        "spec_proposed_drafts_histogram": [0, 0, 0, 2],
                        "spec_num_proposed_drafts_by_position": [2, 2, 2],
                        "spec_num_correct_drafts_by_position": [2, 1, 1],
                        "spec_accept_rate_by_position": [1.0, 0.5, 0.5],
                    },
                }
            ],
        )
        summary = summarize(records, 2.0)
        self.assertEqual(summary["batch_size"], 1)
        self.assertEqual(summary["completion_tokens"], 4)
        self.assertEqual(summary["output_tokens_per_s"], 2.0)
        self.assertNotIn("requests_per_s", summary)
        self.assertAlmostEqual(summary["spec_accept_rate"], 4 / 6)
        self.assertAlmostEqual(summary["spec_accept_length"], 2.0)
        self.assertAlmostEqual(summary["spec_draft_occupancy_rate"], 0.75)
        self.assertEqual(summary["spec_proposed_draft_length"], 3.0)
        self.assertEqual(records[0]["spec_proposed_drafts_histogram"], [0, 0, 0, 2])
        self.assertEqual(summary["spec_proposed_drafts_histogram"], [0, 0, 0, 2])
        self.assertEqual(summary["spec_num_proposed_drafts_by_position"], [2, 2, 2])
        self.assertEqual(summary["spec_num_correct_drafts_by_position"], [2, 1, 1])
        self.assertEqual(summary["spec_accept_rate_by_position"], [1.0, 0.5, 0.5])

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_records(run_dir / "client" / "request_metrics.csv", records)
            manifest = render_speculative(run_dir)
            self.assertEqual(len(manifest["outputs"]), 2)
            self.assertTrue((run_dir / "plots" / "request_speculative.svg").is_file())

    def test_client_posts_exactly_one_http_batch(self):
        async def exercise():
            received_payloads = []

            async def model_info(_request):
                return web.json_response({"model_path": "target"})

            async def generate(request):
                payload = await request.json()
                received_payloads.append(payload)
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                for completion_tokens in (1, 4):
                    for index in range(len(payload["input_ids"])):
                        is_decode = completion_tokens > 1
                        event = {
                            "index": index,
                            "text": f"answer-{index}",
                            "meta_info": {
                                "completion_tokens": completion_tokens,
                                "spec_verify_ct": 2 if is_decode else 0,
                                "spec_num_proposed_drafts": 3 if is_decode else 0,
                                "spec_num_correct_drafts": 2 if is_decode else 0,
                            },
                        }
                        await response.write(
                            b"data: " + json.dumps(event).encode() + b"\n\n"
                        )
                    await asyncio.sleep(0.001)
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response

            app = web.Application()
            app.router.add_get("/model_info", model_info)
            app.router.add_post("/generate", generate)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                requests = load_requests(
                    {
                        "batch": {"size": 2},
                        "dataset": {
                            "format": "synthetic_ids",
                            "prompt_len": 3,
                            "output_len": 4,
                        },
                    }
                )
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = Path(directory)
                    records, _ = await run_batch(
                        {
                            "server": {"base_url": f"http://127.0.0.1:{port}"},
                            "generation": {"temperature": 0},
                        },
                        requests,
                        run_dir,
                    )
                    stream_events = [
                        json.loads(line)
                        for line in (run_dir / "client" / "stream_timing_events.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
            finally:
                await runner.cleanup()
            self.assertEqual(len(received_payloads), 1)
            self.assertEqual(len(received_payloads[0]["input_ids"]), 2)
            self.assertTrue(received_payloads[0]["stream"])
            self.assertEqual(len(records), 2)
            self.assertIsNotNone(records[0]["ttft_ms"])
            self.assertIsNotNone(records[0]["tpot_ms"])
            self.assertEqual(stream_events[-1]["spec_verify_ct"], 2)
            self.assertEqual(stream_events[-1]["spec_num_proposed_drafts"], 3)
            self.assertEqual(stream_events[-1]["spec_num_correct_drafts"], 2)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()

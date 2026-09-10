"""CPU unit tests for the decoupled-spec benchmark client contract."""

import argparse
import asyncio
import copy
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aiohttp import web
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4] / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "plot"))
sys.path.insert(0, str(_ROOT / "server"))

from client.client import (
    add_cli_args,
    apply_cli_overrides,
    build_batch_payload,
    run_batch,
)
from client.observer import _summarize_decode_metric_windows
from client.observer import collect as collect_observability
from generate_report import generate_report
from client.metrics import (
    BATCH_JSON_FIELDS,
    CONTENT_JSON_FIELDS,
    REQUESTS_CSV_FIELDS,
    build_result_artifacts,
    write_result_artifacts,
)
from plot_latency import render_latency
from plot_observability import (
    _extract_transport_mean_points,
    _latency_histogram_quantile,
    render_observability,
)
from plot_speculative import render_speculative
from plot_utils import upper_iqr_outlier_threshold
from client.request_loader import load_requests
from config import load_yaml, validate_role_pair
from run_io import require_run_dir, update_run_config, update_status


def _integer_histogram(offset, counts):
    return {
        "offset": offset,
        "counts": counts,
        "underflow_count": 0,
        "overflow_count": 0,
    }


def _latency_histogram(counts, sum_us):
    return {
        "count": sum(counts),
        "sum_us": sum_us,
        "bucket_upper_bounds_us": [5, 10],
        "bucket_counts": counts,
    }


def _tail_select(
    rows,
    *,
    direct,
    mismatch,
    selected_counts,
    pending_fast_forwards=0,
    seqlock_retry_rows=0,
    seqlock_retries=0,
    max_seqlock_retries=0,
):
    num_steps = len(selected_counts) - 1
    tail_histogram_size = 2 * num_steps + 3
    raw_counts = [0] * tail_histogram_size
    consumable_counts = [0] * tail_histogram_size
    pending_counts = [0] * tail_histogram_size
    raw_counts[num_steps + 1] = rows
    consumable_counts[num_steps] = rows
    pending_counts[1] = rows
    return {
        "num_select_rows": rows,
        "num_select_valid_rows": direct,
        "reason_counts": {"direct": direct, "bonus_mismatch": mismatch},
        "selected_draft_length_histogram": _integer_histogram(0, selected_counts),
        "raw_draft_tail_length_histogram": _integer_histogram(-1, raw_counts),
        "consumable_draft_tail_length_histogram": _integer_histogram(
            -1, consumable_counts
        ),
        "logical_delta_histogram": _integer_histogram(
            -(2 * num_steps + 1),
            [0] * (2 * num_steps + 1) + [rows] + [0] * (2 * num_steps + 1),
        ),
        "pending_prefix_length_histogram": _integer_histogram(-1, pending_counts),
        "num_publish_seq_initial": 0,
        "num_publish_seq_same": mismatch,
        "num_publish_seq_advance": direct,
        "num_pending_prefix_fast_forwards": pending_fast_forwards,
        "num_protocol_errors": 0,
        "num_seqlock_retry_rows": seqlock_retry_rows,
        "num_seqlock_retries": seqlock_retries,
        "max_seqlock_retries": max_seqlock_retries,
    }


class TestDecoupledSpecBenchmark(CustomTestCase):
    def test_coupled_mtp_histogram_uses_common_position_contract(self):
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
        request_rows, batch_result, _ = build_result_artifacts(
            requests,
            [
                {
                    "text": "answer",
                    "output_ids": [10, 11, 12, 13],
                    "meta_info": {
                        "completion_tokens": 4,
                        "spec_verify_ct": 4,
                        "spec_accept_length": 1.0,
                        "spec_proposed_draft_length": 3.0,
                        "spec_correct_drafts_histogram": [1, 1, 1, 1],
                    },
                }
            ],
            [{"e2e_latency_ms": 1000.0}],
            verifier_rank=0,
        )

        self.assertEqual(
            request_rows[0]["spec_num_proposed_drafts_by_position"], [4, 4, 4]
        )
        self.assertEqual(
            request_rows[0]["spec_num_correct_drafts_by_position"], [3, 2, 1]
        )
        self.assertEqual(
            request_rows[0]["spec_accept_rate_by_position"], [0.75, 0.5, 0.25]
        )
        self.assertEqual(batch_result["output_tokens"], 4)

    def test_run_dir_must_be_created_by_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "create it before"):
                require_run_dir(missing)

    def test_runtime_status_stays_in_server_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            update_status(runtime_dir, "server", "ready", engine_count=2)

            status = json.loads(
                (runtime_dir / "server" / "status.json").read_text(encoding="utf-8")
            )

        self.assertEqual(status["component"], "server")
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["engine_count"], 2)

    def test_component_status_rejects_run_dir_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "inside RUN_DIR"):
                update_status(Path(directory), "../outside", "failed")

    def test_run_config_contains_only_server_and_client(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            update_run_config(run_dir, "client", {"batch": {"size": 2}})
            config = update_run_config(run_dir, "server", {"verifier": {}})

            saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(config, saved)
        self.assertEqual(tuple(saved), ("server", "client"))
        with self.assertRaisesRegex(ValueError, "unsupported run config section"):
            update_run_config(run_dir, "observer", {})

    def test_observer_keeps_healthy_peer_cross_node_samples(self):
        windows = {}
        for window_id, valid_peers, invalid_peers in ((1, 2, 0), (2, 1, 1)):
            windows[("verifier", 0, window_id)] = {
                "target_id": "verifier",
                "role": "verifier",
                "rank": 0,
                "base_url": "http://verifier",
                "dp_rank": 0,
                "window": {
                    "window_id": window_id,
                    "end_time": 100.0 + window_id,
                    "num_decode_iters": 40,
                    "iter_latency_ms": 10.0,
                    "num_decode_rows": 320,
                    "sum_context_lens": 3200,
                    "num_verify_rows": 0,
                    "num_accept_tokens": 0,
                    "num_proposed_drafts": 0,
                    "decoupled_spec": {
                        "transport": {
                            "num_draft_result_frames": 1,
                            "num_draft_result_tokens": 3,
                            "clock_sync_valid": invalid_peers == 0,
                            "clock_error_bound_us": 2.0,
                            "num_clock_sync_valid_peers": valid_peers,
                            "num_clock_sync_invalid_peers": invalid_peers,
                            "draft_transport_one_way_latency_us": (
                                _latency_histogram([1, 0, 0], 4.0)
                            ),
                            "draft_result_ready_to_receive_latency_us": (
                                _latency_histogram([1, 0, 0], 5.0)
                            ),
                        }
                    },
                },
            }
        by_target, _ = _summarize_decode_metric_windows(
            windows,
            [
                {
                    "target_id": "verifier",
                    "role": "verifier",
                    "rank": 0,
                    "base_url": "http://verifier",
                }
            ],
        )

        transport = by_target["verifier"]["decoupled_spec"]["transport"]
        self.assertEqual(
            transport["clock_sync_peers"],
            {
                "window_count": 2,
                "valid_min": 1,
                "valid_max": 2,
                "invalid_min": 0,
                "invalid_max": 1,
                "all_valid_window_count": 1,
            },
        )
        self.assertEqual(transport["draft_transport_one_way_latency_us"]["count"], 2)
        self.assertEqual(
            transport["draft_result_ready_to_receive_latency_us"]["count"], 2
        )

    def test_observer_merges_raw_decoupled_spec_histograms(self):
        transport_bounds = [5, 10]
        windows = {}
        for window_id, bucket_counts, sum_us in (
            (1, [1, 2, 1], 35.0),
            (2, [0, 2, 0], 18.0),
        ):
            windows[("drafter", 0, window_id)] = {
                "target_id": "drafter",
                "role": "drafter",
                "rank": 0,
                "base_url": "http://drafter",
                "dp_rank": 0,
                "window": {
                    "window_id": window_id,
                    "end_time": 100.0 + window_id,
                    "num_decode_iters": 40,
                    "iter_latency_ms": 4.0,
                    "num_decode_rows": 320,
                    "sum_context_lens": 3200,
                    "num_verify_rows": 0,
                    "num_accept_tokens": 0,
                    "num_proposed_drafts": 0,
                    "decoupled_spec": {
                        "transport": {
                            "num_draft_result_frames": sum(bucket_counts),
                            "num_draft_result_tokens": 2 * sum(bucket_counts),
                            "draft_send_queue_depth_max": window_id,
                            "draft_send_queue_latency_us": {
                                "count": sum(bucket_counts),
                                "sum_us": sum_us,
                                "bucket_upper_bounds_us": transport_bounds,
                                "bucket_counts": bucket_counts,
                            },
                        }
                    },
                },
            }
        by_target, by_role = _summarize_decode_metric_windows(
            windows,
            [
                {
                    "target_id": "drafter",
                    "role": "drafter",
                    "rank": 0,
                    "base_url": "http://drafter",
                }
            ],
        )

        for summary in (by_target["drafter"], by_role["drafter"]):
            transport = summary["decoupled_spec"]["transport"]
            histogram = transport["draft_send_queue_latency_us"]
            self.assertEqual(histogram["count"], 6)
            self.assertEqual(histogram["sum_us"], 53.0)
            self.assertEqual(histogram["bucket_counts"], [1, 4, 1])
            self.assertEqual(histogram["p50_us"], 10.0)
            self.assertIsNone(histogram["p95_us"])
            self.assertEqual(transport["draft_send_queue_depth_max"], 2)

    def test_observer_sums_seqlock_retries_and_preserves_window_max(self):
        windows = {}
        for window_id, retry_rows, retries, max_retries in (
            (1, 2, 5, 3),
            (2, 1, 7, 7),
        ):
            windows[("verifier", 0, window_id)] = {
                "target_id": "verifier",
                "role": "verifier",
                "rank": 0,
                "base_url": "http://verifier",
                "dp_rank": 0,
                "window": {
                    "window_id": window_id,
                    "end_time": 100.0 + window_id,
                    "num_decode_iters": 1,
                    "iter_latency_ms": 10.0,
                    "num_decode_rows": 4,
                    "sum_context_lens": 40,
                    "num_verify_rows": 4,
                    "num_accept_tokens": 4,
                    "num_proposed_drafts": 4,
                    "decoupled_spec": {
                        "tail_select": _tail_select(
                            4,
                            direct=4,
                            mismatch=0,
                            selected_counts=[1, 1, 1, 1],
                            seqlock_retry_rows=retry_rows,
                            seqlock_retries=retries,
                            max_seqlock_retries=max_retries,
                        )
                    },
                },
            }

        by_target, _ = _summarize_decode_metric_windows(
            windows,
            [
                {
                    "target_id": "verifier",
                    "role": "verifier",
                    "rank": 0,
                    "base_url": "http://verifier",
                }
            ],
        )

        tail = by_target["verifier"]["decoupled_spec"]["tail_select"]
        self.assertEqual(tail["num_seqlock_retry_rows"], 3)
        self.assertEqual(tail["num_seqlock_retries"], 12)
        self.assertEqual(tail["max_seqlock_retries"], 7)

    def test_latency_histogram_zero_count_and_overflow_are_gaps(self):
        self.assertIsNone(
            _latency_histogram_quantile(_latency_histogram([0, 0, 0], 0), 0.5)
        )
        self.assertIsNone(
            _latency_histogram_quantile(_latency_histogram([0, 0, 1], 11), 0.5)
        )

    def test_transport_plot_merges_only_new_windows_per_observer_poll(self):
        def window(window_id, end_time, count, sum_us):
            return {
                "window_id": window_id,
                "end_time": end_time,
                "decoupled_spec": {
                    "transport": {
                        "draft_send_queue_latency_us": _latency_histogram(
                            [count, 0, 0], sum_us
                        )
                    }
                },
            }

        baseline = window(0, 99.4, 1, 99.0)
        first = window(1, 100.4, 2, 20.0)
        second = window(2, 101.1, 3, 90.0)
        third = window(3, 101.3, 1, 50.0)
        zero_count = window(4, 102.2, 0, 0.0)

        def sample(sample_id, collected_wall_time, windows):
            return {
                "sample_id": sample_id,
                "target_id": "drafter-0",
                "role": "drafter",
                "rank": 0,
                "collected_wall_time": collected_wall_time,
                "error": None,
                "payload": {
                    "loads": [
                        {
                            "dp_rank": 0,
                            "decode_metrics_windows": windows,
                        }
                    ]
                },
            }

        points = _extract_transport_mean_points(
            [
                sample(0, 99.5, [baseline]),
                sample(1, 100.5, [baseline, first]),
                sample(2, 101.5, [baseline, first, second, third]),
                sample(3, 102.5, [first, second, third, zero_count]),
                sample(4, 103.5, [second, third, zero_count]),
            ],
            started_wall_time=100.0,
            finished_wall_time=104.0,
        )

        field = "draft_send_queue_latency_us"
        self.assertEqual([point["sample_id"] for point in points], [1, 2, 3, 4])
        self.assertEqual([point["time_s"] for point in points], [0.5, 1.5, 2.5, 3.5])
        self.assertEqual(
            [point["transport_means"][field] for point in points],
            [10.0, 35.0, None, None],
        )
        self.assertEqual(
            [point["transport_counts"][field] for point in points], [2, 4, 0, 0]
        )

    def test_iteration_latency_presentation_outlier_threshold(self):
        self.assertIsNone(upper_iqr_outlier_threshold([8.0] * 7))
        threshold = upper_iqr_outlier_threshold([8.0] * 7 + [800.0])
        self.assertEqual(threshold, 8.0)

    def test_report_keeps_verifier_and_drafter_cycle_stats_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "observer").mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "server": {
                            "verifier": {"server_args": {}},
                            "drafter": {"server_args": {}},
                        },
                        "client": {"dataset": {}, "generation": {}},
                    }
                ),
                encoding="utf-8",
            )
            request_rows = [
                {
                    "batch_row_index": index,
                    "dataset_idx": index,
                    "verifier_rank": 0,
                    "prompt_len": 1,
                    "resp_len": 1,
                    "spec_verify_ct": 1,
                    "valid_draft_len": 1.0,
                    "acc_len": 1.0,
                    "e2e_latency_s": 1.0,
                    "spec_num_proposed_drafts_by_position": [1],
                    "spec_num_correct_drafts_by_position": [1],
                    "spec_accept_rate_by_position": [1.0],
                }
                for index in range(8)
            ]
            content_rows = [
                {
                    "batch_row_idx": index,
                    "dataset_idx": index,
                    "input_len": 1,
                    "output_len": 1,
                    "input_ids": [1],
                    "input_text": "input",
                    "output_ids": [2],
                    "output_text": "output",
                }
                for index in range(8)
            ]
            write_result_artifacts(
                run_dir / "client",
                request_rows,
                {
                    "output_tokens": 8,
                    "batch_elapsed_latency_s": 1.0,
                    "batch_thpt": 8.0,
                    "mean_valid_draft_len": 1.0,
                    "acclen": 1.0,
                },
                content_rows,
            )
            records = []
            for role, count, latency in (("verifier", 3, 10.0), ("drafter", 4, 7.0)):
                windows = [
                    {
                        "window_id": index,
                        "end_time": 100.0 + index,
                        "num_decode_iters": 1,
                        "iter_latency_ms": latency,
                        "num_decode_rows": 8,
                        "sum_context_lens": 8000,
                        "num_verify_rows": 8 if role == "verifier" else 0,
                        "num_accept_tokens": 16 if role == "verifier" else 0,
                        "num_proposed_drafts": 12 if role == "verifier" else 0,
                    }
                    for index in range(count)
                ]
                records.append(
                    {
                        "sample_id": 0,
                        "target_id": role,
                        "role": role,
                        "rank": 0,
                        "base_url": f"http://{role}",
                        "collected_wall_time": 100.0,
                        "error": None,
                        "payload": {"loads": [{"decode_metrics_windows": windows}]},
                    }
                )
            (run_dir / "observer" / "samples.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (run_dir / "observer" / "bench_timeline.json").write_text(
                json.dumps(
                    {
                        "observer_started_wall_time": 99.0,
                        "client_started_wall_time": 100.0,
                        "client_finished_wall_time": 103.0,
                        "observer_finished_wall_time": 104.0,
                        "observer_elapsed_s": 5.0,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "plots").mkdir()
            (run_dir / "plots" / "decoupled_spec_metrics.png").write_bytes(b"png")

            generate_report(run_dir)

            report = (run_dir / "plots" / "run_report.md").read_text(encoding="utf-8")
            self.assertIn("| verifier | 3 | 10.000 ms", report)
            self.assertIn("| drafter | 4 | 7.000 ms", report)
            self.assertIn("decoupled_spec_metrics.png", report)
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"config.json", "client", "observer", "plots"},
            )
            self.assertEqual(
                {path.name for path in (run_dir / "client").iterdir()},
                {"requests.csv", "batch.json", "content.json"},
            )
            self.assertEqual(
                {path.name for path in (run_dir / "observer").iterdir()},
                {"samples.jsonl", "bench_timeline.json"},
            )
            self.assertFalse(any((run_dir / "plots").glob("*manifest*.json")))

    def test_observer_preserves_decode_metrics_http_payload(self):
        async def exercise():
            window_end_times = {}

            async def metadata(_request):
                return web.json_response({"model_path": "mock"})

            async def loads(request):
                role = request.match_info["role"]
                window_end_time = window_end_times.setdefault(role, time.time())
                is_verifier = role == "verifier"
                return web.json_response(
                    {
                        "loads": [
                            {
                                "dp_rank": 0,
                                "decode_metrics_windows": [
                                    {
                                        "window_id": 9,
                                        "end_time": window_end_time,
                                        "num_decode_iters": 40,
                                        "iter_latency_ms": (
                                            10.0 if is_verifier else 7.0
                                        ),
                                        "num_decode_rows": 320,
                                        "sum_context_lens": 3_200_000,
                                        "mean_batch_size": 8.0,
                                        "mean_context_length": 10_000.0,
                                        "num_verify_rows": 320 if is_verifier else 0,
                                        "num_accept_tokens": (
                                            640 if is_verifier else 0
                                        ),
                                        "num_proposed_drafts": (
                                            480 if is_verifier else 0
                                        ),
                                        "accept_length": (2.0 if is_verifier else None),
                                        "proposed_draft_length": (
                                            1.5 if is_verifier else None
                                        ),
                                        "decoupled_spec": (
                                            {
                                                "tail_select": _tail_select(
                                                    4,
                                                    direct=3,
                                                    mismatch=1,
                                                    selected_counts=[1, 1, 1, 1],
                                                    pending_fast_forwards=1,
                                                    seqlock_retry_rows=1,
                                                    seqlock_retries=3,
                                                    max_seqlock_retries=3,
                                                ),
                                                "transport": {
                                                    "num_draft_result_frames": 4,
                                                    "num_draft_result_tokens": 12,
                                                    "clock_sync_valid": True,
                                                    "clock_error_bound_us": 2.0,
                                                    "num_clock_sync_valid_peers": 1,
                                                    "num_clock_sync_invalid_peers": 0,
                                                    "gpu_publish_staging_slots_max": 2,
                                                    "draft_result_ready_to_receive_latency_us": _latency_histogram(
                                                        [1, 3, 0], 30.0
                                                    ),
                                                    "draft_receive_to_gpu_publish_enqueue_latency_us": _latency_histogram(
                                                        [2, 2, 0], 26.0
                                                    ),
                                                },
                                            }
                                            if is_verifier
                                            else {
                                                "transport": {
                                                    "num_draft_result_frames": 4,
                                                    "num_draft_result_tokens": 12,
                                                    "draft_send_queue_depth_max": 1,
                                                    "draft_send_queue_latency_us": _latency_histogram(
                                                        [2, 2, 0], 25.0
                                                    ),
                                                }
                                            }
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                )

            app = web.Application()
            app.router.add_get("/{role}/model_info", metadata)
            app.router.add_get("/{role}/server_info", metadata)
            app.router.add_get("/{role}/v1/loads", loads)
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
                                "verifier": {
                                    "base_url": f"http://127.0.0.1:{port}/verifier"
                                },
                                "drafter": {
                                    "base_url": f"http://127.0.0.1:{port}/drafter"
                                },
                            },
                            "loads": {"include": ["core", "spec", "queues"]},
                        },
                        run_dir,
                        duration_s=0.025,
                    )
                    samples = [
                        json.loads(line)
                        for line in (run_dir / "observer" / "samples.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
            finally:
                await runner.cleanup()

            self.assertGreaterEqual(summary["sample_ct"], 2)
            self.assertTrue(samples)
            self.assertEqual(set(summary["decode_metrics"]), {"verifier", "drafter"})
            for role in ("verifier", "drafter"):
                metrics = summary["decode_metrics"][role]
                self.assertEqual(metrics["window_count"], 1)
                expected_cycle = 10.0 if role == "verifier" else 7.0
                self.assertEqual(metrics["scheduler_cycle_ms"]["mean"], expected_cycle)
                self.assertEqual(metrics["scheduler_cycle_ms"]["p50"], expected_cycle)
                self.assertEqual(metrics["scheduler_cycle_ms"]["p95"], expected_cycle)
                self.assertEqual(metrics["mean_batch_size"], 8.0)
                self.assertEqual(metrics["mean_context_length"], 10_000.0)
                transport = metrics["decoupled_spec"]["transport"]
                self.assertEqual(transport["num_draft_result_frames"], 4)
                if role == "verifier":
                    self.assertEqual(
                        metrics["decoupled_spec"]["tail_select"]["num_select_rows"],
                        4,
                    )
                    self.assertEqual(
                        metrics["decoupled_spec"]["tail_select"][
                            "num_pending_prefix_fast_forwards"
                        ],
                        1,
                    )
                    self.assertEqual(
                        metrics["decoupled_spec"]["tail_select"][
                            "num_seqlock_retry_rows"
                        ],
                        1,
                    )
                    self.assertEqual(
                        metrics["decoupled_spec"]["tail_select"][
                            "num_seqlock_retries"
                        ],
                        3,
                    )
                    self.assertEqual(
                        metrics["decoupled_spec"]["tail_select"][
                            "max_seqlock_retries"
                        ],
                        3,
                    )
                    self.assertEqual(
                        transport["clock_sync_peers"],
                        {
                            "window_count": 1,
                            "valid_min": 1,
                            "valid_max": 1,
                            "invalid_min": 0,
                            "invalid_max": 0,
                            "all_valid_window_count": 1,
                        },
                    )
                else:
                    self.assertIsNone(metrics["decoupled_spec"]["tail_select"])
            for sample in samples:
                window = sample["payload"]["loads"][0]["decode_metrics_windows"][0]
                self.assertEqual(window["window_id"], 9)
                self.assertEqual(window["mean_batch_size"], 8.0)
                self.assertEqual(window["mean_context_length"], 10_000.0)
                if sample["role"] == "verifier":
                    self.assertEqual(window["proposed_draft_length"], 1.5)
                    self.assertEqual(window["accept_length"], 2.0)
                    self.assertEqual(
                        window["decoupled_spec"]["tail_select"]["reason_counts"],
                        {"direct": 3, "bonus_mismatch": 1},
                    )
                else:
                    self.assertIsNone(window["proposed_draft_length"])
                    self.assertIsNone(window["accept_length"])
                    self.assertEqual(
                        window["decoupled_spec"]["transport"][
                            "draft_send_queue_latency_us"
                        ]["bucket_counts"],
                        [2, 2, 0],
                    )

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
        verifier_windows[0]["decoupled_spec"] = {
            "tail_select": _tail_select(
                4,
                direct=3,
                mismatch=1,
                selected_counts=[1, 1, 1, 1],
                pending_fast_forwards=1,
            ),
            "transport": {
                "num_draft_result_frames": 4,
                "num_draft_result_tokens": 12,
                "clock_sync_valid": True,
                "clock_error_bound_us": 2.0,
                "num_clock_sync_valid_peers": 1,
                "num_clock_sync_invalid_peers": 0,
                "gpu_publish_staging_slots_max": 2,
                "draft_result_ready_to_receive_latency_us": _latency_histogram(
                    [1, 3, 0], 32.0
                ),
                "draft_transport_one_way_latency_us": _latency_histogram(
                    [1, 3, 0], 30.0
                ),
                "draft_receive_to_gpu_publish_enqueue_latency_us": (
                    _latency_histogram([2, 2, 0], 26.0)
                ),
                "draft_gpu_publish_completion_latency_us": _latency_histogram(
                    [0, 1, 0], 8.0
                ),
            },
        }
        verifier_windows[1]["decoupled_spec"] = {
            "tail_select": _tail_select(
                4, direct=2, mismatch=2, selected_counts=[2, 1, 1, 0]
            ),
            "transport": {
                "num_draft_result_frames": 1,
                "num_draft_result_tokens": 3,
                "clock_sync_valid": False,
                "clock_error_bound_us": 3.0,
                "num_clock_sync_valid_peers": 1,
                "num_clock_sync_invalid_peers": 1,
                "gpu_publish_staging_slots_max": 0,
                "draft_result_ready_to_receive_latency_us": _latency_histogram(
                    [1, 0, 0], 4.0
                ),
                "draft_transport_one_way_latency_us": _latency_histogram(
                    [1, 0, 0], 4.0
                ),
                "draft_receive_to_gpu_publish_enqueue_latency_us": (
                    _latency_histogram([1, 0, 0], 4.0)
                ),
                "draft_gpu_publish_completion_latency_us": _latency_histogram(
                    [0, 0, 0], 0.0
                ),
            },
        }
        drafter_windows[0]["decoupled_spec"] = {
            "transport": {
                "num_draft_result_frames": 4,
                "num_draft_result_tokens": 12,
                "draft_send_queue_depth_max": 1,
                "draft_send_queue_latency_us": _latency_histogram([2, 2, 0], 25.0),
            }
        }
        speculative = {
            "accept_length": 2.0,
            "accept_rate": 0.8,
            "draft_occupancy_rate": 0.5,
            "proposed_draft_length": 1.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            observer_dir = run_dir / "observer"
            observer_dir.mkdir(parents=True)
            records = [
                record(0, "verifier", 100.0, verifier_windows[:1], speculative),
                record(0, "drafter", 100.0, drafter_windows),
                record(1, "verifier", 101.0, verifier_windows, speculative),
                record(1, "drafter", 101.0, drafter_windows),
            ]
            verifier_dp1_window = copy.deepcopy(verifier_windows[0])
            verifier_dp1_window["end_time"] = 100.3
            verifier_dp1_window["decoupled_spec"]["tail_select"] = _tail_select(
                12,
                direct=9,
                mismatch=3,
                selected_counts=[3, 3, 3, 3],
            )
            records[0]["payload"]["loads"].append(
                {
                    "dp_rank": 1,
                    "num_running_reqs": 12,
                    "num_waiting_reqs": 0,
                    "gen_throughput": 120.0,
                    "token_usage": 0.3,
                    "decode_metrics_windows": [verifier_dp1_window],
                    "speculative": speculative,
                }
            )
            (observer_dir / "samples.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            (observer_dir / "bench_timeline.json").write_text(
                json.dumps(
                    {
                        "observer_started_wall_time": 100.0,
                        "client_started_wall_time": 100.1,
                        "client_finished_wall_time": 100.9,
                        "observer_finished_wall_time": 101.0,
                        "observer_elapsed_s": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "config.json").write_text(
                json.dumps({"client": {"server": {"target_id": "verifier"}}}),
                encoding="utf-8",
            )

            manifest = render_observability(run_dir)

            self.assertEqual(manifest["decode_metrics_window_ct"], 4)
            self.assertEqual(
                manifest["decode_metrics_window_ct_by_role"],
                {"drafter": 1, "verifier": 3},
            )
            self.assertEqual(len(manifest["outputs"]), 3)
            self.assertEqual(manifest["decoupled_spec_metrics_window_ct"], 4)
            self.assertEqual(manifest["decoupled_spec_time_axis"], "Batch runtime (s)")
            self.assertEqual(manifest["selector_target_id"], "verifier")
            self.assertEqual(manifest["selector_dp_rank"], 1)
            self.assertEqual(
                manifest["selector_series"],
                [
                    {"target_id": "verifier", "dp_rank": 0, "num_select_rows": 8},
                    {"target_id": "verifier", "dp_rank": 1, "num_select_rows": 12},
                ],
            )
            self.assertEqual(manifest["transport_zero_count_policy"], "gap")
            self.assertEqual(manifest["transport_observer_point_ct"], 4)
            self.assertEqual(
                manifest["transport_latency_plot_policy"],
                "per-observer-sample weighted exact mean over newly observed "
                "engine windows: sum(sum_us) / sum(count)",
            )
            self.assertEqual(
                manifest["cross_node_latency_peer_policy"],
                "plot record-time calibrated samples regardless of drain-time peer state",
            )
            self.assertEqual(
                manifest["pending_fast_forward_plot_policy"],
                "events per select row on an unbounded secondary axis",
            )
            self.assertTrue((run_dir / "plots" / "decode_metrics.png").is_file())
            self.assertTrue(
                (run_dir / "plots" / "decoupled_spec_metrics.png").is_file()
            )

    def test_formal_server_templates_use_gpu_backend_without_snapshot_wait(self):
        for name in (
            "qwen35_27b_tp4_0_8b_tp1_k3_nonoverlap_cpp_replayssm_bs64_out4k.yaml",
            "qwen35_27b_tp4_0_8b_tp1_k3_overlap_cpp_replayssm_bs64.yaml",
        ):
            with self.subTest(name=name):
                config = load_yaml(_ROOT / "configs" / "server" / name)
                for role in ("verifier", "drafter"):
                    self.assertNotIn(
                        "SGLANG_DECOUPLED_SPEC_SNAPSHOT_WAIT_MS",
                        config[role]["runtime"]["env"],
                    )

    def test_role_pair_supports_both_drafter_schedule_modes(self):
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
                "disable_radix_cache": True,
                "decoupled_spec_bind_endpoint": "tcp://127.0.0.1:31101",
                "decoupled_spec_connect_endpoints": ["tcp://127.0.0.1:31100"],
            },
        }

        for disable_overlap_schedule in (True, False):
            with self.subTest(disable_overlap_schedule=disable_overlap_schedule):
                candidate = copy.deepcopy(drafter)
                candidate["server_args"]["disable_overlap_schedule"] = (
                    disable_overlap_schedule
                )
                validate_role_pair(verifier, candidate)

        mixed_backend = copy.deepcopy(drafter)
        mixed_backend["runtime"]["env"]["SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND"] = "0"
        validate_role_pair(verifier, mixed_backend)

        verifier_topk_two = copy.deepcopy(verifier)
        drafter_topk_two = copy.deepcopy(drafter)
        verifier_topk_two["server_args"]["speculative_eagle_topk"] = 2
        drafter_topk_two["server_args"]["speculative_eagle_topk"] = 2
        with self.assertRaisesRegex(ValueError, "requires F=1"):
            validate_role_pair(verifier_topk_two, drafter_topk_two)

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
        self.assertEqual(requests[0].batch_row_index, 0)
        self.assertEqual(requests[0].dataset_idx, 0)

    @patch("client.request_loader._read_rows")
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
        self.assertEqual(requests[0].dataset_idx, 0)

    @patch("client.request_loader._read_rows")
    def test_shuffle_preserves_original_dataset_indices(self, read_rows):
        read_rows.return_value = [
            {"prompt": f"row-{index}", "answer": str(index)} for index in range(8)
        ]
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens: [len(text)]
        requests = load_requests(
            {
                "batch": {"size": 4},
                "dataset": {
                    "format": "generic_jsonl",
                    "path": "/unused/in/mocked/test",
                    "shuffle": True,
                    "seed": 17,
                },
                "chat_template": {"mode": "none"},
                "generation": {"output_len": 4},
            },
            tokenizer,
        )
        self.assertEqual(
            [request.batch_row_index for request in requests], [0, 1, 2, 3]
        )
        self.assertEqual([request.dataset_idx for request in requests], [3, 1, 0, 4])

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

    def test_fixed_client_result_artifacts(self):
        self.assertEqual(
            REQUESTS_CSV_FIELDS,
            (
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
            ),
        )
        self.assertEqual(
            BATCH_JSON_FIELDS,
            (
                "output_tokens",
                "batch_elapsed_latency_s",
                "batch_thpt",
                "mean_valid_draft_len",
                "acclen",
            ),
        )
        self.assertEqual(
            CONTENT_JSON_FIELDS,
            (
                "batch_row_idx",
                "dataset_idx",
                "input_len",
                "output_len",
                "input_ids",
                "input_text",
                "output_ids",
                "output_text",
            ),
        )
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
        request_rows, batch_result, content_rows = build_result_artifacts(
            requests,
            [
                {
                    "text": "answer",
                    "output_ids": [10, 11, 12, 13],
                    "meta_info": {
                        "completion_tokens": 4,
                        "spec_verify_ct": 2,
                        "spec_accept_length": 2.0,
                        "spec_proposed_draft_length": 3.0,
                        "spec_num_proposed_drafts_by_position": [2, 2, 2],
                        "spec_num_correct_drafts_by_position": [2, 1, 1],
                        "spec_accept_rate_by_position": [1.0, 0.5, 0.5],
                    },
                }
            ],
            [{"e2e_latency_ms": 2000.0}],
            verifier_rank=3,
        )
        self.assertEqual(tuple(request_rows[0]), REQUESTS_CSV_FIELDS)
        self.assertEqual(tuple(batch_result), BATCH_JSON_FIELDS)
        self.assertEqual(tuple(content_rows[0]), CONTENT_JSON_FIELDS)
        self.assertEqual(request_rows[0]["batch_row_index"], 0)
        self.assertEqual(request_rows[0]["dataset_idx"], 0)
        self.assertEqual(request_rows[0]["verifier_rank"], 3)
        self.assertEqual(request_rows[0]["e2e_latency_s"], 2.0)
        self.assertEqual(batch_result["output_tokens"], 4)
        self.assertEqual(batch_result["batch_elapsed_latency_s"], 2.0)
        self.assertEqual(batch_result["batch_thpt"], 2.0)
        self.assertEqual(batch_result["mean_valid_draft_len"], 3.0)
        self.assertEqual(batch_result["acclen"], 2.0)
        self.assertEqual(content_rows[0]["input_text"], "")
        self.assertEqual(content_rows[0]["output_ids"], [10, 11, 12, 13])

        invalid_output = {
            "text": "answer",
            "output_ids": [10, 11, 12, 13],
            "meta_info": {
                "completion_tokens": 4,
                "spec_verify_ct": 2,
                "spec_accept_length": 2.0,
                "spec_proposed_draft_length": 3.0,
                "spec_num_proposed_drafts_by_position": [2, 2, 2],
                "spec_num_correct_drafts_by_position": [2, 1, 1],
                "spec_accept_rate_by_position": [1.0, 0.75, 0.5],
            },
        }
        with self.assertRaisesRegex(ValueError, "accept rate is inconsistent"):
            build_result_artifacts(
                requests,
                [invalid_output],
                [{"e2e_latency_ms": 2000.0}],
                verifier_rank=3,
            )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            results_dir = write_result_artifacts(
                run_dir / "client", request_rows, batch_result, content_rows
            )
            self.assertEqual(
                {path.name for path in results_dir.iterdir()},
                {"requests.csv", "batch.json", "content.json"},
            )
            with (results_dir / "requests.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames or ()), REQUESTS_CSV_FIELDS)
                serialized = next(reader)
            self.assertEqual(
                serialized["spec_num_proposed_drafts_by_position"], "[2,2,2]"
            )
            self.assertEqual(
                serialized["spec_accept_rate_by_position"], "[1.0,0.5,0.5]"
            )
            speculative_manifest = render_speculative(run_dir)
            latency_manifest = render_latency(run_dir)
            self.assertEqual(len(speculative_manifest["outputs"]), 1)
            self.assertEqual(len(latency_manifest["outputs"]), 1)
            self.assertTrue((run_dir / "plots" / "request_speculative.png").is_file())
            self.assertTrue((run_dir / "plots" / "request_latency.png").is_file())

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
                            "output_ids": list(range(completion_tokens)),
                            "meta_info": {
                                "completion_tokens": completion_tokens,
                                "spec_verify_ct": 2 if is_decode else 0,
                                "spec_num_proposed_drafts": 3 if is_decode else 0,
                                "spec_num_correct_drafts": 2 if is_decode else 0,
                                "spec_accept_length": 2.0 if is_decode else None,
                                "spec_proposed_draft_length": (
                                    1.5 if is_decode else None
                                ),
                                "spec_num_proposed_drafts_by_position": (
                                    [2, 1] if is_decode else None
                                ),
                                "spec_num_correct_drafts_by_position": (
                                    [1, 1] if is_decode else None
                                ),
                                "spec_accept_rate_by_position": (
                                    [0.5, 1.0] if is_decode else None
                                ),
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
                    execution = await run_batch(
                        {
                            "server": {"base_url": f"http://127.0.0.1:{port}"},
                            "generation": {"temperature": 0},
                        },
                        requests,
                    )
                    request_rows, batch_result, content_rows = build_result_artifacts(
                        requests,
                        execution.outputs,
                        execution.timings,
                        verifier_rank=0,
                    )
                    formal_window_exists = (
                        run_dir / "observer" / "bench_timeline.json"
                    ).is_file()
                    legacy_stream_exists = (
                        run_dir / "client" / "stream_timing_events.jsonl"
                    ).exists()
                    legacy_raw_exists = (
                        run_dir / "client" / "raw_batch_response.json"
                    ).exists()
            finally:
                await runner.cleanup()
            self.assertEqual(len(received_payloads), 1)
            self.assertEqual(len(received_payloads[0]["input_ids"]), 2)
            self.assertTrue(received_payloads[0]["stream"])
            self.assertEqual(len(request_rows), 2)
            self.assertGreater(request_rows[0]["e2e_latency_s"], 0)
            self.assertEqual(request_rows[0]["spec_verify_ct"], 2)
            self.assertEqual(content_rows[0]["output_len"], 4)
            self.assertEqual(batch_result["output_tokens"], 8)
            self.assertFalse(formal_window_exists)
            self.assertFalse(legacy_stream_exists)
            self.assertFalse(legacy_raw_exists)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()

"""Queue-validity tests for Agent-operated decoupled-spec benchmarks."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4] / "benchmark" / "decoupled_spec"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VALIDATOR = _load_module(
    "decoupled_spec_validate_samples_test",
    _ROOT / "skills" / "observe-decoupled-spec-run" / "scripts" / "validate_samples.py",
)


def _integer_histogram(offset: int, counts: list[int]) -> dict:
    return {
        "offset": offset,
        "counts": counts,
        "underflow_count": 0,
        "overflow_count": 0,
    }


def _latency_histogram(counts: list[int]) -> dict:
    return {
        "count": sum(counts),
        "sum_us": 12.0,
        "bucket_upper_bounds_us": [5, 10],
        "bucket_counts": counts,
    }


class TestDecoupledSpecQueueGate(CustomTestCase):
    def _write_observability_fixture(
        self, run_dir: Path, *, formal_drafter_waiting: int = 0
    ) -> None:
        observer_dir = run_dir / "observer"
        observer_dir.mkdir(parents=True)
        (observer_dir / "bench_timeline.json").write_text(
            json.dumps(
                {
                    "observer_started_wall_time": 99.0,
                    "client_started_wall_time": 100.0,
                    "client_finished_wall_time": 102.0,
                    "observer_finished_wall_time": 103.2,
                    "observer_elapsed_s": 4.2,
                }
            ),
            encoding="utf-8",
        )
        records = []
        for sample_id, collected_at in enumerate((99.0, 101.0, 103.0)):
            for role in ("verifier", "drafter"):
                waiting = (
                    formal_drafter_waiting
                    if role == "drafter" and collected_at == 101.0
                    else 0
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "target_id": role,
                        "role": role,
                        "rank": 0,
                        "base_url": f"http://{role}",
                        "interval_s": 1.0,
                        "observer_started_wall_time": 98.0,
                        "collected_wall_time": collected_at,
                        "status_code": 200,
                        "error": None,
                        "payload": {
                            "loads": [
                                {
                                    "dp_rank": 0,
                                    "num_waiting_reqs": waiting,
                                }
                            ]
                        },
                    }
                )
        (observer_dir / "samples.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_formal_window_requires_zero_waiting_on_both_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_observability_fixture(run_dir)
            report = _VALIDATOR.validate_samples(
                run_dir, ["verifier", "drafter"], require_formal_window=True
            )

        self.assertTrue(report["ok"], report["errors"])
        for role in ("verifier", "drafter"):
            self.assertEqual(
                report["roles"][role]["max_waiting_reqs_in_formal_window"], 0
            )
            self.assertEqual(report["roles"][role]["formal_waiting_sample_count"], 0)

    def test_formal_window_drafter_queue_invalidates_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_observability_fixture(run_dir, formal_drafter_waiting=16)
            report = _VALIDATOR.validate_samples(
                run_dir, ["verifier", "drafter"], require_formal_window=True
            )

        self.assertFalse(report["ok"])
        self.assertEqual(
            report["roles"]["drafter"]["max_waiting_reqs_in_formal_window"],
            16,
        )
        self.assertTrue(
            any(
                "drafter: waiting requests observed inside the formal window" in error
                for error in report["errors"]
            )
        )

    def test_validator_accepts_coupled_target_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_observability_fixture(run_dir)
            samples_path = run_dir / "observer" / "samples.jsonl"
            records = []
            for line in samples_path.read_text().splitlines():
                record = json.loads(line)
                if record["role"] != "verifier":
                    continue
                record.update(
                    {
                        "target_id": "target-0",
                        "role": "target",
                        "base_url": "http://target",
                    }
                )
                records.append(record)
            samples_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = _VALIDATOR.validate_samples(
                run_dir, ["target"], require_formal_window=True
            )

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["roles"]["target"]["success_count"], 3)

    def test_decode_windows_before_observer_start_are_not_recounted(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_observability_fixture(run_dir)
            samples_path = run_dir / "observer" / "samples.jsonl"
            records = [
                json.loads(line) for line in samples_path.read_text().splitlines()
            ]
            for record in records:
                record["payload"]["loads"][0]["decode_metrics_windows"] = [
                    {
                        "window_id": 1,
                        "end_time": 97.5,
                        "num_decode_iters": 40,
                    },
                    {
                        "window_id": 2,
                        "end_time": 101.0,
                        "num_decode_iters": 40,
                    },
                ]
            samples_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = _VALIDATOR.validate_samples(
                run_dir, ["verifier", "drafter"], require_formal_window=True
            )

        self.assertTrue(report["ok"], report["errors"])
        for role in ("verifier", "drafter"):
            self.assertEqual(report["roles"][role]["decode_metrics_window_count"], 1)

    def test_validator_accepts_role_owned_decoupled_spec_metrics(self):
        rows = 4
        tail = {
            "num_select_rows": rows,
            "num_select_valid_rows": 3,
            "reason_counts": {"direct": 3, "bonus_mismatch": 1},
            "selected_draft_length_histogram": _integer_histogram(0, [1, 1, 1, 1]),
            "raw_draft_tail_length_histogram": _integer_histogram(
                -1, [0, 4, 0, 0, 0, 0, 0, 0, 0]
            ),
            "consumable_draft_tail_length_histogram": _integer_histogram(
                -1, [0, 4, 0, 0, 0, 0, 0, 0, 0]
            ),
            "logical_delta_histogram": _integer_histogram(
                -7, [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]
            ),
            "pending_prefix_length_histogram": _integer_histogram(
                -1, [0, 4, 0, 0, 0, 0, 0, 0, 0]
            ),
            "num_publish_seq_initial": 0,
            "num_publish_seq_same": 1,
            "num_publish_seq_advance": 3,
            "num_pending_prefix_fast_forwards": 1,
            "num_protocol_errors": 0,
            "num_seqlock_retry_rows": 1,
            "num_seqlock_retries": 3,
            "max_seqlock_retries": 3,
        }
        verifier_window = {
            "decoupled_spec": {
                "tail_select": tail,
                "transport": {
                    "num_draft_result_frames": 4,
                    "num_draft_result_tokens": 12,
                    "clock_sync_valid": True,
                    "clock_error_bound_us": 2.0,
                    "num_clock_sync_valid_peers": 1,
                    "num_clock_sync_invalid_peers": 0,
                    "gpu_publish_staging_slots_max": 2,
                    "draft_result_ready_to_receive_latency_us": _latency_histogram(
                        [1, 3, 0]
                    ),
                    "draft_transport_one_way_latency_us": _latency_histogram([1, 3, 0]),
                    "draft_receive_to_gpu_publish_enqueue_latency_us": (
                        _latency_histogram([2, 2, 0])
                    ),
                },
            }
        }
        drafter_window = {
            "decoupled_spec": {
                "transport": {
                    "num_draft_result_frames": 4,
                    "num_draft_result_tokens": 12,
                    "draft_send_queue_depth_max": 1,
                    "draft_send_queue_latency_us": _latency_histogram([2, 2, 0]),
                }
            }
        }
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        _VALIDATOR._validate_decoupled_spec_window(
            "drafter", "drafter", 0, 1, drafter_window, errors
        )
        self.assertEqual(errors, [])

        verifier_transport = verifier_window["decoupled_spec"]["transport"]
        verifier_transport["clock_sync_valid"] = False
        verifier_transport["num_clock_sync_invalid_peers"] = 1
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        self.assertEqual(errors, [])

        verifier_transport["num_clock_sync_valid_peers"] = 0
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        # Peer gauges describe calibration state at drain time. Histograms were
        # already gated at record time, so an in-progress recalibration must not
        # invalidate samples retained from the same engine window.
        self.assertEqual(errors, [])
        error_bound = verifier_transport.pop("clock_error_bound_us")
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        self.assertTrue(
            any("clock_error_bound_us must be finite" in error for error in errors),
            errors,
        )
        verifier_transport["clock_error_bound_us"] = error_bound
        verifier_transport["num_clock_sync_valid_peers"] = 1

        # Landing may process several recovery events between two stable
        # selector observations, so event count is not bounded by row count.
        tail["num_pending_prefix_fast_forwards"] = rows + 1
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        self.assertEqual(errors, [])

        tail["num_seqlock_retry_rows"] = rows + 1
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        self.assertTrue(
            any("seqlock retry rows exceed rows" in error for error in errors), errors
        )
        tail["num_seqlock_retry_rows"] = 1

        tail["max_seqlock_retries"] = 4
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "verifier", "verifier", 0, 1, verifier_window, errors
        )
        self.assertTrue(
            any("seqlock retry counters disagree" in error for error in errors), errors
        )
        tail["max_seqlock_retries"] = 3

    def test_validator_rejects_wrong_role_and_malformed_histograms(self):
        window = {
            "decoupled_spec": {
                "tail_select": {},
                "transport": {
                    "num_draft_result_frames": 1,
                    "num_draft_result_tokens": 1,
                    "clock_sync_valid": False,
                    "draft_transport_one_way_latency_us": {
                        "count": 2,
                        "sum_us": 10.0,
                        "bucket_upper_bounds_us": [5, 10],
                        "bucket_counts": [1, 0],
                    },
                },
            }
        }
        errors = []
        _VALIDATOR._validate_decoupled_spec_window(
            "drafter", "drafter", 0, 1, window, errors
        )
        self.assertTrue(any("tail_select is verifier-only" in item for item in errors))
        self.assertTrue(any("not owned by role=drafter" in item for item in errors))
        self.assertTrue(any("overflow bucket" in item for item in errors))
        self.assertTrue(
            any("clock_error_bound_us must be finite" in item for item in errors)
        )


if __name__ == "__main__":
    unittest.main()

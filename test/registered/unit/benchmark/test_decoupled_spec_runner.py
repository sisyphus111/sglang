"""CPU unit tests for decoupled-spec Client/Observer coordination."""

import argparse
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4] / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_ROOT))

import runner as benchmark_runner


def _sample(sample_id, target_id, collected_at, waiting=0):
    return {
        "sample_id": sample_id,
        "target_id": target_id,
        "collected_wall_time": collected_at,
        "latency_ms": 10.0,
        "status_code": 200,
        "error": None,
        "payload": {"loads": [{"num_waiting_reqs": waiting}]},
    }


class TestDecoupledSpecRunner(CustomTestCase):
    def test_observer_shutdown_uses_sigterm(self):
        process = MagicMock()
        process.poll.return_value = None
        process.communicate.return_value = ("summary", "")
        process.returncode = 0

        output = benchmark_runner._stop_observer(process, timeout_s=5.0)

        process.send_signal.assert_called_once_with(signal.SIGTERM)
        self.assertEqual(output, ("summary", ""))

    def test_concurrent_reader_ignores_an_incomplete_final_line(self):
        for suffix in ("{", "{\n"):
            with self.subTest(suffix=repr(suffix)):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "samples.jsonl"
                    path.write_text(
                        json.dumps(_sample(0, "verifier-0", 99.0))
                        + "\n"
                        + suffix,
                        encoding="utf-8",
                    )

                    records = benchmark_runner._read_samples(path)

                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["sample_id"], 0)

    def test_concurrent_reader_rejects_an_invalid_middle_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text(
                json.dumps(_sample(0, "verifier-0", 99.0))
                + "\n{\n"
                + json.dumps(_sample(1, "verifier-0", 100.0))
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2"):
                benchmark_runner._read_samples(path)

    def test_sample_barriers_require_a_complete_round(self):
        records = [
            _sample(0, "verifier-0", 99.0),
            _sample(0, "drafter-0", 99.0, waiting=1),
            _sample(1, "verifier-0", 100.0),
            _sample(1, "drafter-0", 100.0),
            _sample(2, "verifier-0", 102.0),
            _sample(2, "drafter-0", 102.0),
        ]
        targets = {"verifier-0", "drafter-0"}

        self.assertEqual(
            benchmark_runner._find_complete_sample_round(
                records, targets, require_zero_waiting=True
            ),
            (1, 100.01),
        )
        self.assertEqual(
            benchmark_runner._find_complete_sample_round(
                records, targets, collected_at_or_after=101.0
            ),
            (2, 102.01),
        )

    def test_runner_writes_ordered_completed_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            observer_dir = run_dir / "observer"
            observer_dir.mkdir()
            (observer_dir / "samples.jsonl").write_text(
                json.dumps(
                    {
                        **_sample(0, "verifier-0", 99.0),
                        "observer_started_wall_time": 98.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                run_dir=str(run_dir),
                client_config="client.yaml",
                observer_config="observer.yaml",
                server_manifest="manifest.json",
                observer_interval_s=None,
                observer_request_timeout_s=None,
                observer_loads_include=None,
                observer_barrier_timeout_s=5.0,
                observer_stop_timeout_s=5.0,
            )
            execution = SimpleNamespace(
                started_wall_time=100.0,
                finished_wall_time=102.0,
                started_monotonic_ns=10,
                finished_monotonic_ns=2_000_000_010,
                elapsed_s=2.0,
            )
            events = []

            def wait_for_round(*_args, collected_at_or_after=None, **_kwargs):
                if collected_at_or_after is None:
                    events.append("baseline")
                    return 0, 99.01
                events.append("trailing")
                self.assertEqual(collected_at_or_after, 102.0)
                return 2, 102.01

            def execute_client(*_args):
                events.append("client")
                return {"output_tokens": 8}, execution

            def stop_observer(*_args):
                events.append("stop")
                return json.dumps(
                    {"started_wall_time": 98.5, "finished_wall_time": 103.0}
                ), ""

            process = MagicMock()
            process.poll.return_value = None
            with (
                patch.object(
                    benchmark_runner,
                    "_resolved_configs",
                    return_value=({"batch": {"size": 8}}, {"targets": {}}),
                ),
                patch.object(
                    benchmark_runner,
                    "_normalize_targets",
                    return_value=[{"target_id": "verifier-0"}],
                ),
                patch.object(
                    benchmark_runner, "_observer_command", return_value=["observer"]
                ),
                patch.object(
                    benchmark_runner.subprocess, "Popen", return_value=process
                ),
                patch.object(
                    benchmark_runner,
                    "_wait_for_sample_round",
                    side_effect=wait_for_round,
                ),
                patch.object(
                    benchmark_runner, "execute_client", side_effect=execute_client
                ),
                patch.object(
                    benchmark_runner, "_stop_observer", side_effect=stop_observer
                ),
            ):
                result = benchmark_runner.run(args)

            timeline = json.loads(
                (observer_dir / "bench_timeline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(events, ["baseline", "client", "trailing", "stop"])
            self.assertEqual(result, {"output_tokens": 8})
            self.assertEqual(
                set(timeline),
                {
                    "observer_started_wall_time",
                    "client_started_wall_time",
                    "client_finished_wall_time",
                    "observer_finished_wall_time",
                    "observer_elapsed_s",
                },
            )
            self.assertEqual(timeline["observer_started_wall_time"], 98.5)
            self.assertEqual(timeline["client_started_wall_time"], 100.0)
            self.assertEqual(timeline["client_finished_wall_time"], 102.0)
            self.assertEqual(timeline["observer_finished_wall_time"], 103.0)
            self.assertEqual(timeline["observer_elapsed_s"], 4.5)


if __name__ == "__main__":
    unittest.main()

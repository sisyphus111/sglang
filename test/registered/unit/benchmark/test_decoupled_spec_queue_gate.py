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
_AUDITOR = _load_module(
    "decoupled_spec_audit_run_test",
    _ROOT / "skills" / "audit-decoupled-spec-artifacts" / "scripts" / "audit_run.py",
)


class TestDecoupledSpecQueueGate(CustomTestCase):
    def _write_observability_fixture(
        self, run_dir: Path, *, formal_drafter_waiting: int = 0
    ) -> None:
        observability_dir = run_dir / "observability"
        client_dir = run_dir / "client"
        observability_dir.mkdir(parents=True)
        client_dir.mkdir(parents=True)
        (observability_dir / "resolved_config.json").write_text(
            json.dumps({"interval_s": 1.0}), encoding="utf-8"
        )
        (observability_dir / "summary.json").write_text(
            json.dumps({"error_ct": 0, "target_ct": 2, "sample_ct": 3}),
            encoding="utf-8",
        )
        (client_dir / "formal_window.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "started_wall_time": 100.0,
                    "finished_wall_time": 102.0,
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
                        "role": role,
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
        (observability_dir / "samples.jsonl").write_text(
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

    def test_decode_windows_before_collector_start_are_not_recounted(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_observability_fixture(run_dir)
            summary_path = run_dir / "observability" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(
                {
                    "started_wall_time": 100.0,
                    "decode_metrics": {
                        role: {
                            "window_count": 1,
                            "scheduler_cycle_ms": {"mean": 1.0},
                        }
                        for role in ("verifier", "drafter")
                    },
                }
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            samples_path = run_dir / "observability" / "samples.jsonl"
            records = [
                json.loads(line) for line in samples_path.read_text().splitlines()
            ]
            for record in records:
                record["payload"]["loads"][0]["decode_metrics_windows"] = [
                    {
                        "window_id": 1,
                        "end_time": 99.5,
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

    def test_any_positive_role_log_queue_invalidates_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            logs = run_dir / "logs"
            logs.mkdir()
            (logs / "verifier.log").write_text(
                "Decode batch, #running-req: 64, #queue-req: 0\n",
                encoding="utf-8",
            )
            (logs / "drafter.log").write_text(
                "Prefill batch, #running-req: 48, #queue-req: 16\n",
                encoding="utf-8",
            )
            errors = []
            report = _AUDITOR._check_zero_waiting_queues(run_dir, errors)

        self.assertEqual(report["verifier"]["max_waiting_reqs"], 0)
        self.assertEqual(report["drafter"]["max_waiting_reqs"], 16)
        self.assertTrue(
            any(
                "drafter: role log observed waiting requests" in error
                for error in errors
            )
        )


if __name__ == "__main__":
    unittest.main()

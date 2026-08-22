"""CPU-only tests for decoupled-spec profiler trace analyzers."""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ANALYSIS_ROOT = _REPO_ROOT / "benchmark" / "decoupled_spec" / "analysis"
sys.path.insert(0, str(_ANALYSIS_ROOT))

from analyze_forward_trace import (  # noqa: E402
    _is_device_event,
    _scope_invocations,
    _scope_summary,
    analyze_trace,
)
from analyze_nsys_forward import _require_cupti_tables  # noqa: E402
from compare_terminal_scatter import (  # noqa: E402
    build_metric_row,
    compare_metric_rows,
)
from trace_window_gate import build_window_gate  # noqa: E402


class TestDecoupledSpecTraceAnalysis(CustomTestCase):
    def test_cpu_scope_does_not_mix_gpu_projection(self):
        name = "sglang.decoupled_spec.target_verify"
        events = [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": name,
                "pid": 10,
                "tid": 20,
                "ts": 100.0,
                "dur": 50.0,
                "args": {"External id": 7},
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": name,
                "pid": 0,
                "tid": 80,
                "ts": 200.0,
                "dur": 900.0,
                "args": {"External id": 7, "stream": 80, "device": 0},
            },
        ]
        invocations = _scope_invocations(events)
        summary = _scope_summary(invocations, [events[1]], events)[name]

        self.assertEqual(len(invocations), 1)
        self.assertEqual(summary["cpu_duration_us"]["count"], 1)
        self.assertEqual(summary["cpu_duration_us"]["p50"], 50.0)
        self.assertEqual(summary["gpu_projection_duration_us"]["count"], 1)
        self.assertEqual(summary["gpu_projection_duration_us"]["p50"], 900.0)
        self.assertFalse(_is_device_event(events[1]))

    def test_cpu_only_torch_trace_is_marked_gpu_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "cpu-only.trace.json"
            trigger_path = root / "trigger.txt"
            log_path = root / "verifier.log"
            trace_path.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "user_annotation",
                                "name": "sglang.decoupled_spec.target_verify",
                                "pid": 10,
                                "tid": 20,
                                "ts": 100.0,
                                "dur": 50.0,
                                "args": {"External id": 7},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            trigger_path.write_text(
                "2026-08-21T22:25:37,400000000+08:00\n", encoding="utf-8"
            )
            log_path.write_text(
                "[2026-08-21 22:25:37 TP0] Decode batch, #running-req: 8, "
                "cuda graph: True\n",
                encoding="utf-8",
            )

            report = analyze_trace(
                trace_path,
                expected_bs=8,
                trigger_path=trigger_path,
                verifier_log_path=log_path,
            )
            self.assertTrue(report["window_gate"]["passed"])
            self.assertFalse(report["activity_gate"]["gpu_activity_present"])
            self.assertFalse(report["activity_gate"]["gpu_structure_valid"])

    def test_nsys_missing_cupti_tables_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            sqlite_path = Path(directory) / "empty.sqlite"
            connection = sqlite3.connect(sqlite_path)
            try:
                with self.assertRaisesRegex(
                    ValueError, "missing required CUPTI tables"
                ):
                    _require_cupti_tables(connection)
            finally:
                connection.close()

    def test_nsys_fold_grid_y_batch_mismatch_fails_gate(self):
        gate = build_window_gate(expected_bs=8, fold_grid_y_values=[1, 1, 1])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["inferred_bs_from_fold_grid_y"], 1)
        self.assertIn("fold gridY batch mismatch", gate["errors"][0])

    def test_scatter_comparison_factorizes_tps(self):
        baseline = build_metric_row(
            {
                "batch_elapsed_s": 4.0,
                "completion_tokens": 80,
                "output_tokens_per_s": 20.0,
                "spec_verify_ct": 40,
                "spec_num_proposed_drafts": 60,
                "spec_num_correct_drafts": 40,
                "spec_draft_occupancy_rate": 0.5,
                "spec_accept_rate": 2 / 3,
            }
        )
        candidate = build_metric_row(
            {
                "batch_elapsed_s": 2.5,
                "completion_tokens": 80,
                "output_tokens_per_s": 32.0,
                "spec_verify_ct": 50,
                "spec_num_proposed_drafts": 50,
                "spec_num_correct_drafts": 40,
                "spec_draft_occupancy_rate": 1 / 3,
                "spec_accept_rate": 0.8,
            }
        )
        comparison = compare_metric_rows(baseline, candidate)
        factorization = comparison["tps_ratio_factorization"]
        self.assertAlmostEqual(factorization["observed"], 1.6)
        self.assertAlmostEqual(factorization["product"], 1.6)
        self.assertAlmostEqual(factorization["absolute_error"], 0.0)
        self.assertLess(
            comparison["spec_draft_occupancy_rate"]["relative_change_pct"], 0
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused tests for fluid oracle replay."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from replay_oracle import _write_summary, parse_args, replay


def point(label: str, step: int, ctx: int, tokens: float, latency_ms: float):
    return {
        "label": label,
        "allow_partial": "True",
        "dynamic": "False",
        "active_step": str(step),
        "point_index": "1",
        "batch_size": "8",
        "ctx_per_req": str(ctx),
        "matched_profile_batch_size": "8",
        "matched_profile_ctx_len": str(ctx),
        "observed_itl_ms": str(latency_ms),
        "output_tokens": str(tokens),
        "queue_req": "0",
    }


class ReplayTest(unittest.TestCase):
    def test_markdown_reports_decision_grade_before_speedup(self):
        summary = {
            "decision_grade": False,
            "decision_grade_reasons": ["token drift"],
            "reference_label": "ap1_static_dl1",
            "candidate_set_scope": "full_matrix",
            "reference_e2e_time_s": 10.0,
            "oracle_e2e_time_s": 8.0,
            "oracle_speedup": 1.25,
            "target_capture": 0.7,
            "target_capture_time_s": 8.6,
            "candidate_coverage_token_share": 1.0,
            "all_tier_coverage_token_share": 1.0,
            "fallback_token_share": 0.0,
            "queued_token_share": 0.0,
            "max_queue_req": 0,
            "static_queued_point_count": 0,
            "static_max_queue_req": 0,
            "static_queued_points": [],
            "incomplete_states": [],
            "fallback_states": [],
            "oracle_step_token_share": {"1": 1.0},
            "dynamic_label": None,
            "assumption": "fixed token-progress state path",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_summary(path, summary)
            content = path.read_text()
        self.assertLess(
            content.index("Decision-grade"), content.index("Oracle speedup")
        )

    def test_selects_per_state_tier_and_interpolates_time_saved(self):
        rows = []
        for _ in range(2):
            rows.extend(
                [
                    point("ap1_static_dl1", 1, 1024, 10, 5),
                    point("ap1_static_dl1", 1, 2048, 10, 10),
                    point("ap1_static_dl2", 2, 1024, 10, 10),
                    point("ap1_static_dl2", 2, 2048, 10, 5),
                ]
            )
        e2e = [
            {
                "label": "ap1_static_dl1",
                "allow_partial": "True",
                "dynamic": "False",
                "max_step": "1",
                "output_throughput_tok_per_s": str(20 / 15),
                "generation_time_s": "15",
                "total_generated_tokens": "20",
            },
            {
                "label": "ap1_static_dl2",
                "allow_partial": "True",
                "dynamic": "False",
                "max_step": "2",
                "output_throughput_tok_per_s": "1.2",
                "generation_time_s": str(20 / 1.2),
                "total_generated_tokens": "20",
            },
            {
                "label": "ap1_dynamic_dl2",
                "allow_partial": "True",
                "dynamic": "True",
                "max_step": "2",
                "output_throughput_tok_per_s": str(20 / 12),
                "generation_time_s": "12",
                "total_generated_tokens": "20",
            },
        ]

        summary, _, replay_rows = replay(
            rows,
            e2e,
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
        )

        self.assertAlmostEqual(summary["oracle_speedup"], 1.5)
        self.assertAlmostEqual(summary["oracle_e2e_time_s"], 10.0)
        self.assertAlmostEqual(summary["target_capture_time_s"], 11.5)
        self.assertAlmostEqual(summary["dynamic_time_saving_capture"], 0.6)
        self.assertEqual([row["oracle_step"] for row in replay_rows], [1, 2, 1, 2])
        self.assertEqual(summary["all_tier_coverage_token_share"], 1.0)
        self.assertEqual(summary["fallback_token_share"], 0.0)
        self.assertEqual(summary["queued_token_share"], 0.0)
        self.assertEqual(summary["fallback_states"], [])
        self.assertEqual(summary["oracle_step_token_share"], {"1": 0.5, "2": 0.5})
        self.assertTrue(summary["decision_grade"])

    def test_queueing_prevents_decision_grade_and_states_are_auditable(self):
        rows = []
        for step in (1, 2):
            for _ in range(2):
                row = point(f"ap1_static_dl{step}", step, 1024, 10, 5)
                row["queue_req"] = "3"
                rows.append(row)
        e2e = [
            {
                "label": f"ap1_static_dl{step}",
                "allow_partial": "True",
                "dynamic": "False",
                "max_step": str(step),
                "output_throughput_tok_per_s": "2",
                "generation_time_s": "10",
                "total_generated_tokens": "20",
            }
            for step in (1, 2)
        ]

        summary, _, _ = replay(
            rows,
            e2e,
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
        )

        self.assertFalse(summary["decision_grade"])
        self.assertEqual(summary["queued_token_share"], 1.0)
        self.assertEqual(summary["max_queue_req"], 3)
        self.assertIn(
            "static matrix contains queued requests",
            summary["decision_grade_reasons"],
        )
        self.assertEqual(summary["state_coverage"][0]["available_steps"], [1, 2])
        self.assertEqual(summary["state_coverage"][0]["missing_steps"], [])

    def test_reports_exact_fallback_state_identity(self):
        rows = [
            point("ap1_static_dl1", 1, 1024, 10, 5),
            point("ap1_static_dl1", 1, 1024, 10, 5),
            point("ap1_static_dl1", 1, 2048, 10, 5),
            point("ap1_static_dl2", 2, 1024, 10, 5),
            point("ap1_static_dl2", 2, 1024, 10, 5),
        ]
        e2e = [
            {
                "label": f"ap1_static_dl{step}",
                "allow_partial": "True",
                "dynamic": "False",
                "max_step": str(step),
                "output_throughput_tok_per_s": "2",
                "generation_time_s": "15",
                "total_generated_tokens": "30",
            }
            for step in (1, 2)
        ]

        summary, _, _ = replay(
            rows,
            e2e,
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
        )

        self.assertEqual(len(summary["fallback_states"]), 1)
        fallback = summary["fallback_states"][0]
        self.assertEqual((fallback["batch_slot"], fallback["ctx_bucket"]), (8, 2048))
        self.assertEqual(fallback["available_steps"], [])
        self.assertEqual(fallback["missing_steps"], [1, 2])

    def test_queueing_above_latency_cutoff_prevents_decision_grade(self):
        rows = []
        for step in (1, 2):
            rows.extend(
                [
                    point(f"ap1_static_dl{step}", step, 1024, 10, 5),
                    point(f"ap1_static_dl{step}", step, 1024, 10, 5),
                ]
            )
        excluded = point("ap1_static_dl1", 1, 1024, 10, 101)
        excluded["queue_req"] = "7"
        excluded["point_index"] = "99"
        rows.append(excluded)
        summary, _, _ = replay(
            rows,
            self._two_static_e2e(),
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
        )

        self.assertFalse(summary["decision_grade"])
        self.assertEqual(summary["queued_token_share"], 0.0)
        self.assertEqual(summary["static_queued_point_count"], 1)
        self.assertEqual(summary["static_max_queue_req"], 7)
        self.assertEqual(summary["static_queued_points"][0]["point_index"], 99)

    def test_queueing_in_non_reference_tier_prevents_decision_grade(self):
        rows = []
        for step in (1, 2):
            for _ in range(2):
                row = point(f"ap1_static_dl{step}", step, 1024, 10, 5)
                if step == 2:
                    row["queue_req"] = "4"
                rows.append(row)
        summary, _, _ = replay(
            rows,
            self._two_static_e2e(),
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
        )

        self.assertFalse(summary["decision_grade"])
        self.assertEqual(summary["queued_token_share"], 0.0)
        self.assertEqual(summary["static_queued_point_count"], 2)
        self.assertEqual(summary["static_max_queue_req"], 4)
        self.assertEqual(
            {item["label"] for item in summary["static_queued_points"]},
            {"ap1_static_dl2"},
        )

    def test_explicit_sensitivity_filter_preserves_actions_and_is_labeled(self):
        rows = []
        for step, latency in ((0, 20), (1, 5), (2, 5)):
            for _ in range(2):
                rows.append(point(f"ap1_static_dl{step}", step, 1024, 10, latency))
        e2e = self._two_static_e2e()
        e2e.append(
            {
                "label": "ap1_static_dl0",
                "allow_partial": "True",
                "dynamic": "False",
                "max_step": "0",
                "output_throughput_tok_per_s": "0.5",
                "generation_time_s": "200",
                "total_generated_tokens": "100",
            }
        )
        full, _, full_replay = replay(
            rows,
            e2e,
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
        )
        sensitivity, _, sensitivity_replay = replay(
            rows,
            e2e,
            reference_label="ap1_static_dl1",
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
            static_step_filter={1, 2},
        )

        self.assertFalse(full["decision_grade"])
        self.assertTrue(sensitivity["decision_grade"])
        self.assertEqual(sensitivity["candidate_set_scope"], "filtered_sensitivity")
        self.assertEqual(sensitivity["static_steps"], [1, 2])
        self.assertEqual(
            [row["oracle_step"] for row in full_replay],
            [row["oracle_step"] for row in sensitivity_replay],
        )
        self.assertEqual(
            [row["oracle_time_ms"] for row in full_replay],
            [row["oracle_time_ms"] for row in sensitivity_replay],
        )

        e2e[2]["output_throughput_tok_per_s"] = "100"
        auto, _, _ = replay(
            rows,
            e2e,
            reference_label=None,
            target_capture=0.7,
            min_points=2,
            latency_cutoff_ms=100,
            static_step_filter={1, 2},
        )
        self.assertEqual(auto["reference_label"], "ap1_static_dl1")

    def test_sensitivity_cli_requires_distinct_output_dir(self):
        argv = [
            "replay_oracle.py",
            "--analysis-dir",
            "/tmp/analysis",
            "--static-steps",
            "1,2,3",
        ]
        with (
            patch("sys.argv", argv),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    @staticmethod
    def _two_static_e2e():
        return [
            {
                "label": f"ap1_static_dl{step}",
                "allow_partial": "True",
                "dynamic": "False",
                "max_step": str(step),
                "output_throughput_tok_per_s": "2",
                "generation_time_s": "10",
                "total_generated_tokens": "20",
            }
            for step in (1, 2)
        ]

    def test_rejects_missing_reference(self):
        with self.assertRaisesRegex(ValueError, "reference static case not found"):
            replay(
                [],
                [
                    {
                        "label": "ap1_static_dl1",
                        "allow_partial": "True",
                        "dynamic": "False",
                        "max_step": "1",
                        "output_throughput_tok_per_s": "1",
                        "generation_time_s": "1",
                        "total_generated_tokens": "1",
                    }
                ],
                reference_label="missing",
                target_capture=0.7,
                min_points=1,
                latency_cutoff_ms=100,
            )


if __name__ == "__main__":
    unittest.main()

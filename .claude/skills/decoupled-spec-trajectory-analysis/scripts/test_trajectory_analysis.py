#!/usr/bin/env python3
"""Focused tests for normalized decoupled-spec trajectories."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis_common import (
    Case,
    ProfileTable,
    parse_candidate_scores,
    parse_decode_points,
    parse_switches,
)
from analysis_render import plot_trajectories
from analyze import filter_points, filter_trajectory_points, smooth_points


class ProfileTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = ProfileTable(
            [
                {
                    "batch_size": bs,
                    "steps": 2,
                    "ctx_len": ctx,
                    "cost_ms": bs + ctx / 1000,
                }
                for bs in (8, 16)
                for ctx in (1000, 3000)
            ]
        )

    def test_bs_ceil_ctx_nearest_and_lower_tie(self) -> None:
        match = self.table.lookup(batch_size=9, steps=2, ctx_len=2000)
        self.assertEqual(match.batch_size, 16)
        self.assertEqual(match.ctx_len, 1000)
        self.assertEqual(match.cost_ms, 17.0)

    def test_bs_clamps_above_largest(self) -> None:
        self.assertEqual(self.table.lookup(64, 2, 4000).batch_size, 16)


class LogParsingTest(unittest.TestCase):
    def test_trajectory_keeps_small_batch_tail_points(self) -> None:
        rows = [
            {"label": "case", "batch_size": 64, "observed_itl_ms": 20.0},
            {"label": "case", "batch_size": 8, "observed_itl_ms": 25.0},
            {"label": "case", "batch_size": 1, "observed_itl_ms": 120.0},
        ]
        profile_fit = filter_points(rows, 100.0, include_partial_batches=False)
        trajectory = filter_trajectory_points(rows, 100.0)
        self.assertEqual([row["batch_size"] for row in profile_fit], [64])
        self.assertEqual([row["batch_size"] for row in trajectory], [64, 8])

    def test_trajectory_plot_contains_throughput_acclen_and_step(self) -> None:
        rows = [
            {
                "label": "ap1_dynamic_dl3",
                "elapsed_s": float(index),
                "observed_throughput_tok_s": 1000.0 + index,
                "accept_len": 1.5 + index / 10,
                "active_step": 2 + index % 2,
                "observed_itl_ms": 20.0,
                "modeled_itl_ms": 19.0,
                "modeled_throughput_tok_s": 1050.0,
            }
            for index in range(3)
        ]
        smooth = smooth_points(rows, 2)
        with tempfile.TemporaryDirectory() as directory:
            paths = plot_trajectories(rows, smooth, Path(directory), 2)
            content = paths[0].read_text()
            self.assertTrue(
                (Path(directory) / "trajectory_ap1_dynamic_dl3.png").exists()
            )
        self.assertIn("Observed throughput", content)
        self.assertIn("Acceptance length", content)
        self.assertIn("Active draft length", content)

    def test_missing_acclen_is_preserved_as_unavailable(self) -> None:
        rows = [
            {
                "label": "ap0_static_dl0",
                "elapsed_s": 0.0,
                "observed_throughput_tok_s": 1000.0,
                "accept_len": "",
                "active_step": 0,
                "observed_itl_ms": 20.0,
                "modeled_itl_ms": 20.0,
                "modeled_throughput_tok_s": 1000.0,
            }
        ]
        smooth = smooth_points(rows, 1)
        self.assertEqual(smooth[0]["accept_len_smooth"], "")
        with tempfile.TemporaryDirectory() as directory:
            path = plot_trajectories(rows, smooth, Path(directory), 1)[0]
            content = path.read_text()
        self.assertIn("accept_len unavailable", content)

    def test_scheduler_modeled_throughput_is_not_charged_twice(self) -> None:
        line = (
            "[2026-07-10 04:44:07 TP0] Decode batch, #running-req: 8, "
            "#full token: 8000, iter latency (ms): 20.0000, accept len: 2.00, "
            "valid draft len: 1.00, accept rate: 1.00, "
            "modeled throughput (token/s): 1000.00, "
            "modeled step: 1, "
            "gen throughput (token/s): 800.00, #queue-req: 0\n"
        )
        table = ProfileTable(
            [
                {"batch_size": 8, "steps": 1, "ctx_len": 1000, "cost_ms": 12.0},
                {"batch_size": 8, "steps": 2, "ctx_len": 1000, "cost_ms": 13.0},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap0_dynamic_dl2.log"
            path.write_text(line)
            rows = parse_decode_points(
                Case("ap0_dynamic_dl2", path, False, True, 2), table, 3.0
            )
        self.assertEqual(rows[0]["modeled_itl_ms"], 16.0)
        self.assertEqual(rows[0]["model_source"], "scheduler_modeled_throughput")
        self.assertEqual(rows[0]["active_step"], 1)
        self.assertEqual(rows[0]["profile_cost_ms"], 12.0)

    def test_static_profile_cost_adds_cpu_overhead_once(self) -> None:
        line = (
            "[2026-07-10 04:44:07 TP0] Decode batch, #running-req: 8, "
            "#full token: 8000, iter latency (ms): 20.0000, accept len: 2.00, "
            "valid draft len: 1.00, accept rate: 1.00, "
            "gen throughput (token/s): 800.00, #queue-req: 0\n"
        )
        table = ProfileTable(
            [{"batch_size": 8, "steps": 2, "ctx_len": 1000, "cost_ms": 13.0}]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap0_static_dl2.log"
            path.write_text(line)
            rows = parse_decode_points(
                Case("ap0_static_dl2", path, False, False, 2), table, 3.0
            )
        self.assertEqual(rows[0]["modeled_itl_ms"], 16.0)
        self.assertEqual(rows[0]["model_source"], "profile_lookup_plus_overhead")
        self.assertEqual(rows[0]["active_step"], 2)

    def test_runtime_switch_updates_dynamic_active_step_timeline(self) -> None:
        log = "\n".join(
            [
                "[2026-07-10 04:44:06 TP0] Switch decoupled verifier adaptive "
                "state: steps 3 -> 2, draft_tokens 4 -> 3",
                "[2026-07-10 04:44:07 TP0] Decode batch, #running-req: 8, "
                "#full token: 8000, iter latency (ms): 20.0000, accept len: 2.00, "
                "valid draft len: 1.00, accept rate: 1.00, "
                "gen throughput (token/s): 800.00, #queue-req: 0",
            ]
        )
        table = ProfileTable(
            [{"batch_size": 8, "steps": 2, "ctx_len": 1000, "cost_ms": 13.0}]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap1_dynamic_dl3.log"
            path.write_text(log)
            case = Case("ap1_dynamic_dl3", path, True, True, 3)
            rows = parse_decode_points(case, table, 3.0)
            switches = parse_switches(case)
        self.assertEqual(rows[0]["active_step"], 2)
        self.assertEqual(switches[0]["from_steps"], 3)
        self.assertEqual(switches[0]["to_steps"], 2)
        self.assertEqual(switches[0]["reason"], "runtime_adaptive_state")

    def test_switch_candidates_preserve_acceptance_positions(self) -> None:
        raw = (
            "[S=1:E=1.800/cost=20.0000ms=0.090000,profile_cost=17.0000ms,"
            "cpu_overhead=3.0000ms,ctx=200->1024,rates=[p1=0.800:ema]*, "
            "S=3:E=2.400/cost=30.0000ms=0.080000,profile_cost=27.0000ms,"
            "cpu_overhead=3.0000ms,ctx=200->1024,"
            "rates=[p1=0.800:ema,p2=0.400:projected,p3=0.200:probe],"
            "expected_source=tier_ema,tier_updates=15:blocked, "
            "S=2:E=2.100/cost=25.0000ms=0.084000,profile_cost=22.0000ms,"
            "cpu_overhead=3.0000ms,ctx=200->1024,rates=[p1=0.700:ema],"
            "expected_source=tier_ema,tier_updates=10*]"
        )
        candidates = parse_candidate_scores(raw)
        self.assertEqual(candidates[0]["position_accept_rates"][0]["position"], 1)
        self.assertEqual(
            candidates[1]["position_accept_rates"][1]["source"], "projected"
        )
        self.assertTrue(candidates[0]["selected"])
        self.assertFalse(candidates[1]["eligible"])
        self.assertEqual(candidates[1]["expected_source"], "tier_ema")
        self.assertEqual(candidates[1]["tier_updates"], 15)
        self.assertTrue(candidates[2]["selected"])

    def test_switch_parser_accepts_decision_reason(self) -> None:
        line = (
            "[2026-07-11 10:44:37 TP0] Decoupled verifier throughput-aware "
            "step switch: steps 1 -> 2, bs=64, avg_ctx_len=163, "
            "batch_count=10, current_score=0.060863, best_score=0.063773, "
            "best_over_current=1.047813, reason=acceptance_probe, "
            "scores=[S=1:E=1.866/cost=30.6574ms=0.060863,"
            "profile_cost=27.6574ms,cpu_overhead=3.0000ms,ctx=163->1024,"
            "rates=[p1=0.866:ema]*]"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap1_dynamic_dl3.log"
            path.write_text(line)
            rows = parse_switches(Case("ap1_dynamic_dl3", path, True, True, 3))
        self.assertEqual(rows[0]["reason"], "acceptance_probe")

    def test_detailed_and_generic_runtime_switch_are_counted_once(self) -> None:
        log = "\n".join(
            [
                "[2026-07-11 10:44:37 TP0] Decoupled verifier throughput-aware "
                "step switch: steps 1 -> 2, bs=64, avg_ctx_len=163, "
                "batch_count=10, current_score=0.060863, best_score=0.063773, "
                "best_over_current=1.047813, reason=score_hysteresis, "
                "scores=[S=1:E=1.866/cost=30.6574ms=0.060863,"
                "profile_cost=27.6574ms,cpu_overhead=3.0000ms,ctx=163->1024,"
                "rates=[p1=0.866:ema],expected_source=tier_ema_decay,"
                "tier_updates=10,tier_age=5*]",
                "[2026-07-11 10:44:38 TP0] Switch decoupled verifier adaptive "
                "state: steps 1 -> 2, draft_tokens 2 -> 3",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap1_dynamic_dl3.log"
            path.write_text(log)
            rows = parse_switches(Case("ap1_dynamic_dl3", path, True, True, 3))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"], "2026-07-11 10:44:37")
        candidate = json.loads(rows[0]["candidate_scores_json"])[0]
        self.assertEqual(candidate["expected_source"], "tier_ema_decay")
        self.assertEqual(candidate["tier_age_batches"], 5)

    def test_intervening_reverse_preserves_repeated_direction(self) -> None:
        detailed = (
            "Decoupled verifier throughput-aware step switch: steps {old} -> {new}, "
            "bs=64, avg_ctx_len=163, batch_count=10, current_score=0.1, "
            "best_score=0.2, best_over_current=2.0, reason=score_hysteresis, "
            "scores=[]"
        )
        generic = (
            "Switch decoupled verifier adaptive state: steps {old} -> {new}, "
            "draft_tokens 2 -> 3"
        )
        transitions = [(1, 2), (2, 1), (1, 2)]
        lines = []
        for second, (old, new) in enumerate(transitions, start=37):
            lines.append(
                f"[2026-07-11 10:44:{second} TP0] " + detailed.format(old=old, new=new)
            )
            lines.append(
                f"[2026-07-11 10:44:{second + 1} TP0] "
                + generic.format(old=old, new=new)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap1_dynamic_dl3.log"
            path.write_text("\n".join(lines))
            rows = parse_switches(Case("ap1_dynamic_dl3", path, True, True, 3))

        self.assertEqual(
            [(row["from_steps"], row["to_steps"]) for row in rows], transitions
        )


if __name__ == "__main__":
    unittest.main()

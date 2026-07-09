#!/usr/bin/env python3
"""Focused unit tests for controller-compatible analysis primitives."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis_common import Case, ProfileTable, parse_candidate_scores, parse_decode_points


class ProfileTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = ProfileTable(
            [
                {"batch_size": bs, "steps": 2, "ctx_len": ctx, "cost_ms": bs + ctx / 1000}
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
    def test_scheduler_modeled_throughput_is_not_charged_twice(self) -> None:
        line = (
            "[2026-07-10 04:44:07 TP0] Decode batch, #running-req: 8, "
            "#full token: 8000, iter latency (ms): 20.0000, accept len: 2.00, "
            "valid draft len: 1.00, accept rate: 1.00, "
            "modeled throughput (token/s): 1000.00, "
            "gen throughput (token/s): 800.00, #queue-req: 0\n"
        )
        table = ProfileTable(
            [{"batch_size": 8, "steps": 2, "ctx_len": 1000, "cost_ms": 13.0}]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ap0_dynamic_dl2.log"
            path.write_text(line)
            rows = parse_decode_points(
                Case("ap0_dynamic_dl2", path, False, True, 2), table, 3.0
            )
        self.assertEqual(rows[0]["modeled_itl_ms"], 16.0)
        self.assertEqual(rows[0]["model_source"], "scheduler_modeled_throughput")

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

    def test_switch_candidates_preserve_acceptance_positions(self) -> None:
        raw = (
            "[S=1:E=1.800/cost=20.0000ms=0.090000,profile_cost=17.0000ms,"
            "cpu_overhead=3.0000ms,ctx=200->1024,rates=[p1=0.800:win]*]"
        )
        candidates = parse_candidate_scores(raw)
        self.assertEqual(candidates[0]["position_accept_rates"][0]["position"], 1)
        self.assertTrue(candidates[0]["selected"])


if __name__ == "__main__":
    unittest.main()

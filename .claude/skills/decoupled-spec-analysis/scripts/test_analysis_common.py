#!/usr/bin/env python3
"""Focused unit tests for controller-compatible analysis primitives."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis_common import (
    Case,
    ProfileTable,
    parse_candidate_scores,
    parse_decode_points,
    parse_switches,
)
from run_matrix import base_command, build_env


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
        self.assertEqual(candidates[1]["position_accept_rates"][1]["source"], "projected")
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
                f"[2026-07-11 10:44:{second} TP0] "
                + detailed.format(old=old, new=new)
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


class MatrixCommandTest(unittest.TestCase):
    @staticmethod
    def _config(strategy: str) -> dict:
        return {
            "paths": {
                "repo": "/repo",
                "entrypoint": "single-node.py",
                "dataset": "/data/input.parquet",
                "target_model": "/models/target",
                "draft_model": "/models/draft",
                "profile": "/profiles/cost.json",
            },
            "parallelism": {
                "target_tp_size": 4,
                "draft_tp_size": 1,
                "n_gpu_per_node": 5,
                "verify_ngpus": 4,
                "draft_ngpus": 1,
            },
            "workload": {
                "dataset_format": "dapo_math_17k",
                "batch_size": 64,
                "max_running_requests": 65,
                "max_new_tokens": 32768,
                "temperature": 1.0,
                "sampling_seed": 42,
                "deterministic": True,
            },
            "profile": {
                "capture_bs": [64],
                "ctx_lens": [32768],
                "steps": [0, 1, 2, 3],
            },
            "adaptive": {
                "strategy": strategy,
                "config": "/configs/ema.json",
            },
        }

    def test_ema_dynamic_command_uses_accept_length_config(self) -> None:
        command = base_command(
            self._config("ema"), Path("/output"), steps=3, dynamic=True
        )

        self.assertIn("ema", command)
        self.assertIn("--speculative-adaptive-config", command)
        self.assertNotIn("--decoupled-verify-throughput-profile-path", command)

    def test_profile_command_can_force_throughput_aware(self) -> None:
        command = base_command(
            self._config("ema"),
            Path("/output"),
            steps=3,
            dynamic=True,
            adaptive_strategy="throughput_aware",
        )

        self.assertIn("throughput_aware", command)
        self.assertIn("--decoupled-verify-throughput-profile-path", command)
        self.assertNotIn("--speculative-adaptive-config", command)

    def test_all_formal_commands_lock_sampling_seed_and_determinism(self) -> None:
        config = self._config("ema")
        cases = ((0, False), (1, False), (2, False), (3, False), (3, True))
        for steps, dynamic in cases:
            with self.subTest(steps=steps, dynamic=dynamic):
                command = base_command(
                    config, Path("/output"), steps=steps, dynamic=dynamic
                )
                seed_index = command.index("--sampling-seed")
                self.assertEqual(command[seed_index + 1], "42")
                self.assertIn("--deterministic", command)

    def test_child_environment_prefers_configured_worktree(self) -> None:
        config = self._config("ema")
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/ambient/python"}):
            env = build_env(config, allow_partial=True)

        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep),
            ["/repo/python", "/ambient/python"],
        )


if __name__ == "__main__":
    unittest.main()

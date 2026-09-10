from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.speculative.decoupled_verify_cost_model import (
    BatchSizeCostTable,
    DraftPositionStatsTracker,
    score_verify_candidates,
)
from sglang.srt.speculative.decoupled_verify_profile import (
    DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR,
    DECOUPLED_VERIFY_PROFILE_COST_SCOPE,
    DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE,
    DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE,
    DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER,
    DECOUPLED_VERIFY_PROFILE_KIND,
    DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK,
    DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION,
    DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE,
    resolve_profile_input_context_len,
    validate_decoupled_verify_profile,
)
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    DecoupledVerifyThroughputController,
    load_decoupled_verify_adaptive_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _profile() -> dict:
    return {
        "schema_version": DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION,
        "kind": DECOUPLED_VERIFY_PROFILE_KIND,
        "status": "complete",
        "profile_abi_version": 1,
        "measurement_clock": DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK,
        "cost_scope": DECOUPLED_VERIFY_PROFILE_COST_SCOPE,
        "cost_estimator": DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR,
        "draft_provider": DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER,
        "profile_control_plane": DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE,
        "context_anchor_mode": DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE,
        "trajectory_acceptance": DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE,
        "fingerprint": {},
        "points": [
            {
                "step": step,
                "batch_size": 8,
                "context_len": 1024,
                "cost_ms": 10.0 + step,
                "cuda_graph_fraction": 1.0,
                "mean_selected_draft_length": float(step),
                "mean_accept_length": float(step + 1),
            }
            for step in range(4)
        ],
    }


class TestDecoupledVerifyAdaptive(CustomTestCase):
    def test_measurement_midpoint(self):
        self.assertEqual(
            resolve_profile_input_context_len(
                target_context_len=1024,
                step=3,
                warmup_iters=10,
                measure_iters=50,
            ),
            881,
        )

    def test_profile_validation_rejects_partial_for_production(self):
        profile = _profile()
        profile["status"] = "partial"
        with self.assertRaisesRegex(ValueError, "complete profile"):
            validate_decoupled_verify_profile(profile, require_complete=True)

    def test_profile_validation_rejects_obsolete_tail_producer(self):
        profile = _profile()
        profile["draft_provider"] = "daemon_synthetic_tail_producer"
        with self.assertRaisesRegex(ValueError, "forward-stream mock"):
            validate_decoupled_verify_profile(profile, require_complete=True)

    def test_profile_validation_rejects_eager_measurement(self):
        profile = _profile()
        profile["points"][2]["cuda_graph_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "only CUDA Graph"):
            validate_decoupled_verify_profile(profile, require_complete=True)

    def test_supply_and_conditional_accept_model(self):
        tracker = DraftPositionStatsTracker(max_steps=3, ema_alpha=1.0)
        tracker.update(
            correct_drafts=[3, 1],
            consumable_drafts=[3, 2],
            applied_steps=3,
        )
        self.assertEqual(tracker.snapshot()["supply_ema"], [1.0, 1.0, 0.5])
        self.assertEqual(tracker.snapshot()["conditional_accept_ema"], [1.0, 0.5, 1.0])
        self.assertEqual(tracker.expected_accept_length(3), 3.0)

    def test_inactive_positions_do_not_decay_supply_ema(self):
        tracker = DraftPositionStatsTracker(max_steps=3, ema_alpha=1.0)
        tracker.update(
            correct_drafts=[3],
            consumable_drafts=[3],
            applied_steps=3,
        )
        tracker.update(
            correct_drafts=[1],
            consumable_drafts=[1],
            applied_steps=1,
        )
        self.assertEqual(tracker.snapshot()["supply_ema"], [1.0, 1.0, 1.0])

    def test_candidate_score_uses_scheduler_cycle_itl(self):
        tracker = DraftPositionStatsTracker(max_steps=1, ema_alpha=1.0)
        tracker.update(
            correct_drafts=[1] * 8,
            consumable_drafts=[1] * 8,
            applied_steps=1,
        )
        costs = BatchSizeCostTable()
        costs.set(batch_size=8, steps=0, context_len=1024, cost_ms=10)
        costs.set(batch_size=8, steps=1, context_len=1024, cost_ms=15)
        rows = score_verify_candidates(
            tracker=tracker,
            cost_table=costs,
            candidate_steps=[0, 1],
            batch_size=8,
            context_len=1024,
        )
        self.assertEqual(rows[0]["modeled_tps"], 800.0)
        self.assertAlmostEqual(rows[1]["modeled_tps"], 8 * 2 * 1000 / 15)

    def test_controller_starts_from_maximum_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "adaptive.json"
            config_path.write_text(
                json.dumps(
                    {
                        "candidate_steps": [0, 1, 2, 3],
                        "ema_alpha": 1.0,
                        "update_interval_rounds": 1,
                        "switch_hysteresis": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            controller = DecoupledVerifyThroughputController(
                max_steps=3,
                config_path=str(config_path),
                profile=_profile(),
                profile_sha256="abc",
            )
            self.assertEqual(controller.current_steps, 3)
            controller.observe(
                correct_drafts=[3] * 8,
                consumable_drafts=[3] * 8,
                applied_steps=3,
            )
            decision = controller.decide(batch_size=8, context_len=1024)
            self.assertEqual(decision["new_steps"], 3)

    def test_controller_downshifts_without_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "adaptive.json"
            config_path.write_text(
                json.dumps(
                    {
                        "candidate_steps": [0, 1, 2, 3],
                        "ema_alpha": 1.0,
                        "update_interval_rounds": 1,
                        "switch_hysteresis": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            controller = DecoupledVerifyThroughputController(
                max_steps=3,
                config_path=str(config_path),
                profile=_profile(),
                profile_sha256="abc",
            )
            controller.observe(
                correct_drafts=[0] * 8,
                consumable_drafts=[0] * 8,
                applied_steps=3,
            )
            decision = controller.decide(batch_size=8, context_len=1024)
            self.assertEqual(decision["new_steps"], 0)
            self.assertEqual(decision["reason"], "score_hysteresis")

    def test_zero_supply_is_a_zero_contribution_observation(self):
        tracker = DraftPositionStatsTracker(max_steps=3, ema_alpha=1.0)
        tracker.update(
            correct_drafts=[0] * 8,
            consumable_drafts=[0] * 8,
            applied_steps=3,
        )
        self.assertEqual(tracker.expected_accept_length(3), 1.0)

    def test_obsolete_warmup_and_initial_step_config_is_rejected(self):
        for key, value in (
            ("initial_steps", 1),
            ("warmup_updates", 1),
            ("warmup_batches", 1),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "adaptive.json"
                config_path.write_text(
                    json.dumps({"candidate_steps": [0, 1, 2, 3], key: value}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "obsolete config keys"):
                    load_decoupled_verify_adaptive_config(
                        str(config_path), max_steps=3
                    )


if __name__ == "__main__":
    unittest.main()

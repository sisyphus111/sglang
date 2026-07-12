import unittest

from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    BatchSizeCostTable,
    DecoupledVerifyThroughputAwareController,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES,
    DraftPositionStatsTracker,
    pick_best_step,
    pick_best_step_with_hysteresis,
    parse_decoupled_verify_throughput_profile_ctx_lens,
    score_decoupled_verify_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Worker:
    def __init__(self, initial_steps: int):
        self.speculative_num_steps = initial_steps
        self.applied_steps = []

    def apply_runtime_state(self, state):
        self.speculative_num_steps = state.speculative_num_steps
        self.applied_steps.append(state.speculative_num_steps)


def _state(steps: int) -> SpecRuntimeState:
    return SpecRuntimeState.for_decoupled_verify(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        target_attn_backend=object(),
        target_graph_runner=object(),
    )


def _observe(
    controller: DecoupledVerifyThroughputAwareController,
    num_correct: list[int],
    num_consumable: list[int],
    verified_steps: int,
) -> None:
    controller.on_verify_complete(
        num_correct,
        num_consumable_drafts_per_req=num_consumable,
        verified_steps=verified_steps,
        batch_size=len(num_correct),
    )


class TestBatchSizeCostTable(CustomTestCase):
    def test_lookup_uses_exact_ceiling_and_largest_profiled_bs(self):
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=8, steps=0, ctx_len=256, cost_ms=12.0)

        self.assertEqual(table.lookup(batch_size=4, steps=0, ctx_len=256), 10.0)
        self.assertEqual(table.lookup(batch_size=5, steps=0, ctx_len=256), 12.0)
        self.assertEqual(table.lookup(batch_size=16, steps=0, ctx_len=256), 12.0)
        self.assertIsNone(table.lookup(batch_size=4, steps=1, ctx_len=256))

    def test_lookup_uses_batch_ceiling_and_nearest_ctx_len(self):
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=1, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=8, steps=1, ctx_len=256, cost_ms=20.0)
        table.set(batch_size=8, steps=1, ctx_len=1024, cost_ms=30.0)

        cost_ms, matched_bs, matched_ctx_len = table.lookup_with_match(
            batch_size=5, steps=1, ctx_len=700
        )

        self.assertEqual(cost_ms, 30.0)
        self.assertEqual(matched_bs, 8)
        self.assertEqual(matched_ctx_len, 1024)

    def test_lookup_uses_ctx_boundaries_outside_profiled_range(self):
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=2, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=4, steps=2, ctx_len=1024, cost_ms=20.0)

        self.assertEqual(table.lookup(batch_size=4, steps=2, ctx_len=1), 10.0)
        self.assertEqual(table.lookup(batch_size=4, steps=2, ctx_len=4096), 20.0)


class TestProfileCtxLensParser(CustomTestCase):
    def test_parse_profile_ctx_lens(self):
        self.assertIsNone(parse_decoupled_verify_throughput_profile_ctx_lens(None))
        self.assertIsNone(parse_decoupled_verify_throughput_profile_ctx_lens(""))
        self.assertEqual(
            parse_decoupled_verify_throughput_profile_ctx_lens("1024,256,1024"),
            [256, 1024],
        )

    def test_parse_rejects_invalid_profile_ctx_lens(self):
        for raw in ["abc", "0", "-1", "256,,1024"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_decoupled_verify_throughput_profile_ctx_lens(raw)


class TestDraftPositionStatsTracker(CustomTestCase):
    def test_tracks_supply_and_conditional_accept_separately(self):
        tracker = DraftPositionStatsTracker(
            max_steps=3, ema_alpha=1.0, warmup_batches=1
        )

        tracker.update([0, 1, 2, 0], [0, 1, 3, 2], verified_steps=2)

        self.assertEqual(
            [tracker.supply_rate(position) for position in range(3)],
            [0.75, 0.5, 0.25],
        )
        self.assertEqual(tracker.accept_rate(0), 2 / 3)
        self.assertEqual(tracker.accept_rate(1), 0.5)
        self.assertIsNone(tracker.accept_rate(2))
        self.assertAlmostEqual(tracker.get_expected_tokens(2), 1.75)
        self.assertIsNone(tracker.get_expected_tokens(3))
        self.assertAlmostEqual(tracker.get_optimistic_expected_tokens(3), 2.0)

    def test_accept_warmup_counts_only_batches_with_supplied_samples(self):
        tracker = DraftPositionStatsTracker(
            max_steps=2, ema_alpha=1.0, warmup_batches=2
        )

        tracker.update([0, 0], [2, 2], verified_steps=2)
        tracker.update([0, 0], [2, 2], verified_steps=1)

        self.assertTrue(tracker.all_positions_warmed(1))
        self.assertFalse(tracker.all_positions_warmed(2))
        self.assertEqual(tracker.accept_update_count(1), 1)
        self.assertEqual(tracker.accept_sample_count(1), 2)

    def test_ema_updates_globally_across_batch_sizes(self):
        tracker = DraftPositionStatsTracker(
            max_steps=1, ema_alpha=0.5, warmup_batches=1
        )

        tracker.update([1, 1], [1, 1], verified_steps=1)
        tracker.update([0] * 8, [0] * 8, verified_steps=1)

        self.assertEqual(tracker.supply_rate(0), 0.5)
        self.assertEqual(tracker.accept_rate(0), 1.0)
        self.assertEqual(tracker.accept_update_count(0), 1)

    def test_rejects_mismatched_or_impossible_observations(self):
        tracker = DraftPositionStatsTracker(
            max_steps=2, ema_alpha=1.0, warmup_batches=1
        )

        with self.assertRaisesRegex(ValueError, "same number of requests"):
            tracker.update([0], [1, 1], verified_steps=1)
        with self.assertRaisesRegex(ValueError, "matching snapshot supply"):
            tracker.update([1], [0], verified_steps=1)
        with self.assertRaisesRegex(ValueError, "verified_steps"):
            tracker.update([2], [2], verified_steps=1)

    def test_rejects_invalid_ema_configuration(self):
        for alpha in [0.0, -0.1, 1.1]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                DraftPositionStatsTracker(max_steps=1, ema_alpha=alpha)


class TestDecoupledVerifyScoring(CustomTestCase):
    def test_expected_tokens_use_supply_times_conditional_accept(self):
        tracker = DraftPositionStatsTracker(
            max_steps=2, ema_alpha=1.0, warmup_batches=1
        )
        tracker.update([1, 2, 0, 0], [1, 2, 2, 0], verified_steps=2)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=4, steps=2, ctx_len=256, cost_ms=19.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0, 2], batch_size=4, ctx_len=256
        )

        self.assertEqual(rows[0]["expected"], 1.0)
        self.assertAlmostEqual(rows[1]["expected"], 1.75)
        self.assertEqual(rows[1]["position_supply_rates"], [0.75, 0.5])
        self.assertEqual(rows[1]["position_accept_rates"], [2 / 3, 0.5])
        self.assertEqual(pick_best_step(rows, fallback=0), 2)

    def test_hysteresis_blocks_marginal_switch(self):
        rows = [
            {"steps": 1, "score": 1.0},
            {"steps": 3, "score": 1.05},
        ]

        self.assertEqual(
            pick_best_step_with_hysteresis(
                rows, current_steps=1, hysteresis=0.1
            ),
            1,
        )
        self.assertEqual(
            DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS, 0.05
        )


class TestDecoupledVerifyThroughputAwareController(CustomTestCase):
    def _controller(
        self,
        *,
        candidate_steps: list[int],
        initial_steps: int,
        warmup_batches: int = 1,
        costs: dict[int, float],
    ) -> tuple[_Worker, DecoupledVerifyThroughputAwareController]:
        worker = _Worker(initial_steps=initial_steps)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=candidate_steps,
            initial_steps=initial_steps,
            ema_alpha=1.0,
            warmup_batches=warmup_batches,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in candidate_steps:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(
                batch_size=4,
                steps=steps,
                ctx_len=256,
                cost_ms=costs[steps],
            )
        controller.init_states(cuda_graph_bs=[4])
        return worker, controller

    def test_default_estimator_uses_upstream_ema_parameters(self):
        controller = DecoupledVerifyThroughputAwareController(
            _Worker(initial_steps=1),
            candidate_steps=[0, 1],
            initial_steps=1,
        )

        self.assertEqual(DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA, 0.2)
        self.assertEqual(DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES, 10)
        self.assertEqual(controller._tracker.ema_alpha, 0.2)
        self.assertEqual(controller._tracker.warmup_batches, 10)

    def test_sparse_positive_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            DecoupledVerifyThroughputAwareController(
                _Worker(initial_steps=1),
                candidate_steps=[0, 1, 3],
                initial_steps=1,
            )

    def test_cold_position_uses_adjacent_probe_after_warmup(self):
        worker, controller = self._controller(
            candidate_steps=[0, 1, 2],
            initial_steps=1,
            warmup_batches=2,
            costs={0: 100.0, 1: 10.0, 2: 11.0},
        )

        _observe(controller, [1, 1], [2, 2], verified_steps=1)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 1)

        _observe(controller, [1, 1], [2, 2], verified_steps=1)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 2)
        self.assertEqual(controller._probe_steps, 2)

        _observe(controller, [1, 1], [2, 2], verified_steps=2)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 2)

        _observe(controller, [1, 1], [2, 2], verified_steps=2)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 1)

    def test_cold_position_is_not_probed_when_optimistic_score_loses(self):
        worker, controller = self._controller(
            candidate_steps=[1, 2],
            initial_steps=1,
            costs={1: 1.0, 2: 100.0},
        )

        _observe(controller, [1, 1], [2, 2], verified_steps=1)
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertIsNone(controller._probe_steps)

    def test_zero_step_forces_first_positive_probe_without_supply_history(self):
        worker, controller = self._controller(
            candidate_steps=[0, 1],
            initial_steps=1,
            costs={0: 1.0, 1: 100.0},
        )
        controller._current_steps = 0
        controller._probe_steps = None
        worker.speculative_num_steps = 0

        _observe(controller, [0, 0], [0, 0], verified_steps=0)
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertEqual(controller._probe_steps, 1)

    def test_observation_uses_result_verified_steps_not_current_state(self):
        _, controller = self._controller(
            candidate_steps=[1, 2, 3],
            initial_steps=3,
            costs={1: 10.0, 2: 10.0, 3: 10.0},
        )

        _observe(controller, [1, 1], [3, 3], verified_steps=1)

        self.assertEqual(controller._tracker.accept_update_count(0), 1)
        self.assertEqual(controller._tracker.accept_update_count(1), 0)
        self.assertEqual(controller._tracker.supply_rate(2), 1.0)

    def test_modeled_throughput_uses_supply_accept_ema(self):
        _, controller = self._controller(
            candidate_steps=[1],
            initial_steps=1,
            costs={1: 10.0},
        )
        _observe(controller, [1, 0, 0, 0], [1, 1, 0, 0], verified_steps=1)

        modeled = controller.get_modeled_throughput(batch_size=4, ctx_len=256)

        self.assertAlmostEqual(modeled["modeled_throughput"], 4 * 1.25 * 1000 / 13)

    def test_modeled_throughput_waits_for_ema(self):
        _, controller = self._controller(
            candidate_steps=[1],
            initial_steps=1,
            costs={1: 10.0},
        )

        self.assertIsNone(
            controller.get_modeled_throughput(batch_size=4, ctx_len=256)
        )


if __name__ == "__main__":
    unittest.main()

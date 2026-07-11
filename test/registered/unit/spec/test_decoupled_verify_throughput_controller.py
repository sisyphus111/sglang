import unittest

from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    BatchSizeCostTable,
    DecoupledVerifyThroughputAwareController,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES,
    PositionAcceptanceTracker,
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
        self.assertIsNone(parse_decoupled_verify_throughput_profile_ctx_lens("  "))
        self.assertEqual(
            parse_decoupled_verify_throughput_profile_ctx_lens("256,1024,4096"),
            [256, 1024, 4096],
        )
        self.assertEqual(
            parse_decoupled_verify_throughput_profile_ctx_lens("1024,256,1024"),
            [256, 1024],
        )

    def test_parse_rejects_invalid_profile_ctx_lens(self):
        for raw in ["abc", "0", "-1", "256,,1024"]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_decoupled_verify_throughput_profile_ctx_lens(raw)


class TestDecoupledVerifyScoring(CustomTestCase):
    def test_zero_step_expected_tokens_is_one(self):
        tracker = PositionAcceptanceTracker(max_steps=0)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, ctx_len=256, cost_ms=10.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0], batch_size=4, ctx_len=256
        )

        self.assertEqual(rows[0]["expected"], 1.0)
        self.assertAlmostEqual(rows[0]["score"], 1.0 / 13.0)
        self.assertEqual(rows[0]["matched_ctx_len"], 256)

    def test_scoring_passes_runtime_ctx_len_to_lookup(self):
        tracker = PositionAcceptanceTracker(max_steps=0)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=4, steps=0, ctx_len=1024, cost_ms=20.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0], batch_size=4, ctx_len=800
        )

        self.assertEqual(rows[0]["ctx_len"], 800)
        self.assertEqual(rows[0]["matched_ctx_len"], 1024)
        self.assertEqual(rows[0]["cost_ms"], 23.0)

    def test_high_acceptance_can_prefer_larger_step(self):
        tracker = PositionAcceptanceTracker(
            max_steps=3, ema_alpha=1.0, warmup_batches=1
        )
        tracker.update([3, 3], current_steps=3)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=4, steps=3, ctx_len=256, cost_ms=20.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0, 3], batch_size=4, ctx_len=256
        )

        self.assertEqual(pick_best_step(rows, fallback=0), 3)

    def test_high_verifier_cost_can_prefer_smaller_step(self):
        tracker = PositionAcceptanceTracker(
            max_steps=3, ema_alpha=1.0, warmup_batches=1
        )
        tracker.update([3, 3], current_steps=3)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, ctx_len=256, cost_ms=10.0)
        table.set(batch_size=4, steps=3, ctx_len=256, cost_ms=100.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0, 3], batch_size=4, ctx_len=256
        )

        self.assertEqual(pick_best_step(rows, fallback=3), 0)

    def test_hysteresis_blocks_marginal_switch(self):
        rows = [
            {"steps": 1, "expected": 2.0, "cost_ms": 2.0, "score": 1.0},
            {"steps": 3, "expected": 2.1, "cost_ms": 2.0, "score": 1.05},
        ]

        selected = pick_best_step_with_hysteresis(
            rows, current_steps=1, hysteresis=0.1
        )

        self.assertEqual(selected, 1)

    def test_default_hysteresis_requires_more_than_five_percent(self):
        for best_score, expected_step in [(1.05, 1), (1.051, 3)]:
            with self.subTest(best_score=best_score):
                rows = [
                    {"steps": 1, "score": 1.0},
                    {"steps": 3, "score": best_score},
                ]
                self.assertEqual(
                    pick_best_step_with_hysteresis(
                        rows,
                        current_steps=1,
                        hysteresis=DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS,
                    ),
                    expected_step,
                )


class TestPositionAcceptanceTracker(CustomTestCase):
    def test_updates_per_position_ema(self):
        tracker = PositionAcceptanceTracker(
            max_steps=3, ema_alpha=0.25, warmup_batches=2
        )

        tracker.update([3, 1, 0, 0], current_steps=3)
        self.assertEqual(tracker.snapshot_position_rates(3), [0.5, 0.25, 0.25])
        self.assertFalse(tracker.all_positions_warmed(3))

        tracker.update([3, 3, 3, 3], current_steps=3)
        rates = tracker.snapshot_position_rates(3)
        self.assertEqual(rates, [0.625, 0.4375, 0.4375])
        self.assertTrue(tracker.all_positions_warmed(3))
        self.assertGreaterEqual(rates[0], rates[1])
        self.assertGreaterEqual(rates[1], rates[2])

    def test_only_first_unseen_position_is_extrapolated(self):
        tracker = PositionAcceptanceTracker(
            max_steps=3, ema_alpha=1.0, warmup_batches=1
        )

        tracker.update([1, 1, 1, 0], current_steps=1)

        self.assertEqual(tracker.snapshot_position_rates(3), [0.75, 0.5625, None])
        self.assertAlmostEqual(tracker.get_expected_tokens(2), 2.3125)
        self.assertIsNone(tracker.get_expected_tokens(3))
        self.assertEqual(tracker.position_update_count(0), 1)
        self.assertEqual(tracker.position_update_count(1), 0)

    def test_rejects_invalid_ema_configuration(self):
        for alpha in [0.0, -0.1, 1.1]:
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    PositionAcceptanceTracker(max_steps=1, ema_alpha=alpha)

    def test_empty_and_zero_step_updates_are_noops(self):
        tracker = PositionAcceptanceTracker(
            max_steps=2, ema_alpha=1.0, warmup_batches=1
        )

        tracker.update([], current_steps=2)
        tracker.update([1, 1], current_steps=0)

        self.assertEqual(tracker.snapshot_position_rates(2), [None, None])
        self.assertEqual(tracker.position_update_count(0), 0)
        self.assertEqual(tracker.position_update_count(1), 0)


class TestDecoupledVerifyThroughputAwareController(CustomTestCase):
    def test_default_estimator_uses_upstream_ema_parameters(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=1,
        )

        self.assertEqual(DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA, 0.2)
        self.assertEqual(DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES, 10)
        self.assertEqual(DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS, 0.05)
        self.assertEqual(controller._tracker.ema_alpha, 0.2)
        self.assertEqual(controller._tracker.warmup_batches, 10)

    def test_initial_step_uses_smallest_positive_candidate(self):
        worker = _Worker(initial_steps=0)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=0,
            warmup_batches=1,
            update_interval=1,
        )
        for steps in [0, 1]:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(
                batch_size=4,
                steps=steps,
                ctx_len=256,
                cost_ms=10.0,
            )
        controller.init_states(cuda_graph_bs=[4])

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertEqual(worker.applied_steps[-1], 1)

    def test_sparse_positive_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            DecoupledVerifyThroughputAwareController(
                _Worker(initial_steps=1),
                candidate_steps=[0, 1, 3],
                initial_steps=1,
            )

    def test_zero_step_probes_positive_candidate_to_collect_acceptance_data(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=1,
            warmup_batches=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [0, 1]:
            controller.register(_state(steps), steps=steps)
        controller.set_profile_cost(batch_size=4, steps=0, ctx_len=256, cost_ms=1.0)
        controller.set_profile_cost(batch_size=4, steps=1, ctx_len=256, cost_ms=100.0)
        controller.init_states(cuda_graph_bs=[4])

        controller._current_steps = 0
        worker.speculative_num_steps = 0
        controller.on_verify_complete([0, 0], batch_size=4)
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertEqual(worker.applied_steps[-1], 1)

    def test_cold_start_probes_one_position_before_larger_step(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1, 2, 3],
            initial_steps=1,
            ema_alpha=1.0,
            warmup_batches=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [0, 1, 2, 3]:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(
                batch_size=4,
                steps=steps,
                ctx_len=256,
                cost_ms=100.0 if steps == 0 else 10.0,
            )
        controller.init_states(cuda_graph_bs=[4])

        controller.on_verify_complete([1, 1, 1, 0], batch_size=4)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 2)

        controller.on_verify_complete([2, 2, 2, 2], batch_size=4)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 3)

    def test_shrink_preserves_higher_position_ema(self):
        worker = _Worker(initial_steps=3)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1, 2, 3],
            initial_steps=3,
            ema_alpha=1.0,
            warmup_batches=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [1, 2, 3]:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(
                batch_size=4,
                steps=steps,
                ctx_len=256,
                cost_ms=1.0 if steps == 1 else 100.0,
            )
        controller.init_states(cuda_graph_bs=[4])

        controller.on_verify_complete([3, 3], batch_size=2)
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertFalse(controller._tracker.is_position_extrapolated(1))
        self.assertFalse(controller._tracker.is_position_extrapolated(2))

        controller.on_verify_complete([0, 0], batch_size=2)
        self.assertEqual(controller._tracker.snapshot_position_rates(3), [0.0] * 3)
        self.assertEqual(controller._tracker.position_update_count(1), 0)
        self.assertEqual(controller._tracker.position_update_count(2), 0)
        self.assertEqual(controller._tracker.position_source(1), "projected")
        self.assertEqual(controller._tracker.position_source(2), "projected")
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)

    def test_unseen_position_is_not_forced_when_projected_score_is_lower(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1, 2],
            initial_steps=1,
            ema_alpha=1.0,
            warmup_batches=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [0, 1, 2]:
            controller.register(_state(steps), steps=steps)
        controller.set_profile_cost(batch_size=4, steps=0, ctx_len=256, cost_ms=1.0)
        controller.set_profile_cost(batch_size=4, steps=1, ctx_len=256, cost_ms=1.0)
        controller.set_profile_cost(
            batch_size=4, steps=2, ctx_len=256, cost_ms=100.0
        )
        controller.init_states(cuda_graph_bs=[4])

        controller.on_verify_complete([1, 1], batch_size=2)
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)

    def test_tier_ema_preserves_width_specific_partial_output(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1, 2, 3],
            initial_steps=1,
            ema_alpha=0.2,
            warmup_batches=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps, cost_ms in [(1, 15.0), (2, 100.0), (3, 10.0)]:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(
                batch_size=4, steps=steps, ctx_len=256, cost_ms=cost_ms
            )
        controller.init_states(cuda_graph_bs=[4])

        controller._tracker.update([3, 3], current_steps=3)
        controller.on_verify_complete([1, 1], batch_size=4)
        controller.activate_step_by_batch(4, 256)
        self.assertEqual(worker.speculative_num_steps, 3)

        # A deep verify can drain partial draft tails and produce no accepted
        # drafts.  Its p1 observation must not overwrite DL1's measured output.
        controller.on_verify_complete([0, 0], batch_size=4)
        controller.activate_step_by_batch(4, 256)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertEqual(controller._tier_expected_tokens_ema[(4, 1)], 2.0)
        self.assertEqual(controller._tier_expected_tokens_ema[(4, 3)], 1.0)

    def test_tier_ema_is_isolated_by_cuda_graph_batch_slot(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1, 2],
            initial_steps=1,
            ema_alpha=1.0,
            warmup_batches=1,
            update_interval=1,
        )
        for batch_size in [4, 8]:
            for steps in [1, 2]:
                controller.set_profile_cost(
                    batch_size=batch_size,
                    steps=steps,
                    ctx_len=256,
                    cost_ms=10.0,
                )
        for steps in [1, 2]:
            controller.register(_state(steps), steps=steps)
        controller.init_states(cuda_graph_bs=[4, 8])

        controller.on_verify_complete([1, 1], batch_size=4)
        controller.on_verify_complete([0, 0], batch_size=5)

        self.assertEqual(controller._batch_slot(5), 8)
        self.assertEqual(controller._tier_expected_tokens_ema[(4, 1)], 2.0)
        self.assertEqual(controller._tier_expected_tokens_ema[(8, 1)], 1.0)
        self.assertEqual(controller._tier_update_counts[(4, 1)], 1)
        self.assertEqual(controller._tier_update_counts[(8, 1)], 1)

    def test_new_batch_slot_does_not_force_cold_tier_probe(self):
        worker = _Worker(initial_steps=2)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1, 2],
            initial_steps=2,
            ema_alpha=1.0,
            warmup_batches=2,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [1, 2]:
            controller.register(_state(steps), steps=steps)
            for batch_size in [4, 8]:
                controller.set_profile_cost(
                    batch_size=batch_size,
                    steps=steps,
                    ctx_len=256,
                    cost_ms=1.0 if steps == 1 else 100.0,
                )
        controller.init_states(cuda_graph_bs=[4, 8])

        controller.on_verify_complete([2, 2], batch_size=4)
        controller.activate_step_by_batch(8, 256)
        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertNotIn((8, 2), controller._tier_update_counts)

    def test_old_tier_ema_decays_toward_current_position_estimate(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1, 2],
            initial_steps=1,
            ema_alpha=0.2,
            warmup_batches=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps, cost_ms in [(1, 10.0), (2, 15.0)]:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(
                batch_size=4, steps=steps, ctx_len=256, cost_ms=cost_ms
            )
        controller.init_states(cuda_graph_bs=[4])

        controller._tracker.update([1, 0], current_steps=2)
        controller._batch_count = 11
        controller._tier_expected_tokens_ema.update({(4, 1): 1.5, (4, 2): 3.0})
        controller._tier_update_counts.update({(4, 1): 1, (4, 2): 1})
        controller._tier_last_update_batch.update({(4, 1): 11, (4, 2): 1})

        # The raw old DL2 sample would win (3/18 > 1.5/13, including runtime
        # overhead).  Its decayed blend with the current 1.5-token position
        # estimate must not pull the controller into the stale tier.
        controller._reevaluate_and_switch(batch_size=4, ctx_len=256)

        self.assertEqual(controller._current_steps, 1)

    def test_tier_observation_records_current_batch_age(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1],
            initial_steps=1,
            warmup_batches=1,
        )
        controller.on_verify_complete([1, 0], batch_size=4)

        self.assertEqual(controller._batch_count, 1)
        self.assertEqual(controller._tier_last_update_batch[(4, 1)], 1)


if __name__ == "__main__":
    unittest.main()

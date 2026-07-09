import unittest

from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    BatchSizeCostTable,
    DecoupledVerifyThroughputAwareController,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WINDOW_SIZE,
    PositionAcceptanceTracker,
    pick_best_step,
    pick_best_step_with_hysteresis,
    score_decoupled_verify_candidates,
)
from sglang.test.ci.ci_register import register_cpu_ci

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


class TestBatchSizeCostTable(unittest.TestCase):
    def test_lookup_uses_exact_ceiling_and_largest_profiled_bs(self):
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, cost_ms=10.0)
        table.set(batch_size=8, steps=0, cost_ms=12.0)

        self.assertEqual(table.lookup(batch_size=4, steps=0), 10.0)
        self.assertEqual(table.lookup(batch_size=5, steps=0), 12.0)
        self.assertEqual(table.lookup(batch_size=16, steps=0), 12.0)
        self.assertIsNone(table.lookup(batch_size=4, steps=1))


class TestDecoupledVerifyScoring(unittest.TestCase):
    def test_zero_step_expected_tokens_is_one(self):
        tracker = PositionAcceptanceTracker(max_steps=0, window_size=1)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, cost_ms=10.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0], batch_size=4
        )

        self.assertEqual(rows[0]["expected"], 1.0)
        self.assertEqual(rows[0]["score"], 0.1)

    def test_high_acceptance_can_prefer_larger_step(self):
        tracker = PositionAcceptanceTracker(max_steps=3, window_size=1)
        tracker.update([3, 3], current_steps=3)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, cost_ms=10.0)
        table.set(batch_size=4, steps=3, cost_ms=20.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0, 3], batch_size=4
        )

        self.assertEqual(pick_best_step(rows, fallback=0), 3)

    def test_high_verifier_cost_can_prefer_smaller_step(self):
        tracker = PositionAcceptanceTracker(max_steps=3, window_size=1)
        tracker.update([3, 3], current_steps=3)
        table = BatchSizeCostTable()
        table.set(batch_size=4, steps=0, cost_ms=10.0)
        table.set(batch_size=4, steps=3, cost_ms=100.0)

        rows = score_decoupled_verify_candidates(
            tracker, table, candidate_steps=[0, 3], batch_size=4
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


class TestDecoupledVerifyThroughputAwareController(unittest.TestCase):
    def test_default_sliding_window_size_is_fifty_batches(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=1,
        )

        self.assertEqual(DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WINDOW_SIZE, 50)
        self.assertEqual(controller._tracker.window_size, 50)

    def test_initial_step_uses_smallest_positive_candidate(self):
        worker = _Worker(initial_steps=0)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=0,
            window_size=1,
            update_interval=1,
        )
        for steps in [0, 1]:
            controller.register(_state(steps), steps=steps)
            controller.set_profile_cost(batch_size=4, steps=steps, cost_ms=10.0)
        controller.init_states(cuda_graph_bs=[4])

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertEqual(worker.applied_steps[-1], 1)

    def test_zero_step_probes_positive_candidate_to_collect_acceptance_data(self):
        worker = _Worker(initial_steps=1)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=1,
            window_size=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [0, 1]:
            controller.register(_state(steps), steps=steps)
        controller.set_profile_cost(batch_size=4, steps=0, cost_ms=1.0)
        controller.set_profile_cost(batch_size=4, steps=1, cost_ms=100.0)
        controller.init_states(cuda_graph_bs=[4])

        controller.on_verify_complete([0, 0], batch_size=4)
        controller.activate_step_by_batch(4)

        self.assertEqual(worker.speculative_num_steps, 0)

        controller.on_verify_complete([0, 0], batch_size=4)
        controller.activate_step_by_batch(4)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertEqual(worker.applied_steps[-1], 1)

    def test_shrink_preserves_higher_position_windows(self):
        worker = _Worker(initial_steps=3)
        controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[1, 3],
            initial_steps=3,
            window_size=1,
            update_interval=1,
            switch_hysteresis=0.0,
        )
        for steps in [1, 3]:
            controller.register(_state(steps), steps=steps)
        controller.set_profile_cost(batch_size=4, steps=1, cost_ms=1.0)
        controller.set_profile_cost(batch_size=4, steps=3, cost_ms=100.0)
        controller.init_states(cuda_graph_bs=[4])

        controller.on_verify_complete([3, 3], batch_size=2)
        controller.activate_step_by_batch(4)

        self.assertEqual(worker.speculative_num_steps, 1)
        self.assertFalse(controller._tracker.is_position_extrapolated(1))
        self.assertFalse(controller._tracker.is_position_extrapolated(2))


if __name__ == "__main__":
    unittest.main()

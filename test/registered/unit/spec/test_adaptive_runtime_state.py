import json
import tempfile
import unittest

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
    _SpecAdaptiveBase,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSpecRuntimeState(unittest.TestCase):
    def test_for_eagle_carries_flat_phase_resources(self):
        draft_attn_backend = object()
        cuda_graph_runner = object()
        target_attn_backend = object()
        target_graph_runner = object()
        draft_extend_attn_backend = object()
        cuda_graph_runner_for_draft_extend = object()

        state = SpecRuntimeState.for_eagle(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_attn_backend=draft_attn_backend,
            cuda_graph_runner=cuda_graph_runner,
            target_attn_backend=target_attn_backend,
            target_graph_runner=target_graph_runner,
            draft_extend_attn_backend=draft_extend_attn_backend,
            cuda_graph_runner_for_draft_extend=cuda_graph_runner_for_draft_extend,
        )

        self.assertIs(state.draft_attn_backend, draft_attn_backend)
        self.assertIs(state.cuda_graph_runner, cuda_graph_runner)
        self.assertIs(state.target_attn_backend, target_attn_backend)
        self.assertIs(state.target_graph_runner, target_graph_runner)
        self.assertIs(state.draft_extend_attn_backend, draft_extend_attn_backend)
        self.assertIs(
            state.cuda_graph_runner_for_draft_extend,
            cuda_graph_runner_for_draft_extend,
        )

    def test_eagle_allows_optional_draft_resources(self):
        state = SpecRuntimeState(
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
        )

        self.assertIsNone(state.draft_attn_backend)
        self.assertIsNone(state.cuda_graph_runner)
        self.assertIsNone(state.draft_extend_attn_backend)
        self.assertIsNone(state.cuda_graph_runner_for_draft_extend)

    def test_for_decoupled_verify_carries_target_resources_only(self):
        attn_backend = object()
        graph_runner = object()

        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=8,
            speculative_num_draft_tokens=9,
            target_attn_backend=attn_backend,
            target_graph_runner=graph_runner,
        )

        self.assertEqual(state.speculative_num_steps, 8)
        self.assertEqual(state.speculative_num_draft_tokens, 9)
        self.assertIsNone(state.draft_attn_backend)
        self.assertIsNone(state.cuda_graph_runner)
        self.assertIsNone(state.draft_extend_attn_backend)
        self.assertIsNone(state.cuda_graph_runner_for_draft_extend)
        self.assertIs(state.target_attn_backend, attn_backend)
        self.assertIs(state.target_graph_runner, graph_runner)


class _Worker:
    def __init__(self):
        self.speculative_num_steps = 1
        self.applied = []
        self.build_calls = []

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        self.build_calls.append(
            (speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs)
        )
        return SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            target_attn_backend=object(),
            target_graph_runner=object(),
        )

    def apply_runtime_state(self, state):
        self.speculative_num_steps = state.speculative_num_steps
        self.applied.append(state)


class TestAdaptiveController(unittest.TestCase):
    def test_base_register_activate_and_default_profiling_hook(self):
        worker = _Worker()
        base = _SpecAdaptiveBase(worker)
        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            target_attn_backend=object(),
            target_graph_runner=object(),
        )

        base.register(state)
        result = base.run_profiling(tree_cache=object())
        base._activate(2)

        self.assertIsNone(result)
        self.assertEqual(worker.speculative_num_steps, 2)
        self.assertIs(worker.applied[-1], state)

    def test_activate_step_by_batch_applies_selected_state(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1]},
                    "8": {"candidate_steps": [3]},
                },
                f,
            )
            f.flush()
            worker = _Worker()
            controller = AdaptiveController(worker, config_path=f.name)
            controller.init_states(cuda_graph_bs=None)

        result = controller.activate_step_by_batch(8)

        self.assertIsNone(result)
        self.assertEqual(worker.speculative_num_steps, 3)
        self.assertEqual(worker.applied[-1].speculative_num_steps, 3)

    def test_init_states_reuses_registered_zero_step_state(self):
        worker = _Worker()
        worker.speculative_num_steps = 0
        controller = AdaptiveController(
            worker,
            config={
                "1": {"candidate_steps": [0]},
                "8": {"candidate_steps": [3]},
            },
        )
        zero_state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=0,
            speculative_num_draft_tokens=1,
            target_attn_backend=object(),
            target_graph_runner=object(),
        )
        controller.register(zero_state, steps=0)

        controller.init_states(cuda_graph_bs=[1, 8])

        self.assertEqual(worker.build_calls, [(3, 4, [8])])
        self.assertIs(worker.applied[-1], zero_state)

    def test_init_states_reuses_registered_profile_states(self):
        worker = _Worker()
        worker.speculative_num_steps = 0
        controller = AdaptiveController(
            worker,
            config={
                "1": {"candidate_steps": [0, 1]},
                "8": {"candidate_steps": [0, 1, 3]},
            },
        )
        states = {
            steps: SpecRuntimeState.for_decoupled_verify(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                target_attn_backend=object(),
                target_graph_runner=object(),
            )
            for steps in (0, 1, 3)
        }
        for steps, state in states.items():
            controller.register(state, steps=steps)

        controller.init_states(cuda_graph_bs=[1, 8])

        self.assertEqual(worker.build_calls, [])
        self.assertIs(worker.applied[-1], states[0])


if __name__ == "__main__":
    unittest.main()

import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
)
from sglang.srt.speculative.spec_info import (
    compute_decoupled_verify_runtime_state,
    compute_decoupled_verify_shape,
    compute_runtime_state_from_capture_bs,
    compute_shape_from_capture_bs,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid: str):
        self.rid = rid


class TestDecoupledVerifyDynamic(unittest.TestCase):
    def test_budget_formula_uses_captured_batch_size(self):
        shape = compute_decoupled_verify_shape(
            raw_batch_size=7,
            captured_batch_size=8,
            budget=33,
            max_speculative_num_steps=8,
        )

        self.assertEqual(shape.num_speculative_steps, 3)
        self.assertEqual(shape.verify_tokens_per_req, 4)
        self.assertEqual(shape.raw_verify_tokens, 28)
        self.assertEqual(shape.padded_verify_tokens, 32)
        self.assertLess(shape.padded_verify_tokens, shape.budget)

    def test_budget_clamps_to_zero_step_decode_graph(self):
        shape = compute_decoupled_verify_shape(
            raw_batch_size=31,
            captured_batch_size=32,
            budget=64,
            max_speculative_num_steps=8,
        )

        self.assertEqual(shape.num_speculative_steps, 0)
        self.assertEqual(shape.verify_tokens_per_req, 1)
        self.assertTrue(shape.uses_decode_graph)
        self.assertLess(shape.padded_verify_tokens, shape.budget)

    def test_budget_rejects_padded_batch_that_cannot_verify_one_token(self):
        with self.assertRaises(ValueError):
            compute_decoupled_verify_shape(
                raw_batch_size=63,
                captured_batch_size=64,
                budget=64,
                max_speculative_num_steps=8,
            )

    def test_capture_bs_selection_uses_first_covering_shape(self):
        shape = compute_shape_from_capture_bs(
            raw_batch_size=9,
            capture_bs=[1, 4, 8, 16],
            budget=65,
            max_speculative_num_steps=8,
        )

        self.assertEqual(shape.captured_batch_size, 16)
        self.assertEqual(shape.verify_tokens_per_req, 4)
        self.assertLess(shape.padded_verify_tokens, shape.budget)

    def test_runtime_state_wraps_shape_and_target_resources(self):
        attn_backend = object()
        graph_runner = object()
        state = compute_decoupled_verify_runtime_state(
            raw_batch_size=7,
            captured_batch_size=8,
            budget=33,
            max_speculative_num_steps=8,
            target_attn_backend=attn_backend,
            target_graph_runner=graph_runner,
        )

        self.assertEqual(state.num_speculative_steps, 3)
        self.assertEqual(state.verify_tokens_per_req, 4)
        self.assertEqual(state.raw_verify_tokens, 28)
        self.assertEqual(state.padded_verify_tokens, 32)
        self.assertIs(state.target_attn_backend, attn_backend)
        self.assertIs(state.target_graph_runner, graph_runner)
        self.assertEqual(state.shape.verify_tokens_per_req, state.verify_tokens_per_req)

    def test_runtime_state_capture_bs_selection(self):
        state = compute_runtime_state_from_capture_bs(
            raw_batch_size=9,
            capture_bs=[1, 4, 8, 16],
            budget=65,
            max_speculative_num_steps=8,
        )

        self.assertEqual(state.captured_batch_size, 16)
        self.assertEqual(state.verify_tokens_per_req, 4)
        self.assertLess(state.padded_verify_tokens, state.budget)

    def test_draft_tail_buffer_caps_snapshot_tail_without_hiding_raw_tail(self):
        rid = "req-a"
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=4)
        try:
            buffer.apply_control_batch(
                DraftControlBatch(
                    0,
                    sync_messages=[
                        DraftSync(
                            request_id=rid,
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            prompt_token_ids=[1, 2],
                            committed_output_ids=[3],
                        )
                    ],
                )
            )
            buffer.append_draft_stream_batch(
                DraftTailStreamOutputBatch(
                    [
                        DraftTailStreamOutput(0, 0, rid, 1, 1, 10),
                        DraftTailStreamOutput(0, 0, rid, 1, 2, 11),
                        DraftTailStreamOutput(0, 0, rid, 1, 3, 12),
                    ]
                )
            )

            snapshot = buffer.get_draft_snapshots(
                [_Req(rid)],
                allow_partial=True,
                include_raw_tail_tokens=True,
                max_tail_len=1,
            )[0]

            self.assertEqual(snapshot.tail_tokens, [10])
            self.assertEqual(snapshot.raw_tail_tokens, [10, 11, 12])
            self.assertEqual(snapshot.raw_tail_len, 3)
        finally:
            buffer.close()


if __name__ == "__main__":
    unittest.main()

"""CPU contract tests for verifier-side GPU snapshot selection and TP relay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.decoupled_verify_worker import (
    DecoupledVerifyWorker,
    select_decoupled_gpu_tail_snapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _GpuTail:
    def __init__(
        self,
        compact: torch.Tensor,
        cursor: torch.Tensor,
        debug: torch.Tensor | None = None,
    ) -> None:
        self.compact = compact
        self.cursor = cursor
        self.debug = debug
        self.calls = []

    def select_snapshot(
        self,
        gpu_seats,
        seq_lens,
        bonus_tokens,
        *,
        request_epochs=None,
        out=None,
        out_cursor=None,
        debug_out=None,
    ):
        self.calls.append((gpu_seats, request_epochs, seq_lens, bonus_tokens))
        if out is not None:
            out.copy_(self.compact)
            self.compact = out
        if out_cursor is not None:
            out_cursor.copy_(self.cursor)
            self.cursor = out_cursor
        if debug_out is not None:
            debug_out.copy_(self.debug)
        return self.compact, self.cursor


class _BroadcastGroup:
    def __init__(self, source: torch.Tensor | None = None) -> None:
        self.world_size = 4
        self.source = source
        self.calls = []

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        self.calls.append((tensor, src))
        if self.source is not None:
            tensor.copy_(self.source)


class TestDecoupledVerifyGpuSnapshot(CustomTestCase):
    def test_forward_applies_critical_commit_before_future_publish(self):
        events = []
        commit_call = {}

        class _CriticalTail:
            mock_profile = False

            def apply_verify_commit_from_device(
                self,
                gpu_seats,
                expected_request_epochs,
                pre_verify_seq_lens,
                accept_tokens,
                num_accept_tokens,
                *,
                accept_token_stride,
                commit_mask=None,
            ):
                events.append("commit")
                commit_call.update(
                    gpu_seats=gpu_seats,
                    expected_request_epochs=expected_request_epochs,
                    pre_verify_seq_lens=pre_verify_seq_lens,
                    accept_tokens=accept_tokens,
                    num_accept_tokens=num_accept_tokens,
                    accept_token_stride=accept_token_stride,
                    commit_mask=commit_mask,
                )

        worker = object.__new__(DecoupledVerifyWorker)
        worker.ps = SimpleNamespace(tp_rank=0)
        worker.gpu_tail_buffer = _CriticalTail()
        worker._target_worker = SimpleNamespace()
        worker.req_to_token_pool = SimpleNamespace()
        worker.token_to_kv_pool_allocator = SimpleNamespace()
        worker.plan_stream = None
        worker.plan_stream_ctx = None
        worker.topk = 1
        worker.device = torch.device("cpu")

        verify_input = SimpleNamespace()

        def build_verify_input(batch):
            events.append("select")
            return verify_input

        worker._build_verify_input = build_verify_input
        gpu_seats = torch.tensor([1, 2], dtype=torch.int64)
        expected_request_epochs = torch.tensor([5, 6], dtype=torch.int64)
        seq_lens = torch.tensor([10, 12], dtype=torch.int64)
        pre_verify_output_lens = torch.tensor([7, 9], dtype=torch.int64)
        accept_tokens = torch.tensor([11, 12, 0, 0, 21, 22, 23, 0], dtype=torch.int32)
        accept_lens = torch.tensor([2, 3], dtype=torch.int32)
        batch_output = SimpleNamespace(
            next_token_ids=accept_tokens,
            accept_lens=accept_lens,
            next_draft_input=SimpleNamespace(
                bonus_tokens=torch.tensor([12, 23], dtype=torch.int32)
            ),
            new_seq_lens=torch.tensor([9, 12], dtype=torch.int64),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
            decoupled_verify_steps=3,
            req_pool_indices=gpu_seats,
            decoupled_expected_request_epochs=expected_request_epochs,
            decoupled_has_new_lifecycle=False,
            decoupled_landing_event=None,
            seq_lens=seq_lens,
            decoupled_pre_verify_output_lens=pre_verify_output_lens,
            decoupled_rebase_valid=torch.ones(2, dtype=torch.int64),
            decoupled_selected_draft_lens=torch.tensor([1, 2]),
            decoupled_tail_select_debug=torch.zeros((2, 10), dtype=torch.int64),
            spec_info=None,
        )

        def publish(new_seq_lens):
            events.append("publish")
            self.assertIs(new_seq_lens, batch_output.new_seq_lens)

        with (
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.run_eagle_verify",
                return_value=batch_output,
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.build_decoupled_next_input",
                return_value=SimpleNamespace(),
            ),
        ):
            actual = worker.forward_batch_generation(batch, on_publish=publish)

        self.assertIs(actual, batch_output)
        self.assertEqual(events, ["select", "commit", "publish"])
        self.assertIs(commit_call["gpu_seats"], gpu_seats)
        self.assertIs(commit_call["expected_request_epochs"], expected_request_epochs)
        self.assertIs(commit_call["pre_verify_seq_lens"], seq_lens)
        self.assertIs(commit_call["accept_tokens"], accept_tokens)
        self.assertIs(commit_call["num_accept_tokens"], accept_lens)
        self.assertEqual(commit_call["accept_token_stride"], 4)

    def test_extend_and_mixed_commit_live_final_rows_before_publish(self):
        def make_req(*, finished=False, retracted=False, middle_chunks=0):
            return SimpleNamespace(
                finished=lambda: finished,
                is_retracted=retracted,
                inflight_middle_chunks=middle_chunks,
            )

        cases = (
            (
                "extend_all_live",
                True,
                False,
                [make_req(), make_req()],
                None,
                True,
                None,
            ),
            (
                "final_chunk_with_prior_result_in_flight",
                True,
                False,
                [make_req(middle_chunks=1), make_req()],
                None,
                False,
                None,
            ),
            (
                "mixed_filters_non_committing_rows",
                False,
                True,
                [
                    make_req(),
                    make_req(middle_chunks=1),
                    make_req(finished=True),
                    make_req(retracted=True),
                ],
                [True, False, False, False],
                False,
                1,
            ),
        )

        for (
            name,
            is_extend,
            is_extend_in_batch,
            reqs,
            expected_mask,
            has_new_lifecycle,
            chunked_row,
        ) in cases:
            with self.subTest(name=name):
                events = []
                commit_call = {}
                batch_size = len(reqs)
                gpu_seats = torch.arange(1, batch_size + 1, dtype=torch.int64)
                expected_request_epochs = torch.arange(
                    11, 11 + batch_size, dtype=torch.int64
                )
                sampled_tokens = torch.arange(101, 101 + batch_size, dtype=torch.int64)
                seq_lens = torch.arange(20, 20 + batch_size, dtype=torch.int64)
                batch_output = SimpleNamespace(next_token_ids=sampled_tokens)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_extend=lambda: is_extend),
                    is_extend_in_batch=is_extend_in_batch,
                    reqs=reqs,
                    chunked_req=None if chunked_row is None else reqs[chunked_row],
                    req_pool_indices=gpu_seats,
                    decoupled_expected_request_epochs=expected_request_epochs,
                    decoupled_has_new_lifecycle=has_new_lifecycle,
                    decoupled_landing_event=(
                        object() if has_new_lifecycle else None
                    ),
                    seq_lens=seq_lens,
                )

                class _CriticalTail:
                    mock_profile = False

                    def wait_for_landing_event(self, event):
                        self.event = event
                        events.append("open_wait")

                    def apply_verify_commit_from_device(
                        self,
                        actual_gpu_seats,
                        actual_expected_request_epochs,
                        pre_verify_seq_lens,
                        accept_tokens,
                        num_accept_tokens,
                        *,
                        accept_token_stride,
                        commit_mask=None,
                    ):
                        events.append("commit")
                        commit_call.update(
                            gpu_seats=actual_gpu_seats,
                            expected_request_epochs=actual_expected_request_epochs,
                            pre_verify_seq_lens=pre_verify_seq_lens,
                            accept_tokens=accept_tokens,
                            num_accept_tokens=num_accept_tokens,
                            accept_token_stride=accept_token_stride,
                            commit_mask=commit_mask,
                        )

                class _TargetWorker:
                    def forward_batch_generation(
                        self, actual_batch, *, capture_hidden_mode
                    ):
                        events.append("target")
                        return batch_output

                worker = object.__new__(DecoupledVerifyWorker)
                worker.ps = SimpleNamespace(tp_rank=0)
                worker.gpu_tail_buffer = _CriticalTail()
                worker._target_worker = _TargetWorker()
                worker.topk = 1

                def build_next_input(tokens, *, topk):
                    return SimpleNamespace(bonus_tokens=tokens.to(torch.int32))

                def publish(new_seq_lens):
                    events.append("publish")
                    self.assertIs(new_seq_lens, seq_lens)

                with patch(
                    "sglang.srt.speculative.decoupled_verify_worker."
                    "build_decoupled_next_input",
                    side_effect=build_next_input,
                ):
                    actual = worker.forward_batch_generation(
                        batch,
                        on_publish=publish,
                    )

                self.assertIs(actual, batch_output)
                expected_events = ["target"]
                if has_new_lifecycle:
                    expected_events.append("open_wait")
                expected_events.extend(("commit", "publish"))
                self.assertEqual(events, expected_events)
                self.assertIs(commit_call["gpu_seats"], gpu_seats)
                self.assertIs(
                    commit_call["expected_request_epochs"],
                    expected_request_epochs,
                )
                self.assertIsNone(commit_call["pre_verify_seq_lens"])
                self.assertIs(
                    commit_call["accept_tokens"],
                    batch_output.next_draft_input.bonus_tokens,
                )
                self.assertEqual(
                    commit_call["num_accept_tokens"].tolist(),
                    [1] * batch_size,
                )
                self.assertEqual(commit_call["accept_token_stride"], 1)
                if expected_mask is None:
                    self.assertIsNone(commit_call["commit_mask"])
                else:
                    self.assertEqual(
                        commit_call["commit_mask"].tolist(),
                        expected_mask,
                    )

    def test_attach_preallocates_two_forward_slots(self):
        worker = object.__new__(DecoupledVerifyWorker)
        worker.ps = SimpleNamespace(tp_rank=1)
        worker.req_to_token_pool = SimpleNamespace(
            size=8, req_to_token=SimpleNamespace(shape=(9, 64))
        )
        worker.num_draft_tokens = 3
        worker.max_draft_tokens = 3
        worker._runtime_candidate_steps = ()
        worker.device = torch.device("cpu")
        worker.gpu_tail_buffer = None
        worker._gpu_tail_buffer_attached = False
        worker._gpu_tail_snapshot_buffers = None
        worker._gpu_tail_cursor_buffers = None
        worker._gpu_tail_expected_epoch_buffers = None
        worker._gpu_tail_landing_events = None
        worker._gpu_tail_debug_buffers = None
        worker._linear_selected_index_by_step = {}
        worker._linear_parent_list_by_step = {}

        worker.attach_gpu_tail_buffer(None)

        self.assertEqual(tuple(worker._gpu_tail_snapshot_buffers.shape), (2, 9, 5))
        self.assertEqual(tuple(worker._gpu_tail_cursor_buffers.shape), (2, 9))
        self.assertIsNone(worker._gpu_tail_expected_epoch_buffers)
        self.assertIsNone(worker._gpu_tail_landing_events)
        self.assertIsNone(worker._gpu_tail_debug_buffers)
        self.assertEqual(tuple(worker._linear_selected_index_by_step[3].shape), (9, 3))
        self.assertEqual(tuple(worker._linear_parent_list_by_step[3].shape), (9, 3))
        self.assertEqual(
            worker._linear_selected_index_by_step[3][0].tolist(), [0, 1, 2]
        )
        self.assertEqual(worker._linear_parent_list_by_step[3][0].tolist(), [-1, 0, 1])
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            worker.attach_gpu_tail_buffer(None)

    def test_tp0_preallocates_debug_without_widening_compact_snapshot(self):
        worker = object.__new__(DecoupledVerifyWorker)
        worker.ps = SimpleNamespace(tp_rank=0)
        worker.req_to_token_pool = SimpleNamespace(
            req_to_token=SimpleNamespace(shape=(9, 64))
        )
        worker.num_draft_tokens = 3
        worker.max_draft_tokens = 3
        worker._runtime_candidate_steps = ()
        worker.device = torch.device("cpu")
        worker.gpu_tail_buffer = None
        worker._gpu_tail_buffer_attached = False
        worker._gpu_tail_snapshot_buffers = None
        worker._gpu_tail_cursor_buffers = None
        worker._gpu_tail_expected_epoch_buffers = None
        worker._gpu_tail_landing_events = None
        worker._gpu_tail_debug_buffers = None
        worker._linear_selected_index_by_step = {}
        worker._linear_parent_list_by_step = {}
        gpu_tail = SimpleNamespace(num_seats=9, num_draft_tokens=3)

        worker.attach_gpu_tail_buffer(gpu_tail)

        self.assertEqual(tuple(worker._gpu_tail_snapshot_buffers.shape), (2, 9, 5))
        self.assertEqual(tuple(worker._gpu_tail_expected_epoch_buffers.shape), (2, 9))
        self.assertEqual(len(worker._gpu_tail_landing_events), 2)
        self.assertEqual(tuple(worker._gpu_tail_debug_buffers.shape), (2, 9, 10))

    def test_tp0_captures_batch_owned_request_epochs_in_overlap_slot(self):
        calls = []

        class _EpochTail:
            mock_profile = False
            landing_stream = torch.cpu.Stream()

            def capture_active_request_epochs(self, gpu_seats, *, out):
                calls.append((gpu_seats, out))
                out.copy_(torch.tensor([41, 43], dtype=torch.int64))
                return out

        worker = object.__new__(DecoupledVerifyWorker)
        worker.ps = SimpleNamespace(tp_rank=0)
        worker.gpu_tail_buffer = _EpochTail()
        worker._gpu_tail_expected_epoch_buffers = torch.empty((2, 4), dtype=torch.int64)
        worker._gpu_tail_landing_events = (torch.cpu.Event(), torch.cpu.Event())
        gpu_seats = torch.tensor([1, 3], dtype=torch.int64)
        batch = SimpleNamespace(
            forward_iter=3,
            req_pool_indices=gpu_seats,
            decoupled_has_new_lifecycle=True,
            decoupled_needs_landing_fence=False,
        )

        worker.capture_expected_request_epochs(batch)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], gpu_seats)
        self.assertEqual(batch.decoupled_expected_request_epochs.tolist(), [41, 43])
        self.assertEqual(
            batch.decoupled_expected_request_epochs.data_ptr(),
            worker._gpu_tail_expected_epoch_buffers[1, :2].data_ptr(),
        )
        self.assertIs(
            batch.decoupled_landing_event, worker._gpu_tail_landing_events[1]
        )

        lifecycle_fence_batch = SimpleNamespace(
            forward_iter=2,
            req_pool_indices=gpu_seats,
            decoupled_has_new_lifecycle=False,
            decoupled_needs_landing_fence=True,
        )
        worker.capture_expected_request_epochs(lifecycle_fence_batch)
        self.assertIs(
            lifecycle_fence_batch.decoupled_landing_event,
            worker._gpu_tail_landing_events[0],
        )

    def test_mock_profile_skips_request_epoch_capture(self):
        worker = object.__new__(DecoupledVerifyWorker)
        worker.ps = SimpleNamespace(tp_rank=0)
        worker.gpu_tail_buffer = SimpleNamespace(mock_profile=True)
        worker._gpu_tail_expected_epoch_buffers = torch.empty((2, 4), dtype=torch.int64)
        worker._gpu_tail_landing_events = (torch.cpu.Event(), torch.cpu.Event())
        batch = SimpleNamespace(
            forward_iter=0,
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            decoupled_expected_request_epochs=torch.tensor([99], dtype=torch.int64),
            decoupled_has_new_lifecycle=True,
            decoupled_needs_landing_fence=False,
            decoupled_landing_event=object(),
        )

        worker.capture_expected_request_epochs(batch)

        self.assertIsNone(batch.decoupled_expected_request_epochs)
        self.assertIsNone(batch.decoupled_landing_event)

    def test_tp0_selects_fixed_compact_snapshot(self):
        compact = torch.tensor(
            [[11, 12, 13, 3, 1], [21, 0, 0, 1, 1]], dtype=torch.int64
        )
        cursor = torch.tensor([4, 7], dtype=torch.int64)
        native_debug = torch.tensor(
            [
                [5, 10, 0, 3, 3, 0, 4, 0, 0, 0],
                [10, 11, 1, 2, 2, 0, 7, 0, 0, 1],
            ],
            dtype=torch.int64,
        )
        gpu_tail = _GpuTail(compact, cursor, native_debug)
        # The selector must preserve both CUDA-graph's padding seat 0 and the
        # final real seat in the physical request-pool tensor.
        seats = torch.tensor([0, 8], dtype=torch.int64)
        expected_request_epochs = torch.tensor([3, 5], dtype=torch.int64)
        seq_lens = torch.tensor([13, 17], dtype=torch.int64)
        bonus_tokens = torch.tensor([10, 20], dtype=torch.int32)

        debug_out = torch.empty((2, 10), dtype=torch.int64)
        selected, selected_lens, row_valid, debug, actual_cursor = (
            select_decoupled_gpu_tail_snapshot(
                gpu_tail_buffer=gpu_tail,
                tp_rank=0,
                tp_group=SimpleNamespace(world_size=1),
                gpu_seats=seats,
                expected_request_epochs=expected_request_epochs,
                seq_lens=seq_lens,
                bonus_tokens=bonus_tokens,
                num_draft_tokens=3,
                debug_out=debug_out,
            )
        )

        self.assertEqual(
            gpu_tail.calls,
            [(seats, expected_request_epochs, seq_lens, bonus_tokens)],
        )
        self.assertEqual(selected.tolist(), [[11, 12, 13], [21, 0, 0]])
        self.assertEqual(selected_lens.tolist(), [3, 1])
        self.assertEqual(row_valid.tolist(), [1, 1])
        self.assertIs(debug, debug_out)
        self.assertEqual(debug.tolist(), native_debug.tolist())
        self.assertIs(actual_cursor, cursor)

    def test_non_entry_rank_consumes_one_gpu_broadcast(self):
        source = torch.tensor([[31, 32, 0, 2, 1], [0, 0, 0, 0, 0]], dtype=torch.int64)
        tp_group = _BroadcastGroup(source)

        selected, selected_lens, row_valid, debug, cursor = (
            select_decoupled_gpu_tail_snapshot(
                gpu_tail_buffer=None,
                tp_rank=1,
                tp_group=tp_group,
                gpu_seats=torch.tensor([1, 2], dtype=torch.int64),
                expected_request_epochs=None,
                seq_lens=torch.tensor([8, 9], dtype=torch.int64),
                bonus_tokens=torch.tensor([7, 8], dtype=torch.int32),
                num_draft_tokens=3,
            )
        )

        self.assertEqual(len(tp_group.calls), 1)
        self.assertEqual(tp_group.calls[0][1], 0)
        self.assertEqual(selected.tolist(), [[31, 32, 0], [0, 0, 0]])
        self.assertEqual(selected_lens.tolist(), [2, 0])
        self.assertEqual(row_valid.tolist(), [1, 0])
        self.assertIsNone(debug)
        self.assertIsNone(cursor)

    def test_tp0_requires_gpu_tail(self):
        with self.assertRaisesRegex(RuntimeError, "no attached GPU draft-tail"):
            select_decoupled_gpu_tail_snapshot(
                gpu_tail_buffer=None,
                tp_rank=0,
                tp_group=SimpleNamespace(world_size=1),
                gpu_seats=torch.tensor([0], dtype=torch.int64),
                expected_request_epochs=torch.tensor([1], dtype=torch.int64),
                seq_lens=torch.tensor([8], dtype=torch.int64),
                bonus_tokens=torch.tensor([7], dtype=torch.int32),
                num_draft_tokens=3,
            )

    def test_malformed_native_snapshot_fails_before_verify(self):
        seats = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([8], dtype=torch.int64)
        bonus_tokens = torch.tensor([7], dtype=torch.int32)
        for compact, message in (
            (torch.zeros((1, 4), dtype=torch.int64), "shape mismatch"),
            (torch.zeros((1, 5), dtype=torch.int32), "must use int64"),
        ):
            with self.subTest(message=message):
                gpu_tail = _GpuTail(compact, torch.tensor([0], dtype=torch.int64))
                with self.assertRaisesRegex(RuntimeError, message):
                    select_decoupled_gpu_tail_snapshot(
                        gpu_tail_buffer=gpu_tail,
                        tp_rank=0,
                        tp_group=SimpleNamespace(world_size=1),
                        gpu_seats=seats,
                        expected_request_epochs=torch.tensor([1], dtype=torch.int64),
                        seq_lens=seq_lens,
                        bonus_tokens=bonus_tokens,
                        num_draft_tokens=3,
                    )

    def test_verify_input_terminal_markers_match_advanced_index_semantics(self):
        num_draft_tokens = 3
        batch_size = num_draft_tokens + 1
        selected_lens = torch.arange(batch_size, dtype=torch.int64)
        initial_retrieve = torch.arange(
            batch_size * (num_draft_tokens + 1), dtype=torch.int64
        ).reshape(batch_size, num_draft_tokens + 1)
        expected = initial_retrieve.clone()
        expected[
            torch.arange(batch_size),
            torch.clamp(selected_lens, max=num_draft_tokens),
        ] = -1

        worker = object.__new__(DecoupledVerifyWorker)
        worker.num_draft_tokens = num_draft_tokens
        worker.max_draft_tokens = num_draft_tokens
        worker.num_verify_tokens = num_draft_tokens + 1
        worker.topk = 1
        worker.device = torch.device("cpu")
        worker.ps = SimpleNamespace(tp_rank=0)
        worker._target_worker = SimpleNamespace()
        landing_order = []
        landing_event = object()
        worker.gpu_tail_buffer = SimpleNamespace(
            wait_for_landing_event=lambda event: landing_order.append(
                ("wait", event)
            )
        )
        worker._gpu_tail_snapshot_buffers = torch.empty(
            (2, batch_size, num_draft_tokens + 2), dtype=torch.int64
        )
        worker._gpu_tail_cursor_buffers = torch.empty(
            (2, batch_size), dtype=torch.int64
        )
        worker._gpu_tail_debug_buffers = torch.empty(
            (2, batch_size, 10), dtype=torch.int64
        )
        worker._linear_selected_index_by_step = {
            num_draft_tokens: (
                torch.arange(num_draft_tokens).expand(batch_size, -1).contiguous()
            )
        }
        worker._linear_parent_list_by_step = {
            num_draft_tokens: (
                torch.arange(-1, num_draft_tokens - 1)
                .expand(batch_size, -1)
                .contiguous()
            )
        }

        draft_input = SimpleNamespace(
            bonus_tokens=torch.arange(batch_size, dtype=torch.int32)
        )
        verify_input = SimpleNamespace(
            retrieve_next_token=initial_retrieve.clone(),
            capture_hidden_mode=None,
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            spec_info=draft_input,
            forward_iter=0,
            seq_lens=torch.arange(batch_size, dtype=torch.int64) + 8,
            req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
            decoupled_expected_request_epochs=torch.arange(
                1, batch_size + 1, dtype=torch.int64
            ),
            decoupled_landing_event=landing_event,
            decoupled_verify_steps=num_draft_tokens,
            device=torch.device("cpu"),
        )
        selected_tokens = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64)
        row_valid = torch.ones(batch_size, dtype=torch.int64)
        tail_select_debug = torch.zeros((batch_size, 10), dtype=torch.int64)
        logical_committed_lens = torch.arange(batch_size, dtype=torch.int64)

        def select_snapshot(**kwargs):
            landing_order.append(("select", None))
            return (
                selected_tokens,
                selected_lens,
                row_valid,
                tail_select_debug,
                logical_committed_lens,
            )

        with (
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.EagleDraftInput",
                SimpleNamespace,
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.get_tp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.select_decoupled_gpu_tail_snapshot",
                side_effect=select_snapshot,
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.build_eagle_verify_input",
                return_value=verify_input,
            ),
        ):
            actual = worker._build_verify_input(batch)

        self.assertIs(actual, verify_input)
        self.assertEqual(landing_order, [("wait", landing_event), ("select", None)])
        self.assertIs(batch.decoupled_tail_select_debug, tail_select_debug)
        self.assertTrue(torch.equal(actual.retrieve_next_token, expected))
        for row, terminal_index in enumerate(selected_lens.tolist()):
            self.assertEqual(actual.retrieve_next_token[row, terminal_index].item(), -1)


if __name__ == "__main__":
    unittest.main()

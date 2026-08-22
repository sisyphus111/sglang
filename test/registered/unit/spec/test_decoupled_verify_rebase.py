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
    def __init__(self, compact: torch.Tensor, cursor: torch.Tensor) -> None:
        self.compact = compact
        self.cursor = cursor
        self.calls = []

    def select_snapshot(
        self, gpu_seats, seq_lens, bonus_tokens, *, out=None, out_cursor=None
    ):
        self.calls.append((gpu_seats, seq_lens, bonus_tokens))
        if out is not None:
            out.copy_(self.compact)
            self.compact = out
        if out_cursor is not None:
            out_cursor.copy_(self.cursor)
            self.cursor = out_cursor
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
    def test_attach_preallocates_two_forward_slots(self):
        worker = object.__new__(DecoupledVerifyWorker)
        worker.ps = SimpleNamespace(tp_rank=1)
        worker.req_to_token_pool = SimpleNamespace(
            size=8, req_to_token=SimpleNamespace(shape=(9, 64))
        )
        worker.num_draft_tokens = 3
        worker.device = torch.device("cpu")
        worker.gpu_tail_buffer = None
        worker._gpu_tail_buffer_attached = False
        worker._gpu_tail_snapshot_buffers = None
        worker._gpu_tail_cursor_buffers = None
        worker._linear_selected_index = None
        worker._linear_parent_list = None

        worker.attach_gpu_tail_buffer(None)

        self.assertEqual(tuple(worker._gpu_tail_snapshot_buffers.shape), (2, 9, 5))
        self.assertEqual(tuple(worker._gpu_tail_cursor_buffers.shape), (2, 9))
        self.assertEqual(tuple(worker._linear_selected_index.shape), (9, 3))
        self.assertEqual(tuple(worker._linear_parent_list.shape), (9, 3))
        self.assertEqual(worker._linear_selected_index[0].tolist(), [0, 1, 2])
        self.assertEqual(worker._linear_parent_list[0].tolist(), [-1, 0, 1])
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            worker.attach_gpu_tail_buffer(None)

    def test_tp0_selects_fixed_compact_snapshot(self):
        compact = torch.tensor(
            [[11, 12, 13, 3, 1], [21, 0, 0, 1, 1]], dtype=torch.int64
        )
        cursor = torch.tensor([4, 7], dtype=torch.int64)
        gpu_tail = _GpuTail(compact, cursor)
        # The selector must preserve both CUDA-graph's padding seat 0 and the
        # final real seat in the physical request-pool tensor.
        seats = torch.tensor([0, 8], dtype=torch.int64)
        seq_lens = torch.tensor([13, 17], dtype=torch.int64)
        bonus_tokens = torch.tensor([10, 20], dtype=torch.int32)

        selected, selected_lens, row_valid, actual_cursor = (
            select_decoupled_gpu_tail_snapshot(
                gpu_tail_buffer=gpu_tail,
                tp_rank=0,
                tp_group=SimpleNamespace(world_size=1),
                gpu_seats=seats,
                seq_lens=seq_lens,
                bonus_tokens=bonus_tokens,
                num_draft_tokens=3,
            )
        )

        self.assertEqual(gpu_tail.calls, [(seats, seq_lens, bonus_tokens)])
        self.assertEqual(selected.tolist(), [[11, 12, 13], [21, 0, 0]])
        self.assertEqual(selected_lens.tolist(), [3, 1])
        self.assertEqual(row_valid.tolist(), [1, 1])
        self.assertIs(actual_cursor, cursor)

    def test_non_entry_rank_consumes_one_gpu_broadcast(self):
        source = torch.tensor([[31, 32, 0, 2, 1], [0, 0, 0, 0, 0]], dtype=torch.int64)
        tp_group = _BroadcastGroup(source)

        selected, selected_lens, row_valid, cursor = select_decoupled_gpu_tail_snapshot(
            gpu_tail_buffer=None,
            tp_rank=1,
            tp_group=tp_group,
            gpu_seats=torch.tensor([1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([8, 9], dtype=torch.int64),
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int32),
            num_draft_tokens=3,
        )

        self.assertEqual(len(tp_group.calls), 1)
        self.assertEqual(tp_group.calls[0][1], 0)
        self.assertEqual(selected.tolist(), [[31, 32, 0], [0, 0, 0]])
        self.assertEqual(selected_lens.tolist(), [2, 0])
        self.assertEqual(row_valid.tolist(), [1, 0])
        self.assertIsNone(cursor)

    def test_tp0_requires_gpu_tail(self):
        with self.assertRaisesRegex(RuntimeError, "no attached GPU draft-tail"):
            select_decoupled_gpu_tail_snapshot(
                gpu_tail_buffer=None,
                tp_rank=0,
                tp_group=SimpleNamespace(world_size=1),
                gpu_seats=torch.tensor([0], dtype=torch.int64),
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
        worker.num_verify_tokens = num_draft_tokens + 1
        worker.topk = 1
        worker.device = torch.device("cpu")
        worker.ps = SimpleNamespace(tp_rank=0)
        worker._target_worker = SimpleNamespace()
        worker.gpu_tail_buffer = SimpleNamespace()
        worker._gpu_tail_snapshot_buffers = torch.empty(
            (2, batch_size, num_draft_tokens + 2), dtype=torch.int64
        )
        worker._gpu_tail_cursor_buffers = torch.empty(
            (2, batch_size), dtype=torch.int64
        )
        worker._linear_selected_index = (
            torch.arange(num_draft_tokens).expand(batch_size, -1).contiguous()
        )
        worker._linear_parent_list = (
            torch.arange(-1, num_draft_tokens - 1).expand(batch_size, -1).contiguous()
        )

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
            device=torch.device("cpu"),
        )
        selected_tokens = torch.zeros((batch_size, num_draft_tokens), dtype=torch.int64)
        row_valid = torch.ones(batch_size, dtype=torch.int64)
        logical_committed_lens = torch.arange(batch_size, dtype=torch.int64)

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
                return_value=(
                    selected_tokens,
                    selected_lens,
                    row_valid,
                    logical_committed_lens,
                ),
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.build_eagle_verify_input",
                return_value=verify_input,
            ),
        ):
            actual = worker._build_verify_input(batch)

        self.assertIs(actual, verify_input)
        self.assertTrue(torch.equal(actual.retrieve_next_token, expected))
        for row, terminal_index in enumerate(selected_lens.tolist()):
            self.assertEqual(actual.retrieve_next_token[row, terminal_index].item(), -1)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for ForwardBatch MRoPE position source selection."""

import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.model_executor.forward_batch_info as forward_batch_info
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid: str) -> MagicMock:
    req = MagicMock()
    req.rid = rid
    req.lora_id = None
    req.token_type_ids = None
    return req


def _make_decode_batch(
    *,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    defer_decode_kv_binding: bool,
) -> ScheduleBatch:
    reqs = [_make_req(f"req-{index}") for index in range(seq_lens.shape[0])]
    batch = ScheduleBatch(
        reqs=reqs,
        model_config=MagicMock(),
        device="cpu",
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.forward_mode = ForwardMode.DECODE
    batch.input_ids = torch.arange(seq_lens.shape[0], dtype=torch.int64)
    batch.req_pool_indices = torch.arange(seq_lens.shape[0], dtype=torch.int64)
    batch.seq_lens = seq_lens
    batch.seq_lens_cpu = seq_lens_cpu
    batch.orig_seq_lens = seq_lens.to(torch.int32)
    batch.out_cache_loc = torch.arange(seq_lens.shape[0], dtype=torch.int64)
    batch.seq_lens_sum = None
    batch.defer_decode_kv_binding = defer_decode_kv_binding
    batch.multimodal_inputs = [None] * seq_lens.shape[0]
    return batch


def _make_mrope_runner() -> MagicMock:
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.model_config.model_is_mrope = True
    runner.ngram_embedding_manager.enabled = False
    runner.server_args.enable_lora = False
    runner.dcp_size = 1
    return runner


class TestForwardBatchMropePositions(CustomTestCase):
    def _init_forward_batch(self, batch: ScheduleBatch) -> ForwardBatch:
        with (
            get_context().override_server_args(),
            get_parallel().override(moe_ep_size=1),
            patch.object(
                forward_batch_info,
                "clamp_position",
                forward_batch_info._clamp_position_native,
            ),
        ):
            return ForwardBatch.init_new(
                batch,
                _make_mrope_runner(),
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_hidden_states_before_norm=False,
            )

    def test_gpu_only_decode_uses_reconciled_device_positions(self):
        batch = _make_decode_batch(
            seq_lens=torch.tensor([7, 11], dtype=torch.int64),
            # A stale host mirror, if present at the boundary, must be ignored.
            seq_lens_cpu=torch.tensor([70, 110], dtype=torch.int64),
            defer_decode_kv_binding=True,
        )

        with patch.object(
            ForwardBatch,
            "_compute_mrope_positions",
            side_effect=AssertionError("GPU-only decode entered the CPU MRoPE path"),
        ):
            forward_batch = self._init_forward_batch(batch)

        expected = torch.tensor(
            [[6, 10], [6, 10], [6, 10]], dtype=torch.int64
        )
        torch.testing.assert_close(forward_batch.mrope_positions, expected)
        self.assertEqual(forward_batch.mrope_positions.stride(), (0, 1))
        self.assertEqual(
            forward_batch.mrope_positions.data_ptr(),
            forward_batch.positions.data_ptr(),
        )
        self.assertIsNone(forward_batch.seq_lens_cpu)

    def test_ordinary_decode_preserves_cpu_mrope_path(self):
        batch = _make_decode_batch(
            seq_lens=torch.tensor([7, 11], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([7, 11], dtype=torch.int64),
            defer_decode_kv_binding=False,
        )

        with patch.object(
            ForwardBatch,
            "compute_spec_mrope_positions",
            side_effect=AssertionError("ordinary decode bypassed the CPU MRoPE path"),
        ):
            forward_batch = self._init_forward_batch(batch)

        expected = torch.tensor(
            [[6, 10], [6, 10], [6, 10]], dtype=torch.int64
        )
        torch.testing.assert_close(forward_batch.mrope_positions, expected)
        torch.testing.assert_close(
            forward_batch.seq_lens_cpu, torch.tensor([7, 11], dtype=torch.int64)
        )


if __name__ == "__main__":
    unittest.main()

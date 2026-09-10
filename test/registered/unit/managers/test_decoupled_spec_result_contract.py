"""CPU-only contracts for GPU-managed decoupled-drafter results."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecoupledSpecResultDispatch(CustomTestCase):
    @staticmethod
    def _make_scheduler(before_result):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.publish_load_snapshot = MagicMock()
        scheduler.decoupled_spec_manager = MagicMock()
        scheduler.decoupled_spec_manager.before_process_batch_result.return_value = (
            before_result
        )
        scheduler.batch_result_processor = MagicMock()
        scheduler._record_step_counters = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = MagicMock()
        scheduler.maybe_send_health_check_signal = MagicMock()
        return scheduler

    def test_explicit_true_skips_generic_processor_but_runs_bookkeeping(self):
        scheduler = self._make_scheduler(True)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        result = SimpleNamespace()

        scheduler.process_batch_result(batch, result)

        scheduler.batch_result_processor.process_batch_result_decode.assert_not_called()
        scheduler.decoupled_spec_manager.after_process_batch_result.assert_called_once_with(
            batch, result
        )
        scheduler._record_step_counters.assert_called_once_with(batch, result)
        scheduler.metrics_reporter.log_batch_result_stats.assert_called_once_with(
            batch, result
        )
        scheduler.metrics_reporter.update_device_timer.assert_called_once_with()

    def test_none_or_false_keeps_generic_processor(self):
        for before_result in (None, False):
            with self.subTest(before_result=before_result):
                scheduler = self._make_scheduler(before_result)
                batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                result = SimpleNamespace()

                scheduler.process_batch_result(batch, result)

                scheduler.batch_result_processor.process_batch_result_decode.assert_called_once_with(
                    batch, result
                )
                scheduler.decoupled_spec_manager.after_process_batch_result.assert_called_once_with(
                    batch, result
                )


class TestGenerationBatchResultGpuManagedCopy(CustomTestCase):
    def test_gpu_managed_decode_copies_only_allocator_metadata(self):
        next_token_ids = object()
        copy_done = MagicMock()
        kv_outcomes = torch.tensor([[1, 7, 11]])
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            copy_done=copy_done,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_kv_outcomes=kv_outcomes,
        )

        with patch("sglang.srt.managers.utils._async_d2h") as async_d2h:
            async_d2h.side_effect = lambda value: value
            result.copy_to_cpu(return_logprob=False, return_hidden_states=False)

        self.assertIs(result.next_token_ids, next_token_ids)
        self.assertEqual(
            async_d2h.call_args_list,
            [call(kv_outcomes)],
        )
        copy_done.record.assert_called_once_with()


class TestDecoupledDraftRetractionOrdering(CustomTestCase):
    def test_gpu_guard_runs_before_generic_retraction_mutates_resources(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.decoupled_spec_manager = MagicMock()
        scheduler.decoupled_spec_manager.sleep_overrun_requests.side_effect = (
            lambda batch: batch
        )
        scheduler.decoupled_spec_manager.before_decode_retraction.side_effect = (
            RuntimeError("unsupported GPU retraction")
        )
        batch = MagicMock()
        batch.batch_size.return_value = 1
        batch.is_empty.return_value = False
        batch.check_decode_mem.return_value = False

        with self.assertRaisesRegex(RuntimeError, "unsupported GPU retraction"):
            scheduler.update_running_batch(batch)

        scheduler.decoupled_spec_manager.before_decode_retraction.assert_called_once_with(
            batch
        )
        batch.retract_decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()

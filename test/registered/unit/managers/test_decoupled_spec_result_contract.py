"""CPU-only contracts for GPU-managed decoupled-drafter results."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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
    def test_mixed_copies_only_prefill_tokens_and_preserves_verifier_copy(self):
        outcomes = torch.tensor([[1, 7, 11]])
        tokens = torch.tensor([10, 20, 30])
        accept_lens = torch.tensor([2])
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=tokens,
            copy_done=MagicMock(),
            decoupled_draft_num_prefill_reqs=2,
            decoupled_draft_kv_outcomes=outcomes,
            accept_lens=accept_lens,
        )
        with patch(
            "sglang.srt.managers.utils._async_d2h", side_effect=lambda t: t.clone()
        ) as copy:
            result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
        self.assertEqual(result.next_token_ids.tolist(), [10, 20])
        self.assertIs(result.decoupled_draft_kv_outcomes, outcomes)
        self.assertIsNot(result.accept_lens, accept_lens)
        self.assertEqual(result.accept_lens.tolist(), [2])
        self.assertEqual(copy.call_count, 2)

    def test_gpu_managed_decode_leaves_allocator_copy_to_manager(self):
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
        async_d2h.assert_not_called()
        self.assertIs(result.decoupled_draft_kv_outcomes, kv_outcomes)
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


class TestDecoupledDraftForwardScheduling(CustomTestCase):
    def test_both_schedules_use_the_same_forward_transaction(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap), get_context().override_server_args(
                model_path="dummy", decoupled_spec_role="drafter"
            ):
                events = []
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_overlap = overlap
                scheduler.enable_pdmux = False
                scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
                scheduler.forward_ct = 0
                scheduler.scripted_scheduler_hook = None
                scheduler.profiler_manager = MagicMock()
                scheduler.forward_sleep_time = None
                scheduler.disaggregation_mode = None
                scheduler.is_generation = True
                scheduler.future_map = MagicMock()
                scheduler._confidence_budget_prepare = None
                scheduler.forward_stream_ctx = nullcontext()
                scheduler.forward_stream = MagicMock()
                scheduler.device_module = MagicMock()
                scheduler.forward_stream.synchronize.side_effect = (
                    lambda: events.append("retire_forward")
                )
                scheduler.schedule_stream = object()
                scheduler._forward_isolation = MagicMock(return_value=nullcontext())
                scheduler.enable_unified_memory = False
                scheduler._maybe_report_active_ranks = MagicMock()
                scheduler._relay_forward_payload = MagicMock()
                scheduler.decoupled_spec_manager = MagicMock()
                manager = scheduler.decoupled_spec_manager
                manager.prepare_batch.side_effect = lambda _: events.append("batch")
                manager.prepare_forward.side_effect = lambda _: events.append("prepare")
                manager.finish_forward.side_effect = (
                    lambda *_: events.append("finish") or True
                )
                result = GenerationBatchResult(
                    logits_output=LogitsProcessorOutput(next_token_logits=None),
                    next_token_ids=torch.tensor([11]),
                    decoupled_draft_gpu_managed=True,
                )
                scheduler.model_worker = MagicMock()
                scheduler.model_worker.forward_batch_generation.side_effect = (
                    lambda *_, **__: events.append("model") or result
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    req_pool_indices=torch.tensor([1]),
                    return_logprob=False,
                    return_hidden_states=False,
                )
                with patch(
                    "sglang.srt.managers.scheduler.resolve_forward_inputs",
                    side_effect=lambda *_: events.append("resolve"),
                ):
                    self.assertIs(scheduler.run_batch(batch), result)

                expected = ["batch", "resolve", "prepare", "model", "finish"]
                self.assertEqual(events, expected)
                self.assertIsNone(batch.input_ids)
                self.assertIsNone(result.copy_done)
                scheduler._relay_forward_payload.assert_not_called()
                scheduler.future_map.publish.assert_not_called()
                scheduler.forward_stream.synchronize.assert_not_called()
                if overlap:
                    scheduler.forward_stream.wait_stream.assert_called_once_with(
                        scheduler.schedule_stream
                    )
                else:
                    scheduler.forward_stream.wait_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()

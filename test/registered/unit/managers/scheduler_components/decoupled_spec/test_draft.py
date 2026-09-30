"""CPU-only contract tests for the decoupled drafter scheduler component."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context, get_observability
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler_components.decoupled_spec.draft import (  # noqa: E402
    DecoupledDraftManager,
    _DraftRequestState,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.decoupled_draft_checkpoint import (  # noqa: E402
    DraftRequestGeneration,
)
from sglang.srt.speculative.decoupled_spec_io import (  # noqa: E402
    DecoupledSpecIpcConfig,
    DraftReqKey,
    DraftSync,
    ReadyDraftControls,
)

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_DRAFT_MODULE = "sglang.srt.managers.scheduler_components.decoupled_spec.draft"


class _DraftBatch:
    def __init__(self, reqs, *, forward_mode=ForwardMode.DECODE) -> None:
        self.reqs = list(reqs)
        self.forward_mode = forward_mode
        self.batch_is_full = True
        self.defer_decode_kv_binding = False
        self.decoupled_draft_mirror_seats = None
        self.decoupled_draft_num_prefill_reqs = None
        self.decoupled_draft_num_prefill_tokens = None
        self.chunked_req = None

    def filter_batch(self, *, keep_indices) -> None:
        self.reqs = [self.reqs[index] for index in keep_indices]

    def is_empty(self) -> bool:
        return not self.reqs

    def merge_batch(self, other) -> None:
        self.reqs.extend(other.reqs)


class TestDecoupledDraftManager(CustomTestCase):
    enable_overlap = False

    def setUp(self) -> None:
        context = get_context().override_server_args(
            model_path="dummy", enable_metrics=False
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        stream_patcher = patch(f"{_DRAFT_MODULE}.get_stream", return_value=object())
        stream_patcher.start()
        self.addCleanup(stream_patcher.stop)
        data_plane_patcher = patch(
            f"{_DRAFT_MODULE}.create_drafter_decoupled_spec_data_plane",
            autospec=True,
        )
        self.addCleanup(data_plane_patcher.stop)
        data_plane_factory = data_plane_patcher.start()
        self.data_plane = data_plane_factory.return_value
        self.data_plane.collect_lifecycle_controls.return_value = ReadyDraftControls()
        self.data_plane.drain_gpu_progress.return_value = []
        self.data_plane.pending_control_count.return_value = 0
        self.data_plane.take_transport_metrics.return_value = {
            "num_draft_result_frames": 0,
            "num_draft_result_tokens": 0,
            "draft_send_queue_latency_us": {
                "count": 0,
                "sum_us": 0,
                "bucket_upper_bounds_us": [10, 100],
                "bucket_counts": [0, 0, 0],
            },
            "draft_send_queue_depth_max": 0,
        }
        self.scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0, tp_size=1),
            enable_overlap=self.enable_overlap,
            max_running_requests=16,
            max_total_num_tokens=128,
            device_module=SimpleNamespace(Event=MagicMock()),
            server_args=SimpleNamespace(
                speculative_num_steps=3,
                enable_metrics=False,
                enable_mixed_chunk=False,
            ),
            metrics_reporter=MagicMock(),
            req_to_token_pool=SimpleNamespace(
                mamba_pool=None,
                req_to_token=torch.zeros((16, 128), dtype=torch.int32),
            ),
            tokenizer=MagicMock(),
            model_config=SimpleNamespace(
                vocab_size=128,
                hf_eos_token_id={2},
            ),
            metrics_collector=None,
            device=torch.device("cpu"),
            init_req_max_new_tokens=MagicMock(),
            _add_request_to_queue=MagicMock(),
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=1,
                free=MagicMock(),
                alloc=MagicMock(),
                available_size=MagicMock(return_value=128),
            ),
            schedule_stream=SimpleNamespace(
                wait_stream=MagicMock(), wait_event=MagicMock()
            ),
            forward_stream=object(),
            result_queue=[],
            chunked_req=None,
            _pending_chunked_abort_req=None,
            waiting_queue=[],
            running_batch=None,
            last_batch=None,
            cur_batch_for_debug=None,
            tree_cache=SimpleNamespace(supports_mamba=lambda: False),
            future_map=SimpleNamespace(
                needs_cpu_seq_lens=False,
                output_tokens_buf=torch.zeros(16, dtype=torch.int64),
                stash=MagicMock(),
            ),
        )
        config = DecoupledSpecIpcConfig(
            bind_endpoint="ipc:///tmp/unused-drafter",
            connect_endpoints=("ipc:///tmp/unused-verifier",),
            rank=0,
        )
        self.manager = DecoupledDraftManager(self.scheduler, config)
        self.manager._routing_indices = torch.empty((2, 16), dtype=torch.int64)
        self.data_plane.lookup_gpu_binding.return_value = (1, 0)
        self.data_plane.start.assert_called_once_with()
        self.data_plane.reset_mock()

    def test_transport_metrics_exchange_keeps_empty_latency_window(self):
        window = self.manager.take_decode_metrics_window()

        self.data_plane.take_transport_metrics.assert_called_once_with()
        self.assertIsNone(window.tail_select)
        self.assertEqual(window.transport.num_draft_result_frames, 0)
        self.assertEqual(window.transport.num_draft_result_tokens, 0)
        histogram = window.transport.draft_send_queue_latency_us
        self.assertEqual(histogram.count, 0)
        self.assertEqual(histogram.bucket_counts, [0, 0, 0])

    def test_two_decode_results_finalize_two_metrics_windows(self):
        metrics_reporter = MagicMock()
        self.scheduler.metrics_reporter = metrics_reporter
        batch = _DraftBatch([])
        result = GenerationBatchResult()

        self.manager.after_process_batch_result(batch, result)
        self.manager.after_process_batch_result(batch, result)

        self.assertEqual(
            metrics_reporter.finish_decoupled_decode_metrics_window.call_count,
            2,
        )

    @staticmethod
    def _sync(request_id="req"):
        return DraftSync(
            request_id=request_id,
            src_verifier_rank=0,
            dst_drafter_rank=0,
            max_new_tokens=32,
            prompt_token_ids=[1, 2, 3],
            committed_outputs=[10],
        )

    def _install_request(
        self,
        *,
        output_tokens,
        committed_len=1,
        request_id="req",
        src_verifier_rank=0,
    ):
        req = SimpleNamespace(
            rid=f"draft:{src_verifier_rank}:{request_id}",
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q", output_tokens),
            req_pool_idx=1,
            mamba_pool_idx=torch.tensor(2),
            kv=SimpleNamespace(kv_allocated_len=6),
            kv_committed_len=6,
            cache_protected_len=6,
            inflight_middle_chunks=0,
            full_untruncated_fill_ids=array("q"),
            finished_reason=None,
            finished_len=None,
            _refresh_fill_ids=MagicMock(),
        )
        key = DraftRequestGeneration(
            src_verifier_rank=src_verifier_rank,
            request_id=request_id,
            request_epoch=0,
        )
        req.decoupled_draft_generation = key
        state = _DraftRequestState(
            key=key,
            req=req,
            is_sleeping=False,
            gpu_seat=1,
            gpu_checkpoint_slots_initialized=True,
            kv_highwater_len=6,
            gpu_ahead_len=len(output_tokens) - committed_len,
        )
        self.manager._requests[(src_verifier_rank, request_id)] = state
        return req, state

    def test_sync_materializes_internal_req_before_queueing(self):
        events = []
        sampling_params = MagicMock()
        fake_req = SimpleNamespace(
            rid="draft:0:req",
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q"),
            _refresh_fill_ids=MagicMock(side_effect=lambda: events.append("refresh")),
        )
        self.scheduler.init_req_max_new_tokens.side_effect = lambda req: events.append(
            "init_max_new_tokens"
        )
        self.scheduler._add_request_to_queue.side_effect = lambda req: events.append(
            "enqueue"
        )
        self.data_plane.collect_lifecycle_controls.return_value = ReadyDraftControls(
            sync_messages=[self._sync()]
        )

        with patch(
            f"{_DRAFT_MODULE}.SamplingParams", return_value=sampling_params
        ) as sampling_cls, patch(
            f"{_DRAFT_MODULE}.Req", return_value=fake_req
        ) as req_cls:
            self.manager.process_pending_controls()

        self.assertEqual(sampling_cls.call_args.kwargs["max_new_tokens"], 1 << 30)
        sampling_params.normalize.assert_called_once_with(self.scheduler.tokenizer)
        sampling_params.verify.assert_called_once_with(128)
        self.assertEqual(req_cls.call_args.args[0], "draft:0:req")
        self.assertEqual(list(req_cls.call_args.args[2]), [1, 2, 3])
        self.assertEqual(list(fake_req.output_ids), [10])
        self.assertEqual(events, ["refresh", "init_max_new_tokens", "enqueue"])
        state = self.manager._requests[(0, "req")]
        self.assertIs(state.req, fake_req)
        self.assertEqual(state.gpu_ahead_len, 0)

    def test_sleep_overrun_transfers_ahead_request_before_decode_alloc(self):
        behind_req, _ = self._install_request(
            request_id="behind", output_tokens=range(self.manager.schedule_ahead_limit)
        )
        ahead_req, _ = self._install_request(request_id="ahead", output_tokens=range(8))
        batch = _DraftBatch([behind_req, ahead_req])
        batch.out_cache_loc = None

        self.assertIs(self.manager.sleep_overrun_requests(batch), batch)

        self.assertEqual(batch.reqs, [behind_req])
        ahead_state = self.manager._state_for_req(ahead_req)
        self.assertIs(self.manager._sleeping_requests[ahead_state.key], ahead_req)
        self.assertTrue(ahead_state.is_sleeping)
        self.assertFalse(batch.batch_is_full)
        self.assertIsNone(batch.out_cache_loc)

    def test_gpu_state_uses_lifecycle_with_compact_gpu_credit(self):
        req, state = self._install_request(output_tokens=range(7))
        self.manager._gpu_lifecycle_poll_count = 15
        self.data_plane.pending_control_count.return_value = 1
        batch = _DraftBatch([req])
        routed_capture = MagicMock()
        indexer_capture = MagicMock()
        result = GenerationBatchResult(
            copy_done=None,
            next_token_ids=torch.tensor([12], dtype=torch.int64),
            routed_experts_output=routed_capture,
            indexer_topk_output=indexer_capture,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[0, -1, -1]]),
            can_run_cuda_graph=True,
        )

        self.manager.process_pending_controls()
        self.scheduler.metrics_collector = MagicMock()
        with get_observability().override(enable_metrics=True):
            self.assertTrue(self.manager.before_process_batch_result(batch, result))
        self.scheduler.metrics_collector.increment_decode_cuda_graph_pass.assert_called_once_with(
            value=True
        )
        routed_capture.finalize.assert_called_once_with()
        indexer_capture.finalize.assert_called_once_with()
        self.assertIsNone(result.routed_experts_output)
        self.assertIsNone(result.indexer_topk_output)
        self.manager.after_process_batch_result(batch, result)

        self.data_plane.collect_lifecycle_controls.assert_called_once_with()
        self.data_plane.drain_gpu_progress.assert_called()
        # Native snapshot egress owns ACK/tail publication in GPU mode. The
        # CPU Req transcript must not replay even an apparently unpublished tail.
        self.data_plane.publish_tails.assert_not_called()
        self.assertEqual(list(req.output_ids), list(range(7)))

    def test_gpu_state_parks_a_row_without_runnable_credit(self):
        req, state = self._install_request(output_tokens=range(32))
        state.gpu_ahead_len = self.manager.schedule_ahead_limit
        batch = _DraftBatch([req])

        self.assertIs(self.manager.sleep_overrun_requests(batch), batch)
        self.assertEqual(batch.reqs, [])
        self.assertTrue(state.is_sleeping)
        self.assertIs(self.manager._sleeping_requests[state.key], req)

    def test_gpu_state_accepts_forced_replay_credit_without_token_shadow(self):
        _, state = self._install_request(output_tokens=[10])
        self.data_plane.drain_gpu_progress.return_value = [("req", 0, 0, 3, 3, 0)]

        self.manager.process_pending_controls()

        self.assertEqual(state.gpu_ahead_len, 0)

    def test_gpu_state_discards_queued_result_after_close_and_same_id_reopen(self):
        old_req, _ = self._install_request(output_tokens=[10, 11])
        stale_batch = _DraftBatch([old_req])
        stale_result = GenerationBatchResult(
            copy_done=None,
            next_token_ids=torch.tensor([99], dtype=torch.int64),
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[0, -1, -1]]),
        )

        with patch(f"{_DRAFT_MODULE}.release_kv_cache"):
            self.manager._close_request_key(DraftReqKey(0, "req"))

        new_req, new_state = self._install_request(output_tokens=[20])
        new_key = DraftRequestGeneration(
            src_verifier_rank=0,
            request_id="req",
            request_epoch=1,
        )
        new_req.decoupled_draft_generation = new_key
        new_state.key = new_key
        self.data_plane.publish_tails.reset_mock()

        self.assertTrue(
            self.manager.before_process_batch_result(stale_batch, stale_result)
        )
        self.manager.after_process_batch_result(stale_batch, stale_result)

        self.assertTrue(old_req.is_retracted)
        self.assertEqual(list(new_req.output_ids), [20])
        self.assertIs(self.manager._requests[(0, "req")].req, new_req)
        self.data_plane.publish_tails.assert_not_called()

    def test_gpu_state_final_prefill_rejection_is_a_stale_result_not_assertion(self):
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        batch.contains_last_prefill_chunk = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1], dtype=torch.int64)
        batch.decoupled_draft_request_epochs = torch.tensor([0], dtype=torch.int64)
        batch.req_pool_indices = torch.tensor([1], dtype=torch.int64)
        batch.seq_lens = torch.tensor([5], dtype=torch.int64)
        batch.out_cache_loc = torch.tensor([17], dtype=torch.int64)
        batch.decoupled_draft_captured_state_positions = None
        result = GenerationBatchResult(
            copy_done=None,
            next_token_ids=torch.tensor([77], dtype=torch.int64),
        )
        self.scheduler.future_map = SimpleNamespace(
            output_tokens_buf=torch.zeros(4, dtype=torch.int64),
            new_seq_lens_buf=torch.zeros(4, dtype=torch.int64),
            stash=MagicMock(),
        )
        self.manager._gpu_identity_done_event = MagicMock()

        def reject_prefill(*args, accept_out, **kwargs):
            accept_out.fill_(False)

        self.data_plane.gpu_tail_buffer.append_prefill_sample.side_effect = (
            reject_prefill
        )

        self.assertTrue(self.manager.finish_forward(batch, result))
        self.manager._gpu_identity_done_event.record.assert_not_called()
        self.assertEqual(self.scheduler.future_map.output_tokens_buf.tolist(), [0] * 4)
        self.assertEqual(self.scheduler.future_map.new_seq_lens_buf.tolist(), [0] * 4)
        self.scheduler.future_map.stash.assert_called_once()
        self.assertEqual(result.decoupled_draft_candidate_committed.tolist(), [False])
        self.assertFalse(getattr(result, "decoupled_draft_gpu_managed", False))
        self.assertFalse(self.manager.before_process_batch_result(batch, result))
        self.assertTrue(req.is_retracted)

    def test_gpu_state_accepted_prefill_keeps_generic_result_processing(self):
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        result = GenerationBatchResult(
            copy_done=None,
            next_token_ids=torch.tensor([77], dtype=torch.int64),
            decoupled_draft_gpu_managed=False,
            decoupled_draft_candidate_committed=torch.tensor([True]),
        )

        self.assertFalse(self.manager.before_process_batch_result(batch, result))

    def test_gpu_state_mixed_prefill_defers_middle_chunk_state(self):
        final_req, _ = self._install_request(request_id="final", output_tokens=[10])
        middle_req, _ = self._install_request(request_id="middle", output_tokens=[])
        middle_req.inflight_middle_chunks = 1
        batch = ScheduleBatch(
            reqs=[final_req, middle_req], forward_mode=ForwardMode.EXTEND
        )
        batch.chunked_req = middle_req
        batch.contains_last_prefill_chunk = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1, 2])
        batch.decoupled_draft_request_epochs = torch.tensor([0, 0])
        batch.req_pool_indices = torch.tensor([1, 2])
        result = GenerationBatchResult(
            copy_done=None, next_token_ids=torch.tensor([11, 99])
        )

        def append_prefill(seats, epochs, tokens, *, accept_out):
            accept_out.copy_(seats >= 0)

        self.data_plane.gpu_tail_buffer.append_prefill_sample.side_effect = (
            append_prefill
        )
        self.assertTrue(self.manager.finish_forward(batch, result))
        call_args = self.data_plane.gpu_tail_buffer.append_prefill_sample.call_args
        self.assertEqual(call_args.args[0].tolist(), [1, -1])
        self.assertEqual(batch.decoupled_draft_mirror_seats.tolist(), [1, 2])
        self.assertEqual(
            result.decoupled_draft_candidate_committed.tolist(), [True, False]
        )
        batch = batch.copy()
        self.assertIs(batch.chunked_req, middle_req)
        self.assertFalse(self.manager.before_process_batch_result(batch, result))
        self.assertFalse(getattr(middle_req, "is_retracted", False))
        self.assertEqual(middle_req.inflight_middle_chunks, 1)
        self.assertEqual(list(middle_req.output_ids), [])

        # Checkpoint tags are published by GPU finish, never reconstructed
        # from the final-prefill CPU result.
        self.manager.checkpoints = MagicMock()
        self.manager.after_process_batch_result(batch, result)
        self.assertEqual(self.manager.checkpoints.mock_calls, [])
        self.assertIsNone(batch.mamba_cache_src_indices)
        self.assertIsNone(batch.mamba_cache_dst_indices)

    def test_gpu_state_candidate_ownership_advances_exact_kv_highwater(self):
        owned_req, owned_state = self._install_request(
            request_id="owned",
            output_tokens=[10],
        )
        rejected_req, rejected_state = self._install_request(
            request_id="rejected",
            output_tokens=[20],
        )
        batch = _DraftBatch([owned_req, rejected_req])
        result = GenerationBatchResult(
            copy_done=None,
            next_token_ids=torch.tensor([11, 21], dtype=torch.int64),
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[1, 9, -1], [0, -1, -1]]),
        )

        self.assertTrue(self.manager.before_process_batch_result(batch, result))
        self.assertEqual(len(self.manager._pending_kv_outcomes), 1)
        self.manager._flush_pending_kv_outcomes()

        self.assertEqual(owned_state.kv_highwater_len, 10)
        self.assertEqual(rejected_state.kv_highwater_len, 6)

    def test_gpu_state_decode_result_defers_control_poll_to_next_loop(self):
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        result = GenerationBatchResult(
            copy_done=None,
            next_token_ids=torch.tensor([11], dtype=torch.int64),
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[1, 6, -1]]),
        )
        self.data_plane.collect_lifecycle_controls.reset_mock()

        self.assertTrue(self.manager.before_process_batch_result(batch, result))

        self.data_plane.collect_lifecycle_controls.assert_not_called()
        self.assertEqual(len(self.manager._pending_kv_outcomes), 1)

    def test_gpu_state_reclaims_only_finish_reported_kv_locations(self):
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        result = GenerationBatchResult(
            copy_done=None,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[1, 6, 123]]),
        )

        self.assertTrue(self.manager.before_process_batch_result(batch, result))
        self.manager._flush_pending_kv_outcomes()

        freed = self.scheduler.token_to_kv_pool_allocator.free.call_args.args[0]
        self.assertEqual(freed.tolist(), [123])

    def test_gpu_state_reclaim_waits_for_copy_without_blocking_decode(self):
        req, state = self._install_request(output_tokens=[10])
        done = MagicMock()
        done.query.return_value = False
        self.manager._inflight_kv_outcomes = (
            [req],
            torch.tensor([[1, 9, 123]]),
            done,
            [],
        )
        self.manager._flush_pending_kv_outcomes(blocking=False)
        done.synchronize.assert_not_called()
        self.assertEqual(state.kv_highwater_len, 6)
        self.scheduler.token_to_kv_pool_allocator.free.assert_not_called()

        done.query.return_value = True
        self.manager._flush_pending_kv_outcomes(blocking=False)
        self.assertEqual(state.kv_highwater_len, 10)
        self.assertIsNone(self.manager._inflight_kv_outcomes)
        self.manager._flush_pending_kv_outcomes(blocking=False)
        self.scheduler.token_to_kv_pool_allocator.free.assert_called_once()

    def test_gpu_state_close_drains_inflight_kv_before_release(self):
        req, state = self._install_request(output_tokens=[10])
        self.manager.checkpoints = MagicMock()
        done = MagicMock()
        self.manager._inflight_kv_outcomes = (
            [req],
            torch.tensor([[1, 34, 123]]),
            done,
            [],
        )
        with patch(f"{_DRAFT_MODULE}.release_kv_cache") as release:
            self.manager._close_request_key(DraftReqKey(0, "req"))
        done.synchronize.assert_called_once()
        self.assertEqual(req.kv_committed_len, 35)
        self.assertIsNone(self.manager._inflight_kv_outcomes)
        release.assert_called_once()

    def test_gpu_state_close_releases_exact_kv_highwater(self):
        req, state = self._install_request(output_tokens=[10])
        state.kv_highwater_len = 35
        self.manager.checkpoints = MagicMock()

        with patch(f"{_DRAFT_MODULE}.release_kv_cache") as release:
            self.manager._close_request_key(DraftReqKey(0, "req"))

        if self.enable_overlap:
            self.scheduler.schedule_stream.wait_stream.assert_called_once_with(
                self.scheduler.forward_stream
            )
        else:
            self.scheduler.schedule_stream.wait_stream.assert_not_called()
        self.assertEqual(req.kv_committed_len, 35)
        self.assertEqual(req.kv.kv_allocated_len, 35)
        self.manager.checkpoints.release.assert_called_once_with(state.key)
        release.assert_called_once_with(
            req,
            self.scheduler.tree_cache,
            is_insert=False,
        )

    def test_gpu_state_close_accounts_for_queued_kv_outcome(self):
        req, state = self._install_request(output_tokens=[10])
        live_req, live_state = self._install_request(
            request_id="live", output_tokens=[20]
        )
        copy_done = MagicMock()
        queued_result = GenerationBatchResult(
            copy_done=copy_done,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[1, 9, -1], [1, 99, -1]]),
        )
        queued_batch = _DraftBatch([req, live_req])
        self.scheduler.result_queue = [(queued_batch, queued_result)]
        self.manager.checkpoints = MagicMock()

        with patch(f"{_DRAFT_MODULE}.release_kv_cache"):
            self.manager._close_request_key(DraftReqKey(0, "req"))

        copy_done.synchronize.assert_not_called()
        self.assertEqual(state.kv_highwater_len, 10)
        self.assertEqual(req.kv_committed_len, 10)
        self.assertEqual(req.kv.kv_allocated_len, 10)
        self.assertEqual(live_state.kv_highwater_len, 100)
        self.assertTrue(queued_result.decoupled_draft_kv_outcomes_drained)
        self.assertTrue(
            self.manager.before_process_batch_result(queued_batch, queued_result)
        )
        self.assertEqual(self.manager._pending_kv_outcomes, [])

    def test_gpu_state_mixed_decode_batch_fails_before_allocation(self):
        decoupled_req, _ = self._install_request(output_tokens=[10])
        ordinary_req = SimpleNamespace(decoupled_draft_generation=None)
        batch = _DraftBatch([decoupled_req, ordinary_req])
        batch.defer_decode_kv_binding = False

        with self.assertRaisesRegex(RuntimeError, "cannot mix"):
            self.manager.prepare_decode_allocation(batch)

        batch = _DraftBatch([decoupled_req])
        batch.defer_decode_kv_binding = False
        self.manager.prepare_decode_allocation(batch)
        self.assertTrue(batch.defer_decode_kv_binding)
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)

    def test_gpu_state_retraction_fails_before_generic_resource_mutation(self):
        req, _ = self._install_request(output_tokens=[10])

        with self.assertRaisesRegex(RuntimeError, "cannot retract"):
            self.manager.before_decode_retraction(_DraftBatch([req]))

    def test_pause_retraction_preserves_sleeping_gpu_ownership(self):
        req, state = self._install_request(output_tokens=range(8))
        self.manager.sleep_overrun_requests(_DraftBatch([req]))
        self.manager.checkpoints = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "cannot re-prefill"):
            self.manager.prepare_pause_retract([])

        self.assertIs(self.manager._sleeping_requests[state.key], req)
        self.assertTrue(state.is_sleeping)
        self.manager.checkpoints.release.assert_not_called()
        self.scheduler.token_to_kv_pool_allocator.free.assert_not_called()

    def test_ordinary_retraction_preserves_generic_abort_decision(self):
        retry = [SimpleNamespace(decoupled_draft_generation=None)]
        abort = [SimpleNamespace(decoupled_draft_generation=None)]
        retry_result, abort_result = self.manager.handle_retracted_requests(
            retry, abort
        )
        self.assertIs(retry_result, retry)
        self.assertIs(abort_result, abort)

    def test_gpu_state_identity_table_fences_only_topology_changes(self):
        self.manager.checkpoints = SimpleNamespace(capacity=7)
        self.manager._routing_indices = torch.empty((2, 4), dtype=torch.int64)
        self.manager._gpu_batch_seats = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_batch_epochs = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_batch_seats_cpu = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_batch_epochs_cpu = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_identity_done_event = MagicMock()
        self.manager._gpu_identity_in_use = False
        self.scheduler.schedule_stream.wait_event = MagicMock()

        req, state = self._install_request(request_id="ring", output_tokens=[10])
        state.gpu_seat = 10
        state.key = DraftRequestGeneration(0, "ring", 20)
        req.decoupled_draft_generation = state.key
        batches = []
        for index in range(4):
            batch = _DraftBatch([req])
            batch.defer_decode_kv_binding = True
            self.manager._assign_gpu_identity(batch)
            batches.append(batch)

        self.scheduler.schedule_stream.wait_event.assert_not_called()
        self.manager._gpu_identity_done_event.record.assert_not_called()
        self.assertEqual(int(batches[-1].decoupled_draft_mirror_seats[0]), 10)
        self.assertEqual(int(batches[-1].decoupled_draft_request_epochs[0]), 20)

        self.scheduler.schedule_stream.wait_event.reset_mock()
        self.manager._gpu_decode_binding_validated = True
        previous_epochs = self.manager._gpu_batch_epochs_cpu
        state.key = DraftRequestGeneration(0, "ring", 21)
        req.decoupled_draft_generation = state.key
        replacement = _DraftBatch([req])
        replacement.defer_decode_kv_binding = True
        self.manager._assign_gpu_identity(replacement)
        self.assertEqual(
            self.scheduler.schedule_stream.wait_event.call_args_list,
            [call(self.manager._gpu_identity_done_event)],
        )
        self.assertEqual(int(replacement.decoupled_draft_request_epochs[0]), 21)
        self.assertEqual(previous_epochs.tolist(), [20])
        self.assertFalse(self.manager._gpu_decode_binding_validated)
        self.manager._gpu_identity_done_event.record.assert_called_once_with(
            self.scheduler.forward_stream
            if self.enable_overlap
            else self.scheduler.schedule_stream
        )

    def test_gpu_state_bs1_uses_zero_copy_future_token_view(self):
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        batch.defer_decode_kv_binding = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1])
        batch.input_ids = None
        self.manager._gpu_identity_reqs = batch.reqs

        self.manager.prepare_batch(batch)

        expected = self.scheduler.future_map.output_tokens_buf[1:2]
        self.assertEqual(batch.input_ids.data_ptr(), expected.data_ptr())

    def test_decode_resolves_device_state_and_retires_allocator_outcomes(self):
        req, state = self._install_request(output_tokens=[10])
        batch = ScheduleBatch(reqs=[req], forward_mode=ForwardMode.DECODE)
        batch.req_pool_indices = torch.tensor([req.req_pool_idx])
        batch.req_to_token_pool = self.scheduler.req_to_token_pool
        batch.out_cache_loc = torch.tensor([41])
        batch.seq_lens = torch.tensor([100])
        batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
        batch.input_ids = torch.tensor([10])
        self.manager.prepare_decode_allocation(batch)
        self.manager.prepare_batch(batch)

        def resolve(*args, **kwargs):
            # Simulate a GPU rewind after CPU batch preparation. The next
            # model input must come from the device transaction in both modes.
            kwargs["resolved_input_ids"].fill_(99)
            kwargs["resolved_seq_lens"].fill_(7)
            kwargs["resolved_orig_seq_lens"].fill_(7)
            kwargs["captured_state_positions"].fill_(6)
            kwargs["old_cache_locs"].fill_(31)

        def finish(*args, **kwargs):
            kwargs["kv_outcomes"].copy_(torch.tensor([[1, 6, 31]]))
            kwargs["future_output_tokens"][req.req_pool_idx] = args[4][0]

        gpu = self.data_plane.gpu_tail_buffer
        gpu.prepare_decode.side_effect = resolve
        gpu.finish_decode.side_effect = finish
        self.manager.prepare_forward(batch)
        self.assertEqual(batch.input_ids.tolist(), [99])
        self.assertEqual(batch.seq_lens.tolist(), [7])
        self.assertIsNone(batch.seq_lens_cpu)

        result = GenerationBatchResult(next_token_ids=torch.tensor([12]))
        self.assertTrue(self.manager.finish_forward(batch, result))
        self.assertTrue(result.decoupled_draft_gpu_managed)
        self.assertTrue(self.manager.before_process_batch_result(batch, result))
        self.manager._flush_pending_kv_outcomes()
        self.assertEqual(state.kv_highwater_len, 7)
        self.assertEqual(list(req.output_ids), [10])
        self.assertEqual(
            self.scheduler.token_to_kv_pool_allocator.free.call_args.args[0].tolist(),
            [31],
        )
        self.assertEqual(
            self.scheduler.future_map.output_tokens_buf[req.req_pool_idx].item(), 12
        )

    def test_gpu_state_rebinds_identity_after_filter_and_intervening_batch(self):
        self.manager.checkpoints = MagicMock()
        first, first_state = self._install_request(
            request_id="first", output_tokens=[10]
        )
        second, second_state = self._install_request(
            request_id="second", output_tokens=[20]
        )
        first_state.gpu_seat = 3
        second_state.gpu_seat = 7
        batch = _DraftBatch([first, second])
        batch.defer_decode_kv_binding = True
        self.manager.prepare_batch(batch)
        batch.filter_batch(keep_indices=[1])
        self.manager.prepare_batch(batch)
        self.assertEqual(batch.decoupled_draft_mirror_seats.tolist(), [7])

        other = _DraftBatch([first])
        other.defer_decode_kv_binding = True
        self.manager.prepare_batch(other)
        self.manager.prepare_batch(batch)
        self.assertEqual(batch.decoupled_draft_mirror_seats.tolist(), [7])

    def test_gpu_state_steady_decode_restores_cleared_mamba_routes(self):
        self.manager.checkpoints = MagicMock()
        req0, _ = self._install_request(request_id="req0", output_tokens=[10])
        req1, _ = self._install_request(request_id="req1", output_tokens=[20])
        batch = _DraftBatch([req0, req1])
        batch.defer_decode_kv_binding = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1, 2])
        batch.mamba_cache_src_indices = None
        batch.mamba_cache_dst_indices = None
        self.manager._gpu_identity_reqs = batch.reqs

        self.manager.prepare_batch(batch)

        self.assertEqual(
            batch.mamba_cache_src_indices.data_ptr(),
            self.manager._routing_indices[0, :2].data_ptr(),
        )
        self.assertEqual(
            batch.mamba_cache_dst_indices.data_ptr(),
            self.manager._routing_indices[1, :2].data_ptr(),
        )

    def test_gpu_state_prefill_launch_keeps_allocator_output_locations(self):
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        prefill_out_cache_loc = torch.tensor([37], dtype=torch.int64)
        batch.out_cache_loc = prefill_out_cache_loc

        self.manager._assign_gpu_identity(batch)

        self.assertIs(batch.out_cache_loc, prefill_out_cache_loc)

    def test_dense_prefill_preserves_kv_ownership_without_state_routes(self):
        self.manager._routing_indices = None
        req, state = self._install_request(output_tokens=[10])
        req.mamba_pool_idx = None
        state.kv_highwater_len = 0
        req.kv.kv_allocated_len = 25
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        batch.mamba_cache_src_indices = None
        batch.mamba_cache_dst_indices = None

        self.manager.prepare_batch(batch)

        self.assertEqual(state.kv_highwater_len, 25)
        self.assertEqual(batch.decoupled_draft_mirror_seats.tolist(), [state.gpu_seat])
        self.assertIsNone(batch.mamba_cache_src_indices)
        self.assertIsNone(batch.mamba_cache_dst_indices)
        self.assertIsNone(self.manager.checkpoints)

    def test_gpu_state_mixed_prefill_batch_fails_before_forward(self):
        self.manager.checkpoints = MagicMock()
        self.manager._routing_indices = torch.empty((2, 4), dtype=torch.int64)
        decoupled_req, _ = self._install_request(output_tokens=[10])
        ordinary_req = SimpleNamespace(decoupled_draft_generation=None)
        batch = _DraftBatch(
            [decoupled_req, ordinary_req], forward_mode=ForwardMode.EXTEND
        )

        with self.assertRaisesRegex(RuntimeError, "cannot mix"):
            self.manager.prepare_batch(batch)

    def test_mixed_packing_preserves_prefill_tracking_and_uses_gpu_prefix(self):
        prefill, _ = self._install_request(request_id="prefill", output_tokens=[10])
        decode, _ = self._install_request(request_id="decode", output_tokens=[20])
        model_config = SimpleNamespace(is_encoder_decoder=False)
        prefill_batch = ScheduleBatch(
            reqs=[prefill],
            model_config=model_config,
            req_pool_indices=torch.tensor([1]),
            req_pool_indices_cpu=torch.tensor([1]),
            seq_lens=torch.tensor([3]),
            orig_seq_lens=torch.tensor([3], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([3]),
            out_cache_loc=torch.tensor([40, 41, 42]),
            input_ids=torch.tensor([1, 2, 3]),
            prefix_lens=[0],
            extend_lens=[3],
            extend_num_tokens=3,
            extend_logprob_start_lens=[0],
            sampling_info=MagicMock(),
            mamba_track_indices=torch.tensor([9]),
            mamba_track_mask=torch.tensor([True]),
            mamba_track_seqlens=torch.tensor([3]),
        )
        decode_batch = ScheduleBatch(
            reqs=[decode],
            model_config=model_config,
            req_pool_indices=torch.tensor([2]),
            req_pool_indices_cpu=torch.tensor([2]),
            seq_lens=torch.tensor([99]),
            orig_seq_lens=torch.tensor([99], dtype=torch.int32),
            seq_lens_cpu=None,
            out_cache_loc=torch.tensor([43]),
            input_ids=None,
            sampling_info=MagicMock(),
            defer_decode_kv_binding=True,
        )
        prefill_batch.mix_with_running(decode_batch)
        self.assertEqual(prefill_batch.decoupled_draft_num_prefill_reqs, 1)
        self.assertEqual(prefill_batch.decoupled_draft_num_prefill_tokens, 3)
        self.assertEqual(prefill_batch.extend_lens, [3, 1])
        self.assertEqual(prefill_batch.prefix_lens, [0, 0])
        self.assertEqual(prefill_batch.out_cache_loc.tolist(), [40, 41, 42, 43])
        self.assertEqual(prefill_batch.mamba_track_mask.tolist(), [True, False])
        self.assertEqual(prefill_batch.mamba_track_indices.tolist(), [9, 0])
        self.assertIsNone(prefill_batch.seq_lens_cpu)
        self.assertTrue(prefill_batch.defer_decode_kv_binding)
        decode._refresh_fill_ids.assert_not_called()

    def test_radix_prefill_keeps_active_state_and_evicts_for_ring(self):
        from sglang.srt.speculative.decoupled_draft_checkpoint import (
            DecoupledDraftMambaCheckpointStore,
        )

        req, state = self._install_request(output_tokens=[10])
        state.gpu_checkpoint_slots_initialized = False
        allocator = MagicMock()
        allocator.available_size.return_value = 0
        allocator.schedulable_available_size.return_value = 0
        allocator.alloc.return_value = torch.arange(3, 9, dtype=torch.int64)
        pool = self.scheduler.req_to_token_pool
        pool.mamba_pool = SimpleNamespace(replayssm_write_pos=None)
        pool.mamba_allocator = allocator
        self.manager.checkpoints = DecoupledDraftMambaCheckpointStore(
            pool, max_draft_tokens=3
        )
        self.manager._gpu_checkpoint_slots = torch.full((16, 7), -1, dtype=torch.int64)
        self.scheduler.tree_cache.evict = MagicMock()
        batch = ScheduleBatch(reqs=[req], forward_mode=ForwardMode.EXTEND)
        tracking = torch.tensor([True])
        batch.mamba_track_mask = tracking

        self.manager.prepare_batch(batch)
        self.manager.prepare_batch(batch)

        allocator.alloc.assert_called_once_with(6)
        self.scheduler.tree_cache.evict.assert_called_once()
        self.assertEqual(self.scheduler.tree_cache.evict.call_args.args[0].mamba_num, 6)
        self.assertEqual(batch.mamba_cache_src_indices.tolist(), [2])
        self.assertEqual(batch.mamba_cache_dst_indices.tolist(), [2])
        self.assertEqual(self.manager._gpu_checkpoint_slots[1, 4].item(), 2)
        self.assertIs(batch.mamba_track_mask, tracking)

    def test_mixed_transaction_uses_distinct_request_and_token_boundaries(self):
        prefill, prefill_state = self._install_request(
            request_id="prefill", output_tokens=[10]
        )
        middle, _ = self._install_request(request_id="middle", output_tokens=[])
        decode, decode_state = self._install_request(
            request_id="decode", output_tokens=[20]
        )
        middle.inflight_middle_chunks = 1
        decode.req_pool_idx = 3
        batch = ScheduleBatch(
            reqs=[prefill, middle, decode], forward_mode=ForwardMode.MIXED
        )
        batch.decoupled_draft_num_prefill_reqs = 2
        batch.decoupled_draft_num_prefill_tokens = 5
        batch.defer_decode_kv_binding = True
        batch.chunked_req = middle
        batch.contains_last_prefill_chunk = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1, 2, 3])
        batch.decoupled_draft_request_epochs = torch.tensor([0, 0, 0])
        batch.req_pool_indices = torch.tensor([1, 2, 3])
        batch.req_to_token_pool = self.scheduler.req_to_token_pool
        batch.input_ids = torch.tensor([1, 2, 3, 4, 5, 20])
        batch.seq_lens = torch.tensor([4, 6, 100])
        batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
        batch.out_cache_loc = torch.tensor([40, 41, 42, 43, 44, 45])
        gpu = self.data_plane.gpu_tail_buffer

        def prepare(seats, epochs, req_indices, locations, *args, **kw):
            self.assertEqual(seats.tolist(), [3])
            self.assertEqual(locations.tolist(), [45])
            kw["resolved_input_ids"].fill_(99)
            kw["resolved_seq_lens"].fill_(8)
            kw["resolved_orig_seq_lens"].fill_(8)
            kw["captured_state_positions"].fill_(7)
            kw["old_cache_locs"].fill_(31)

        def finish(seats, epochs, req_indices, locations, samples, *args, **kw):
            self.assertEqual(samples.tolist(), [23])
            self.assertEqual(kw["resolved_input_tokens"].tolist(), [99])
            kw["kv_outcomes"].copy_(torch.tensor([[1, 7, 31]]))
            kw["future_output_tokens"][3] = 77

        def append(seats, epochs, samples, *, accept_out):
            self.assertEqual(seats.tolist(), [1, -1])
            self.assertEqual(samples.tolist(), [11, 12])
            accept_out.copy_(seats >= 0)

        gpu.prepare_decode.side_effect = prepare
        gpu.finish_decode.side_effect = finish
        gpu.append_prefill_sample.side_effect = append
        self.manager.prepare_forward(batch)
        self.assertEqual(batch.input_ids.tolist(), [1, 2, 3, 4, 5, 99])
        self.assertEqual(batch.seq_lens.tolist(), [4, 6, 8])
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([11, 12, 23]), copy_done=MagicMock()
        )
        self.assertTrue(self.manager.finish_forward(batch, result))
        self.assertFalse(result.decoupled_draft_gpu_managed)
        self.assertFalse(self.manager.before_process_batch_result(batch, result))
        self.manager._flush_pending_kv_outcomes()
        self.assertEqual(decode_state.kv_highwater_len, 8)
        self.assertEqual(prefill_state.kv_highwater_len, 6)
        self.assertEqual(list(decode.output_ids), [20])
        self.assertEqual(self.scheduler.future_map.output_tokens_buf[3].item(), 77)
        self.assertEqual(
            self.scheduler.future_map.stash.call_args.args[0].tolist(), [1, 2]
        )
        self.assertEqual(
            self.scheduler.token_to_kv_pool_allocator.free.call_args.args[0].tolist(),
            [31],
        )
        self.assertFalse(getattr(middle, "is_retracted", False))

    def test_serial_decode_retires_transaction_before_result_processing(self):
        req, _ = self._install_request(output_tokens=[10])
        ready = MagicMock()
        result = GenerationBatchResult(
            decoupled_draft_gpu_managed=True,
            decoupled_draft_kv_outcomes=torch.tensor([[1, 7, -1]]),
            decoupled_draft_kv_outcomes_ready=ready,
        )
        self.assertTrue(
            self.manager.before_process_batch_result(_DraftBatch([req]), result)
        )
        self.assertEqual(ready.synchronize.call_count, int(not self.enable_overlap))

    def test_prefill_result_keeps_row_identity_until_next_control_boundary(self):
        first, _ = self._install_request(request_id="first", output_tokens=[10])
        second, _ = self._install_request(request_id="second", output_tokens=[20])
        batch = _DraftBatch([first, second], forward_mode=ForwardMode.EXTEND)
        self.scheduler.cur_batch_for_debug = batch
        self.manager._gpu_lifecycle_poll_count = 15
        self.data_plane.pending_control_count.return_value = 1
        self.data_plane.collect_lifecycle_controls.return_value = ReadyDraftControls(
            close_keys={DraftReqKey(0, "first")}
        )
        result = GenerationBatchResult(
            copy_done=None,
            decoupled_draft_candidate_committed=torch.tensor([False, True]),
        )
        self.assertFalse(self.manager.before_process_batch_result(batch, result))
        self.assertEqual(batch.reqs, [first, second])
        self.assertTrue(first.is_retracted)
        self.assertFalse(getattr(second, "is_retracted", False))
        self.assertTrue(self.manager.has_pending_work())
        self.data_plane.collect_lifecycle_controls.assert_not_called()
        with patch(f"{_DRAFT_MODULE}.release_kv_cache"):
            self.manager.process_pending_controls(force_lifecycle=True)
        self.assertEqual(batch.reqs, [second])


class TestDecoupledDraftManagerOverlap(TestDecoupledDraftManager):
    enable_overlap = True


if __name__ == "__main__":
    unittest.main()

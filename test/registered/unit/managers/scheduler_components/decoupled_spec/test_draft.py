"""CPU-only contract tests for the decoupled drafter scheduler component."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.decoupled_spec.draft import (  # noqa: E402
    DecoupledDraftManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.decoupled_draft_checkpoint import (  # noqa: E402
    DraftRequestGeneration,
)
from sglang.srt.speculative.decoupled_spec_io import (  # noqa: E402
    DecoupledSpecIpcConfig,
    DraftCommitAction,
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

    def filter_batch(self, *, keep_indices) -> None:
        self.reqs = [self.reqs[index] for index in keep_indices]

    def is_empty(self) -> bool:
        return not self.reqs

    def merge_batch(self, other) -> None:
        self.reqs.extend(other.reqs)


class TestDecoupledDraftManager(CustomTestCase):
    def setUp(self) -> None:
        observability_patcher = patch(
            f"{_DRAFT_MODULE}.get_observability",
            return_value=SimpleNamespace(enable_metrics=False),
        )
        self.addCleanup(observability_patcher.stop)
        observability_patcher.start()
        data_plane_patcher = patch(
            f"{_DRAFT_MODULE}.create_drafter_decoupled_spec_data_plane",
            autospec=True,
        )
        self.addCleanup(data_plane_patcher.stop)
        data_plane_factory = data_plane_patcher.start()
        self.data_plane = data_plane_factory.return_value
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls()
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
            enable_overlap=False,
            server_args=SimpleNamespace(
                speculative_num_steps=3,
                enable_metrics=False,
            ),
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
                free=MagicMock(),
                alloc=MagicMock(),
                available_size=MagicMock(return_value=128),
            ),
            schedule_stream=SimpleNamespace(wait_stream=MagicMock()),
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
        result = SimpleNamespace()

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
            mamba_pool_idx=2,
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
        state = SimpleNamespace(
            key=key,
            req=req,
            src_verifier_rank=src_verifier_rank,
            committed_len=committed_len,
            published_len=len(output_tokens),
            is_sleeping=False,
            gpu_seat=1,
            gpu_checkpoint_slots_initialized=True,
            kv_highwater_len=6,
            gpu_ahead_len=len(output_tokens) - committed_len,
        )
        self.manager._requests[(src_verifier_rank, request_id)] = state
        return req, state

    def _enable_gpu_overlap(self) -> None:
        """Switch this CPU-only fixture onto the host-facing overlap hooks."""

        self.manager.gpu_overlap = True
        self.manager.schedule_ahead_limit = self.manager.ahead_window - 1
        shape = (16,)
        self.manager._gpu_batch_seats = torch.empty(shape, dtype=torch.int64)
        self.manager._gpu_batch_epochs = torch.empty(shape, dtype=torch.int64)
        self.manager._gpu_batch_seats_cpu = torch.empty(shape, dtype=torch.int64)
        self.manager._gpu_batch_epochs_cpu = torch.empty(shape, dtype=torch.int64)
        self.manager._gpu_identity_done_event = MagicMock()
        self.manager._gpu_identity_in_use = False
        self.manager._gpu_identity = None
        self.manager._gpu_lifecycle_poll_count = 0
        self.manager._gpu_lifecycle_poll_interval = 16
        self.manager._routing_indices = torch.empty((2, 16), dtype=torch.int64)
        self.manager._routing_cpu = torch.empty((2, 16), dtype=torch.int64)
        self.manager._routing_cpu_array = self.manager._routing_cpu.numpy()
        self.scheduler.schedule_stream.wait_event = MagicMock()
        self.data_plane.collect_lifecycle_controls.return_value = ReadyDraftControls()
        self.data_plane.pending_control_count.return_value = 0

    @staticmethod
    def _action(
        request_id="req",
        *,
        expected_output_len,
        pre_verify_committed_len,
        new_committed_len,
        rewrite_position=-1,
        rewrite_token=-1,
    ):
        return DraftCommitAction(
            draft_key=DraftReqKey(0, request_id),
            dst_drafter_rank=0,
            expected_output_len=expected_output_len,
            pre_verify_committed_len=pre_verify_committed_len,
            new_committed_len=new_committed_len,
            rewrite_position=rewrite_position,
            rewrite_token=rewrite_token,
            echo_position=new_committed_len - 1,
            echo_token=rewrite_token if rewrite_position >= 0 else 0,
        )

    def test_published_tail_targets_the_true_sparse_verifier_rank(self):
        req, _ = self._install_request(
            output_tokens=[10],
            src_verifier_rank=7,
        )
        batch = _DraftBatch([req])
        self.assertFalse(
            self.manager.before_process_batch_result(batch, SimpleNamespace())
        )
        req.output_ids.append(11)

        self.manager.after_process_batch_result(batch, SimpleNamespace())

        output = self.data_plane.publish_tails.call_args.args[0].outputs[0]
        self.assertEqual(output.src_drafter_rank, 0)
        self.assertEqual(output.dst_verifier_rank, 7)

    def test_decode_publication_coalesces_contiguous_tokens(self):
        req, state = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        req.output_ids.extend([11, 12, 13])
        state.published_len = 1

        self.manager.after_process_batch_result(batch, SimpleNamespace())

        self.data_plane.publish_tails.assert_called_once()
        outputs = self.data_plane.publish_tails.call_args.args[0].outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].base_committed_len, 1)
        self.assertEqual(outputs[0].start_token_pos, 1)
        self.assertEqual(outputs[0].tokens, (11, 12, 13))
        self.assertFalse(outputs[0].is_commit_echo)
        self.assertEqual(state.published_len, 4)

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
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls(
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
            request_id="behind", output_tokens=range(7)
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

    def test_decode_routing_prunes_consumed_committed_state_before_ring_wrap(self):
        req, state = self._install_request(
            output_tokens=range(7),
            committed_len=1,
        )
        self.manager.checkpoints = MagicMock()
        self.manager.checkpoints.prepare_decode_route.return_value = (21, 22)
        self.manager._routing_indices = torch.empty((2, 4), dtype=torch.int64)
        self.manager._routing_cpu = torch.empty((2, 4), dtype=torch.int64)
        self.manager._routing_cpu_array = self.manager._routing_cpu.numpy()
        batch = _DraftBatch([req])

        self.manager.prepare_batch(batch)

        # prompt=3 and committed_len=1: position 3 represented the state before
        # the now-consumed committed tail and must be free before K=3 wraps.
        self.manager.checkpoints.prune.assert_called_once_with(
            state.key,
            min_position=4,
            max_position=9,
        )
        self.manager.checkpoints.prepare_decode_route.assert_called_once_with(
            state.key,
            position=9,
        )
        self.assertEqual(batch.mamba_cache_src_indices.tolist(), [21])
        self.assertEqual(batch.mamba_cache_dst_indices.tolist(), [22])

    def test_decode_routing_keeps_unconsumed_committed_tail_state(self):
        req, state = self._install_request(
            output_tokens=[10],
            committed_len=1,
        )
        self.manager.checkpoints = MagicMock()
        self.manager.checkpoints.prepare_decode_route.return_value = (21, 22)
        self.manager._routing_indices = torch.empty((2, 4), dtype=torch.int64)
        self.manager._routing_cpu = torch.empty((2, 4), dtype=torch.int64)
        self.manager._routing_cpu_array = self.manager._routing_cpu.numpy()

        self.manager.prepare_batch(_DraftBatch([req]))

        self.manager.checkpoints.prune.assert_called_once_with(
            state.key,
            min_position=3,
            max_position=3,
        )

    def test_full_match_advances_cursor_without_rewrite(self):
        req, state = self._install_request(output_tokens=[10, 11, 12, 13])
        self.manager.checkpoints = MagicMock()
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls(
            commit_actions=[
                self._action(
                    expected_output_len=4,
                    pre_verify_committed_len=1,
                    new_committed_len=3,
                )
            ]
        )

        self.manager.process_pending_controls()

        self.assertEqual(state.committed_len, 3)
        self.assertEqual(state.published_len, 4)
        self.assertEqual(list(req.output_ids), [10, 11, 12, 13])
        self.manager.checkpoints.rewind_for_rewrite.assert_not_called()
        self.manager.checkpoints.prune.assert_called_once_with(
            state.key,
            min_position=5,
        )
        self.data_plane.publish_tails.assert_called_once()
        outputs = self.data_plane.publish_tails.call_args.args[0].outputs
        self.assertEqual(len(outputs), 2)
        echo, retained = outputs
        self.assertEqual(echo.base_committed_len, 3)
        self.assertEqual(echo.start_token_pos, 2)
        self.assertEqual(echo.tokens, (0,))
        self.assertTrue(echo.is_commit_echo)
        self.assertEqual(retained.base_committed_len, 3)
        self.assertEqual(retained.start_token_pos, 3)
        self.assertEqual(retained.tokens, (13,))
        self.assertFalse(retained.is_commit_echo)
        self.assertEqual(self.data_plane.publish_tails.call_args.kwargs, {})

        # The retained suffix was published with the ACK and must not replay
        # on the next scheduler result.
        self.data_plane.publish_tails.reset_mock()
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls()
        self.manager.after_process_batch_result(_DraftBatch([req]), SimpleNamespace())
        self.data_plane.publish_tails.assert_not_called()

    def test_ready_segment_wakes_only_its_sleeping_request(self):
        running_req, _ = self._install_request(
            request_id="running", output_tokens=range(7)
        )
        sleeping_req, sleeping_state = self._install_request(
            request_id="sleeping", output_tokens=range(8)
        )
        sleeping_req.kv.kv_allocated_len = 10
        sleeping_req.kv_committed_len = 10
        running_batch = _DraftBatch([running_req, sleeping_req])
        self.scheduler.running_batch = running_batch
        self.manager.sleep_overrun_requests(running_batch)
        self.assertEqual(running_batch.reqs, [running_req])
        self.manager.checkpoints = MagicMock()
        wake_batch = _DraftBatch([sleeping_req])
        self.manager._build_draft_decode_batch = MagicMock(return_value=wake_batch)
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls(
            commit_actions=[
                self._action(
                    "sleeping",
                    expected_output_len=8,
                    pre_verify_committed_len=1,
                    new_committed_len=2,
                )
            ]
        )

        self.manager.process_pending_controls()

        self.assertEqual(sleeping_state.committed_len, 2)
        self.assertFalse(sleeping_state.is_sleeping)
        self.assertNotIn(sleeping_state.key, self.manager._sleeping_requests)
        self.assertEqual(running_batch.reqs, [running_req, sleeping_req])
        self.manager._build_draft_decode_batch.assert_called_once_with([sleeping_req])

    def test_short_tail_does_not_block_ready_commit_for_another_request(self):
        blocked_req, blocked_state = self._install_request(
            request_id="blocked",
            output_tokens=[10],
        )
        ready_req, ready_state = self._install_request(
            request_id="ready",
            output_tokens=[20, 21],
        )
        self.manager.checkpoints = MagicMock()
        ready_action = self._action(
            "ready",
            expected_output_len=2,
            pre_verify_committed_len=1,
            new_committed_len=2,
        )

        def collect_ready(model_ready):
            self.assertFalse(model_ready(DraftReqKey(0, "blocked"), 2))
            self.assertTrue(model_ready(DraftReqKey(0, "ready"), 2))
            return ReadyDraftControls(commit_actions=[ready_action])

        self.data_plane.collect_ready_actions.side_effect = collect_ready

        self.manager.process_pending_controls()

        self.assertEqual(blocked_state.committed_len, 1)
        self.assertEqual(ready_state.committed_len, 2)

        blocked_req.output_ids.append(11)
        blocked_state.published_len = len(blocked_req.output_ids)
        self.data_plane.collect_ready_actions.side_effect = None
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls(
            commit_actions=[
                self._action(
                    "blocked",
                    expected_output_len=2,
                    pre_verify_committed_len=1,
                    new_committed_len=2,
                )
            ]
        )
        self.manager.process_pending_controls()

        self.assertEqual(blocked_state.committed_len, 2)
        self.assertEqual(list(ready_req.output_ids), [20, 21])

    def test_retract_releases_invalidated_checkpoint_ring(self):
        req, state = self._install_request(output_tokens=[10, 11, 12])
        self.manager.checkpoints = MagicMock()

        self.manager.retract_request(req)

        self.manager.checkpoints.release.assert_called_once_with(state.key)
        self.assertEqual(list(req.output_ids), [10])
        self.assertEqual(state.published_len, 1)
        req._refresh_fill_ids.assert_called_once_with()

    def test_last_oom_mirror_is_retried_instead_of_aborted(self):
        retracted_req, _ = self._install_request(
            request_id="retracted", output_tokens=[10, 11]
        )
        last_oom_req, _ = self._install_request(
            request_id="last-oom", output_tokens=[20, 21]
        )
        last_oom_req.to_finish = object()
        self.manager.checkpoints = MagicMock()

        retry_reqs, abort_reqs = self.manager.handle_retracted_requests(
            [retracted_req], [last_oom_req]
        )

        self.assertEqual(retry_reqs, [retracted_req, last_oom_req])
        self.assertEqual(abort_reqs, [])
        self.assertIsNone(last_oom_req.to_finish)
        self.assertEqual(list(last_oom_req.output_ids), [20])

    def test_pause_retract_folds_sleeping_request_and_resets_each_once(self):
        running_req, running_state = self._install_request(
            request_id="running", output_tokens=[10, 11, 12]
        )
        sleeping_req, sleeping_state = self._install_request(
            request_id="sleeping", output_tokens=[20, 21, 22]
        )
        sleeping_state.is_sleeping = True
        self.manager._sleeping_requests[sleeping_state.key] = sleeping_req
        self.manager.checkpoints = MagicMock()

        retract_reqs = self.manager.prepare_pause_retract([running_req, running_req])

        self.assertEqual(retract_reqs, [running_req, sleeping_req])
        self.assertEqual(list(running_req.output_ids), [10])
        self.assertEqual(list(sleeping_req.output_ids), [20])
        self.assertEqual(running_state.published_len, 1)
        self.assertEqual(sleeping_state.published_len, 1)
        self.assertFalse(sleeping_state.is_sleeping)
        self.assertEqual(self.manager._sleeping_requests, {})
        self.manager.checkpoints.release.assert_has_calls(
            [call(running_state.key), call(sleeping_state.key)]
        )
        self.assertEqual(self.manager.checkpoints.release.call_count, 2)

    def test_bonus_rewrite_restores_then_truncates_before_publishing_echo(self):
        req, state = self._install_request(output_tokens=[10, 11, 99, 100])
        events = []
        self.manager.checkpoints = MagicMock()
        self.manager.checkpoints.rewind_for_rewrite.side_effect = (
            lambda *args, **kwargs: events.append("restore")
        )
        self.manager._truncate_kv = MagicMock(
            side_effect=lambda *args, **kwargs: events.append("truncate")
        )
        req._refresh_fill_ids.side_effect = lambda: events.append("refresh_fill")
        self.manager._refresh_active_batches = MagicMock(
            side_effect=lambda *args, **kwargs: events.append("refresh_batch")
        )
        self.data_plane.publish_tails.side_effect = (
            lambda *args, **kwargs: events.append("publish_echo")
        )
        self.manager.checkpoints.prune.side_effect = (
            lambda *args, **kwargs: events.append("prune")
        )
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls(
            commit_actions=[
                self._action(
                    expected_output_len=4,
                    pre_verify_committed_len=1,
                    new_committed_len=3,
                    rewrite_position=2,
                    rewrite_token=22,
                )
            ]
        )

        self.manager.process_pending_controls()

        self.assertEqual(
            events,
            [
                "restore",
                "truncate",
                "refresh_fill",
                "prune",
                "refresh_batch",
                "publish_echo",
            ],
        )
        self.assertEqual(list(req.output_ids), [10, 11, 22])
        self.assertEqual(state.committed_len, 3)
        self.assertEqual(state.published_len, 3)
        echo = self.data_plane.publish_tails.call_args.args[0].outputs[0]
        self.assertEqual(echo.base_committed_len, 3)
        self.assertEqual(echo.start_token_pos, 2)
        self.assertEqual(echo.tokens, (22,))
        self.assertTrue(echo.is_commit_echo)
        self.assertEqual(
            len(self.data_plane.publish_tails.call_args.args[0].outputs), 1
        )

    def test_rewrite_rebases_publication_cursor_before_next_result(self):
        req, state = self._install_request(output_tokens=[10, 11, 99, 100])
        batch = _DraftBatch([req])
        # Freeze the old batch boundary before a verifier rewrite. Publication
        # must use request-owned state rather than this stale round-local length.
        self.manager.before_process_batch_result(batch, SimpleNamespace())
        self.manager.checkpoints = MagicMock()
        self.manager._truncate_kv = MagicMock()
        self.manager._refresh_active_batches = MagicMock()
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls(
            commit_actions=[
                self._action(
                    expected_output_len=4,
                    pre_verify_committed_len=1,
                    new_committed_len=3,
                    rewrite_position=2,
                    rewrite_token=22,
                )
            ]
        )

        self.manager.process_pending_controls()
        req.output_ids.append(33)
        self.data_plane.publish_tails.reset_mock()
        self.data_plane.collect_ready_actions.return_value = ReadyDraftControls()

        self.manager.after_process_batch_result(batch, SimpleNamespace())

        outputs = self.data_plane.publish_tails.call_args.args[0].outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].base_committed_len, 3)
        self.assertEqual(outputs[0].start_token_pos, 3)
        self.assertEqual(outputs[0].tokens, (33,))
        self.assertEqual(state.published_len, 4)

    def test_gpu_overlap_uses_lifecycle_with_compact_gpu_credit(self):
        self._enable_gpu_overlap()
        req, state = self._install_request(output_tokens=range(7))
        state.published_len = 1
        self.manager._gpu_lifecycle_poll_count = 15
        self.data_plane.pending_control_count.return_value = 1
        batch = _DraftBatch([req])
        routed_capture = MagicMock()
        indexer_capture = MagicMock()
        result = SimpleNamespace(
            copy_done=None,
            next_token_ids=torch.tensor([12], dtype=torch.int64),
            routed_experts_output=routed_capture,
            indexer_topk_output=indexer_capture,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[0, -1, -1]]),
            can_run_cuda_graph=True,
        )
        self.data_plane.collect_ready_actions.side_effect = AssertionError(
            "GPU overlap must not materialize verifier commits on CPU"
        )

        self.manager.process_pending_controls()
        self.scheduler.metrics_collector = MagicMock()
        with patch(
            f"{_DRAFT_MODULE}.get_observability",
            return_value=SimpleNamespace(enable_metrics=True),
        ):
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
        self.data_plane.collect_ready_actions.assert_not_called()
        # Native snapshot egress owns ACK/tail publication in GPU mode. The
        # CPU Req transcript must not replay even an apparently unpublished tail.
        self.data_plane.publish_tails.assert_not_called()
        self.assertEqual(state.committed_len, 1)
        self.assertEqual(state.published_len, 1)
        self.assertEqual(list(req.output_ids), list(range(7)))

    def test_gpu_overlap_parks_a_row_without_runnable_credit(self):
        self._enable_gpu_overlap()
        req, state = self._install_request(output_tokens=range(32))
        state.gpu_ahead_len = self.manager.schedule_ahead_limit
        batch = _DraftBatch([req])

        self.assertIs(self.manager.sleep_overrun_requests(batch), batch)
        self.assertEqual(batch.reqs, [])
        self.assertTrue(state.is_sleeping)
        self.assertIs(self.manager._sleeping_requests[state.key], req)

    def test_gpu_overlap_accepts_forced_replay_credit_without_token_shadow(self):
        self._enable_gpu_overlap()
        _, state = self._install_request(output_tokens=[10])
        self.data_plane.drain_gpu_progress.return_value = [
            ("req", 0, 0, 3, 3, 0)
        ]

        self.manager.process_pending_controls()

        self.assertEqual(state.gpu_ahead_len, 0)

    def test_gpu_overlap_discards_queued_result_after_close_and_same_id_reopen(self):
        self._enable_gpu_overlap()
        old_req, _ = self._install_request(output_tokens=[10, 11])
        stale_batch = _DraftBatch([old_req])
        stale_result = SimpleNamespace(
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

    def test_gpu_overlap_final_prefill_rejection_is_a_stale_result_not_assertion(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        batch.contains_last_prefill_chunk = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1], dtype=torch.int64)
        batch.decoupled_draft_request_epochs = torch.tensor([0], dtype=torch.int64)
        batch.req_pool_indices = torch.tensor([1], dtype=torch.int64)
        batch.seq_lens = torch.tensor([5], dtype=torch.int64)
        batch.out_cache_loc = torch.tensor([17], dtype=torch.int64)
        batch.decoupled_draft_captured_state_positions = None
        result = SimpleNamespace(
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

    def test_gpu_overlap_accepted_prefill_keeps_generic_result_processing(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        result = SimpleNamespace(
            copy_done=None,
            next_token_ids=torch.tensor([77], dtype=torch.int64),
            decoupled_draft_gpu_managed=False,
            decoupled_draft_candidate_committed=torch.tensor([True]),
        )

        self.assertFalse(self.manager.before_process_batch_result(batch, result))

    def test_gpu_overlap_candidate_ownership_advances_exact_kv_highwater(self):
        self._enable_gpu_overlap()
        owned_req, owned_state = self._install_request(
            request_id="owned",
            output_tokens=[10],
        )
        rejected_req, rejected_state = self._install_request(
            request_id="rejected",
            output_tokens=[20],
        )
        batch = _DraftBatch([owned_req, rejected_req])
        result = SimpleNamespace(
            copy_done=None,
            next_token_ids=torch.tensor([11, 21], dtype=torch.int64),
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor(
                [[1, 9, -1], [0, -1, -1]]
            ),
        )

        self.assertTrue(self.manager.before_process_batch_result(batch, result))
        self.assertEqual(len(self.manager._pending_kv_outcomes), 1)
        self.manager._flush_pending_kv_outcomes()

        self.assertEqual(owned_state.kv_highwater_len, 10)
        self.assertEqual(rejected_state.kv_highwater_len, 6)

    def test_gpu_overlap_decode_result_defers_control_poll_to_next_loop(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        result = SimpleNamespace(
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

    def test_gpu_overlap_reclaims_only_finish_reported_kv_locations(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        result = SimpleNamespace(
            copy_done=None,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor([[1, 6, 123]]),
        )

        self.assertTrue(self.manager.before_process_batch_result(batch, result))
        self.manager._flush_pending_kv_outcomes()

        freed = self.scheduler.token_to_kv_pool_allocator.free.call_args.args[0]
        self.assertEqual(freed.tolist(), [123])

    def test_gpu_overlap_reclaim_waits_for_copy_without_blocking_decode(self):
        self._enable_gpu_overlap()
        req, state = self._install_request(output_tokens=[10])
        done = MagicMock()
        done.query.return_value = False
        self.manager._inflight_kv_outcomes = (
            [req], torch.tensor([[1, 9, 123]]), done, []
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

    def test_gpu_overlap_close_drains_inflight_kv_before_release(self):
        self._enable_gpu_overlap()
        req, state = self._install_request(output_tokens=[10])
        self.manager.checkpoints = MagicMock()
        done = MagicMock()
        self.manager._inflight_kv_outcomes = (
            [req], torch.tensor([[1, 34, 123]]), done, []
        )
        with patch(f"{_DRAFT_MODULE}.release_kv_cache") as release:
            self.manager._close_request_key(DraftReqKey(0, "req"))
        done.synchronize.assert_called_once()
        self.assertEqual(req.kv_committed_len, 35)
        self.assertIsNone(self.manager._inflight_kv_outcomes)
        release.assert_called_once()

    def test_gpu_overlap_close_releases_exact_kv_highwater(self):
        self._enable_gpu_overlap()
        req, state = self._install_request(output_tokens=[10])
        state.kv_highwater_len = 35
        self.manager.checkpoints = MagicMock()

        with patch(f"{_DRAFT_MODULE}.release_kv_cache") as release:
            self.manager._close_request_key(DraftReqKey(0, "req"))

        self.scheduler.schedule_stream.wait_stream.assert_called_once_with(
            self.scheduler.forward_stream
        )
        self.assertEqual(req.kv_committed_len, 35)
        self.assertEqual(req.kv.kv_allocated_len, 35)
        self.manager.checkpoints.release.assert_called_once_with(state.key)
        release.assert_called_once_with(
            req,
            self.scheduler.tree_cache,
            is_insert=False,
        )

    def test_gpu_overlap_close_accounts_for_queued_kv_outcome(self):
        self._enable_gpu_overlap()
        req, state = self._install_request(output_tokens=[10])
        live_req, live_state = self._install_request(
            request_id="live", output_tokens=[20]
        )
        copy_done = MagicMock()
        queued_result = SimpleNamespace(
            copy_done=copy_done,
            decoupled_draft_gpu_managed=True,
            decoupled_draft_candidate_committed=None,
            decoupled_draft_kv_outcomes=torch.tensor(
                [[1, 9, -1], [1, 99, -1]]
            ),
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

    def test_gpu_overlap_mixed_decode_batch_fails_before_allocation(self):
        self._enable_gpu_overlap()
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

    def test_gpu_overlap_retraction_fails_before_generic_resource_mutation(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])

        with self.assertRaisesRegex(RuntimeError, "cannot retract"):
            self.manager.before_decode_retraction(_DraftBatch([req]))

    def test_gpu_overlap_identity_table_fences_only_topology_changes(self):
        self._enable_gpu_overlap()
        self.manager.checkpoints = SimpleNamespace(capacity=7)
        self.manager._routing_indices = torch.empty((2, 4), dtype=torch.int64)
        self.manager._gpu_batch_seats = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_batch_epochs = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_batch_seats_cpu = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_batch_epochs_cpu = torch.empty((4,), dtype=torch.int64)
        self.manager._gpu_identity_done_event = MagicMock()
        self.manager._gpu_identity_in_use = False
        self.scheduler.schedule_stream.wait_event = MagicMock()

        req, state = self._install_request(
            request_id="ring", output_tokens=[10]
        )
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
        self.assertEqual(
            int(batches[-1].decoupled_draft_mirror_seats[0]), 10
        )
        self.assertEqual(
            int(batches[-1].decoupled_draft_request_epochs[0]), 20
        )

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
        )

    def test_gpu_overlap_bs1_uses_zero_copy_future_token_view(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req])
        batch.defer_decode_kv_binding = True
        batch.decoupled_draft_mirror_seats = torch.tensor([1])
        batch.input_ids = None
        self.manager._gpu_identity_reqs = batch.reqs

        self.manager.prepare_batch(batch)

        expected = self.scheduler.future_map.output_tokens_buf[1:2]
        self.assertEqual(batch.input_ids.data_ptr(), expected.data_ptr())

    def test_gpu_overlap_rebinds_identity_after_filter_and_intervening_batch(self):
        self._enable_gpu_overlap()
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

    def test_gpu_overlap_steady_decode_restores_cleared_mamba_routes(self):
        self._enable_gpu_overlap()
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

    def test_gpu_overlap_prefill_launch_keeps_allocator_output_locations(self):
        self._enable_gpu_overlap()
        req, _ = self._install_request(output_tokens=[10])
        batch = _DraftBatch([req], forward_mode=ForwardMode.EXTEND)
        prefill_out_cache_loc = torch.tensor([37], dtype=torch.int64)
        batch.out_cache_loc = prefill_out_cache_loc

        self.manager._assign_gpu_identity(batch)

        self.assertIs(batch.out_cache_loc, prefill_out_cache_loc)

    def test_dense_overlap_prefill_preserves_kv_ownership_without_state_routes(self):
        self._enable_gpu_overlap()
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

    def test_dense_nonoverlap_rewrite_preserves_prefix_and_releases_suffix_kv(self):
        req, state = self._install_request(output_tokens=[10, 11, 12, 13])
        self.manager._truncate_kv = MagicMock()
        action = self._action(
            expected_output_len=4,
            pre_verify_committed_len=1,
            new_committed_len=3,
            rewrite_position=2,
            rewrite_token=99,
        )

        rewritten, outputs = self.manager._apply_commit_action(action)

        self.assertIs(rewritten, req)
        self.assertEqual(list(req.output_ids), [10, 11, 99])
        self.manager._truncate_kv.assert_called_once_with(req, 5)
        self.assertEqual(state.committed_len, 3)
        self.assertEqual(outputs[0].tokens, (99,))

    def test_gpu_overlap_mixed_prefill_batch_fails_before_forward(self):
        self._enable_gpu_overlap()
        self.manager.checkpoints = MagicMock()
        self.manager._routing_indices = torch.empty((2, 4), dtype=torch.int64)
        decoupled_req, _ = self._install_request(output_tokens=[10])
        ordinary_req = SimpleNamespace(decoupled_draft_generation=None)
        batch = _DraftBatch(
            [decoupled_req, ordinary_req], forward_mode=ForwardMode.EXTEND
        )

        with self.assertRaisesRegex(RuntimeError, "cannot mix"):
            self.manager.prepare_batch(batch)


if __name__ == "__main__":
    unittest.main()

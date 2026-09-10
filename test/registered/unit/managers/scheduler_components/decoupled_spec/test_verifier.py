"""CPU-only contract tests for the decoupled verifier scheduler component."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.decoupled_spec.verifier import (  # noqa: E402
    DecoupledVerifyManager,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.decoupled_spec_io import (  # noqa: E402
    DecoupledSpecIpcConfig,
    DecoupledSpecPeerConfig,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_VERIFIER_MODULE = "sglang.srt.managers.scheduler_components.decoupled_spec.verifier"


class _Req:
    def __init__(
        self,
        rid: str,
        *,
        prompt_tokens=(1, 2),
        output_tokens=(),
        retraction_count: int = 0,
        req_pool_idx: int = 0,
    ) -> None:
        self.rid = rid
        self.origin_input_ids = list(prompt_tokens)
        self.output_ids = list(output_tokens)
        self.retraction_count = retraction_count
        self.req_pool_idx = req_pool_idx
        self.sampling_params = SimpleNamespace(max_new_tokens=128)
        self.is_retracted = False
        self._finished = False

    def finished(self) -> bool:
        return self._finished


def _batch(reqs, forward_mode: ForwardMode):
    return SimpleNamespace(
        reqs=list(reqs),
        forward_mode=forward_mode,
        device=torch.device("cpu"),
    )


class TestDecoupledVerifyManager(CustomTestCase):
    def setUp(self) -> None:
        stream_patcher = patch(f"{_VERIFIER_MODULE}.get_stream", autospec=True)
        self.addCleanup(stream_patcher.stop)
        self.get_stream = stream_patcher.start()
        self.landing_stream = object()
        self.get_stream.return_value = self.landing_stream
        data_plane_patcher = patch(
            f"{_VERIFIER_MODULE}.create_verifier_decoupled_spec_data_plane",
            autospec=True,
        )
        self.addCleanup(data_plane_patcher.stop)
        self.data_plane_factory = data_plane_patcher.start()
        self.data_plane = self.data_plane_factory.return_value
        self.gpu_tail_buffer = object()
        self.data_plane.gpu_tail_buffer = self.gpu_tail_buffer
        self.data_plane.take_transport_metrics.return_value = {
            "num_draft_result_frames": 0,
            "num_draft_result_tokens": 0,
        }
        self.verify_worker = MagicMock()
        self.scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0, tp_size=1),
            server_args=SimpleNamespace(
                speculative_num_steps=3,
                speculative_adaptive=False,
            ),
            device=torch.device("cpu"),
            req_to_token_pool=SimpleNamespace(
                size=16, req_to_token=SimpleNamespace(shape=(17, 64))
            ),
            draft_worker=self.verify_worker,
            enable_overlap=True,
        )
        self.config = DecoupledSpecIpcConfig(
            bind_endpoint="ipc:///tmp/unused-verifier",
            connect_endpoints=("ipc:///tmp/unused-drafter",),
            rank=0,
        )
        self.manager = DecoupledVerifyManager(self.scheduler, self.config)
        self.data_plane_factory.assert_called_once_with(
            self.config,
            required_tail_len=0,
            device=torch.device("cpu"),
            num_gpu_seats=17,
            num_draft_tokens=3,
            landing_stream=self.landing_stream,
            mock_profile=False,
        )
        self.get_stream.assert_called_once_with("decoupled_spec_landing")
        self.data_plane.start.assert_called_once_with()
        self.verify_worker.attach_gpu_tail_buffer.assert_called_once_with(
            self.gpu_tail_buffer
        )
        self.data_plane.reset_mock()

    def _replace_manager(self, *, verifier_rank: int, weighted_drafter_ranks):
        self.manager.close()
        self.data_plane.reset_mock()
        self.data_plane_factory.reset_mock()
        self.config = DecoupledSpecIpcConfig(
            bind_endpoint="tcp://verifier:30000",
            connect_endpoints=(),
            rank=verifier_rank,
            peers=tuple(
                DecoupledSpecPeerConfig(
                    rank=drafter_rank,
                    endpoint=f"tcp://drafter-{drafter_rank}:31{drafter_rank:03d}",
                    quota=quota,
                )
                for drafter_rank, quota in weighted_drafter_ranks
            ),
        )
        self.manager = DecoupledVerifyManager(self.scheduler, self.config)
        self.data_plane.reset_mock()

    def _open_rows(self):
        return [
            row
            for call in self.data_plane.open_requests.call_args_list
            for row in call.args[0]
        ]

    def test_decode_reuses_only_an_open_lifecycle_identity(self):
        req_a = _Req("req-a", retraction_count=0)
        req_b = _Req("req-b", retraction_count=2)
        extend_batch = _batch([req_a, req_b], ForwardMode.EXTEND)
        self.manager.prepare_batch(extend_batch)
        self.assertTrue(extend_batch.decoupled_has_new_lifecycle)
        self.verify_worker.capture_expected_request_epochs.assert_called_with(
            extend_batch
        )
        self.data_plane.reset_mock()
        self.verify_worker.capture_expected_request_epochs.reset_mock()
        batch = _batch([req_a, req_b], ForwardMode.DECODE)

        self.manager.prepare_batch(batch)

        self.assertEqual(
            batch.decoupled_launch_mirror_ids,
            ["req-a::draft-epoch::1", "req-b::draft-epoch::2"],
        )
        self.assertFalse(batch.decoupled_has_new_lifecycle)
        self.verify_worker.capture_expected_request_epochs.assert_called_once_with(
            batch
        )
        self.data_plane.assert_not_called()

    def test_sparse_weighted_routing_is_sticky_for_each_lifecycle(self):
        self._replace_manager(
            verifier_rank=5,
            weighted_drafter_ranks=((3, 2), (9, 1)),
        )
        reqs = [_Req(f"req-{index}", req_pool_idx=index) for index in range(6)]

        self.manager.prepare_batch(_batch(reqs, ForwardMode.EXTEND))

        assigned_ranks = [
            self.manager._open_drafter_rank_by_req[req.rid] for req in reqs
        ]
        self.assertEqual(assigned_ranks, [3, 9, 3, 3, 9, 3])
        open_batches = [
            call.args[0] for call in self.data_plane.open_requests.call_args_list
        ]
        self.assertEqual(
            [[row[0].dst_drafter_rank for row in rows] for rows in open_batches],
            [[3, 3, 3, 3], [9, 9]],
        )

        self.data_plane.reset_mock()
        self.manager.prepare_batch(_batch(reqs, ForwardMode.DECODE))
        self.data_plane.open_requests.assert_not_called()
        self.assertEqual(
            [self.manager._open_drafter_rank_by_req[req.rid] for req in reqs],
            assigned_ranks,
        )

    def test_one_result_batch_is_grouped_by_sparse_destination_rank(self):
        self._replace_manager(
            verifier_rank=5,
            weighted_drafter_ranks=((3, 1), (9, 1)),
        )
        req_a = _Req("req-a", output_tokens=(30,), req_pool_idx=0)
        req_b = _Req("req-b", output_tokens=(40,), req_pool_idx=1)
        req_c = _Req("req-c", output_tokens=(50,), req_pool_idx=2)
        self.manager.prepare_batch(_batch([req_a, req_b, req_c], ForwardMode.EXTEND))
        self.assertEqual(
            [
                self.manager._open_drafter_rank_by_req[req.rid]
                for req in (req_a, req_b, req_c)
            ],
            [3, 9, 3],
        )
        self.data_plane.reset_mock()

        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
        )
        decode_batch = _batch([req_a, req_b, req_c], ForwardMode.DECODE)
        self.manager.prepare_batch(decode_batch)
        self.manager.before_process_batch_result(decode_batch, result)
        req_a.output_ids.append(31)
        req_b._finished = True
        req_c.output_ids.append(51)

        self.manager.after_process_batch_result(decode_batch, result)

        control_batches = [
            call.args[0] for call in self.data_plane.submit_control_batch.call_args_list
        ]
        self.assertTrue(
            all(
                call.kwargs == {"apply_local_verify_commits": False}
                for call in self.data_plane.submit_control_batch.call_args_list
            )
        )
        self.assertEqual([batch.dst_drafter_rank for batch in control_batches], [3, 9])
        self.assertEqual(
            [
                message.request_id
                for message in control_batches[0].verify_commit_messages
            ],
            ["req-a::draft-epoch::1", "req-c::draft-epoch::3"],
        )
        self.assertEqual(
            [message.request_id for message in control_batches[1].close_messages],
            ["req-b::draft-epoch::2"],
        )

    def test_delayed_result_cannot_cross_a_retracted_request_route(self):
        self._replace_manager(
            verifier_rank=5,
            weighted_drafter_ranks=((3, 1), (9, 1)),
        )
        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        self.assertEqual(self.manager._open_drafter_rank_by_req[req.rid], 3)
        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
        )
        old_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(old_batch)
        self.manager.before_process_batch_result(old_batch, result)

        req.retraction_count = 1
        req.is_retracted = True
        self.manager.retract_request(req)
        old_close = self.data_plane.close_request.call_args.args[0]
        self.assertEqual(old_close.dst_drafter_rank, 3)
        req.is_retracted = False
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        new_open = self._open_rows()[-1][0]
        self.assertEqual(new_open.request_id, "req::draft-epoch::2")
        self.assertEqual(new_open.dst_drafter_rank, 9)

        self.data_plane.reset_mock()
        self.manager.after_process_batch_result(old_batch, result)
        self.data_plane.submit_control_batch.assert_not_called()

        new_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(new_batch)
        self.manager.before_process_batch_result(new_batch, result)
        req.output_ids.append(31)
        self.manager.after_process_batch_result(new_batch, result)
        new_commit_batch = self.data_plane.submit_control_batch.call_args.args[0]
        self.assertEqual(new_commit_batch.dst_drafter_rank, 9)
        self.assertEqual(
            new_commit_batch.verify_commit_messages[0].request_id,
            "req::draft-epoch::2",
        )

    def test_decode_without_an_open_lifecycle_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "no open draft mirror"):
            self.manager.prepare_batch(_batch([_Req("req")], ForwardMode.DECODE))

    def test_extend_opens_gpu_seat_with_monotonic_request_epoch(self):
        req = _Req("req", prompt_tokens=(1, 2, 3), req_pool_idx=5)

        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))

        sync, gpu_seat, request_epoch = self._open_rows()[0]
        self.assertEqual(sync.request_id, "req::draft-epoch::1")
        self.assertEqual(sync.prompt_token_ids, [1, 2, 3])
        self.assertEqual(sync.max_new_tokens, 128)
        self.assertEqual(gpu_seat, 5)
        self.assertEqual(request_epoch, 1)

    def test_extend_accepts_last_physical_request_pool_seat(self):
        req = _Req("req", req_pool_idx=16)

        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))

        self.assertEqual(self._open_rows()[0][1], 16)

    def test_same_req_and_retraction_lifecycle_keeps_one_wire_identity(self):
        req = _Req("req", req_pool_idx=5)
        first_batch = _batch([req], ForwardMode.EXTEND)
        second_batch = _batch([req], ForwardMode.EXTEND)

        self.manager.prepare_batch(first_batch)
        self.manager.prepare_batch(second_batch)

        self.assertEqual(
            first_batch.decoupled_launch_mirror_ids,
            ["req::draft-epoch::1"],
        )
        self.assertEqual(
            second_batch.decoupled_launch_mirror_ids,
            first_batch.decoupled_launch_mirror_ids,
        )
        self.data_plane.open_requests.assert_called_once()

    def test_schedule_batch_copy_freezes_launch_mirror_ids(self):
        launch_ids = ["req::draft-epoch::1"]
        expected_request_epochs = torch.tensor([1], dtype=torch.int64)
        batch = ScheduleBatch(
            reqs=[],
            forward_mode=ForwardMode.DECODE,
            decoupled_launch_mirror_ids=launch_ids,
            decoupled_expected_request_epochs=expected_request_epochs,
            decoupled_has_new_lifecycle=True,
            decoupled_needs_landing_fence=True,
            decoupled_landing_event=object(),
        )

        result_batch = batch.copy()
        launch_ids[0] = "req::draft-epoch::2"

        self.assertEqual(
            result_batch.decoupled_launch_mirror_ids,
            ["req::draft-epoch::1"],
        )
        self.assertIsNone(result_batch.decoupled_expected_request_epochs)
        self.assertFalse(result_batch.decoupled_has_new_lifecycle)
        self.assertFalse(result_batch.decoupled_needs_landing_fence)
        self.assertIsNone(result_batch.decoupled_landing_event)

    def test_tail_selector_metrics_are_fixed_window_counters(self):
        batch = SimpleNamespace(
            decoupled_result_mirror_ids=[
                "req-a::draft-epoch::1",
                "req-b::draft-epoch::2",
            ]
        )
        self.manager._record_tail_select_result(
            batch,
            SimpleNamespace(
                decoupled_rebase_valid=torch.tensor([1, 0], dtype=torch.int64),
                decoupled_selected_draft_lens=torch.tensor([3, 0], dtype=torch.int64),
                num_proposed_drafts_per_req_cpu=[3, 0],
                decoupled_tail_select_debug=torch.tensor(
                    [
                        [5, 10, 0, 5, 5, 0, 7, 0, 0, 0, 2],
                        [8, 20, 2, 0, 0, 2, 9, 0, 0, 0, 0],
                    ],
                    dtype=torch.int64,
                ),
            ),
        )
        self.manager._record_tail_select_result(
            batch,
            SimpleNamespace(
                decoupled_rebase_valid=torch.tensor([1, 1], dtype=torch.int64),
                decoupled_selected_draft_lens=torch.tensor([3, 2], dtype=torch.int64),
                num_proposed_drafts_per_req_cpu=[3, 2],
                decoupled_tail_select_debug=torch.tensor(
                    [
                        [10, 11, 1, 5, 5, 0, 8, 0, 0, 1, 3],
                        [5, 20, 0, 2, 2, 0, 11, 0, 0, 0, 0],
                    ],
                    dtype=torch.int64,
                ),
            ),
        )

        window = self.manager.take_decode_metrics_window()
        tail = window.tail_select
        self.assertEqual(tail.num_select_rows, 4)
        self.assertEqual(tail.num_select_valid_rows, 3)
        self.assertEqual(
            tail.reason_counts,
            {"direct": 2, "delta_beyond_consumable": 1, "rebased": 1},
        )
        self.assertEqual(tail.selected_draft_length_histogram.counts, [1, 0, 1, 2])
        self.assertEqual(tail.pending_prefix_length_histogram.offset, -1)
        self.assertEqual(tail.pending_prefix_length_histogram.counts[1], 3)
        self.assertEqual(tail.pending_prefix_length_histogram.counts[3], 1)
        self.assertEqual(tail.num_publish_seq_initial, 2)
        self.assertEqual(tail.num_publish_seq_same, 1)
        self.assertEqual(tail.num_publish_seq_advance, 1)
        self.assertEqual(tail.num_pending_prefix_fast_forwards, 1)
        self.assertEqual(tail.num_protocol_errors, 0)
        self.assertEqual(tail.num_seqlock_retry_rows, 2)
        self.assertEqual(tail.num_seqlock_retries, 5)
        self.assertEqual(tail.max_seqlock_retries, 3)

        reset = self.manager.take_decode_metrics_window().tail_select
        self.assertEqual(reset.num_select_rows, 0)
        self.assertEqual(reset.num_select_valid_rows, 0)
        self.assertEqual(reset.reason_counts, {})
        self.assertFalse(any(reset.selected_draft_length_histogram.counts))
        self.assertEqual(reset.num_seqlock_retry_rows, 0)
        self.assertEqual(reset.num_seqlock_retries, 0)
        self.assertEqual(reset.max_seqlock_retries, 0)

    def test_tail_debug_is_read_only_after_result_processor_barrier(self):
        class DeferredDebugRows:
            ndim = 2
            shape = (1, 10)

            def __init__(self):
                self.ready = False
                self.tolist_calls = 0

            def tolist(self):
                self.tolist_calls += 1
                if not self.ready:
                    raise RuntimeError("async D2H is not complete")
                return [[8, 1, 1, 0, 0, 0, 1, 0, 0, 0]]

        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        decode_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(decode_batch)
        debug_rows = DeferredDebugRows()
        result = SimpleNamespace(
            decoupled_rebase_valid=torch.tensor([0], dtype=torch.int64),
            decoupled_selected_draft_lens=torch.tensor([0], dtype=torch.int64),
            num_proposed_drafts_per_req_cpu=[0],
            decoupled_tail_select_debug=debug_rows,
        )
        metrics_reporter = MagicMock()
        self.scheduler.metrics_reporter = metrics_reporter

        self.manager.before_process_batch_result(decode_batch, result)

        self.assertEqual(debug_rows.tolist_calls, 0)
        metrics_reporter.finish_decoupled_decode_metrics_window.assert_not_called()

        # This transition represents BatchResultProcessor's copy_done barrier.
        debug_rows.ready = True
        self.manager.after_process_batch_result(decode_batch, result)

        self.assertEqual(debug_rows.tolist_calls, 1)
        metrics_reporter.finish_decoupled_decode_metrics_window.assert_called_once_with()

    def test_tail_selector_reason_must_match_row_valid(self):
        with self.assertRaisesRegex(RuntimeError, "reason disagrees"):
            self.manager._record_tail_select_result(
                SimpleNamespace(decoupled_result_mirror_ids=["req::draft-epoch::1"]),
                SimpleNamespace(
                    decoupled_rebase_valid=torch.tensor([0], dtype=torch.int64),
                    decoupled_selected_draft_lens=torch.tensor([0], dtype=torch.int64),
                    num_proposed_drafts_per_req_cpu=[0],
                    decoupled_tail_select_debug=torch.tensor(
                        [[5, 1, 0, 0, 0, 0, 0]], dtype=torch.int64
                    ),
                ),
            )

    def test_gpu_tail_update_error_fails_fast_with_request_identity(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "request_id=req::draft-epoch::1 error_code=7 error_op_seq=123",
        ):
            self.manager._record_tail_select_result(
                SimpleNamespace(decoupled_result_mirror_ids=["req::draft-epoch::1"]),
                SimpleNamespace(
                    decoupled_rebase_valid=torch.tensor([0], dtype=torch.int64),
                    decoupled_selected_draft_lens=torch.tensor([0], dtype=torch.int64),
                    num_proposed_drafts_per_req_cpu=[0],
                    decoupled_tail_select_debug=torch.tensor(
                        [[4, 1, 0, 0, 0, 0, 0, 7, 123, 0]], dtype=torch.int64
                    ),
                ),
            )

        self.assertEqual(self.manager._protocol_error_ct, 1)

    def test_transient_selector_row_allows_unavailable_fast_forward_counter(self):
        self.manager._record_tail_select_result(
            SimpleNamespace(decoupled_result_mirror_ids=["req::draft-epoch::1"]),
            SimpleNamespace(
                decoupled_rebase_valid=torch.tensor([0], dtype=torch.int64),
                decoupled_selected_draft_lens=torch.tensor([0], dtype=torch.int64),
                num_proposed_drafts_per_req_cpu=[0],
                decoupled_tail_select_debug=torch.tensor(
                    [[2, -1, -1, -1, -1, -1, -1, 0, -1, -1]],
                    dtype=torch.int64,
                ),
            ),
        )

        tail = self.manager.take_decode_metrics_window().tail_select
        self.assertEqual(tail.reason_counts, {"writer_in_progress": 1})
        self.assertEqual(tail.num_pending_prefix_fast_forwards, 0)

    def test_commit_is_emitted_only_after_result_processing(self):
        req = _Req("req", output_tokens=(30,))
        extend_batch = _batch([req], ForwardMode.EXTEND)
        self.manager.prepare_batch(extend_batch)
        self.data_plane.reset_mock()

        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
        )
        decode_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(decode_batch)
        self.manager.before_process_batch_result(decode_batch, result)
        self.data_plane.submit_control_batch.assert_not_called()

        # This mutation stands in for the normal scheduler result processor.
        req.output_ids.extend([31, 32])
        self.manager.after_process_batch_result(decode_batch, result)

        self.data_plane.submit_control_batch.assert_called_once()
        control_batch = self.data_plane.submit_control_batch.call_args.args[0]
        self.assertEqual(len(control_batch.verify_commit_messages), 1)
        self.assertEqual(control_batch.close_messages, [])
        commit = control_batch.verify_commit_messages[0]
        self.assertEqual(commit.request_id, "req::draft-epoch::1")
        self.assertEqual(commit.pre_verify_committed_len, 1)
        self.assertEqual(commit.committed_tokens, [31, 32])

    def test_finished_planned_row_consumes_close_landing_fence(self):
        req = _Req("req", output_tokens=(30,))
        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
        )
        finished_batch = _batch([req], ForwardMode.EXTEND)
        self.manager.prepare_batch(finished_batch)
        self.manager.before_process_batch_result(finished_batch, result)
        req._finished = True
        self.manager.after_process_batch_result(finished_batch, result)
        self.assertTrue(self.manager._has_pending_lifecycle_landing_fence)
        self.verify_worker.capture_expected_request_epochs.reset_mock()

        stale_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(stale_batch)

        self.assertEqual(stale_batch.decoupled_launch_mirror_ids, ["req"])
        self.assertTrue(stale_batch.decoupled_needs_landing_fence)
        self.assertFalse(self.manager._has_pending_lifecycle_landing_fence)
        self.verify_worker.capture_expected_request_epochs.assert_called_once_with(
            stale_batch
        )

        next_stale_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(next_stale_batch)
        self.assertFalse(next_stale_batch.decoupled_needs_landing_fence)

    def test_non_entry_rank_tolerates_retracted_planned_row_without_fence(self):
        self.manager.close()
        self.scheduler.ps.tp_rank = 1
        self.data_plane_factory.reset_mock()
        self.verify_worker.reset_mock()
        self.manager = DecoupledVerifyManager(self.scheduler, self.config)
        req = _Req("req")
        req.is_retracted = True
        batch = _batch([req], ForwardMode.DECODE)

        self.manager.prepare_batch(batch)

        self.data_plane_factory.assert_not_called()
        self.assertEqual(batch.decoupled_launch_mirror_ids, ["req"])
        self.assertFalse(batch.decoupled_needs_landing_fence)
        self.verify_worker.capture_expected_request_epochs.assert_called_once_with(batch)

    def test_result_cursor_overrides_mutable_host_output_length(self):
        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        self.data_plane.reset_mock()

        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
            decoupled_pre_output_lens=torch.tensor([1], dtype=torch.int64),
        )
        decode_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(decode_batch)

        # A shared Req may already reflect another delayed result by the time
        # this result is processed. The forward-derived cursor must win.
        req.output_ids.append(31)
        self.manager.before_process_batch_result(decode_batch, result)
        req.output_ids.append(32)
        self.manager.after_process_batch_result(decode_batch, result)

        control_batch = self.data_plane.submit_control_batch.call_args.args[0]
        commit = control_batch.verify_commit_messages[0]
        self.assertEqual(commit.pre_verify_committed_len, 1)
        self.assertEqual(commit.committed_tokens, [31, 32])

    def test_unpublished_gpu_cursor_falls_back_to_host_result_cursor(self):
        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        self.data_plane.reset_mock()

        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
            decoupled_pre_output_lens=torch.tensor([-1], dtype=torch.int64),
        )
        decode_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(decode_batch)
        self.manager.before_process_batch_result(decode_batch, result)
        req.output_ids.append(31)
        self.manager.after_process_batch_result(decode_batch, result)

        control_batch = self.data_plane.submit_control_batch.call_args.args[0]
        commit = control_batch.verify_commit_messages[0]
        self.assertEqual(commit.pre_verify_committed_len, 1)
        self.assertEqual(commit.committed_tokens, [31])

    def test_reseat_closes_old_generation_before_opening_new_generation(self):
        req = _Req("req", output_tokens=(30,))
        batch = _batch([req], ForwardMode.EXTEND)
        self.manager.prepare_batch(batch)
        self.data_plane.reset_mock()

        req.retraction_count = 1
        self.manager.prepare_batch(batch)

        self.assertEqual(
            [call[0] for call in self.data_plane.method_calls],
            ["close_request", "open_requests"],
        )
        close = self.data_plane.close_request.call_args.args[0]
        sync, _, request_epoch = self._open_rows()[0]
        self.assertEqual(close.request_id, "req::draft-epoch::1")
        self.assertEqual(close.reason, "reseated")
        self.assertEqual(sync.request_id, "req::draft-epoch::2")
        self.assertEqual(request_epoch, 2)

    def test_finished_same_rid_reopens_with_a_new_wire_epoch(self):
        old_req = _Req("req", output_tokens=(30,))
        old_batch = _batch([old_req], ForwardMode.EXTEND)
        self.manager.prepare_batch(old_batch)
        self.manager.before_process_batch_result(
            old_batch,
            SimpleNamespace(
                decoupled_rebase_valid=None,
                decoupled_selected_draft_lens=None,
            ),
        )
        self.data_plane.reset_mock()

        old_req._finished = True
        self.manager.after_process_batch_result(
            old_batch,
            SimpleNamespace(
                decoupled_rebase_valid=None,
                decoupled_selected_draft_lens=None,
            ),
        )
        close_batch = self.data_plane.submit_control_batch.call_args.args[0]
        self.assertEqual(
            [message.request_id for message in close_batch.close_messages],
            ["req::draft-epoch::1"],
        )
        self.assertNotIn(old_req.rid, self.manager._open_mirror_by_req)

        self.data_plane.reset_mock()
        new_req = _Req("req", output_tokens=(30,), req_pool_idx=7)
        new_batch = _batch([new_req], ForwardMode.EXTEND)
        self.manager.prepare_batch(new_batch)

        self.assertEqual(
            new_batch.decoupled_launch_mirror_ids,
            ["req::draft-epoch::2"],
        )
        sync, _, request_epoch = self._open_rows()[0]
        self.assertEqual(sync.request_id, "req::draft-epoch::2")
        self.assertEqual(request_epoch, 2)

        # An abort for the completed Req object must not close the replacement.
        self.data_plane.reset_mock()
        self.manager.abort_request(old_req)
        self.assertEqual(
            self.manager._open_mirror_by_req[new_req.rid],
            "req::draft-epoch::2",
        )
        self.data_plane.close_request.assert_not_called()

    def test_one_result_batch_coalesces_commits_and_finished_closes(self):
        continuing_req = _Req("continuing", output_tokens=(30,))
        finished_req = _Req("finished", output_tokens=(40,), req_pool_idx=1)
        self.manager.prepare_batch(
            _batch([continuing_req, finished_req], ForwardMode.EXTEND)
        )
        self.data_plane.reset_mock()

        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
        )
        decode_batch = _batch(
            [continuing_req, finished_req],
            ForwardMode.DECODE,
        )
        self.manager.prepare_batch(decode_batch)
        self.manager.before_process_batch_result(decode_batch, result)
        continuing_req.output_ids.append(31)
        finished_req._finished = True

        self.manager.after_process_batch_result(decode_batch, result)

        self.data_plane.submit_control_batch.assert_called_once()
        control_batch = self.data_plane.submit_control_batch.call_args.args[0]
        self.assertEqual(
            [message.request_id for message in control_batch.verify_commit_messages],
            ["continuing::draft-epoch::1"],
        )
        self.assertEqual(
            [message.request_id for message in control_batch.close_messages],
            ["finished::draft-epoch::2"],
        )
        self.assertIn(continuing_req.rid, self.manager._open_mirror_by_req)
        self.assertNotIn(finished_req.rid, self.manager._open_mirror_by_req)
        self.data_plane.commit.assert_not_called()
        self.data_plane.close_request.assert_not_called()

    def test_stale_result_cannot_affect_new_generation(self):
        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
        )
        old_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(old_batch)
        self.manager.before_process_batch_result(old_batch, result)

        req.retraction_count = 1
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        calls_after_reseat = list(self.data_plane.method_calls)
        req._finished = True
        self.manager.after_process_batch_result(old_batch, result)

        self.assertEqual(
            self.manager._open_mirror_by_req[req.rid], "req::draft-epoch::2"
        )
        self.assertEqual(self.data_plane.method_calls, calls_after_reseat)

    def test_abort_closes_live_generation_and_suppresses_inflight_commit(self):
        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        self.data_plane.reset_mock()

        result = SimpleNamespace(
            decoupled_rebase_valid=None,
            decoupled_selected_draft_lens=None,
            decoupled_pre_output_lens=torch.tensor([1], dtype=torch.int64),
        )
        decode_batch = _batch([req], ForwardMode.DECODE)
        self.manager.prepare_batch(decode_batch)
        self.manager.before_process_batch_result(decode_batch, result)

        self.manager.abort_request(req)
        req.output_ids.append(31)
        self.manager.after_process_batch_result(decode_batch, result)

        self.assertNotIn(req.rid, self.manager._open_mirror_by_req)
        self.data_plane.submit_control_batch.assert_not_called()
        self.data_plane.close_request.assert_called_once()
        self.assertEqual(
            self.data_plane.close_request.call_args.args[0].request_id,
            "req::draft-epoch::1",
        )

    def test_retract_closes_old_mirror_before_recompute(self):
        req = _Req("req", output_tokens=(30,))
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        self.data_plane.reset_mock()
        req.retraction_count = 1
        req.is_retracted = True

        self.manager.retract_request(req)

        self.assertNotIn(req.rid, self.manager._open_mirror_by_req)
        close = self.data_plane.close_request.call_args.args[0]
        self.assertEqual(close.request_id, "req::draft-epoch::1")
        self.assertEqual(close.reason, "retracted")

    def test_pause_retract_closes_unique_mirrors_before_generic_release(self):
        req_a = _Req("req-a", output_tokens=(30,))
        req_b = _Req("req-b", output_tokens=(40,), req_pool_idx=1)
        self.manager.prepare_batch(_batch([req_a, req_b], ForwardMode.EXTEND))
        self.data_plane.reset_mock()
        events = []
        self.data_plane.close_request.side_effect = lambda close: events.append(
            ("close", close.request_id)
        )

        retract_reqs = self.manager.prepare_pause_retract([req_a, req_a, req_b])
        events.append(("generic-release", None))

        self.assertEqual(retract_reqs, [req_a, req_b])
        self.assertEqual(
            events,
            [
                ("close", "req-a::draft-epoch::1"),
                ("close", "req-b::draft-epoch::2"),
                ("generic-release", None),
            ],
        )
        self.assertEqual(self.manager._open_mirror_by_req, {})
        self.assertEqual(self.manager._open_lifecycle_by_req, {})
        self.assertEqual(self.manager._open_drafter_rank_by_req, {})

    def test_abort_after_retraction_closes_the_still_owned_old_epoch(self):
        req = _Req("req")
        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))
        self.data_plane.reset_mock()

        # reset_for_retract increments this before the request is reseated.
        req.retraction_count = 1
        req.is_retracted = True
        self.manager.abort_request(req)

        self.data_plane.close_request.assert_called_once()
        close = self.data_plane.close_request.call_args.args[0]
        self.assertEqual(close.request_id, "req::draft-epoch::1")
        self.assertEqual(close.reason, "abort")
        self.assertNotIn(req.rid, self.manager._open_mirror_by_req)


if __name__ == "__main__":
    unittest.main()

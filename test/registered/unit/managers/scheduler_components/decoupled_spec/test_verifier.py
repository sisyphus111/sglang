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
        self.verify_worker = MagicMock()
        self.scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0, tp_size=1),
            server_args=SimpleNamespace(speculative_num_steps=3),
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
        )
        self.get_stream.assert_called_once_with("decoupled_spec_landing")
        self.data_plane.start.assert_called_once_with()
        self.verify_worker.attach_gpu_tail_buffer.assert_called_once_with(
            self.gpu_tail_buffer
        )
        self.data_plane.reset_mock()

    def test_decode_reuses_only_an_open_lifecycle_identity(self):
        req_a = _Req("req-a", retraction_count=0)
        req_b = _Req("req-b", retraction_count=2)
        self.manager.prepare_batch(_batch([req_a, req_b], ForwardMode.EXTEND))
        self.data_plane.reset_mock()
        batch = _batch([req_a, req_b], ForwardMode.DECODE)

        self.manager.prepare_batch(batch)

        self.assertEqual(
            batch.decoupled_launch_mirror_ids,
            ["req-a::draft-epoch::1", "req-b::draft-epoch::2"],
        )
        self.data_plane.assert_not_called()

    def test_decode_without_an_open_lifecycle_fails_fast(self):
        with self.assertRaisesRegex(RuntimeError, "no open draft mirror"):
            self.manager.prepare_batch(_batch([_Req("req")], ForwardMode.DECODE))

    def test_extend_opens_gpu_seat_with_monotonic_request_epoch(self):
        req = _Req("req", prompt_tokens=(1, 2, 3), req_pool_idx=5)

        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))

        sync = self.data_plane.open_request.call_args.args[0]
        self.assertEqual(sync.request_id, "req::draft-epoch::1")
        self.assertEqual(sync.prompt_token_ids, [1, 2, 3])
        self.assertEqual(self.data_plane.open_request.call_args.kwargs["gpu_seat"], 5)
        self.assertEqual(
            self.data_plane.open_request.call_args.kwargs["request_epoch"], 1
        )

    def test_extend_accepts_last_physical_request_pool_seat(self):
        req = _Req("req", req_pool_idx=16)

        self.manager.prepare_batch(_batch([req], ForwardMode.EXTEND))

        self.assertEqual(self.data_plane.open_request.call_args.kwargs["gpu_seat"], 16)

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
        self.data_plane.open_request.assert_called_once()

    def test_schedule_batch_copy_freezes_launch_mirror_ids(self):
        launch_ids = ["req::draft-epoch::1"]
        batch = ScheduleBatch(
            reqs=[],
            forward_mode=ForwardMode.DECODE,
            decoupled_launch_mirror_ids=launch_ids,
        )

        result_batch = batch.copy()
        launch_ids[0] = "req::draft-epoch::2"

        self.assertEqual(
            result_batch.decoupled_launch_mirror_ids,
            ["req::draft-epoch::1"],
        )

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
            ["close_request", "open_request"],
        )
        close = self.data_plane.close_request.call_args.args[0]
        sync = self.data_plane.open_request.call_args.args[0]
        self.assertEqual(close.request_id, "req::draft-epoch::1")
        self.assertEqual(close.reason, "reseated")
        self.assertEqual(sync.request_id, "req::draft-epoch::2")
        self.assertEqual(
            self.data_plane.open_request.call_args.kwargs["request_epoch"], 2
        )

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
        sync = self.data_plane.open_request.call_args.args[0]
        self.assertEqual(sync.request_id, "req::draft-epoch::2")
        self.assertEqual(
            self.data_plane.open_request.call_args.kwargs["request_epoch"], 2
        )

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

"""CPU-only contract tests for the decoupled drafter scheduler component."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
    DraftReqKey,
    DraftSync,
    ReadyDraftControls,
    VerifierCommitSegment,
)

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_DRAFT_MODULE = "sglang.srt.managers.scheduler_components.decoupled_spec.draft"


class _DraftBatch:
    def __init__(self, reqs, *, forward_mode=ForwardMode.DECODE) -> None:
        self.reqs = list(reqs)
        self.forward_mode = forward_mode
        self.batch_is_full = True

    def filter_batch(self, *, keep_indices) -> None:
        self.reqs = [self.reqs[index] for index in keep_indices]

    def is_empty(self) -> bool:
        return not self.reqs

    def merge_batch(self, other) -> None:
        self.reqs.extend(other.reqs)


class TestDecoupledDraftManager(CustomTestCase):
    def setUp(self) -> None:
        data_plane_patcher = patch(
            f"{_DRAFT_MODULE}.create_drafter_decoupled_spec_data_plane",
            autospec=True,
        )
        self.addCleanup(data_plane_patcher.stop)
        data_plane_factory = data_plane_patcher.start()
        self.data_plane = data_plane_factory.return_value
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls()
        self.data_plane.pending_control_count.return_value = 0
        self.scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0, tp_size=1),
            server_args=SimpleNamespace(
                speculative_num_steps=3,
                enable_metrics=False,
            ),
            req_to_token_pool=SimpleNamespace(mamba_pool=None),
            tokenizer=MagicMock(),
            model_config=SimpleNamespace(
                vocab_size=128,
                hf_eos_token_id={2},
            ),
            metrics_collector=None,
            init_req_max_new_tokens=MagicMock(),
            _add_request_to_queue=MagicMock(),
            waiting_queue=[],
            running_batch=None,
            last_batch=None,
            cur_batch_for_debug=None,
            tree_cache=SimpleNamespace(supports_mamba=lambda: False),
        )
        config = DecoupledSpecIpcConfig(
            bind_endpoint="ipc:///tmp/unused-drafter",
            connect_endpoints=("ipc:///tmp/unused-verifier",),
            rank=0,
        )
        self.manager = DecoupledDraftManager(self.scheduler, config)
        self.data_plane.start.assert_called_once_with()
        self.data_plane.reset_mock()

    @staticmethod
    def _sync(request_id="req"):
        return DraftSync(
            request_id=request_id,
            src_verifier_rank=0,
            dst_drafter_rank=0,
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
            full_untruncated_fill_ids=array("q"),
            finished_reason=None,
            finished_len=None,
            _refresh_fill_ids=MagicMock(),
        )
        key = DraftRequestGeneration(
            src_verifier_rank=src_verifier_rank,
            request_id=request_id,
            generation=0,
        )
        req.decoupled_draft_generation = key
        state = SimpleNamespace(
            key=key,
            req=req,
            src_verifier_rank=src_verifier_rank,
            committed_len=committed_len,
            published_len=len(output_tokens),
            is_sleeping=False,
        )
        self.manager._requests[(src_verifier_rank, request_id)] = state
        return req, state

    def test_published_tail_targets_the_true_sparse_verifier_rank(self):
        req, _ = self._install_request(
            output_tokens=[10],
            src_verifier_rank=7,
        )
        batch = _DraftBatch([req])
        self.manager.before_process_batch_result(batch, SimpleNamespace())
        req.output_ids.append(11)

        self.manager.after_process_batch_result(batch, SimpleNamespace())

        output = self.data_plane.publish_tails.call_args.args[0].outputs[0]
        self.assertEqual(output.src_drafter_rank, 0)
        self.assertEqual(output.dst_verifier_rank, 7)

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
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls(
            sync_messages=[self._sync()]
        )

        with patch(
            f"{_DRAFT_MODULE}.SamplingParams", return_value=sampling_params
        ), patch(f"{_DRAFT_MODULE}.Req", return_value=fake_req) as req_cls:
            self.manager.process_pending_controls()

        sampling_params.normalize.assert_called_once_with(self.scheduler.tokenizer)
        sampling_params.verify.assert_called_once_with(128)
        self.assertEqual(req_cls.call_args.args[0], "draft:0:req")
        self.assertEqual(list(req_cls.call_args.args[2]), [1, 2, 3])
        self.assertEqual(list(fake_req.output_ids), [10])
        self.assertEqual(events, ["refresh", "init_max_new_tokens", "enqueue"])
        self.assertIs(self.manager._requests[(0, "req")].req, fake_req)

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

    def test_full_match_advances_cursor_without_rewrite(self):
        req, state = self._install_request(output_tokens=[10, 11, 12, 13])
        self.manager.checkpoints = MagicMock()
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls(
            ready_commit_segments=[
                VerifierCommitSegment(
                    draft_key=DraftReqKey(0, "req"),
                    dst_drafter_rank=0,
                    pre_verify_committed_len=1,
                    committed_tokens=[11, 12],
                )
            ]
        )

        self.manager.process_pending_controls()

        self.assertEqual(state.committed_len, 3)
        self.assertEqual(list(req.output_ids), [10, 11, 12, 13])
        self.manager.checkpoints.restore_for_rewrite.assert_not_called()
        self.manager.checkpoints.prune.assert_called_once_with(
            state.key,
            min_position=6,
        )
        self.data_plane.publish_tail.assert_not_called()

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
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls(
            ready_commit_segments=[
                VerifierCommitSegment(
                    draft_key=DraftReqKey(0, "sleeping"),
                    dst_drafter_rank=0,
                    pre_verify_committed_len=1,
                    committed_tokens=[1],
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
        blocked_segment = VerifierCommitSegment(
            draft_key=DraftReqKey(0, "blocked"),
            dst_drafter_rank=0,
            pre_verify_committed_len=1,
            committed_tokens=[11],
        )
        ready_segment = VerifierCommitSegment(
            draft_key=DraftReqKey(0, "ready"),
            dst_drafter_rank=0,
            pre_verify_committed_len=1,
            committed_tokens=[21],
        )

        def collect_ready(consumable):
            self.assertEqual(consumable(blocked_segment), 0)
            self.assertEqual(consumable(ready_segment), 1)
            return ReadyDraftControls(ready_commit_segments=[ready_segment])

        self.data_plane.collect_ready_controls.side_effect = collect_ready

        self.manager.process_pending_controls()

        self.assertEqual(blocked_state.committed_len, 1)
        self.assertEqual(ready_state.committed_len, 2)

        blocked_req.output_ids.append(11)
        self.data_plane.collect_ready_controls.side_effect = None
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls(
            ready_commit_segments=[blocked_segment]
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
        self.manager.checkpoints.restore_for_rewrite.side_effect = (
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
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls(
            ready_commit_segments=[
                VerifierCommitSegment(
                    draft_key=DraftReqKey(0, "req"),
                    dst_drafter_rank=0,
                    pre_verify_committed_len=1,
                    committed_tokens=[11, 22],
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
        echo = self.data_plane.publish_tails.call_args.args[0].outputs[0]
        self.assertEqual(echo.base_committed_len, 2)
        self.assertEqual(echo.new_token_pos, 2)
        self.assertEqual(echo.new_token, 22)


if __name__ == "__main__":
    unittest.main()

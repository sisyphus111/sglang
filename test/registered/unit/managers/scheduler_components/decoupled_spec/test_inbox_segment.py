"""Regression oracles for decoupled-draft inbox segments and sleeping rows."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.decoupled_spec.draft import (  # noqa: E402
    DecoupledDraftManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.decoupled_draft_checkpoint import (  # noqa: E402
    DraftRequestGeneration,
)
from sglang.srt.speculative.decoupled_spec_io import (  # noqa: E402
    DecoupledSpecIpcConfig,
    DraftControlInbox,
    DraftReqKey,
    ReadyDraftControls,
    VerifierCommitSegment,
    VerifyCommit,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DRAFT_MODULE = "sglang.srt.managers.scheduler_components.decoupled_spec.draft"
_SCHEDULER_MODULE = "sglang.srt.managers.scheduler"


def _commit(request_id: str, pre: int, tokens: list[int]) -> VerifyCommit:
    return VerifyCommit(
        request_id=request_id,
        src_verifier_rank=0,
        dst_drafter_rank=0,
        pre_verify_committed_len=pre,
        committed_tokens=tokens,
    )


class _DraftBatch:
    def __init__(self, reqs) -> None:
        self.reqs = list(reqs)
        self.forward_mode = ForwardMode.DECODE
        self.batch_is_full = True

    def filter_batch(self, *, keep_indices) -> None:
        self.reqs = [self.reqs[index] for index in keep_indices]

    def is_empty(self) -> bool:
        return not self.reqs

    def merge_batch(self, other) -> None:
        self.reqs.extend(other.reqs)


class TestDraftControlInboxSegments(CustomTestCase):
    def test_blocked_request_does_not_head_of_line_block_ready_request(self):
        inbox = DraftControlInbox()
        inbox.add_verify_commit_locked(_commit("blocked", 1, [11, 12]))
        inbox.add_verify_commit_locked(_commit("ready", 1, [21]))

        ready = inbox.extract_ready_controls_locked(
            lambda segment: 0 if segment.draft_key.request_id == "blocked" else 1
        )

        self.assertEqual(
            [segment.draft_key.request_id for segment in ready.ready_commit_segments],
            ["ready"],
        )
        blocked = inbox.verifier_commit_segments[DraftReqKey(0, "blocked")]
        self.assertEqual(blocked.pre_verify_committed_len, 1)
        self.assertEqual(blocked.committed_tokens, [11, 12])

    def test_contiguous_commits_coalesce_and_partially_consume_in_order(self):
        inbox = DraftControlInbox()
        inbox.add_verify_commit_locked(_commit("req", 1, [11, 12]))
        inbox.add_verify_commit_locked(_commit("req", 3, [13, 14]))

        first = inbox.extract_ready_controls_locked(lambda _segment: 1)
        self.assertEqual(first.ready_commit_segments[0].committed_tokens, [11])
        remainder = inbox.verifier_commit_segments[DraftReqKey(0, "req")]
        self.assertEqual(remainder.pre_verify_committed_len, 2)
        self.assertEqual(remainder.committed_tokens, [12, 13, 14])

        second = inbox.extract_ready_controls_locked(lambda _segment: 2)
        self.assertEqual(second.ready_commit_segments[0].committed_tokens, [12, 13])
        remainder = inbox.verifier_commit_segments[DraftReqKey(0, "req")]
        self.assertEqual(remainder.pre_verify_committed_len, 4)
        self.assertEqual(remainder.committed_tokens, [14])

    def test_close_discards_pending_segment_and_wins_over_later_commit(self):
        inbox = DraftControlInbox()
        key = DraftReqKey(0, "req")
        inbox.add_verify_commit_locked(_commit("req", 1, [11, 12]))
        inbox.add_close_key_locked(key)
        inbox.add_verify_commit_locked(_commit("req", 1, [99]))

        ready = inbox.extract_ready_controls_locked(lambda _segment: 99)

        self.assertEqual(ready.close_keys, {key})
        self.assertEqual(ready.ready_commit_segments, [])
        self.assertTrue(inbox.is_empty())


class TestSchedulerSleepPlacement(CustomTestCase):
    def test_sleep_filter_precedes_decode_memory_check_and_allocation(self):
        events = []

        class Batch:
            batch_is_full = True
            out_cache_loc = None

            @staticmethod
            def batch_size():
                return 2

            @staticmethod
            def is_empty():
                return False

            def filter_batch(self):
                events.append("filter")

            def check_decode_mem(self):
                events.append("check_decode_mem")
                self.assert_prepared(False)
                return True

            def prepare_for_decode(self):
                events.append("prepare_for_decode")
                self.out_cache_loc = object()

            def assert_prepared(self, expected):
                self_test.assertEqual(self.out_cache_loc is not None, expected)

        self_test = self
        batch = Batch()

        def sleep_overrun_requests(input_batch):
            events.append("sleep_overrun_requests")
            self.assertIs(input_batch, batch)
            self.assertIsNone(batch.out_cache_loc)
            return input_batch

        scheduler = SimpleNamespace(
            decoupled_spec_manager=SimpleNamespace(
                sleep_overrun_requests=sleep_overrun_requests
            ),
            forward_ct=0,
            new_token_ratio_tracker=SimpleNamespace(decay_step=MagicMock()),
        )
        with patch(f"{_SCHEDULER_MODULE}.TEST_RETRACT", False):
            result = Scheduler.update_running_batch(scheduler, batch)

        self.assertIs(result, batch)
        self.assertEqual(
            events,
            [
                "filter",
                "sleep_overrun_requests",
                "check_decode_mem",
                "prepare_for_decode",
            ],
        )
        self.assertIsNotNone(batch.out_cache_loc)
        scheduler.new_token_ratio_tracker.decay_step.assert_called_once_with()


class TestDecoupledDraftSegmentAndSleep(CustomTestCase):
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
            token_to_kv_pool_allocator=MagicMock(),
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
            chunked_req=None,
            _pending_chunked_abort_req=None,
            tree_cache=SimpleNamespace(supports_mamba=lambda: False),
            device="cpu",
            enable_overlap=False,
            spec_algorithm=MagicMock(),
            future_map=SimpleNamespace(
                stash=MagicMock(),
                output_tokens_buf=torch.full((16,), -1, dtype=torch.int64),
            ),
        )
        config = DecoupledSpecIpcConfig(
            bind_endpoint="ipc:///tmp/unused-drafter-segments",
            connect_endpoints=("ipc:///tmp/unused-verifier-segments",),
            rank=0,
        )
        self.manager = DecoupledDraftManager(self.scheduler, config)
        self.data_plane.start.assert_called_once_with()
        self.data_plane.reset_mock()
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls()
        self.data_plane.pending_control_count.return_value = 0

    def _install_request(
        self,
        *,
        request_id: str,
        output_tokens,
        committed_len: int = 1,
        req_pool_idx: int = 1,
    ):
        req = SimpleNamespace(
            rid=f"draft:0:{request_id}",
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q", output_tokens),
            req_pool_idx=req_pool_idx,
            mamba_pool_idx=req_pool_idx + 10,
            kv=SimpleNamespace(kv_allocated_len=16),
            kv_committed_len=16,
            cache_protected_len=16,
            full_untruncated_fill_ids=array("q"),
            finished_reason=None,
            finished_len=None,
            multimodal_inputs=None,
            _refresh_fill_ids=MagicMock(),
        )
        key = DraftRequestGeneration(
            src_verifier_rank=0,
            request_id=request_id,
            generation=0,
        )
        req.decoupled_draft_generation = key
        state = SimpleNamespace(
            key=key,
            req=req,
            src_verifier_rank=0,
            committed_len=committed_len,
            published_len=len(req.output_ids),
            is_sleeping=False,
        )
        self.manager._requests[(0, request_id)] = state
        return req, state

    def test_mismatch_consumes_only_through_rewrite_token(self):
        req, state = self._install_request(
            request_id="req",
            output_tokens=[10, 11, 12, 13],
        )
        segment = VerifierCommitSegment(
            draft_key=DraftReqKey(0, "req"),
            dst_drafter_rank=0,
            pre_verify_committed_len=1,
            committed_tokens=[11, 99, 100],
        )
        inbox = DraftControlInbox(verifier_commit_segments={segment.draft_key: segment})
        self.manager.checkpoints = MagicMock()
        self.manager._truncate_kv = MagicMock()
        self.data_plane.collect_ready_controls.side_effect = (
            inbox.extract_ready_controls_locked
        )

        self.manager.process_pending_controls()

        self.assertEqual(list(req.output_ids), [10, 11, 99])
        self.assertEqual(state.committed_len, 3)
        remainder = inbox.verifier_commit_segments[DraftReqKey(0, "req")]
        self.assertEqual(remainder.pre_verify_committed_len, 3)
        self.assertEqual(remainder.committed_tokens, [100])
        self.assertEqual(self.scheduler.future_map.output_tokens_buf[1].item(), 99)
        echo = self.data_plane.publish_tails.call_args.args[0].outputs[0]
        self.assertEqual((echo.new_token_pos, echo.new_token), (2, 99))

    def test_sleep_retains_ownership_then_commit_rebuilds_and_merges(self):
        ready_req, _ = self._install_request(
            request_id="ready",
            output_tokens=[20, 21],
            req_pool_idx=1,
        )
        sleep_req, sleep_state = self._install_request(
            request_id="sleep",
            output_tokens=range(8),
            req_pool_idx=2,
        )
        checkpoint_store = MagicMock()
        self.manager.checkpoints = checkpoint_store
        original_kv = sleep_req.kv
        running_batch = _DraftBatch([ready_req, sleep_req])
        self.scheduler.running_batch = running_batch

        with patch(f"{_DRAFT_MODULE}.release_kv_cache") as release_kv:
            self.manager.sleep_overrun_requests(running_batch)

        self.assertEqual(running_batch.reqs, [ready_req])
        self.assertIs(self.manager._requests[(0, "sleep")], sleep_state)
        self.assertIs(self.manager._sleeping_requests[sleep_state.key], sleep_req)
        self.assertTrue(sleep_state.is_sleeping)
        self.assertEqual(sleep_req.req_pool_idx, 2)
        self.assertIs(sleep_req.kv, original_kv)
        checkpoint_store.release.assert_not_called()
        release_kv.assert_not_called()

        ready_segment = VerifierCommitSegment(
            draft_key=DraftReqKey(0, "sleep"),
            dst_drafter_rank=0,
            pre_verify_committed_len=1,
            committed_tokens=[1],
        )
        self.data_plane.collect_ready_controls.return_value = ReadyDraftControls(
            ready_commit_segments=[ready_segment]
        )
        built_batch = _DraftBatch([sleep_req])
        with patch.object(
            self.manager,
            "_build_draft_decode_batch",
            return_value=built_batch,
        ) as build_batch:
            self.manager.process_pending_controls()

        build_batch.assert_called_once_with([sleep_req])
        self.assertEqual(running_batch.reqs, [ready_req, sleep_req])
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(sleep_state.committed_len, 2)
        self.assertFalse(sleep_state.is_sleeping)
        self.assertNotIn(sleep_state.key, self.manager._sleeping_requests)

    def test_all_sleeping_bypasses_idle_and_is_not_fully_idle(self):
        sleep_req, sleep_state = self._install_request(
            request_id="sleep",
            output_tokens=range(8),
        )
        batch = _DraftBatch([sleep_req])
        self.scheduler.running_batch = batch
        self.manager.sleep_overrun_requests(batch)
        self.assertTrue(batch.is_empty())

        self.assertTrue(self.manager.on_no_batch())
        self.data_plane.wait_for_control.assert_called_once_with(0.01)

        idle_scheduler = SimpleNamespace(
            running_batch=batch,
            chunked_req=None,
            dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
            last_batch=None,
            enable_overlap=False,
            _pp_microbatches_drained=lambda: True,
            waiting_queue=[],
            decoupled_spec_manager=self.manager,
            _engine_paused=False,
            disaggregation_mode=None,
            disagg_decode_prealloc_queue=None,
        )
        self.assertFalse(Scheduler.is_fully_idle(idle_scheduler, for_health_check=True))

        no_batch_manager = SimpleNamespace(
            adjust_plan=lambda plan: plan,
            on_no_batch=MagicMock(return_value=True),
        )
        loop_scheduler = SimpleNamespace(
            gracefully_exit=False,
            request_receiver=SimpleNamespace(recv_requests=lambda: []),
            process_input_requests=MagicMock(),
            _engine_paused=False,
            running_batch=batch,
            last_batch=None,
            decoupled_spec_manager=no_batch_manager,
            cur_batch_for_debug=None,
            run_batch=MagicMock(),
            process_batch_result=MagicMock(),
            on_idle=MagicMock(),
            invariant_checker=MagicMock(),
        )

        def one_plan(**_kwargs):
            loop_scheduler.gracefully_exit = True
            return SimpleNamespace(running_batch=batch, batch_to_run=None)

        loop_scheduler.get_next_batch_to_run = one_plan
        event_loop = Scheduler.event_loop_normal.__wrapped__
        with patch(
            f"{_SCHEDULER_MODULE}.envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get",
            return_value=False,
        ):
            event_loop(loop_scheduler)

        no_batch_manager.on_no_batch.assert_called_once_with()
        loop_scheduler.on_idle.assert_not_called()
        self.assertTrue(sleep_state.is_sleeping)

    def test_close_and_retract_remove_sleeping_ownership(self):
        close_req, close_state = self._install_request(
            request_id="close",
            output_tokens=range(8),
            req_pool_idx=1,
        )
        close_batch = _DraftBatch([close_req])
        self.scheduler.running_batch = close_batch
        self.manager.sleep_overrun_requests(close_batch)
        self.manager.checkpoints = MagicMock()

        with patch(f"{_DRAFT_MODULE}.release_kv_cache") as release_kv:
            self.manager._close_request_key(DraftReqKey(0, "close"))

        self.assertNotIn((0, "close"), self.manager._requests)
        self.assertNotIn(close_state.key, self.manager._sleeping_requests)
        self.assertFalse(close_state.is_sleeping)
        self.manager.checkpoints.release.assert_called_once_with(close_state.key)
        release_kv.assert_called_once_with(
            close_req, self.scheduler.tree_cache, is_insert=False
        )

        retract_req, retract_state = self._install_request(
            request_id="retract",
            output_tokens=range(8),
            req_pool_idx=2,
        )
        retract_batch = _DraftBatch([retract_req])
        self.scheduler.running_batch = retract_batch
        self.manager.sleep_overrun_requests(retract_batch)
        self.manager.checkpoints.reset_mock()

        self.manager.retract_request(retract_req)

        self.assertNotIn(retract_state.key, self.manager._sleeping_requests)
        self.assertFalse(retract_state.is_sleeping)
        self.assertEqual(list(retract_req.output_ids), [0])
        self.manager.checkpoints.release.assert_called_once_with(retract_state.key)

    def test_mid_chunk_close_clears_owner_before_releasing_resources(self):
        req, state = self._install_request(
            request_id="mid-chunk-close",
            output_tokens=[10],
            req_pool_idx=3,
        )
        self.scheduler.chunked_req = req
        self.scheduler._pending_chunked_abort_req = req
        self.manager.checkpoints = MagicMock()
        release_order = []

        def release_checkpoint(key):
            self.assertIsNone(self.scheduler.chunked_req)
            self.assertIsNone(self.scheduler._pending_chunked_abort_req)
            release_order.append(("checkpoint", key))

        def release_request_kv(released_req, tree_cache, *, is_insert):
            self.assertIsNone(self.scheduler.chunked_req)
            self.assertIsNone(self.scheduler._pending_chunked_abort_req)
            release_order.append(("kv", released_req))

        self.manager.checkpoints.release.side_effect = release_checkpoint
        with patch(f"{_DRAFT_MODULE}.release_kv_cache", side_effect=release_request_kv):
            self.manager._close_request_key(DraftReqKey(0, "mid-chunk-close"))

        self.assertIsNone(self.scheduler.chunked_req)
        self.assertIsNone(self.scheduler._pending_chunked_abort_req)
        self.assertEqual(
            release_order,
            [("checkpoint", state.key), ("kv", req)],
        )

    def test_verifier_chunk_abort_delegates_before_scheduler_release(self):
        req = SimpleNamespace(
            rid="verifier-chunk",
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            to_finish=object(),
        )
        manager = MagicMock()
        scheduler = SimpleNamespace(
            _pending_chunked_abort_req=req,
            chunked_req=req,
            decoupled_spec_manager=manager,
            disaggregation_mode=None,
            enable_hicache_storage=False,
            tree_cache=MagicMock(),
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
            ),
        )
        release_order = []
        manager.abort_request.side_effect = lambda aborted_req: release_order.append(
            ("draft-close", aborted_req)
        )

        with (
            patch(f"{_SCHEDULER_MODULE}.prepare_abort"),
            patch(
                f"{_SCHEDULER_MODULE}.release_kv_cache",
                side_effect=lambda released_req, *_args, **_kwargs: release_order.append(
                    ("kv", released_req)
                ),
            ),
        ):
            Scheduler.process_pending_chunked_abort(scheduler)

        manager.abort_request.assert_called_once_with(req)
        self.assertEqual(release_order, [("draft-close", req), ("kv", req)])
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)


if __name__ == "__main__":
    unittest.main()

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler_decoupled_draft_mixin import (
    DraftBatchMetadataUpdate,
    DraftKVTruncation,
    DraftReqState,
    SchedulerDecoupledDraftMixin,
)
from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_spec_io import DraftReqKey, DraftSync
from sglang.srt.speculative.decoupled_draft_mamba import (
    DecoupledDraftMambaStateManager,
)
from sglang.srt.utils.common import flatten_arrays_to_pinned_cpu
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Batch(SimpleNamespace):
    def is_empty(self):
        return not self.reqs


class _RecordingAllocator:
    def __init__(self):
        self.freed = None

    def free(self, indices):
        self.freed = indices.clone()


class _DraftScheduler(SchedulerDecoupledDraftMixin):
    def init_req_max_new_tokens(self, req):
        return None


class TestDecoupledDraftBatchState(unittest.TestCase):
    def test_mamba_checkpoint_manager_prunes_and_reuses_ring_slots(self):
        allocator = SimpleNamespace(
            alloc=lambda size: torch.arange(10, 10 + size),
            available_size=lambda: 8,
            free=lambda slots: None,
        )
        scheduler = SimpleNamespace(
            _draft_ahead_window=lambda: 3,
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(size=16),
                mamba_allocator=allocator,
            ),
        )
        manager = DecoupledDraftMambaStateManager(scheduler)
        state = DraftReqState(
            key=DraftReqKey(src_verifier_rank=0, request_id="req-mamba"),
            req=SimpleNamespace(output_ids=[1, 2, 3, 4]),
            verifier_committed_prefix_len=3,
            mamba_checkpoint_positions={0, 2, 3},
        )

        manager.prune(state)

        self.assertEqual(state.mamba_checkpoint_positions, {3})
        self.assertEqual(
            manager.checkpoint_slot(state, 4, for_write=True).tolist(), [11]
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "decoupled draft metadata flush is implemented by a CUDA Triton kernel",
    )
    def test_metadata_flush_recomputes_missing_seq_lens_sum(self):
        device = torch.device("cuda")
        batch = _Batch(
            reqs=[object(), object()],
            seq_lens=torch.tensor([10, 20], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int64),
            orig_seq_lens=torch.tensor([10, 20], dtype=torch.int32, device=device),
            seq_lens_sum=None,
            output_ids=None,
            input_ids=None,
            req_pool_indices=torch.tensor([2, 3], dtype=torch.int64, device=device),
        )
        scheduler = SimpleNamespace(
            running_batch=batch,
            future_map=SimpleNamespace(
                output_tokens_buf=torch.zeros(8, dtype=torch.int64, device=device)
            ),
        )
        updates = [DraftBatchMetadataUpdate(1, 25, 99)]

        SchedulerDecoupledDraftMixin._flush_draft_batch_metadata_updates(
            scheduler, updates
        )

        self.assertEqual(batch.seq_lens_sum, 35)
        self.assertTrue(torch.equal(batch.seq_lens_cpu, torch.tensor([10, 25])))
        self.assertTrue(torch.equal(batch.seq_lens.cpu(), torch.tensor([10, 25])))
        self.assertTrue(
            torch.equal(
                batch.orig_seq_lens.cpu(), torch.tensor([10, 25], dtype=torch.int32)
            )
        )
        self.assertEqual(int(scheduler.future_map.output_tokens_buf[3]), 99)
        self.assertEqual(updates, [])

    def test_create_draft_request_uses_array_token_buffers(self):
        scheduler = _DraftScheduler()
        scheduler.draft_req_table = {}
        scheduler.tokenizer = None
        scheduler.model_config = SimpleNamespace(
            vocab_size=1000, hf_eos_token_id={99}
        )
        scheduler.server_args = SimpleNamespace(enable_metrics=False)
        message = DraftSync(
            request_id="req-a",
            src_verifier_rank=0,
            dst_drafter_rank=0,
            prompt_token_ids=[1, 2],
            committed_output_ids=[10, 11],
        )

        req = SchedulerDecoupledDraftMixin._create_draft_request(
            scheduler, message
        )

        self.assertIsInstance(req.origin_input_ids, array)
        self.assertIsInstance(req.output_ids, array)
        self.assertIsInstance(req.full_untruncated_fill_ids, array)
        self.assertIsInstance(req.get_fill_ids(), array)
        self.assertEqual(list(req.get_fill_ids()), [1, 2, 10, 11])
        self.assertEqual(
            flatten_arrays_to_pinned_cpu([req.get_fill_ids()], False).tolist(),
            [1, 2, 10, 11],
        )

    def test_kv_truncation_frees_slots_and_clears_req_to_token_mapping(self):
        req_to_token = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
            ],
            dtype=torch.int32,
        )
        allocator = _RecordingAllocator()
        scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=allocator,
        )
        truncations = [DraftKVTruncation(req_pool_idx=1, kv_start=2, kv_end=5)]

        SchedulerDecoupledDraftMixin._flush_draft_kv_truncations(
            scheduler, truncations
        )

        self.assertTrue(torch.equal(allocator.freed, torch.tensor([12, 13, 14])))
        self.assertTrue(
            torch.equal(req_to_token[1], torch.tensor([10, 11, 0, 0, 0, 15]))
        )
        self.assertTrue(
            torch.equal(req_to_token[2], torch.tensor([20, 21, 22, 23, 24, 25]))
        )
        self.assertEqual(truncations, [])

    def test_wake_decode_batch_refreshes_future_map_tail_tokens(self):
        reqs = [
            SimpleNamespace(
                req_pool_idx=2,
                origin_input_ids=array("q", [1, 2]),
                output_ids=array("q", [10, 101]),
                multimodal_inputs=None,
            ),
            SimpleNamespace(
                req_pool_idx=5,
                origin_input_ids=array("q", [3, 4]),
                output_ids=array("q", [20, 202]),
                multimodal_inputs=None,
            ),
        ]
        batch = SimpleNamespace()
        scheduler = SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=SimpleNamespace(device=torch.device("cpu")),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            model_config=SimpleNamespace(vocab_size=8),
            enable_overlap=False,
            spec_algorithm=object(),
            future_map=SimpleNamespace(
                output_tokens_buf=torch.full((8,), -1, dtype=torch.int64)
            ),
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_decoupled_draft_mixin.ScheduleBatch.init_new",
                return_value=batch,
            ),
            patch(
                "sglang.srt.managers.scheduler_decoupled_draft_mixin.SamplingBatchInfo.from_schedule_batch",
                return_value=object(),
            ),
        ):
            built = SchedulerDecoupledDraftMixin._build_draft_decode_batch(
                scheduler, reqs
            )

        self.assertIs(built, batch)
        self.assertTrue(
            torch.equal(batch.req_pool_indices_cpu, torch.tensor([2, 5]))
        )
        self.assertTrue(
            torch.equal(
                scheduler.future_map.output_tokens_buf,
                torch.tensor([-1, -1, 101, -1, -1, 202, -1, -1]),
            )
        )

    def test_flush_draft_updates_streams_current_tail_token(self):
        key = DraftReqKey(src_verifier_rank=7, request_id="req-1")
        req = SimpleNamespace(
            output_ids=array("q", [10, 11, 12, 13]),
            _decoupled_draft_state=None,
        )
        state = DraftReqState(
            key=key,
            req=req,
            verifier_committed_prefix_len=1,
        )
        req._decoupled_draft_state = state
        scheduler = SimpleNamespace(
            is_draft_worker_batch=lambda batch: True,
            is_draft_entry_rank=lambda: False,
            get_decoupled_spec_rank=lambda: 3,
            _get_draft_state_by_req=lambda item: item._decoupled_draft_state,
            draft_req_table={key: state},
        )
        batch = SimpleNamespace(reqs=[req])

        output_batch = SchedulerDecoupledDraftMixin.flush_draft_updates(
            scheduler, batch
        )

        self.assertEqual([item.new_token_pos for item in output_batch.outputs], [3])
        self.assertEqual([item.new_token_id for item in output_batch.outputs], [13])
        self.assertEqual([item.base_committed_len for item in output_batch.outputs], [1])

        state.verifier_committed_prefix_len = 4
        replay_batch = SchedulerDecoupledDraftMixin.flush_draft_updates(
            scheduler, batch
        )

        self.assertEqual(replay_batch.outputs, [])

    def test_flush_draft_updates_cpp_env_submits_native_rows(self):
        key = DraftReqKey(src_verifier_rank=7, request_id="req-1")
        req = SimpleNamespace(
            output_ids=array("q", [10, 11, 12, 13]),
            _decoupled_draft_state=None,
        )
        state = DraftReqState(
            key=key,
            req=req,
            verifier_committed_prefix_len=1,
        )
        req._decoupled_draft_state = state

        class _TokenSync:
            def __init__(self):
                self.rows = []

            def submit_draft_result_rows(self, rows):
                self.rows.append(list(rows))

            def submit_draft_results(self, _batch):
                raise AssertionError("C++ env path must use native rows")

        token_sync = _TokenSync()
        scheduler = SimpleNamespace(
            is_draft_worker_batch=lambda batch: True,
            is_draft_entry_rank=lambda: True,
            get_decoupled_spec_rank=lambda: 3,
            _get_draft_state_by_req=lambda item: item._decoupled_draft_state,
            _get_token_sync_thread=lambda: token_sync,
            draft_req_table={key: state},
        )
        batch = SimpleNamespace(reqs=[req])

        with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(True):
            result = SchedulerDecoupledDraftMixin.flush_draft_updates(
                scheduler, batch
            )

        self.assertIsNone(result)
        self.assertEqual(
            token_sync.rows,
            [[(3, 7, "req-1", 1, 3, 13)]],
        )


if __name__ == "__main__":
    unittest.main()

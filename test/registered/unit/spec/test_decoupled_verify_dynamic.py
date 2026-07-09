import unittest
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.managers.scheduler_decoupled_verify_mixin import (
    SchedulerDecoupledVerifyMixin,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
)
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import DecoupledVerifySpecAlgo
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid: str):
        self.rid = rid


class _SchedReq:
    def __init__(self, rid: str):
        self.rid = rid
        self.is_retracted = False
        self.output_ids = [1, 2]
        self.kv_committed_len = 2

    def finished(self):
        return False


class _ForwardMode:
    def __str__(self):
        return "DECODE"


class _Batch:
    def __init__(self):
        self.reqs = [_SchedReq("req-a"), _SchedReq("req-b")]
        self.forward_mode = _ForwardMode()

    def batch_size(self):
        return len(self.reqs)


class _Scheduler(SchedulerDecoupledVerifyMixin):
    def __init__(self, steps: int, draft_tokens: int):
        self.activations = []
        self.server_args = SimpleNamespace(
            speculative_adaptive=True,
            speculative_algorithm="DECOUPLED_VERIFY",
            speculative_num_steps=8,
            speculative_num_draft_tokens=9,
            disable_cuda_graph=False,
        )

        def activate_step_by_batch(batch_size):
            self.activations.append(batch_size)

        self.model_worker = SimpleNamespace(
            speculative_num_steps=steps,
            speculative_num_draft_tokens=draft_tokens,
            activate_step_by_batch=activate_step_by_batch,
        )

    def is_verify_entry_rank(self):
        return False

    def _broadcast_verify_snapshots(self, local_snapshots):
        return []

    def _bind_verify_snapshots(
        self, target_reqs, synced_snapshots, *, collect_trace_stats=False
    ):
        return 0


class TestDecoupledVerifyDynamic(unittest.TestCase):
    runtime_state_attr_name = "_".join(
        ["decoupled", "verify", "runtime", "state"]
    )

    def _make_validate_args(self, **overrides):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = SimpleNamespace(
            pp_size=1,
            decoupled_spec_rank_base=0,
            page_size=1,
            max_running_requests=1,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            mamba_radix_cache_strategy="no_buffer",
            enable_mixed_chunk=False,
            disable_piecewise_cuda_graph=False,
            speculative_algorithm="DECOUPLED_VERIFY",
            speculative_adaptive=True,
            speculative_adaptive_strategy="throughput_aware",
            speculative_adaptive_config=None,
            speculative_num_steps=8,
            speculative_num_draft_tokens=9,
            speculative_eagle_topk=1,
            speculative_use_rejection_sampling=False,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED),
                decode=SimpleNamespace(backend=Backend.FULL),
            ),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _make_throughput_profile_worker(self, cache_path=None):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.gpu_id = 0
        worker.enable_adaptive_verify = True
        worker.speculative_num_steps = 1
        worker.speculative_num_draft_tokens = 2
        worker.server_args = SimpleNamespace(
            speculative_adaptive_strategy="throughput_aware",
            _decoupled_verify_max_speculative_steps=2,
            decoupled_verify_throughput_profile_path=cache_path,
            decoupled_verify_throughput_profile_ctx_lens="256",
            model_path="/models/target-model",
            tp_size=4,
            dp_size=1,
            enable_dp_attention=False,
        )
        worker._throughput_profile_done = False
        worker._throughput_profile_states_by_step = {
            steps: SpecRuntimeState.for_decoupled_verify(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                target_attn_backend=object(),
                target_graph_runner=SimpleNamespace(capture_bs=[4, 8]),
            )
            for steps in (0, 1, 2)
        }
        worker._throughput_profile_capture_bs_by_step = {
            steps: [4, 8] for steps in (0, 1, 2)
        }
        worker._throughput_profile_bs_by_step = {
            steps: [4, 8] for steps in (0, 1, 2)
        }
        worker._throughput_profile_capture_bs = [4, 8]

        def apply_runtime_state(state):
            worker.speculative_num_steps = state.speculative_num_steps
            worker.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        worker.apply_runtime_state = apply_runtime_state
        return worker

    def _throughput_profile_cache_payload(
        self,
        worker,
        *,
        omit_last=False,
        invalid_last=False,
        fingerprint=None,
        include_fingerprint=True,
        include_costs=True,
    ):
        costs = [
            {
                "batch_size": bs,
                "steps": steps,
                "ctx_len": 256,
                "cost_ms": float(bs + steps + 1),
            }
            for steps in (0, 1, 2)
            for bs in (4, 8)
        ]
        if omit_last:
            costs.pop()
        if invalid_last:
            costs[-1] = {
                "batch_size": 8,
                "steps": 2,
                "ctx_len": 256,
                "cost_ms": -1.0,
            }
        payload = {"summary": "cached"}
        if include_fingerprint:
            payload["fingerprint"] = (
                worker._throughput_profile_fingerprint()
                if fingerprint is None
                else fingerprint
            )
        if include_costs:
            payload["costs"] = costs
        return payload

    def test_throughput_aware_validation_accepts_adaptive_full_cuda_graph(self):
        args = self._make_validate_args()

        DecoupledVerifySpecAlgo.validate_server_args(args)

        self.assertEqual(args.speculative_adaptive_strategy, "throughput_aware")

    def test_throughput_aware_validation_requires_adaptive_decoupled_verify(self):
        args = self._make_validate_args(speculative_adaptive=False)

        with self.assertRaisesRegex(ValueError, "requires adaptive"):
            DecoupledVerifySpecAlgo.validate_server_args(args)

    def test_throughput_aware_validation_requires_positive_max_steps(self):
        for speculative_num_steps in (None, 0):
            with self.subTest(speculative_num_steps=speculative_num_steps):
                args = self._make_validate_args(
                    speculative_num_steps=speculative_num_steps
                )
                with self.assertRaisesRegex(
                    ValueError, "positive --speculative-num-steps"
                ):
                    DecoupledVerifySpecAlgo.validate_server_args(args)

    def test_throughput_aware_validation_requires_full_decode_cuda_graph(self):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = self._make_validate_args(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED),
                decode=SimpleNamespace(backend=Backend.BREAKABLE),
            )
        )

        with self.assertRaisesRegex(ValueError, "full decode CUDA Graph"):
            DecoupledVerifySpecAlgo.validate_server_args(args)

    def test_decoupled_ema_requires_explicit_config(self):
        args = self._make_validate_args(speculative_adaptive_strategy="ema")

        with self.assertRaisesRegex(ValueError, "requires --speculative-adaptive-config"):
            DecoupledVerifySpecAlgo.validate_server_args(args)

    def test_decoupled_ema_explicit_config_is_accepted(self):
        args = self._make_validate_args(
            speculative_adaptive_strategy="ema",
            speculative_adaptive_config="/tmp/adaptive-config.json",
        )

        DecoupledVerifySpecAlgo.validate_server_args(args)

    def test_scheduler_prepare_uses_active_worker_config_for_snapshot(self):
        scheduler = _Scheduler(steps=3, draft_tokens=4)
        batch = _Batch()

        SchedulerDecoupledVerifyMixin._prepare_verify_decode_inputs(
            scheduler, batch
        )

        self.assertEqual(scheduler.activations, [2])
        self.assertFalse(hasattr(batch, self.runtime_state_attr_name))
        for req in batch.reqs:
            self.assertEqual(req._decoupled_verify_num_speculative_steps, 3)
            self.assertEqual(req._decoupled_verify_tokens_per_req, 4)
            self.assertEqual(req.kv_committed_len, 2)

    def test_scheduler_zero_step_uses_active_worker_config(self):
        scheduler = _Scheduler(steps=0, draft_tokens=1)
        batch = _Batch()

        SchedulerDecoupledVerifyMixin._prepare_verify_decode_inputs(
            scheduler, batch
        )

        self.assertEqual(scheduler.activations, [2])
        self.assertFalse(hasattr(batch, self.runtime_state_attr_name))
        for req in batch.reqs:
            self.assertEqual(req._decoupled_verify_num_speculative_steps, 0)
            self.assertEqual(req._decoupled_verify_tokens_per_req, 1)
            self.assertEqual(req.kv_committed_len, 2)

    def test_verify_worker_activate_step_by_batch_applies_state(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        graph_runner = SimpleNamespace(capture_bs=[16], num_tokens_per_bs=4)
        attn_backend = object()
        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            target_attn_backend=attn_backend,
            target_graph_runner=graph_runner,
        )
        worker = object.__new__(VerifyWorker)
        worker.speculative_num_steps = 8
        worker.speculative_num_draft_tokens = 9
        worker.server_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=[16], max_bs=None)
            ),
            cuda_graph_bs=None,
            cuda_graph_max_bs=None,
        )
        model_runner = SimpleNamespace(
            attn_backend=object(), decode_cuda_graph_runner=None
        )
        worker._target_worker = SimpleNamespace(model_runner=model_runner)

        class _Controller:
            def __init__(self):
                self.activated_steps = []

            def activate_step_by_batch(self, batch_size):
                self.batch_size = batch_size
                self.activated_steps.append(3)
                VerifyWorker.apply_runtime_state(worker, state)

        controller = _Controller()
        worker.adaptive_controller = controller

        result = VerifyWorker.activate_step_by_batch(worker, batch_size=16)

        self.assertIsNone(result)
        self.assertEqual(controller.batch_size, 16)
        self.assertEqual(controller.activated_steps, [3])
        self.assertEqual(worker.speculative_num_steps, 3)
        self.assertEqual(worker.speculative_num_draft_tokens, 4)
        self.assertIs(model_runner.attn_backend, attn_backend)
        self.assertIs(model_runner.decode_cuda_graph_runner, graph_runner)

    def test_verify_worker_verify_returns_generation_batch_result(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        req = SimpleNamespace(
            rid="req-a",
            draft_buffer=[10],
            kv_committed_len=2,
            kv_allocated_len=2,
            spec_valid_draft_tokens=0,
            spec_valid_accepted_tokens=0,
            spec_valid_draft_tokens_by_position=[],
            spec_valid_accepted_tokens_by_position=[],
        )
        spec_info = EagleVerifyInput(
            draft_token=torch.tensor([2, 10], dtype=torch.long),
            custom_mask=torch.ones(4, dtype=torch.bool),
            positions=torch.tensor([2, 3], dtype=torch.int64),
            retrieve_index=torch.tensor([[0, 1]], dtype=torch.long),
            retrieve_next_token=torch.tensor([[1, -1]], dtype=torch.long),
            retrieve_next_sibling=torch.full((1, 2), -1, dtype=torch.long),
            retrieve_cum_len=None,
            spec_steps=1,
            topk=1,
            draft_token_num=2,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([2], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            input_ids=torch.tensor([2], dtype=torch.long),
            out_cache_loc=torch.tensor([4], dtype=torch.int64),
            tree_cache=object(),
            return_hidden_states=True,
            return_logprob=False,
            spec_info=spec_info,
            has_grammar=False,
        )

        routed_output = object()
        indexer_output = object()
        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros((2, 8), dtype=torch.float32)
        )
        verify_forward_batch = SimpleNamespace(
            capture_hidden_mode=CaptureHiddenMode.NULL
        )

        class _TargetWorker:
            def __init__(self):
                self.forward_kwargs = None

            def forward_batch_generation(self, **kwargs):
                self.forward_kwargs = kwargs
                return GenerationBatchResult(
                    logits_output=logits_output,
                    can_run_cuda_graph=True,
                    routed_experts_output=routed_output,
                    indexer_topk_output=indexer_output,
                )

        target_worker = _TargetWorker()
        freed = []
        worker = object.__new__(VerifyWorker)
        worker.device = "cpu"
        worker.page_size = 1
        worker.speculative_num_steps = 1
        worker.speculative_num_draft_tokens = 2
        worker.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.empty((1, 4), dtype=torch.int64)
        )
        worker.token_to_kv_pool_allocator = SimpleNamespace(
            free=lambda slots: freed.append(slots.clone())
        )
        worker._target_worker = target_worker

        def fake_prepare(verify_input, req_to_token_pool, batch, target_worker):
            batch.input_ids = verify_input.draft_token
            batch.out_cache_loc = torch.tensor([5, 6], dtype=torch.int64)
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            return verify_forward_batch, True

        with (
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "torch.get_device_module",
                return_value=SimpleNamespace(current_stream=lambda: object()),
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "eagle_prepare_for_verify",
                side_effect=fake_prepare,
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "eagle_sample",
                return_value=(
                    torch.tensor([2, 10], dtype=torch.long),
                    torch.tensor([2], dtype=torch.int64),
                    torch.tensor([[0, 1]], dtype=torch.long),
                ),
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "commit_mamba_states_after_verify"
            ) as commit_mamba,
        ):
            result = VerifyWorker.verify(worker, batch)

        self.assertIsInstance(result, GenerationBatchResult)
        self.assertIs(result.logits_output, logits_output)
        self.assertEqual(result.next_token_ids.tolist(), [2, 10])
        self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [10])
        self.assertEqual(result.next_draft_input.topk_p.shape, (1, 1))
        self.assertEqual(result.next_draft_input.topk_index.shape, (1, 1))
        self.assertTrue(
            torch.equal(result.next_draft_input.topk_p, torch.zeros(1, 1))
        )
        self.assertTrue(
            torch.equal(
                result.next_draft_input.topk_index,
                torch.zeros(1, 1, dtype=torch.int64),
            )
        )
        self.assertEqual(
            result.next_draft_input.capture_hidden_mode, CaptureHiddenMode.NULL
        )
        self.assertEqual(result.num_correct_drafts, 1)
        self.assertEqual(result.num_correct_drafts_per_req_cpu, [1])
        self.assertTrue(result.can_run_cuda_graph)
        self.assertEqual(result.speculative_num_draft_tokens, 2)
        self.assertEqual(result.accept_lens.tolist(), [2])
        self.assertEqual(result.new_seq_lens.tolist(), [4])
        self.assertIs(result.routed_experts_output, routed_output)
        self.assertIs(result.indexer_topk_output, indexer_output)
        self.assertEqual(result.extra_keep_alive_refs, [verify_forward_batch])
        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertIsNone(batch.spec_info)
        self.assertEqual(req.kv_allocated_len, 2)
        self.assertEqual(len(freed), 0)
        commit_mamba.assert_called_once()

    def test_verify_worker_extend_result_uses_shape_valid_next_draft_input(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        batch_output = GenerationBatchResult(
            next_token_ids=torch.tensor([31, 32], dtype=torch.long)
        )
        target_worker = SimpleNamespace(
            forward_batch_generation=lambda batch: batch_output
        )
        worker = object.__new__(VerifyWorker)
        worker.topk = 1
        worker._target_worker = target_worker
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            is_extend_in_batch=False,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens=torch.tensor([4, 5], dtype=torch.int64),
        )

        result = VerifyWorker.forward_batch_generation(worker, batch)

        self.assertIs(result, batch_output)
        self.assertEqual(batch.capture_hidden_mode, CaptureHiddenMode.NULL)
        self.assertEqual(result.new_seq_lens.tolist(), [4, 5])
        self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [31, 32])
        self.assertEqual(result.next_draft_input.topk_p.shape, (2, 1))
        self.assertEqual(result.next_draft_input.topk_index.shape, (2, 1))
        self.assertTrue(
            torch.equal(result.next_draft_input.topk_p, torch.zeros(2, 1))
        )
        self.assertTrue(
            torch.equal(
                result.next_draft_input.topk_index,
                torch.zeros(2, 1, dtype=torch.int64),
            )
        )

    def test_decoupled_verify_factory_imports_verify_worker(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        self.assertEqual(VerifyWorker.__name__, "VerifyWorker")

    def test_verify_worker_builds_zero_step_verify_input(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        req = SimpleNamespace(
            rid="req-a",
            output_ids=[5, 7],
            origin_input_ids=[1, 2],
            decode_batch_idx=0,
        )
        batch = SimpleNamespace(
            reqs=[req],
            device=torch.device("cpu"),
            seq_lens=torch.tensor([2], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([2], dtype=torch.int64),
            seq_lens_sum=None,
            sampling_info=None,
            forward_mode=ForwardMode.DECODE,
            maybe_evict_swa=lambda: None,
            batch_size=lambda: 1,
        )
        worker = object.__new__(VerifyWorker)
        worker.topk = 1
        worker.speculative_num_steps = 0
        worker.speculative_num_draft_tokens = 1
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )

        spec_info = VerifyWorker.draft(worker, batch)

        self.assertEqual(spec_info.spec_steps, 0)
        self.assertEqual(spec_info.draft_token_num, 1)
        self.assertEqual(spec_info.capture_hidden_mode, CaptureHiddenMode.NULL)
        self.assertEqual(spec_info.draft_token.tolist(), [7])
        self.assertEqual(spec_info.retrieve_next_token.tolist(), [[-1]])
        self.assertEqual(req.decode_batch_idx, 0)

    def test_verify_worker_update_weights_delegates_to_target_only(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        calls = []
        target_worker = SimpleNamespace(
            update_weights_from_disk=lambda req: calls.append(("disk", req))
            or (True, "disk"),
            update_weights_from_ipc=lambda req: calls.append(("ipc", req))
            or (True, "ipc"),
            update_weights_from_tensor=lambda req: calls.append(("tensor", req))
            or (True, "tensor"),
        )
        worker = object.__new__(VerifyWorker)
        worker._target_worker = target_worker
        disk_req = object()
        ipc_req = object()
        tensor_req = object()

        self.assertEqual(
            VerifyWorker.update_weights_from_disk(worker, disk_req), (True, "disk")
        )
        self.assertEqual(
            VerifyWorker.update_weights_from_ipc(worker, ipc_req), (True, "ipc")
        )
        self.assertEqual(
            VerifyWorker.update_weights_from_tensor(worker, tensor_req),
            (True, "tensor"),
        )
        self.assertEqual(
            calls,
            [("disk", disk_req), ("ipc", ipc_req), ("tensor", tensor_req)],
        )

    def test_verify_worker_rejects_state_with_draft_resources(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        state = SpecRuntimeState(
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
            draft_attn_backend=object(),
            target_attn_backend=object(),
            target_graph_runner=object(),
        )
        worker = object.__new__(VerifyWorker)

        with self.assertRaisesRegex(ValueError, "must not carry draft resources"):
            VerifyWorker._validate_decoupled_runtime_state(worker, state)

    def test_verify_worker_validation_allows_candidate_state_without_budget(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.server_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=None)),
            cuda_graph_bs=None,
            cuda_graph_max_bs=None,
        )
        worker.adaptive_controller = None
        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            target_attn_backend=object(),
            target_graph_runner=SimpleNamespace(capture_bs=[128]),
        )

        VerifyWorker._validate_decoupled_runtime_state(worker, state)

    def test_verify_worker_validation_rejects_unregistered_throughput_step(self):
        from sglang.srt.speculative.decoupled_verify_throughput_controller import (
            DecoupledVerifyThroughputAwareController,
        )
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.speculative_num_steps = 1
        worker.server_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=None)),
            cuda_graph_bs=None,
            cuda_graph_max_bs=None,
        )
        worker.adaptive_controller = DecoupledVerifyThroughputAwareController(
            worker,
            candidate_steps=[0, 1],
            initial_steps=1,
        )
        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            target_attn_backend=object(),
            target_graph_runner=SimpleNamespace(capture_bs=[16]),
        )

        with self.assertRaisesRegex(RuntimeError, "not selected"):
            VerifyWorker._validate_decoupled_runtime_state(worker, state)

    def test_throughput_profile_draft_buffers_populate_nonzero_steps(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.model_config = SimpleNamespace(vocab_size=100)
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(output_ids=[1]),
                SimpleNamespace(output_ids=[1, 2]),
            ]
        )

        VerifyWorker._prepare_throughput_profile_draft_buffers(worker, batch, 3)

        self.assertEqual([len(req.draft_buffer) for req in batch.reqs], [3, 3])
        for req in batch.reqs:
            self.assertTrue(all(1 <= token_id < 100 for token_id in req.draft_buffer))

    def test_throughput_profile_capture_bs_uses_decode_graph_buckets(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.server_args = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=[8, 4, 0], max_bs=None)
            ),
            cuda_graph_bs_decode=None,
        )

        capture_bs = VerifyWorker._resolve_throughput_profile_capture_bs(worker)

        self.assertEqual(capture_bs, [4, 8])
        self.assertEqual(
            VerifyWorker._throughput_profile_padded_graph_bs(worker, 5, [4, 8]), 8
        )

    def test_throughput_profile_decode_result_applies_accept_lens(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.model_config = SimpleNamespace(vocab_size=100)
        next_draft_input = object()
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(output_ids=[1]),
                SimpleNamespace(output_ids=[1, 2]),
            ],
            spec_info=None,
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int64),
            seq_lens_sum=30,
            input_ids=object(),
        )
        result = GenerationBatchResult(
            can_run_cuda_graph=True,
            next_draft_input=next_draft_input,
            accept_lens=torch.tensor([3, 1], dtype=torch.int32),
            new_seq_lens=torch.tensor([13, 21], dtype=torch.int64),
        )

        VerifyWorker._apply_throughput_decode_result(worker, batch, result)

        self.assertIs(batch.spec_info, next_draft_input)
        self.assertIsNone(batch.input_ids)
        self.assertEqual([len(req.output_ids) for req in batch.reqs], [4, 3])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [13, 21])
        self.assertEqual(batch.seq_lens_sum, 34)

    def test_throughput_profile_capture_uses_decode_graph_bs_and_steps(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.server_args = SimpleNamespace(
            _decoupled_verify_max_speculative_steps=2,
            speculative_num_steps=1,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=[8, 4], max_bs=None)
            ),
            cuda_graph_bs_decode=None,
        )
        captured = []

        def build_state(speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs):
            captured.append(
                (
                    speculative_num_steps,
                    speculative_num_draft_tokens,
                    list(cuda_graph_bs),
                )
            )
            return SpecRuntimeState.for_decoupled_verify(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                target_attn_backend=object(),
                target_graph_runner=SimpleNamespace(capture_bs=list(cuda_graph_bs)),
            )

        worker.build_adaptive_runtime_state = build_state

        VerifyWorker._capture_throughput_profile_states(worker)

        self.assertEqual(
            captured,
            [
                (0, 1, [4, 8]),
                (1, 2, [4, 8]),
                (2, 3, [4, 8]),
            ],
        )
        self.assertEqual(
            worker._throughput_profile_bs_by_step,
            {0: [4, 8], 1: [4, 8], 2: [4, 8]},
        )

    def test_startup_throughput_profile_writes_controller_cost_table(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        worker = object.__new__(VerifyWorker)
        worker.gpu_id = 0
        worker.enable_adaptive_verify = True
        worker.speculative_num_steps = 1
        worker.speculative_num_draft_tokens = 2
        worker.server_args = SimpleNamespace(
            speculative_adaptive_strategy="throughput_aware",
            _decoupled_verify_max_speculative_steps=2,
            decoupled_verify_throughput_profile_path=None,
            decoupled_verify_throughput_profile_ctx_lens="256",
        )
        worker._throughput_profile_done = False
        worker._throughput_profile_states_by_step = {
            steps: SpecRuntimeState.for_decoupled_verify(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                target_attn_backend=object(),
                target_graph_runner=SimpleNamespace(capture_bs=[4, 8]),
            )
            for steps in (0, 1, 2)
        }
        worker._throughput_profile_capture_bs_by_step = {
            steps: [4, 8] for steps in (0, 1, 2)
        }
        worker._throughput_profile_bs_by_step = {
            steps: [4, 8] for steps in (0, 1, 2)
        }
        worker._throughput_profile_capture_bs = [4, 8]
        profile_calls = []

        def apply_runtime_state(state):
            worker.speculative_num_steps = state.speculative_num_steps
            worker.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        def profile_shape(_worker, batch_size, steps, ctx_len, state, tree_cache):
            profile_calls.append((batch_size, steps, ctx_len))
            return float(batch_size + steps + 1)

        worker.apply_runtime_state = apply_runtime_state

        with (
            patch.object(VerifyWorker, "_profile_throughput_shape", profile_shape),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.log_info_on_rank0"
            ),
        ):
            VerifyWorker.run_startup_spec_profiling(worker, tree_cache=object())

        self.assertEqual(
            profile_calls,
            [
                (4, 0, 256),
                (8, 0, 256),
                (4, 1, 256),
                (8, 1, 256),
                (4, 2, 256),
                (8, 2, 256),
            ],
        )
        self.assertEqual(worker.adaptive_controller.candidate_steps, [0, 1, 2])
        self.assertEqual(
            worker.adaptive_controller._cost_table.lookup(
                batch_size=5, steps=2, ctx_len=256
            ),
            11.0,
        )
        self.assertIn(
            "(bs=8, steps=2, ctx=256): 11.0000ms",
            worker.server_args._decoupled_verify_throughput_cost_table_summary,
        )

    def test_startup_throughput_profile_loads_matching_cache_without_measurement(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.get_device_name",
                return_value="NVIDIA H100",
            ),
        ):
            worker = self._make_throughput_profile_worker()
            fingerprint = VerifyWorker._throughput_profile_fingerprint(worker)
            self.assertNotIn("draft_model_path", fingerprint)
            self.assertNotIn("draft_tp_size", fingerprint)
            self.assertEqual(fingerprint["gpu_name"], "NVIDIA H100")
            cache_path = os.path.join(tmpdir, "any-cache-file-name.json")
            worker.server_args.decoupled_verify_throughput_profile_path = cache_path
            with open(cache_path, "w") as f:
                payload = self._throughput_profile_cache_payload(worker)
                payload["fingerprint"]["extra_ignored"] = "ok"
                payload["top_level_extra_ignored"] = "ok"
                json.dump(payload, f)

            with (
                patch.object(
                    VerifyWorker,
                    "_profile_throughput_shape",
                    side_effect=AssertionError("cache hit should not profile"),
                ),
                patch(
                    "sglang.srt.speculative.decoupled_verify_worker.log_info_on_rank0"
                ) as log_mock,
            ):
                VerifyWorker.run_startup_spec_profiling(worker, tree_cache=object())

        self.assertEqual(worker.adaptive_controller.candidate_steps, [0, 1, 2])
        self.assertEqual(
            worker.adaptive_controller._cost_table.lookup(
                batch_size=5, steps=2, ctx_len=256
            ),
            11.0,
        )
        self.assertIn(
            "(bs=8, steps=2, ctx=256): 11.0000ms",
            worker.server_args._decoupled_verify_throughput_cost_table_summary,
        )
        self.assertTrue(
            any(
                "Loaded decoupled verifier throughput-aware profile data" in call.args[1]
                for call in log_mock.call_args_list
            )
        )

    def test_startup_throughput_profile_reuses_partial_cache(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.get_device_name",
                return_value="NVIDIA H100",
            ),
        ):
            worker = self._make_throughput_profile_worker()
            cache_path = os.path.join(tmpdir, "partial.json")
            worker.server_args.decoupled_verify_throughput_profile_path = cache_path
            with open(cache_path, "w") as f:
                json.dump(
                    self._throughput_profile_cache_payload(worker, omit_last=True),
                    f,
                )

            profile_calls = []

            def profile_shape(_worker, batch_size, steps, ctx_len, state, tree_cache):
                profile_calls.append((batch_size, steps, ctx_len))
                return float(batch_size + steps + 1)

            with (
                patch.object(
                    VerifyWorker,
                    "_profile_throughput_shape",
                    profile_shape,
                ),
                patch(
                    "sglang.srt.speculative.decoupled_verify_worker."
                    "log_info_on_rank0"
                ),
            ):
                VerifyWorker.run_startup_spec_profiling(worker, tree_cache=object())

            self.assertEqual(profile_calls, [(8, 2, 256)])
            self.assertEqual(
                worker.adaptive_controller._cost_table.lookup(
                    batch_size=5, steps=2, ctx_len=256
                ),
                11.0,
            )
            with open(cache_path) as f:
                payload = json.load(f)
            self.assertNotIn("schema_version", payload)
            self.assertEqual(len(payload["costs"]), 6)

    def test_startup_throughput_profile_skips_invalid_cache_row(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.get_device_name",
                return_value="NVIDIA H100",
            ),
        ):
            worker = self._make_throughput_profile_worker()
            cache_path = os.path.join(tmpdir, "invalid-row.json")
            worker.server_args.decoupled_verify_throughput_profile_path = cache_path
            with open(cache_path, "w") as f:
                json.dump(
                    self._throughput_profile_cache_payload(worker, invalid_last=True),
                    f,
                )

            profile_calls = []

            def profile_shape(_worker, batch_size, steps, ctx_len, state, tree_cache):
                profile_calls.append((batch_size, steps, ctx_len))
                return float(batch_size + steps + 1)

            with (
                patch.object(
                    VerifyWorker,
                    "_profile_throughput_shape",
                    profile_shape,
                ),
                patch(
                    "sglang.srt.speculative.decoupled_verify_worker."
                    "log_info_on_rank0"
                ) as log_mock,
            ):
                VerifyWorker.run_startup_spec_profiling(worker, tree_cache=object())

            self.assertEqual(profile_calls, [(8, 2, 256)])
            self.assertTrue(
                any(
                    "skipped_invalid_entries=1" in call.args[1]
                    for call in log_mock.call_args_list
                )
            )
            with open(cache_path) as f:
                payload = json.load(f)
            self.assertNotIn("schema_version", payload)
            self.assertEqual(len(payload["costs"]), 6)

    def test_startup_throughput_profile_cache_misses_profile_and_rewrite(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        cases = (
            "missing",
            "malformed_json",
            "missing_fingerprint",
            "missing_costs",
            "metadata_mismatch",
            "gpu_mismatch",
        )
        for case in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory() as tmpdir,
                patch(
                    "sglang.srt.speculative.decoupled_verify_worker.get_device_name",
                    return_value="NVIDIA H100",
                ),
            ):
                worker = self._make_throughput_profile_worker()
                cache_path = os.path.join(tmpdir, f"{case}.json")
                worker.server_args.decoupled_verify_throughput_profile_path = cache_path
                if case == "malformed_json":
                    with open(cache_path, "w") as f:
                        f.write("{not-json")
                elif case == "missing_fingerprint":
                    with open(cache_path, "w") as f:
                        json.dump(
                            self._throughput_profile_cache_payload(
                                worker, include_fingerprint=False
                            ),
                            f,
                        )
                elif case == "missing_costs":
                    with open(cache_path, "w") as f:
                        json.dump(
                            self._throughput_profile_cache_payload(
                                worker, include_costs=False
                            ),
                            f,
                        )
                elif case in ("metadata_mismatch", "gpu_mismatch"):
                    fingerprint = worker._throughput_profile_fingerprint()
                    if case == "metadata_mismatch":
                        fingerprint["target_model_path"] = "/models/other-target"
                    else:
                        fingerprint["gpu_name"] = "NVIDIA A100"
                    with open(cache_path, "w") as f:
                        json.dump(
                            self._throughput_profile_cache_payload(
                                worker, fingerprint=fingerprint
                            ),
                            f,
                        )

                profile_calls = []

                def profile_shape(_worker, batch_size, steps, ctx_len, state, tree_cache):
                    profile_calls.append((batch_size, steps, ctx_len))
                    return float(batch_size + steps + 1)

                with (
                    patch.object(
                        VerifyWorker,
                        "_profile_throughput_shape",
                        profile_shape,
                    ),
                    patch(
                        "sglang.srt.speculative.decoupled_verify_worker."
                        "log_info_on_rank0"
                    ),
                ):
                    VerifyWorker.run_startup_spec_profiling(
                        worker, tree_cache=object()
                    )

                self.assertEqual(
                    profile_calls,
                    [
                        (4, 0, 256),
                        (8, 0, 256),
                        (4, 1, 256),
                        (8, 1, 256),
                        (4, 2, 256),
                        (8, 2, 256),
                    ],
                )
                with open(cache_path) as f:
                    payload = json.load(f)
                self.assertNotIn("schema_version", payload)
                self.assertEqual(payload["fingerprint"]["gpu_name"], "NVIDIA H100")
                self.assertEqual(len(payload["costs"]), 6)

    def test_zero_step_runtime_state_builds_decode_graph_runner(self):
        from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

        attn_backend = object()
        graph_runner = SimpleNamespace(capture_bs=[8], num_tokens_per_bs=1)
        server_args = SimpleNamespace(
            speculative_num_steps=8,
            speculative_num_draft_tokens=9,
            cuda_graph_bs_decode=None,
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=None)),
        )
        model_runner = SimpleNamespace(
            gpu_id=0,
            init_new_workspace=False,
            server_args=server_args,
            _get_attention_backend=lambda init_new_workspace: attn_backend,
        )
        worker = object.__new__(VerifyWorker)
        worker.device = "cuda"
        worker.server_args = server_args
        worker.speculative_num_steps = server_args.speculative_num_steps
        worker.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        worker._target_worker = SimpleNamespace(model_runner=model_runner)

        with (
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "check_cuda_graph_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "get_available_gpu_memory",
                return_value=10.0,
            ),
            patch(
                "sglang.srt.speculative.decoupled_verify_worker."
                "DecodeCudaGraphRunner",
                return_value=graph_runner,
            ) as graph_runner_cls,
            patch(
                "sglang.srt.speculative.decoupled_verify_worker.log_info_on_rank0"
            ),
        ):
            state = VerifyWorker.build_adaptive_runtime_state(
                worker,
                speculative_num_steps=0,
                speculative_num_draft_tokens=1,
                cuda_graph_bs=[8],
            )

        _, kwargs = graph_runner_cls.call_args
        self.assertIs(kwargs["attn_backend"], attn_backend)
        self.assertEqual(kwargs["speculative_num_steps"], 0)
        self.assertEqual(kwargs["speculative_num_draft_tokens"], 1)

        self.assertEqual(state.speculative_num_steps, 0)
        self.assertEqual(state.speculative_num_draft_tokens, 1)
        self.assertIs(state.target_attn_backend, attn_backend)
        self.assertIs(state.target_graph_runner, graph_runner)
        self.assertIsNone(server_args.cuda_graph_config.decode.bs)

    def test_decoupled_adaptive_allows_zero_step_candidates(self):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        cfg = {"1": {"candidate_steps": [0, 1]}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(cfg, f)
            f.flush()

            args = SimpleNamespace(
                pp_size=1,
                decoupled_spec_rank_base=0,
                page_size=1,
                max_running_requests=1,
                disable_overlap_schedule=True,
                disable_radix_cache=False,
                mamba_radix_cache_strategy="extra_buffer",
                enable_mixed_chunk=False,
                disable_piecewise_cuda_graph=False,
                speculative_algorithm="DECOUPLED_VERIFY",
                speculative_adaptive=True,
                speculative_adaptive_strategy="ema",
                speculative_adaptive_config=f.name,
                speculative_num_steps=1,
                speculative_num_draft_tokens=2,
                speculative_eagle_topk=1,
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend=Backend.DISABLED),
                    decode=SimpleNamespace(backend=Backend.FULL),
                ),
            )

            DecoupledVerifySpecAlgo.validate_server_args(args)

        self.assertEqual(args.speculative_num_draft_tokens, 2)
        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.mamba_radix_cache_strategy, "no_buffer")

    def test_draft_tail_buffer_caps_snapshot_tail_without_hiding_raw_tail(self):
        rid = "req-a"
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=4)
        try:
            buffer.apply_control_batch(
                DraftControlBatch(
                    0,
                    sync_messages=[
                        DraftSync(
                            request_id=rid,
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            prompt_token_ids=[1, 2],
                            committed_output_ids=[3],
                        )
                    ],
                )
            )
            buffer.append_draft_stream_batch(
                DraftTailStreamOutputBatch(
                    [
                        DraftTailStreamOutput(0, 0, rid, 1, 1, 10),
                        DraftTailStreamOutput(0, 0, rid, 1, 2, 11),
                        DraftTailStreamOutput(0, 0, rid, 1, 3, 12),
                    ]
                )
            )

            snapshot = buffer.get_draft_snapshots(
                [_Req(rid)],
                allow_partial=True,
                include_raw_tail_tokens=True,
                max_tail_len=1,
            )[0]

            self.assertEqual(snapshot.tail_tokens, [10])
            self.assertEqual(snapshot.raw_tail_tokens, [10, 11, 12])
            self.assertEqual(snapshot.raw_tail_len, 3)
        finally:
            buffer.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_draft_checkpoint import (
    DecoupledDraftMambaCheckpointStore,
    DraftRequestGeneration,
    draft_active_state_position,
    plan_draft_rewrite,
)
from sglang.srt.speculative.decoupled_spec_data_plane import (
    create_drafter_decoupled_spec_data_plane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifierCommitSegment,
    VerifyCommit,
    build_draft_scheduler_rid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import NextBatchPlan
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


class _DraftRequestState(msgspec.Struct):
    key: DraftRequestGeneration
    req: Any
    src_verifier_rank: int
    committed_len: int
    published_len: int
    is_sleeping: bool = False


class DecoupledDraftManager:
    """Drafter-side request, rollback, pacing, and tail-publish component."""

    def __init__(
        self,
        scheduler: Scheduler,
        config: DecoupledSpecIpcConfig,
    ) -> None:
        self.scheduler = scheduler
        self.config = config
        self.src_drafter_rank = int(config.rank)
        self.num_draft_tokens = int(scheduler.server_args.speculative_num_steps)
        self.ahead_window = 2 * self.num_draft_tokens + 1
        self.is_entry_rank = scheduler.ps.tp_rank == 0
        if scheduler.ps.tp_size != 1:
            raise ValueError("The phase-one decoupled drafter requires tp_size == 1.")
        self.data_plane = create_drafter_decoupled_spec_data_plane(config)
        self.data_plane.start()
        self._requests: dict[tuple[int, str], _DraftRequestState] = {}
        self._sleeping_requests: dict[DraftRequestGeneration, Req] = {}

        req_pool = scheduler.req_to_token_pool
        self.checkpoints = (
            DecoupledDraftMambaCheckpointStore(
                req_pool,
                max_draft_tokens=self.num_draft_tokens,
            )
            if getattr(req_pool, "mamba_pool", None) is not None
            else None
        )

    def close(self) -> None:
        try:
            if self.checkpoints is not None:
                self.checkpoints.close()
        finally:
            self.data_plane.close()

    def prepare_batch(self, batch: ScheduleBatch) -> None:
        return None

    def abort_request(self, req: Req, reason: str = "abort") -> None:
        # Drafter requests are owned by verifier controls, not HTTP clients.
        return None

    def retract_request(self, req: Req) -> None:
        """Restart a retracted drafter from its authoritative committed prefix."""

        if getattr(req, "decoupled_draft_generation", None) is None:
            return
        state = self._state_for_req(req)
        self._sleeping_requests.pop(state.key, None)
        state.is_sleeping = False
        if self.checkpoints is not None:
            self.checkpoints.release(state.key)
        if len(req.output_ids) > state.committed_len:
            req.output_ids = array("q", req.output_ids[: state.committed_len])
            req.full_untruncated_fill_ids = array("q")
            req._refresh_fill_ids()
        req.to_finish = None
        req.finished_reason = None
        req.finished_len = None
        state.published_len = state.committed_len

    def handle_retracted_requests(
        self,
        retracted_reqs: list[Req],
        reqs_to_abort: list[Req],
    ) -> tuple[list[Req], list[Req]]:
        # A drafter mirror has no client-visible abort path. Truncating its
        # speculative suffix can make the cacheless re-prefill fit again.
        retry_reqs = [*retracted_reqs, *reqs_to_abort]
        for req in retry_reqs:
            self.retract_request(req)
        return retry_reqs, []

    def prepare_pause_retract(self, reqs: list[Req]) -> list[Req]:
        """Fold sleeping mirrors into pause retraction and reset draft state."""

        unique_reqs = []
        seen_req_ids = set()
        candidates = [*reqs, *self._sleeping_requests.values()]
        for req in candidates:
            if id(req) in seen_req_ids:
                continue
            seen_req_ids.add(id(req))
            self.retract_request(req)
            unique_reqs.append(req)
        return unique_reqs

    def _build_draft_decode_batch(self, reqs: list[Req]) -> ScheduleBatch:
        """Rebuild scheduler metadata for rows that retained Req/KV ownership."""

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
            tree_cache=self.scheduler.tree_cache,
            model_config=self.scheduler.model_config,
            enable_overlap=self.scheduler.enable_overlap,
            spec_algorithm=self.scheduler.spec_algorithm,
        )
        device = self.scheduler.device
        req_pool_indices = [int(req.req_pool_idx) for req in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        seq_lens = [
            len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0) for req in reqs
        ]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        last_tokens = torch.tensor(
            [
                (
                    int(req.output_ids[-1])
                    if req.output_ids
                    else int(req.origin_input_ids[-1])
                )
                for req in reqs
            ],
            dtype=torch.int64,
            device=device,
        )
        self.scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(bonus_tokens=last_tokens),
        )
        batch.input_ids = None
        batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )
        return batch

    def _wake_sleeping_requests(self) -> None:
        wake_reqs = []
        for key, req in list(self._sleeping_requests.items()):
            state = self._requests.get((key.src_verifier_rank, key.request_id))
            if state is None or state.req is not req:
                self._sleeping_requests.pop(key, None)
                continue
            if len(req.output_ids) - state.committed_len >= self.ahead_window:
                continue
            if req.req_pool_idx is None or req.kv is None:
                raise RuntimeError(
                    "Sleeping decoupled draft request lost its KV ownership: "
                    f"request_id={key.request_id}"
                )
            state.is_sleeping = False
            self._sleeping_requests.pop(key, None)
            wake_reqs.append(req)

        if not wake_reqs:
            return
        wake_batch = self._build_draft_decode_batch(wake_reqs)
        running_batch = self.scheduler.running_batch
        if running_batch is None or running_batch.is_empty():
            self.scheduler.running_batch = wake_batch
        else:
            running_batch.merge_batch(wake_batch)
            running_batch.batch_is_full = False

    def process_pending_controls(self) -> None:
        ready = self.data_plane.collect_ready_controls(self._consumable_commit_len)
        for draft_key in ready.close_keys:
            self._close_request_key(draft_key)
        for message in ready.sync_messages:
            self._open_request(message)
        rewritten_reqs = []
        echo_outputs = []
        for segment in ready.ready_commit_segments:
            req, echo_output = self._apply_commit_segment(segment)
            if echo_output is not None:
                rewritten_reqs.append(req)
                echo_outputs.append(echo_output)
        self._refresh_active_batches(rewritten_reqs)
        if echo_outputs:
            self.data_plane.publish_tails(
                DraftTailStreamOutputBatch(outputs=echo_outputs)
            )
        self._wake_sleeping_requests()

    def adjust_plan(self, plan: NextBatchPlan) -> NextBatchPlan:
        return plan

    def sleep_overrun_requests(self, batch: ScheduleBatch) -> ScheduleBatch:
        """Transfer ahead rows before decode allocates this round's KV slots."""

        if batch is None or batch.is_empty():
            return batch
        keep_indices = []
        for index, req in enumerate(batch.reqs):
            if getattr(req, "decoupled_draft_generation", None) is None:
                keep_indices.append(index)
                continue
            state = self._state_for_req(req)
            if len(req.output_ids) - state.committed_len >= self.ahead_window:
                if state.is_sleeping:
                    raise RuntimeError(
                        "A sleeping decoupled draft request remained in running_batch: "
                        f"request_id={state.key.request_id}"
                    )
                state.is_sleeping = True
                self._sleeping_requests[state.key] = req
            else:
                keep_indices.append(index)
        if len(keep_indices) != len(batch.reqs):
            batch.filter_batch(keep_indices=keep_indices)
            batch.batch_is_full = False
        return batch

    def has_pending_work(self) -> bool:
        # Sleeping rows retain Req/KV/checkpoint ownership outside running_batch.
        # Transport controls alone do not own scheduler resources.
        return bool(self._sleeping_requests)

    def on_no_batch(self) -> bool:
        """Keep the control-driven drafter alive without entering idle cleanup."""

        if not self.has_pending_work():
            return False
        # Wake immediately on verifier control; the finite bound only lets the
        # scheduler service shutdown/health work when the peer is silent.
        self.data_plane.wait_for_control(0.01)
        return True

    def before_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return
        batch.decoupled_pre_output_lens = [len(req.output_ids) for req in batch.reqs]

    def after_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return
        outputs = []
        for req, pre_output_len in zip(batch.reqs, batch.decoupled_pre_output_lens):
            if getattr(req, "decoupled_draft_generation", None) is None:
                continue
            state = self._state_for_req(req)
            if len(req.output_ids) <= int(pre_output_len):
                continue
            if req.req_pool_idx is None:
                raise RuntimeError("Draft result lost its request pool slot.")
            if self.checkpoints is not None:
                self.checkpoints.checkpoint_after_forward(
                    state.key,
                    position=draft_active_state_position(
                        len(req.origin_input_ids), len(req.output_ids)
                    ),
                    active_slot=req.mamba_pool_idx,
                )
            for output_position in range(int(pre_output_len), len(req.output_ids)):
                outputs.append(
                    DraftTailStreamOutput(
                        src_drafter_rank=self.src_drafter_rank,
                        dst_verifier_rank=state.src_verifier_rank,
                        request_id=state.key.request_id,
                        base_committed_len=state.committed_len,
                        new_token_pos=output_position,
                        new_token=int(req.output_ids[output_position]),
                    )
                )
            state.published_len = len(req.output_ids)
        if outputs:
            self.data_plane.publish_tails(DraftTailStreamOutputBatch(outputs=outputs))
        self.process_pending_controls()

    def _open_request(self, message: DraftSync) -> None:
        table_key = (int(message.src_verifier_rank), str(message.request_id))
        if table_key in self._requests:
            raise RuntimeError(
                f"DraftSync reopened a live request: request={table_key}"
            )
        sampling_params = SamplingParams(
            max_new_tokens=1 << 30,
            temperature=0.0,
            top_k=1,
            ignore_eos=True,
        )
        sampling_params.normalize(self.scheduler.tokenizer)
        sampling_params.verify(self.scheduler.model_config.vocab_size)
        req = Req(
            build_draft_scheduler_rid(message.draft_key),
            "",
            array("q", [int(token) for token in message.prompt_token_ids]),
            sampling_params,
            return_logprob=False,
            stream=False,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            vocab_size=self.scheduler.model_config.vocab_size,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.server_args.enable_metrics
                else None
            ),
        )
        req.tokenizer = self.scheduler.tokenizer
        req.output_ids = array("q", [int(token) for token in message.committed_outputs])
        req._refresh_fill_ids()
        self.scheduler.init_req_max_new_tokens(req)
        generation_key = DraftRequestGeneration(
            src_verifier_rank=int(message.src_verifier_rank),
            request_id=str(message.request_id),
            generation=0,
        )
        req.decoupled_draft_generation = generation_key
        state = _DraftRequestState(
            key=generation_key,
            req=req,
            src_verifier_rank=int(message.src_verifier_rank),
            committed_len=len(req.output_ids),
            published_len=len(req.output_ids),
        )
        self._requests[table_key] = state
        self.scheduler._add_request_to_queue(req)

    def _close_request(self, message: DraftClose) -> None:
        self._close_request_key(message.draft_key)

    def _close_request_key(self, draft_key: DraftReqKey) -> None:
        table_key = (int(draft_key.src_verifier_rank), str(draft_key.request_id))
        state = self._requests.pop(table_key, None)
        if state is None:
            return
        req = state.req
        self._sleeping_requests.pop(state.key, None)
        state.is_sleeping = False
        # A close owns the request lifecycle, including a partially scheduled
        # prefill. Drop the scheduler's chunk owner before releasing any request
        # resources so the next scheduling step cannot stash or reschedule it.
        if self.scheduler.chunked_req is req:
            self.scheduler.chunked_req = None
        if self.scheduler._pending_chunked_abort_req is req:
            self.scheduler._pending_chunked_abort_req = None
        self.scheduler.waiting_queue = [
            queued for queued in self.scheduler.waiting_queue if queued is not req
        ]
        seen_batches = set()
        for batch in (
            self.scheduler.running_batch,
            self.scheduler.last_batch,
            self.scheduler.cur_batch_for_debug,
        ):
            if batch is None or id(batch) in seen_batches or batch.is_empty():
                continue
            seen_batches.add(id(batch))
            keep_indices = [
                index
                for index, batch_req in enumerate(batch.reqs)
                if batch_req is not req
            ]
            if len(keep_indices) != len(batch.reqs):
                batch.filter_batch(keep_indices=keep_indices)
                batch.batch_is_full = False
        if self.checkpoints is not None:
            self.checkpoints.release(state.key)
        if req.req_pool_idx is not None or self.scheduler.tree_cache.supports_mamba():
            release_kv_cache(req, self.scheduler.tree_cache, is_insert=False)

    def _consumable_commit_len(self, segment: VerifierCommitSegment) -> int:
        state = self._requests.get(
            (segment.draft_key.src_verifier_rank, segment.draft_key.request_id)
        )
        if state is None or state.req.req_pool_idx is None:
            return 0
        if int(segment.pre_verify_committed_len) != state.committed_len:
            raise RuntimeError(
                "Verifier commit segment does not start at the drafter cursor: "
                f"request_id={segment.draft_key.request_id} "
                f"expected={state.committed_len} "
                f"actual={segment.pre_verify_committed_len}"
            )
        available = len(state.req.output_ids) - state.committed_len
        if available <= 0:
            return 0
        probe_len = min(available, len(segment.committed_tokens))
        for index in range(probe_len):
            if int(state.req.output_ids[state.committed_len + index]) != int(
                segment.committed_tokens[index]
            ):
                return index + 1
        return probe_len

    def _apply_commit_segment(
        self, segment: VerifierCommitSegment
    ) -> tuple[Req, DraftTailStreamOutput | None]:
        state = self._requests.get(
            (segment.draft_key.src_verifier_rank, segment.draft_key.request_id)
        )
        if state is None:
            raise RuntimeError(
                "Ready verifier commit segment has no live draft request: "
                f"draft_key={segment.draft_key}"
            )
        echo_output = self._apply_commit(
            state,
            VerifyCommit(
                request_id=segment.draft_key.request_id,
                src_verifier_rank=segment.draft_key.src_verifier_rank,
                dst_drafter_rank=segment.dst_drafter_rank,
                pre_verify_committed_len=segment.pre_verify_committed_len,
                committed_tokens=list(segment.committed_tokens),
            ),
        )
        return state.req, echo_output

    def _apply_commit(
        self,
        state: _DraftRequestState,
        message: VerifyCommit,
    ) -> DraftTailStreamOutput | None:
        req = state.req
        if int(message.pre_verify_committed_len) != state.committed_len:
            raise RuntimeError(
                "Verifier commit does not start at the drafter committed cursor: "
                f"request_id={message.request_id} "
                f"expected={state.committed_len} "
                f"actual={message.pre_verify_committed_len}"
            )
        if req.kv is None:
            raise RuntimeError("Verifier commit arrived before draft KV allocation.")
        plan = plan_draft_rewrite(
            prompt_len=len(req.origin_input_ids),
            current_output_tokens=req.output_ids,
            pre_verify_committed_len=message.pre_verify_committed_len,
            commit_tokens=message.committed_tokens,
            kv_committed_len=req.kv_committed_len,
            kv_allocated_len=req.kv.kv_allocated_len,
        )
        if plan.needs_rewrite:
            if plan.replay_tokens:
                raise RuntimeError(
                    "Phase-one F=1 drafter rewrite unexpectedly requires "
                    f"multi-token replay: replay_tokens={plan.replay_tokens}"
                )
            if self.checkpoints is None:
                raise RuntimeError(
                    "Draft rewrite requires a recurrent-state checkpoint store."
                )
            self.checkpoints.restore_for_rewrite(
                state.key,
                position=int(plan.state_restore_position),
                active_slot=req.mamba_pool_idx,
            )
            self._truncate_kv(req, int(plan.kv_keep_len))
            req.output_ids = array("q", plan.new_output_tokens)
            req.full_untruncated_fill_ids = array("q")
            req._refresh_fill_ids()
            req.finished_reason = None
            req.finished_len = None
            mismatch_index = int(plan.num_match_tokens)
            echo_position = int(message.pre_verify_committed_len) + mismatch_index
            echo_output = DraftTailStreamOutput(
                src_drafter_rank=self.src_drafter_rank,
                dst_verifier_rank=state.src_verifier_rank,
                request_id=state.key.request_id,
                base_committed_len=echo_position,
                new_token_pos=echo_position,
                new_token=int(message.committed_tokens[mismatch_index]),
            )
        else:
            echo_output = None

        state.committed_len = int(plan.new_committed_len)
        if self.checkpoints is not None:
            self.checkpoints.prune(
                state.key,
                min_position=len(req.origin_input_ids) + state.committed_len,
            )
        return echo_output

    def _truncate_kv(self, req: Req, keep_len: int) -> None:
        old_allocated_len = int(req.kv.kv_allocated_len)
        if keep_len < old_allocated_len:
            row = self.scheduler.req_to_token_pool.req_to_token[req.req_pool_idx]
            free_indices = row[keep_len:old_allocated_len].clone()
            row[keep_len:old_allocated_len].zero_()
            self.scheduler.token_to_kv_pool_allocator.free_segment(
                free_indices,
                start_pos=keep_len,
            )
        req.kv_committed_len = min(int(req.kv_committed_len), keep_len)
        req.kv.kv_allocated_len = min(old_allocated_len, keep_len)
        req.cache_protected_len = min(int(req.cache_protected_len), keep_len)

    def _refresh_active_batches(self, reqs: list[Req]) -> None:
        if not reqs:
            return
        req_ids = {id(req) for req in reqs}
        seen_batches = set()
        for batch in (
            self.scheduler.running_batch,
            self.scheduler.last_batch,
            self.scheduler.cur_batch_for_debug,
        ):
            if batch is None or id(batch) in seen_batches or batch.is_empty():
                continue
            seen_batches.add(id(batch))
            batch_indices = [
                index for index, req in enumerate(batch.reqs) if id(req) in req_ids
            ]
            if not batch_indices:
                continue
            if batch.seq_lens_cpu is None:
                raise RuntimeError(
                    "Decoupled draft rewrite requires active CPU sequence lengths."
                )
            seq_lens = [
                len(batch.reqs[index].origin_input_ids)
                + len(batch.reqs[index].output_ids)
                - 1
                for index in batch_indices
            ]
            batch.seq_lens_cpu[batch_indices] = torch.tensor(
                seq_lens, dtype=torch.int64
            )
            indices_device = torch.tensor(
                batch_indices, dtype=torch.int64, device=batch.seq_lens.device
            )
            seq_lens_device = torch.tensor(
                seq_lens, dtype=torch.int64, device=batch.seq_lens.device
            )
            batch.seq_lens[indices_device] = seq_lens_device
            batch.orig_seq_lens[indices_device] = seq_lens_device.to(torch.int32)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            batch.input_ids = None

        live_reqs = [req for req in reqs if req.req_pool_idx is not None]
        if live_reqs:
            pool_indices = torch.tensor(
                [int(req.req_pool_idx) for req in live_reqs],
                dtype=torch.int64,
                device=self.scheduler.device,
            )
            tail_tokens = torch.tensor(
                [int(req.output_ids[-1]) for req in live_reqs],
                dtype=torch.int64,
                device=self.scheduler.device,
            )
            self.scheduler.future_map.output_tokens_buf[pool_indices] = tail_tokens

    def _state_for_req(self, req: Req) -> _DraftRequestState:
        key = getattr(req, "decoupled_draft_generation", None)
        if key is None:
            raise RuntimeError(f"Request {req.rid} is not a decoupled draft request.")
        state = self._requests.get((key.src_verifier_rank, key.request_id))
        if state is None or state.req is not req:
            raise RuntimeError(f"Missing decoupled draft state for request {req.rid}.")
        return state

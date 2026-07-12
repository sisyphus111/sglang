from __future__ import annotations

from contextlib import suppress
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlInbox,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
    VerifierCommitSegment,
    build_draft_scheduler_rid,
    parse_draft_scheduler_rid,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    get_decoupled_spec_transport,
)
from sglang.srt.speculative.decoupled_draft_mamba import (
    DecoupledDraftMambaStateManager,
)
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

@dataclass
class DraftReqState:
    key: DraftReqKey
    req: Optional[Req] = None
    verifier_committed_prefix_len: int = 0
    is_sleeping: bool = False
    mamba_checkpoint_positions: set[int] = field(default_factory=set)
    mamba_checkpoint_slots: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DraftKVTruncation:
    req_pool_idx: int
    kv_start: int
    kv_end: int


@dataclass(frozen=True)
class DraftBatchMetadataUpdate:
    req_batch_idx: int
    new_seq_len: int
    new_tail_token_id: int


@triton.jit
def _flush_draft_batch_metadata_updates_kernel(
    metadata_ptr,
    seq_lens_ptr,
    orig_seq_lens_ptr,
    req_pool_indices_ptr,
    future_output_tokens_ptr,
    num_updates,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_updates

    req_batch_indices = tl.load(metadata_ptr + offsets, mask=mask, other=0)
    new_seq_lens = tl.load(
        metadata_ptr + num_updates + offsets, mask=mask, other=0
    )
    new_tail_token_ids = tl.load(
        metadata_ptr + 2 * num_updates + offsets, mask=mask, other=0
    )
    req_pool_indices = tl.load(
        req_pool_indices_ptr + req_batch_indices, mask=mask, other=0
    )

    tl.store(seq_lens_ptr + req_batch_indices, new_seq_lens, mask=mask)
    tl.store(
        orig_seq_lens_ptr + req_batch_indices,
        new_seq_lens.to(tl.int32),
        mask=mask,
    )
    tl.store(
        future_output_tokens_ptr + req_pool_indices,
        new_tail_token_ids,
        mask=mask,
    )



class SchedulerDecoupledDraftMixin:
    """Drafter-side scheduler hooks for decoupled speculation."""

    def start_token_sync_thread(self: Scheduler) -> None:
        self.decoupled_draft_mamba = DecoupledDraftMambaStateManager(self)
        self.token_sync_thread = None
        if not self.is_draft_entry_rank():
            return
        transport = get_decoupled_spec_transport()
        self.token_sync_thread = transport.token_sync_thread_cls(
            context=self.ipc_channels.context,
            drafter_rank=self.get_decoupled_spec_rank(),
        )
        self.token_sync_thread.start()

    def flush_draft_updates(
        self: Scheduler,
        batch: ScheduleBatch,
        req_indices: Optional[list[int]] = None,
    ) -> Optional[DraftTailStreamOutputBatch]:
        if not self.is_draft_worker_batch(batch):
            return None
        if req_indices is None:
            emit_candidate_indices = list(range(len(batch.reqs)))
        else:
            emit_candidate_indices = list(req_indices)
        if get_decoupled_spec_transport().supports_native_rows:
            stream_output_rows: list[tuple] = []
            src_drafter_rank = self.get_decoupled_spec_rank()
            for req_batch_idx in emit_candidate_indices:
                if not (0 <= req_batch_idx < len(batch.reqs)):
                    continue
                req = batch.reqs[req_batch_idx]
                if not req.output_ids:
                    continue
                state = self._get_draft_state_by_req(req)
                token_pos = len(req.output_ids) - 1
                token_id = int(req.output_ids[-1])
                committed_len = int(state.verifier_committed_prefix_len)
                if token_pos < committed_len:
                    continue
                stream_output_rows.append(
                    (
                        src_drafter_rank,
                        int(state.key.src_verifier_rank),
                        state.key.request_id,
                        committed_len,
                        token_pos,
                        token_id,
                    )
                )

            if stream_output_rows and self.is_draft_entry_rank():
                submit_rows = getattr(
                    self._get_token_sync_thread(), "submit_draft_result_rows", None
                )
                if submit_rows is None:
                    raise RuntimeError(
                        "C++ decoupled token sync thread does not expose row API"
                    )
                submit_rows(stream_output_rows)
            return None

        stream_output_batch = DraftTailStreamOutputBatch()
        src_drafter_rank = self.get_decoupled_spec_rank()
        for req_batch_idx in emit_candidate_indices:
            if not (0 <= req_batch_idx < len(batch.reqs)):
                continue
            req = batch.reqs[req_batch_idx]
            if not req.output_ids:
                continue
            state = self._get_draft_state_by_req(req)
            token_pos = len(req.output_ids) - 1
            token_id = int(req.output_ids[-1])
            committed_len = int(state.verifier_committed_prefix_len)
            if token_pos < committed_len:
                continue
            stream_output_batch.outputs.append(
                DraftTailStreamOutput(
                    src_drafter_rank=src_drafter_rank,
                    dst_verifier_rank=int(state.key.src_verifier_rank),
                    request_id=state.key.request_id,
                    base_committed_len=committed_len,
                    new_token_pos=token_pos,
                    new_token_id=token_id,
                )
            )

        if stream_output_batch.outputs and self.is_draft_entry_rank():
            self._get_token_sync_thread().submit_draft_results(stream_output_batch)
        return stream_output_batch

    def _broadcast_ready_draft_controls(
        self: Scheduler,
        ready_controls: ReadyDraftControls | None,
    ) -> ReadyDraftControls:
        """
        Broadcast ready draft controls among all ranks:
        DraftSync: build a new draft request based on its prompt token_ids
        VerifierCommitSegment: apply the verifier-committed segment and
        truncate suffix if needed
        """
        def broadcast_ready_controls(rank, group, src) -> ReadyDraftControls | None:
            payload = (
                [ready_controls]
                if ready_controls is not None and rank == src
                else []
            )
            payload = broadcast_pyobj(payload, rank, group, src=src)
            if not payload:
                return None
            if len(payload) != 1:
                raise RuntimeError(
                    "Expected a single ReadyDraftControls payload, "
                    f"got {len(payload)}"
                )
            return payload[0]

        if getattr(self.server_args, "enable_dp_attention", False):
            if self.ps.attn_tp_size != 1:
                ready_controls = broadcast_ready_controls(
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.ps.attn_cp_size != 1:
                ready_controls = broadcast_ready_controls(
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )
            return (
                ready_controls
                if ready_controls is not None
                else ReadyDraftControls()
            )

        if self.ps.tp_size != 1:
            ready_controls = broadcast_ready_controls(
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return (
            ready_controls
            if ready_controls is not None
            else ReadyDraftControls()
        )

    def _get_token_sync_thread(self: Scheduler):
        token_sync_thread = self.token_sync_thread
        if token_sync_thread is None:
            raise RuntimeError("Decoupled draft entry rank has no token sync thread")
        return token_sync_thread

    def _get_or_create_draft_state(
        self: Scheduler,
        draft_key: DraftReqKey,
    ) -> DraftReqState:
        state = self.draft_req_table.get(draft_key)
        if state is None:
            state = DraftReqState(key=draft_key)
            self.draft_req_table[draft_key] = state
        return state

    def _get_draft_state_by_req(self: Scheduler, req: Req) -> DraftReqState:
        draft_key = parse_draft_scheduler_rid(req.rid)
        state = self.draft_req_table.get(draft_key)
        if state is None:
            raise RuntimeError(
                "Decoupled draft request has no scheduler state: "
                f"rid={req.rid} draft_key={draft_key}"
            )
        if state.req is not req:
            raise RuntimeError(
                "Decoupled draft scheduler state points to a different request: "
                f"rid={req.rid} draft_key={draft_key}"
            )
        return state

    def commit_draft_mamba_ckpts(
        self: Scheduler,
        batch: ScheduleBatch,
        req_indices: Optional[list[int]] = None,
    ) -> None:
        self.decoupled_draft_mamba.commit(batch, req_indices)

    def prepare_draft_mamba_routing(self: Scheduler, batch: ScheduleBatch) -> None:
        self.decoupled_draft_mamba.prepare_routing(batch)

    def _flush_draft_kv_truncations(
        self: Scheduler,
        kv_truncations: list[DraftKVTruncation],
    ) -> None:
        if not kv_truncations:
            return

        indices_to_free: list[torch.Tensor] = []
        req_to_token = self.req_to_token_pool.req_to_token
        for truncation in kv_truncations:
            if truncation.kv_start >= truncation.kv_end:
                continue
            kv_slice = req_to_token[
                truncation.req_pool_idx, truncation.kv_start : truncation.kv_end
            ]
            if len(kv_slice) > 0:
                indices_to_free.append(kv_slice.clone())
                kv_slice.zero_()

        if indices_to_free:
            self.token_to_kv_pool_allocator.free(torch.cat(indices_to_free))
        kv_truncations.clear()

    def _flush_draft_batch_metadata_updates(
        self: Scheduler,
        batch_metadata_updates: list[DraftBatchMetadataUpdate],
    ) -> None:
        if not batch_metadata_updates:
            return

        batch = self.running_batch
        if batch is None or batch.is_empty():
            raise RuntimeError(
                "Decoupled draft batch metadata update requires a non-empty "
                "running_batch. Verifier commit segment metadata updates should "
                "only be queued for requests in running_batch."
            )
        if (
            batch.seq_lens_cpu is None
            or batch.seq_lens is None
            or batch.orig_seq_lens is None
        ):
            raise RuntimeError(
                "Decoupled draft batch metadata update requires complete "
                "running_batch metadata: seq_lens_cpu, seq_lens, "
                "and orig_seq_lens must be set."
            )
        if batch.seq_lens_sum is None:
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())

        req_batch_idx_set = {update.req_batch_idx for update in batch_metadata_updates}
        if len(req_batch_idx_set) != len(batch_metadata_updates):
            raise RuntimeError(
                "Decoupled draft batch metadata update received duplicate batch "
                "indices in one flush. This indicates multiple verifier commit "
                "segment rewrites for the same in-flight request."
            )

        device = batch.seq_lens.device
        if device.type != "cuda":
            raise RuntimeError(
                "Decoupled draft batch metadata update requires CUDA because "
                "metadata flush is implemented by a Triton kernel."
            )

        num_updates = len(batch_metadata_updates)
        req_batch_indices = []
        new_seq_lens = []
        new_tail_token_ids = []
        seq_lens_delta = 0
        seq_lens_cpu_np = batch.seq_lens_cpu.numpy()
        for update in batch_metadata_updates:
            req_batch_idx = int(update.req_batch_idx)
            new_seq_len = int(update.new_seq_len)
            old_seq_len = int(seq_lens_cpu_np[req_batch_idx])

            req_batch_indices.append(req_batch_idx)
            new_seq_lens.append(new_seq_len)
            new_tail_token_ids.append(int(update.new_tail_token_id))
            seq_lens_delta += new_seq_len - old_seq_len

        metadata_cpu = torch.tensor(
            [req_batch_indices, new_seq_lens, new_tail_token_ids],
            dtype=torch.int64,
            pin_memory=True,
        )

        for req_batch_idx, new_seq_len in zip(req_batch_indices, new_seq_lens):
            seq_lens_cpu_np[req_batch_idx] = new_seq_len

        metadata_device = metadata_cpu.to(device=device, non_blocking=True)

        block_size = 1024
        _flush_draft_batch_metadata_updates_kernel[
            (triton.cdiv(num_updates, block_size),)
        ](
            metadata_device,
            batch.seq_lens,
            batch.orig_seq_lens,
            batch.req_pool_indices,
            self.future_map.output_tokens_buf,
            num_updates,
            BLOCK_SIZE=block_size,
        )

        batch.seq_lens_sum += seq_lens_delta

        batch_metadata_updates.clear()

    def apply_verifier_commit_segment(
        self: Scheduler,
        req: Req,
        segment: VerifierCommitSegment,
        *,
        req_batch_idx: Optional[int] = None,
        kv_truncations: list[DraftKVTruncation],
        batch_metadata_updates: list[DraftBatchMetadataUpdate],
    ) -> None:
        """
        Apply the verifier-committed output segment to the draft request.

        This path handles already-materialized matching segments and a single
        divergent verifier token. Callers must split verifier commit segments
        before applying them so mismatched suffixes are committed incrementally.
        """
        state = self._get_draft_state_by_req(req)
        if state.key != segment.draft_key:
            raise RuntimeError(
                "VerifierCommitSegment arrived for a mismatched draft request: "
                f"req_rid={req.rid} req_draft_key={state.key} "
                f"segment_draft_key={segment.draft_key}"
            )
        pre_verify_committed_len = int(segment.pre_verify_committed_len)
        committed_token_ids = [
            int(token_id) for token_id in segment.committed_token_ids
        ]
        if not committed_token_ids:
            raise ValueError(
                "VerifierCommitSegment committed_token_ids must be non-empty: "
                f"request_id={segment.draft_key.request_id} "
                f"pre_verify_committed_len={pre_verify_committed_len}"
            )
        committed_segment_len = len(committed_token_ids)
        current_committed_len = int(state.verifier_committed_prefix_len)
        new_committed_len = pre_verify_committed_len + committed_segment_len
        output_len = len(req.output_ids)
        prompt_len = len(req.origin_input_ids)
        materialized_kv_len = prompt_len + max(output_len - 1, 0)

        if new_committed_len <= current_committed_len:
            raise RuntimeError(
                "VerifierCommitSegment must advance the drafter committed prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"committed_segment_len={committed_segment_len} "
                f"current_committed_len={current_committed_len} "
                f"new_committed_len={new_committed_len}"
            )

        if pre_verify_committed_len > current_committed_len:
            raise RuntimeError(
                "VerifierCommitSegment depends on a prefix the drafter has not "
                "committed: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"current_committed_len={current_committed_len}"
            )

        if req.kv_committed_freed:
            raise RuntimeError(
                "Decoupled draft verify commit found freed KV cache: "
                f"request_id={state.key.request_id} "
                f"new_committed_len={new_committed_len} "
                f"output_len={output_len} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        if (
            req.kv_committed_len < materialized_kv_len
            or req.kv_allocated_len < materialized_kv_len
        ):
            raise RuntimeError(
                "Decoupled draft KV prefix is shorter than the materialized "
                "output prefix: "
                f"request_id={state.key.request_id} "
                f"new_committed_len={new_committed_len} "
                f"output_len={output_len} "
                f"prompt_len={prompt_len} "
                f"materialized_kv_len={materialized_kv_len} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        matched_segment_len = 0
        max_possible_match_len = min(
            committed_segment_len,
            max(0, output_len - pre_verify_committed_len),
        )
        while (
            matched_segment_len < max_possible_match_len
            and int(req.output_ids[pre_verify_committed_len + matched_segment_len])
            == committed_token_ids[matched_segment_len]
        ):
            matched_segment_len += 1

        if matched_segment_len == committed_segment_len:
            # all committed_tokens match the drafter's output, simply advance the committed prefix
            state.verifier_committed_prefix_len = new_committed_len
            self.decoupled_draft_mamba.prune(state)
            return

        remaining_committed_token_ids = committed_token_ids[matched_segment_len:]
        if len(remaining_committed_token_ids) > 1:
            raise RuntimeError(
                "VerifierCommitSegment committed_token_ids contain a multi-token "
                "mismatched verifier segment. Split the segment before applying: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"matched_segment_len={matched_segment_len} "
                f"committed_token_ids={committed_token_ids} "
                f"draft_segment={req.output_ids[pre_verify_committed_len:new_committed_len]}"
            )

        committed_token_pos = pre_verify_committed_len + matched_segment_len
        committed_token_id = int(remaining_committed_token_ids[0])
        if committed_token_pos < current_committed_len:
            raise RuntimeError(
                "VerifierCommitSegment conflicts with the already committed "
                "drafter prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"committed_token_pos={committed_token_pos} "
                f"current_committed_len={current_committed_len} "
                f"committed_token_ids={committed_token_ids}"
            )
        if committed_token_pos >= output_len:
            raise RuntimeError(
                "VerifierCommitSegment cannot skip a non-materialized drafter "
                "output gap. Keep the segment pending until the drafter "
                "materializes the prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"committed_token_pos={committed_token_pos} "
                f"output_len={output_len} "
                f"committed_token_ids={committed_token_ids}"
            )

        # The verifier-selected token replaces the drafter suffix starting at
        # `committed_token_pos`.
        #
        # Positions here are in req.output_ids, not in the full prompt+output
        # sequence. The kept output range is [0, truncate_from), and the removed
        # output range is [truncate_from, len(req.output_ids)). In other words,
        # `truncate_from` itself is removed. After the removal, committed_token_id
        # is appended at exactly that position.
        truncate_from = committed_token_pos

        # Number of output tokens removed from the drafter suffix:
        # len(req.output_ids[truncate_from:]).
        removed = output_len - truncate_from

        # KV positions are in the full sequence coordinate system:
        # [0, prompt_len) are prompt tokens, and output_ids[i] corresponds to
        # full-sequence position prompt_len + i. Therefore the KV entries to
        # discard start at `kv_truncate_from`, inclusive.
        kv_truncate_from = prompt_len + truncate_from

        if kv_truncate_from > min(req.kv_committed_len, req.kv_allocated_len):
            raise RuntimeError(
                "Decoupled draft cannot truncate beyond materialized KV prefix: "
                f"request_id={state.key.request_id} "
                f"committed_token_pos={committed_token_pos} "
                f"output_len={output_len} "
                f"prompt_len={prompt_len} "
                f"kv_truncate_from={kv_truncate_from} "
                f"kv_committed_len={req.kv_committed_len} "
                f"kv_allocated_len={req.kv_allocated_len}"
            )

        if isinstance(self.req_to_token_pool, HybridReqToTokenPool):
            # check the committed_token_pos's mamba ckpt exiests
            self.decoupled_draft_mamba.checkpoint_slot(
                state, committed_token_pos, for_write=False
            )

        if req.grammar is not None:
            with suppress(Exception):
                req.grammar.rollback(removed)

        if req.req_pool_idx is not None and not req.kv_committed_freed:
            # Only free KV slots that are currently allocated for this req.
            # `trimmed_end` is exclusive. The freed full-sequence KV range is
            # [kv_truncate_from, trimmed_end). If kv_truncate_from ==
            # trimmed_end, there is nothing to free.
            trimmed_end = min(
                req.kv_allocated_len, prompt_len + len(req.output_ids)
            )
            if kv_truncate_from < trimmed_end:
                kv_truncations.append(
                    DraftKVTruncation(
                        req_pool_idx=int(req.req_pool_idx),
                        kv_start=kv_truncate_from,
                        kv_end=trimmed_end,
                    )
                )
            req.kv_committed_len = min(req.kv_committed_len, kv_truncate_from)
            req.kv_allocated_len = min(req.kv_allocated_len, kv_truncate_from)
            req.cache_protected_len = min(req.cache_protected_len, kv_truncate_from)

        # Truncate per-output arrays with the same output-index interval:
        # delete [truncate_from, old_output_len).
        del req.output_ids[truncate_from:]
        if req.return_logprob:
            del req.logprob.output_token_logprobs_val[truncate_from:]
            del req.logprob.output_token_logprobs_idx[truncate_from:]
            del req.logprob.output_top_logprobs_val[truncate_from:]
            del req.logprob.output_top_logprobs_idx[truncate_from:]
            del req.logprob.output_token_ids_logprobs_val[truncate_from:]
            del req.logprob.output_token_ids_logprobs_idx[truncate_from:]
        if req.hidden_states:
            del req.hidden_states[truncate_from:]

        req.output_ids.append(committed_token_id)
        if req.grammar is not None:
            with suppress(Exception):
                req.grammar.accept_token(committed_token_id)
        req.finished_reason = None
        req.finished_len = None
        req.finished_output = None
        req.to_finish = None
        req.decoded_text = ""

        if len(req.output_ids) != new_committed_len:
            raise RuntimeError(
                "Decoupled draft verify commit produced an unexpected output "
                "length: "
                f"request_id={state.key.request_id} "
                f"expected_output_len={new_committed_len} "
                f"actual_output_len={len(req.output_ids)} "
                f"committed_token_pos={committed_token_pos}"
            )
        if int(req.output_ids[-1]) != committed_token_id:
            raise RuntimeError(
                "Decoupled draft verify commit failed to install committed token: "
                f"request_id={state.key.request_id} "
                f"committed_token_pos={committed_token_pos} "
                f"committed_token_id={committed_token_id} "
                f"tail_token_id={int(req.output_ids[-1])}"
            )

        state.verifier_committed_prefix_len = new_committed_len
        self.decoupled_draft_mamba.prune(state)

        if req_batch_idx is not None:
            batch = self.running_batch
            if batch is None or batch.is_empty():
                raise RuntimeError(
                    "Decoupled draft verify commit received a running batch "
                    "index, but running_batch is empty: "
                    f"request_id={state.key.request_id} "
                    f"req_batch_idx={req_batch_idx}"
                )
            if not (0 <= req_batch_idx < len(batch.reqs)):
                raise RuntimeError(
                    "Decoupled draft verify commit received an invalid batch "
                    "index: "
                    f"request_id={state.key.request_id} "
                    f"req_batch_idx={req_batch_idx} "
                    f"batch_size={len(batch.reqs)}"
                )
            if batch.reqs[req_batch_idx] is not req:
                raise RuntimeError(
                    "Decoupled draft verify commit batch index points to a "
                    "different request: "
                    f"request_id={state.key.request_id} "
                    f"req_batch_idx={req_batch_idx} "
                    f"batch_req_rid={batch.reqs[req_batch_idx].rid}"
                )
            # Keep the in-flight decode batch consistent with the rewritten request
            # state. This block is only needed when the verifier committed token
            # changed req.output_ids above: either an existing suffix was truncated
            # and replaced, or a committed token was appended at the current tail.
            #
            # Decode seq_len is the number of tokens **already present in KV** before the
            # next tail token is consumed. For a drafter request, output_ids[-1] is the
            # current tail token used as the next decode input, so the KV-backed prefix
            # is origin_input_ids plus output_ids[0:-1]. The slice [0, -1) excludes the
            # tail token itself, hence len(origin_input_ids) + max(len(output_ids)-1, 0).
            new_seq_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

            # Prefer the last output token as the next decode input. If no output
            # token exists yet, fall back to the last prompt token. In both cases
            # the selected token is included as the decode input via future_map,
            # but excluded from new_seq_len above.
            if req.output_ids:
                new_tail_token_id = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_tail_token_id = int(req.origin_input_ids[-1])
            else:
                raise AssertionError(
                    f"Draft request {req.rid} has no token to decode from"
                )

            if new_seq_len > min(req.kv_committed_len, req.kv_allocated_len):
                raise RuntimeError(
                    "Decoupled draft batch seq_len points beyond materialized KV "
                    "after verify commit: "
                    f"request_id={state.key.request_id} "
                    f"new_seq_len={new_seq_len} "
                    f"kv_committed_len={req.kv_committed_len} "
                    f"kv_allocated_len={req.kv_allocated_len}"
                )
            batch_metadata_updates.append(
                DraftBatchMetadataUpdate(
                    req_batch_idx=req_batch_idx,
                    new_seq_len=new_seq_len,
                    new_tail_token_id=new_tail_token_id,
                )
            )

    def release_draft_request(self: Scheduler, req: Req) -> None:
        """
        release a draft request only when it has completed at the verifier side:
        1. evict the req from waiting_queue or running_batch
        2. remove the req from the draft request table
        3. release its kvcache
        """
        state = self._get_draft_state_by_req(req)

        # remove the req from waiting_queue, running_batch, and sleeping table
        self.waiting_queue = [
            queued_req for queued_req in self.waiting_queue if queued_req is not req
        ]
        for batch in (self.running_batch, self.last_batch, self.cur_batch):
            if batch is None or batch.is_empty():
                continue
            keep_indices = [
                i for i, batch_req in enumerate(batch.reqs) if batch_req is not req
            ]
            if len(keep_indices) != len(batch.reqs):
                batch.filter_batch(keep_indices=keep_indices)
                batch.batch_is_full = False
        self.draft_sleeping_reqs.pop(state.key, None)
        self.decoupled_draft_mamba.release(state)
        self.draft_req_table.pop(state.key, None)
        if req.req_pool_idx is not None or self.tree_cache.supports_mamba():
            release_kv_cache(req, self.tree_cache, is_insert=False)

    def _create_draft_request(
        self: Scheduler,
        message: DraftSync,
    ) -> Req:
        """
        Create and register a new drafter-side request from DraftSync.
        """
        state = self._get_or_create_draft_state(message.draft_key)
        if state.req is not None:
            raise RuntimeError(
                "Received DraftSync for an existing decoupled draft request: "
                f"request_id={message.request_id}"
            )

        sampling_params = SamplingParams(
            # Keep sampling until receiving DraftClose.
            max_new_tokens=1 << 30,
            temperature=0.0,
            top_k=1,
            ignore_eos=True,
        )
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        req = Req(
            build_draft_scheduler_rid(message.draft_key),
            "",
            array("q", [int(token_id) for token_id in message.prompt_token_ids]),
            sampling_params,
            return_logprob=False,
            stream=False,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
            metrics_collector=(
                self.metrics_collector if self.server_args.enable_metrics else None
            ),
        )
        req.tokenizer = self.tokenizer
        req.output_ids = array(
            "q", [int(token_id) for token_id in message.committed_output_ids]
        )
        req._refresh_fill_ids()
        req.fill_len = len(req.full_untruncated_fill_ids)
        self.init_req_max_new_tokens(req)
        state.req = req
        state.verifier_committed_prefix_len = len(req.output_ids)
        return req

    def _draft_commit_segment_consumable_len(
        self: Scheduler,
        segment: VerifierCommitSegment,
        req: Req,
        state: DraftReqState,
    ) -> int:
        pre_verify_committed_len = int(segment.pre_verify_committed_len)
        current_committed_len = int(state.verifier_committed_prefix_len)
        if pre_verify_committed_len != current_committed_len:
            raise RuntimeError(
                "Verifier commit segment does not match the drafter committed prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"current_committed_len={current_committed_len}"
            )
        if not segment.committed_token_ids:
            return 0

        output_len = len(req.output_ids)
        if pre_verify_committed_len > output_len:
            raise RuntimeError(
                "Verifier commit segment is ahead of the drafter committed prefix: "
                f"request_id={segment.draft_key.request_id} "
                f"src_verifier_rank={segment.draft_key.src_verifier_rank} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"output_len={output_len}"
            )

        if pre_verify_committed_len == output_len:
            return 0

        matched_len = 0
        max_possible_match_len = min(
            len(segment.committed_token_ids),
            output_len - pre_verify_committed_len,
        )
        while (
            matched_len < max_possible_match_len
            and int(req.output_ids[pre_verify_committed_len + matched_len])
            == int(segment.committed_token_ids[matched_len])
        ):
            matched_len += 1

        if matched_len == len(segment.committed_token_ids):
            return matched_len
        if matched_len < max_possible_match_len:
            # the first token that doesn't match the drafter's suffix is still considered consumable
            return matched_len + 1
        return matched_len

    def _collect_ready_draft_controls(
        self: Scheduler,
        control_inbox: DraftControlInbox,
    ) -> ReadyDraftControls:
        def consumable_commit_len(segment: VerifierCommitSegment) -> int:
            state = self.draft_req_table.get(segment.draft_key)
            if state is None:
                return 0
            req = state.req
            if (
                req is None
                or req.req_pool_idx is None
                or req.kv_committed_freed
            ):
                return 0

            return self._draft_commit_segment_consumable_len(segment, req, state)

        return control_inbox.extract_ready_controls_locked(consumable_commit_len)

    def _apply_ready_verifier_commit_segments(
        self: Scheduler,
        ready_commit_segments: list[VerifierCommitSegment],
    ) -> int:
        kv_truncations: list[DraftKVTruncation] = []
        batch_metadata_updates: list[DraftBatchMetadataUpdate] = []
        use_native_rows = get_decoupled_spec_transport().supports_native_rows
        commit_echo_batch = None if use_native_rows else DraftTailStreamOutputBatch()
        commit_echo_rows: list[tuple] = []
        if not ready_commit_segments:
            return 0

        running_req_to_idx = {}
        if self.running_batch is not None and not self.running_batch.is_empty():
            running_req_to_idx = {
                id(req): req_batch_idx
                for req_batch_idx, req in enumerate(self.running_batch.reqs)
        }

        for segment in ready_commit_segments:
            state = self.draft_req_table.get(segment.draft_key)
            if state is None:
                raise RuntimeError(
                    "Ready verifier commit segment has no draft state: "
                    f"draft_key={segment.draft_key}"
                )
            req = state.req
            if req is None:
                raise RuntimeError(
                    "Ready verifier commit segment has no draft request: "
                    f"draft_key={segment.draft_key}"
                )

            req_batch_idx = running_req_to_idx.get(id(req))
            self.apply_verifier_commit_segment(
                req,
                segment,
                req_batch_idx=req_batch_idx,
                kv_truncations=kv_truncations,
                batch_metadata_updates=batch_metadata_updates,
            )
            if self.is_draft_entry_rank():
                if not segment.committed_token_ids:
                    raise ValueError(
                        "VerifierCommitSegment committed_token_ids must be "
                        "non-empty before echoing applied segment: "
                        f"request_id={segment.draft_key.request_id} "
                        f"pre_verify_committed_len={segment.pre_verify_committed_len}"
                    )
                committed_token_pos = (
                    int(segment.pre_verify_committed_len)
                    + len(segment.committed_token_ids)
                    - 1
                )
                # Echo the last applied committed token so verifier-side
                # pending expected tokens can advance when no comparable
                # draft-tail anchor was available in its buffer.
                if use_native_rows:
                    commit_echo_rows.append(
                        (
                            self.get_decoupled_spec_rank(),
                            int(segment.draft_key.src_verifier_rank),
                            segment.draft_key.request_id,
                            committed_token_pos,
                            committed_token_pos,
                            int(segment.committed_token_ids[-1]),
                        )
                    )
                else:
                    assert commit_echo_batch is not None
                    commit_echo_batch.outputs.append(
                        DraftTailStreamOutput(
                            src_drafter_rank=self.get_decoupled_spec_rank(),
                            dst_verifier_rank=int(segment.draft_key.src_verifier_rank),
                            request_id=segment.draft_key.request_id,
                            base_committed_len=committed_token_pos,
                            new_token_pos=committed_token_pos,
                            new_token_id=int(segment.committed_token_ids[-1]),
                        )
                    )

        self._flush_draft_kv_truncations(kv_truncations)
        self._flush_draft_batch_metadata_updates(batch_metadata_updates)
        if use_native_rows:
            if commit_echo_rows:
                submit_rows = getattr(
                    self._get_token_sync_thread(), "submit_draft_result_rows", None
                )
                if submit_rows is None:
                    raise RuntimeError(
                        "C++ decoupled token sync thread does not expose row API"
                    )
                submit_rows(commit_echo_rows)
            return len(commit_echo_rows)
        assert commit_echo_batch is not None
        if commit_echo_batch.outputs:
            self._get_token_sync_thread().submit_draft_results(commit_echo_batch)
        return len(commit_echo_batch.outputs)

    def _handle_draft_sync_message(
        self: Scheduler,
        message: DraftSync,
    ) -> None:
        req = self._create_draft_request(message)
        running_batch = self.running_batch
        if (
            req not in self.waiting_queue
            and req not in running_batch.reqs
            and req not in self.draft_sleeping_reqs.values()
        ):
            self._add_request_to_queue(req)

    def _handle_draft_close_key(self: Scheduler, draft_key: DraftReqKey) -> None:
        entry = self.draft_req_table.get(draft_key)
        if entry is None:
            return

        req = entry.req
        if req is not None:
            self.release_draft_request(req)
            return

        raise RuntimeError(
            "DraftClose found drafter state without a live request: "
            f"draft_key={draft_key} is_sleeping={entry.is_sleeping}"
        )

    def sync_draft_requests(self: Scheduler) -> None:
        """
        (called by decoupled drafter)
        Collect ready verifier-to-drafter controls in arrival order.
        DraftSync creates requests, ready VerifierCommitSegment objects
        advance/truncate existing requests, and DraftClose releases drafter-side
        state.
        """
        if not self.spec_algorithm.is_decoupled_draft():
            return None

        ready_controls: ReadyDraftControls | None = None
        if self.is_draft_entry_rank():
            ready_controls = (
                self._get_token_sync_thread().collect_ready_draft_controls(
                    self._collect_ready_draft_controls
                )
            )

        ready_controls = self._broadcast_ready_draft_controls(ready_controls)
        closed_keys = ready_controls.close_keys
        for draft_key in closed_keys:
            self._handle_draft_close_key(draft_key)

        for message in ready_controls.sync_messages:
            draft_key = message.draft_key
            if draft_key in closed_keys:
                continue
            self._handle_draft_sync_message(message)

        self._apply_ready_verifier_commit_segments(
            ready_controls.ready_commit_segments
        )

    def _draft_ahead_window(self: Scheduler) -> int:
        draft_tokens = self.server_args.speculative_num_draft_tokens
        return max(0, int(draft_tokens or 0) * 2)

    def _draft_req_ahead(self: Scheduler, state: DraftReqState) -> int:
        req = state.req
        if req is None:
            return 0
        return len(req.output_ids) - int(state.verifier_committed_prefix_len)

    def has_draft_sleeping_requests(self: Scheduler) -> bool:
        # check whether decoupled drafter has sleeping requests
        return bool(self.draft_sleeping_reqs)

    def _build_draft_decode_batch(self: Scheduler, reqs: list[Req]) -> ScheduleBatch:
        device = self.device
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64
        )
        seq_lens = [
            len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
            for req in reqs
        ]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        tail_token_ids = torch.tensor(
            [
                int(req.output_ids[-1])
                if req.output_ids
                else int(req.origin_input_ids[-1])
                for req in reqs
            ],
            dtype=torch.int64,
            device=device,
        )
        self.future_map.output_tokens_buf[batch.req_pool_indices] = tail_token_ids
        batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        return batch

    def sleep_overrun_draft_requests(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
    ) -> Optional[ScheduleBatch]:
        if batch is None or batch.is_empty():
            return batch
        window = self._draft_ahead_window()
        if window <= 0:
            return batch

        keep_indices: list[int] = []
        slept_any = False
        for req_batch_idx, req in enumerate(batch.reqs):
            state = self._get_draft_state_by_req(req)
            ahead = self._draft_req_ahead(state)
            if ahead >= window:
                state.is_sleeping = True
                self.draft_sleeping_reqs[state.key] = req
                slept_any = True
            else:
                keep_indices.append(req_batch_idx)

        if slept_any:
            batch.filter_batch(keep_indices=keep_indices)
            batch.batch_is_full = False
        return batch

    def wake_draft_sleeping_requests(self: Scheduler) -> None:
        window = self._draft_ahead_window()
        if window <= 0 or not self.draft_sleeping_reqs:
            return None
        max_batch_size = getattr(self.server_args, "pp_max_micro_batch_size", None)
        if not max_batch_size:
            max_batch_size = getattr(self, "max_running_requests", None)
        max_batch_size = int(max_batch_size or 0)
        if max_batch_size > 0:
            available_num_reqs = max(0, max_batch_size - len(self.running_batch.reqs))
            if available_num_reqs == 0:
                return None
        else:
            available_num_reqs = len(self.draft_sleeping_reqs)

        wake_reqs: list[Req] = []
        for draft_key, req in list(self.draft_sleeping_reqs.items()):
            state = self.draft_req_table.get(draft_key)
            if state is None or state.req is not req:
                self.draft_sleeping_reqs.pop(draft_key, None)
                continue
            ahead = self._draft_req_ahead(state)
            if ahead < window:
                state.is_sleeping = False
                self.draft_sleeping_reqs.pop(draft_key, None)
                wake_reqs.append(req)
                if len(wake_reqs) >= available_num_reqs:
                    break

        if not wake_reqs:
            return None

        # build decode batch for these woken reqs, and merge them into the current batch
        wake_batch = self._build_draft_decode_batch(wake_reqs)
        if self.running_batch.is_empty():
            self.running_batch = wake_batch
        else:
            self.running_batch.merge_batch(wake_batch)
        self.running_batch.batch_is_full = False

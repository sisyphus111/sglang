from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class DecoupledDraftMambaStateManager:
    """Owns rollback checkpoint allocation and routing for a decoupled drafter."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def release(self, state) -> None:
        slots = state.mamba_checkpoint_slots
        state.mamba_checkpoint_positions.clear()
        state.mamba_checkpoint_slots = None
        if slots is not None:
            self.scheduler.req_to_token_pool.mamba_allocator.free(slots)

    def _ensure_slots(self, state) -> torch.Tensor:
        if state.mamba_checkpoint_slots is not None:
            return state.mamba_checkpoint_slots

        scheduler = self.scheduler
        window = scheduler._draft_ahead_window()
        if window <= 0:
            raise RuntimeError(
                "Decoupled drafter mamba rollback requires a positive draft "
                f"ahead window. request_id={state.key.request_id}, window={window}"
            )
        mamba_pool = scheduler.req_to_token_pool.mamba_pool
        mamba_allocator = getattr(scheduler.req_to_token_pool, "mamba_allocator", None)
        if mamba_pool is None or mamba_allocator is None:
            raise RuntimeError(
                "Decoupled drafter mamba checkpoint requested without a mamba pool: "
                f"request_id={state.key.request_id}"
            )
        slots = mamba_allocator.alloc(window)
        if slots is None:
            raise RuntimeError(
                "Not enough space for decoupled drafter mamba rollback "
                "checkpoints. Try to increase --mamba-full-memory-ratio or "
                f"--max-mamba-cache-size. request_id={state.key.request_id}, "
                f"window={window}, mamba_pool_size={mamba_pool.size}, "
                f"mamba_available_size={mamba_allocator.available_size()}"
            )
        state.mamba_checkpoint_slots = slots
        return slots

    def checkpoint_slot(
        self, state, token_pos: int, *, for_write: bool
    ) -> torch.Tensor:
        if for_write:
            slots = self._ensure_slots(state)
        else:
            if token_pos not in state.mamba_checkpoint_positions:
                req = state.req
                raise RuntimeError(
                    "Missing decoupled drafter mamba checkpoint. "
                    f"request_id={state.key.request_id}, token_pos={token_pos}, "
                    f"output_len={len(req.output_ids) if req else None}, "
                    "available_checkpoint_positions="
                    f"{sorted(state.mamba_checkpoint_positions)}"
                )
            slots = state.mamba_checkpoint_slots
            if slots is None:
                raise RuntimeError(
                    "Decoupled drafter mamba checkpoint metadata exists without "
                    "allocated checkpoint slots. "
                    f"request_id={state.key.request_id}, token_pos={token_pos}"
                )

        slot_count = int(slots.numel())
        slot_offset = token_pos % slot_count
        if for_write:
            for existing_pos in state.mamba_checkpoint_positions:
                if existing_pos != token_pos and existing_pos % slot_count == slot_offset:
                    raise RuntimeError(
                        "Decoupled drafter mamba checkpoint ring would overwrite a "
                        "live checkpoint. This indicates the drafter exceeded its "
                        "rollback window. "
                        f"request_id={state.key.request_id}, token_pos={token_pos}, "
                        f"existing_pos={existing_pos}, slot_count={slot_count}"
                    )
        return slots[slot_offset : slot_offset + 1]

    def prune(self, state) -> None:
        req = state.req
        if req is None or not state.mamba_checkpoint_positions:
            return
        committed_len = int(state.verifier_committed_prefix_len)
        output_len = len(req.output_ids)
        tail_pos = output_len - 1
        positions_to_invalidate = [
            pos
            for pos in state.mamba_checkpoint_positions
            if pos >= output_len or (pos < committed_len and pos != tail_pos)
        ]
        for pos in positions_to_invalidate:
            state.mamba_checkpoint_positions.discard(pos)

    def commit(
        self, batch: ScheduleBatch, req_indices: Optional[list[int]] = None
    ) -> None:
        scheduler = self.scheduler
        if not scheduler.is_draft_worker_batch(batch):
            return
        try:
            candidate_indices = (
                range(len(batch.reqs)) if req_indices is None else req_indices
            )
            for req_batch_idx in candidate_indices:
                if not (0 <= req_batch_idx < len(batch.reqs)):
                    continue
                req = batch.reqs[req_batch_idx]
                if req.mamba_pool_idx is None or not req.output_ids:
                    continue
                state = scheduler._get_draft_state_by_req(req)
                self.prune(state)
                token_pos = len(req.output_ids) - 1
                if token_pos in state.mamba_checkpoint_positions:
                    continue
                if batch.mamba_cache_dst_indices is None:
                    raise RuntimeError(
                        "Decoupled drafter emitted a token without mamba "
                        "routing metadata. "
                        f"request_id={state.key.request_id}, token_pos={token_pos}"
                    )
                state.mamba_checkpoint_positions.add(token_pos)
        finally:
            if (
                batch.forward_mode.is_decode()
                or batch.forward_mode.is_extend(include_draft_extend_v2=True)
            ):
                batch.mamba_cache_src_indices = None
                batch.mamba_cache_dst_indices = None

    def prepare_routing(self, batch: ScheduleBatch) -> None:
        scheduler = self.scheduler
        if not scheduler.is_draft_worker_batch(batch):
            return
        if not isinstance(scheduler.req_to_token_pool, HybridReqToTokenPool):
            return
        is_decode = batch.forward_mode.is_decode()
        is_prefill = batch.forward_mode.is_extend(include_draft_extend_v2=True)
        if not is_decode and not is_prefill:
            return

        src_indices: list[torch.Tensor] = []
        dst_indices: list[torch.Tensor] = []
        for req in batch.reqs:
            if req.mamba_pool_idx is None:
                raise RuntimeError(
                    "Decoupled drafter mamba routing requires every req "
                    f"to own a mamba slot. rid={req.rid}"
                )
            if is_decode and not req.output_ids:
                raise RuntimeError(
                    "Decoupled drafter mamba routing requires a tail token. "
                    f"rid={req.rid}"
                )
            state = scheduler._get_draft_state_by_req(req)
            if is_decode:
                self.prune(state)
                token_pos = len(req.output_ids) - 1
                src_indices.append(
                    self.checkpoint_slot(state, token_pos, for_write=False)
                )
                dst_indices.append(
                    self.checkpoint_slot(state, token_pos + 1, for_write=True)
                )
            else:
                token_pos = len(req.output_ids)
                dst_slot = self.checkpoint_slot(state, token_pos, for_write=True)
                src_indices.append(dst_slot)
                dst_indices.append(dst_slot)

        if not src_indices:
            return
        device = batch.seq_lens.device
        batch.mamba_cache_src_indices = torch.cat(src_indices).to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        batch.mamba_cache_dst_indices = torch.cat(dst_indices).to(
            device=device, dtype=torch.int64, non_blocking=True
        )

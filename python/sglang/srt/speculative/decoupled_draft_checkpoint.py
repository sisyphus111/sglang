from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool


class DraftRequestGeneration(msgspec.Struct, frozen=True):
    """Identity of one drafter-side request lifetime.

    ``request_id`` is only verifier-local, so the verifier rank is part of the
    key. ``generation`` separates a re-opened request from stale checkpoints
    left by an earlier lifetime with the same wire identity.
    """

    src_verifier_rank: int
    request_id: str
    generation: int


class DraftRewritePlan(msgspec.Struct, frozen=True):
    """Pure token/KV/state mutation plan for one verifier commit.

    Positions in this plan are prefix lengths in the full prompt-plus-output
    sequence. After a draft forward appends its newly predicted tail token, the
    active recurrent state and materialized KV correspond to
    ``prompt_tokens + output_tokens[:-1]``. Consequently their position is
    ``prompt_len + len(output_tokens) - 1``.

    ``state_restore_position`` is the checkpoint needed immediately before the
    first rewritten output. ``replay_tokens`` are authoritative tokens that
    must be processed after the restore to reach the state immediately before
    ``new_output_tokens[-1]``. For the common F=1 bonus-token rewrite,
    ``replay_tokens`` is empty.
    """

    pre_verify_committed_len: int
    new_committed_len: int
    num_match_tokens: int
    new_output_tokens: tuple[int, ...]
    rewrite_output_position: Optional[int] = None
    state_restore_position: Optional[int] = None
    replay_tokens: tuple[int, ...] = ()
    kv_keep_len: Optional[int] = None
    kv_free_start: Optional[int] = None
    kv_free_end: Optional[int] = None

    @property
    def needs_rewrite(self) -> bool:
        return self.rewrite_output_position is not None


class _RequestCheckpointRing(msgspec.Struct):
    checkpoint_slots: torch.Tensor
    position_to_slot_offset: dict[int, int] = {}
    slot_offset_to_position: dict[int, int] = {}


def draft_active_state_position(prompt_len: int, num_output_tokens: int) -> int:
    """Return the full-sequence prefix length represented by active state/KV.

    The scheduler calls this after a forward result has appended one newly
    predicted output token. That token is the next decode input and has not yet
    been consumed by the model, hence the ``-1``.
    """

    prompt_len = int(prompt_len)
    num_output_tokens = int(num_output_tokens)
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")
    if num_output_tokens <= 0:
        raise ValueError(
            "active draft state requires a predicted tail token, "
            f"got num_output_tokens={num_output_tokens}"
        )
    return prompt_len + num_output_tokens - 1


def plan_draft_rewrite(
    *,
    prompt_len: int,
    current_output_tokens: Sequence[int],
    pre_verify_committed_len: int,
    commit_tokens: Sequence[int],
    kv_committed_len: int,
    kv_allocated_len: int,
) -> DraftRewritePlan:
    """Plan drafter alignment without mutating Req, KV, or recurrent state.

    A fully matching verifier commit only advances the committed cursor and
    preserves the valid ahead suffix. At the first mismatch, every later draft
    token belongs to the old branch. The plan truncates that suffix, restores
    the checkpoint immediately before the mismatch, and reports any
    authoritative intermediate tokens that must be replayed before decode can
    resume from the new tail token.

    The verifier may only commit outputs already materialized by the drafter.
    Waiting for a short drafter tail is owned by the data plane / DrafterManager,
    not hidden inside this pure planner.
    """

    prompt_len = int(prompt_len)
    pre_verify_committed_len = int(pre_verify_committed_len)
    kv_committed_len = int(kv_committed_len)
    kv_allocated_len = int(kv_allocated_len)
    current_outputs = tuple(int(token) for token in current_output_tokens)
    verifier_tokens = tuple(int(token) for token in commit_tokens)

    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")
    if pre_verify_committed_len < 0:
        raise ValueError(
            "pre_verify_committed_len must be non-negative, "
            f"got {pre_verify_committed_len}"
        )
    if not verifier_tokens:
        raise ValueError("commit_tokens must be non-empty")
    if not current_outputs:
        raise ValueError("current_output_tokens must contain a draft tail token")
    if pre_verify_committed_len > len(current_outputs):
        raise ValueError(
            "pre_verify_committed_len exceeds the materialized drafter output: "
            f"pre_verify_committed_len={pre_verify_committed_len} "
            f"output_len={len(current_outputs)}"
        )

    new_committed_len = pre_verify_committed_len + len(verifier_tokens)
    if new_committed_len > len(current_outputs):
        raise ValueError(
            "verifier commit exceeds the materialized drafter output: "
            f"new_committed_len={new_committed_len} "
            f"output_len={len(current_outputs)}"
        )

    active_position = draft_active_state_position(prompt_len, len(current_outputs))
    if kv_committed_len < active_position or kv_allocated_len < active_position:
        raise ValueError(
            "draft KV is shorter than the active recurrent-state prefix: "
            f"active_position={active_position} "
            f"kv_committed_len={kv_committed_len} "
            f"kv_allocated_len={kv_allocated_len}"
        )

    num_match_tokens = 0
    while (
        num_match_tokens < len(verifier_tokens)
        and current_outputs[pre_verify_committed_len + num_match_tokens]
        == verifier_tokens[num_match_tokens]
    ):
        num_match_tokens += 1

    if num_match_tokens == len(verifier_tokens):
        return DraftRewritePlan(
            pre_verify_committed_len=pre_verify_committed_len,
            new_committed_len=new_committed_len,
            num_match_tokens=num_match_tokens,
            new_output_tokens=current_outputs,
        )

    rewrite_output_position = pre_verify_committed_len + num_match_tokens
    rewritten_outputs = (
        current_outputs[:rewrite_output_position] + verifier_tokens[num_match_tokens:]
    )
    state_restore_position = prompt_len + rewrite_output_position
    if state_restore_position > min(kv_committed_len, kv_allocated_len):
        raise ValueError(
            "rewrite checkpoint lies beyond materialized KV: "
            f"state_restore_position={state_restore_position} "
            f"kv_committed_len={kv_committed_len} "
            f"kv_allocated_len={kv_allocated_len}"
        )

    # These tokens follow the restored boundary but precede the new tail. They
    # must be replayed through the model before normal one-token decode resumes.
    replay_tokens = rewritten_outputs[rewrite_output_position:-1]
    return DraftRewritePlan(
        pre_verify_committed_len=pre_verify_committed_len,
        new_committed_len=new_committed_len,
        num_match_tokens=num_match_tokens,
        new_output_tokens=rewritten_outputs,
        rewrite_output_position=rewrite_output_position,
        state_restore_position=state_restore_position,
        replay_tokens=replay_tokens,
        kv_keep_len=state_restore_position,
        kv_free_start=state_restore_position,
        kv_free_end=kv_allocated_len,
    )


class DecoupledDraftMambaCheckpointStore:
    """GPU checkpoint rings for decoupled Qwen3.5/GDN drafter state.

    Checkpoints are ordinary slots in the v0.5.17 ``HybridReqToTokenPool``.
    This store reserves them with ``mamba_allocator`` and copies state only via
    the official ``mamba_pool.copy_from`` API. It deliberately has no knowledge
    of Scheduler batches, CUDA-graph routing buffers, or data-plane messages.

    ``max_draft_tokens`` is K, the maximum number of drafter-proposed tokens in
    one verifier round; it excludes the target model's bonus token. The minimum
    ring capacity is therefore exactly ``2 * K + 1``.

    All state-copy and release methods must be called on the same ordered model
    execution stream used by the drafter. Logical pruning makes ring offsets
    reusable; physical slots stay allocated until the request generation is
    released.
    """

    def __init__(
        self,
        req_to_token_pool: HybridReqToTokenPool,
        *,
        max_draft_tokens: int,
        capacity: Optional[int] = None,
    ) -> None:
        max_draft_tokens = int(max_draft_tokens)
        if max_draft_tokens <= 0:
            raise ValueError(
                f"max_draft_tokens must be positive, got {max_draft_tokens}"
            )
        required_capacity = 2 * max_draft_tokens + 1
        capacity = required_capacity if capacity is None else int(capacity)
        if capacity < required_capacity:
            raise ValueError(
                "decoupled draft checkpoint capacity must cover two draft "
                f"windows plus the boundary token: capacity={capacity} "
                f"required_capacity={required_capacity}"
            )
        if (
            getattr(req_to_token_pool, "mamba_pool", None) is None
            or getattr(req_to_token_pool, "mamba_allocator", None) is None
        ):
            raise TypeError(
                "decoupled draft checkpoints require HybridReqToTokenPool "
                "mamba_pool and mamba_allocator"
            )
        if (
            getattr(req_to_token_pool.mamba_pool, "replayssm_write_pos", None)
            is not None
        ):
            raise ValueError(
                "decoupled draft checkpoints require dense active GDN state; "
                "ReplaySSM copy_from omits its pending ring updates"
            )

        self.req_to_token_pool = req_to_token_pool
        self.capacity = capacity
        self._rings: dict[DraftRequestGeneration, _RequestCheckpointRing] = {}

    def checkpoint_after_forward(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
        active_slot: torch.Tensor,
    ) -> None:
        """Checkpoint active state after the newly predicted tail is appended."""

        self._validate_key(key)
        position = self._validate_position(position)
        active_slot = self._normalize_one_slot(active_slot, "active_slot")
        ring = self._get_or_allocate_ring(key)

        if position in ring.position_to_slot_offset:
            raise RuntimeError(
                "duplicate decoupled draft checkpoint position: "
                f"key={key} position={position}"
            )
        slot_offset = position % self.capacity
        live_position = ring.slot_offset_to_position.get(slot_offset)
        if live_position is not None:
            raise RuntimeError(
                "decoupled draft checkpoint ring would overwrite a live state; "
                "the committed cursor was not pruned or the drafter exceeded "
                f"its rollback window: key={key} position={position} "
                f"live_position={live_position} capacity={self.capacity}"
            )

        checkpoint_slot = ring.checkpoint_slots[slot_offset : slot_offset + 1]
        self._copy_state(active_slot, checkpoint_slot)
        ring.position_to_slot_offset[position] = slot_offset
        ring.slot_offset_to_position[slot_offset] = position

    def restore_for_rewrite(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
        active_slot: torch.Tensor,
    ) -> None:
        """Restore the matching prefix state and invalidate its old future."""

        self._validate_key(key)
        position = self._validate_position(position)
        active_slot = self._normalize_one_slot(active_slot, "active_slot")
        ring = self._rings.get(key)
        if ring is None or position not in ring.position_to_slot_offset:
            available = (
                () if ring is None else tuple(sorted(ring.position_to_slot_offset))
            )
            raise RuntimeError(
                "missing decoupled draft recurrent-state checkpoint: "
                f"key={key} position={position} available_positions={available}"
            )

        slot_offset = ring.position_to_slot_offset[position]
        checkpoint_slot = ring.checkpoint_slots[slot_offset : slot_offset + 1]
        self._copy_state(checkpoint_slot, active_slot)
        self.prune(key, max_position=position)

    def has_checkpoint(self, key: DraftRequestGeneration, position: int) -> bool:
        ring = self._rings.get(key)
        return ring is not None and int(position) in ring.position_to_slot_offset

    def positions(self, key: DraftRequestGeneration) -> tuple[int, ...]:
        ring = self._rings.get(key)
        if ring is None:
            return ()
        return tuple(sorted(ring.position_to_slot_offset))

    def prune(
        self,
        key: DraftRequestGeneration,
        *,
        min_position: Optional[int] = None,
        max_position: Optional[int] = None,
    ) -> None:
        """Forget checkpoints outside an inclusive live position interval."""

        ring = self._rings.get(key)
        if ring is None:
            return
        if min_position is not None:
            min_position = self._validate_position(min_position)
        if max_position is not None:
            max_position = self._validate_position(max_position)
        if (
            min_position is not None
            and max_position is not None
            and min_position > max_position
        ):
            raise ValueError(
                f"min_position={min_position} exceeds max_position={max_position}"
            )

        stale_positions = [
            position
            for position in ring.position_to_slot_offset
            if (min_position is not None and position < min_position)
            or (max_position is not None and position > max_position)
        ]
        for position in stale_positions:
            slot_offset = ring.position_to_slot_offset.pop(position)
            del ring.slot_offset_to_position[slot_offset]

    def release(self, key: DraftRequestGeneration) -> None:
        """Release all physical checkpoint slots owned by one generation."""

        ring = self._rings.pop(key, None)
        if ring is not None:
            self.req_to_token_pool.mamba_allocator.free(ring.checkpoint_slots)

    def close(self) -> None:
        for key in tuple(self._rings):
            self.release(key)

    def _get_or_allocate_ring(
        self, key: DraftRequestGeneration
    ) -> _RequestCheckpointRing:
        ring = self._rings.get(key)
        if ring is not None:
            return ring

        slots = self.req_to_token_pool.mamba_allocator.alloc(self.capacity)
        if slots is None:
            available_size = self.req_to_token_pool.mamba_allocator.available_size()
            raise RuntimeError(
                "not enough Mamba slots for decoupled draft rollback "
                f"checkpoints: key={key} capacity={self.capacity} "
                f"available_size={available_size}"
            )
        if not isinstance(slots, torch.Tensor) or slots.numel() != self.capacity:
            raise RuntimeError(
                "mamba_allocator returned an invalid checkpoint allocation: "
                f"expected_slots={self.capacity} got={slots!r}"
            )
        ring = _RequestCheckpointRing(checkpoint_slots=slots)
        self._rings[key] = ring
        return ring

    def _copy_state(self, src_slots: torch.Tensor, dst_slots: torch.Tensor) -> None:
        src_slots = self.req_to_token_pool.translate_mamba_indices(src_slots)
        dst_slots = self.req_to_token_pool.translate_mamba_indices(dst_slots)
        self.req_to_token_pool.mamba_pool.copy_from(src_slots, dst_slots)

    @staticmethod
    def _normalize_one_slot(slot: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(slot, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(slot).__name__}")
        if slot.numel() != 1:
            raise ValueError(
                f"{name} must contain exactly one slot, got {slot.numel()}"
            )
        return slot.reshape(1)

    @staticmethod
    def _validate_key(key: DraftRequestGeneration) -> None:
        if not isinstance(key, DraftRequestGeneration):
            raise TypeError(
                "checkpoint key must be DraftRequestGeneration, "
                f"got {type(key).__name__}"
            )
        if key.src_verifier_rank < 0 or key.generation < 0 or not key.request_id:
            raise ValueError(f"invalid draft request generation key: {key}")

    @staticmethod
    def _validate_position(position: int) -> int:
        position = int(position)
        if position < 0:
            raise ValueError(
                f"checkpoint position must be non-negative, got {position}"
            )
        return position

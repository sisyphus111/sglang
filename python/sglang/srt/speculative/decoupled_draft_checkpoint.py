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
    key. ``request_epoch`` separates a re-opened request from stale checkpoints
    left by an earlier lifetime with the same wire identity.
    """

    src_verifier_rank: int
    request_id: str
    request_epoch: int


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
    checkpoint_slot_ids: tuple[int, ...]
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
    """GPU state-slot rings for decoupled Qwen3.5/GDN drafter rollback.

    Every live logical state position owns one ordinary Mamba slot. Decode reads
    the slot for position ``p`` and writes the slot for ``p + 1`` directly in the
    GDN kernels. A rewrite therefore selects an older source slot and invalidates
    its old future; it never copies the full recurrent state.

    ``max_draft_tokens`` is K, the maximum number of drafter-proposed tokens in
    one verifier round; it excludes the target model's bonus token. The minimum
    ring capacity is therefore exactly ``2 * K + 1``.

    Slot ids are materialized on allocation so the scheduler hot path never calls
    ``Tensor.item()`` on a CUDA tensor. Logical pruning makes ring offsets
    reusable; physical slots stay allocated until the request generation ends.
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
                "decoupled draft state routing requires dense GDN state; "
                "ReplaySSM pending ring updates are not independently routable"
            )

        self.req_to_token_pool = req_to_token_pool
        self.capacity = capacity
        self._rings: dict[DraftRequestGeneration, _RequestCheckpointRing] = {}

    def prepare_prefill_route(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
    ) -> int:
        """Return the in-place destination used by a (possibly chunked) prefill."""

        self._validate_key(key)
        position = self._validate_position(position)
        ring = self._get_or_allocate_ring(key)
        slot_offset = position % self.capacity
        self._check_writable_offset(key, ring, position, slot_offset)
        return ring.checkpoint_slot_ids[slot_offset]

    def prepare_decode_route(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
    ) -> tuple[int, int]:
        """Return physical source/destination slot ids for one decode step."""

        self._validate_key(key)
        position = self._validate_position(position)
        ring = self._rings.get(key)
        if ring is None or position not in ring.position_to_slot_offset:
            available = (
                () if ring is None else tuple(sorted(ring.position_to_slot_offset))
            )
            raise RuntimeError(
                "missing decoupled draft recurrent-state checkpoint: "
                f"key={key} position={position} available_positions={available}"
            )

        src_offset = ring.position_to_slot_offset[position]
        dst_position = position + 1
        dst_offset = dst_position % self.capacity
        self._check_writable_offset(key, ring, dst_position, dst_offset)
        return (
            ring.checkpoint_slot_ids[src_offset],
            ring.checkpoint_slot_ids[dst_offset],
        )

    def commit_after_forward(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
    ) -> None:
        """Mark the state written by the just-finished forward as live."""

        self._validate_key(key)
        position = self._validate_position(position)
        ring = self._rings.get(key)
        if ring is None:
            raise RuntimeError(
                "decoupled draft forward completed without a state-slot ring: "
                f"key={key} position={position}"
            )
        if position in ring.position_to_slot_offset:
            raise RuntimeError(
                "duplicate decoupled draft checkpoint position: "
                f"key={key} position={position}"
            )
        slot_offset = position % self.capacity
        self._check_writable_offset(key, ring, position, slot_offset)
        ring.position_to_slot_offset[position] = slot_offset
        ring.slot_offset_to_position[slot_offset] = position

    def rewind_for_rewrite(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
    ) -> None:
        """Select a matching prefix state by invalidating only its old future."""

        self._validate_key(key)
        position = self._validate_position(position)
        ring = self._rings.get(key)
        if ring is None or position not in ring.position_to_slot_offset:
            available = (
                () if ring is None else tuple(sorted(ring.position_to_slot_offset))
            )
            raise RuntimeError(
                "missing decoupled draft recurrent-state checkpoint: "
                f"key={key} position={position} available_positions={available}"
            )

        self.prune(key, max_position=position)

    def has_checkpoint(self, key: DraftRequestGeneration, position: int) -> bool:
        ring = self._rings.get(key)
        return ring is not None and int(position) in ring.position_to_slot_offset

    def positions(self, key: DraftRequestGeneration) -> tuple[int, ...]:
        ring = self._rings.get(key)
        if ring is None:
            return ()
        return tuple(sorted(ring.position_to_slot_offset))

    def checkpoint_slots(self, key: DraftRequestGeneration) -> torch.Tensor:
        """Return the fixed physical-slot ring owned by one request lifetime.

        The overlap drafter indexes this tensor on GPU by logical state
        position modulo ``capacity``. Allocation remains a lifecycle operation;
        steady decode only changes the device-side position.
        """

        self._validate_key(key)
        return self._get_or_allocate_ring(key).checkpoint_slots

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
        slot_ids = tuple(int(slot_id) for slot_id in slots.tolist())
        ring = _RequestCheckpointRing(
            checkpoint_slots=slots,
            checkpoint_slot_ids=slot_ids,
        )
        self._rings[key] = ring
        return ring

    @staticmethod
    def _check_writable_offset(
        key: DraftRequestGeneration,
        ring: _RequestCheckpointRing,
        position: int,
        slot_offset: int,
    ) -> None:
        live_position = ring.slot_offset_to_position.get(slot_offset)
        if live_position is not None and live_position != position:
            raise RuntimeError(
                "decoupled draft checkpoint ring would overwrite a live state; "
                "the committed cursor was not pruned or the drafter exceeded "
                f"its rollback window: key={key} position={position} "
                f"live_position={live_position} "
                f"capacity={ring.checkpoint_slots.numel()}"
            )

    @staticmethod
    def _validate_key(key: DraftRequestGeneration) -> None:
        if not isinstance(key, DraftRequestGeneration):
            raise TypeError(
                "checkpoint key must be DraftRequestGeneration, "
                f"got {type(key).__name__}"
            )
        if key.src_verifier_rank < 0 or key.request_epoch < 0 or not key.request_id:
            raise ValueError(f"invalid draft request generation key: {key}")

    @staticmethod
    def _validate_position(position: int) -> int:
        position = int(position)
        if position < 0:
            raise ValueError(
                f"checkpoint position must be non-negative, got {position}"
            )
        return position

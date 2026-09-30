from __future__ import annotations

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


class _RequestCheckpointRing(msgspec.Struct):
    checkpoint_slots: torch.Tensor
    owned_slots: torch.Tensor


class DecoupledDraftMambaCheckpointStore:
    """GPU state-slot rings for decoupled Qwen3.5/GDN drafter rollback.

    Every live logical state position owns one ordinary Mamba slot. Decode reads
    the slot for position ``p`` and writes the slot for ``p + 1`` directly in the
    GDN kernels. A rewrite therefore selects an older source slot and invalidates
    its old future; it never copies the full recurrent state.

    ``max_draft_tokens`` is K, the maximum number of drafter-proposed tokens in
    one verifier round; it excludes the target model's bonus token. The minimum
    ring capacity is therefore exactly ``2 * K + 1``.

    Prefill retains the request's ordinary active slot, including radix-cache
    COW and chunked-prefill tracking. The ring borrows that slot at the initial
    decode position and owns only the other ``capacity - 1`` slots. The request
    remains the sole owner responsible for freeing its active slot.

    GPU position tags govern slot reuse and rollback during decode in both
    scheduler modes. Physical slots stay allocated until the generation ends;
    no recurrent state is copied to initialize the ring or roll it back.
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

    def initialize(
        self,
        key: DraftRequestGeneration,
        *,
        position: int,
        active_slot: torch.Tensor,
    ) -> torch.Tensor:
        """Borrow the prefill slot at its final position and allocate the ring.

        Call once after the request's active slot has been allocated. All
        prefill chunks continue to use that slot; the first decode reads it
        through this table after prefill publishes the initial position tag.
        """

        self._validate_key(key)
        position = self._validate_position(position)
        if key in self._rings:
            raise RuntimeError(
                f"decoupled draft checkpoint ring already initialized: {key}"
            )
        if active_slot.numel() != 1:
            raise ValueError("decoupled draft checkpoint requires one active slot")

        num_owned_slots = self.capacity - 1
        slots = self.req_to_token_pool.mamba_allocator.alloc(num_owned_slots)
        if slots is None:
            available_size = self.req_to_token_pool.mamba_allocator.available_size()
            raise RuntimeError(
                "not enough Mamba slots for decoupled draft rollback "
                f"checkpoints: key={key} required_slots={num_owned_slots} "
                f"available_size={available_size}"
            )
        if not isinstance(slots, torch.Tensor) or slots.numel() != num_owned_slots:
            raise RuntimeError(
                "mamba_allocator returned an invalid checkpoint allocation: "
                f"expected_slots={num_owned_slots} got={slots!r}"
            )

        offset = position % self.capacity
        checkpoint_slots = torch.cat(
            (slots[:offset], active_slot.reshape(1), slots[offset:])
        )
        self._rings[key] = _RequestCheckpointRing(
            checkpoint_slots=checkpoint_slots, owned_slots=slots
        )
        return checkpoint_slots

    def checkpoint_slots(self, key: DraftRequestGeneration) -> torch.Tensor:
        """Return the initialized physical-slot ring for one request lifetime.

        The drafter indexes this tensor on GPU by logical state
        position modulo ``capacity``. Allocation remains a lifecycle operation;
        steady decode only changes the device-side position.
        """

        self._validate_key(key)
        return self._rings[key].checkpoint_slots

    def release(self, key: DraftRequestGeneration) -> None:
        """Release all physical checkpoint slots owned by one generation."""

        ring = self._rings.pop(key, None)
        if ring is not None:
            self.req_to_token_pool.mamba_allocator.free(ring.owned_slots)

    def close(self) -> None:
        for key in tuple(self._rings):
            self.release(key)

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

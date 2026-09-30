"""Unit tests for decoupled drafter token/KV and GDN state rollback."""

import unittest

import torch

from sglang.srt.speculative.decoupled_draft_checkpoint import (
    DecoupledDraftMambaCheckpointStore,
    DraftRequestGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeMambaAllocator:
    def __init__(self, size: int):
        self.free_slots = torch.arange(1, size + 1, dtype=torch.int64)
        self.freed: list[torch.Tensor] = []

    def alloc(self, size: int):
        if size > len(self.free_slots):
            return None
        slots = self.free_slots[:size]
        self.free_slots = self.free_slots[size:]
        return slots

    def free(self, slots: torch.Tensor):
        self.freed.append(slots.clone())
        self.free_slots = torch.cat((self.free_slots, slots))

    def available_size(self) -> int:
        return len(self.free_slots)


class _FakeMambaPool:
    def __init__(self, size: int):
        self.state = torch.zeros(size + 1, dtype=torch.int64)
        self.copies: list[tuple[torch.Tensor, torch.Tensor]] = []

    def copy_from(self, src_slots: torch.Tensor, dst_slots: torch.Tensor):
        self.copies.append((src_slots.clone(), dst_slots.clone()))
        self.state[dst_slots] = self.state[src_slots]


class _FakeHybridReqToTokenPool:
    def __init__(self, size: int = 64):
        self.mamba_allocator = _FakeMambaAllocator(size)
        self.mamba_pool = _FakeMambaPool(size)

    def translate_mamba_indices(self, slots: torch.Tensor) -> torch.Tensor:
        return slots


class TestDecoupledDraftMambaCheckpointStore(CustomTestCase):
    def setUp(self):
        self.req_pool = _FakeHybridReqToTokenPool()
        self.store = DecoupledDraftMambaCheckpointStore(
            self.req_pool, max_draft_tokens=2
        )
        self.key = DraftRequestGeneration(
            src_verifier_rank=0, request_id="req-a", request_epoch=3
        )
        # The request owns its prefill slot; the checkpoint ring borrows it.
        self.active_slot = self.req_pool.mamba_allocator.alloc(1)

    def test_prefill_borrows_active_slot_at_final_position_without_copy(self):
        self.req_pool.mamba_pool.state[self.active_slot] = 42
        slots = self.store.initialize(
            self.key, position=13, active_slot=self.active_slot
        )
        self.assertIs(slots, self.store.checkpoint_slots(self.key))
        self.assertEqual(slots.numel(), self.store.capacity)
        self.assertEqual(int(slots[13 % self.store.capacity]), int(self.active_slot[0]))
        # Chunked prefill and COW continue updating the same physical slot.
        self.req_pool.mamba_pool.state[self.active_slot] = 43
        self.assertEqual(
            int(self.req_pool.mamba_pool.state[slots[13 % self.store.capacity]]), 43
        )
        self.assertEqual(self.req_pool.mamba_pool.copies, [])
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), 59)

    def test_duplicate_initialize_does_not_allocate_or_replace_ring(self):
        slots = self.store.initialize(
            self.key, position=10, active_slot=self.active_slot
        )
        available = self.req_pool.mamba_allocator.available_size()
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            self.store.initialize(self.key, position=11, active_slot=self.active_slot)
        self.assertIs(self.store.checkpoint_slots(self.key), slots)
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), available)

    def test_generation_isolation_and_release(self):
        next_generation = DraftRequestGeneration(0, "req-a", 4)
        slots = self.store.initialize(
            self.key, position=10, active_slot=self.active_slot
        )
        other_active_slot = self.req_pool.mamba_allocator.alloc(1)
        other_slots = self.store.initialize(
            next_generation, position=13, active_slot=other_active_slot
        )
        self.assertFalse(set(slots.tolist()) & set(other_slots.tolist()))
        self.store.release(self.key)
        self.assertEqual(len(self.req_pool.mamba_allocator.freed), 1)
        self.assertEqual(
            set(self.req_pool.mamba_allocator.freed[0].tolist()),
            set(slots.tolist()) - set(self.active_slot.tolist()),
        )
        self.assertIs(self.store.checkpoint_slots(next_generation), other_slots)
        self.store.close()
        self.assertEqual(len(self.req_pool.mamba_allocator.freed), 2)
        # Request teardown is the only path that frees the borrowed active slots.
        self.req_pool.mamba_allocator.free(self.active_slot)
        self.req_pool.mamba_allocator.free(other_active_slot)
        all_freed = torch.cat(self.req_pool.mamba_allocator.freed)
        self.assertEqual(all_freed.numel(), all_freed.unique().numel())
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), 64)

    def test_allocation_failure_does_not_install_partial_ring(self):
        self.req_pool.mamba_allocator.free_slots = torch.arange(2, 5)
        with self.assertRaisesRegex(RuntimeError, "not enough Mamba slots"):
            self.store.initialize(self.key, position=10, active_slot=self.active_slot)
        self.assertEqual(self.store._rings, {})
        self.assertEqual(self.req_pool.mamba_allocator.freed, [])
        self.assertEqual(self.req_pool.mamba_allocator.available_size(), 3)

    def test_rejects_ring_smaller_than_two_windows_plus_boundary(self):
        with self.assertRaisesRegex(ValueError, "required_capacity=7"):
            DecoupledDraftMambaCheckpointStore(
                self.req_pool, max_draft_tokens=3, capacity=6
            )

    def test_rejects_replayssm_active_state(self):
        self.req_pool.mamba_pool.replayssm_write_pos = torch.zeros(65)
        with self.assertRaisesRegex(ValueError, "not independently routable"):
            DecoupledDraftMambaCheckpointStore(self.req_pool, max_draft_tokens=2)


if __name__ == "__main__":
    unittest.main()

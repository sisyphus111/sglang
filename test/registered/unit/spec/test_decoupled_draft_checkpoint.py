"""Unit tests for decoupled drafter token/KV and GDN state rollback."""

import unittest

import torch

from sglang.srt.speculative.decoupled_draft_checkpoint import (
    DecoupledDraftMambaCheckpointStore,
    DraftRequestGeneration,
    draft_active_state_position,
    plan_draft_rewrite,
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


class TestDraftRewritePlanner(CustomTestCase):
    def test_active_state_position_excludes_predicted_tail(self):
        self.assertEqual(draft_active_state_position(8, 1), 8)
        self.assertEqual(draft_active_state_position(8, 4), 11)

    def test_matching_commit_preserves_ahead_suffix(self):
        plan = plan_draft_rewrite(
            prompt_len=10,
            current_output_tokens=[20, 21, 22, 23, 24],
            pre_verify_committed_len=1,
            commit_tokens=[21, 22],
            kv_committed_len=14,
            kv_allocated_len=14,
        )

        self.assertFalse(plan.needs_rewrite)
        self.assertEqual(plan.new_committed_len, 3)
        self.assertEqual(plan.num_match_tokens, 2)
        self.assertEqual(plan.new_output_tokens, (20, 21, 22, 23, 24))
        self.assertIsNone(plan.state_restore_position)

    def test_bonus_mismatch_restores_boundary_before_bonus(self):
        plan = plan_draft_rewrite(
            prompt_len=10,
            current_output_tokens=[20, 21, 22, 23, 24],
            pre_verify_committed_len=1,
            commit_tokens=[21, 22, 99],
            kv_committed_len=14,
            kv_allocated_len=14,
        )

        self.assertTrue(plan.needs_rewrite)
        self.assertEqual(plan.num_match_tokens, 2)
        self.assertEqual(plan.rewrite_output_position, 3)
        self.assertEqual(plan.state_restore_position, 13)
        self.assertEqual(plan.kv_keep_len, 13)
        self.assertEqual((plan.kv_free_start, plan.kv_free_end), (13, 14))
        self.assertEqual(plan.new_output_tokens, (20, 21, 22, 99))
        self.assertEqual(plan.replay_tokens, ())

    def test_early_mismatch_reports_authoritative_replay_tokens(self):
        plan = plan_draft_rewrite(
            prompt_len=4,
            current_output_tokens=[10, 11, 12, 13, 14],
            pre_verify_committed_len=1,
            commit_tokens=[91, 92, 93],
            kv_committed_len=8,
            kv_allocated_len=8,
        )

        self.assertEqual(plan.rewrite_output_position, 1)
        self.assertEqual(plan.state_restore_position, 5)
        self.assertEqual(plan.new_output_tokens, (10, 91, 92, 93))
        self.assertEqual(plan.replay_tokens, (91, 92))


class TestDecoupledDraftMambaCheckpointStore(CustomTestCase):
    def setUp(self):
        self.req_pool = _FakeHybridReqToTokenPool()
        self.store = DecoupledDraftMambaCheckpointStore(
            self.req_pool, max_draft_tokens=2
        )
        self.key = DraftRequestGeneration(
            src_verifier_rank=0, request_id="req-a", generation=3
        )
        # Mirror production ownership: the request's active slot is allocated
        # before this store reserves its disjoint checkpoint slots.
        self.active_slot = self.req_pool.mamba_allocator.alloc(1)

    def test_allocates_two_windows_plus_boundary_and_restores(self):
        self.req_pool.mamba_pool.state[self.active_slot] = 101
        self.store.checkpoint_after_forward(
            self.key, position=10, active_slot=self.active_slot
        )
        self.req_pool.mamba_pool.state[self.active_slot] = 202
        self.store.checkpoint_after_forward(
            self.key, position=11, active_slot=self.active_slot
        )
        self.req_pool.mamba_pool.state[self.active_slot] = 303

        self.store.restore_for_rewrite(
            self.key, position=10, active_slot=self.active_slot
        )

        self.assertEqual(int(self.req_pool.mamba_pool.state[self.active_slot]), 101)
        self.assertEqual(self.store.positions(self.key), (10,))
        checkpoint_dst = self.req_pool.mamba_pool.copies[0][1]
        self.assertEqual(checkpoint_dst.numel(), 1)
        self.assertEqual(len(self.req_pool.mamba_allocator.free_slots), 58)

    def test_live_ring_collision_fails_until_old_position_is_pruned(self):
        self.req_pool.mamba_pool.state[self.active_slot] = 1
        self.store.checkpoint_after_forward(
            self.key, position=7, active_slot=self.active_slot
        )
        with self.assertRaisesRegex(RuntimeError, "overwrite a live state"):
            self.store.checkpoint_after_forward(
                self.key, position=12, active_slot=self.active_slot
            )

        self.store.prune(self.key, min_position=8)
        self.store.checkpoint_after_forward(
            self.key, position=12, active_slot=self.active_slot
        )
        self.assertEqual(self.store.positions(self.key), (12,))

    def test_generation_isolation_and_release(self):
        next_generation = DraftRequestGeneration(
            src_verifier_rank=0, request_id="req-a", generation=4
        )
        self.store.checkpoint_after_forward(
            self.key, position=10, active_slot=self.active_slot
        )
        self.store.checkpoint_after_forward(
            next_generation, position=10, active_slot=self.active_slot
        )

        self.store.release(self.key)

        self.assertEqual(self.store.positions(self.key), ())
        self.assertEqual(self.store.positions(next_generation), (10,))
        self.assertEqual(len(self.req_pool.mamba_allocator.freed), 1)
        self.assertEqual(self.req_pool.mamba_allocator.freed[0].numel(), 5)

    def test_rejects_ring_smaller_than_two_windows_plus_boundary(self):
        with self.assertRaisesRegex(ValueError, "required_capacity=7"):
            DecoupledDraftMambaCheckpointStore(
                self.req_pool, max_draft_tokens=3, capacity=6
            )

    def test_rejects_replayssm_active_state(self):
        self.req_pool.mamba_pool.replayssm_write_pos = torch.zeros(65)
        with self.assertRaisesRegex(ValueError, "ReplaySSM copy_from"):
            DecoupledDraftMambaCheckpointStore(self.req_pool, max_draft_tokens=2)


if __name__ == "__main__":
    unittest.main()

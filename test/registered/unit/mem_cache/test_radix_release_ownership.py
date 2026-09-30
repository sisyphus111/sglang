"""Finished requests release owned KV independently of CPU token history."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRadixReleaseOwnership(CustomTestCase):
    def test_no_insert_frees_private_suffix_and_unlocks_shared_prefix(self):
        for page_size in (1, 4):
            for disable_finished_insert in (False, True):
                with self.subTest(
                    page_size=page_size,
                    disable_finished_insert=disable_finished_insert,
                ):
                    allocator = SimpleNamespace(device="cpu", free_segment=Mock())
                    cache = RadixCache.create_simulated(
                        mock_allocator=allocator, page_size=page_size
                    )
                    cache.disable_finished_insert = disable_finished_insert
                    kv_indices = torch.arange(20, 32, dtype=torch.int64)
                    cache.req_to_token_pool = SimpleNamespace(
                        req_to_token=kv_indices.unsqueeze(0)
                    )
                    prefix = RadixKey(array("q", [1, 2, 3, 4]))
                    cache.insert(InsertParams(key=prefix, value=kv_indices[:4]))
                    match = cache.match_prefix(MatchPrefixParams(key=prefix))
                    cache.inc_lock_ref(match.last_device_node)
                    req = SimpleNamespace(
                        req_pool_idx=0,
                        cache_protected_len=4,
                        last_node=match.last_device_node,
                        origin_input_ids=array("q", [1, 2, 3, 4]),
                        # GPU decode has materialized more KV than this mirror.
                        output_ids=array("q", [5]),
                        extra_key=None,
                    )

                    cache.cache_finished_req(
                        req,
                        is_insert=disable_finished_insert,
                        kv_len_to_handle=9,
                    )

                    allocator.free_segment.assert_called_once()
                    freed = allocator.free_segment.call_args.args[0]
                    self.assertTrue(torch.equal(freed, kv_indices[4:9]))
                    self.assertEqual(
                        allocator.free_segment.call_args.kwargs, {"start_pos": 4}
                    )
                    self.assertEqual(match.last_device_node.lock_ref, 0)
                    self.assertEqual(cache.protected_size(), 0)
                    self.assertEqual(cache.evictable_size(), 4)
                    rematch = cache.match_prefix(MatchPrefixParams(key=prefix))
                    self.assertTrue(torch.equal(rematch.device_indices, kv_indices[:4]))

    def test_mamba_no_insert_releases_state_and_full_private_kv_suffix(self):
        kv_indices = torch.arange(20, 32, dtype=torch.int64)
        req_pool = SimpleNamespace(
            req_to_token=kv_indices.unsqueeze(0), free_mamba_cache=Mock()
        )
        allocator = SimpleNamespace(free_segment=Mock())
        cache = SimpleNamespace(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            enable_mamba_extra_buffer=True,
            int8_ckpt_pool=None,
            dec_lock_ref=Mock(),
        )
        req = SimpleNamespace(
            req_pool_idx=0,
            cache_protected_len=4,
            last_node=object(),
            origin_input_ids=array("q", [1, 2, 3, 4]),
            output_ids=array("q", [5]),
        )

        MambaRadixCache.cache_finished_req(
            cache, req, is_insert=False, kv_len_to_handle=9
        )

        allocator.free_segment.assert_called_once()
        self.assertTrue(
            torch.equal(allocator.free_segment.call_args.args[0], kv_indices[4:9])
        )
        req_pool.free_mamba_cache.assert_called_once_with(
            req, mamba_ping_pong_track_buffer_to_keep=None
        )
        cache.dec_lock_ref.assert_called_once_with(req.last_node)


if __name__ == "__main__":
    unittest.main()

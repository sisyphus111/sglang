"""Compare one GDN mixed layer with separate prefill/decode kernel execution."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA GDN kernels")
class TestDecoupledGdnMixed(CustomTestCase):
    @staticmethod
    def _batch(query_lengths, prefix_lengths, source, destination, mode):
        lengths = torch.tensor(query_lengths, dtype=torch.int32, device="cuda")
        prefixes = torch.tensor(prefix_lengths, dtype=torch.int32, device="cuda")
        return ForwardBatch(
            forward_mode=mode,
            batch_size=len(query_lengths),
            input_ids=torch.zeros(sum(query_lengths), dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.arange(len(query_lengths), device="cuda"),
            seq_lens=(lengths + prefixes).to(torch.int64),
            seq_lens_sum=None,
            out_cache_loc=torch.zeros(
                sum(query_lengths), dtype=torch.int64, device="cuda"
            ),
            extend_seq_lens=lengths,
            extend_seq_lens_cpu=query_lengths,
            extend_prefix_lens=prefixes,
            extend_start_loc=torch.nn.functional.pad(lengths.cumsum(0)[:-1], (1, 0)),
            extend_num_tokens=sum(query_lengths),
            mamba_cache_src_indices=torch.tensor(source, device="cuda"),
            mamba_cache_dst_indices=torch.tensor(destination, device="cuda"),
        )

    @staticmethod
    def _backend(conv, state):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            GDNKernelDispatcher,
        )
        from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend

        backend = object.__new__(GDNAttnBackend)
        backend.device = "cuda"
        backend.topk = 0
        backend.enable_state_routing = True
        backend.enable_unified_memory = False
        backend.conv_states_shape = conv.shape
        cache = SimpleNamespace(
            conv=[conv],
            temporal=state,
            replayssm_d=None,
            replayssm_k=None,
            replayssm_g=None,
        )
        backend.req_to_token_pool = SimpleNamespace(
            translate_mamba_indices=lambda indices: indices,
            mamba_pool=SimpleNamespace(enable_linear_replayssm=False),
            mamba2_layer_cache=lambda _: cache,
        )
        backend.kernel_dispatcher = GDNKernelDispatcher(
            LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.TRITON
        )
        return backend

    def test_mixed_preserves_decode_sources_and_matches_separate_kernels(self):
        torch.manual_seed(13)
        device = "cuda"
        dtype = torch.bfloat16
        heads, dim, channels = 2, 64, 384
        conv = torch.randn(12, channels, 4, device=device, dtype=dtype)
        state = torch.randn(12, heads, dim, dim, device=device, dtype=torch.float32)
        actual_conv, actual_state = conv.clone(), state.clone()
        expected_conv, expected_state = conv.clone(), state.clone()
        layer = SimpleNamespace(
            layer_id=0,
            conv_weights=torch.randn(channels, 4, device=device, dtype=dtype),
            bias=None,
            activation="silu",
            A_log=torch.randn(heads, device=device, dtype=torch.float32),
            dt_bias=torch.randn(heads, device=device, dtype=dtype),
            num_q_heads=heads,
            num_k_heads=heads,
            num_v_heads=heads,
            head_q_dim=dim,
            head_k_dim=dim,
            head_v_dim=dim,
            q_dim=heads * dim,
            k_dim=heads * dim,
            v_dim=heads * dim,
        )
        qkv = torch.randn(10, channels, device=device, dtype=dtype)
        gates = torch.randn(10, 2 * heads, device=device, dtype=dtype)
        a, b = gates.split(heads, dim=-1)

        mixed = self._batch(
            [5, 3, 1, 1], [0, 3, 11, 17], [2, 3, 4, 6], [2, 3, 5, 7], ForwardMode.MIXED
        )
        mixed.decoupled_draft_num_prefill_reqs = 2
        mixed.decoupled_draft_num_prefill_tokens = 8
        prefill = self._batch([5, 3], [0, 3], [2, 3], [2, 3], ForwardMode.EXTEND)
        decode = self._batch([1, 1], [11, 17], [4, 6], [5, 7], ForwardMode.DECODE)

        with get_context().override_server_args():
            backend = self._backend(actual_conv, actual_state)
            backend.init_forward_metadata(mixed)
            actual = backend.forward_extend(layer, mixed, qkv.clone(), a, b)

            reference = self._backend(expected_conv, expected_state)
            reference.init_forward_metadata(prefill)
            expected_prefill = reference.forward_extend(
                layer, prefill, qkv[:8].clone(), a[:8], b[:8]
            )
            reference.init_forward_metadata(decode)
            expected_decode = reference.forward_decode(
                layer, decode, qkv[8:].clone(), a[8:], b[8:]
            )

        torch.testing.assert_close(
            actual, torch.cat((expected_prefill, expected_decode), dim=1)
        )
        torch.testing.assert_close(actual_conv, expected_conv)
        torch.testing.assert_close(actual_state, expected_state)
        torch.testing.assert_close(actual_conv[[4, 6]], conv[[4, 6]], rtol=0, atol=0)
        torch.testing.assert_close(actual_state[[4, 6]], state[[4, 6]], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

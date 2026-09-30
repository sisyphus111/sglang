"""GPU-owned mixed-prefix metadata and GDN phase dispatch, without GPU kernels."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase

with (
    patch.object(
        torch.ops.sgl_kernel, "fused_sigmoid_gating_delta_rule_update_cpu", create=True
    ),
    patch.object(torch.ops.sgl_kernel, "fused_gdn_gating_cpu", create=True),
):
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import (
    _forward_input_buffer_pool,
    share_input_buffer,
)
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as decode_graph
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDecoupledMixedPrefill(CustomTestCase):
    def setUp(self):
        context = get_context().override_server_args(page_size=1)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        parallel = get_parallel().override(moe_ep_size=1)
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)
        model_config = patch.object(
            ServerArgs,
            "get_model_config",
            return_value=SimpleNamespace(hf_config=SimpleNamespace(mamba_chunk_size=2)),
        )
        model_config.start()
        self.addCleanup(model_config.stop)

    def test_decode_graph_does_not_inherit_drafter_prefill_tracking(self):
        for role in ("null", "drafter"):
            with (
                self.subTest(role=role),
                get_context().override_server_args(
                    decoupled_spec_role=role,
                    disable_radix_cache=False,
                    mamba_radix_cache_strategy="extra_buffer",
                ),
                get_parallel().override(attn_tp_size=1, attn_tp_rank=0),
                patch.dict(_forward_input_buffer_pool, clear=True),
            ):
                # Eager prefill publishes a live radix-tracking mask into the
                # shared pool. Drafter decode must not capture that mask.
                prefill_mask = share_input_buffer(
                    "mamba_track_mask", torch.ones(4, dtype=torch.bool)
                )
                model_runner = SimpleNamespace(
                    device="cpu",
                    server_args=get_server_args(),
                    is_draft_worker=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=SimpleNamespace(
                        is_encoder_decoder=False,
                        hidden_size=8,
                        vocab_size=16,
                        dtype=torch.float32,
                    ),
                    ngram_embedding_manager=SimpleNamespace(enabled=False),
                    attn_backend=MagicMock(),
                    graph_shared_output=MagicMock(),
                    decode_num_tokens_per_req=lambda **kwargs: 1,
                    get_pp_proxy_topk_size=lambda: None,
                    get_pp_proxy_residual_num_blocks=lambda: None,
                )
                model_runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = (
                    1
                )
                model_runner.graph_shared_output.get_logits_buffer.return_value = (
                    torch.zeros(4, 16)
                )
                with (
                    patch.object(
                        decode_graph,
                        "get_batch_sizes_to_capture",
                        return_value=([1, 4], []),
                    ),
                    patch.object(
                        decode_graph, "mambaish_config", return_value=object()
                    ),
                    patch.object(
                        decode_graph.DecodeCudaGraphRunner,
                        "_cache_loc_dtype",
                        return_value=torch.int64,
                    ),
                    patch.object(decode_graph.DecodeCudaGraphRunner, "capture"),
                    patch.object(decode_graph, "resolve_decode_backend"),
                ):
                    runner = decode_graph.DecodeCudaGraphRunner(model_runner)
                if role == "drafter":
                    self.assertIsNone(runner.buffers.mamba_track_mask)
                    self.assertIsNone(runner.buffers.mamba_track_indices)
                    self.assertFalse(
                        runner.buffer_registry.has_slot("mamba_track_mask")
                    )
                else:
                    self.assertEqual(
                        runner.buffers.mamba_track_mask.data_ptr(),
                        prefill_mask.data_ptr(),
                    )
                    self.assertTrue(runner.buffer_registry.has_slot("mamba_track_mask"))

    @staticmethod
    def _batch():
        reqs = [MagicMock(rid=f"req-{index}", lora_id=None) for index in range(4)]
        return ScheduleBatch(
            reqs=reqs,
            model_config=MagicMock(),
            device="cpu",
            spec_algorithm=SpeculativeAlgorithm.NONE,
            forward_mode=ForwardMode.MIXED,
            input_ids=torch.arange(7),
            req_pool_indices=torch.arange(4),
            seq_lens=torch.tensor([7, 2, 10, 4]),
            seq_lens_cpu=torch.tensor([70, 20, 100, 40]),
            orig_seq_lens=torch.tensor([7, 2, 10, 4], dtype=torch.int32),
            out_cache_loc=torch.arange(7),
            extend_lens=[3, 2, 1, 1],
            prefix_lens=[40, 0, 90, 30],
            extend_num_tokens=7,
            multimodal_inputs=[None] * 4,
            decoupled_draft_num_prefill_reqs=2,
            decoupled_draft_num_prefill_tokens=5,
            mamba_cache_src_indices=torch.tensor([2, 3, 4, 6]),
            mamba_cache_dst_indices=torch.tensor([2, 3, 5, 7]),
        )

    def _forward_batch(self, batch=None):
        runner = MagicMock()
        runner.device = torch.device("cpu")
        runner.model_config.model_is_mrope = False
        runner.ngram_embedding_manager.enabled = False
        runner.server_args.enable_lora = False
        runner.dcp_size = 1
        with patch.object(
            forward_batch_info,
            "compute_position",
            side_effect=lambda _, prefix, lengths, __: (
                forward_batch_info.compute_position_torch(prefix, lengths)
            ),
        ):
            return ForwardBatch.init_new(
                self._batch() if batch is None else batch,
                runner,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_hidden_states_before_norm=False,
            )

    def test_positions_ignore_stale_cpu_decode_prefix(self):
        batch = self._batch()
        forward = self._forward_batch(batch)
        torch.testing.assert_close(
            forward.extend_prefix_lens, torch.tensor([4, 0, 9, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            forward.positions, torch.tensor([4, 5, 6, 0, 1, 9, 3])
        )
        self.assertIsNone(forward.seq_lens_cpu)
        self.assertIsNone(forward.extend_prefix_lens_cpu)
        self.assertEqual(forward.extend_seq_lens_cpu, [3, 2, 1, 1])
        self.assertEqual(batch.prefix_lens, [40, 0, 90, 30])

    def test_rejects_misaligned_boundaries(self):
        # Backend capability is checked once when the drafter starts.
        for overrides in (
            {"decoupled_draft_num_prefill_tokens": None},
            {"decoupled_draft_num_prefill_tokens": 4},
            {"extend_lens": [3, 1, 2, 1]},
        ):
            with self.subTest(overrides=overrides):
                batch = self._batch()
                for name, value in overrides.items():
                    setattr(batch, name, value)
                with self.assertRaises(ValueError):
                    self._forward_batch(batch)

    def test_mixed_does_not_replay_a_pure_prefill_graph(self):
        runner = object.__new__(PrefillCudaGraphRunner)
        runner.can_replay_locally = MagicMock(return_value=True)
        forward = self._forward_batch()
        self.assertFalse(runner.can_run_graph(forward))
        runner.can_replay_locally.assert_not_called()

    def test_flashattention_builds_query_offsets_without_cpu_prefix(self):
        forward = self._forward_batch()
        backend = object.__new__(FlashAttentionBackend)
        backend.max_context_len = 32
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(128, dtype=torch.int32).view(4, 32)
        )
        backend.attn_cp_size = 1
        backend.is_prefill_aware_swa = False
        backend.use_sliding_window_kv_pool = False
        backend._unified_dense = False
        backend.page_size = 1
        for version in (3, 4):
            with self.subTest(version=version):
                backend.fa_impl_ver = version
                backend.init_forward_metadata(forward)
                metadata = backend.forward_metadata
                torch.testing.assert_close(
                    metadata.cu_seqlens_q,
                    torch.tensor([0, 3, 5, 6, 7], dtype=torch.int32),
                )
                torch.testing.assert_close(
                    metadata.cache_seqlens_int32,
                    torch.tensor([7, 2, 10, 4], dtype=torch.int32),
                )
                self.assertEqual(metadata.max_seq_len_q, 3)
                self.assertEqual(metadata.max_seq_len_k, 32)

    def test_gdn_splits_tracking_and_checkpoint_routes_once_per_forward(self):
        forward = self._forward_batch()
        forward.mamba_track_mask = torch.tensor([True, False, False, False])
        forward.mamba_track_indices = torch.tensor([12, 0, 0, 0])
        forward.mamba_track_seqlens = torch.tensor([6, 0, 0, 0])
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.topk = 0
        backend.enable_state_routing = True
        backend.enable_unified_memory = False
        backend.conv_states_shape = (16, 6, 2)
        layer_cache = SimpleNamespace(
            conv=[torch.zeros(16, 6, 2)],
            temporal=torch.zeros(16, 1, 2, 2),
            replayssm_d=None,
            replayssm_k=None,
            replayssm_g=None,
        )
        backend.req_to_token_pool = SimpleNamespace(
            translate_mamba_indices=lambda indices: indices,
            mamba_pool=SimpleNamespace(enable_linear_replayssm=False),
            mamba2_layer_cache=lambda _: layer_cache,
        )
        backend.kernel_dispatcher = MagicMock(
            extend_uses_state_checkpoints=False, supports_packed_decode=True
        )
        backend.init_forward_metadata(forward)
        prefill, decode, decode_metadata = backend._decoupled_mixed_forward
        prefill_metadata = backend.forward_metadata
        self.assertEqual(prefill.batch_size, 2)
        self.assertEqual(decode.batch_size, 2)
        self.assertIsNone(decode.mamba_track_mask)
        self.assertIsNone(prefill.seq_lens_cpu)
        self.assertIsNone(prefill.extend_prefix_lens_cpu)
        self.assertIsNone(decode.seq_lens_cpu)
        self.assertIsNone(decode.extend_seq_lens_cpu)
        torch.testing.assert_close(
            prefill_metadata.track_ssm_final_dst, torch.tensor([12])
        )
        torch.testing.assert_close(
            decode_metadata.mamba_cache_src_indices, torch.tensor([4, 6])
        )
        torch.testing.assert_close(
            decode_metadata.mamba_cache_indices, torch.tensor([5, 7])
        )
        torch.testing.assert_close(
            decode_metadata.query_start_loc, torch.tensor([0, 1, 2], dtype=torch.int32)
        )

        qkv = torch.arange(42, dtype=torch.float32).view(7, 6)
        layer = SimpleNamespace(
            layer_id=0,
            conv_weights=None,
            bias=None,
            activation="silu",
            A_log=None,
            dt_bias=None,
            head_k_dim=2,
            head_q_dim=2,
            head_v_dim=2,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            q_dim=2,
            k_dim=2,
            v_dim=2,
        )
        backend.kernel_dispatcher.extend.side_effect = lambda **args: (
            args["q"],
            None,
            None,
        )
        backend.kernel_dispatcher.packed_decode.side_effect = lambda **args: args[
            "mixed_qkv"
        ][:, :2].reshape(1, 2, 1, 2)
        backend._track_mamba_state_extend = MagicMock()
        with (
            patch.object(
                gdn_backend,
                "causal_conv1d_fn",
                side_effect=lambda x, *_args, **_kwargs: x,
            ),
            patch.object(
                gdn_backend,
                "causal_conv1d_update",
                side_effect=lambda x, *_args, **_kwargs: x,
            ) as decode_conv,
            patch.object(gdn_backend, "fused_gdn_gating", return_value=(None, None)),
            patch.object(gdn_backend, "is_cuda", return_value=False),
            patch.object(gdn_backend, "is_hip", return_value=False),
        ):
            output = backend.forward_extend(layer, forward, qkv, qkv[:, :1], qkv[:, :1])
        torch.testing.assert_close(output, qkv[:, :2].reshape(1, 7, 1, 2))
        self.assertIs(backend.forward_metadata, prefill_metadata)
        torch.testing.assert_close(
            decode_conv.call_args.kwargs["conv_state_src_indices"], torch.tensor([4, 6])
        )
        torch.testing.assert_close(
            backend.kernel_dispatcher.packed_decode.call_args.kwargs[
                "final_state_indices"
            ],
            torch.tensor([5, 7]),
        )

    def test_replacing_eager_or_graph_metadata_releases_mixed_batch_storage(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                backend = object.__new__(GDNAttnBackend)
                previous = self._forward_batch()
                previous.input_embeds = torch.empty(128, 16)
                retained = weakref.ref(previous.input_embeds)
                backend._decoupled_mixed_forward = (previous, previous, None)
                del previous
                self.assertIsNotNone(retained())

                next_batch = SimpleNamespace(decoupled_draft_num_prefill_reqs=None)
                backend.forward_metadata = SimpleNamespace(has_mamba_track_mask=False)
                initializer = (
                    "init_forward_metadata_out_graph"
                    if graph
                    else "init_forward_metadata"
                )
                with patch.object(MambaAttnBackendBase, initializer):
                    getattr(backend, initializer)(next_batch)

                self.assertIsNone(backend._decoupled_mixed_forward)
                self.assertIsNone(retained())


if __name__ == "__main__":
    unittest.main()

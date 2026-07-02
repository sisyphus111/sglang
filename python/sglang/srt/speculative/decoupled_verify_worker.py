from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
)
from sglang.srt.speculative.spec_info import dynamic_verify_enabled
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.async_probe import maybe_detect_nan

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput

logger = logging.getLogger(__name__)


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external draft verification."
    )


def _normalize_token_id(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_token_id(item)
            if normalized is not None:
                return normalized
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_linear_topk1_tree_metadata(
    batch_size: int,
    spec_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_index = (
        torch.arange(
            spec_steps,
            dtype=torch.long,
            device=device,
        )
        .expand(batch_size, -1)
        .contiguous()
    )

    if spec_steps <= 1:
        parent_list = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    else:
        parent_list = (
            torch.arange(
                -1,
                spec_steps - 1,
                dtype=torch.long,
                device=device,
            )
            .expand(batch_size, -1)
            .contiguous()
        )

    return selected_index, parent_list


class VerifyWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del gpu_id, moe_ep_rank, moe_dp_rank, nccl_port
        self.server_args = server_args
        self.target_worker = target_worker
        self.tp_rank = int(tp_rank)
        self.attn_cp_rank = int(attn_cp_rank)
        self.dp_rank = 0 if dp_rank is None else int(dp_rank)
        self.pp_rank = int(getattr(target_worker, "pp_rank", 0))
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.page_size = server_args.page_size
        self.topk = 1
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(
            server_args.speculative_num_draft_tokens
        )
        self.dynamic_verify_length = dynamic_verify_enabled(server_args)
        self.device = self.model_runner.device
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.total_accept_length = 0
        self.total_num_verified_reqs = 0

    def clear_cache_pool(self):
        return

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool
        if token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        return

    def init_cuda_graphs(self):
        return

    def on_verify_complete_cpu(self, *args, **kwargs):
        return

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _decoupled_verify_runtime_state(self, batch: ScheduleBatch):
        return getattr(batch, "decoupled_verify_runtime_state", None)

    def _get_verify_buffers(self, draft_token_num: int, batch: ScheduleBatch | None):
        if draft_token_num > self.speculative_num_draft_tokens:
            return None, None
        if (
            draft_token_num != self.speculative_num_draft_tokens
            and not self.dynamic_verify_length
        ):
            return None, None

        runtime_state = (
            self._decoupled_verify_runtime_state(batch) if batch is not None else None
        )
        attn_backend = (
            runtime_state.target_attn_backend
            if runtime_state is not None
            and runtime_state.target_attn_backend is not None
            else getattr(self.target_worker.model_runner, "attn_backend", None)
        )
        if attn_backend is None:
            return None, None

        get_buffers = getattr(
            attn_backend, "get_verify_buffers_to_fill_after_draft", None
        )
        if get_buffers is None:
            return None, None

        try:
            return get_buffers()
        except Exception as exc:
            logger.debug("Falling back to eager verify buffers: %s", exc)
            return None, None

    def _get_pad_token_id(self) -> int:
        """Return an EOS token id used to pad short external draft tails."""
        hf_generation_config = getattr(self.model_config, "hf_generation_config", None)
        eos_token_id = _normalize_token_id(
            getattr(hf_generation_config, "eos_token_id", None)
        )
        if eos_token_id is not None:
            return eos_token_id

        hf_config = getattr(self.model_config, "hf_config", None)
        eos_token_id = _normalize_token_id(getattr(hf_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        get_text_config = getattr(hf_config, "get_text_config", None)
        text_config = (
            get_text_config()
            if callable(get_text_config)
            else getattr(hf_config, "text_config", None)
        )
        eos_token_id = _normalize_token_id(getattr(text_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        eos_token_ids = getattr(self.model_config, "hf_eos_token_id", None)
        if eos_token_ids:
            return min(int(token_id) for token_id in eos_token_ids)

        raise RuntimeError("External draft verification requires an EOS token id.")

    def _decoupled_verify_shape(self, batch: ScheduleBatch):
        runtime_state = self._decoupled_verify_runtime_state(batch)
        if runtime_state is not None:
            return runtime_state.shape
        return getattr(batch, "decoupled_verify_shape", None)

    def _effective_verify_shape(self, batch: ScheduleBatch) -> tuple[int, int]:
        shape = self._decoupled_verify_shape(batch)
        if shape is None:
            return self.speculative_num_steps, self.speculative_num_draft_tokens
        return int(shape.num_speculative_steps), int(shape.verify_tokens_per_req)

    def _build_req_verify_tokens(
        self, req, pad_token_id: int, spec_depth: int
    ) -> list[int]:
        tail_token = _get_req_tail_token_id(req)
        draft_buffer = list(getattr(req, "draft_buffer", []) or [])
        draft_tokens = list(draft_buffer[:spec_depth])
        if len(draft_tokens) < spec_depth:
            draft_tokens.extend([int(pad_token_id)] * (spec_depth - len(draft_tokens)))
        return [tail_token, *draft_tokens]

    def _get_snapshot_tail_lens(
        self, batch: ScheduleBatch, spec_depth: int
    ) -> list[int]:
        return [
            min(
                len(list(getattr(req, "draft_buffer", []) or [])),
                spec_depth,
            )
            for req in batch.reqs
        ]

    def _assert_num_correct_within_snapshot_tail(
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: list[int]
    ) -> list[int]:
        # req.draft_buffer is a per-forward snapshot bound before verify. Any
        # concurrent drafter appends belong to later verify rounds.
        spec_steps, _ = self._effective_verify_shape(batch)
        real_tail_lens = self._get_snapshot_tail_lens(batch, spec_steps)
        raw_accept_lens = [int(x) for x in num_correct_drafts_per_req_cpu]
        for req, raw_accept_len, real_tail_len in zip(
            batch.reqs, raw_accept_lens, real_tail_lens
        ):
            assert raw_accept_len <= real_tail_len, (
                "Decoupled verify has accepted padded draft tokens: "
                f"request_id={req.rid} "
                f"raw_accept_len={raw_accept_len} "
                f"snapshot_tail_len={real_tail_len}"
            )

        return raw_accept_lens

    def _record_valid_draft_metrics(
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: list[int]
    ) -> None:
        spec_steps, _ = self._effective_verify_shape(batch)
        for req, accepted_drafts in zip(batch.reqs, num_correct_drafts_per_req_cpu):
            valid_draft_tokens = min(
                len(list(getattr(req, "draft_buffer", []) or [])), spec_steps
            )
            valid_accepted_tokens = min(int(accepted_drafts), valid_draft_tokens)
            req.spec_valid_draft_tokens += valid_draft_tokens
            req.spec_valid_accepted_tokens += valid_accepted_tokens
            metric_len = max(
                spec_steps,
                len(req.spec_valid_draft_tokens_by_position),
                len(req.spec_valid_accepted_tokens_by_position),
            )
            req.spec_valid_draft_tokens_by_position = (
                req.spec_valid_draft_tokens_by_position + [0] * metric_len
            )[:metric_len]
            req.spec_valid_accepted_tokens_by_position = (
                req.spec_valid_accepted_tokens_by_position + [0] * metric_len
            )[:metric_len]
            for pos in range(valid_draft_tokens):
                req.spec_valid_draft_tokens_by_position[pos] += 1
            for pos in range(valid_accepted_tokens):
                req.spec_valid_accepted_tokens_by_position[pos] += 1

    def _forward_decode_as_zero_step(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        shape = self._decoupled_verify_shape(batch)
        if shape is None or not shape.uses_decode_graph:
            raise RuntimeError(
                "zero-step decoupled verifier decode fallback requires a "
                "dynamic verify shape with num_speculative_steps == 0."
            )
        log_key = (int(shape.captured_batch_size), int(shape.verify_tokens_per_req))
        if log_key != getattr(self, "_last_zero_step_decode_graph_log_key", None):
            setattr(self, "_last_zero_step_decode_graph_log_key", log_key)
            log_info_on_rank0(
                logger,
                "Decoupled verifier dynamic verify selected zero speculative "
                "steps; running decode CUDA Graph: "
                f"raw_bs={shape.raw_batch_size}, "
                f"captured_bs={shape.captured_batch_size}, "
                f"verify_tokens_per_req={shape.verify_tokens_per_req}, "
                f"padded_verify_tokens={shape.padded_verify_tokens}, "
                f"budget={shape.budget}",
            )

        batch.spec_info = None
        batch.return_hidden_states = False
        batch.forward_mode = ForwardMode.DECODE
        batch.input_ids = torch.tensor(
            [_get_req_tail_token_id(req) for req in batch.reqs],
            dtype=torch.int64,
            device=batch.device,
        )
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1
        batch.seq_lens.add_(1)
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu.add_(1)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

        batch_result = self.target_worker.forward_batch_generation(batch)
        if self.dynamic_verify_length and not batch_result.can_run_cuda_graph:
            raise RuntimeError(
                "Decoupled verifier dynamic verify length selected zero "
                "speculative steps, but the fallback decode forward did not use "
                f"CUDA Graph: shape={shape}"
            )

        next_token_ids_obj = batch_result.next_token_ids
        if isinstance(next_token_ids_obj, torch.Tensor):
            next_token_ids = [int(x) for x in next_token_ids_obj.tolist()]
        else:
            next_token_ids = [int(x) for x in next_token_ids_obj]
        if len(next_token_ids) != len(batch.reqs):
            raise RuntimeError(
                "zero-step decoupled verifier decode returned unexpected token "
                "count: "
                f"num_tokens={len(next_token_ids)}, batch_size={len(batch.reqs)}"
            )

        batch_result.num_correct_drafts = 0
        batch_result.num_correct_drafts_per_req_cpu = [0] * len(batch.reqs)
        batch_result.next_token_ids = (
            next_token_ids_obj
            if isinstance(next_token_ids_obj, torch.Tensor)
            else torch.tensor(next_token_ids, dtype=torch.long, device=batch.device)
        )
        return batch_result

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        spec_steps, draft_token_num = self._effective_verify_shape(batch)
        if draft_token_num < 1:
            raise RuntimeError(
                "External draft verification requires at least one verify token per request."
            )

        if batch.forward_mode.is_idle():
            spec_info = EagleVerifyInput.create_idle_input(
                self.topk,
                spec_steps,
                draft_token_num,
            )
            # Decoupled verify does not consume target hidden states. Keep idle
            # companion ranks on the same NULL-hidden CUDA graph as active ranks.
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
            setattr(
                spec_info,
                "decoupled_verify_runtime_state",
                getattr(batch, "decoupled_verify_runtime_state", None),
            )
            setattr(
                spec_info,
                "decoupled_verify_shape",
                self._decoupled_verify_shape(batch),
            )
            return spec_info

        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1
        seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        batch.seq_lens_sum = seq_lens_sum

        # Accumulate penalty
        sampling_info = getattr(batch, "sampling_info", None)
        penalizer_orchestrator = getattr(sampling_info, "penalizer_orchestrator", None)
        if (
            penalizer_orchestrator is not None
            and penalizer_orchestrator.is_required
            and batch.reqs
        ):
            penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(
                    [_get_req_tail_token_id(req) for req in batch.reqs],
                    dtype=torch.int64,
                    device=batch.device,
                )
            )

        pad_token_id = self._get_pad_token_id()

        full_draft_tokens_by_req = [
            self._build_req_verify_tokens(req, pad_token_id, spec_steps)
            for req in batch.reqs
        ]
        bonus_tokens = torch.tensor(
            [tokens[0] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens = torch.tensor(
            [tokens[1:] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )

        batch_size = batch.batch_size()
        selected_index, parent_list = _build_linear_topk1_tree_metadata(
            batch_size,
            spec_steps,
            batch.device,
        )

        tree_mask_buf, position_buf = self._get_verify_buffers(
            draft_token_num, batch=batch
        )

        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            bonus_tokens=bonus_tokens,
            parent_list=parent_list,
            top_scores_index=selected_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=1,
            spec_steps=spec_steps,
            num_verify_tokens=draft_token_num,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        terminal_indices = torch.tensor(
            self._get_snapshot_tail_lens(batch, spec_steps),
            dtype=torch.long,
            device=batch.device,
        )
        row_indices = torch.arange(batch_size, dtype=torch.long, device=batch.device)
        terminal_indices = torch.clamp(terminal_indices, max=draft_token_num - 1)
        retrieve_next_token[row_indices, terminal_indices] = -1

        spec_info = EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=spec_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )
        setattr(
            spec_info,
            "decoupled_verify_runtime_state",
            getattr(batch, "decoupled_verify_runtime_state", None),
        )
        setattr(
            spec_info,
            "decoupled_verify_shape",
            self._decoupled_verify_shape(batch),
        )
        return spec_info

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
    ):
        was_idle = batch.forward_mode.is_idle()
        seq_lens_pre_verify = batch.seq_lens.clone()

        spec_info.num_tokens_per_req = spec_info.draft_token_num
        batch.return_hidden_states = False
        batch.spec_info = spec_info

        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            spec_info,
            self.req_to_token_pool,
            batch,
            self.target_worker,
            capture_hidden_mode=spec_info.capture_hidden_mode,
            allocate_verify_slots=True,
            page_size=self.page_size,
        )
        assert verify_forward_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrieve_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrieve_next_token.shape
            ).cpu()

        batch_result = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            bool(batch_result.can_run_cuda_graph or can_run_cuda_graph),
        )

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(
            logits_output.next_token_logits, "decoupled_verify: target model logits"
        )

        predict, accept_lens, accept_index = eagle_sample(
            spec_info, batch, logits_output, vocab_mask
        )
        num_correct_drafts_per_req_cpu = (accept_lens - 1).cpu().tolist()
        self._assert_num_correct_within_snapshot_tail(
            batch, num_correct_drafts_per_req_cpu
        )
        self._record_valid_draft_metrics(batch, num_correct_drafts_per_req_cpu)

        if not was_idle:
            if self.page_size != 1:
                raise RuntimeError(
                    "Decoupled verifier currently requires page_size == 1."
                )
            accepted_indices = accept_index[accept_index != -1]
            evict_mask = torch.full_like(
                spec_info.draft_token, True, dtype=torch.bool
            )
            evict_mask[accepted_indices] = False
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[accepted_indices]

        if (not was_idle) and (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            commit_mamba_states_after_verify(
                self.target_worker,
                batch,
                accept_lens,
                accept_index,
                spec_info.draft_token_num,
            )

        if batch.return_logprob:
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, spec_info.spec_steps
            )

        batch.forward_mode = ForwardMode.IDLE if was_idle else ForwardMode.DECODE
        # Decoupled verify rebuilds verify inputs from fresh external draft
        # snapshots each round, so there is no in-process draft state to carry.
        batch.spec_info = None
        return (
            logits_output,
            predict,
            accept_lens,
            num_correct_drafts_per_req_cpu,
            seq_lens_pre_verify + accept_lens,
            can_run_cuda_graph,
        )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        # When a peer DP rank is running prefill/extend, IDLE batches are just
        # DP-attention companions. They must preserve the plain target-forward
        # MLP sync shape instead of installing an idle verify spec_info.
        if not batch.forward_mode.is_target_verify() and (
            batch.forward_mode.is_extend() or batch.is_extend_in_batch
        ):
            model_worker_batch = batch.get_model_worker_batch()
            result = self.target_worker.forward_batch_generation(model_worker_batch)
            return result

        shape = self._decoupled_verify_shape(batch)
        if (
            self.dynamic_verify_length
            and shape is not None
            and shape.uses_decode_graph
            and batch.forward_mode.is_decode_or_idle()
        ):
            result = self._forward_decode_as_zero_step(batch)
            num_verified_reqs = len(result.num_correct_drafts_per_req_cpu or [])
            self.total_num_verified_reqs += num_verified_reqs
            return result

        spec_info = self.draft(batch)
        (
            logits_output,
            predict,
            accept_lens,
            num_correct_drafts_per_req_cpu,
            new_seq_lens,
            can_run_cuda_graph,
        ) = self.verify(batch, spec_info)

        num_correct_drafts = sum(num_correct_drafts_per_req_cpu)
        reported_can_run_cuda_graph = can_run_cuda_graph
        if self.dynamic_verify_length and not reported_can_run_cuda_graph:
            shape = self._decoupled_verify_shape(batch)
            raise RuntimeError(
                "Decoupled verifier dynamic verify length requires full CUDA "
                "Graph replay for target verify, but this forward ran without it: "
                f"shape={shape}, draft_token_num={spec_info.draft_token_num}"
            )

        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            num_correct_drafts=num_correct_drafts,
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=reported_can_run_cuda_graph,
            speculative_num_draft_tokens=spec_info.draft_token_num,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
        )
        num_verified_reqs = len(num_correct_drafts_per_req_cpu)
        self.total_accept_length += int(result.num_correct_drafts)
        self.total_num_verified_reqs += num_verified_reqs
        return result

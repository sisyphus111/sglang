from __future__ import annotations

import logging
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info_v2 import EagleDraftInputV2Mixin
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    align_evict_mask_to_page_size,
    assign_req_to_token_pool_func,
    create_num_accept_tokens_filter,
    filter_finished_cache_loc_kernel,
    generate_simulated_accept_index,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_musa, next_power_of_2

if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


def _draft_runner_of(worker):
    """Draft model_runner accessor across worker shapes.

    v2 draft workers (`EagleDraftWorker` and subclasses) expose the draft
    model_runner as `draft_runner`; fall back to `model_runner` for workers
    that run the draft model directly.
    """
    return (
        worker.draft_runner if hasattr(worker, "draft_runner") else worker.model_runner
    )


@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    retrieve_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None
    # Stacked per-step draft proposal distribution q, shape (bs, num_steps,
    # vocab); only set under rejection sampling. Consumed by the verify kernel.
    draft_probs: torch.Tensor = None

    # Shape info for padding
    num_tokens_per_req: int = -1  # -1 auto-fills from draft_token_num.

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        if self.num_tokens_per_req < 0:
            self.num_tokens_per_req = self.draft_token_num

    @property
    def max_tree_depth(self) -> int:
        """Longest root-to-leaf chain of the verify tree, incl. the root;
        bounds the accept_index row width. EAGLE trees are depth-bounded by
        the draft loop. Algorithms with other tree shapes override this."""
        return self.spec_steps + 1

    @property
    def tree_topk(self) -> int:
        """Branching factor passed to the tree-verify kernels; -1 means an
        irregular tree (no fixed per-level branching)."""
        return self.topk

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrieve_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrieve_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,
    ):
        if batch.forward_mode.is_idle():
            draft_extend_input = EagleDraftExtendInput.create_idle_input(
                device=batch.device,
                hidden_size=None,
                dtype=None,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )
            return EagleVerifyOutput.create_idle(
                draft_extend_input=draft_extend_input,
                logits_output=logits_output,
                device=batch.device,
                spec_steps=self.spec_steps,
            )

        bs = self.retrieve_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            sampling_info.filter_batch(
                self.retrieve_index.tolist(), self.retrieve_index
            )

        if sampling_info.has_custom_logit_processor:
            from sglang.srt.layers.sampler import apply_custom_logit_processor

            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            sampling_info.penalizer_orchestrator.apply(
                logits_output.next_token_logits, repeat=self.draft_token_num
            )
            if sampling_info.logit_bias is not None:
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(
                        sampling_info.logit_bias, self.draft_token_num, dim=0
                    )
                )

        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable. Falling back to "
                "greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=num_correct_drafts,
                candidates=candidates,
                retrieve_index=self.retrieve_index,
                retrieve_next_token=self.retrieve_next_token,
                retrieve_next_sibling=self.retrieve_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )
            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )
            coins = torch.rand_like(
                candidates, dtype=torch.float32, device=batch.device
            )
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            tree_speculative_sampling_target_only(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=num_correct_drafts,
                candidates=candidates,
                retrive_index=self.retrieve_index,
                retrive_next_token=self.retrieve_next_token,
                retrive_next_sibling=self.retrieve_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)

        if SIMULATE_ACC_LEN > 0.0:
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,
                num_correct_drafts=num_correct_drafts,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False
        think_end_id = batch.model_config.think_end_id

        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            num_accept_tokens = 0
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                num_accept_tokens += 1
                token_id = predict_cpu[idx]
                req.output_ids.append(token_id)
                if req.require_reasoning and think_end_id is not None:
                    req.update_reasoning_tokens(token_id, think_end_id)
                req.check_finished()
                if not req.finished() and req.grammar is not None:
                    try:
                        req.grammar.accept_token(token_id)
                    except ValueError as exc:
                        logger.info(
                            f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                        )
                        raise exc
                    req.check_finished()
                if req.finished():
                    has_finished = True
                    accept_index[i, j + 1 :] = -1
                    break

            req.kv_committed_len += num_accept_tokens
            req.kv_allocated_len = req.kv_committed_len
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            num_correct_drafts_this_req = (
                sum(1 for idx in accept_index_row if idx != -1) - 1
            )
            req.spec_num_correct_drafts += num_correct_drafts_this_req
            req.update_spec_correct_drafts_histogram(num_correct_drafts_this_req)

        if has_finished:
            num_correct_drafts = (accept_index != -1).sum(dim=1) - 1

        accept_index = accept_index[accept_index != -1]
        accept_tokens = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        num_correct_drafts_cpu = num_correct_drafts.cpu()
        num_accept_tokens_cpu = num_correct_drafts_cpu + 1
        num_correct_drafts_list = num_correct_drafts_cpu.tolist()
        num_accept_tokens_list = num_accept_tokens_cpu.tolist()

        if page_size == 1:
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    num_correct_drafts,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    num_correct_drafts,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )
                token_to_kv_pool_allocator.free(to_free_slots)
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + num_correct_drafts + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(num_correct_drafts + 1)
            batch.seq_lens_cpu.add_(num_accept_tokens_cpu)

            draft_extend_input = EagleDraftExtendInput(
                hidden_states=(
                    batch.spec_info.hidden_states[accept_index]
                    if batch.spec_info.hidden_states is not None
                    else None
                ),
                num_correct_drafts=num_correct_drafts,
                num_accept_tokens=num_correct_drafts + 1,
                num_accept_tokens_cpu=num_accept_tokens_list,
                input_ids=accept_tokens,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                req_pool_indices=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_extend_input=draft_extend_input,
                logits_output=logits_output,
                accept_tokens=accept_tokens,
                num_correct_drafts_per_req_cpu=num_correct_drafts_list,
                accept_indices=accept_index,
            )

        if page_size == 1 or self.topk == 1:
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.seq_lens + num_correct_drafts + 1,
                batch.out_cache_loc[accept_index],
                bs,
            )
            batch.seq_lens.add_(num_correct_drafts + 1)
            batch.seq_lens_cpu.add_(num_accept_tokens_cpu)

        if len(unfinished_accept_index) > 0:
            unfinished_accept_index = torch.cat(unfinished_accept_index)
            unfinished_index_device = torch.tensor(
                unfinished_index, dtype=torch.int64, device=predict.device
            )
            draft_input_num_correct_drafts_cpu = [
                num_correct_drafts_list[i] for i in unfinished_index
            ]
            draft_input_num_accept_tokens_cpu = [
                num_accept_tokens_list[i] for i in unfinished_index
            ]
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
            else:
                batch.out_cache_loc = torch.empty(
                    len(unfinished_index) + sum(draft_input_num_correct_drafts_cpu),
                    dtype=torch.int64,
                    device=predict.device,
                )
                num_accept_tokens_filter = create_num_accept_tokens_filter(
                    num_correct_drafts,
                    unfinished_index_device,
                    batch.seq_lens,
                )
                batch.seq_lens_cpu.add_(num_accept_tokens_cpu)
                filter_finished_cache_loc_kernel[(bs,)](
                    batch.out_cache_loc,
                    tgt_cache_loc,
                    num_correct_drafts,
                    num_accept_tokens_filter,
                    next_power_of_2(bs),
                    next_power_of_2(self.draft_token_num),
                )

            unfinished_num_correct_drafts = num_correct_drafts[unfinished_index_device]
            draft_extend_input = EagleDraftExtendInput(
                hidden_states=(
                    batch.spec_info.hidden_states[unfinished_accept_index]
                    if batch.spec_info.hidden_states is not None
                    else None
                ),
                num_accept_tokens_cpu=draft_input_num_accept_tokens_cpu,
                num_correct_drafts=unfinished_num_correct_drafts,
                num_accept_tokens=unfinished_num_correct_drafts + 1,
                input_ids=predict[unfinished_accept_index],
                seq_lens=batch.seq_lens[unfinished_index_device],
                seq_lens_cpu=batch.seq_lens_cpu[unfinished_index],
                req_pool_indices=batch.req_pool_indices[unfinished_index_device],
            )
        else:
            draft_extend_input = EagleDraftExtendInput.create_idle_input(
                device=batch.device,
                hidden_size=None,
                dtype=None,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        return EagleVerifyOutput(
            draft_extend_input=draft_extend_input,
            logits_output=logits_output,
            accept_tokens=accept_tokens,
            num_correct_drafts_per_req_cpu=num_correct_drafts_list,
            accept_indices=accept_index,
        )


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    # For idle stubs use `create_idle_input`, not the bare ctor: `filter_batch`
    # / `merge_batch` slice / cat `topk_p` / `topk_index` / `hidden_states` /
    # `bonus_tokens` unconditionally.

    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, vocab) - single-step draft proposal q from draft-extend;
    # only set under rejection sampling.
    draft_probs: torch.Tensor = None
    # shape: (b, hidden_size) - one hidden per req, consumed by `draft` forward.
    # None when the spec algorithm's draft doesn't read hidden_states
    # (e.g., STANDALONE — vanilla LLM draft).
    hidden_states: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Per-req bonus token (the "+1" target prediction at end of each accept
    # chain); the worker copies it here post-extend for next iter's draft.
    bonus_tokens: torch.Tensor = None

    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = -1

    # V2 overlap worker only: req_pool_indices used as buf slot keys.
    future_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Decode-phase `hidden_states` width: draft self-chain output
        (draft model writes its own last hidden back via `capture_for_decode`
        and the draft loop). Returns None when the draft architecture doesn't
        consume the field (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.spec_hidden_size

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return _draft_runner_of(worker).model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
        vocab_size: int = 0,
    ):
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            draft_probs=(
                torch.empty((0, vocab_size), device=device, dtype=torch.float32)
                if get_global_server_args().speculative_use_rejection_sampling
                else None
            ),
            capture_hidden_mode=capture_hidden_mode,
        )

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            error_msg = f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            if self.draft_probs is not None:
                self.draft_probs = self.draft_probs[: len(new_indices)]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[: len(new_indices)]
            self.bonus_tokens = self.bonus_tokens[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            if self.draft_probs is not None:
                self.draft_probs = self.draft_probs[new_indices]
            if self.hidden_states is not None:
                self.hidden_states = self.hidden_states[new_indices]
            self.bonus_tokens = self.bonus_tokens[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            return

        # Detect idle stub by `topk_index` length (idle inputs have
        # shape[0] == 0 across all fields). Don't use `hidden_states is None`:
        # for STANDALONE all non-idle inputs also have None hidden_states.
        if len(self.topk_index) == 0:
            self.hidden_states = spec_info.hidden_states
            self.bonus_tokens = spec_info.bonus_tokens
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            self.draft_probs = spec_info.draft_probs
            return
        if len(spec_info.topk_index) == 0:
            return
        if self.hidden_states is not None and spec_info.hidden_states is not None:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], axis=0
            )
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], axis=0
        )
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])
        if self.draft_probs is not None and spec_info.draft_probs is not None:
            self.draft_probs = torch.cat([self.draft_probs, spec_info.draft_probs])


@dataclass
class EagleDraftExtendInput(SpecInput):
    """Inputs to the draft-extend forward (the fill-draft-kvcache pass after
    target prefill / verify).

    Installed on `batch.spec_info` by the worker's `_draft_extend_for_*`
    (and synthetically by draft-extend cuda-graph capture), then replaced
    with a fresh `EagleDraftInput` for the next iter's draft.
    """

    # Target-model hidden states for the draft-extend forward; None when the
    # draft doesn't read hidden_states (e.g., STANDALONE). Shape: decode
    # (bs * num_draft_tokens, hidden), prefill (extend_num_tokens, hidden).
    hidden_states: Optional[torch.Tensor] = None

    # Per-req accept counts. `num_accept_tokens = num_correct_drafts + 1`.
    # Both kept for cuda-graph buffer indexing.
    num_correct_drafts: torch.Tensor = None
    num_accept_tokens: torch.Tensor = None
    # CPU view, read by attention backends during the extend forward.
    num_accept_tokens_cpu: List[int] = None

    # Per-req batch-state slices for the draft-extend forward:
    #   - input_ids:        accept tokens flat over surviving reqs
    #   - seq_lens / _cpu:  per-req sequence length (post-accept)
    #   - req_pool_indices: per-req kv-pool slot
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    req_pool_indices: torch.Tensor = None

    #   - positions: shape `[total_accepted]`.
    #   - bonus_tokens: shape `[bs]`; read post-extend to populate next iter's
    #     `EagleDraftInput.bonus_tokens`.
    positions: Optional[torch.Tensor] = None
    bonus_tokens: Optional[torch.Tensor] = None

    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST
    num_tokens_per_req: int = -1
    num_tokens_for_logprob_per_req: int = 1

    # None for draft-extend's idle batch; attention backends fall back to
    # rebuilding plain metadata from seq_lens when this is None.
    kv_indptr: torch.Tensor = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT_EXTEND)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_req, self.num_tokens_for_logprob_per_req

    @classmethod
    def hidden_size_for(cls, worker) -> Optional[int]:
        """Extend-phase `hidden_states` width: target's `spec_hidden_size`,
        widened to `num_aux * target_hidden` for EAGLE-3 aux mode. Returns
        None when the draft architecture doesn't consume the field
        (e.g., STANDALONE)."""
        if worker.speculative_algorithm.is_standalone():
            return None
        target_cfg = worker.target_worker.model_runner.model_config
        if not (
            worker.speculative_algorithm.is_eagle3()
            and worker.eagle_use_aux_hidden_state
        ):
            return target_cfg.spec_hidden_size

        hf_config = target_cfg.hf_config

        # `num_aux` resolution: explicit attr > eagle_config layer_ids > default 3.
        num_aux = getattr(hf_config, "num_aux_hidden_states", None)
        if num_aux is None:
            eagle_config = getattr(hf_config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            num_aux = len(layer_ids) if layer_ids else 3

        target_hidden = getattr(hf_config, "target_hidden_size", target_cfg.hidden_size)
        return target_hidden * num_aux

    @classmethod
    def dtype_for(cls, worker) -> Optional[torch.dtype]:
        if worker.speculative_algorithm.is_standalone():
            return None
        return worker.target_worker.model_runner.model_config.dtype

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: Optional[int],
        dtype: Optional[torch.dtype],
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
    ) -> "EagleDraftExtendInput":
        return cls(
            hidden_states=(
                torch.empty((0, hidden_size), device=device, dtype=dtype)
                if hidden_size is not None
                else None
            ),
            num_correct_drafts=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            num_accept_tokens_cpu=[],
            input_ids=torch.empty((0,), device=device, dtype=torch.long),
            seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: Optional[int],
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = self.num_correct_drafts.numel()
        # Constant num_tokens_per_req qo layout (required for cuda-graph capture).
        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.num_tokens_per_req,
            step=self.num_tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None


@dataclass
class EagleVerifyOutput:
    draft_extend_input: EagleDraftExtendInput
    logits_output: LogitsProcessorOutput
    accept_tokens: torch.Tensor
    num_correct_drafts_per_req_cpu: List[int]
    accept_indices: torch.Tensor

    @classmethod
    def create_idle(
        cls,
        *,
        draft_extend_input: EagleDraftExtendInput,
        logits_output: LogitsProcessorOutput,
        device: torch.device,
        spec_steps: int,
    ):
        return cls(
            draft_extend_input=draft_extend_input,
            logits_output=logits_output,
            accept_tokens=torch.empty(0, dtype=torch.long, device=device),
            num_correct_drafts_per_req_cpu=[],
            accept_indices=torch.full(
                (0, spec_steps + 1), -1, dtype=torch.int32, device=device
            ),
        )

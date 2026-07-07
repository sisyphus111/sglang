import bisect
import contextlib
import dataclasses
import logging
import time
from array import array
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.server_args import ServerArgs
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import (
    build_decoupled_verify_profiled_adaptive_config,
    is_decoupled_verify_roofline_budget,
    resolve_decoupled_verify_adaptive_config_from_server_args,
    resolve_decoupled_verify_roofline_capture_bs_candidates,
    resolve_decoupled_verify_roofline_profile_bs_candidates,
    select_decoupled_verify_roofline_steps_by_bs,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    dynamic_verify_enabled,
)
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)

_ROOFLINE_PROFILE_WARMUP_ITERS = 5
_ROOFLINE_PROFILE_MEASURE_ITERS = 1000
_ROOFLINE_PROFILE_PLATEAU_RATIO = 0.95


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external "
        "draft verification."
    )


def _build_bonus_tokens_from_accepts(
    predict: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
) -> torch.Tensor:
    accept_tokens = predict[accept_index]
    row_indices = torch.arange(accept_lens.numel(), device=accept_lens.device)
    bonus_indices = torch.clamp(accept_lens.to(torch.long) - 1, min=0)
    return accept_tokens[row_indices, bonus_indices].to(dtype=torch.int32)


def _build_next_draft_input_stub(
    bonus_tokens: torch.Tensor,
    topk: int,
) -> EagleDraftInput:
    bonus_tokens = bonus_tokens.to(dtype=torch.int32)
    batch_size = int(bonus_tokens.numel())
    device = bonus_tokens.device
    return EagleDraftInput(
        bonus_tokens=bonus_tokens,
        topk_p=torch.zeros(
            (batch_size, int(topk)), device=device, dtype=torch.float32
        ),
        topk_index=torch.zeros(
            (batch_size, int(topk)), device=device, dtype=torch.int64
        ),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _normalize_token_id(value) -> Optional[int]:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected_index = (
        torch.arange(spec_steps, dtype=torch.long, device=device)
        .expand(batch_size, -1)
        .contiguous()
    )

    if spec_steps <= 1:
        parent_list = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    else:
        parent_list = (
            torch.arange(-1, spec_steps - 1, dtype=torch.long, device=device)
            .expand(batch_size, -1)
            .contiguous()
        )

    return selected_index, parent_list


class VerifyWorker(BaseSpecWorker):
    """Spec-v2 worker for target-side decoupled verification.

    Decoupled verifier has no local draft model.  Its ``draft()`` method turns
    scheduler-bound external draft-tail snapshots into an ``EagleVerifyInput``;
    ``verify()`` then follows the EAGLE v2 target-verify/result contract.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self._decoupled_verify_max_speculative_steps = (
            None
            if server_args.speculative_num_steps is None
            else int(server_args.speculative_num_steps)
        )
        if not hasattr(server_args, "_decoupled_verify_max_speculative_steps"):
            server_args._decoupled_verify_max_speculative_steps = (
                self._decoupled_verify_max_speculative_steps
            )
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.pp_rank = getattr(target_worker, "pp_rank", 0)
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.enable_adaptive_verify = dynamic_verify_enabled(server_args)
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.total_accept_length = 0
        self.total_num_verified_reqs = 0
        self.adaptive_controller: Optional[AdaptiveController] = None
        self._roofline_zero_state: Optional[SpecRuntimeState] = None
        self._roofline_profile_states_by_step: dict[int, SpecRuntimeState] = {}
        self._roofline_profile_capture_bs_by_step: dict[int, List[int]] = {}
        self._roofline_profile_bs_by_step: dict[int, List[int]] = {}
        self._roofline_profile_bs_candidates: Optional[List[int]] = None
        self._roofline_profile_capture_bs: Optional[List[int]] = None
        self._roofline_profile_capture_in_progress = False
        self._roofline_profile_in_progress = False
        self._roofline_profile_done = False

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        return None

    def clear_cache_pool(self):
        return

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        return

    def init_cuda_graphs(self):
        if self.enable_adaptive_verify:
            capture_bs = (
                None
                if self.server_args.disable_cuda_graph
                else self.server_args.cuda_graph_bs_decode
            )
            if capture_bs is None and not self.server_args.disable_cuda_graph:
                capture_bs, _ = get_batch_sizes_to_capture(
                    self.target_worker.model_runner, num_tokens_per_bs=1
                )

            if is_decoupled_verify_roofline_budget(
                self.server_args.decoupled_spec_target_verify_token_budget
            ):
                # Capture profile states here; profiling needs scheduler-owned tree_cache.
                self._capture_roofline_profile_states()
                log_info_on_rank0(
                    logger,
                    "Captured decoupled verifier roofline profile runtime states; "
                    "startup profiling will run after scheduler KV cache init. "
                    f"profile_capture_bs_by_step="
                    f"{self._roofline_profile_capture_bs_by_step}",
                )
                return

            adaptive_config = resolve_decoupled_verify_adaptive_config_from_server_args(
                self.server_args, cuda_graph_bs=capture_bs
            )
            self.adaptive_controller = AdaptiveController(
                self,
                config=adaptive_config,
            )
            log_info_on_rank0(
                logger,
                "Capture decoupled verifier adaptive runtime states begin: "
                f"budget={self.server_args.decoupled_spec_target_verify_token_budget}, "
                f"candidate_steps={self.adaptive_controller.candidate_steps}, "
                f"global_max_steps={self.speculative_num_steps}, "
                f"cuda_graph_bs={capture_bs}",
            )

            model_runner = self.target_worker.model_runner
            initial_graph_runner = model_runner.decode_cuda_graph_runner
            if initial_graph_runner is None:
                raise RuntimeError(
                    "Decoupled verifier adaptive initial state requires the target "
                    "worker's captured decode CUDA Graph."
                )
            initial_state = SpecRuntimeState.for_decoupled_verify(
                speculative_num_steps=self.speculative_num_steps,
                speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                target_attn_backend=model_runner.attn_backend,
                target_graph_runner=initial_graph_runner,
            )
            self._validate_decoupled_runtime_state(initial_state)
            self.adaptive_controller.register(initial_state)
            self.adaptive_controller.init_states(cuda_graph_bs=capture_bs)
            log_info_on_rank0(
                logger,
                "Capture decoupled verifier adaptive runtime states end.",
            )

    def _capture_roofline_profile_states(self) -> None:
        profile_bs_candidates = self._generate_roofline_profile_bs_candidates()
        profile_capture_bs = self._generate_roofline_profile_capture_bs(
            profile_bs_candidates
        )
        self._roofline_profile_bs_candidates = profile_bs_candidates
        self._roofline_profile_states_by_step = {}
        self._roofline_profile_capture_bs_by_step = {}
        self._roofline_profile_bs_by_step = {}
        self._roofline_profile_capture_in_progress = True
        try:
            for steps in self._roofline_profile_steps():
                state = self.build_adaptive_runtime_state(
                    speculative_num_steps=steps,
                    speculative_num_draft_tokens=steps + 1,
                    cuda_graph_bs=profile_capture_bs,
                )
                capture_bs = self._get_capture_bs(state.target_graph_runner)
                self._roofline_profile_states_by_step[steps] = state
                self._roofline_profile_capture_bs_by_step[steps] = capture_bs
                self._roofline_profile_bs_by_step[steps] = (
                    self._filter_roofline_profile_bs_by_capture(
                        profile_bs_candidates, capture_bs
                    )
                )
        finally:
            self._roofline_profile_capture_in_progress = False

        empty_profile_steps = [
            steps
            for steps, profile_bs in self._roofline_profile_bs_by_step.items()
            if not profile_bs
        ]
        if empty_profile_steps:
            raise RuntimeError(
                "Decoupled verifier roofline profiling has no replayable profile "
                f"batch sizes for steps={empty_profile_steps}; "
                f"profile_bs_candidates={profile_bs_candidates}, "
                f"capture_bs_by_step={self._roofline_profile_capture_bs_by_step}"
            )

        zero_state = self._roofline_profile_states_by_step.get(0)
        if zero_state is None:
            raise RuntimeError(
                "Decoupled verifier roofline profiling failed to capture step=0."
            )
        self._roofline_zero_state = zero_state

        profile_capture_bs_union = sorted(
            {
                int(bs)
                for capture_bs in self._roofline_profile_capture_bs_by_step.values()
                for bs in capture_bs
            }
        )
        if not profile_capture_bs_union:
            raise RuntimeError(
                "Decoupled verifier roofline profiling has no captured CUDA Graph "
                "batch sizes."
            )
        self._roofline_profile_capture_bs = profile_capture_bs_union

    def _generate_roofline_profile_bs_candidates(self) -> List[int]:
        return resolve_decoupled_verify_roofline_profile_bs_candidates(
            self.server_args,
        )

    def _generate_roofline_profile_capture_bs(
        self, profile_bs_candidates: List[int]
    ) -> List[int]:
        decode_capture_bs = getattr(
            self.server_args.cuda_graph_config.decode, "bs", None
        )
        return resolve_decoupled_verify_roofline_capture_bs_candidates(
            self.server_args,
            profile_bs_candidates,
            cuda_graph_bs=decode_capture_bs,
        )

    def _filter_roofline_profile_bs_by_capture(
        self, profile_bs_candidates: List[int], capture_bs: List[int]
    ) -> List[int]:
        if not capture_bs:
            return []
        max_capture_bs = max(int(bs) for bs in capture_bs)
        return [
            int(bs)
            for bs in profile_bs_candidates
            if int(bs) > 0 and int(bs) <= max_capture_bs
        ]

    def _roofline_padded_graph_bs(self, raw_bs: int, capture_bs: List[int]) -> int:
        sorted_capture_bs = sorted(int(bs) for bs in capture_bs)
        index = bisect.bisect_left(sorted_capture_bs, int(raw_bs))
        if index >= len(sorted_capture_bs):
            raise RuntimeError(
                "Decoupled verifier roofline profile batch size cannot be "
                f"replayed by captured CUDA Graphs: raw_bs={raw_bs}, "
                f"capture_bs={capture_bs}"
            )
        return sorted_capture_bs[index]

    def _roofline_profile_steps(self) -> List[int]:
        max_steps = self._roofline_max_speculative_steps()
        if max_steps is None:
            max_steps = int(self.speculative_num_steps or 0)
        if max_steps < 0:
            raise RuntimeError(
                "Decoupled verifier roofline profiling requires non-negative "
                f"max speculative steps, got {max_steps}."
            )
        return list(range(int(max_steps) + 1))

    def run_startup_spec_profiling(self, tree_cache) -> None:
        if not is_decoupled_verify_roofline_budget(
            self.server_args.decoupled_spec_target_verify_token_budget
        ):
            return
        if self._roofline_profile_done:
            return

        if not self._roofline_profile_states_by_step:
            self._capture_roofline_profile_states()

        profile_capture_bs_by_step = self._roofline_profile_capture_bs_by_step
        profile_bs_by_step = self._roofline_profile_bs_by_step
        profile_capture_bs = self._roofline_profile_capture_bs or sorted(
            {
                int(bs)
                for capture_bs in profile_capture_bs_by_step.values()
                for bs in capture_bs
            }
        )
        if not profile_capture_bs or not profile_capture_bs_by_step:
            raise RuntimeError(
                "Decoupled verifier roofline profiling has no captured "
                "CUDA Graph batch sizes."
            )

        log_info_on_rank0(
            logger,
            "Profile decoupled verifier multi-step roofline begin: "
            f"profile_bs_by_step={profile_bs_by_step}, "
            f"profile_capture_bs_by_step={profile_capture_bs_by_step}, "
            f"warmup_iters={_ROOFLINE_PROFILE_WARMUP_ITERS}, "
            f"measure_iters={_ROOFLINE_PROFILE_MEASURE_ITERS}",
        )

        profile_rows: list[tuple[int, int, float, float]] = []
        self._roofline_profile_in_progress = True
        try:
            for steps in sorted(self._roofline_profile_states_by_step):
                state = self._roofline_profile_states_by_step[steps]
                capture_bs = profile_capture_bs_by_step.get(steps, [])
                for bs in profile_bs_by_step.get(steps, []):
                    padded_graph_bs = self._roofline_padded_graph_bs(bs, capture_bs)
                    avg_decode_ms = self._profile_roofline_shape(
                        batch_size=int(bs),
                        steps=int(steps),
                        state=state,
                        tree_cache=tree_cache,
                    )
                    avg_decode_ms = self._max_reduce_profile_ms(avg_decode_ms)
                    throughput = (
                        int(bs) * (int(steps) + 1) * 1000.0 / avg_decode_ms
                    )
                    profile_rows.append(
                        (int(bs), int(steps), avg_decode_ms, throughput)
                    )
                    log_info_on_rank0(
                        logger,
                        "Decoupled verifier roofline profile point: "
                        f"bs={int(bs)}, steps={int(steps)}, "
                        f"padded_graph_bs={padded_graph_bs}, "
                        f"verify_tokens_per_req={int(steps) + 1}, "
                        f"avg_decode_ms={avg_decode_ms:.4f}, "
                        f"throughput={throughput:.2f} tok/s",
                    )
        finally:
            self._roofline_profile_in_progress = False

        selected_steps_by_bs, selection_summaries, global_peak = (
            select_decoupled_verify_roofline_steps_by_bs(
                [
                    (bs, steps, throughput)
                    for bs, steps, _, throughput in profile_rows
                ],
                plateau_ratio=_ROOFLINE_PROFILE_PLATEAU_RATIO,
            )
        )
        adaptive_config = build_decoupled_verify_profiled_adaptive_config(
            selected_steps_by_bs
        )
        self.server_args._decoupled_verify_roofline_selected_steps_by_bs = dict(
            selected_steps_by_bs
        )
        self.server_args._decoupled_verify_roofline_adaptive_config = adaptive_config

        for bs, summary in sorted(selection_summaries.items()):
            log_info_on_rank0(
                logger,
                "Decoupled verifier roofline selected step: "
                f"bs={bs}, peak_step={summary['peak_step']}, "
                f"peak_throughput={summary['peak_throughput']:.2f} tok/s, "
                f"p95_threshold={summary['threshold']:.2f} tok/s, "
                f"selected_step={summary['selected_step']}",
            )
        global_peak_bs, global_peak_step, global_peak_throughput = global_peak
        log_info_on_rank0(
            logger,
            "Decoupled verifier multi-step roofline selected: "
            f"global_peak_bs={global_peak_bs}, "
            f"global_peak_step={global_peak_step}, "
            f"global_peak_throughput={global_peak_throughput:.2f} tok/s, "
            f"adaptive_config={adaptive_config}",
        )

        zero_state = self._roofline_profile_states_by_step.get(0)
        if zero_state is None:
            raise RuntimeError(
                "Decoupled verifier roofline profiling has no captured step=0 "
                "runtime state."
            )
        self.apply_runtime_state(zero_state)
        self.adaptive_controller = AdaptiveController(self, config=adaptive_config)
        for steps, state in self._roofline_profile_states_by_step.items():
            self.adaptive_controller.register(state, steps=steps)
        self.adaptive_controller.init_states(cuda_graph_bs=profile_capture_bs)
        self._roofline_profile_done = True
        log_info_on_rank0(
            logger,
            "Profile decoupled verifier multi-step roofline end.",
        )

    def _roofline_max_speculative_steps(self) -> Optional[int]:
        max_steps = getattr(
            self.server_args,
            "_decoupled_verify_max_speculative_steps",
            None,
        )
        if max_steps is None:
            max_steps = getattr(self, "_decoupled_verify_max_speculative_steps", None)
        if max_steps is None:
            max_steps = getattr(self.server_args, "speculative_num_steps", None)
        return None if max_steps is None else int(max_steps)

    def _profile_roofline_shape(
        self,
        batch_size: int,
        steps: int,
        state: SpecRuntimeState,
        tree_cache,
    ) -> float:
        if self.device != "cuda":
            raise RuntimeError(
                "Decoupled verifier roofline profiling currently requires CUDA "
                "event timing."
            )
        self.apply_runtime_state(state)

        reqs, batch = self._build_roofline_profile_batch(
            batch_size=batch_size,
            tree_cache=tree_cache,
        )

        try:
            keep_alive_refs = []
            keep_alive_refs.extend(
                self._run_roofline_prefill(batch).extra_keep_alive_refs or []
            )
            for _ in range(_ROOFLINE_PROFILE_WARMUP_ITERS):
                self._prepare_roofline_profile_draft_buffers(batch, steps)
                keep_alive_refs.extend(
                    self._run_roofline_decode(batch).extra_keep_alive_refs or []
                )
            torch.cuda.current_stream().synchronize()
            keep_alive_refs.clear()

            events = []
            for _ in range(_ROOFLINE_PROFILE_MEASURE_ITERS):
                self._prepare_roofline_profile_draft_buffers(batch, steps)
                batch.prepare_for_decode()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                result = self._run_roofline_profile_forward(batch)
                end_event.record()
                self._apply_roofline_decode_result(batch, result)
                keep_alive_refs.extend(result.extra_keep_alive_refs or [])
                events.append((start_event, end_event))

            torch.cuda.current_stream().synchronize()
            total_ms = sum(start.elapsed_time(end) for start, end in events)
            return total_ms / max(1, _ROOFLINE_PROFILE_MEASURE_ITERS)
        finally:
            self._teardown_roofline_profile_reqs(reqs, tree_cache)

    def _build_roofline_profile_batch(
        self,
        *,
        batch_size: int,
        tree_cache,
    ) -> tuple[list[Req], ScheduleBatch]:
        seq_len = self._roofline_profile_prompt_len()
        vocab_size = int(getattr(self.model_config, "vocab_size", 32000) or 32000)
        token_mod = max(1, vocab_size - 1)
        max_new_tokens = self._roofline_profile_decode_headroom(extra_iters=4)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            ignore_eos=True,
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        reqs = []
        for i in range(int(batch_size)):
            token_ids = array(
                "q",
                [((i + j) % token_mod) + 1 for j in range(seq_len)],
            )
            req = Req(
                rid=f"decoupled-roofline-profile-{batch_size}-{i}-{time.time_ns()}",
                origin_input_text="",
                origin_input_ids=token_ids,
                sampling_params=sampling_params,
                extra_key=f"decoupled-roofline-profile-{batch_size}-{i}-{time.time_ns()}",
            )
            req.skip_radix_cache_insert = True
            req.init_next_round_input(tree_cache)
            req.fill_len = len(req.full_untruncated_fill_ids)
            reqs.append(req)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=self.speculative_algorithm,
        )
        return reqs, batch

    def _roofline_profile_decode_headroom(self, *, extra_iters: int) -> int:
        max_steps = self._roofline_max_speculative_steps()
        if max_steps is None:
            max_steps = int(self.speculative_num_steps or 0)
        tokens_per_decode = max(1, int(max_steps) + 1)
        return (
            _ROOFLINE_PROFILE_WARMUP_ITERS
            + _ROOFLINE_PROFILE_MEASURE_ITERS
            + int(extra_iters)
        ) * tokens_per_decode

    def _roofline_profile_prompt_len(self) -> int:
        context_len = int(getattr(self.model_config, "context_len", 4096) or 4096)
        decode_headroom = self._roofline_profile_decode_headroom(extra_iters=8)
        max_profile_bs = max(self._roofline_profile_capture_bs or [1])
        available_tokens = getattr(
            self.token_to_kv_pool_allocator, "available_size", lambda: 0
        )()
        pool_limited_len = (
            max(
                1,
                int(available_tokens) // max(1, int(max_profile_bs))
                - decode_headroom,
            )
            if available_tokens
            else 256
        )
        prefill_budget_candidates = [
            int(value)
            for value in (
                getattr(self.server_args, "chunked_prefill_size", None),
                getattr(self.server_args, "max_prefill_tokens", None),
            )
            if value is not None and int(value) > 0
        ]
        prefill_limited_len = (
            max(1, min(prefill_budget_candidates) // max(1, int(max_profile_bs)))
            if prefill_budget_candidates
            else 256
        )
        return max(
            1,
            min(
                256,
                context_len - decode_headroom,
                pool_limited_len,
                prefill_limited_len,
            ),
        )

    def _run_roofline_prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.prepare_for_extend()
        if batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        result = self._run_roofline_profile_forward(batch)
        self._apply_roofline_prefill_result(batch, result)
        return result

    def _run_roofline_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.prepare_for_decode()
        result = self._run_roofline_profile_forward(batch)
        self._apply_roofline_decode_result(batch, result)
        return result

    def _prepare_roofline_profile_draft_buffers(
        self, batch: ScheduleBatch, steps: int
    ) -> None:
        steps = int(steps)
        for req_idx, req in enumerate(batch.reqs):
            if steps <= 0:
                req.draft_buffer = []
                continue
            req.draft_buffer = [
                self._roofline_dummy_token_id(req_idx, len(req.output_ids) + i)
                for i in range(steps)
            ]

    def _run_roofline_profile_forward(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            batch.sampling_info = sampling_info.copy_for_forward()
        try:
            result = self.forward_batch_generation(batch)
            if result.extra_keep_alive_refs is None:
                result.extra_keep_alive_refs = []
            result.extra_keep_alive_refs.append(snapshot)
            return result
        finally:
            for name, value in snapshot.items():
                setattr(batch, name, value)

    def _apply_roofline_prefill_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        batch.spec_info = result.next_draft_input
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
        batch.input_ids = None
        for i, req in enumerate(batch.reqs):
            req.output_ids.append(self._roofline_dummy_token_id(i, len(req.output_ids)))

    def _roofline_decode_accept_lens(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> List[int]:
        if result.accept_lens is not None:
            return [int(x) for x in result.accept_lens.to("cpu").tolist()]
        if result.new_seq_lens is not None and batch.seq_lens is not None:
            accept_lens = (
                result.new_seq_lens.to(batch.seq_lens.device) - batch.seq_lens
            )
            return [int(x) for x in accept_lens.to("cpu").tolist()]
        if result.num_correct_drafts_per_req_cpu is not None:
            return [int(x) + 1 for x in result.num_correct_drafts_per_req_cpu]
        return [1 for _ in batch.reqs]

    def _apply_roofline_decode_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        if not result.can_run_cuda_graph:
            raise RuntimeError(
                "Decoupled verifier roofline profiling expected target verify "
                "CUDA Graph replay, but the measured decode ran eagerly."
            )
        accept_lens = self._roofline_decode_accept_lens(batch, result)
        if len(accept_lens) != len(batch.reqs):
            raise RuntimeError(
                "Decoupled verifier roofline profiling decode result has "
                f"{len(accept_lens)} accept lengths for {len(batch.reqs)} requests."
            )
        batch.spec_info = result.next_draft_input
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = result.new_seq_lens.to("cpu")
        else:
            if batch.seq_lens is not None:
                batch.seq_lens = batch.seq_lens + torch.tensor(
                    accept_lens,
                    dtype=batch.seq_lens.dtype,
                    device=batch.seq_lens.device,
                )
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = batch.seq_lens_cpu + torch.tensor(
                    accept_lens,
                    dtype=batch.seq_lens_cpu.dtype,
                    device=batch.seq_lens_cpu.device,
                )
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif batch.seq_lens is not None:
            batch.seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        batch.input_ids = None
        for i, req in enumerate(batch.reqs):
            for _ in range(max(0, int(accept_lens[i]))):
                req.output_ids.append(
                    self._roofline_dummy_token_id(i, len(req.output_ids))
                )

    def _roofline_dummy_token_id(self, req_idx: int, output_idx: int) -> int:
        vocab_size = int(getattr(self.model_config, "vocab_size", 32000) or 32000)
        return ((int(req_idx) + int(output_idx)) % max(1, vocab_size - 1)) + 1

    def _teardown_roofline_profile_reqs(self, reqs: list[Req], tree_cache) -> None:
        for req in reqs:
            if req.req_pool_idx is None and getattr(req, "mamba_pool_idx", None) is None:
                continue
            try:
                release_kv_cache(req, tree_cache, is_insert=False)
            except Exception:
                logger.exception(
                    "Failed to release decoupled verifier roofline profile request "
                    f"{req.rid}."
                )

    def _max_reduce_profile_ms(self, avg_ms: float) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return float(avg_ms)
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return float(avg_ms)
        value = torch.tensor([float(avg_ms)], dtype=torch.float32, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=tp_group.device_group)
        return float(value.item())

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: List[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: Optional[List[int]] = None,
    ) -> SpecRuntimeState:
        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            raise RuntimeError(
                "Decoupled verifier adaptive runtime states require target verify "
                "CUDA Graph."
            )
        speculative_num_steps = int(speculative_num_steps)
        speculative_num_draft_tokens = int(speculative_num_draft_tokens)
        verify_tokens_per_req = speculative_num_steps + 1
        if speculative_num_draft_tokens != verify_tokens_per_req:
            speculative_num_draft_tokens = verify_tokens_per_req
        capture_bs = (
            None if cuda_graph_bs is None else [int(bs) for bs in cuda_graph_bs]
        )
        if capture_bs is not None and not capture_bs:
            raise RuntimeError(
                "Decoupled verifier adaptive runtime state got an empty CUDA "
                f"Graph batch-size list for steps={speculative_num_steps}."
            )

        model_runner = self.target_worker.model_runner
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(
            self.device, getattr(model_runner, "gpu_id", 0), empty_cache=False
        )
        log_info_on_rank0(
            logger,
            "Capture decoupled verifier adaptive runtime state: "
            f"step={speculative_num_steps}, "
            f"verify_tokens_per_req={verify_tokens_per_req}, "
            f"capture_bs={capture_bs}",
        )

        with self._override_worker_state(
            speculative_num_steps,
            verify_tokens_per_req,
            cuda_graph_bs=capture_bs,
        ):
            backup_init = model_runner.init_new_workspace
            try:
                target_attn_backend = model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                model_runner.init_new_workspace = backup_init
            target_graph_runner = DecodeCudaGraphRunner(
                model_runner,
                attn_backend=target_attn_backend,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=verify_tokens_per_req,
            )
        after_mem = get_available_gpu_memory(
            self.device, getattr(model_runner, "gpu_id", 0), empty_cache=False
        )
        log_info_on_rank0(
            logger,
            "Captured decoupled verifier adaptive runtime state: "
            f"step={speculative_num_steps}, "
            f"verify_tokens_per_req={verify_tokens_per_req}, "
            f"capture_bs={target_graph_runner.capture_bs}, "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )
        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=verify_tokens_per_req,
            target_attn_backend=target_attn_backend,
            target_graph_runner=target_graph_runner,
        )
        # During roofline profiling, every profiled step is captured before the
        # final per-BS adaptive config exists. Validation allows that temporary
        # capture phase and tightens again after profiling resolves the config.
        self._validate_decoupled_runtime_state(state)
        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        self._validate_decoupled_runtime_state(state)
        if state.target_attn_backend is None or state.target_graph_runner is None:
            raise RuntimeError(
                "Decoupled verifier dynamic runtime state is missing target "
                "CUDA Graph resources."
            )
        if (
            self.speculative_num_steps == state.speculative_num_steps
            and self.speculative_num_draft_tokens == state.speculative_num_draft_tokens
            and self.target_worker.model_runner.attn_backend
            is state.target_attn_backend
            and self.target_worker.model_runner.decode_cuda_graph_runner
            is state.target_graph_runner
        ):
            return

        old_steps = self.speculative_num_steps
        old_draft_tokens = self.speculative_num_draft_tokens
        self.speculative_num_steps = int(state.speculative_num_steps)
        self.speculative_num_draft_tokens = int(state.speculative_num_draft_tokens)
        self.target_worker.model_runner.attn_backend = state.target_attn_backend
        self.target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )
        self.server_args.speculative_num_steps = self.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            self.speculative_num_draft_tokens
        )
        if (
            old_steps != self.speculative_num_steps
            or old_draft_tokens != self.speculative_num_draft_tokens
        ):
            log_info_on_rank0(
                logger,
                "Switch decoupled verifier adaptive state: "
                f"steps {old_steps} -> {self.speculative_num_steps}, "
                f"draft_tokens {old_draft_tokens} -> "
                f"{self.speculative_num_draft_tokens}",
            )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: Optional[List[int]] = None,
    ):
        """Temporarily override target-only adaptive state for graph capture."""
        sa = self.server_args
        decode_config = sa.cuda_graph_config.decode
        decode_config_bs = decode_config.bs
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs_decode,
            None if decode_config_bs is None else list(decode_config_bs),
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        if cuda_graph_bs is not None:
            sa.cuda_graph_bs_decode = cuda_graph_bs
            decode_config.bs = cuda_graph_bs

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
                sa.cuda_graph_bs_decode,
                decode_config.bs,
            ) = backup

    def _validate_decoupled_runtime_state(self, state: SpecRuntimeState) -> None:
        if (
            state.draft_attn_backend is not None
            or state.cuda_graph_runner is not None
            or state.draft_extend_attn_backend is not None
            or state.cuda_graph_runner_for_draft_extend is not None
        ):
            raise ValueError(
                "Decoupled verifier runtime state must not carry draft resources."
            )
        expected_draft_tokens = int(state.speculative_num_steps) + 1
        if int(state.speculative_num_draft_tokens) != expected_draft_tokens:
            raise RuntimeError(
                "Decoupled verifier runtime state has inconsistent verify width: "
                f"steps={state.speculative_num_steps}, "
                f"draft_tokens={state.speculative_num_draft_tokens}, "
                f"expected={expected_draft_tokens}"
            )
        graph_runner = state.target_graph_runner
        if graph_runner is None:
            raise RuntimeError(
                "Decoupled verifier adaptive runtime state has no target graph runner."
            )
        capture_bs = self._get_capture_bs(graph_runner)
        if not capture_bs:
            raise RuntimeError(
                "Decoupled verifier adaptive runtime state has no captured "
                f"batch sizes for steps={state.speculative_num_steps}."
            )
        budget_value = self.server_args.decoupled_spec_target_verify_token_budget
        if is_decoupled_verify_roofline_budget(budget_value):
            if (
                int(state.speculative_num_steps) == 0
                and int(state.speculative_num_draft_tokens) == 1
            ):
                return
            if (
                getattr(self, "_roofline_profile_capture_in_progress", False)
                or getattr(self, "_roofline_profile_in_progress", False)
            ):
                return

            adaptive_config = getattr(
                self.server_args, "_decoupled_verify_roofline_adaptive_config", None
            )
            if adaptive_config is not None:
                selected_bs = self._roofline_config_bs_for_step(
                    adaptive_config, int(state.speculative_num_steps)
                )
                if not selected_bs:
                    raise RuntimeError(
                        "Decoupled verifier roofline runtime state "
                        f"steps={state.speculative_num_steps} is not selected "
                        "by roofline adaptive config."
                    )
                unsupported_bs = [
                    bs
                    for bs in selected_bs
                    if not self._roofline_profile_bs_supported_by_capture(
                        bs, capture_bs
                    )
                ]
                if unsupported_bs:
                    raise RuntimeError(
                        "Decoupled verifier roofline runtime state does not "
                        "have a captured graph bucket for all selected raw "
                        "profile batch sizes: "
                        f"steps={state.speculative_num_steps}, "
                        f"unsupported_bs={unsupported_bs}, capture_bs={capture_bs}"
                    )
                return

            roofline_bs = getattr(
                self.server_args, "_decoupled_verify_roofline_bs", None
            )
            if roofline_bs is None:
                raise RuntimeError(
                    "Decoupled verifier roofline runtime state validation needs "
                    "a resolved roofline adaptive config for nonzero-step states."
                )
            roofline_bs = int(roofline_bs)
            max_capture_bs = max(int(bs) for bs in capture_bs)
            padded_verify_tokens = max_capture_bs * int(
                state.speculative_num_draft_tokens
            )
            if padded_verify_tokens > roofline_bs:
                raise RuntimeError(
                    "decoupled verifier roofline budget violated by adaptive "
                    "runtime state: "
                    f"steps={state.speculative_num_steps}, "
                    f"verify_tokens_per_req={state.speculative_num_draft_tokens}, "
                    f"max_capture_bs={max_capture_bs}, "
                    f"padded_verify_tokens={padded_verify_tokens}, "
                    f"roofline_bs={roofline_bs}"
                )
            return

        budget = int(budget_value)
        max_capture_bs = max(int(bs) for bs in capture_bs)
        padded_verify_tokens = max_capture_bs * int(
            state.speculative_num_draft_tokens
        )
        if padded_verify_tokens >= budget:
            raise RuntimeError(
                "decoupled verifier target verify budget violated by "
                "adaptive runtime state: "
                f"steps={state.speculative_num_steps}, "
                f"verify_tokens_per_req={state.speculative_num_draft_tokens}, "
                f"max_capture_bs={max_capture_bs}, "
                f"padded_verify_tokens={padded_verify_tokens}, "
                f"budget={budget}"
            )

    def _roofline_config_bs_for_step(self, config: dict, step: int) -> List[int]:
        selected_bs = []
        for raw_bs, raw_entry in config.items():
            entry = raw_entry or {}
            candidate_steps = entry.get("candidate_steps", [])
            if int(step) in [int(value) for value in candidate_steps]:
                selected_bs.append(int(raw_bs))
        return sorted(selected_bs)

    def _roofline_profile_bs_supported_by_capture(
        self, raw_bs: int, capture_bs: List[int]
    ) -> bool:
        return bool(capture_bs) and int(raw_bs) <= max(int(bs) for bs in capture_bs)

    def _get_capture_bs(self, graph_runner) -> List[int]:
        if graph_runner is not None and getattr(graph_runner, "capture_bs", None):
            return [int(bs) for bs in graph_runner.capture_bs]
        cuda_graph_config = getattr(self.server_args, "cuda_graph_config", None)
        decode_config = getattr(cuda_graph_config, "decode", None)
        capture_bs = getattr(decode_config, "bs", None)
        if capture_bs:
            return [int(bs) for bs in capture_bs]
        max_bs = getattr(decode_config, "max_bs", None)
        if max_bs is not None:
            return [int(max_bs)]
        capture_bs = getattr(self.server_args, "cuda_graph_bs", None)
        if capture_bs:
            return [int(bs) for bs in capture_bs]
        max_bs = getattr(self.server_args, "cuda_graph_max_bs", None)
        if max_bs is not None:
            return [int(max_bs)]
        raise RuntimeError(
            "Cannot determine decoupled verifier CUDA graph batch sizes for "
            "dynamic verify length."
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        return self.target_worker.update_weights_from_disk(recv_req)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        return self.target_worker.update_weights_from_ipc(recv_req)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _get_verify_buffers(
        self, draft_token_num: int, batch: Optional[ScheduleBatch]
    ):
        expected_draft_tokens = int(self.speculative_num_draft_tokens)
        if int(draft_token_num) != expected_draft_tokens:
            raise RuntimeError(
                "Decoupled verifier verify buffer width mismatch: "
                f"draft_token_num={draft_token_num}, "
                f"expected={expected_draft_tokens}"
            )
        attn_backend = getattr(self.target_worker.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None, None

        get_buffers = getattr(
            attn_backend, "get_verify_buffers_to_fill_after_draft", None
        )
        if get_buffers is None:
            return None, None

        return get_buffers()

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

    def _build_req_verify_tokens(
        self, req, pad_token_id: int, spec_depth: int
    ) -> List[int]:
        tail_token = _get_req_tail_token_id(req)
        draft_buffer = list(getattr(req, "draft_buffer", []) or [])
        draft_tokens = list(draft_buffer[:spec_depth])
        if len(draft_tokens) < spec_depth:
            draft_tokens.extend([int(pad_token_id)] * (spec_depth - len(draft_tokens)))
        return [tail_token, *draft_tokens]

    def _get_snapshot_tail_lens(
        self, batch: ScheduleBatch, spec_depth: int
    ) -> List[int]:
        return [
            min(
                len(list(getattr(req, "draft_buffer", []) or [])),
                spec_depth,
            )
            for req in batch.reqs
        ]

    def _assert_num_correct_within_snapshot_tail(
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: List[int]
    ) -> List[int]:
        # req.draft_buffer is a per-forward snapshot bound before verify. Any
        # concurrent drafter appends belong to later verify rounds.
        spec_steps = int(self.speculative_num_steps)
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
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: List[int]
    ) -> None:
        spec_steps = int(self.speculative_num_steps)
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

    def _build_trivial_verify_input(
        self, batch: ScheduleBatch, seq_lens_sum: int
    ) -> EagleVerifyInput:
        batch_size = batch.batch_size()
        device = batch.device
        draft_token = torch.tensor(
            [_get_req_tail_token_id(req) for req in batch.reqs],
            dtype=torch.long,
            device=device,
        )
        retrieve_index = torch.arange(
            batch_size, dtype=torch.long, device=device
        ).unsqueeze(1)
        retrieve_next_token = torch.full(
            (batch_size, 1), -1, dtype=torch.long, device=device
        )
        retrieve_next_sibling = torch.full(
            (batch_size, 1), -1, dtype=torch.long, device=device
        )

        tree_mask_buf, position_buf = self._get_verify_buffers(1, batch=batch)
        if tree_mask_buf is not None:
            custom_mask = tree_mask_buf
            custom_mask.fill_(True)
        else:
            custom_mask = torch.ones(
                seq_lens_sum + batch_size, dtype=torch.bool, device=device
            )

        if position_buf is not None:
            positions = position_buf
            positions[:batch_size].copy_(batch.seq_lens)
        else:
            positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_token,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        spec_steps = int(self.speculative_num_steps)
        draft_token_num = int(self.speculative_num_draft_tokens)
        if draft_token_num < 1:
            raise RuntimeError(
                "External draft verification requires at least one verify token "
                "per request."
            )

        if batch.forward_mode.is_idle():
            spec_info = EagleVerifyInput.create_idle_input(
                self.topk,
                spec_steps,
                draft_token_num,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
            return spec_info

        seq_lens_sum = (
            int(batch.seq_lens_cpu.sum())
            if batch.seq_lens_cpu is not None
            else int(torch.sum(batch.seq_lens).item())
        )
        batch.seq_lens_sum = seq_lens_sum

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

        if spec_steps == 0:
            return self._build_trivial_verify_input(batch, seq_lens_sum)

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

        return EagleVerifyInput(
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

    def verify(self, batch: ScheduleBatch) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        assert verify_input is not None
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        was_idle = batch.forward_mode.is_idle()
        seq_lens_pre_verify = batch.seq_lens.clone()

        verify_input.num_tokens_per_req = verify_input.draft_token_num
        batch.return_hidden_states = False

        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            self.req_to_token_pool,
            batch,
            self.target_worker,
        )
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)
        assert (
            verify_forward_batch.capture_hidden_mode
            == verify_input.capture_hidden_mode
        )

        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(
            logits_output.next_token_logits, "decoupled_verify: target model logits"
        )
        maybe_detect_inf(
            logits_output.next_token_logits, "decoupled_verify: target model logits"
        )

        predict, accept_lens, accept_index = eagle_sample(
            verify_input, batch, logits_output, vocab_mask
        )
        num_correct_drafts_per_req_cpu = (accept_lens - 1).cpu().tolist()
        self._assert_num_correct_within_snapshot_tail(
            batch, num_correct_drafts_per_req_cpu
        )
        self._record_valid_draft_metrics(batch, num_correct_drafts_per_req_cpu)
        new_seq_lens = seq_lens_pre_verify + accept_lens

        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_index,
            verify_input.draft_token_num,
        )

        if not was_idle:
            if self.page_size != 1:
                raise RuntimeError(
                    "Decoupled verifier currently requires page_size == 1."
                )

        if not was_idle:
            bonus_tokens = _build_bonus_tokens_from_accepts(
                predict, accept_lens, accept_index
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not was_idle:
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, verify_input.spec_steps
            )

        batch.forward_mode = ForwardMode.IDLE if was_idle else ForwardMode.DECODE
        batch.spec_info = None
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            num_correct_drafts=sum(num_correct_drafts_per_req_cpu),
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=verify_input.draft_token_num,
            next_draft_input=_build_next_draft_input_stub(
                bonus_tokens, verify_input.topk
            ),
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            batch_output = self.target_worker.forward_batch_generation(batch)
            if isinstance(batch_output.next_token_ids, torch.Tensor):
                batch_output.next_draft_input = _build_next_draft_input_stub(
                    bonus_tokens=batch_output.next_token_ids.flatten().to(
                        dtype=torch.int32
                    ),
                    topk=self.topk,
                )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        spec_info = self.draft(batch)
        assert spec_info.is_verify_input()
        batch.spec_info = spec_info
        batch_output = self.verify(batch)

        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if self.enable_adaptive_verify and not batch_output.can_run_cuda_graph:
            raise RuntimeError(
                "Decoupled verifier dynamic verify length requires full CUDA "
                "Graph replay for target verify, but this forward ran without it: "
                f"speculative_num_steps={self.speculative_num_steps}, "
                f"speculative_num_draft_tokens={self.speculative_num_draft_tokens}, "
                f"draft_token_num={spec_info.draft_token_num}"
            )

        num_verified_reqs = len(batch_output.num_correct_drafts_per_req_cpu or [])
        self.total_accept_length += int(batch_output.num_correct_drafts)
        self.total_num_verified_reqs += num_verified_reqs
        return batch_output

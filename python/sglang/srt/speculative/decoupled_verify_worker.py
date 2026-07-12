import contextlib
import logging
import time
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
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
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
    _SpecAdaptiveBase,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    DecoupledVerifyThroughputAwareController,
)
from sglang.srt.speculative.decoupled_verify_input import (
    build_next_draft_input_stub,
    get_req_tail_token_id,
)
from sglang.srt.speculative.decoupled_verify_profiler import (
    DecoupledVerifyProfilerMixin,
)
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

_THROUGHPUT_PROFILE_WARMUP_ITERS = 5
_THROUGHPUT_PROFILE_MEASURE_ITERS = 1000


def _get_req_draft_tokens(req) -> list[int]:
    snapshot = getattr(req, "decoupled_verify_snapshot", None)
    return snapshot.draft_tokens if snapshot is not None else []


def _build_bonus_tokens_from_accepts(
    predict: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
) -> torch.Tensor:
    accept_tokens = predict[accept_index]
    row_indices = torch.arange(accept_lens.numel(), device=accept_lens.device)
    bonus_indices = torch.clamp(accept_lens.to(torch.long) - 1, min=0)
    return accept_tokens[row_indices, bonus_indices].to(dtype=torch.int32)


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


class VerifyWorker(DecoupledVerifyProfilerMixin, BaseSpecWorker):
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
        self.adaptive_config = getattr(server_args, "_adaptive_spec_config", None)
        if server_args.speculative_adaptive and self.adaptive_config is None:
            raise RuntimeError("Adaptive speculative configuration was not normalized.")
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
        self.adaptive_controller: Optional[_SpecAdaptiveBase] = None
        self._throughput_profile_states_by_step: dict[int, SpecRuntimeState] = {}
        self._throughput_profile_capture_bs_by_step: dict[int, List[int]] = {}
        self._throughput_profile_bs_by_step: dict[int, List[int]] = {}
        self._throughput_profile_capture_bs: Optional[List[int]] = None
        self._throughput_profile_done = False

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

    def _use_throughput_aware_adaptive_verify(self) -> bool:
        return (
            self.enable_adaptive_verify
            and getattr(self.server_args, "speculative_adaptive_strategy", "ema")
            == "throughput_aware"
        )

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

            if self._use_throughput_aware_adaptive_verify():
                # Capture profile states here; profiling needs scheduler-owned tree_cache.
                self._capture_throughput_profile_states(capture_bs=capture_bs)
                log_info_on_rank0(
                    logger,
                    "Captured decoupled verifier throughput-aware profile runtime states; "
                    "startup profiling will run after scheduler KV cache init. "
                    f"profile_capture_bs_by_step="
                    f"{self._throughput_profile_capture_bs_by_step}",
                )
                return

            self.adaptive_controller = AdaptiveController(
                self,
                config=self.adaptive_config.params_config,
            )
            log_info_on_rank0(
                logger,
                "Capture decoupled verifier adaptive runtime states begin: "
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

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: List[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int, ctx_len: int = 1) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size, ctx_len)

    def get_modeled_throughput(
        self, batch_size: int, ctx_len: int, accept_length: float
    ) -> Optional[dict]:
        if self.adaptive_controller is None:
            return None
        return self.adaptive_controller.get_modeled_throughput(
            batch_size=batch_size,
            ctx_len=ctx_len,
            accept_length=accept_length,
        )

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
        adaptive_controller = getattr(self, "adaptive_controller", None)
        if isinstance(adaptive_controller, DecoupledVerifyThroughputAwareController):
            if int(state.speculative_num_steps) not in [
                int(step) for step in adaptive_controller.candidate_steps
            ]:
                raise RuntimeError(
                    "Decoupled verifier throughput-aware runtime state "
                    f"steps={state.speculative_num_steps} is not selected "
                    "by the controller."
                )

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
        tail_token = get_req_tail_token_id(req)
        draft_tokens = list(_get_req_draft_tokens(req)[:spec_depth])
        if len(draft_tokens) < spec_depth:
            draft_tokens.extend([int(pad_token_id)] * (spec_depth - len(draft_tokens)))
        return [tail_token, *draft_tokens]

    def _get_snapshot_tail_lens(
        self, batch: ScheduleBatch, spec_depth: int
    ) -> List[int]:
        return [
            min(
                len(_get_req_draft_tokens(req)),
                spec_depth,
            )
            for req in batch.reqs
        ]

    def _record_valid_draft_metrics(
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: List[int]
    ) -> int:
        spec_steps = int(self.speculative_num_steps)
        total_valid_draft_tokens = 0
        for req, accepted_drafts in zip(batch.reqs, num_correct_drafts_per_req_cpu):
            valid_draft_tokens = min(
                len(_get_req_draft_tokens(req)), spec_steps
            )
            valid_accepted_tokens = min(int(accepted_drafts), valid_draft_tokens)
            total_valid_draft_tokens += valid_draft_tokens
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
        return total_valid_draft_tokens

    def _build_trivial_verify_input(
        self, batch: ScheduleBatch, seq_lens_sum: int
    ) -> EagleVerifyInput:
        batch_size = batch.batch_size()
        device = batch.device
        draft_token = torch.tensor(
            [get_req_tail_token_id(req) for req in batch.reqs],
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
                    [get_req_tail_token_id(req) for req in batch.reqs],
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
        valid_draft_tokens = self._record_valid_draft_metrics(
            batch, num_correct_drafts_per_req_cpu
        )
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
            spec_valid_draft_tokens=valid_draft_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=verify_input.draft_token_num,
            next_draft_input=build_next_draft_input_stub(
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
                batch_output.next_draft_input = build_next_draft_input_stub(
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

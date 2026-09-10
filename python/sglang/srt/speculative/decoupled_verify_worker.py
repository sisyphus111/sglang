from __future__ import annotations

import contextlib
import logging
import time

import torch

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.runtime_context import get_context, get_exec, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.cpp_decoupled_spec import GPU_DRAFT_TAIL_DEBUG_WIDTH
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    resolve_decoupled_verify_candidate_steps,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.spec_utils import get_plan_stream
from sglang.srt.utils.common import get_available_gpu_memory, log_info_on_rank0
from sglang.srt.utils.nvtx_utils import operations_nvtx_range

logger = logging.getLogger(__name__)

# The first seven columns are the stable selector state. The four optional
# columns expose update errors, pending-prefix recovery, and seqlock retries.
DECOUPLED_TAIL_SELECT_DEBUG_WIDTH = GPU_DRAFT_TAIL_DEBUG_WIDTH + 4


def build_decoupled_next_input(
    bonus_tokens: torch.Tensor,
    *,
    topk: int,
) -> EagleDraftInput:
    """Build the standard V2 relay payload without a local draft model."""

    bonus_tokens = bonus_tokens.flatten().to(dtype=torch.int32)
    batch_size = int(bonus_tokens.shape[0])
    device = bonus_tokens.device
    return EagleDraftInput(
        bonus_tokens=bonus_tokens,
        topk_p=torch.zeros((batch_size, topk), dtype=torch.float32, device=device),
        topk_index=torch.zeros((batch_size, topk), dtype=torch.int64, device=device),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def select_decoupled_gpu_tail_snapshot(
    *,
    gpu_tail_buffer,
    tp_rank: int,
    tp_group,
    gpu_seats: torch.Tensor,
    expected_request_epochs: torch.Tensor | None,
    seq_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    num_draft_tokens: int,
    out: torch.Tensor | None = None,
    out_cursor: torch.Tensor | None = None,
    debug_out: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Select on TP0 and GPU-broadcast one fixed-shape verifier snapshot."""

    batch_size = int(seq_lens.shape[0])
    compact_width = int(num_draft_tokens) + 2
    logical_committed_lens = None
    if int(tp_rank) == 0:
        if gpu_tail_buffer is None:
            raise RuntimeError(
                "Decoupled verify TP0 has no attached GPU draft-tail buffer."
            )
        if expected_request_epochs is None and not gpu_tail_buffer.mock_profile:
            raise RuntimeError(
                "Decoupled verify TP0 has no batch-owned request epochs."
            )
        with operations_nvtx_range("sglang.decoupled_spec.gpu_tail_select"):
            compact_snapshot, logical_committed_lens = gpu_tail_buffer.select_snapshot(
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=expected_request_epochs,
                out=out,
                out_cursor=out_cursor,
                debug_out=debug_out,
            )
    else:
        compact_snapshot = (
            torch.empty(
                (batch_size, compact_width),
                dtype=torch.int64,
                device=seq_lens.device,
            )
            if out is None
            else out
        )

    expected_shape = (batch_size, compact_width)
    if tuple(compact_snapshot.shape) != expected_shape:
        raise RuntimeError(
            "Decoupled GPU snapshot shape mismatch: "
            f"expected={expected_shape} actual={tuple(compact_snapshot.shape)}"
        )
    if compact_snapshot.dtype != torch.int64:
        raise RuntimeError(
            "Decoupled GPU snapshot must use int64 storage, got "
            f"{compact_snapshot.dtype}."
        )
    if compact_snapshot.device != seq_lens.device:
        raise RuntimeError(
            "Decoupled GPU snapshot is on the wrong device: "
            f"snapshot={compact_snapshot.device} batch={seq_lens.device}."
        )
    if int(tp_rank) == 0 and (
        logical_committed_lens is None
        or tuple(logical_committed_lens.shape) != (batch_size,)
        or logical_committed_lens.dtype != torch.int64
        or logical_committed_lens.device != seq_lens.device
    ):
        raise RuntimeError(
            "Decoupled GPU snapshot returned an invalid committed-length cursor."
        )
    if (
        int(tp_rank) == 0
        and debug_out is not None
        and (
            debug_out.ndim != 2
            or int(debug_out.shape[0]) != batch_size
            or int(debug_out.shape[1]) < 7
            or debug_out.dtype != torch.int64
            or debug_out.device != seq_lens.device
        )
    ):
        raise RuntimeError(
            "Decoupled GPU snapshot returned an invalid TP0 debug tensor: "
            f"shape={tuple(debug_out.shape)} dtype={debug_out.dtype} "
            f"device={debug_out.device}."
        )

    if tp_group.world_size > 1:
        # This call runs under Scheduler.forward_stream_ctx. It neither reads a
        # host snapshot nor introduces a CPU-group collective.
        with operations_nvtx_range("sglang.decoupled_spec.tp_broadcast"):
            tp_group.broadcast(compact_snapshot, src=0)

    return (
        compact_snapshot[:, :num_draft_tokens],
        compact_snapshot[:, num_draft_tokens],
        compact_snapshot[:, num_draft_tokens + 1],
        debug_out if int(tp_rank) == 0 else None,
        logical_committed_lens,
    )


class DecoupledVerifyWorker(BaseSpecWorker):
    """Target-only Spec V2 worker fed by a verifier-resident GPU draft tail."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        super().__init__()
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.device = server_args.device
        self.topk = int(server_args.speculative_eagle_topk)
        self.max_draft_tokens = int(server_args.speculative_num_steps)
        self.max_verify_tokens = int(server_args.speculative_num_draft_tokens)
        self.num_draft_tokens = self.max_draft_tokens
        self.num_verify_tokens = self.max_verify_tokens
        self.active_verify_steps = self.max_draft_tokens
        self.active_verify_tokens = self.max_verify_tokens
        self.speculative_num_steps = self.active_verify_steps
        self.speculative_num_draft_tokens = self.active_verify_tokens
        if self.topk != 1:
            raise ValueError("Decoupled verification requires topk == 1.")
        if self.num_verify_tokens != self.num_draft_tokens + 1:
            raise ValueError(
                "Decoupled verification requires speculative_num_draft_tokens "
                "== speculative_num_steps + 1."
            )

        self._target_worker = target_worker
        self._draft_worker = None
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.gpu_tail_buffer = None
        self._gpu_tail_buffer_attached = False
        self._gpu_tail_snapshot_buffers = None
        self._gpu_tail_cursor_buffers = None
        self._gpu_tail_expected_epoch_buffers = None
        self._gpu_tail_landing_events = None
        self._gpu_tail_debug_buffers = None
        self._linear_selected_index_by_step: dict[int, torch.Tensor] = {}
        self._linear_parent_list_by_step: dict[int, torch.Tensor] = {}
        self._verify_runtime_states: dict[int, tuple[object, object]] = {}
        self._pending_verify_steps: int | None = None
        self._runtime_candidate_steps = self._resolve_runtime_candidate_steps()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def _resolve_runtime_candidate_steps(self) -> list[int]:
        from sglang.srt.managers.scheduler_components.decoupled_spec.offline_profile import (
            DecoupledVerifyOfflineProfiler,
        )

        profile_steps = DecoupledVerifyOfflineProfiler.configured_steps()
        if profile_steps:
            if any(step < 0 or step > self.max_draft_tokens for step in profile_steps):
                raise ValueError(
                    "Offline profile steps exceed verifier Kmax: "
                    f"steps={profile_steps} Kmax={self.max_draft_tokens}."
                )
            return profile_steps
        if not self.server_args.speculative_adaptive:
            return []
        return resolve_decoupled_verify_candidate_steps(
            self.server_args.speculative_adaptive_config,
            max_steps=self.max_draft_tokens,
        )

    def attach_gpu_tail_buffer(self, gpu_tail_buffer) -> None:
        """Attach TP0's GPU tail after scheduler-side transport construction."""

        if self.ps.tp_rank == 0 and gpu_tail_buffer is None:
            raise RuntimeError(
                "Decoupled verification requires a GPU draft-tail data plane on TP0."
            )
        if self._gpu_tail_buffer_attached:
            raise RuntimeError(
                "The decoupled GPU draft-tail buffer is already attached."
            )
        if self.req_to_token_pool is None:
            raise RuntimeError(
                "Decoupled GPU draft-tail buffers require an initialized request pool."
            )
        self.gpu_tail_buffer = gpu_tail_buffer
        # ReqToTokenPool reserves an extra padding row beyond its public
        # request capacity. req_pool_idx can address that final physical row.
        num_gpu_seats = int(self.req_to_token_pool.req_to_token.shape[0])
        if gpu_tail_buffer is not None and (
            int(gpu_tail_buffer.num_seats) != num_gpu_seats
            or int(gpu_tail_buffer.num_draft_tokens) != self.num_draft_tokens
        ):
            raise RuntimeError(
                "Decoupled GPU draft-tail dimensions do not match the verifier: "
                f"tail_seats={gpu_tail_buffer.num_seats} "
                f"req_pool_seats={num_gpu_seats} "
                f"tail_draft_tokens={gpu_tail_buffer.num_draft_tokens} "
                f"verifier_draft_tokens={self.num_draft_tokens}."
            )
        # Two slots are sufficient for Scheduler's one-result overlap queue:
        # slot r is not reused until r+2, after result r's D2H copy completed.
        self._gpu_tail_snapshot_buffers = torch.empty(
            (2, num_gpu_seats, self.num_draft_tokens + 2),
            dtype=torch.int64,
            device=self.device,
        )
        self._gpu_tail_cursor_buffers = torch.empty(
            (2, num_gpu_seats),
            dtype=torch.int64,
            device=self.device,
        )
        if self.ps.tp_rank == 0:
            device_module = torch.get_device_module(self.device)
            self._gpu_tail_expected_epoch_buffers = torch.empty(
                (2, num_gpu_seats),
                dtype=torch.int64,
                device=self.device,
            )
            self._gpu_tail_landing_events = (
                device_module.Event(),
                device_module.Event(),
            )
            self._gpu_tail_debug_buffers = torch.empty(
                (2, num_gpu_seats, DECOUPLED_TAIL_SELECT_DEBUG_WIDTH),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self._gpu_tail_expected_epoch_buffers = None
            self._gpu_tail_landing_events = None
            self._gpu_tail_debug_buffers = None
        # The topk=1 chain topology is identical for every request and round.
        # Keep one request-pool-sized copy instead of rebuilding arange/expand/
        # contiguous tensors on the verifier hot path.
        topology_steps = sorted({self.max_draft_tokens, *self._runtime_candidate_steps})
        for steps in topology_steps:
            self._linear_selected_index_by_step[steps] = (
                torch.arange(steps, dtype=torch.long, device=self.device)
                .expand(num_gpu_seats, -1)
                .contiguous()
            )
            parent_width = steps if steps > 1 else 0
            self._linear_parent_list_by_step[steps] = (
                torch.arange(
                    -1,
                    parent_width - 1,
                    dtype=torch.long,
                    device=self.device,
                )
                .expand(num_gpu_seats, -1)
                .contiguous()
            )
        self._gpu_tail_buffer_attached = True

    def capture_expected_request_epochs(self, batch: ScheduleBatch) -> None:
        """Freeze TP0's per-row lifecycle identity on the scheduler stream."""

        if self.ps.tp_rank != 0:
            batch.decoupled_expected_request_epochs = None
            batch.decoupled_landing_event = None
            return
        if (
            self.gpu_tail_buffer is None
            or self._gpu_tail_expected_epoch_buffers is None
        ):
            raise RuntimeError("Decoupled verifier TP0 has no GPU epoch buffer.")
        if self.gpu_tail_buffer.mock_profile:
            batch.decoupled_expected_request_epochs = None
            batch.decoupled_landing_event = None
            return
        snapshot_slot = int(batch.forward_iter) % 2
        batch_size = int(batch.req_pool_indices.numel())
        batch.decoupled_expected_request_epochs = (
            self.gpu_tail_buffer.capture_active_request_epochs(
                batch.req_pool_indices,
                out=self._gpu_tail_expected_epoch_buffers[snapshot_slot, :batch_size],
            )
        )
        batch.decoupled_landing_event = None
        if (
            batch.decoupled_has_new_lifecycle
            or batch.decoupled_needs_landing_fence
        ):
            if self._gpu_tail_landing_events is None:
                raise RuntimeError("Decoupled verifier TP0 has no landing event ring.")
            batch.decoupled_landing_event = self._gpu_tail_landing_events[
                snapshot_slot
            ]
            batch.decoupled_landing_event.record(
                self.gpu_tail_buffer.landing_stream
            )

    @property
    def war_fastpath_runner(self):
        return self.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (self.target_worker.model_runner.attn_backend,)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        return None

    def init_cuda_graphs(self) -> None:
        # TpModelWorker owns the initial Kmax graph. Dynamic/profile mode adds
        # target-only states; drafter publication and GPU-tail capacity stay Kmax.
        if not self._runtime_candidate_steps:
            return None
        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            raise RuntimeError(
                "Adaptive/offline decoupled verification requires target decode "
                "CUDA Graphs for every active K."
            )
        model_runner = self.target_worker.model_runner
        if model_runner.decode_cuda_graph_runner is None:
            raise RuntimeError("The decoupled verifier Kmax CUDA Graph is missing.")
        self._verify_runtime_states[self.max_draft_tokens] = (
            model_runner.attn_backend,
            model_runner.decode_cuda_graph_runner,
        )
        capture_bs = list(get_exec().graph.cuda_graph_bs_decode or [])
        for steps in self._runtime_candidate_steps:
            if steps == self.max_draft_tokens:
                continue
            self._verify_runtime_states[steps] = self._build_verify_runtime_state(
                steps=steps,
                cuda_graph_bs=capture_bs,
            )
        missing = sorted(
            set(self._runtime_candidate_steps) - self._verify_runtime_states.keys()
        )
        if missing:
            raise RuntimeError(f"Missing decoupled verifier runtime states: {missing}.")
        return None

    def _build_verify_runtime_state(
        self, *, steps: int, cuda_graph_bs: list[int]
    ) -> tuple[object, object]:
        model_runner = self.target_worker.model_runner
        started = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        with self._capture_verify_shape(steps=steps, cuda_graph_bs=cuda_graph_bs):
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
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
            )
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            "Captured decoupled verifier target runtime state: "
            f"steps={steps}, verify_tokens={steps + 1}, "
            f"capture_bs={getattr(target_graph_runner, 'capture_bs', None)}, "
            f"elapsed={time.perf_counter() - started:.2f}s, "
            f"mem={before_mem - after_mem:.2f}GB",
        )
        return target_attn_backend, target_graph_runner

    @contextlib.contextmanager
    def _capture_verify_shape(self, *, steps: int, cuda_graph_bs: list[int]):
        backup = (
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
        )
        get_context().override(
            "decoupled_verify.capture",
            speculative_num_steps=int(steps),
            speculative_num_draft_tokens=int(steps) + 1,
            cuda_graph_bs_decode=list(cuda_graph_bs),
        )
        try:
            yield
        finally:
            get_context().override(
                "decoupled_verify.capture_restore",
                speculative_num_steps=backup[0],
                speculative_num_draft_tokens=backup[1],
                cuda_graph_bs_decode=backup[2],
            )

    def queue_verify_steps(self, steps: int) -> None:
        steps = int(steps)
        if steps not in self._verify_runtime_states:
            raise ValueError(
                f"No captured decoupled verifier runtime state for steps={steps}."
            )
        self._pending_verify_steps = steps

    def activate_pending_verify_steps(self) -> None:
        steps = self._pending_verify_steps
        if steps is None:
            return
        self._pending_verify_steps = None
        self.activate_verify_steps(steps)

    def activate_verify_steps(self, steps: int) -> None:
        steps = int(steps)
        state = self._verify_runtime_states.get(steps)
        if state is None:
            if steps == self.max_draft_tokens and not self._runtime_candidate_steps:
                return
            raise ValueError(
                f"No captured decoupled verifier runtime state for steps={steps}."
            )
        if steps == self.active_verify_steps:
            return
        old_steps = self.active_verify_steps
        target_attn_backend, target_graph_runner = state
        self.target_worker.model_runner.attn_backend = target_attn_backend
        self.target_worker.model_runner.decode_cuda_graph_runner = target_graph_runner
        self.active_verify_steps = steps
        self.active_verify_tokens = steps + 1
        self.speculative_num_steps = steps
        self.speculative_num_draft_tokens = steps + 1
        get_context().override(
            "decoupled_verify.activate",
            speculative_num_steps=steps,
            speculative_num_draft_tokens=steps + 1,
        )
        log_info_on_rank0(
            logger,
            f"Activated decoupled verifier target state: steps {old_steps} -> {steps}.",
        )

    def _build_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        applied_steps = int(
            self.active_verify_steps
            if batch.decoupled_verify_steps is None
            else batch.decoupled_verify_steps
        )
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                applied_steps,
                applied_steps + 1,
                self.device,
            )

        draft_input = batch.spec_info
        if not isinstance(draft_input, EagleDraftInput):
            raise RuntimeError(
                "Decoupled verify decode requires the previous bonus-token relay."
            )

        tp_group = get_tp_group()
        snapshot_slot = int(batch.forward_iter) % 2
        batch_size = int(batch.seq_lens.shape[0])
        if batch.decoupled_landing_event is not None:
            self.gpu_tail_buffer.wait_for_landing_event(
                batch.decoupled_landing_event
            )
        (
            selected_draft_tokens,
            selected_lens,
            row_valid,
            tail_select_debug,
            logical_committed_lens,
        ) = select_decoupled_gpu_tail_snapshot(
            gpu_tail_buffer=self.gpu_tail_buffer,
            tp_rank=self.ps.tp_rank,
            tp_group=tp_group,
            gpu_seats=batch.req_pool_indices,
            expected_request_epochs=batch.decoupled_expected_request_epochs,
            seq_lens=batch.seq_lens,
            bonus_tokens=draft_input.bonus_tokens,
            num_draft_tokens=self.max_draft_tokens,
            out=self._gpu_tail_snapshot_buffers[snapshot_slot, :batch_size],
            out_cursor=self._gpu_tail_cursor_buffers[snapshot_slot, :batch_size],
            debug_out=(
                None
                if self._gpu_tail_debug_buffers is None
                else self._gpu_tail_debug_buffers[snapshot_slot, :batch_size]
            ),
        )
        with operations_nvtx_range("sglang.decoupled_spec.verify_input_prepare"):
            selected_draft_tokens = selected_draft_tokens[:, :applied_steps]
            selected_lens = selected_lens.clamp_max(applied_steps)
            selected_index = self._linear_selected_index_by_step[applied_steps][
                :batch_size
            ]
            parent_list = self._linear_parent_list_by_step[applied_steps][:batch_size]
            if not selected_index.is_contiguous() or not parent_list.is_contiguous():
                raise RuntimeError(
                    "Decoupled topk=1 chain topology must be contiguous for the "
                    f"tree kernel: steps={applied_steps}."
                )

            verify_input = build_eagle_verify_input(
                batch,
                draft_input,
                parent_list,
                selected_index,
                selected_draft_tokens,
                None,
                target_worker=self.target_worker,
                topk=1,
                num_steps=applied_steps,
                num_draft_tokens=applied_steps + 1,
                tree_mask_mode=TreeMaskMode.FULL_MASK,
                device=self.device,
            )
            verify_input.retrieve_next_token.scatter_(1, selected_lens.unsqueeze(1), -1)
            verify_input.capture_hidden_mode = CaptureHiddenMode.NULL

        # These tensors are consumed by manager-side observability after D2H.
        batch.decoupled_rebase_valid = row_valid
        batch.decoupled_selected_draft_lens = selected_lens
        batch.decoupled_tail_select_debug = tail_select_debug
        batch.decoupled_pre_verify_output_lens = logical_committed_lens
        batch.decoupled_verify_steps = applied_steps
        return verify_input

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
    ):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch_output = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=CaptureHiddenMode.NULL
            )
            if isinstance(batch_output.next_token_ids, torch.Tensor):
                batch_output.next_draft_input = build_decoupled_next_input(
                    batch_output.next_token_ids,
                    topk=self.topk,
                )
                if self.ps.tp_rank == 0 and not getattr(
                    self.gpu_tail_buffer, "mock_profile", False
                ):
                    if batch.decoupled_expected_request_epochs is None:
                        raise RuntimeError(
                            "Decoupled critical VerifyCommit requires launch epochs."
                        )
                    if batch.decoupled_landing_event is not None:
                        self.gpu_tail_buffer.wait_for_landing_event(
                            batch.decoupled_landing_event
                        )
                    commit_rows = [
                        not req.finished()
                        and not req.is_retracted
                        # The previous chunk's CPU result can still be in flight.
                        # Only the current batch identifies a non-final row.
                        and req is not batch.chunked_req
                        for req in batch.reqs
                    ]
                    commit_mask = (
                        None
                        if all(commit_rows)
                        else torch.tensor(
                            commit_rows,
                            dtype=torch.bool,
                            device=batch.req_pool_indices.device,
                        )
                    )
                    with operations_nvtx_range("sglang.decoupled_spec.verify_commit"):
                        self.gpu_tail_buffer.apply_verify_commit_from_device(
                            batch.req_pool_indices,
                            batch.decoupled_expected_request_epochs,
                            None,
                            batch_output.next_draft_input.bonus_tokens,
                            torch.ones_like(
                                batch_output.next_draft_input.bonus_tokens,
                                dtype=torch.int32,
                            ),
                            accept_token_stride=1,
                            commit_mask=commit_mask,
                        )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        verify_input = self._build_verify_input(batch)
        applied_steps = int(batch.decoupled_verify_steps)
        batch.spec_info = verify_input
        with operations_nvtx_range("sglang.decoupled_spec.target_verify"):
            batch_output = run_eagle_verify(
                batch,
                target_worker=self.target_worker,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                plan_stream=self.plan_stream,
                plan_stream_ctx=self.plan_stream_ctx,
                topk=self.topk,
                num_steps=applied_steps,
                num_draft_tokens=applied_steps + 1,
                device=self.device,
                metadata_ready_pre_pad=False,
                finalize_tree_path=True,
                grammar_barrier=grammar_barrier,
            )
        if self.ps.tp_rank == 0 and not getattr(
            self.gpu_tail_buffer, "mock_profile", False
        ):
            if batch.decoupled_expected_request_epochs is None:
                raise RuntimeError(
                    "Decoupled critical VerifyCommit requires launch epochs."
                )
            if batch_output.accept_lens is None:
                raise RuntimeError(
                    "Decoupled critical VerifyCommit requires per-request accept lengths."
                )
            with operations_nvtx_range("sglang.decoupled_spec.verify_commit"):
                self.gpu_tail_buffer.apply_verify_commit_from_device(
                    batch.req_pool_indices,
                    batch.decoupled_expected_request_epochs,
                    batch.seq_lens,
                    batch_output.next_token_ids,
                    batch_output.accept_lens,
                    accept_token_stride=applied_steps + 1,
                )
        batch_output.next_draft_input = build_decoupled_next_input(
            batch_output.next_draft_input.bonus_tokens,
            topk=self.topk,
        )
        batch_output.decoupled_rebase_valid = batch.decoupled_rebase_valid
        batch_output.decoupled_selected_draft_lens = batch.decoupled_selected_draft_lens
        batch_output.decoupled_tail_select_debug = batch.decoupled_tail_select_debug
        batch_output.decoupled_pre_output_lens = batch.decoupled_pre_verify_output_lens
        batch_output.decoupled_verify_steps = applied_steps
        batch_output.speculative_num_draft_tokens = applied_steps + 1
        if on_publish is not None:
            with operations_nvtx_range("sglang.decoupled_spec.future_publish"):
                on_publish(batch_output.new_seq_lens)

        return batch_output

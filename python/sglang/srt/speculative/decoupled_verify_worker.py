from __future__ import annotations

import torch

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.spec_utils import get_plan_stream
from sglang.srt.utils.nvtx_utils import operations_nvtx_range


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
    seq_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    num_draft_tokens: int,
    out: torch.Tensor | None = None,
    out_cursor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select on TP0 and GPU-broadcast one fixed-shape verifier snapshot."""

    batch_size = int(seq_lens.shape[0])
    compact_width = int(num_draft_tokens) + 2
    logical_committed_lens = None
    if int(tp_rank) == 0:
        if gpu_tail_buffer is None:
            raise RuntimeError(
                "Decoupled verify TP0 has no attached GPU draft-tail buffer."
            )
        with operations_nvtx_range("sglang.decoupled_spec.gpu_tail_select"):
            compact_snapshot, logical_committed_lens = gpu_tail_buffer.select_snapshot(
                gpu_seats,
                seq_lens,
                bonus_tokens,
                out=out,
                out_cursor=out_cursor,
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

    if tp_group.world_size > 1:
        # This call runs under Scheduler.forward_stream_ctx. It neither reads a
        # host snapshot nor introduces a CPU-group collective.
        with operations_nvtx_range("sglang.decoupled_spec.tp_broadcast"):
            tp_group.broadcast(compact_snapshot, src=0)

    return (
        compact_snapshot[:, :num_draft_tokens],
        compact_snapshot[:, num_draft_tokens],
        compact_snapshot[:, num_draft_tokens + 1],
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
        self.num_draft_tokens = int(server_args.speculative_num_steps)
        self.num_verify_tokens = int(server_args.speculative_num_draft_tokens)
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
        self._linear_selected_index = None
        self._linear_parent_list = None
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

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
        # The topk=1 chain topology is identical for every request and round.
        # Keep one request-pool-sized copy instead of rebuilding arange/expand/
        # contiguous tensors on the verifier hot path.
        self._linear_selected_index = (
            torch.arange(
                self.num_draft_tokens,
                dtype=torch.long,
                device=self.device,
            )
            .expand(num_gpu_seats, -1)
            .contiguous()
        )
        self._linear_parent_list = (
            torch.empty(
                (num_gpu_seats, 0),
                dtype=torch.long,
                device=self.device,
            )
            if self.num_draft_tokens <= 1
            else torch.arange(
                -1,
                self.num_draft_tokens - 1,
                dtype=torch.long,
                device=self.device,
            )
            .expand(num_gpu_seats, -1)
            .contiguous()
        )
        self._gpu_tail_buffer_attached = True

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
        # Target decode/verify graphs are owned and captured by TpModelWorker.
        return None

    def _build_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.num_draft_tokens,
                self.num_verify_tokens,
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
        (
            selected_draft_tokens,
            selected_lens,
            row_valid,
            logical_committed_lens,
        ) = select_decoupled_gpu_tail_snapshot(
            gpu_tail_buffer=self.gpu_tail_buffer,
            tp_rank=self.ps.tp_rank,
            tp_group=tp_group,
            gpu_seats=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            bonus_tokens=draft_input.bonus_tokens,
            num_draft_tokens=self.num_draft_tokens,
            out=self._gpu_tail_snapshot_buffers[snapshot_slot, :batch_size],
            out_cursor=self._gpu_tail_cursor_buffers[snapshot_slot, :batch_size],
        )
        with operations_nvtx_range("sglang.decoupled_spec.verify_input_prepare"):
            selected_index = self._linear_selected_index[:batch_size]
            parent_list = self._linear_parent_list[:batch_size]

            verify_input = build_eagle_verify_input(
                batch,
                draft_input,
                parent_list,
                selected_index,
                selected_draft_tokens,
                None,
                target_worker=self.target_worker,
                topk=1,
                num_steps=self.num_draft_tokens,
                num_draft_tokens=self.num_verify_tokens,
                tree_mask_mode=TreeMaskMode.FULL_MASK,
                device=self.device,
            )
            verify_input.retrieve_next_token.scatter_(1, selected_lens.unsqueeze(1), -1)
            verify_input.capture_hidden_mode = CaptureHiddenMode.NULL

        # These tensors are consumed by manager-side observability after D2H.
        batch.decoupled_rebase_valid = row_valid
        batch.decoupled_selected_draft_lens = selected_lens
        batch.decoupled_pre_verify_output_lens = logical_committed_lens
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
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        verify_input = self._build_verify_input(batch)
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
                num_steps=self.num_draft_tokens,
                num_draft_tokens=self.num_verify_tokens,
                device=self.device,
                metadata_ready_pre_pad=False,
                finalize_tree_path=True,
                grammar_barrier=grammar_barrier,
            )
        batch_output.next_draft_input = build_decoupled_next_input(
            batch_output.next_draft_input.bonus_tokens,
            topk=self.topk,
        )
        batch_output.decoupled_rebase_valid = batch.decoupled_rebase_valid
        batch_output.decoupled_selected_draft_lens = batch.decoupled_selected_draft_lens
        batch_output.decoupled_pre_output_lens = batch.decoupled_pre_verify_output_lens
        if on_publish is not None:
            with operations_nvtx_range("sglang.decoupled_spec.future_publish"):
                on_publish(batch_output.new_seq_lens)

        return batch_output

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.managers.scheduler_components.decoupled_spec.base import (
    build_draft_mirror_request_id,
)
from sglang.srt.runtime_context import get_stream
from sglang.srt.speculative.decoupled_spec_data_plane import (
    create_verifier_decoupled_spec_data_plane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftSync,
    VerifyCommit,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


class DecoupledVerifyManager:
    """Verifier request lifecycle and GPU-tail transport coordinator."""

    def __init__(
        self,
        scheduler: Scheduler,
        config: DecoupledSpecIpcConfig,
    ) -> None:
        self.scheduler = scheduler
        self.config = config
        self.src_verifier_rank = int(config.rank)
        self.dst_drafter_rank = 0
        self.is_entry_rank = scheduler.ps.tp_rank == 0
        self.num_draft_tokens = int(scheduler.server_args.speculative_num_steps)
        self._open_mirror_by_req: dict[str, str] = {}
        self._open_lifecycle_by_req: dict[str, tuple[Req, int]] = {}
        self._next_request_epoch = 1
        self.rebase_row_ct = 0
        self.rebase_hit_ct = 0
        self.selected_draft_tokens_ct = 0
        self._last_logged_rebase_rows = 0
        self.data_plane = (
            create_verifier_decoupled_spec_data_plane(
                config,
                required_tail_len=0,
                device=scheduler.device,
                num_gpu_seats=int(scheduler.req_to_token_pool.req_to_token.shape[0]),
                num_draft_tokens=self.num_draft_tokens,
                landing_stream=get_stream("decoupled_spec_landing"),
            )
            if self.is_entry_rank
            else None
        )
        if self.data_plane is not None:
            if getattr(self.data_plane, "gpu_tail_buffer", None) is None:
                raise RuntimeError(
                    "Decoupled verification requires the C++ GPU draft-tail "
                    "data plane; the CPU snapshot backend is not a runtime fallback."
                )
            self.data_plane.start()

        verify_worker = scheduler.draft_worker
        if verify_worker is None or not hasattr(
            verify_worker, "attach_gpu_tail_buffer"
        ):
            raise RuntimeError(
                "Decoupled verifier scheduler is missing DecoupledVerifyWorker."
            )
        verify_worker.attach_gpu_tail_buffer(
            None if self.data_plane is None else self.data_plane.gpu_tail_buffer
        )

    def close(self) -> None:
        if self.data_plane is not None:
            self.data_plane.close()

    def process_pending_controls(self) -> None:
        return None

    def has_pending_work(self) -> bool:
        return False

    def on_no_batch(self) -> bool:
        return False

    def adjust_plan(self, plan):
        return plan

    def sleep_overrun_requests(self, batch):
        return batch

    def prepare_batch(self, batch: ScheduleBatch) -> None:
        if batch is None or batch.forward_mode.is_idle():
            return
        allow_open = batch.forward_mode.is_extend()
        batch.decoupled_launch_mirror_ids = [
            self._ensure_open_request(req, allow_open=allow_open) for req in batch.reqs
        ]

    def _ensure_open_request(self, req: Req, *, allow_open: bool) -> str:
        mirror_request_id = self._mirror_for_lifecycle(req)
        if mirror_request_id is not None:
            return mirror_request_id

        previous = self._open_mirror_by_req.get(req.rid)
        if not allow_open:
            raise RuntimeError(
                "Decoupled verifier decode request has no open draft mirror for "
                "its current lifecycle: "
                f"request_id={req.rid} retraction_count={req.retraction_count}"
            )
        if previous is not None:
            self._close_request(req, previous, reason="reseated")

        request_epoch = self._next_request_epoch
        self._next_request_epoch += 1
        mirror_request_id = build_draft_mirror_request_id(req.rid, request_epoch)
        if self.data_plane is not None:
            if req.req_pool_idx is None:
                raise RuntimeError(
                    "Decoupled verifier request has no GPU request-pool seat: "
                    f"request_id={req.rid}"
                )
            self.data_plane.open_request(
                DraftSync(
                    request_id=mirror_request_id,
                    src_verifier_rank=self.src_verifier_rank,
                    dst_drafter_rank=self.dst_drafter_rank,
                    prompt_token_ids=[int(token) for token in req.origin_input_ids],
                    committed_outputs=[int(token) for token in req.output_ids],
                ),
                gpu_seat=int(req.req_pool_idx),
                request_epoch=request_epoch,
            )
        self._open_mirror_by_req[req.rid] = mirror_request_id
        self._open_lifecycle_by_req[req.rid] = (
            req,
            int(req.retraction_count),
        )
        return mirror_request_id

    def _mirror_for_lifecycle(self, req: Req) -> str | None:
        mirror_request_id = self._open_mirror_by_req.get(req.rid)
        lifecycle = self._open_lifecycle_by_req.get(req.rid)
        if (mirror_request_id is None) != (lifecycle is None):
            raise RuntimeError(
                "Decoupled verifier draft-mirror lifecycle maps are inconsistent: "
                f"request_id={req.rid}"
            )
        if lifecycle is None:
            return None
        owner, retraction_count = lifecycle
        if owner is req and retraction_count == int(req.retraction_count):
            return mirror_request_id
        return None

    def before_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return
        batch.decoupled_pre_output_lens = [len(req.output_ids) for req in batch.reqs]
        batch.decoupled_result_mirror_ids = list(batch.decoupled_launch_mirror_ids)

    def after_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return
        if result.decoupled_rebase_valid is not None:
            self.rebase_row_ct += int(result.decoupled_rebase_valid.numel())
            self.rebase_hit_ct += int(result.decoupled_rebase_valid.sum().item())
        if result.decoupled_selected_draft_lens is not None:
            self.selected_draft_tokens_ct += int(
                result.decoupled_selected_draft_lens.sum().item()
            )
        if (
            self.is_entry_rank
            and self.rebase_row_ct - self._last_logged_rebase_rows >= 128
        ):
            self._last_logged_rebase_rows = self.rebase_row_ct
            logger.info(
                "Decoupled verifier tail select: rows=%s hits=%s hit_rate=%.4f "
                "selected_draft_tokens=%s",
                self.rebase_row_ct,
                self.rebase_hit_ct,
                self.rebase_hit_ct / max(self.rebase_row_ct, 1),
                self.selected_draft_tokens_ct,
            )
        result_pre_output_lens = getattr(result, "decoupled_pre_output_lens", None)
        gpu_pre_output_lens = (
            None if result_pre_output_lens is None else result_pre_output_lens.tolist()
        )
        host_pre_output_lens = batch.decoupled_pre_output_lens
        mirror_request_ids = batch.decoupled_result_mirror_ids
        control_batch = DraftControlBatch(dst_drafter_rank=self.dst_drafter_rank)
        finished_lifecycles: list[tuple[str, str]] = []
        for row_index, (req, mirror_request_id) in enumerate(
            zip(batch.reqs, mirror_request_ids)
        ):
            pre_output_len = int(host_pre_output_lens[row_index])
            if (
                gpu_pre_output_lens is not None
                and int(gpu_pre_output_lens[row_index]) >= 0
            ):
                pre_output_len = int(gpu_pre_output_lens[row_index])
            # The launch identity is immutable, while the live mapping changes
            # on abort, finish, or retraction/reseat. Only the lifecycle that
            # still owns the request may commit a delayed overlap result.
            if self._open_mirror_by_req.get(req.rid) != mirror_request_id:
                continue
            if getattr(req, "is_retracted", False):
                continue
            if req.finished():
                if self.is_entry_rank:
                    control_batch.close_messages.append(
                        DraftClose(
                            request_id=mirror_request_id,
                            src_verifier_rank=self.src_verifier_rank,
                            dst_drafter_rank=self.dst_drafter_rank,
                            reason="finished",
                        )
                    )
                finished_lifecycles.append((req.rid, mirror_request_id))
                continue
            if not self.is_entry_rank:
                continue
            if len(req.output_ids) <= pre_output_len:
                continue
            control_batch.verify_commit_messages.append(
                VerifyCommit(
                    request_id=mirror_request_id,
                    src_verifier_rank=self.src_verifier_rank,
                    dst_drafter_rank=self.dst_drafter_rank,
                    pre_verify_committed_len=pre_output_len,
                    committed_tokens=[
                        int(token) for token in req.output_ids[pre_output_len:]
                    ],
                )
            )
        if self.data_plane is not None and (
            control_batch.verify_commit_messages or control_batch.close_messages
        ):
            self.data_plane.submit_control_batch(control_batch)
        for request_id, mirror_request_id in finished_lifecycles:
            self._forget_open_request(request_id, mirror_request_id)

    def abort_request(self, req: Req, reason: str = "abort") -> None:
        mirror_request_id = self._open_mirror_by_req.get(req.rid)
        lifecycle = self._open_lifecycle_by_req.get(req.rid)
        if (mirror_request_id is None) != (lifecycle is None):
            raise RuntimeError(
                "Decoupled verifier draft-mirror lifecycle maps are inconsistent: "
                f"request_id={req.rid}"
            )
        # A retracted request increments retraction_count before it is reseated.
        # It still owns the previous mirror until the next extend opens a new
        # epoch, so abort must key on Req ownership rather than the exact count.
        if lifecycle is None or lifecycle[0] is not req:
            return
        self._close_request(req, mirror_request_id, reason=reason)

    def retract_request(self, req: Req) -> None:
        """Close the old mirror before the verifier request is recomputed."""

        self.abort_request(req, reason="retracted")

    def handle_retracted_requests(self, retracted_reqs, reqs_to_abort):
        for req in retracted_reqs:
            self.retract_request(req)
        for req in reqs_to_abort:
            self.abort_request(req, reason="oom")
        return retracted_reqs, reqs_to_abort

    def prepare_pause_retract(self, reqs: list[Req]) -> list[Req]:
        """Close each verifier mirror before generic scheduler retraction."""

        unique_reqs = []
        seen_req_ids = set()
        for req in reqs:
            if id(req) in seen_req_ids:
                continue
            seen_req_ids.add(id(req))
            self.retract_request(req)
            unique_reqs.append(req)
        return unique_reqs

    def _close_request(
        self,
        req: Req,
        mirror_request_id: str,
        *,
        reason: str,
    ) -> None:
        if self.data_plane is not None:
            self.data_plane.close_request(
                DraftClose(
                    request_id=mirror_request_id,
                    src_verifier_rank=self.src_verifier_rank,
                    dst_drafter_rank=self.dst_drafter_rank,
                    reason=reason,
                )
            )
        self._forget_open_request(req.rid, mirror_request_id)

    def _forget_open_request(
        self,
        request_id: str,
        mirror_request_id: str,
    ) -> None:
        if self._open_mirror_by_req.get(request_id) == mirror_request_id:
            self._open_mirror_by_req.pop(request_id, None)
            self._open_lifecycle_by_req.pop(request_id, None)

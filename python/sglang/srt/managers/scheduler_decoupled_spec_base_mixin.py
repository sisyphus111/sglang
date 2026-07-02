from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.managers.io_struct import (
    ConfigureDecoupledSpecPeersReq,
    ConfigureDecoupledSpecPeersReqOutput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.tracer import (
    SpecTraceEvent,
    build_tracer,
    trace_speculative,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerDecoupledSpecBaseMixin:
    """Shared scheduler hooks for decoupled speculation."""

    def get_decoupled_spec_local_dp_rank(self: Scheduler) -> int:
        local_dp_rank = (
            self.ps.attn_dp_rank
            if self.server_args.enable_dp_attention
            else self.ps.dp_rank
        )
        return 0 if local_dp_rank is None else int(local_dp_rank)

    def get_decoupled_spec_rank(self: Scheduler) -> int:
        return (
            int(self.server_args.decoupled_spec_rank_base)
            + self.get_decoupled_spec_local_dp_rank()
        )

    def get_decoupled_spec_endpoint_info(self: Scheduler) -> Optional[dict]:
        if not (self.is_verify_entry_rank() or self.is_draft_entry_rank()):
            return None
        bind_endpoint = None
        role = None
        if self.is_verify_entry_rank():
            role = "verifier"
            bind_endpoint = getattr(
                self.draft_proxy_thread, "result_bind_endpoint", None
            )
        elif self.is_draft_entry_rank():
            role = "drafter"
            bind_endpoint = getattr(
                self.token_sync_thread, "control_bind_endpoint", None
            )
        if bind_endpoint is None:
            return None
        return {
            "role": role,
            "rank": self.get_decoupled_spec_rank(),
            "local_dp_rank": self.get_decoupled_spec_local_dp_rank(),
            "bind_endpoint": bind_endpoint,
        }

    def is_draft_worker_batch(
        self: Scheduler, batch: Optional[ScheduleBatch] = None
    ) -> bool:
        spec_algorithm = (
            batch.spec_algorithm if batch is not None else self.spec_algorithm
        )
        return bool(spec_algorithm.is_decoupled_draft())

    def is_verify_worker_batch(
        self: Scheduler, batch: Optional[ScheduleBatch] = None
    ) -> bool:
        spec_algorithm = (
            batch.spec_algorithm if batch is not None else self.spec_algorithm
        )
        return bool(spec_algorithm.is_decoupled_verify())

    def init_draft_state_tables(self: Scheduler) -> None:
        self.draft_req_table = {}
        self.draft_sleeping_reqs = {}
        self.decoupled_verify_drafter_ranks = []
        self.decoupled_verify_req_to_drafter_rank = {}
        self.decoupled_verify_drafter_loads = {}

    def configure_decoupled_spec_peers(
        self: Scheduler, req: ConfigureDecoupledSpecPeersReq
    ) -> ConfigureDecoupledSpecPeersReqOutput:
        try:
            if self.is_verify_entry_rank():
                if self.draft_proxy_thread is None:
                    raise RuntimeError("Decoupled verify proxy is not initialized")
                self.draft_proxy_thread.configure_peer_endpoints(req.connect_endpoints)
                self._reset_decoupled_verify_drafter_tables(
                    len(req.connect_endpoints)
                )
                self.draft_proxy_thread.start()
            elif self.is_draft_entry_rank():
                if self.token_sync_thread is None:
                    raise RuntimeError("Decoupled token sync thread is not initialized")
                self.token_sync_thread.configure_peer_endpoints(req.connect_endpoints)
                self.token_sync_thread.start()
            logger.info(
                "Handled decoupled-spec configure request: rank=%s endpoints=%s",
                self.get_decoupled_spec_rank(),
                req.connect_endpoints,
            )
            return ConfigureDecoupledSpecPeersReqOutput(success=True)
        except Exception as exc:
            logger.exception(
                "Failed to handle decoupled-spec configure request: endpoints=%s",
                req.connect_endpoints,
            )
            return ConfigureDecoupledSpecPeersReqOutput(
                success=False, message=str(exc)
            )

    def start_forward_timer(self: Scheduler, batch: ScheduleBatch) -> Optional[int]:
        if (
            not batch
            or not self.is_generation
            or not (
                self.spec_algorithm.is_decoupled_verify()
                or self.spec_algorithm.is_decoupled_draft()
                or self.spec_algorithm.is_none()
            )
        ):
            return None
        return self.tracer.start_timer()

    @trace_speculative(SpecTraceEvent.SCHEDULER_FORWARD_BATCH)
    def record_forward_latency(
        self: Scheduler,
        batch: ScheduleBatch,
        start_ns: Optional[int],
        result: object | None = None,
    ) -> None:
        if start_ns is None:
            return

    def create_tracer(self):
        enabled = bool(
            getattr(self.server_args, "spec_trace_dir", None)
            and (
                self.spec_algorithm.is_decoupled_verify()
                or self.spec_algorithm.is_decoupled_draft()
                or self.spec_algorithm.is_none()
                or self.spec_algorithm.is_eagle()
            )
        )
        dp_rank = self.ps.dp_rank or 0
        rank_suffix = f"dp{dp_rank}_tp{self.ps.tp_rank}_pp{self.ps.pp_rank}"
        if self.spec_algorithm.is_decoupled_verify():
            forward_trace_prefix = "verifier"
        elif self.spec_algorithm.is_decoupled_draft():
            forward_trace_prefix = "drafter"
        elif self.spec_algorithm.is_none():
            forward_trace_prefix = "decode"
        elif self.spec_algorithm.is_eagle():
            forward_trace_prefix = "mtp"
        else:
            forward_trace_prefix = "scheduler"
        file_names = {
            "scheduler.forward_batch": (
                f"{forward_trace_prefix}-forward-batch_{rank_suffix}.csv"
            ),
            "mtp.phase": f"mtp-phase_{rank_suffix}.csv",
            "verifier": f"verifier_{rank_suffix}.csv",
            "drafter": f"drafter_{rank_suffix}.csv",
            "draft_proxy": f"draft_proxy_verifier{dp_rank}.csv",
            "token_sync_thread": f"token_sync_thread_drafter{dp_rank}.csv",
        }
        return build_tracer(
            enabled=enabled,
            output_dir=getattr(self.server_args, "spec_trace_dir", None),
            file_names=file_names,
        )

    def _infer_forward_graph_path(
        self: Scheduler, batch: ScheduleBatch, can_run_cuda_graph: bool
    ) -> str:
        if not can_run_cuda_graph:
            return "eager"
        if batch.forward_mode.is_decode():
            if self.spec_algorithm.is_decoupled_verify():
                return "cuda_graph_target_verify"
            return "cuda_graph_decode"
        if batch.forward_mode.is_extend(include_draft_extend_v2=True):
            return "piecewise_cuda_graph"
        if batch.forward_mode.is_idle():
            return "cuda_graph_idle"
        return "cuda_graph"

    def is_draft_entry_rank(self: Scheduler) -> bool:
        return (
            self.spec_algorithm.is_decoupled_draft()
            and self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )

    def is_verify_entry_rank(self) -> bool:
        return (
            self.spec_algorithm.is_decoupled_verify()
            and self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )

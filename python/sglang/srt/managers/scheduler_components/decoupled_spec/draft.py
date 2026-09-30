from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.managers.load_snapshot import (
    DecoupledSpecDecodeMetrics,
    DraftTransportMetrics,
)
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.runtime_context import get_observability, get_stream
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_draft_checkpoint import (
    DecoupledDraftMambaCheckpointStore,
    DraftRequestGeneration,
)
from sglang.srt.speculative.decoupled_spec_data_plane import (
    create_drafter_decoupled_spec_data_plane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftReqKey,
    DraftSync,
    build_draft_scheduler_rid,
)
from sglang.srt.utils.common import is_pin_memory_available
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


class _DraftRequestState(msgspec.Struct):
    key: DraftRequestGeneration
    req: Any
    is_sleeping: bool = False
    gpu_seat: int = -1
    gpu_checkpoint_slots_initialized: bool = False
    # Highest contiguous logical KV prefix owned by this Req. Decode candidates
    # may overwrite old-branch cells, so allocation count is not a high-water.
    kv_highwater_len: int = 0
    # Compact host scheduling credit from the latest cumulative GPU snapshot.
    # No token or model-position transcript is mirrored on CPU.
    gpu_ahead_len: int = 0


class DecoupledDraftManager:
    """Own drafter resources and pacing around the GPU token/state machine.

    Both scheduler modes use the same prepare/forward/finish transaction.
    CPU state tracks lifecycle and physical ownership, never decode tokens or
    rollback positions; verifier commits are applied on the landing stream.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        config: DecoupledSpecIpcConfig,
    ) -> None:
        self.scheduler = scheduler
        self.num_draft_tokens = int(scheduler.server_args.speculative_num_steps)
        self.ahead_window = 2 * self.num_draft_tokens + 1
        if scheduler.future_map.needs_cpu_seq_lens:
            raise ValueError(
                "Decoupled drafter GPU state management requires attention backends "
                "that build decode metadata from device sequence lengths."
            )
        if scheduler.token_to_kv_pool_allocator.page_size != 1:
            raise ValueError(
                "Decoupled drafter GPU state management currently requires page_size=1 "
                "for one-candidate decode allocation."
            )
        if (
            scheduler.server_args.enable_mixed_chunk
            and not scheduler.tp_worker.model_runner.attn_backend.supports_decoupled_mixed_prefix
        ):
            raise ValueError(
                "Decoupled drafter mixed prefill requires GPU prefix metadata "
                "support in the selected attention backend."
            )
        # Generic overlap schedules one decode before the previous result-side
        # pacing update. Reserve that in-flight slot at the scheduling boundary.
        self.schedule_ahead_limit = self.ahead_window - int(scheduler.enable_overlap)
        if scheduler.ps.tp_size != 1:
            raise ValueError("The phase-one decoupled drafter requires tp_size == 1.")
        route_capacity = int(scheduler.max_running_requests)
        self.data_plane = create_drafter_decoupled_spec_data_plane(
            config,
            device=scheduler.device,
            num_gpu_seats=route_capacity,
            num_draft_tokens=self.num_draft_tokens,
            pending_token_capacity=max(
                self.ahead_window,
                (int(scheduler.max_total_num_tokens) + route_capacity - 1)
                // route_capacity,
            ),
            landing_stream=get_stream("decoupled_spec_drafter_landing"),
            schedule_ahead_limit=self.schedule_ahead_limit,
        )
        self.data_plane.start()
        self._requests: dict[tuple[int, str], _DraftRequestState] = {}
        self._sleeping_requests: dict[DraftRequestGeneration, Req] = {}
        self._pending_kv_outcomes = []
        self._inflight_kv_outcomes = None
        self._gpu_decode_binding_validated = False
        self._kv_outcome_flush_interval = 32
        # VerifyCommit is consumed by the GPU mirror without scheduler
        # participation. While decode is active, OPEN/CLOSE may lag
        # a few launches: identity checks turn those launches into padding and
        # lifecycle cleanup is not on the model critical path.
        self._gpu_lifecycle_poll_count = 0
        self._gpu_lifecycle_poll_interval = 16

        req_pool = scheduler.req_to_token_pool
        self.checkpoints = (
            DecoupledDraftMambaCheckpointStore(
                req_pool,
                max_draft_tokens=self.num_draft_tokens,
            )
            if isinstance(req_pool, HybridReqToTokenPool)
            else None
        )
        if self.checkpoints is not None:
            self._routing_indices = torch.empty(
                (2, route_capacity), dtype=torch.int64, device=scheduler.device
            )
        else:
            self._routing_indices = None

        # Request identity is immutable across steady decode. One stable
        # table is shared by overlap generations and updated only after a
        # rare topology/lifecycle fence.
        self._gpu_checkpoint_slots = (
            torch.full(
                (route_capacity, self.checkpoints.capacity),
                -1,
                dtype=torch.int64,
                device=scheduler.device,
            )
            if self.checkpoints is not None
            else None
        )
        self._gpu_batch_seats = torch.empty(
            (route_capacity,),
            dtype=torch.int64,
            device=scheduler.device,
        )
        self._gpu_batch_epochs = torch.empty_like(self._gpu_batch_seats)
        self._gpu_batch_seats_cpu = torch.empty(
            (route_capacity,),
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(scheduler.device),
        )
        self._gpu_batch_epochs_cpu = torch.empty_like(
            self._gpu_batch_seats_cpu,
            pin_memory=self._gpu_batch_seats_cpu.is_pinned(),
        )
        self._gpu_identity_done_event = scheduler.device_module.Event()
        self._gpu_identity_in_use = False
        self._gpu_identity = None
        outcome_capacity = self._kv_outcome_flush_interval * route_capacity
        self._gpu_kv_outcome_stage = torch.empty(
            (outcome_capacity, 3),
            dtype=torch.int64,
            device=scheduler.device,
        )
        self._cpu_kv_outcome_stage = torch.empty(
            (outcome_capacity, 3),
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(scheduler.device),
        )

        # filter/merge replace reqs; an intervening prefill also changes the
        # shared identity table. Reuse it only for the same request-list owner.
        self._gpu_identity_reqs = None

    def close(self) -> None:
        try:
            self._flush_pending_kv_outcomes()
            self.data_plane.close()
        finally:
            if self.checkpoints is not None:
                self.checkpoints.close()

    def take_decode_metrics_window(self) -> DecoupledSpecDecodeMetrics:
        """Exchange the drafter's role-local native transport counters."""

        return DecoupledSpecDecodeMetrics(
            transport=DraftTransportMetrics.from_dict(
                self.data_plane.take_transport_metrics()
            )
        )

    def _assign_gpu_identity(self, batch: ScheduleBatch) -> None:
        num_reqs = len(batch.reqs)
        launch_identity = []
        for req in batch.reqs:
            state = self._state_for_req(req)
            launch_identity.append((state.gpu_seat, state.key.request_epoch))
        launch_identity = tuple(launch_identity)
        identity_changed = launch_identity != self._gpu_identity
        if identity_changed:
            self._gpu_decode_binding_validated = False
            if self._gpu_identity_in_use:
                # Capture all previously submitted readers only when replacing
                # the identity table. Steady decode needs no per-forward event.
                self._gpu_identity_done_event.record(
                    self.scheduler.forward_stream
                    if self.scheduler.enable_overlap
                    else self.scheduler.schedule_stream
                )
                self.scheduler.schedule_stream.wait_event(self._gpu_identity_done_event)
            # A stream wait cannot fence CPU writes to an in-flight pinned H2D
            # source. Each topology change owns immutable staging; PyTorch's
            # pinned allocator retains its storage until the copy completes.
            pin_memory = self._gpu_batch_seats_cpu.is_pinned()
            self._gpu_batch_seats_cpu = torch.tensor(
                [seat for seat, _ in launch_identity],
                dtype=torch.int64,
                pin_memory=pin_memory,
            )
            self._gpu_batch_epochs_cpu = torch.tensor(
                [epoch for _, epoch in launch_identity],
                dtype=torch.int64,
                pin_memory=pin_memory,
            )
            self._gpu_batch_seats[:num_reqs].copy_(
                self._gpu_batch_seats_cpu[:num_reqs],
                non_blocking=self._gpu_batch_seats_cpu.is_pinned(),
            )
            self._gpu_batch_epochs[:num_reqs].copy_(
                self._gpu_batch_epochs_cpu[:num_reqs],
                non_blocking=self._gpu_batch_epochs_cpu.is_pinned(),
            )
            self._gpu_identity = launch_identity
        self._gpu_identity_in_use = True
        self._gpu_identity_reqs = batch.reqs
        batch.decoupled_draft_mirror_seats = self._gpu_batch_seats[:num_reqs]
        batch.decoupled_draft_request_epochs = self._gpu_batch_epochs[:num_reqs]

    @scheduler_nvtx_method("decoupled_draft.prepare_state_routes")
    def prepare_batch(self, batch: ScheduleBatch) -> None:
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return
        num_reqs = len(batch.reqs)
        if not num_reqs:
            return
        same_identity = batch.reqs is self._gpu_identity_reqs
        if not same_identity:
            decoupled_rows = [
                req.decoupled_draft_generation is not None for req in batch.reqs
            ]
            if any(decoupled_rows) and not all(decoupled_rows):
                raise RuntimeError(
                    "Decoupled drafter cannot mix control-owned and ordinary "
                    "rows in one extend/decode batch."
                )
            if all(decoupled_rows):
                self._assign_gpu_identity(batch)
        else:
            batch.decoupled_draft_mirror_seats = self._gpu_batch_seats[:num_reqs]
            batch.decoupled_draft_request_epochs = self._gpu_batch_epochs[:num_reqs]

        gpu_managed = batch.decoupled_draft_mirror_seats is not None
        num_prefill = (
            0
            if batch.forward_mode.is_decode()
            else (
                batch.decoupled_draft_num_prefill_reqs
                if batch.decoupled_draft_num_prefill_reqs is not None
                else num_reqs
            )
        )
        if self.checkpoints is not None:
            if num_reqs > self._routing_indices.shape[1]:
                raise RuntimeError("Decoupled draft state-route capacity exceeded.")
            routes = self._routing_indices[:, :num_reqs]
            batch.mamba_cache_src_indices = routes[0]
            batch.mamba_cache_dst_indices = routes[1]

        if gpu_managed and num_prefill < num_reqs:
            if not batch.defer_decode_kv_binding:
                raise RuntimeError("Decoupled draft decode requires GPU KV binding.")
            if self.checkpoints is not None and not same_identity:
                for req in batch.reqs[num_prefill:]:
                    if not self._state_for_req(req).gpu_checkpoint_slots_initialized:
                        raise RuntimeError(
                            "Decoupled draft decode started before its GPU "
                            "checkpoint ring was published."
                        )
            if num_prefill == 0 and num_reqs == 1:
                req_pool_index = int(batch.reqs[0].req_pool_idx)
                batch.input_ids = self.scheduler.future_map.output_tokens_buf[
                    req_pool_index : req_pool_index + 1
                ]
            if num_prefill == 0:
                return

        # Prefill retains the upstream active slot, including radix COW and
        # tracking. The ring borrows that slot at the initial decode boundary.
        # Routes originate on device; no mutable pinned H2D staging is reused.
        prefill_reqs = batch.reqs[:num_prefill] if gpu_managed else batch.reqs
        for req in prefill_reqs:
            if not gpu_managed:
                continue
            state = self._state_for_req(req)
            state.kv_highwater_len = max(
                state.kv_highwater_len, int(req.kv.kv_allocated_len)
            )
            if (
                self.checkpoints is not None
                and not state.gpu_checkpoint_slots_initialized
            ):
                if req.mamba_pool_idx is None:
                    raise RuntimeError("Decoupled draft row has no Mamba state slot.")
                # Ordinary admission reserves the active/tracking slots only.
                # Reclaim evictable radix snapshots before reserving the ring.
                shortfall = (
                    self.checkpoints.capacity
                    - 1
                    - self.scheduler.req_to_token_pool.mamba_allocator.schedulable_available_size()
                )
                if shortfall > 0:
                    self.scheduler.tree_cache.evict(
                        EvictParams(num_tokens=0, mamba_num=shortfall)
                    )
                slots = self.checkpoints.initialize(
                    state.key,
                    position=len(req.origin_input_ids) + len(req.output_ids),
                    active_slot=req.mamba_pool_idx,
                )
                self._gpu_checkpoint_slots[state.gpu_seat].copy_(slots)
                state.gpu_checkpoint_slots_initialized = True
        if self.checkpoints is not None and prefill_reqs:
            if any(req.mamba_pool_idx is None for req in prefill_reqs):
                raise RuntimeError("Decoupled draft row has no Mamba state slot.")
            active_slots = torch.stack(
                [req.mamba_pool_idx.reshape(()) for req in prefill_reqs]
            )
            routes[:, : len(prefill_reqs)].copy_(active_slots.unsqueeze(0))

    def abort_request(self, req: Req, reason: str = "abort") -> None:
        # Drafter requests are owned by verifier controls, not HTTP clients.
        return None

    def retract_request(self, req: Req) -> None:
        if req.decoupled_draft_generation is not None:
            raise RuntimeError(
                "Decoupled drafter cannot re-prefill a live request; "
                "size KV/Mamba capacity to avoid drafter retraction."
            )

    def handle_retracted_requests(
        self,
        retracted_reqs: list[Req],
        reqs_to_abort: list[Req],
    ) -> tuple[list[Req], list[Req]]:
        # Control-owned rows must fail before their GPU state is discarded.
        for req in retracted_reqs:
            self.retract_request(req)
        for req in reqs_to_abort:
            self.retract_request(req)
        return retracted_reqs, reqs_to_abort

    def prepare_pause_retract(self, reqs: list[Req]) -> list[Req]:
        """Reject pause retraction while any control-owned row is live."""

        unique_reqs = []
        seen_req_ids = set()
        candidates = [*reqs, *self._sleeping_requests.values()]
        for req in candidates:
            if id(req) in seen_req_ids:
                continue
            seen_req_ids.add(id(req))
            self.retract_request(req)
            unique_reqs.append(req)
        return unique_reqs

    def _build_draft_decode_batch(self, reqs: list[Req]) -> ScheduleBatch:
        """Rebuild scheduler metadata for rows that retained Req/KV ownership."""

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
            tree_cache=self.scheduler.tree_cache,
            model_config=self.scheduler.model_config,
            enable_overlap=self.scheduler.enable_overlap,
            spec_algorithm=self.scheduler.spec_algorithm,
        )
        device = self.scheduler.device
        req_pool_indices = [int(req.req_pool_idx) for req in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        # Actual positions and inputs are resolved on GPU at forward entry.
        batch.seq_lens = torch.ones(len(reqs), dtype=torch.int64, device=device)
        batch.seq_lens_cpu = None
        batch.orig_seq_lens = torch.ones(len(reqs), dtype=torch.int32, device=device)
        batch.seq_lens_sum = None
        last_tokens = torch.zeros(len(reqs), dtype=torch.int64, device=device)
        self.scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(bonus_tokens=last_tokens),
        )
        batch.input_ids = None
        batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )
        return batch

    @scheduler_nvtx_method("decoupled_draft.wake_sleeping")
    def _wake_sleeping_requests(self) -> None:
        wake_reqs = []
        for key, req in list(self._sleeping_requests.items()):
            state = self._requests.get((key.src_verifier_rank, key.request_id))
            if state is None or state.req is not req:
                self._sleeping_requests.pop(key, None)
                continue
            ahead_len = state.gpu_ahead_len
            if ahead_len >= self.schedule_ahead_limit:
                continue
            if req.req_pool_idx is None or req.kv is None:
                raise RuntimeError(
                    "Sleeping decoupled draft request lost its KV ownership: "
                    f"request_id={key.request_id}"
                )
            state.is_sleeping = False
            self._sleeping_requests.pop(key, None)
            wake_reqs.append(req)

        if not wake_reqs:
            return
        wake_batch = self._build_draft_decode_batch(wake_reqs)
        running_batch = self.scheduler.running_batch
        if running_batch is None or running_batch.is_empty():
            self.scheduler.running_batch = wake_batch
        else:
            running_batch.merge_batch(wake_batch)
            running_batch.batch_is_full = False

    @scheduler_nvtx_method("decoupled_draft.process_controls")
    def process_pending_controls(self, *, force_lifecycle: bool = False) -> None:
        # The host sees lifecycle and pacing only. VerifyCommit is reconciled
        # directly on GPU, even when scheduler iterations execute serially.
        if self._drain_gpu_progress():
            self._wake_sleeping_requests()
        if self._requests and not force_lifecycle:
            self._gpu_lifecycle_poll_count += 1
            if self._gpu_lifecycle_poll_count < self._gpu_lifecycle_poll_interval:
                return
            self._gpu_lifecycle_poll_count = 0
            if self.data_plane.pending_control_count() == 0:
                return
        ready = self.data_plane.collect_lifecycle_controls()
        for draft_key in ready.close_keys:
            self._close_request_key(draft_key)
        for message in ready.sync_messages:
            self._open_request(message)

    @scheduler_nvtx_method("decoupled_draft.drain_gpu_progress")
    def _drain_gpu_progress(self) -> bool:
        """Refresh only host scheduling credit from cumulative GPU snapshots."""

        changed = False
        for row in self.data_plane.drain_gpu_progress():
            (
                request_id,
                src_verifier_rank,
                request_epoch,
                logical_output_len,
                committed_len,
                raw_tail_len,
            ) = row
            state = self._requests.get((int(src_verifier_rank), str(request_id)))
            if state is None or state.key.request_epoch != int(request_epoch):
                continue
            logical_output_len = int(logical_output_len)
            committed_len = int(committed_len)
            raw_tail_len = int(raw_tail_len)
            if (
                committed_len < 0
                or logical_output_len < committed_len
                or raw_tail_len != logical_output_len - committed_len
                or raw_tail_len < 0
                or raw_tail_len > self.ahead_window
            ):
                raise RuntimeError(
                    "GPU drafter progress violated transcript cursors: "
                    f"request_id={request_id} committed_len={committed_len} "
                    f"logical_output_len={logical_output_len} "
                    f"raw_tail_len={raw_tail_len}"
                )
            if state.gpu_ahead_len != raw_tail_len:
                state.gpu_ahead_len = raw_tail_len
                changed = True
        return changed

    @scheduler_nvtx_method("decoupled_draft.flush_kv_outcomes")
    def _flush_pending_kv_outcomes(self, *, blocking: bool = True) -> None:
        """Drain at CLOSE; during decode only consume completed D2H batches."""

        if self._inflight_kv_outcomes is not None:
            reqs, cpu_outcomes, copy_done, _keep_alive = self._inflight_kv_outcomes
            if blocking:
                copy_done.synchronize()
            elif not copy_done.query():
                return
            self._consume_kv_outcomes(reqs, cpu_outcomes.tolist())
            self._inflight_kv_outcomes = None

        if not self._pending_kv_outcomes:
            return
        if (
            not blocking
            and len(self._pending_kv_outcomes) < self._kv_outcome_flush_interval
        ):
            return
        reqs = []
        outcome_tensors = []
        for outcome_reqs, kv_outcomes, _ in self._pending_kv_outcomes:
            reqs.extend(outcome_reqs)
            outcome_tensors.append(kv_outcomes)
        if outcome_tensors[0].is_cuda:
            if not all(outcome.is_cuda for outcome in outcome_tensors):
                raise RuntimeError("Mixed CPU/GPU drafter KV outcomes.")
            ready_event = self._pending_kv_outcomes[-1][2]
            if ready_event is None:
                raise RuntimeError("GPU drafter KV outcomes have no ready event.")
            num_outcome_rows = len(reqs)
            if num_outcome_rows > self._gpu_kv_outcome_stage.shape[0]:
                raise RuntimeError(
                    "Decoupled drafter KV outcome batch exceeded its fixed "
                    f"capacity: rows={num_outcome_rows} "
                    f"capacity={self._gpu_kv_outcome_stage.shape[0]}"
                )
            self.scheduler.copy_stream.wait_event(ready_event)
            with self.scheduler.copy_stream_ctx:
                gpu_outcomes = self._gpu_kv_outcome_stage[:num_outcome_rows]
                torch.cat(outcome_tensors, dim=0, out=gpu_outcomes)
                cpu_outcomes = self._cpu_kv_outcome_stage[:num_outcome_rows]
                cpu_outcomes.copy_(
                    gpu_outcomes,
                    non_blocking=cpu_outcomes.is_pinned(),
                )
                copy_done = self.scheduler.device_module.Event()
                copy_done.record()
            # Keep forward-allocated source tensors alive until copy_stream has
            # read them. The staging buffers cannot be reused before this event.
            self._pending_kv_outcomes.clear()
            if not blocking:
                self._inflight_kv_outcomes = (
                    reqs,
                    cpu_outcomes,
                    copy_done,
                    outcome_tensors,
                )
                return
            copy_done.synchronize()
            outcome_rows = cpu_outcomes.tolist()
        else:
            outcome_rows = torch.cat(outcome_tensors, dim=0).tolist()

        self._pending_kv_outcomes.clear()
        self._consume_kv_outcomes(reqs, outcome_rows)

    def _consume_kv_outcomes(self, reqs, outcome_rows) -> None:
        """Apply completed ownership records exactly once, before request release."""
        reclaim_locs = []
        for req, outcome in zip(reqs, outcome_rows):
            bound_position = int(outcome[1])
            generation = req.decoupled_draft_generation
            state = (
                None
                if generation is None
                else self._requests.get(
                    (generation.src_verifier_rank, generation.request_id)
                )
            )
            if (
                bound_position >= 0
                and state is not None
                and state.req is req
                and state.key == generation
            ):
                state.kv_highwater_len = max(
                    state.kv_highwater_len,
                    bound_position + 1,
                )
            reclaim_loc = int(outcome[2])
            if reclaim_loc > 0:
                reclaim_locs.append(reclaim_loc)
        if reclaim_locs:
            self.scheduler.token_to_kv_pool_allocator.free(
                torch.tensor(
                    reclaim_locs,
                    dtype=torch.int64,
                    device=self.scheduler.device,
                )
            )

    @scheduler_nvtx_method("decoupled_draft.prepare_decode_allocation")
    def prepare_decode_allocation(self, batch: ScheduleBatch) -> None:
        first_is_decoupled = (
            bool(batch.reqs) and batch.reqs[0].decoupled_draft_generation is not None
        )
        if any(
            (req.decoupled_draft_generation is not None) != first_is_decoupled
            for req in batch.reqs[1:]
        ):
            raise RuntimeError(
                "Decoupled drafter GPU state management cannot mix control-owned and "
                "ordinary decode rows in one batch."
            )
        batch.defer_decode_kv_binding = first_is_decoupled
        if batch.defer_decode_kv_binding:
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None

    def before_decode_retraction(self, batch: ScheduleBatch) -> None:
        if any(req.decoupled_draft_generation is not None for req in batch.reqs):
            raise RuntimeError(
                "Decoupled drafter GPU state management cannot retract a live decode "
                "request; size KV/Mamba capacity to avoid drafter retraction."
            )

    @scheduler_nvtx_method("decoupled_draft.prepare_forward")
    def prepare_forward(self, batch: ScheduleBatch) -> None:
        if batch.decoupled_draft_mirror_seats is None:
            return
        is_decode = batch.forward_mode.is_decode()
        is_mixed = batch.decoupled_draft_num_prefill_reqs is not None
        if batch.forward_mode.is_extend():
            # OPEN must precede the first sample. Decode-only iterations do
            # not wait for remote commits; their GPU transactions reconcile them.
            self.data_plane.gpu_tail_buffer.wait_for_landing()
        if not (is_decode or is_mixed):
            return
        if batch.input_ids is None:
            raise RuntimeError("GPU draft reconcile requires resolved decode inputs.")
        # Ordinary decode passes the original tensors; only MIXED creates
        # suffix views, keeping view construction off the steady decode path.
        request_start = batch.decoupled_draft_num_prefill_reqs if is_mixed else 0
        token_start = batch.decoupled_draft_num_prefill_tokens if is_mixed else 0
        rows = slice(request_start, None)
        tokens = slice(token_start, None)
        batch.decoupled_draft_captured_state_positions = torch.empty_like(
            (batch.seq_lens[rows] if is_mixed else batch.seq_lens)
        )
        batch.decoupled_draft_old_cache_locs = torch.empty_like(
            (batch.out_cache_loc[tokens] if is_mixed else batch.out_cache_loc)
        )
        self.data_plane.gpu_tail_buffer.prepare_decode(
            (
                batch.decoupled_draft_mirror_seats[rows]
                if is_mixed
                else batch.decoupled_draft_mirror_seats
            ),
            (
                batch.decoupled_draft_request_epochs[rows]
                if is_mixed
                else batch.decoupled_draft_request_epochs
            ),
            (batch.req_pool_indices[rows] if is_mixed else batch.req_pool_indices),
            (batch.out_cache_loc[tokens] if is_mixed else batch.out_cache_loc),
            self._gpu_checkpoint_slots,
            batch.req_to_token_pool.req_to_token,
            resolved_input_ids=(
                batch.input_ids[tokens] if is_mixed else batch.input_ids
            ),
            resolved_seq_lens=(batch.seq_lens[rows] if is_mixed else batch.seq_lens),
            resolved_orig_seq_lens=(
                batch.orig_seq_lens[rows] if is_mixed else batch.orig_seq_lens
            ),
            mamba_src_indices=(
                None
                if batch.mamba_cache_src_indices is None
                else (
                    batch.mamba_cache_src_indices[rows]
                    if is_mixed
                    else batch.mamba_cache_src_indices
                )
            ),
            mamba_dst_indices=(
                None
                if batch.mamba_cache_dst_indices is None
                else (
                    batch.mamba_cache_dst_indices[rows]
                    if is_mixed
                    else batch.mamba_cache_dst_indices
                )
            ),
            captured_state_positions=batch.decoupled_draft_captured_state_positions,
            old_cache_locs=batch.decoupled_draft_old_cache_locs,
            validate_inputs=not self._gpu_decode_binding_validated,
        )

    @scheduler_nvtx_method("decoupled_draft.finish_forward")
    def finish_forward(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> bool:
        if batch.decoupled_draft_mirror_seats is None:
            return False
        is_decode = batch.forward_mode.is_decode()
        is_mixed = batch.decoupled_draft_num_prefill_reqs is not None
        has_prefill = (
            batch.forward_mode.is_extend() and batch.contains_last_prefill_chunk
        )
        if not (is_decode or is_mixed or has_prefill):
            return False
        sampled_tokens = result.next_token_ids
        if not isinstance(sampled_tokens, torch.Tensor):
            raise TypeError("GPU draft finish requires device sampled tokens.")
        num_prefill = (
            batch.decoupled_draft_num_prefill_reqs
            if is_mixed
            else 0 if is_decode else len(batch.reqs)
        )
        if is_decode or is_mixed:
            request_start = num_prefill
            token_start = batch.decoupled_draft_num_prefill_tokens if is_mixed else 0
            rows = slice(request_start, None)
            tokens = slice(token_start, None)
            kv_outcomes = torch.empty(
                (len(batch.reqs) - num_prefill, 3),
                dtype=torch.int64,
                device=sampled_tokens.device,
            )
            self.data_plane.gpu_tail_buffer.finish_decode(
                (
                    batch.decoupled_draft_mirror_seats[rows]
                    if is_mixed
                    else batch.decoupled_draft_mirror_seats
                ),
                (
                    batch.decoupled_draft_request_epochs[rows]
                    if is_mixed
                    else batch.decoupled_draft_request_epochs
                ),
                (batch.req_pool_indices[rows] if is_mixed else batch.req_pool_indices),
                (batch.out_cache_loc[tokens] if is_mixed else batch.out_cache_loc),
                (sampled_tokens[rows] if is_mixed else sampled_tokens),
                batch.req_to_token_pool.req_to_token,
                resolved_input_tokens=(
                    batch.input_ids[tokens] if is_mixed else batch.input_ids
                ),
                captured_state_positions=batch.decoupled_draft_captured_state_positions,
                old_cache_locs=batch.decoupled_draft_old_cache_locs,
                kv_outcomes=kv_outcomes,
                future_output_tokens=self.scheduler.future_map.output_tokens_buf,
                validate_inputs=not self._gpu_decode_binding_validated,
            )
            self._gpu_decode_binding_validated = True
            result.decoupled_draft_gpu_managed = is_decode
            result.decoupled_draft_num_prefill_reqs = num_prefill
            result.decoupled_draft_kv_outcomes = kv_outcomes
            result.decoupled_draft_kv_outcomes_ready = (
                self.scheduler.device_module.Event()
            )
            result.decoupled_draft_kv_outcomes_ready.record()
        if has_prefill:
            mirror_seats = batch.decoupled_draft_mirror_seats[:num_prefill]
            if batch.chunked_req is not None:
                mirror_seats = mirror_seats.clone()
                mirror_seats[batch.reqs.index(batch.chunked_req)] = -1
            candidate_committed = torch.empty(
                (num_prefill,),
                dtype=torch.bool,
                device=sampled_tokens.device,
            )
            self.data_plane.gpu_tail_buffer.append_prefill_sample(
                mirror_seats,
                batch.decoupled_draft_request_epochs[:num_prefill],
                sampled_tokens[:num_prefill],
                accept_out=candidate_committed,
            )
            result.decoupled_draft_candidate_committed = candidate_committed
            # finish_decode already relayed the authoritative decode token;
            # raw samples may differ during forced replay or branch rewind.
            self.scheduler.future_map.stash(
                batch.req_pool_indices[:num_prefill],
                RelayPayload(bonus_tokens=sampled_tokens[:num_prefill].to(torch.int64)),
            )
        return True

    @scheduler_nvtx_method("decoupled_draft.sleep_overrun")
    def sleep_overrun_requests(self, batch: ScheduleBatch) -> ScheduleBatch:
        """Transfer ahead rows before decode allocates this round's KV slots."""

        if batch is None or batch.is_empty():
            return batch
        keep_indices = []
        for index, req in enumerate(batch.reqs):
            if req.decoupled_draft_generation is None:
                keep_indices.append(index)
                continue
            state = self._state_for_req(req)
            ahead_len = state.gpu_ahead_len
            if ahead_len >= self.schedule_ahead_limit:
                if state.is_sleeping:
                    raise RuntimeError(
                        "A sleeping decoupled draft request remained in running_batch: "
                        f"request_id={state.key.request_id}"
                    )
                state.is_sleeping = True
                self._sleeping_requests[state.key] = req
            else:
                keep_indices.append(index)
        if len(keep_indices) != len(batch.reqs):
            batch.filter_batch(keep_indices=keep_indices)
            batch.batch_is_full = False
        return batch

    def has_pending_work(self) -> bool:
        # A GPU-rejected prefill can leave no runnable batch before its CLOSE
        # reaches the CPU. Keep lifecycle ownership alive until that is drained.
        return bool(self._requests)

    @scheduler_nvtx_method("decoupled_draft.wait_for_control")
    def on_no_batch(self) -> bool:
        """Keep the control-driven drafter alive without entering idle cleanup."""

        if not self.has_pending_work():
            return False
        # Wake immediately on verifier control; the finite bound only lets the
        # scheduler service shutdown/health work when the peer is silent.
        self.data_plane.wait_for_control(0.01)
        # With no result-side hook, consume lifecycle and compact GPU pacing
        # progress here. VerifyCommit tokens never enter the CPU scheduler.
        self.process_pending_controls(force_lifecycle=True)
        return True

    def before_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> bool:
        gpu_managed_decode = result.decoupled_draft_gpu_managed
        committed_mask = result.decoupled_draft_candidate_committed
        kv_outcomes = result.decoupled_draft_kv_outcomes
        kv_outcomes_drained = result.decoupled_draft_kv_outcomes_drained
        num_prefill = result.decoupled_draft_num_prefill_reqs
        has_decode = gpu_managed_decode or num_prefill is not None
        decode_reqs = batch.reqs[num_prefill:] if num_prefill else batch.reqs
        if not has_decode and committed_mask is None:
            return False

        # GPU-managed decode bypasses the generic result processor, but these
        # optional host-side captures retain their ordinary lifecycle.
        routed_experts_output = result.routed_experts_output
        if routed_experts_output is not None:
            routed_experts_output.finalize()
            result.routed_experts_output = None
        indexer_topk_output = result.indexer_topk_output
        if indexer_topk_output is not None:
            indexer_topk_output.finalize()
            result.indexer_topk_output = None

        if has_decode:
            if not self.scheduler.enable_overlap:
                ready = result.decoupled_draft_kv_outcomes_ready
                if ready is not None:
                    ready.synchronize()
            if kv_outcomes is None and not kv_outcomes_drained:
                for req in decode_reqs:
                    generation = req.decoupled_draft_generation
                    state = (
                        None
                        if generation is None
                        else self._requests.get(
                            (
                                generation.src_verifier_rank,
                                generation.request_id,
                            )
                        )
                    )
                    if state is not None and state.req is req:
                        raise RuntimeError(
                            "Live GPU-managed drafter decode is missing KV outcomes."
                        )
            elif kv_outcomes is not None:
                self._pending_kv_outcomes.append(
                    (
                        tuple(decode_reqs),
                        kv_outcomes,
                        result.decoupled_draft_kv_outcomes_ready,
                    )
                )
                result.decoupled_draft_kv_outcomes = None
                result.decoupled_draft_kv_outcomes_ready = None
            if gpu_managed_decode:
                result.copy_done = None
            # Bound pending ownership without fencing every copy submission.
            # Only fall back to a drain if a whole subsequent batch accumulated
            # while the previous D2H was still in flight.
            if (
                self._inflight_kv_outcomes is not None
                or len(self._pending_kv_outcomes) >= self._kv_outcome_flush_interval
            ):
                self._flush_pending_kv_outcomes(
                    blocking=(
                        len(self._pending_kv_outcomes)
                        >= self._kv_outcome_flush_interval
                        and self._inflight_kv_outcomes is not None
                    )
                )
            # Preserve decode telemetry while intentionally bypassing the normal
            # processor's Req.output_ids append and output streaming.
            can_run_cuda_graph = bool(result.can_run_cuda_graph)
            if get_observability().enable_metrics:
                self.scheduler.metrics_collector.increment_decode_cuda_graph_pass(
                    value=can_run_cuda_graph
                )
            metrics_reporter = self.scheduler.metrics_reporter
            metrics_reporter.num_generated_tokens += len(decode_reqs)
            metrics_reporter.forward_ct_decode = (
                metrics_reporter.forward_ct_decode + 1
            ) % (1 << 30)
            if gpu_managed_decode:
                metrics_reporter.report_decode_stats(
                    can_run_cuda_graph,
                    running_batch=batch,
                    num_correct_drafts=0,
                )
            if gpu_managed_decode:
                return True
        if committed_mask is None:
            return False

        # Final prefill still needs the ordinary processor to append its first
        # sampled token and admit the Req into decode. A rejected old-lifetime
        # row is marked retracted and is skipped by that processor.
        if result.copy_done is not None:
            result.copy_done.synchronize()
            result.copy_done = None
        # Lifecycle polling may filter the live ScheduleBatch. Serial mode
        # processes that same object, so defer CLOSE to the next loop boundary
        # until these launch-time token/mask rows have been consumed.
        for req in batch.reqs:
            generation = req.decoupled_draft_generation
            state = (
                None
                if generation is None
                else self._requests.get(
                    (generation.src_verifier_rank, generation.request_id)
                )
            )
            if state is None or state.req is not req or state.key != generation:
                req.is_retracted = True
        for req, candidate_committed in zip(batch.reqs, committed_mask.tolist()):
            if req is batch.chunked_req:
                # The generic processor still owns middle-chunk accounting.
                continue
            if not bool(candidate_committed):
                req.is_retracted = True
        return False

    @scheduler_nvtx_method("decoupled_draft.after_batch_result")
    def after_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return
        # Native cumulative snapshots own token publication and ACKs. Device
        # finish publishes checkpoint tags; there is no host decode transcript.
        self.scheduler.metrics_reporter.finish_decoupled_decode_metrics_window()
        batch.mamba_cache_src_indices = None
        batch.mamba_cache_dst_indices = None

    def _open_request(self, message: DraftSync) -> None:
        table_key = (int(message.src_verifier_rank), str(message.request_id))
        if table_key in self._requests:
            raise RuntimeError(
                f"DraftSync reopened a live request: request={table_key}"
            )
        max_new_tokens = int(message.max_new_tokens)
        if max_new_tokens <= len(message.committed_outputs):
            raise RuntimeError(
                "DraftSync max_new_tokens must exceed its committed prefix: "
                f"request_id={message.request_id} "
                f"max_new_tokens={max_new_tokens} "
                f"committed_len={len(message.committed_outputs)}"
            )
        sampling_params = SamplingParams(
            # The verifier owns completion and sends CLOSE. Keep the mirror
            # alive until that lifecycle boundary rather than applying a second
            # drafter-side output limit.
            max_new_tokens=1 << 30,
            temperature=0.0,
            top_k=1,
            ignore_eos=True,
        )
        sampling_params.normalize(self.scheduler.tokenizer)
        sampling_params.verify(self.scheduler.model_config.vocab_size)
        req = Req(
            build_draft_scheduler_rid(message.draft_key),
            "",
            array("q", [int(token) for token in message.prompt_token_ids]),
            sampling_params,
            return_logprob=False,
            stream=False,
            eos_token_ids=self.scheduler.model_config.hf_eos_token_id,
            vocab_size=self.scheduler.model_config.vocab_size,
            metrics_collector=(
                self.scheduler.metrics_collector
                if self.scheduler.server_args.enable_metrics
                else None
            ),
        )
        req.tokenizer = self.scheduler.tokenizer
        req.output_ids = array("q", [int(token) for token in message.committed_outputs])
        req._refresh_fill_ids()
        self.scheduler.init_req_max_new_tokens(req)
        binding = self.data_plane.lookup_gpu_binding(
            message.request_id,
            int(message.src_verifier_rank),
        )
        if binding is None:
            raise RuntimeError(
                "Native drafter GPU control did not bind DraftSync before "
                f"scheduler admission: request_id={message.request_id}"
            )
        gpu_seat, request_epoch = (int(binding[0]), int(binding[1]))
        generation_key = DraftRequestGeneration(
            src_verifier_rank=int(message.src_verifier_rank),
            request_id=str(message.request_id),
            request_epoch=request_epoch,
        )
        req.decoupled_draft_generation = generation_key
        state = _DraftRequestState(
            key=generation_key,
            req=req,
            gpu_seat=gpu_seat,
        )
        self._requests[table_key] = state
        self.scheduler._add_request_to_queue(req)

    def _close_request_key(self, draft_key: DraftReqKey) -> None:
        table_key = (int(draft_key.src_verifier_rank), str(draft_key.request_id))
        state = self._requests.get(table_key)
        if state is None:
            return
        req = state.req
        # A queued overlap result may finish before or after CLOSE lands. Mark
        # the old Req unconditionally so generic result handling never appends
        # into resources owned by a later request generation.
        req.is_retracted = True
        self._sleeping_requests.pop(state.key, None)
        state.is_sleeping = False
        # CLOSE is rare lifecycle work. Order resource reuse after any
        # in-flight forward that may still hold this row/checkpoint ring;
        # steady VerifyCommit never takes this fence.
        if self.scheduler.enable_overlap:
            self.scheduler.schedule_stream.wait_stream(self.scheduler.forward_stream)
        for queued_batch, queued_result in self.scheduler.result_queue:
            kv_outcomes = queued_result.decoupled_draft_kv_outcomes
            if kv_outcomes is None:
                continue
            self._pending_kv_outcomes.append(
                (
                    tuple(
                        queued_batch.reqs[
                            queued_result.decoupled_draft_num_prefill_reqs or 0 :
                        ]
                    ),
                    kv_outcomes,
                    queued_result.decoupled_draft_kv_outcomes_ready,
                )
            )
            queued_result.decoupled_draft_kv_outcomes = None
            queued_result.decoupled_draft_kv_outcomes_drained = True
            queued_result.decoupled_draft_kv_outcomes_ready = None
            queued_result.copy_done = None
        # This is the rare host-synchronous ownership boundary. Flush both
        # already-processed and still-queued outcomes before releasing KV.
        self._flush_pending_kv_outcomes()
        popped_state = self._requests.pop(table_key)
        if popped_state is not state:
            raise RuntimeError("Decoupled draft request changed during CLOSE.")
        # A close owns the request lifecycle, including a partially scheduled
        # prefill. Drop the scheduler's chunk owner before releasing any request
        # resources so the next scheduling step cannot stash or reschedule it.
        if self.scheduler.chunked_req is req:
            self.scheduler.chunked_req = None
        if self.scheduler._pending_chunked_abort_req is req:
            self.scheduler._pending_chunked_abort_req = None
        self.scheduler.waiting_queue = [
            queued for queued in self.scheduler.waiting_queue if queued is not req
        ]
        seen_batches = set()
        for batch in (
            self.scheduler.running_batch,
            self.scheduler.last_batch,
            self.scheduler.cur_batch_for_debug,
        ):
            if batch is None or id(batch) in seen_batches or batch.is_empty():
                continue
            seen_batches.add(id(batch))
            keep_indices = [
                index
                for index, batch_req in enumerate(batch.reqs)
                if batch_req is not req
            ]
            if len(keep_indices) != len(batch.reqs):
                batch.filter_batch(keep_indices=keep_indices)
                batch.batch_is_full = False
        if req.kv is not None:
            if state.kv_highwater_len <= 0:
                raise RuntimeError(
                    "Decoupled draft CLOSE has no GPU KV ownership high-water: "
                    f"request_id={state.key.request_id}"
                )
            req.kv_committed_len = state.kv_highwater_len
            req.kv.kv_allocated_len = state.kv_highwater_len
        if self.checkpoints is not None:
            self.checkpoints.release(state.key)
        if req.req_pool_idx is not None or self.scheduler.tree_cache.supports_mamba():
            release_kv_cache(req, self.scheduler.tree_cache, is_insert=False)

    def _state_for_req(self, req: Req) -> _DraftRequestState:
        key = req.decoupled_draft_generation
        if key is None:
            raise RuntimeError(f"Request {req.rid} is not a decoupled draft request.")
        state = self._requests.get((key.src_verifier_rank, key.request_id))
        if state is None or state.req is not req:
            raise RuntimeError(f"Missing decoupled draft state for request {req.rid}.")
        return state

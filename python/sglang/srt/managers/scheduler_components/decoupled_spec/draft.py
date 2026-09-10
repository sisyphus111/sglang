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
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_observability, get_stream
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.decoupled_draft_checkpoint import (
    DecoupledDraftMambaCheckpointStore,
    DraftRequestGeneration,
    draft_active_state_position,
)
from sglang.srt.speculative.decoupled_spec_data_plane import (
    create_drafter_decoupled_spec_data_plane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftCommitAction,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    build_draft_scheduler_rid,
)
from sglang.srt.utils.common import is_pin_memory_available
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import NextBatchPlan
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


class _DraftRequestState(msgspec.Struct):
    key: DraftRequestGeneration
    req: Any
    src_verifier_rank: int
    # Non-overlap CPU reconciliation cursors. In GPU overlap these remain at
    # admission values and must never be treated as a running transcript.
    committed_len: int
    published_len: int
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
    """Drafter-side request, rollback, pacing, and tail-publish component."""

    def __init__(
        self,
        scheduler: Scheduler,
        config: DecoupledSpecIpcConfig,
    ) -> None:
        self.scheduler = scheduler
        self.config = config
        self.src_drafter_rank = int(config.rank)
        self.num_draft_tokens = int(scheduler.server_args.speculative_num_steps)
        self.ahead_window = 2 * self.num_draft_tokens + 1
        self.gpu_overlap = bool(getattr(scheduler, "enable_overlap", False))
        if self.gpu_overlap and scheduler.future_map.needs_cpu_seq_lens:
            raise ValueError(
                "Decoupled drafter GPU overlap requires attention backends "
                "that build decode metadata from device sequence lengths."
            )
        if (
            self.gpu_overlap
            and scheduler.token_to_kv_pool_allocator.page_size != 1
        ):
            raise ValueError(
                "Decoupled drafter GPU overlap currently requires page_size=1 "
                "for one-candidate decode allocation."
            )
        # Generic overlap schedules one decode before the previous result-side
        # pacing update. Reserve that in-flight slot at the scheduling boundary.
        self.schedule_ahead_limit = self.ahead_window - int(self.gpu_overlap)
        self.is_entry_rank = scheduler.ps.tp_rank == 0
        if scheduler.ps.tp_size != 1:
            raise ValueError("The phase-one decoupled drafter requires tp_size == 1.")
        route_capacity = (
            int(scheduler.max_running_requests) if self.gpu_overlap else None
        )
        if self.gpu_overlap:
            self.data_plane = create_drafter_decoupled_spec_data_plane(
                config,
                device=scheduler.device,
                num_gpu_seats=route_capacity,
                num_draft_tokens=self.num_draft_tokens,
                pending_token_capacity=max(
                    self.ahead_window,
                    (
                        int(scheduler.max_total_num_tokens)
                        + route_capacity
                        - 1
                    )
                    // route_capacity,
                ),
                landing_stream=get_stream("decoupled_spec_drafter_landing"),
            )
        else:
            self.data_plane = create_drafter_decoupled_spec_data_plane(config)
        self.data_plane.start()
        self._requests: dict[tuple[int, str], _DraftRequestState] = {}
        self._sleeping_requests: dict[DraftRequestGeneration, Req] = {}
        self._pending_kv_outcomes = []
        self._inflight_kv_outcomes = None
        self._gpu_decode_binding_validated = False
        self._kv_outcome_flush_interval = 32
        # VerifyCommit is consumed by the GPU mirror without scheduler
        # participation. While GPU-managed decode is active, OPEN/CLOSE may lag
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
            if getattr(req_pool, "mamba_pool", None) is not None
            else None
        )
        if self.checkpoints is not None:
            if route_capacity is None:
                route_capacity = int(scheduler.max_running_requests)
            self._routing_indices = torch.empty(
                (2, route_capacity), dtype=torch.int64, device=scheduler.device
            )
            self._routing_cpu = torch.empty(
                (2, route_capacity),
                dtype=torch.int64,
                pin_memory=is_pin_memory_available(scheduler.device),
            )
            self._routing_cpu_array = self._routing_cpu.numpy()
        else:
            self._routing_indices = None
            self._routing_cpu = None
            self._routing_cpu_array = None

        if self.gpu_overlap:
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
        else:
            self._gpu_checkpoint_slots = None
            self._gpu_batch_seats = None
            self._gpu_batch_epochs = None
            self._gpu_batch_seats_cpu = None
            self._gpu_batch_epochs_cpu = None
            self._gpu_identity_done_event = None
            self._gpu_identity_in_use = False
            self._gpu_identity = None
            self._gpu_kv_outcome_stage = None
            self._cpu_kv_outcome_stage = None

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
                self._gpu_identity_done_event.record(self.scheduler.forward_stream)
                self.scheduler.schedule_stream.wait_event(
                    self._gpu_identity_done_event
                )
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
        if (
            batch.defer_decode_kv_binding
            and batch.decoupled_draft_mirror_seats is not None
            and batch.reqs is self._gpu_identity_reqs
        ):
            # Identity is unchanged during steady decode. Only recurrent
            # models need their per-batch state-route views restored here.
            num_reqs = len(batch.reqs)
            if self.checkpoints is not None:
                if num_reqs > self._routing_indices.shape[1]:
                    raise RuntimeError(
                        "decoupled draft Mamba routing exceeded its preallocated "
                        f"capacity: required={num_reqs} "
                        f"capacity={self._routing_indices.shape[1]}"
                    )
                route_slice = self._routing_indices[:, :num_reqs]
                batch.mamba_cache_src_indices = route_slice[0]
                batch.mamba_cache_dst_indices = route_slice[1]
            if num_reqs == 1:
                req_pool_index = int(batch.reqs[0].req_pool_idx)
                batch.input_ids = self.scheduler.future_map.output_tokens_buf[
                    req_pool_index : req_pool_index + 1
                ]
            return
        if self.checkpoints is None:
            if not self.gpu_overlap or not (
                batch.forward_mode.is_extend() or batch.forward_mode.is_decode()
            ):
                return
            decoupled_rows = [
                getattr(req, "decoupled_draft_generation", None) is not None
                for req in batch.reqs
            ]
            if any(decoupled_rows) and not all(decoupled_rows):
                raise RuntimeError(
                    "Decoupled drafter GPU overlap cannot mix control-owned and "
                    "ordinary rows in one extend/decode batch."
                )
            if not decoupled_rows or not all(decoupled_rows):
                return
            if batch.reqs is not self._gpu_identity_reqs:
                self._assign_gpu_identity(batch)
            else:
                num_reqs = len(batch.reqs)
                batch.decoupled_draft_mirror_seats = self._gpu_batch_seats[:num_reqs]
                batch.decoupled_draft_request_epochs = self._gpu_batch_epochs[:num_reqs]
            if batch.forward_mode.is_extend():
                # Dense rollback retains the prefix KV cells; there is no
                # recurrent state pool. CLOSE still owns the prefill allocation
                # even when no decode has retired a KV outcome yet.
                for req in batch.reqs:
                    state = self._state_for_req(req)
                    state.kv_highwater_len = max(
                        state.kv_highwater_len, int(req.kv.kv_allocated_len)
                    )
            elif batch.defer_decode_kv_binding and len(batch.reqs) == 1:
                req_pool_index = int(batch.reqs[0].req_pool_idx)
                batch.input_ids = self.scheduler.future_map.output_tokens_buf[
                    req_pool_index : req_pool_index + 1
                ]
            return
        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return

        num_reqs = len(batch.reqs)
        if num_reqs > self._routing_indices.shape[1]:
            raise RuntimeError(
                "decoupled draft Mamba routing exceeded its preallocated capacity: "
                f"required={num_reqs} capacity={self._routing_indices.shape[1]}"
            )
        is_decode = batch.forward_mode.is_decode()
        decoupled_rows = [
            getattr(req, "decoupled_draft_generation", None) is not None
            for req in batch.reqs
        ]
        if self.gpu_overlap and any(decoupled_rows) and not all(decoupled_rows):
            raise RuntimeError(
                "Decoupled drafter GPU overlap cannot mix control-owned and "
                "ordinary rows in one extend/decode batch."
            )
        gpu_managed_batch = (
            self.gpu_overlap and bool(decoupled_rows) and all(decoupled_rows)
        )
        if gpu_managed_batch:
            self._assign_gpu_identity(batch)
        if self.gpu_overlap and is_decode and batch.defer_decode_kv_binding:
            for req in batch.reqs:
                state = self._state_for_req(req)
                if not state.gpu_checkpoint_slots_initialized:
                    raise RuntimeError(
                        "Decoupled draft decode started before its GPU checkpoint "
                        f"ring was published: request_id={state.key.request_id}"
                    )
            route_slice = self._routing_indices[:, :num_reqs]
            batch.mamba_cache_src_indices = route_slice[0]
            batch.mamba_cache_dst_indices = route_slice[1]
            if num_reqs == 1:
                req_pool_index = int(batch.reqs[0].req_pool_idx)
                batch.input_ids = self.scheduler.future_map.output_tokens_buf[
                    req_pool_index : req_pool_index + 1
                ]
            return

        for req_index, req in enumerate(batch.reqs):
            if req.mamba_pool_idx is None:
                raise RuntimeError(
                    "decoupled draft Mamba routing requires every request to own "
                    f"a state slot: rid={req.rid}"
                )
            key = getattr(req, "decoupled_draft_generation", None)
            if key is None:
                # Non-decoupled maintenance/health rows keep the ordinary in-place
                # route. They are rare and outside the drafter generation hot path.
                slot_id = int(req.mamba_pool_idx.item())
                src_slot_id = dst_slot_id = slot_id
            elif is_decode:
                if not req.output_ids:
                    raise RuntimeError(
                        "decoupled draft Mamba decode requires a predicted tail: "
                        f"rid={req.rid}"
                    )
                state = self._state_for_req(req)
                output_len = len(req.output_ids)
                if output_len < state.committed_len:
                    raise RuntimeError(
                        "decoupled draft output is shorter than its committed "
                        f"prefix: rid={req.rid} output_len={output_len} "
                        f"committed_len={state.committed_len}"
                    )
                active_position = draft_active_state_position(
                    len(req.origin_input_ids), output_len
                )
                # Once decode has consumed the committed tail, rollback can only
                # target the first speculative token. If the committed tail is
                # still the current decode input, retain its preceding state.
                min_position = len(req.origin_input_ids) + state.committed_len
                if output_len == state.committed_len:
                    min_position -= 1
                self.checkpoints.prune(
                    key,
                    min_position=min_position,
                    max_position=active_position,
                )
                src_slot_id, dst_slot_id = self.checkpoints.prepare_decode_route(
                    key,
                    position=active_position,
                )
            else:
                # Chunked prefill repeatedly writes the same slot until the final
                # chunk samples a tail token and commits this logical position.
                dst_slot_id = self.checkpoints.prepare_prefill_route(
                    key,
                    position=len(req.origin_input_ids) + len(req.output_ids),
                )
                src_slot_id = dst_slot_id
                if gpu_managed_batch:
                    state = self._state_for_req(req)
                    state.kv_highwater_len = max(
                        state.kv_highwater_len,
                        int(req.kv.kv_allocated_len),
                    )
                    if not state.gpu_checkpoint_slots_initialized:
                        self._gpu_checkpoint_slots[state.gpu_seat].copy_(
                            self.checkpoints.checkpoint_slots(key)
                        )
                        state.gpu_checkpoint_slots_initialized = True
            self._routing_cpu_array[0, req_index] = src_slot_id
            self._routing_cpu_array[1, req_index] = dst_slot_id

        route_slice = self._routing_indices[:, :num_reqs]
        route_slice.copy_(
            self._routing_cpu[:, :num_reqs],
            non_blocking=self._routing_cpu.is_pinned(),
        )
        batch.mamba_cache_src_indices = route_slice[0]
        batch.mamba_cache_dst_indices = route_slice[1]

    def abort_request(self, req: Req, reason: str = "abort") -> None:
        # Drafter requests are owned by verifier controls, not HTTP clients.
        return None

    def retract_request(self, req: Req) -> None:
        """Restart a retracted drafter from its authoritative committed prefix."""

        if getattr(req, "decoupled_draft_generation", None) is None:
            return
        state = self._state_for_req(req)
        if self.gpu_overlap:
            raise RuntimeError(
                "Decoupled drafter GPU overlap cannot re-prefill a live request; "
                "size KV/Mamba capacity to avoid drafter retraction."
            )
        self._sleeping_requests.pop(state.key, None)
        state.is_sleeping = False
        if self.checkpoints is not None:
            self.checkpoints.release(state.key)
        if len(req.output_ids) > state.committed_len:
            req.output_ids = array("q", req.output_ids[: state.committed_len])
            req.full_untruncated_fill_ids = array("q")
            req._refresh_fill_ids()
        req.to_finish = None
        req.finished_reason = None
        req.finished_len = None
        state.published_len = state.committed_len

    def handle_retracted_requests(
        self,
        retracted_reqs: list[Req],
        reqs_to_abort: list[Req],
    ) -> tuple[list[Req], list[Req]]:
        # A drafter mirror has no client-visible abort path. Truncating its
        # speculative suffix can make the cacheless re-prefill fit again.
        retry_reqs = [*retracted_reqs, *reqs_to_abort]
        for req in retry_reqs:
            self.retract_request(req)
        return retry_reqs, []

    def prepare_pause_retract(self, reqs: list[Req]) -> list[Req]:
        """Fold sleeping mirrors into pause retraction and reset draft state."""

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
            enable_overlap=self.gpu_overlap,
            spec_algorithm=self.scheduler.spec_algorithm,
        )
        device = self.scheduler.device
        req_pool_indices = [int(req.req_pool_idx) for req in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        seq_lens = (
            [1] * len(reqs)
            if self.gpu_overlap
            else [
                len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
                for req in reqs
            ]
        )
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = None if self.gpu_overlap else torch.tensor(
            seq_lens, dtype=torch.int64
        )
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = None if self.gpu_overlap else sum(seq_lens)
        if self.gpu_overlap:
            # Woken rows are reconciled from the authoritative GPU transcript
            # in prepare_forward. Seed FutureMap with executable padding only;
            # Req.output_ids deliberately stops being a decode-token shadow.
            last_tokens = torch.zeros(len(reqs), dtype=torch.int64, device=device)
        else:
            last_tokens = torch.tensor(
                [
                    (
                        int(req.output_ids[-1])
                        if req.output_ids
                        else int(req.origin_input_ids[-1])
                    )
                    for req in reqs
                ],
                dtype=torch.int64,
                device=device,
            )
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
            ahead_len = (
                state.gpu_ahead_len
                if self.gpu_overlap
                else len(req.output_ids) - state.committed_len
            )
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
    def process_pending_controls(self) -> set[DraftRequestGeneration]:
        if self.gpu_overlap:
            # Egress copies only compact cursors. They are scheduling credit,
            # not a host token transcript, and never participate in reconcile.
            if self._drain_gpu_progress():
                self._wake_sleeping_requests()
            if self._requests:
                self._gpu_lifecycle_poll_count += 1
                if (
                    self._gpu_lifecycle_poll_count
                    < self._gpu_lifecycle_poll_interval
                ):
                    return set()
                self._gpu_lifecycle_poll_count = 0
                if self.data_plane.pending_control_count() == 0:
                    return set()
            # VerifyCommit lands directly on the authoritative GPU mirror and is
            # deliberately absent from the CPU inbox in production overlap.
            ready = self.data_plane.collect_lifecycle_controls()
        else:
            ready = self.data_plane.collect_ready_actions(
                self._model_ready_for_commit
            )
        for draft_key in ready.close_keys:
            self._close_request_key(draft_key)
        for message in ready.sync_messages:
            self._open_request(message)
        if self.gpu_overlap:
            if ready.commit_actions or ready.ready_commit_segments:
                raise RuntimeError(
                    "GPU drafter lifecycle polling unexpectedly materialized "
                    "verifier commit tokens on CPU."
                )
            return set()

        rewritten_reqs = []
        commit_outputs = []
        for action in ready.commit_actions:
            req, action_outputs = self._apply_commit_action(action)
            if action.is_rewrite:
                rewritten_reqs.append(req)
            commit_outputs.extend(action_outputs)
        self._refresh_active_batches(rewritten_reqs)
        if commit_outputs:
            self.data_plane.publish_tails(
                DraftTailStreamOutputBatch(outputs=commit_outputs),
            )
        self._wake_sleeping_requests()
        return {
            req.decoupled_draft_generation
            for req in rewritten_reqs
            if getattr(req, "decoupled_draft_generation", None) is not None
        }

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
            state = self._requests.get(
                (int(src_verifier_rank), str(request_id))
            )
            if (
                state is None
                or state.key.request_epoch != int(request_epoch)
            ):
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
            reqs, cpu_outcomes, copy_done, keep_alive = self._inflight_kv_outcomes
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
            generation = getattr(req, "decoupled_draft_generation", None)
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

    def adjust_plan(self, plan: NextBatchPlan) -> NextBatchPlan:
        return plan

    @scheduler_nvtx_method("decoupled_draft.prepare_decode_allocation")
    def prepare_decode_allocation(self, batch: ScheduleBatch) -> None:
        first_is_decoupled = bool(batch.reqs) and getattr(
            batch.reqs[0], "decoupled_draft_generation", None
        ) is not None
        if self.gpu_overlap and any(
            (getattr(req, "decoupled_draft_generation", None) is not None)
            != first_is_decoupled
            for req in batch.reqs[1:]
        ):
            raise RuntimeError(
                "Decoupled drafter GPU overlap cannot mix control-owned and "
                "ordinary decode rows in one batch."
            )
        batch.defer_decode_kv_binding = bool(
            self.gpu_overlap and first_is_decoupled
        )
        if batch.defer_decode_kv_binding:
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None

    def before_decode_retraction(self, batch: ScheduleBatch) -> None:
        if self.gpu_overlap and any(
            getattr(req, "decoupled_draft_generation", None) is not None
            for req in batch.reqs
        ):
            raise RuntimeError(
                "Decoupled drafter GPU overlap cannot retract a live decode "
                "request; size KV/Mamba capacity to avoid drafter retraction."
            )

    @scheduler_nvtx_method("decoupled_draft.prepare_forward")
    def prepare_forward(self, batch: ScheduleBatch) -> None:
        if (
            not self.gpu_overlap
            or batch.decoupled_draft_mirror_seats is None
            or not batch.forward_mode.is_decode()
        ):
            if (
                self.gpu_overlap
                and batch.decoupled_draft_mirror_seats is not None
                and batch.forward_mode.is_extend()
            ):
                # Lifecycle-only fence: OPEN and the checkpoint table must be
                # visible before the first sample initializes device execution
                # state. Steady decode never waits on a landing event.
                self.data_plane.gpu_tail_buffer.wait_for_landing()
            return

        if batch.input_ids is None:
            raise RuntimeError("GPU draft reconcile requires resolved decode inputs.")
        batch.decoupled_draft_captured_state_positions = torch.empty_like(
            batch.seq_lens
        )
        batch.decoupled_draft_old_cache_locs = torch.empty_like(batch.out_cache_loc)
        self.data_plane.gpu_tail_buffer.prepare_decode(
            batch.decoupled_draft_mirror_seats,
            batch.decoupled_draft_request_epochs,
            batch.req_pool_indices,
            batch.out_cache_loc,
            self._gpu_checkpoint_slots,
            batch.req_to_token_pool.req_to_token,
            resolved_input_ids=batch.input_ids,
            resolved_seq_lens=batch.seq_lens,
            resolved_orig_seq_lens=batch.orig_seq_lens,
            mamba_src_indices=batch.mamba_cache_src_indices,
            mamba_dst_indices=batch.mamba_cache_dst_indices,
            captured_state_positions=(
                batch.decoupled_draft_captured_state_positions
            ),
            old_cache_locs=batch.decoupled_draft_old_cache_locs,
            validate_inputs=not self._gpu_decode_binding_validated,
        )

    @scheduler_nvtx_method("decoupled_draft.finish_forward")
    def finish_forward(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> bool:
        if (
            not self.gpu_overlap
            or batch.decoupled_draft_mirror_seats is None
            or not (
                batch.forward_mode.is_decode()
                or (
                    batch.forward_mode.is_extend()
                    and batch.contains_last_prefill_chunk
                )
            )
        ):
            return False
        sampled_tokens = result.next_token_ids
        if not isinstance(sampled_tokens, torch.Tensor):
            raise RuntimeError("GPU draft finish requires device sampled tokens.")

        if batch.forward_mode.is_decode():
            kv_outcomes = torch.empty(
                (sampled_tokens.shape[0], 3),
                dtype=torch.int64,
                device=sampled_tokens.device,
            )
            self.data_plane.gpu_tail_buffer.finish_decode(
                batch.decoupled_draft_mirror_seats,
                batch.decoupled_draft_request_epochs,
                batch.req_pool_indices,
                batch.out_cache_loc,
                sampled_tokens,
                batch.req_to_token_pool.req_to_token,
                resolved_input_tokens=batch.input_ids,
                captured_state_positions=(
                    batch.decoupled_draft_captured_state_positions
                ),
                old_cache_locs=batch.decoupled_draft_old_cache_locs,
                kv_outcomes=kv_outcomes,
                future_output_tokens=(
                    self.scheduler.future_map.output_tokens_buf
                ),
                validate_inputs=not self._gpu_decode_binding_validated,
            )
            # The scheduler owns these fixed-shape tensors. Validate both
            # transactions once per batch identity; changing/reopening any row
            # invalidates the binding before another decode can use it.
            self._gpu_decode_binding_validated = True
            result.decoupled_draft_gpu_managed = True
            result.decoupled_draft_kv_outcomes = kv_outcomes
            result.decoupled_draft_kv_outcomes_ready = (
                self.scheduler.device_module.Event()
            )
            result.decoupled_draft_kv_outcomes_ready.record()
        else:
            mirror_seats = batch.decoupled_draft_mirror_seats
            if batch.chunked_req is not None:
                # A mixed prefill batch includes an unfinished chunk. Its
                # sample is not an output token and must not initialize the
                # authoritative decode state before the final chunk runs.
                mirror_seats = mirror_seats.clone()
                mirror_seats[batch.reqs.index(batch.chunked_req)] = -1
            candidate_committed = torch.empty(
                sampled_tokens.shape,
                dtype=torch.bool,
                device=sampled_tokens.device,
            )
            self.data_plane.gpu_tail_buffer.append_prefill_sample(
                mirror_seats,
                batch.decoupled_draft_request_epochs,
                sampled_tokens,
                accept_out=candidate_committed,
            )
            # CLOSE/reopen may win the GPU row between prefill launch and this
            # epilogue. A rejected old-lifetime sample is discarded by normal
            # lifecycle/result handling; it is not a device invariant failure.
            # Final prefill still uses the ordinary result processor to admit
            # the Req into decode, so only its one-bit lifecycle verdict crosses
            # to CPU. Decode results are fully GPU-managed.
            result.decoupled_draft_candidate_committed = candidate_committed
            self.scheduler.future_map.stash(
                batch.req_pool_indices,
                RelayPayload(bonus_tokens=sampled_tokens.to(torch.int64)),
            )
        return True

    @scheduler_nvtx_method("decoupled_draft.sleep_overrun")
    def sleep_overrun_requests(self, batch: ScheduleBatch) -> ScheduleBatch:
        """Transfer ahead rows before decode allocates this round's KV slots."""

        if batch is None or batch.is_empty():
            return batch
        keep_indices = []
        for index, req in enumerate(batch.reqs):
            if getattr(req, "decoupled_draft_generation", None) is None:
                keep_indices.append(index)
                continue
            state = self._state_for_req(req)
            ahead_len = (
                state.gpu_ahead_len
                if self.gpu_overlap
                else len(req.output_ids) - state.committed_len
            )
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
        # Sleeping rows retain Req/KV/checkpoint ownership outside running_batch.
        # Transport controls alone do not own scheduler resources.
        return bool(self._sleeping_requests)

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
        self.process_pending_controls()
        return True

    def before_process_batch_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> bool:
        if not self.gpu_overlap:
            return False
        gpu_managed_decode = bool(
            getattr(result, "decoupled_draft_gpu_managed", False)
        )
        committed_mask = getattr(
            result, "decoupled_draft_candidate_committed", None
        )
        kv_outcomes = getattr(result, "decoupled_draft_kv_outcomes", None)
        kv_outcomes_drained = bool(
            getattr(result, "decoupled_draft_kv_outcomes_drained", False)
        )
        if not gpu_managed_decode and committed_mask is None:
            return False

        # GPU-managed decode bypasses the generic result processor, but these
        # optional host-side captures retain their ordinary lifecycle.
        routed_experts_output = getattr(result, "routed_experts_output", None)
        if routed_experts_output is not None:
            routed_experts_output.finalize()
            result.routed_experts_output = None
        indexer_topk_output = getattr(result, "indexer_topk_output", None)
        if indexer_topk_output is not None:
            indexer_topk_output.finalize()
            result.indexer_topk_output = None

        if gpu_managed_decode:
            if kv_outcomes is None and not kv_outcomes_drained:
                for req in batch.reqs:
                    generation = getattr(
                        req, "decoupled_draft_generation", None
                    )
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
                            "Live GPU-managed drafter decode is missing KV "
                            "outcomes."
                        )
            elif kv_outcomes is not None:
                self._pending_kv_outcomes.append(
                    (
                        tuple(batch.reqs),
                        kv_outcomes,
                        getattr(
                            result,
                            "decoupled_draft_kv_outcomes_ready",
                            None,
                        ),
                    )
                )
                result.decoupled_draft_kv_outcomes = None
                result.decoupled_draft_kv_outcomes_ready = None
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
                        len(self._pending_kv_outcomes) >= self._kv_outcome_flush_interval
                        and self._inflight_kv_outcomes is not None
                    )
                )
            # Preserve decode telemetry while intentionally bypassing the normal
            # processor's Req.output_ids append and output streaming.
            can_run_cuda_graph = bool(
                getattr(result, "can_run_cuda_graph", False)
            )
            if get_observability().enable_metrics:
                self.scheduler.metrics_collector.increment_decode_cuda_graph_pass(
                    value=can_run_cuda_graph
                )
            metrics_reporter = getattr(self.scheduler, "metrics_reporter", None)
            if metrics_reporter is not None:
                metrics_reporter.num_generated_tokens += len(batch.reqs)
                metrics_reporter.forward_ct_decode = (
                    metrics_reporter.forward_ct_decode + 1
                ) % (1 << 30)
                metrics_reporter.report_decode_stats(
                    can_run_cuda_graph,
                    running_batch=batch,
                    num_correct_drafts=0,
                )
            return True

        # Final prefill still needs the ordinary processor to append its first
        # sampled token and admit the Req into decode. A rejected old-lifetime
        # row is marked retracted and is skipped by that processor.
        if result.copy_done is not None:
            result.copy_done.synchronize()
            result.copy_done = None
        self.process_pending_controls()
        for req in batch.reqs:
            generation = getattr(req, "decoupled_draft_generation", None)
            state = (
                None
                if generation is None
                else self._requests.get(
                    (generation.src_verifier_rank, generation.request_id)
                )
            )
            if state is None or state.req is not req or state.key != generation:
                req.is_retracted = True
        for req, candidate_committed in zip(
            batch.reqs, committed_mask.tolist()
        ):
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
        if self.gpu_overlap:
            prefill_committed = getattr(
                result, "decoupled_draft_candidate_committed", None
            )
            if (
                batch.forward_mode.is_extend()
                and self.checkpoints is not None
                and prefill_committed is not None
            ):
                for req, candidate_committed in zip(
                    batch.reqs, prefill_committed.tolist()
                ):
                    generation = getattr(
                        req, "decoupled_draft_generation", None
                    )
                    state = (
                        None
                        if generation is None
                        else self._requests.get(
                            (generation.src_verifier_rank, generation.request_id)
                        )
                    )
                    if (
                        not bool(candidate_committed)
                        or req.is_retracted
                        or state is None
                        or state.req is not req
                        or state.key != generation
                    ):
                        continue
                    self.checkpoints.commit_after_forward(
                        state.key,
                        position=draft_active_state_position(
                            len(req.origin_input_ids), len(req.output_ids)
                        ),
                    )
            # Native cumulative snapshot egress owns ACK and retained-tail
            # publication. CPU consumes lifecycle plus compact pacing only.
            metrics_reporter = getattr(self.scheduler, "metrics_reporter", None)
            if metrics_reporter is not None:
                metrics_reporter.finish_decoupled_decode_metrics_window()
            batch.mamba_cache_src_indices = None
            batch.mamba_cache_dst_indices = None
            return

        outputs = []
        for req in batch.reqs:
            generation = getattr(req, "decoupled_draft_generation", None)
            if generation is None:
                continue
            state = self._requests.get(
                (generation.src_verifier_rank, generation.request_id)
            )
            if state is None or state.req is not req or state.key != generation:
                # A queued overlap result may outlive CLOSE or a same-id reopen.
                # Lifecycle release owns that old Req; it must not publish into
                # the replacement request's transcript.
                continue
            output_len = len(req.output_ids)
            publish_start = int(state.published_len)
            if not state.committed_len <= publish_start <= output_len:
                raise RuntimeError(
                    "Draft publication cursor is outside the materialized suffix: "
                    f"request_id={state.key.request_id} "
                    f"committed_len={state.committed_len} "
                    f"published_len={publish_start} output_len={output_len}"
                )
            if output_len == publish_start:
                continue
            if req.req_pool_idx is None:
                raise RuntimeError("Draft result lost its request pool slot.")
            if self.checkpoints is not None:
                self.checkpoints.commit_after_forward(
                    state.key,
                    position=draft_active_state_position(
                        len(req.origin_input_ids), len(req.output_ids)
                    ),
                )
            outputs.append(
                DraftTailStreamOutput(
                    src_drafter_rank=self.src_drafter_rank,
                    dst_verifier_rank=state.src_verifier_rank,
                    request_id=state.key.request_id,
                    base_committed_len=state.committed_len,
                    start_token_pos=publish_start,
                    tokens=tuple(
                        int(req.output_ids[output_position])
                        for output_position in range(publish_start, output_len)
                    ),
                )
            )
            state.published_len = output_len
        if outputs:
            self.data_plane.publish_tails(DraftTailStreamOutputBatch(outputs=outputs))
        self.process_pending_controls()
        metrics_reporter = getattr(self.scheduler, "metrics_reporter", None)
        if metrics_reporter is not None:
            metrics_reporter.finish_decoupled_decode_metrics_window()
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
        gpu_seat = -1
        request_epoch = 0
        if self.gpu_overlap:
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
            src_verifier_rank=int(message.src_verifier_rank),
            committed_len=len(req.output_ids),
            published_len=len(req.output_ids),
            gpu_seat=gpu_seat,
        )
        self._requests[table_key] = state
        self.scheduler._add_request_to_queue(req)

    def _close_request(self, message: DraftClose) -> None:
        self._close_request_key(message.draft_key)

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
        if self.gpu_overlap:
            # CLOSE is rare lifecycle work. Order resource reuse after any
            # in-flight forward that may still hold this row/checkpoint ring;
            # steady VerifyCommit never takes this fence.
            self.scheduler.schedule_stream.wait_stream(self.scheduler.forward_stream)
            for queued_batch, queued_result in self.scheduler.result_queue:
                kv_outcomes = getattr(
                    queued_result, "decoupled_draft_kv_outcomes", None
                )
                if kv_outcomes is None:
                    continue
                self._pending_kv_outcomes.append(
                    (
                        tuple(queued_batch.reqs),
                        kv_outcomes,
                        getattr(
                            queued_result,
                            "decoupled_draft_kv_outcomes_ready",
                            None,
                        ),
                    )
                )
                queued_result.decoupled_draft_kv_outcomes = None
                queued_result.decoupled_draft_kv_outcomes_drained = True
                if hasattr(
                    queued_result, "decoupled_draft_kv_outcomes_ready"
                ):
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
        if self.gpu_overlap and req.kv is not None:
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

    def _model_ready_for_commit(
        self,
        draft_key: DraftReqKey,
        expected_output_len: int,
    ) -> bool:
        """Report model readiness without deciding transcript token matching."""

        state = self._requests.get(
            (int(draft_key.src_verifier_rank), str(draft_key.request_id))
        )
        if state is None or state.req.req_pool_idx is None:
            return False
        output_len = len(state.req.output_ids)
        expected_output_len = int(expected_output_len)
        if output_len > expected_output_len:
            raise RuntimeError(
                "Drafter model output is ahead of its published transcript: "
                f"request_id={draft_key.request_id} "
                f"output_len={output_len} "
                f"expected_output_len={expected_output_len}"
            )
        # A retracted/re-prefilling Req may temporarily lag the transcript.
        # Keep its verifier segment queued until all published positions have
        # been rematerialized with matching tokens.
        return output_len == expected_output_len

    @scheduler_nvtx_method("decoupled_draft.apply_commit")
    def _apply_commit_action(
        self,
        action: DraftCommitAction,
    ) -> tuple[Req, list[DraftTailStreamOutput]]:
        if self.gpu_overlap:
            raise RuntimeError(
                "CPU commit actions are disabled for the GPU-authoritative drafter."
            )
        state = self._requests.get(
            (action.draft_key.src_verifier_rank, action.draft_key.request_id)
        )
        if state is None:
            raise RuntimeError(
                "Draft commit action has no live request: "
                f"draft_key={action.draft_key}"
            )
        req = state.req
        output_len = len(req.output_ids)
        if output_len != int(action.expected_output_len):
            raise RuntimeError(
                "Draft commit action model output length changed after planning: "
                f"request_id={action.draft_key.request_id} "
                f"expected_output_len={action.expected_output_len} "
                f"output_len={output_len}"
            )
        if int(action.pre_verify_committed_len) != state.committed_len:
            raise RuntimeError(
                "Draft commit action does not start at the drafter cursor: "
                f"request_id={action.draft_key.request_id} "
                f"expected={state.committed_len} "
                f"actual={action.pre_verify_committed_len}"
            )
        new_committed_len = int(action.new_committed_len)
        if new_committed_len <= state.committed_len:
            raise RuntimeError(
                "Draft commit action must advance the committed cursor: "
                f"request_id={action.draft_key.request_id} "
                f"current={state.committed_len} new={new_committed_len}"
            )
        if req.kv is None:
            raise RuntimeError("Draft commit action arrived before KV allocation.")

        if action.is_rewrite:
            rewrite_position = int(action.rewrite_position)
            rewrite_token = int(action.rewrite_token)
            if new_committed_len != rewrite_position + 1:
                raise RuntimeError(
                    "Draft rewrite action must commit through its replacement token"
                )
            if rewrite_position < state.committed_len or rewrite_position >= output_len:
                raise RuntimeError(
                    "Draft rewrite action targets a non-materialized suffix position: "
                    f"request_id={action.draft_key.request_id} "
                    f"rewrite_position={rewrite_position} "
                    f"committed_len={state.committed_len} output_len={output_len}"
                )
            if int(req.output_ids[rewrite_position]) == rewrite_token:
                raise RuntimeError(
                    "Draft rewrite token unexpectedly matches the model suffix"
                )
            state_restore_position = len(req.origin_input_ids) + rewrite_position
            if self.checkpoints is not None:
                self.checkpoints.rewind_for_rewrite(
                    state.key,
                    position=state_restore_position,
                )
            self._truncate_kv(req, state_restore_position)
            req.output_ids = array(
                "q",
                [*req.output_ids[:rewrite_position], rewrite_token],
            )
            req.full_untruncated_fill_ids = array("q")
            req._refresh_fill_ids()
            req.finished_reason = None
            req.finished_len = None
        else:
            if new_committed_len > output_len:
                raise RuntimeError(
                    "Draft advance action exceeds the materialized output: "
                    f"request_id={action.draft_key.request_id} "
                    f"new_committed_len={new_committed_len} "
                    f"output_len={output_len}"
                )

        state.committed_len = new_committed_len
        # The cumulative echo and any still-valid suffix land in one ordered
        # transport batch, so the next decode only publishes newly generated
        # positions. A rewrite has no retained suffix and remains ACK-only.
        state.published_len = len(req.output_ids)
        if self.checkpoints is not None:
            self.checkpoints.prune(
                state.key,
                # A rewrite can make the replacement token the current unconsumed
                # tail. Keep that state boundary as the next decode source.
                min_position=len(req.origin_input_ids)
                + max(state.committed_len - 1, 0),
            )
        outputs = [
            DraftTailStreamOutput(
                src_drafter_rank=self.src_drafter_rank,
                dst_verifier_rank=state.src_verifier_rank,
                request_id=state.key.request_id,
                base_committed_len=new_committed_len,
                start_token_pos=int(action.echo_position),
                tokens=(int(action.echo_token),),
                is_commit_echo=True,
            )
        ]
        if not action.is_rewrite and output_len > new_committed_len:
            outputs.append(
                DraftTailStreamOutput(
                    src_drafter_rank=self.src_drafter_rank,
                    dst_verifier_rank=state.src_verifier_rank,
                    request_id=state.key.request_id,
                    base_committed_len=new_committed_len,
                    start_token_pos=new_committed_len,
                    tokens=tuple(
                        int(req.output_ids[output_position])
                        for output_position in range(new_committed_len, output_len)
                    ),
                )
            )
        return req, outputs

    def _truncate_kv(self, req: Req, keep_len: int) -> None:
        old_allocated_len = int(req.kv.kv_allocated_len)
        if keep_len < old_allocated_len:
            row = self.scheduler.req_to_token_pool.req_to_token[req.req_pool_idx]
            free_indices = row[keep_len:old_allocated_len].clone()
            row[keep_len:old_allocated_len].zero_()
            self.scheduler.token_to_kv_pool_allocator.free_segment(
                free_indices,
                start_pos=keep_len,
            )
        req.kv_committed_len = min(int(req.kv_committed_len), keep_len)
        req.kv.kv_allocated_len = min(old_allocated_len, keep_len)
        req.cache_protected_len = min(int(req.cache_protected_len), keep_len)

    @scheduler_nvtx_method("decoupled_draft.refresh_rewritten_batches")
    def _refresh_active_batches(self, reqs: list[Req]) -> None:
        if not reqs:
            return
        req_ids = {id(req) for req in reqs}
        seen_batches = set()
        for batch in (
            self.scheduler.running_batch,
            self.scheduler.last_batch,
            self.scheduler.cur_batch_for_debug,
        ):
            if batch is None or id(batch) in seen_batches or batch.is_empty():
                continue
            seen_batches.add(id(batch))
            batch_indices = [
                index for index, req in enumerate(batch.reqs) if id(req) in req_ids
            ]
            if not batch_indices:
                continue
            if batch.seq_lens_cpu is None:
                raise RuntimeError(
                    "Decoupled draft rewrite requires active CPU sequence lengths."
                )
            seq_lens = [
                len(batch.reqs[index].origin_input_ids)
                + len(batch.reqs[index].output_ids)
                - 1
                for index in batch_indices
            ]
            batch.seq_lens_cpu[batch_indices] = torch.tensor(
                seq_lens, dtype=torch.int64
            )
            indices_device = torch.tensor(
                batch_indices, dtype=torch.int64, device=batch.seq_lens.device
            )
            seq_lens_device = torch.tensor(
                seq_lens, dtype=torch.int64, device=batch.seq_lens.device
            )
            batch.seq_lens[indices_device] = seq_lens_device
            batch.orig_seq_lens[indices_device] = seq_lens_device.to(torch.int32)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            batch.input_ids = None

        live_reqs = [req for req in reqs if req.req_pool_idx is not None]
        if live_reqs:
            pool_indices = torch.tensor(
                [int(req.req_pool_idx) for req in live_reqs],
                dtype=torch.int64,
                device=self.scheduler.device,
            )
            tail_tokens = torch.tensor(
                [int(req.output_ids[-1]) for req in live_reqs],
                dtype=torch.int64,
                device=self.scheduler.device,
            )
            self.scheduler.future_map.output_tokens_buf[pool_indices] = tail_tokens

    def _state_for_req(self, req: Req) -> _DraftRequestState:
        key = getattr(req, "decoupled_draft_generation", None)
        if key is None:
            raise RuntimeError(f"Request {req.rid} is not a decoupled draft request.")
        state = self._requests.get((key.src_verifier_rank, key.request_id))
        if state is None or state.req is not req:
            raise RuntimeError(f"Missing decoupled draft state for request {req.rid}.")
        return state

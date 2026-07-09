from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDraftProxyThread,
    CppDraftTailBuffer,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.draft_proxy import DraftProxyThread
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer, DraftTailSnapshot
from sglang.srt.speculative.spec_info import dynamic_verify_enabled
from sglang.srt.speculative.tracer import SpecTraceEvent, trace_speculative
from sglang.srt.utils import broadcast_pyobj, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


class SchedulerDecoupledVerifyMixin:
    """Verifier-side scheduler hooks for decoupled speculation."""

    def create_draft_tail_buffer(self: Scheduler) -> Optional[DraftTailBuffer]:
        if not self.is_verify_entry_rank():
            return None
        tail_buffer_cls = (
            CppDraftTailBuffer
            if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get()
            else DraftTailBuffer
        )
        return tail_buffer_cls(
            verifier_rank=self.get_decoupled_spec_rank(),
            required_tail_len=max(
                0, int(self.server_args.speculative_num_draft_tokens) - 1
            ),
        )

    def start_verify_proxy(self: Scheduler, context) -> None:
        self.draft_proxy_thread = None
        if not self.is_verify_entry_rank():
            return
        if self.draft_tail_buffer is None:
            raise RuntimeError(
                "DraftTailBuffer is required on decoupled_verify entry rank"
            )
        proxy_cls = (
            CppDraftProxyThread
            if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get()
            else DraftProxyThread
        )
        self.draft_proxy_thread = proxy_cls(
            context=context,
            verifier_rank=self.get_decoupled_spec_rank(),
            draft_tail_buffer=self.draft_tail_buffer,
            tracer=self.tracer,
        )
        self.draft_proxy_thread.start()

    def _reset_decoupled_verify_drafter_tables(
        self: Scheduler, num_drafters: int
    ) -> None:
        self.decoupled_verify_drafter_ranks = list(range(num_drafters))
        self.decoupled_verify_drafter_loads = {
            rank: 0 for rank in self.decoupled_verify_drafter_ranks
        }

    def prepare_verify_inputs(self: Scheduler, batch: ScheduleBatch) -> None:
        if not self.is_verify_worker_batch(batch):
            return
        if batch.forward_mode.is_extend():
            self._sync_verify_requests(batch)
        elif batch.forward_mode.is_decode():
            self._prepare_verify_decode_inputs(batch)

    def _activate_decoupled_verify_step_by_batch(
        self: Scheduler, raw_batch_size: int
    ) -> None:
        if not dynamic_verify_enabled(self.server_args):
            return
        if getattr(self.server_args, "disable_cuda_graph", False):
            raise RuntimeError(
                "decoupled verifier dynamic verify length requires full CUDA Graph."
            )
        activate_step_by_batch = getattr(
            self.model_worker, "activate_step_by_batch", None
        )
        if activate_step_by_batch is None:
            raise RuntimeError(
                "decoupled verifier dynamic verify length requires the verifier "
                "worker AdaptiveController runtime-state activator."
            )
        activate_step_by_batch(int(raw_batch_size))

    def _active_decoupled_verify_config(self: Scheduler) -> tuple[int, int]:
        return (
            int(
                getattr(
                    self.model_worker,
                    "speculative_num_steps",
                    self.server_args.speculative_num_steps,
                )
                or 0
            ),
            int(
                getattr(
                    self.model_worker,
                    "speculative_num_draft_tokens",
                    self.server_args.speculative_num_draft_tokens,
                )
                or 0
            ),
        )

    def _maybe_log_decoupled_verify_state_selection(
        self: Scheduler, raw_batch_size: int
    ) -> None:
        if not dynamic_verify_enabled(self.server_args):
            return
        num_speculative_steps, verify_tokens_per_req = (
            self._active_decoupled_verify_config()
        )
        log_key = (
            int(raw_batch_size),
            verify_tokens_per_req,
            num_speculative_steps,
        )
        if log_key == getattr(self, "_last_decoupled_verify_state_log_key", None):
            return
        setattr(self, "_last_decoupled_verify_state_log_key", log_key)

        graph_path = "target verify CUDA Graph"
        zero_step = " zero-step verify" if num_speculative_steps == 0 else ""
        raw_verify_tokens = int(raw_batch_size) * verify_tokens_per_req
        log_info_on_rank0(
            logger,
            "Select decoupled verifier dynamic CUDA Graph state: "
            f"raw_bs={raw_batch_size}, "
            f"captured_bs=, "
            f"num_speculative_steps={num_speculative_steps}, "
            f"verify_tokens_per_req={verify_tokens_per_req}, "
            f"draft_tail_max_len={num_speculative_steps}, "
            f"raw_verify_tokens={raw_verify_tokens}, "
            f"padded_verify_tokens=, "
            f"graph_path={graph_path}{zero_step}",
        )

    def _prepare_verify_decode_inputs(self: Scheduler, batch: ScheduleBatch) -> None:
        self._activate_decoupled_verify_step_by_batch(batch.batch_size())
        self._maybe_log_decoupled_verify_state_selection(batch.batch_size())
        num_speculative_steps, _ = self._active_decoupled_verify_config()

        if dynamic_verify_enabled(self.server_args) and num_speculative_steps == 0:
            self._prepare_zero_step_verify_inputs(batch)
            return

        self._snapshot_verify_inputs(batch)

    @trace_speculative(
        SpecTraceEvent.VERIFIER_SNAPSHOT_TAIL_BATCH,
        inject_trace_enabled="trace_enabled",
    )
    def _prepare_zero_step_verify_inputs(
        self: Scheduler,
        batch: ScheduleBatch,
        *,
        trace_enabled: bool = False,
    ) -> dict | None:
        num_speculative_steps, verify_tokens_per_req = (
            self._active_decoupled_verify_config()
        )
        live_reqs = []
        for req in batch.reqs:
            if req.is_retracted or req.finished():
                continue
            live_reqs.append(req)
            setattr(req, "draft_buffer", None)
            if trace_enabled:
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
            setattr(
                req,
                "_decoupled_verify_pre_committed_len",
                len(req.output_ids),
            )
            setattr(
                req,
                "_decoupled_verify_num_speculative_steps",
                num_speculative_steps,
            )
            setattr(
                req,
                "_decoupled_verify_tokens_per_req",
                verify_tokens_per_req,
            )
        if not live_reqs:
            return None
        if not trace_enabled:
            return None

        raw_verify_tokens = len(live_reqs) * verify_tokens_per_req

        return {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(live_reqs),
            "dynamic_verify_length": dynamic_verify_enabled(self.server_args),
            "raw_batch_size": len(live_reqs),
            "captured_batch_size": "",
            "num_speculative_steps": num_speculative_steps,
            "verify_tokens_per_req": verify_tokens_per_req,
            "raw_verify_tokens": raw_verify_tokens,
            "padded_verify_tokens": "",
            "rids": [req.rid for req in live_reqs],
            "valid_tail_lens_by_req": [0 for _ in live_reqs],
            "raw_tail_lens_by_req": [0 for _ in live_reqs],
            "committed_lens_by_req": [len(req.output_ids) for req in live_reqs],
            "output_lens_by_req": [len(req.output_ids) for req in live_reqs],
            "num_stale_snapshots": 0,
        }

    def validate_verify_outputs(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        accept_lens = result.num_correct_drafts_per_req_cpu
        # Compatibility with older decoupled verifier results during rolling
        # migrations from v0.5.10-dev.
        if accept_lens is None:
            accept_lens = getattr(result, "num_accepted_drafts_per_req_cpu", None)
        if accept_lens is None:
            accept_lens = getattr(result, "accept_length_per_req_cpu", None)
        if accept_lens is None:
            raise RuntimeError("Decoupled verify result is missing accept lengths.")
        if len(accept_lens) != len(batch.reqs):
            raise RuntimeError(
                "Decoupled verify accept length count does not match batch size: "
                f"accept_lens={len(accept_lens)} batch_size={len(batch.reqs)}"
            )
        if result.next_token_ids is None:
            raise RuntimeError("Decoupled verify result is missing verified token ids.")

        verified_ids_obj = result.next_token_ids
        if isinstance(verified_ids_obj, torch.Tensor):
            verified_ids = verified_ids_obj.tolist()
        else:
            verified_ids = []
            for item in verified_ids_obj:
                if isinstance(item, torch.Tensor):
                    item = item.tolist()
                if isinstance(item, list):
                    verified_ids.extend(int(token_id) for token_id in item)
                else:
                    verified_ids.append(int(item))

        offset = 0
        valid_draft_metric_updates: list[tuple[Req, int, int]] = []
        for req, accept_len in zip(batch.reqs, accept_lens):
            accept_len = int(accept_len)
            segment_len = accept_len + 1
            segment = verified_ids[offset : offset + segment_len]
            if len(segment) != segment_len:
                raise RuntimeError(
                    "Decoupled verify returned too few verified ids: "
                    f"request_id={req.rid} accept_len={accept_len} "
                    f"remaining_verified_ids={len(verified_ids) - offset}"
                )
            offset += segment_len

            pre_committed_len = getattr(
                req, "_decoupled_verify_pre_committed_len", None
            )
            if pre_committed_len is None:
                pre_committed_len = len(req.output_ids) - segment_len
            pre_committed_len = int(pre_committed_len)
            if pre_committed_len < 0:
                raise RuntimeError(
                    "Decoupled verify output is shorter than the verified segment: "
                    f"request_id={req.rid} output_len={len(req.output_ids)} "
                    f"segment_len={segment_len}"
                )

            output_segment = req.output_ids[
                pre_committed_len : pre_committed_len + segment_len
            ]
            if output_segment != segment:
                raise RuntimeError(
                    "Decoupled verify result does not match committed output ids: "
                    f"request_id={req.rid} pre_committed_len={pre_committed_len} "
                    f"accept_len={accept_len} verified_segment={segment} "
                    f"output_segment={output_segment}"
                )

            draft_buffer = list(getattr(req, "draft_buffer", []) or [])
            accepted_draft_tokens = segment[:accept_len]
            expected_draft_tokens = draft_buffer[:accept_len]
            if accepted_draft_tokens != expected_draft_tokens:
                raise RuntimeError(
                    "Decoupled verify accepted tokens outside the draft snapshot: "
                    f"request_id={req.rid} accept_len={accept_len} "
                    f"accepted_draft_tokens={accepted_draft_tokens} "
                    f"draft_snapshot_prefix={expected_draft_tokens}"
                )
            valid_draft_metric_updates.append((req, len(draft_buffer), accept_len))

        if offset != len(verified_ids):
            raise RuntimeError(
                "Decoupled verify returned extra verified ids: "
                f"consumed={offset} total={len(verified_ids)}"
            )

        for (
            req,
            valid_draft_tokens,
            valid_accepted_tokens,
        ) in valid_draft_metric_updates:
            spec_steps = getattr(req, "_decoupled_verify_num_speculative_steps", None)
            if spec_steps is None:
                spec_steps = int(self.server_args.speculative_num_steps or 0)
                if spec_steps <= 0:
                    spec_steps = max(
                        0, int(self.server_args.speculative_num_draft_tokens or 1) - 1
                    )
            spec_steps = int(spec_steps)
            valid_draft_tokens = min(valid_draft_tokens, spec_steps)
            valid_accepted_tokens = min(valid_accepted_tokens, valid_draft_tokens)
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

    def assign_drafter_rank(self, request_id: str) -> int:
        """Assign a verifier request to the currently least-loaded drafter rank."""
        drafter_rank = self.decoupled_verify_req_to_drafter_rank.get(request_id)
        if drafter_rank is not None:
            return drafter_rank

        if not self.decoupled_verify_drafter_ranks:
            raise RuntimeError(
                "Decoupled verify drafter ranks are not initialized on entry rank"
            )
        drafter_rank = min(
            self.decoupled_verify_drafter_ranks,
            key=lambda rank: (
                self.decoupled_verify_drafter_loads.get(rank, 0),
                (rank - self.get_decoupled_spec_rank())
                % len(self.decoupled_verify_drafter_ranks),
            ),
        )
        self.decoupled_verify_req_to_drafter_rank[request_id] = drafter_rank
        self.decoupled_verify_drafter_loads[drafter_rank] = (
            self.decoupled_verify_drafter_loads.get(drafter_rank, 0) + 1
        )
        return drafter_rank

    def get_drafter_rank(self, request_id: str) -> int:
        """Return the drafter rank already assigned to a verifier request."""
        drafter_rank = self.decoupled_verify_req_to_drafter_rank.get(request_id)
        if drafter_rank is None:
            raise RuntimeError(
                "Missing decoupled verify drafter assignment for "
                f"request_id={request_id}"
            )
        return drafter_rank

    def release_drafter_rank(self, request_id: str) -> None:
        """Release one request's drafter assignment after close/abort."""
        drafter_rank = self.decoupled_verify_req_to_drafter_rank.pop(request_id, None)
        if drafter_rank is None:
            return
        self.decoupled_verify_drafter_loads[drafter_rank] = max(
            0,
            self.decoupled_verify_drafter_loads.get(drafter_rank, 0) - 1,
        )

    def _submit_verify_control_batch(self, batch: DraftControlBatch) -> None:
        """
        Submit one verifier-to-drafter control batch.

        Called after scheduler build DraftSync, VerifyCommit, or
        DraftClose messages. The entry rank forwards the batch to
        DraftProxyThread, which both updates the entry-rank DraftTailBuffer
        locally and sends the batch asynchronously. Non-entry ranks do not own
        draft transport state and return without side effects.

        Args:
            batch: A batch of control messages for one drafter rank.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return

        if self.draft_proxy_thread is None:
            raise RuntimeError(
                "Draft proxy thread is not initialized on decoupled_verify entry rank"
            )
        self.draft_proxy_thread.submit_control_batch(batch)

    def _send_verify_control_batches(
        self,
        *,
        sync_messages: list[DraftSync] | None = None,
        verify_commit_messages: list[VerifyCommit] | None = None,
        close_messages: list[DraftClose] | None = None,
    ) -> None:
        """
        Group verifier control messages by destination drafter and submit them.

        Used by decoupled verify lifecycle hooks verify input/update processing
        and by abort handling. This keeps verifier-to-drafter communication
        batch-based: each destination drafter rank receives at most one
        DraftControlBatch from this call.

        Args:
            sync_messages: Optional DraftSync messages created when verifier
                first introduces requests to the drafter.
            verify_commit_messages: Optional VerifyCommit messages created
                after verifier accepts tokens for live requests.
            close_messages: Optional DraftClose messages created when verifier
                finishes, retracts, or aborts requests.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return

        batches: dict[int, DraftControlBatch] = {}

        def get_batch(dst_drafter_rank: int) -> DraftControlBatch:
            dst_drafter_rank = int(dst_drafter_rank)
            batch = batches.get(dst_drafter_rank)
            if batch is None:
                batch = DraftControlBatch(dst_drafter_rank=dst_drafter_rank)
                batches[dst_drafter_rank] = batch
            return batch

        for message in sync_messages or []:
            get_batch(message.dst_drafter_rank).sync_messages.append(message)
        for message in verify_commit_messages or []:
            get_batch(message.dst_drafter_rank).verify_commit_messages.append(message)
        for message in close_messages or []:
            get_batch(message.dst_drafter_rank).close_messages.append(message)

        for batch in batches.values():
            self._submit_verify_control_batch(batch)

    def _send_verify_control_rows(
        self,
        *,
        sync_rows: list[tuple] | None = None,
        commit_rows: list[tuple] | None = None,
        close_rows: list[tuple] | None = None,
    ) -> None:
        if not self.is_verify_entry_rank():
            return

        if self.draft_proxy_thread is None:
            raise RuntimeError(
                "Draft proxy thread is not initialized on decoupled_verify entry rank"
            )

        batches: dict[int, dict[str, list[tuple]]] = {}

        def get_batch(dst_drafter_rank: int) -> dict[str, list[tuple]]:
            dst_drafter_rank = int(dst_drafter_rank)
            batch = batches.get(dst_drafter_rank)
            if batch is None:
                batch = {"sync": [], "commit": [], "close": []}
                batches[dst_drafter_rank] = batch
            return batch

        for row in sync_rows or []:
            get_batch(int(row[2]))["sync"].append(row)
        for row in commit_rows or []:
            get_batch(int(row[2]))["commit"].append(row)
        for row in close_rows or []:
            get_batch(int(row[2]))["close"].append(row)

        submit_rows = getattr(self.draft_proxy_thread, "submit_control_rows", None)
        if submit_rows is None:
            raise RuntimeError("C++ decoupled verify proxy does not expose row API")
        for dst_drafter_rank, rows in batches.items():
            submit_rows(
                dst_drafter_rank,
                rows["sync"],
                rows["commit"],
                rows["close"],
            )

    def _broadcast_verify_snapshots(
        self, local_snapshots: list[DraftTailSnapshot] | None
    ) -> list[DraftTailSnapshot]:
        """
        Broadcast per-forward draft tail snapshots from the verifier entry rank.

        Used during decoupled verify batch preparation. The entry rank reads
        currently available draft tail tokens from its DraftTailBuffer, then
        this helper makes the same immutable per-forward snapshot visible to all
        verifier ranks that participate in the forward pass.

        Args:
            local_snapshots: Draft tail snapshots collected on the entry rank.
                This value is ignored on non-entry ranks.

        Returns:
            The broadcast list of DraftTailSnapshot objects.
        """
        source_payload = (
            list(local_snapshots or []) if self.is_verify_entry_rank() else []
        )
        if getattr(self.server_args, "enable_dp_attention", False):
            synced_snapshots = source_payload
            if self.ps.attn_tp_size != 1:
                synced_snapshots = broadcast_pyobj(
                    synced_snapshots,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.ps.attn_cp_size != 1:
                synced_snapshots = broadcast_pyobj(
                    synced_snapshots,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )
            return list(synced_snapshots or [])

        if self.ps.tp_size != 1:
            source_payload = broadcast_pyobj(
                source_payload,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return list(source_payload or [])

    def _bind_verify_snapshots(
        self,
        target_reqs: list[Req],
        snapshots: list[DraftTailSnapshot],
        *,
        collect_trace_stats: bool = False,
    ) -> int:
        """
        Bind one broadcast draft tail snapshot set to the local verifier batch.

        Called after the entry-rank snapshot
        has been broadcast. All ranks, including the entry rank, use this same
        snapshot set for req.draft_buffer so concurrent proxy updates cannot
        affect the current verifier forward pass.

        Args:
            target_reqs: Live verifier requests in the local batch.
            snapshots: Broadcast per-request draft tail snapshots.

        Returns:
            The number of snapshots skipped because their confirmed prefix
            lags behind the verifier request.
        """
        snapshot_by_rid: dict[str, DraftTailSnapshot] = {}
        for snapshot in snapshots:
            if snapshot.request_id in snapshot_by_rid:
                raise RuntimeError(
                    "Duplicate decoupled verify draft tail snapshot: "
                    f"request_id={snapshot.request_id}"
                )
            snapshot_by_rid[snapshot.request_id] = snapshot

        for req in target_reqs:
            snapshot = snapshot_by_rid.get(req.rid)
            if snapshot is None:
                setattr(req, "draft_buffer", None)
                if collect_trace_stats:
                    setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
                continue

            committed_len = int(snapshot.committed_len)
            if committed_len < len(req.output_ids):
                # the drafter has not caught up with the verifier req's committed output prefix
                setattr(req, "draft_buffer", None)
                if collect_trace_stats:
                    setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
                continue
            if committed_len > len(req.output_ids):
                raise RuntimeError(
                    "Decoupled verify draft tail snapshot is out of sync with "
                    "the verifier request: "
                    f"request_id={req.rid} snapshot_committed_len={committed_len} "
                    f"request_output_len={len(req.output_ids)}"
                )
            setattr(req, "draft_buffer", list(snapshot.tail_tokens))
            if collect_trace_stats:
                setattr(
                    req,
                    "_decoupled_verify_snapshot_raw_tail_tokens",
                    list(snapshot.raw_tail_tokens),
                )
        if not collect_trace_stats:
            return 0
        return sum(
            1
            for req in target_reqs
            if snapshot_by_rid.get(req.rid) is not None
            and int(snapshot_by_rid[req.rid].committed_len) < len(req.output_ids)
        )

    @trace_speculative(
        SpecTraceEvent.VERIFIER_BUILD_SYNC_BATCH,
        inject_trace_enabled="trace_enabled",
    )
    def _sync_verify_requests(
        self,
        batch: ScheduleBatch,
        *,
        trace_enabled: bool = False,
    ) -> dict | None:
        """
        Send DraftSync messages before verifier prefill/extend processing.

        Called from process_batch_result setup for decoupled verify batches
        before an extend batch is run. Only the entry rank owns DraftTailBuffer
        and draft transport; for each live, unsynced request, it records the
        verifier's current prompt/output prefix and sends one DraftSync to the
        corresponding drafter so draft generation can start from the committed
        prefix.

        Args:
            batch: The ScheduleBatch about to be processed by the verifier.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return None

        if not batch.forward_mode.is_extend() or batch.is_dllm():
            return None

        draft_tail_buffer = self.draft_tail_buffer
        assert draft_tail_buffer is not None

        if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get():
            sync_rows: list[tuple] = []
            for req in batch.reqs:
                if not req.is_retracted and not req.finished():
                    setattr(
                        req,
                        "_decoupled_verify_pre_committed_len",
                        len(req.output_ids),
                    )
                if (
                    getattr(req, "inflight_middle_chunks", getattr(req, "is_chunked", 0))
                    > 0
                    or req.is_retracted
                    or req.finished()
                ):
                    continue
                if draft_tail_buffer.has_request(req.rid):
                    continue
                dst_drafter_rank = self.assign_drafter_rank(req.rid)
                sync_rows.append(
                    (
                        req.rid,
                        self.get_decoupled_spec_rank(),
                        dst_drafter_rank,
                        [int(token_id) for token_id in req.origin_input_ids],
                        [int(token_id) for token_id in req.output_ids],
                    )
                )
                setattr(req, "draft_buffer", None)
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
            trace_payload = {
                "forward_mode": str(batch.forward_mode),
                "batch_size": len(batch.reqs),
                "rids": [row[0] for row in sync_rows],
                "committed_lens_by_req": [len(row[4]) for row in sync_rows],
                "output_lens_by_req": [len(row[4]) for row in sync_rows],
                "dst_drafter_ranks": [int(row[2]) for row in sync_rows],
            }
            self._send_verify_control_rows(sync_rows=sync_rows)
            return trace_payload

        sync_messages: list[DraftSync] = []
        for req in batch.reqs:
            if not req.is_retracted and not req.finished():
                setattr(
                    req,
                    "_decoupled_verify_pre_committed_len",
                    len(req.output_ids),
                )
            if (
                getattr(req, "inflight_middle_chunks", getattr(req, "is_chunked", 0))
                > 0
                or req.is_retracted
                or req.finished()
            ):
                continue
            if draft_tail_buffer.has_request(req.rid):
                continue
            sync_messages.append(
                DraftSync(
                    request_id=req.rid,
                    src_verifier_rank=self.get_decoupled_spec_rank(),
                    dst_drafter_rank=self.assign_drafter_rank(req.rid),
                    prompt_token_ids=list(req.origin_input_ids),
                    committed_output_ids=list(req.output_ids),
                )
            )
            setattr(req, "draft_buffer", None)
            if trace_enabled:
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
        self._send_verify_control_batches(sync_messages=sync_messages)
        if not trace_enabled:
            return None
        return {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(batch.reqs),
            "rids": [message.request_id for message in sync_messages],
            "committed_lens_by_req": [
                len(message.committed_output_ids) for message in sync_messages
            ],
            "output_lens_by_req": [
                len(message.committed_output_ids) for message in sync_messages
            ],
            "dst_drafter_ranks": [
                int(message.dst_drafter_rank) for message in sync_messages
            ],
        }

    @trace_speculative(
        SpecTraceEvent.VERIFIER_SNAPSHOT_TAIL_BATCH,
        inject_trace_enabled="trace_enabled",
    )
    def _snapshot_verify_inputs(
        self,
        batch: ScheduleBatch,
        *,
        trace_enabled: bool = False,
    ) -> dict | None:
        """
        Collect currently available draft tails, and bind them to a verifier request batch.

        Called immediately before a decoupled verify forward pass is prepared.
        The default path is non-blocking: the verifier entry rank snapshots the
        draft tail tokens already received by DraftProxyThread, broadcasts that
        stable per-forward snapshot to peer TP ranks, and all ranks bind
        req.draft_buffer from the broadcast snapshot.

        Args:
            batch: The ScheduleBatch that will run verifier extend/decode.

        Returns:
            None.
        """
        live_reqs = []
        for req in batch.reqs:
            if req.is_retracted or req.finished():
                continue
            live_reqs.append(req)
            setattr(req, "draft_buffer", None)
            if trace_enabled:
                setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
            setattr(
                req,
                "_decoupled_verify_pre_committed_len",
                len(req.output_ids),
            )
        target_reqs = live_reqs
        if not target_reqs:
            return None

        is_dynamic = dynamic_verify_enabled(self.server_args)
        active_spec_steps, active_verify_tokens = (
            self._active_decoupled_verify_config()
        )
        snapshot_tail_cap = active_spec_steps if is_dynamic else None
        verify_tokens_per_req = active_verify_tokens if is_dynamic else None
        for req in target_reqs:
            setattr(
                req,
                "_decoupled_verify_num_speculative_steps",
                snapshot_tail_cap,
            )
            setattr(
                req,
                "_decoupled_verify_tokens_per_req",
                verify_tokens_per_req,
            )
        local_snapshots: list[DraftTailSnapshot] = []
        if self.is_verify_entry_rank():
            draft_tail_buffer = self.draft_tail_buffer
            assert draft_tail_buffer is not None
            allow_partial = envs.SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL.get()
            local_snapshots = draft_tail_buffer.get_draft_snapshots(
                target_reqs,
                allow_partial=allow_partial,
                include_raw_tail_tokens=trace_enabled,
                max_tail_len=snapshot_tail_cap,
            )

        synced_snapshots = self._broadcast_verify_snapshots(local_snapshots)
        num_stale_snapshots = self._bind_verify_snapshots(
            target_reqs,
            synced_snapshots,
            collect_trace_stats=trace_enabled,
        )
        if not trace_enabled:
            return None
        snapshot_by_rid = {
            snapshot.request_id: snapshot for snapshot in synced_snapshots
        }
        static_spec_steps = int(self.server_args.speculative_num_steps or 0)
        static_verify_tokens = int(self.server_args.speculative_num_draft_tokens or 0)
        active_spec_steps = (
            snapshot_tail_cap if snapshot_tail_cap is not None else static_spec_steps
        )
        active_verify_tokens = (
            verify_tokens_per_req
            if verify_tokens_per_req is not None
            else static_verify_tokens
        )
        return {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(target_reqs),
            "dynamic_verify_length": is_dynamic,
            "raw_batch_size": len(target_reqs) if is_dynamic else "",
            "captured_batch_size": "",
            "num_speculative_steps": active_spec_steps,
            "verify_tokens_per_req": active_verify_tokens,
            "raw_verify_tokens": len(target_reqs) * active_verify_tokens,
            "padded_verify_tokens": "",
            "rids": [req.rid for req in target_reqs],
            "valid_tail_lens_by_req": [
                len(getattr(req, "draft_buffer", None) or []) for req in target_reqs
            ],
            "raw_tail_lens_by_req": [
                int(getattr(snapshot_by_rid.get(req.rid), "raw_tail_len", 0))
                for req in target_reqs
            ],
            "committed_lens_by_req": [
                int(getattr(snapshot_by_rid.get(req.rid), "committed_len", 0))
                for req in target_reqs
            ],
            "output_lens_by_req": [len(req.output_ids) for req in target_reqs],
            "num_stale_snapshots": num_stale_snapshots,
        }

    @trace_speculative(
        SpecTraceEvent.VERIFIER_BUILD_UPDATE_BATCH,
        inject_trace_enabled="trace_enabled",
    )
    def submit_verify_updates(
        self,
        batch: ScheduleBatch,
        *,
        trace_enabled: bool = False,
    ) -> dict | None:
        """
        Send verifier commit or close messages after batch result processing.

        Called after verifier extend/decode results have updated request output
        ids and finish state. Only the entry rank owns DraftTailBuffer and emits
        control messages. Live synced requests emit VerifyCommit with the latest
        committed output segment. Finished or retracted synced requests
        emit DraftClose so the drafter can release its request state.

        Args:
            batch: The ScheduleBatch whose verifier results were just applied.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return None

        if batch.forward_mode.is_extend() and batch.is_dllm():
            return None

        if not (batch.forward_mode.is_extend() or batch.forward_mode.is_decode()):
            return None

        draft_tail_buffer = self.draft_tail_buffer
        assert draft_tail_buffer is not None

        if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get():
            commit_rows: list[tuple] = []
            close_rows: list[tuple] = []
            commit_pre_committed_lens: list[int] = []
            commit_draft_buffer_lens: list[int] = []
            commit_segment_lens: list[int] = []
            commit_last_token_ids: list[int] = []
            commit_committed_lens: list[int] = []
            commit_output_lens: list[int] = []
            close_output_lens: list[int] = []
            for req in batch.reqs:
                has_request = draft_tail_buffer.has_request(req.rid)

                if req.is_retracted or req.finished():
                    if has_request:
                        dst_drafter_rank = self.get_drafter_rank(req.rid)
                        close_rows.append(
                            (
                                req.rid,
                                self.get_decoupled_spec_rank(),
                                dst_drafter_rank,
                                "abort" if req.is_retracted else "finished",
                            )
                        )
                        close_output_lens.append(len(req.output_ids))
                        self.release_drafter_rank(req.rid)
                    setattr(req, "draft_buffer", None)
                    setattr(req, "_decoupled_verify_snapshot_raw_tail_tokens", [])
                    continue

                if not has_request:
                    continue
                if not req.output_ids:
                    continue

                pre_verify_committed_len = getattr(
                    req,
                    "_decoupled_verify_pre_committed_len",
                    None,
                )
                if pre_verify_committed_len is None:
                    pre_verify_committed_len = draft_tail_buffer.get_committed_len(req.rid)
                if pre_verify_committed_len is None:
                    continue
                pre_verify_committed_len = int(pre_verify_committed_len)
                if pre_verify_committed_len > len(req.output_ids):
                    raise RuntimeError(
                        "Verifier VerifyCommit pre-commit prefix is beyond the "
                        "current output ids: "
                        f"request_id={req.rid} "
                        f"pre_verify_committed_len={pre_verify_committed_len} "
                        f"output_len={len(req.output_ids)}"
                    )
                if pre_verify_committed_len == len(req.output_ids):
                    if hasattr(req, "_decoupled_verify_pre_committed_len"):
                        delattr(req, "_decoupled_verify_pre_committed_len")
                    if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                        delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
                    continue

                draft_buffer = list(getattr(req, "draft_buffer", None) or [])
                committed_token_ids = [
                    int(token_id)
                    for token_id in req.output_ids[pre_verify_committed_len:]
                ]
                dst_drafter_rank = self.get_drafter_rank(req.rid)
                commit_rows.append(
                    (
                        req.rid,
                        self.get_decoupled_spec_rank(),
                        dst_drafter_rank,
                        pre_verify_committed_len,
                        committed_token_ids,
                    )
                )
                commit_pre_committed_lens.append(pre_verify_committed_len)
                commit_draft_buffer_lens.append(len(draft_buffer))
                commit_segment_lens.append(len(committed_token_ids))
                commit_last_token_ids.append(int(committed_token_ids[-1]))
                commit_committed_lens.append(
                    pre_verify_committed_len + len(committed_token_ids)
                )
                commit_output_lens.append(len(req.output_ids))
                if hasattr(req, "_decoupled_verify_pre_committed_len"):
                    delattr(req, "_decoupled_verify_pre_committed_len")
                if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                    delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")

            trace_payload = {
                "forward_mode": str(batch.forward_mode),
                "batch_size": len(batch.reqs),
                "commit_rids": [row[0] for row in commit_rows],
                "close_rids": [row[0] for row in close_rows],
                "num_commit": len(commit_rows),
                "num_close": len(close_rows),
                "pre_committed_lens_by_req": commit_pre_committed_lens,
                "draft_buffer_lens_by_req": commit_draft_buffer_lens,
                "committed_segment_lens_by_req": commit_segment_lens,
                "last_committed_token_ids_by_req": commit_last_token_ids,
                "committed_lens_by_req": commit_committed_lens,
                "commit_output_lens_by_req": commit_output_lens,
                "commit_dst_drafter_ranks": [int(row[2]) for row in commit_rows],
                "close_output_lens_by_req": close_output_lens,
                "close_dst_drafter_ranks": [int(row[2]) for row in close_rows],
            }
            self._send_verify_control_rows(
                commit_rows=commit_rows,
                close_rows=close_rows,
            )
            return trace_payload

        verify_commit_messages: list[VerifyCommit] = []
        close_messages: list[DraftClose] = []
        commit_pre_committed_lens: list[int] = []
        commit_draft_buffer_lens: list[int] = []
        commit_segment_lens: list[int] = []
        commit_last_token_ids: list[int] = []
        commit_committed_lens: list[int] = []
        commit_output_lens: list[int] = []
        close_output_lens: list[int] = []
        for req in batch.reqs:
            has_request = draft_tail_buffer.has_request(req.rid)

            if req.is_retracted or req.finished():
                if has_request:
                    dst_drafter_rank = self.get_drafter_rank(req.rid)
                    close_messages.append(
                        DraftClose(
                            request_id=req.rid,
                            src_verifier_rank=self.get_decoupled_spec_rank(),
                            dst_drafter_rank=dst_drafter_rank,
                            reason="abort" if req.is_retracted else "finished",
                        )
                    )
                    if trace_enabled:
                        close_output_lens.append(len(req.output_ids))
                    self.release_drafter_rank(req.rid)
                setattr(req, "draft_buffer", None)
                if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                    delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
                continue

            if not has_request:
                continue
            if not req.output_ids:
                continue

            pre_verify_committed_len = getattr(
                req,
                "_decoupled_verify_pre_committed_len",
                None,
            )
            if pre_verify_committed_len is None:
                pre_verify_committed_len = draft_tail_buffer.get_committed_len(req.rid)
            if pre_verify_committed_len is None:
                continue
            pre_verify_committed_len = int(pre_verify_committed_len)
            if pre_verify_committed_len > len(req.output_ids):
                raise RuntimeError(
                    "Verifier VerifyCommit pre-commit prefix is beyond the "
                    "current output ids: "
                    f"request_id={req.rid} "
                    f"pre_verify_committed_len={pre_verify_committed_len} "
                    f"output_len={len(req.output_ids)}"
                )
            if pre_verify_committed_len == len(req.output_ids):
                # no tokens are generated during this forward(e.g. chunked prefill)
                if hasattr(req, "_decoupled_verify_pre_committed_len"):
                    delattr(req, "_decoupled_verify_pre_committed_len")
                if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                    delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
                continue

            draft_buffer = list(getattr(req, "draft_buffer", None) or [])
            committed_token_ids = [
                int(token_id)
                for token_id in req.output_ids[pre_verify_committed_len:]
            ]

            verify_commit_messages.append(
                VerifyCommit(
                    request_id=req.rid,
                    src_verifier_rank=self.get_decoupled_spec_rank(),
                    dst_drafter_rank=self.get_drafter_rank(req.rid),
                    pre_verify_committed_len=pre_verify_committed_len,
                    committed_token_ids=committed_token_ids,
                )
            )
            if trace_enabled:
                commit_pre_committed_lens.append(pre_verify_committed_len)
                commit_draft_buffer_lens.append(len(draft_buffer))
                commit_segment_lens.append(len(committed_token_ids))
                commit_last_token_ids.append(int(committed_token_ids[-1]))
                commit_committed_lens.append(
                    pre_verify_committed_len + len(committed_token_ids)
                )
                commit_output_lens.append(len(req.output_ids))
            if hasattr(req, "_decoupled_verify_pre_committed_len"):
                delattr(req, "_decoupled_verify_pre_committed_len")
            if hasattr(req, "_decoupled_verify_snapshot_raw_tail_tokens"):
                delattr(req, "_decoupled_verify_snapshot_raw_tail_tokens")
        self._send_verify_control_batches(
            verify_commit_messages=verify_commit_messages,
            close_messages=close_messages,
        )
        if not trace_enabled:
            return None
        return {
            "forward_mode": str(batch.forward_mode),
            "batch_size": len(batch.reqs),
            "commit_rids": [
                message.request_id for message in verify_commit_messages
            ],
            "close_rids": [message.request_id for message in close_messages],
            "num_commit": len(verify_commit_messages),
            "num_close": len(close_messages),
            "pre_committed_lens_by_req": commit_pre_committed_lens,
            "draft_buffer_lens_by_req": commit_draft_buffer_lens,
            "committed_segment_lens_by_req": commit_segment_lens,
            "last_committed_token_ids_by_req": commit_last_token_ids,
            "committed_lens_by_req": commit_committed_lens,
            "commit_output_lens_by_req": commit_output_lens,
            "commit_dst_drafter_ranks": [
                int(message.dst_drafter_rank) for message in verify_commit_messages
            ],
            "close_output_lens_by_req": close_output_lens,
            "close_dst_drafter_ranks": [
                int(message.dst_drafter_rank) for message in close_messages
            ],
        }

    def abort_verify_request(self, request_id: str) -> None:
        """
        Close a drafter-side request when the verifier aborts it.

        Called from scheduler abort paths. Only the entry rank owns
        DraftTailBuffer; if the request has decoupled verify state there, this
        sends a DraftClose with ABORT to the recorded drafter rank. Requests
        that were never synced have no drafter state and do not need a close
        message.

        Args:
            request_id: Verifier request id to abort on the drafter side.

        Returns:
            None.
        """
        if not self.is_verify_entry_rank():
            return
        draft_tail_buffer = self.draft_tail_buffer
        assert draft_tail_buffer is not None
        if draft_tail_buffer.has_request(request_id):
            dst_drafter_rank = self.get_drafter_rank(request_id)
            if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get():
                self._send_verify_control_rows(
                    close_rows=[
                        (
                            request_id,
                            self.get_decoupled_spec_rank(),
                            dst_drafter_rank,
                            "abort",
                        )
                    ]
                )
                self.release_drafter_rank(request_id)
                return
            self._send_verify_control_batches(
                close_messages=[
                    DraftClose(
                        request_id=request_id,
                        src_verifier_rank=self.get_decoupled_spec_rank(),
                        dst_drafter_rank=dst_drafter_rank,
                        reason="abort",
                    )
                ]
            )
            self.release_drafter_rank(request_id)

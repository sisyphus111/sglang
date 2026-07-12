from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftSync,
    VerifyCommit,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    get_decoupled_spec_transport,
)
from sglang.srt.speculative.decoupled_verify_state import (
    prepare_decoupled_verify_snapshot,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer, DraftTailSnapshot
from sglang.srt.speculative.spec_info import dynamic_verify_enabled
from sglang.srt.utils import broadcast_pyobj, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerDecoupledVerifyMixin:
    """Verifier-side scheduler hooks for decoupled speculation."""

    def create_draft_tail_buffer(self: Scheduler) -> Optional[DraftTailBuffer]:
        if not self.is_verify_entry_rank():
            return None
        transport = get_decoupled_spec_transport()
        return transport.draft_tail_buffer_cls(
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
        transport = get_decoupled_spec_transport()
        self.draft_proxy_thread = transport.draft_proxy_thread_cls(
            context=context,
            verifier_rank=self.get_decoupled_spec_rank(),
            draft_tail_buffer=self.draft_tail_buffer,
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
        self: Scheduler, raw_batch_size: int, avg_ctx_len: int
    ) -> None:
        if not dynamic_verify_enabled(self.server_args):
            return
        if getattr(self.server_args, "disable_cuda_graph", False):
            raise RuntimeError(
                "decoupled verifier dynamic verify length requires full CUDA Graph."
            )
        self.model_worker.activate_step_by_batch(
            int(raw_batch_size), int(avg_ctx_len)
        )

    def _decoupled_verify_avg_ctx_len(self: Scheduler, batch: ScheduleBatch) -> int:
        if batch.batch_size() <= 0:
            return 1

        seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and int(seq_lens_cpu.numel()) > 0:
            return max(1, int(round(float(seq_lens_cpu.float().mean().item()))))

        return 1

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

    def _decoupled_verify_modeled_throughput(
        self: Scheduler, batch: ScheduleBatch, accept_length: float
    ) -> Optional[dict]:
        if (
            batch is None
            or not dynamic_verify_enabled(self.server_args)
            or not self.spec_algorithm.is_decoupled_verify()
        ):
            return None
        return self.model_worker.get_modeled_throughput(
            batch.batch_size(),
            self._decoupled_verify_avg_ctx_len(batch),
            accept_length,
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
        avg_ctx_len = self._decoupled_verify_avg_ctx_len(batch)
        self._activate_decoupled_verify_step_by_batch(
            batch.batch_size(),
            avg_ctx_len,
        )
        self._maybe_log_decoupled_verify_state_selection(batch.batch_size())
        num_speculative_steps, _ = self._active_decoupled_verify_config()

        if dynamic_verify_enabled(self.server_args) and num_speculative_steps == 0:
            self._prepare_zero_step_verify_inputs(batch)
            return

        self._snapshot_verify_inputs(batch)

    def _prepare_zero_step_verify_inputs(
        self: Scheduler,
        batch: ScheduleBatch,
    ) -> None:
        for req in batch.reqs:
            if req.is_retracted or req.finished():
                continue
            prepare_decoupled_verify_snapshot(req, len(req.output_ids))

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
    ) -> None:
        """
        Bind one broadcast draft tail snapshot set to the local verifier batch.

        Called after the entry-rank snapshot
        has been broadcast. All ranks, including the entry rank, use this same
        snapshot set for the request state so concurrent proxy updates cannot
        affect the current verifier forward pass.

        Args:
            target_reqs: Live verifier requests in the local batch.
            snapshots: Broadcast per-request draft tail snapshots.

        Returns:
            None.
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
                continue

            committed_len = int(snapshot.committed_len)
            if committed_len < len(req.output_ids):
                # the drafter has not caught up with the verifier req's committed output prefix
                continue
            if committed_len > len(req.output_ids):
                raise RuntimeError(
                    "Decoupled verify draft tail snapshot is out of sync with "
                    "the verifier request: "
                    f"request_id={req.rid} snapshot_committed_len={committed_len} "
                    f"request_output_len={len(req.output_ids)}"
                )
            verify_snapshot = req.decoupled_verify_snapshot
            if verify_snapshot is None:
                raise RuntimeError(
                    f"Missing decoupled verify state for request {req.rid}"
                )
            verify_snapshot.draft_tokens.extend(snapshot.tail_tokens)
            verify_snapshot.num_consumable_drafts = int(
                snapshot.num_consumable_drafts
            )

    def _sync_verify_requests(
        self,
        batch: ScheduleBatch,
    ) -> None:
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

        if get_decoupled_spec_transport().supports_native_rows:
            sync_rows: list[tuple] = []
            for req in batch.reqs:
                if not req.is_retracted and not req.finished():
                    prepare_decoupled_verify_snapshot(req, len(req.output_ids))
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
            self._send_verify_control_rows(sync_rows=sync_rows)
            return

        sync_messages: list[DraftSync] = []
        for req in batch.reqs:
            if not req.is_retracted and not req.finished():
                prepare_decoupled_verify_snapshot(req, len(req.output_ids))
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
        self._send_verify_control_batches(sync_messages=sync_messages)

    def _snapshot_verify_inputs(
        self,
        batch: ScheduleBatch,
    ) -> None:
        """
        Collect currently available draft tails, and bind them to a verifier request batch.

        Called immediately before a decoupled verify forward pass is prepared.
        The default path is non-blocking: the verifier entry rank snapshots the
        draft tail tokens already received by DraftProxyThread, broadcasts that
        stable per-forward snapshot to peer TP ranks, and all ranks bind
        the request's stable state from the broadcast snapshot.

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
            prepare_decoupled_verify_snapshot(req, len(req.output_ids))
        target_reqs = live_reqs
        if not target_reqs:
            return None

        is_dynamic = dynamic_verify_enabled(self.server_args)
        active_spec_steps, _ = self._active_decoupled_verify_config()
        snapshot_tail_cap = active_spec_steps if is_dynamic else None
        local_snapshots: list[DraftTailSnapshot] = []
        if self.is_verify_entry_rank():
            draft_tail_buffer = self.draft_tail_buffer
            assert draft_tail_buffer is not None
            allow_partial = envs.SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL.get()
            local_snapshots = draft_tail_buffer.get_draft_snapshots(
                target_reqs,
                allow_partial=allow_partial,
                max_tail_len=snapshot_tail_cap,
            )
            self._decoupled_verify_draft_wait_ns_since_log = getattr(
                self, "_decoupled_verify_draft_wait_ns_since_log", 0
            ) + int(draft_tail_buffer.last_draft_wait_ns)

        synced_snapshots = self._broadcast_verify_snapshots(local_snapshots)
        self._bind_verify_snapshots(target_reqs, synced_snapshots)

    def submit_verify_updates(
        self,
        batch: ScheduleBatch,
    ) -> None:
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

        if get_decoupled_spec_transport().supports_native_rows:
            commit_rows: list[tuple] = []
            close_rows: list[tuple] = []
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
                        self.release_drafter_rank(req.rid)
                    req.decoupled_verify_snapshot = None
                    continue

                if not has_request:
                    continue
                if not req.output_ids:
                    continue

                verify_snapshot = req.decoupled_verify_snapshot
                pre_verify_committed_len = (
                    verify_snapshot.pre_committed_len
                    if verify_snapshot is not None
                    else None
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
                    if verify_snapshot is not None:
                        verify_snapshot.reset(len(req.output_ids))
                    continue

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
                if verify_snapshot is not None:
                    verify_snapshot.reset(len(req.output_ids))

            self._send_verify_control_rows(
                commit_rows=commit_rows,
                close_rows=close_rows,
            )
            return

        verify_commit_messages: list[VerifyCommit] = []
        close_messages: list[DraftClose] = []
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
                    self.release_drafter_rank(req.rid)
                req.decoupled_verify_snapshot = None
                continue

            if not has_request:
                continue
            if not req.output_ids:
                continue

            verify_snapshot = req.decoupled_verify_snapshot
            pre_verify_committed_len = (
                verify_snapshot.pre_committed_len
                if verify_snapshot is not None
                else None
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
                if verify_snapshot is not None:
                    verify_snapshot.reset(len(req.output_ids))
                continue

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
            if verify_snapshot is not None:
                verify_snapshot.reset(len(req.output_ids))
        self._send_verify_control_batches(
            verify_commit_messages=verify_commit_messages,
            close_messages=close_messages,
        )

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
            if get_decoupled_spec_transport().supports_native_rows:
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

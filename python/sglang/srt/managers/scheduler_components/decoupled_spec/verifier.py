from __future__ import annotations

import logging
import statistics
import time
from typing import TYPE_CHECKING

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.managers.load_snapshot import (
    DecoupledSpecDecodeMetrics,
    DraftTailSelectMetrics,
    DraftTransportMetrics,
    IntegerHistogram,
)
from sglang.srt.managers.scheduler_components.decoupled_spec.base import (
    build_draft_mirror_request_id,
)
from sglang.srt.runtime_context import get_stream
from sglang.srt.speculative.cpp_decoupled_spec import (
    GPU_DRAFT_TAIL_SELECT_REASON_NAMES,
)
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
from sglang.srt.speculative.decoupled_verify_profile import (
    build_decoupled_verify_profile_fingerprint,
    file_sha256,
    load_decoupled_verify_profile,
    validate_profile_fingerprint,
)
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    DecoupledVerifyThroughputController,
    load_decoupled_verify_adaptive_config,
)
from sglang.srt.utils import broadcast_pyobj

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
        self._drafter_ranks = [int(peer.rank) for peer in config.peers]
        self._drafter_quotas = {
            int(peer.rank): int(peer.quota) for peer in config.peers
        }
        self._drafter_current_weights = {
            drafter_rank: 0 for drafter_rank in self._drafter_ranks
        }
        self._total_drafter_quota = sum(self._drafter_quotas.values())
        self.is_entry_rank = scheduler.ps.tp_rank == 0
        self.num_draft_tokens = int(scheduler.server_args.speculative_num_steps)
        self._open_mirror_by_req: dict[str, str] = {}
        self._open_lifecycle_by_req: dict[str, tuple[Req, int]] = {}
        self._open_drafter_rank_by_req: dict[str, int] = {}
        self._has_pending_lifecycle_landing_fence = False
        self._next_request_epoch = 1
        from sglang.srt.managers.scheduler_components.decoupled_spec.offline_profile import (
            DecoupledVerifyOfflineProfiler,
        )

        self.offline_profiler = DecoupledVerifyOfflineProfiler.maybe_create(scheduler)
        if (
            self.offline_profiler is not None
            and scheduler.server_args.speculative_adaptive
        ):
            raise ValueError(
                "Offline profiling and production adaptive verification cannot run "
                "in the same verifier process."
            )
        self.adaptive_controller: DecoupledVerifyThroughputController | None = None
        self._adaptive_config = None
        self._adaptive_decode_round_count = 0
        self.rebase_row_ct = 0
        self.rebase_hit_ct = 0
        self.selected_draft_tokens_ct = 0
        self._last_logged_rebase_rows = 0
        tail_capacity = 2 * self.num_draft_tokens + 1
        self._tail_select_row_ct = 0
        self._tail_select_valid_row_ct = 0
        self._tail_select_reason_ct = [0] * len(GPU_DRAFT_TAIL_SELECT_REASON_NAMES)
        self._selected_draft_length_ct = [0] * (self.num_draft_tokens + 1)
        self._raw_draft_tail_length_ct = [0] * (tail_capacity + 2)
        self._raw_draft_tail_length_outliers = [0, 0]
        self._consumable_draft_tail_length_ct = [0] * (tail_capacity + 2)
        self._consumable_draft_tail_length_outliers = [0, 0]
        self._logical_delta_ct = [0] * (2 * tail_capacity + 1)
        self._logical_delta_outliers = [0, 0]
        self._pending_prefix_length_ct = [0] * (tail_capacity + 2)
        self._pending_prefix_length_outliers = [0, 0]
        self._publish_seq_initial_ct = 0
        self._publish_seq_same_ct = 0
        self._publish_seq_advance_ct = 0
        self._pending_prefix_fast_forward_ct = 0
        self._protocol_error_ct = 0
        self._seqlock_retry_row_ct = 0
        self._seqlock_retry_ct = 0
        self._max_seqlock_retries = 0
        self._last_publish_seq_by_mirror: dict[str, int] = {}
        self._last_pending_prefix_fast_forward_ct_by_mirror: dict[str, int] = {}
        self.data_plane = (
            create_verifier_decoupled_spec_data_plane(
                config,
                required_tail_len=0,
                device=scheduler.device,
                num_gpu_seats=int(scheduler.req_to_token_pool.req_to_token.shape[0]),
                num_draft_tokens=self.num_draft_tokens,
                landing_stream=get_stream("decoupled_spec_landing"),
                mock_profile=self.offline_profiler is not None,
            )
            if self.is_entry_rank
            else None
        )
        if self.data_plane is not None:
            if getattr(self.data_plane, "gpu_tail_buffer", None) is None:
                raise RuntimeError(
                    "Decoupled verification requires the GPU draft-tail "
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
        self.verify_worker = verify_worker
        self._init_adaptive_controller()

    def _init_adaptive_controller(self) -> None:
        if not self.scheduler.server_args.speculative_adaptive:
            return
        profile_path = envs.SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH.get()
        if not profile_path:
            raise ValueError(
                "Adaptive decoupled verification requires "
                "SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH."
            )
        profile = load_decoupled_verify_profile(profile_path, require_complete=True)
        runtime_fingerprint = build_decoupled_verify_profile_fingerprint(
            server_args=self.scheduler.server_args,
            model_config=self.scheduler.model_config,
        )
        validate_profile_fingerprint(profile["fingerprint"], runtime_fingerprint)
        self._adaptive_config = load_decoupled_verify_adaptive_config(
            self.scheduler.server_args.speculative_adaptive_config,
            max_steps=self.num_draft_tokens,
        )
        startup_steps = max(self._adaptive_config["candidate_steps"])
        self.verify_worker.queue_verify_steps(startup_steps)
        if self.is_entry_rank:
            self.adaptive_controller = DecoupledVerifyThroughputController(
                max_steps=self.num_draft_tokens,
                config_path=self.scheduler.server_args.speculative_adaptive_config,
                profile=profile,
                profile_sha256=file_sha256(profile_path),
            )
        logger.info(
            "Loaded decoupled verifier adaptive profile: path=%s sha256=%s "
            "startup_steps=%s candidate_steps=%s",
            profile_path,
            file_sha256(profile_path),
            startup_steps,
            self._adaptive_config["candidate_steps"],
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

    def prepare_decode_allocation(self, batch: ScheduleBatch) -> None:
        return None

    def before_decode_retraction(self, batch: ScheduleBatch) -> None:
        return None

    def prepare_forward(self, batch: ScheduleBatch) -> None:
        return None

    def finish_forward(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> bool:
        return False

    def sleep_overrun_requests(self, batch):
        return batch

    def prepare_batch(self, batch: ScheduleBatch) -> None:
        if batch is None:
            return
        batch.decoupled_needs_landing_fence = False
        if batch.forward_mode.is_idle():
            return
        if batch.forward_mode.is_decode():
            profile_steps = (
                None
                if self.offline_profiler is None
                else self.offline_profiler.step_for_batch(batch)
            )
            if profile_steps is not None:
                self.verify_worker.activate_verify_steps(profile_steps)
            else:
                self.verify_worker.activate_pending_verify_steps()
            batch.decoupled_verify_steps = int(self.verify_worker.active_verify_steps)
        allow_open = batch.forward_mode.is_extend()
        pending_opens: dict[int, list[tuple[DraftSync, int, int]]] = {}
        batch.decoupled_launch_mirror_ids = []
        batch.decoupled_has_new_lifecycle = False
        for req in batch.reqs:
            mirror_request_id, opened = self._ensure_open_request(
                req,
                allow_open=allow_open,
                pending_opens=pending_opens,
            )
            batch.decoupled_launch_mirror_ids.append(mirror_request_id)
            batch.decoupled_has_new_lifecycle |= opened
        if self.data_plane is not None:
            for dst_drafter_rank in sorted(pending_opens):
                self.data_plane.open_requests(pending_opens[dst_drafter_rank])
        consume_lifecycle_fence = (
            self.is_entry_rank and self._has_pending_lifecycle_landing_fence
        )
        batch.decoupled_needs_landing_fence |= consume_lifecycle_fence
        self.verify_worker.capture_expected_request_epochs(batch)
        if consume_lifecycle_fence:
            self._has_pending_lifecycle_landing_fence = False

    def _ensure_open_request(
        self,
        req: Req,
        *,
        allow_open: bool,
        pending_opens: dict[int, list[tuple[DraftSync, int, int]]],
    ) -> tuple[str, bool]:
        mirror_request_id = self._mirror_for_lifecycle(req)
        if mirror_request_id is not None:
            return mirror_request_id, False

        if not allow_open and (
            req.finished() or getattr(req, "is_retracted", False)
        ):
            # The overlap plan can retain a row whose prior result already
            # closed and forgot its mirror. This stable identity is never sent
            # on wire; result processing rejects it against the live mapping.
            return str(req.rid), False

        previous = self._open_mirror_by_req.get(req.rid)
        if not allow_open:
            raise RuntimeError(
                "Decoupled verifier decode request has no open draft mirror for "
                "its current lifecycle: "
                f"request_id={req.rid} retraction_count={req.retraction_count}"
            )
        if previous is not None:
            self._close_request(req, previous, reason="reseated")

        dst_drafter_rank = self._select_drafter_rank()
        request_epoch = self._next_request_epoch
        self._next_request_epoch += 1
        mirror_request_id = build_draft_mirror_request_id(req.rid, request_epoch)
        if self.data_plane is not None:
            if req.req_pool_idx is None:
                raise RuntimeError(
                    "Decoupled verifier request has no GPU request-pool seat: "
                    f"request_id={req.rid}"
                )
            pending_opens.setdefault(dst_drafter_rank, []).append(
                (
                    DraftSync(
                        request_id=mirror_request_id,
                        src_verifier_rank=self.src_verifier_rank,
                        dst_drafter_rank=dst_drafter_rank,
                        max_new_tokens=int(req.sampling_params.max_new_tokens),
                        prompt_token_ids=[int(token) for token in req.origin_input_ids],
                        committed_outputs=[int(token) for token in req.output_ids],
                    ),
                    int(req.req_pool_idx),
                    request_epoch,
                )
            )
        self._open_mirror_by_req[req.rid] = mirror_request_id
        self._open_lifecycle_by_req[req.rid] = (
            req,
            int(req.retraction_count),
        )
        self._open_drafter_rank_by_req[req.rid] = dst_drafter_rank
        return mirror_request_id, True

    def _select_drafter_rank(self) -> int:
        """Choose one peer with integer smooth weighted round robin."""

        if not self._drafter_ranks or self._total_drafter_quota <= 0:
            raise RuntimeError("Decoupled verifier has no configured drafter peers.")
        for drafter_rank in self._drafter_ranks:
            self._drafter_current_weights[drafter_rank] += self._drafter_quotas[
                drafter_rank
            ]
        drafter_rank = min(
            self._drafter_ranks,
            key=lambda rank: (
                -self._drafter_current_weights[rank],
                (rank - self.src_verifier_rank) % self._total_drafter_quota,
                rank,
            ),
        )
        self._drafter_current_weights[drafter_rank] -= self._total_drafter_quota
        return drafter_rank

    def _mirror_for_lifecycle(self, req: Req) -> str | None:
        mirror_request_id = self._open_mirror_by_req.get(req.rid)
        lifecycle = self._open_lifecycle_by_req.get(req.rid)
        drafter_rank = self._open_drafter_rank_by_req.get(req.rid)
        if not (
            (mirror_request_id is not None)
            == (lifecycle is not None)
            == (drafter_rank is not None)
        ):
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
        # BatchResultProcessor synchronized the standard async D2H copy before
        # returning to this hook; consuming pinned host tensors is now safe.
        self._record_tail_select_result(batch, result)
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
        control_batches: dict[int, DraftControlBatch] = {}
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
            dst_drafter_rank = self._open_drafter_rank_by_req.get(req.rid)
            if dst_drafter_rank is None:
                raise RuntimeError(
                    "Decoupled verifier live mirror has no drafter route: "
                    f"request_id={req.rid} mirror_request_id={mirror_request_id}"
                )
            if getattr(req, "is_retracted", False):
                continue
            if req.finished():
                if self.is_entry_rank:
                    control_batch = control_batches.setdefault(
                        dst_drafter_rank,
                        DraftControlBatch(dst_drafter_rank=dst_drafter_rank),
                    )
                    control_batch.close_messages.append(
                        DraftClose(
                            request_id=mirror_request_id,
                            src_verifier_rank=self.src_verifier_rank,
                            dst_drafter_rank=dst_drafter_rank,
                            reason="finished",
                        )
                    )
                finished_lifecycles.append((req.rid, mirror_request_id))
                continue
            if not self.is_entry_rank:
                continue
            if len(req.output_ids) <= pre_output_len:
                continue
            control_batch = control_batches.setdefault(
                dst_drafter_rank,
                DraftControlBatch(dst_drafter_rank=dst_drafter_rank),
            )
            control_batch.verify_commit_messages.append(
                VerifyCommit(
                    request_id=mirror_request_id,
                    src_verifier_rank=self.src_verifier_rank,
                    dst_drafter_rank=dst_drafter_rank,
                    pre_verify_committed_len=pre_output_len,
                    committed_tokens=[
                        int(token) for token in req.output_ids[pre_output_len:]
                    ],
                )
            )
        if self.data_plane is not None:
            for dst_drafter_rank in sorted(control_batches):
                control_batch = control_batches[dst_drafter_rank]
                self.data_plane.submit_control_batch(
                    control_batch,
                    apply_local_verify_commits=False,
                )
                if control_batch.close_messages:
                    self._has_pending_lifecycle_landing_fence = True
        for request_id, mirror_request_id in finished_lifecycles:
            self._forget_open_request(request_id, mirror_request_id)
        metrics_reporter = getattr(self.scheduler, "metrics_reporter", None)
        if metrics_reporter is not None:
            metrics_reporter.finish_decoupled_decode_metrics_window()
        if batch.forward_mode.is_decode():
            self._observe_adaptive_verify(batch, result)
            if self.offline_profiler is not None:
                self.offline_profiler.record_decode_completion(
                    completion_ns=time.perf_counter_ns(),
                    batch=batch,
                    result=result,
                )

    def _observe_adaptive_verify(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        if self._adaptive_config is None:
            return
        if any(
            isinstance(req.rid, str) and req.rid.startswith(HEALTH_CHECK_RID_PREFIX)
            for req in batch.reqs
        ):
            return
        self._adaptive_decode_round_count += 1
        applied_steps = int(
            batch.decoupled_verify_steps
            if batch.decoupled_verify_steps is not None
            else int(result.speculative_num_draft_tokens or 1) - 1
        )
        correct_drafts = [
            int(value) for value in (result.num_correct_drafts_per_req_cpu or [])
        ]
        if self.is_entry_rank:
            proposed_drafts = result.num_proposed_drafts_per_req_cpu
            if proposed_drafts is None:
                raise RuntimeError(
                    "Adaptive decoupled verification requires selected draft lengths."
                )
            consumable_drafts = [
                min(self.num_draft_tokens, max(0, int(value)))
                for value in proposed_drafts
            ]
            if len(correct_drafts) != len(consumable_drafts):
                raise RuntimeError(
                    "Adaptive accept and GPU-tail supply rows do not align."
                )
            assert self.adaptive_controller is not None
            self.adaptive_controller.observe(
                correct_drafts=correct_drafts,
                consumable_drafts=consumable_drafts,
                applied_steps=applied_steps,
            )

        interval = max(1, int(self._adaptive_config["update_interval_rounds"]))
        if self._adaptive_decode_round_count % interval != 0:
            return
        pre_output_lens = list(batch.decoupled_pre_output_lens or [])
        context_lens = [
            len(req.origin_input_ids)
            + (
                int(pre_output_lens[row])
                if row < len(pre_output_lens)
                else len(req.output_ids)
            )
            for row, req in enumerate(batch.reqs)
        ]
        payload = None
        if self.is_entry_rank:
            assert self.adaptive_controller is not None
            payload = self.adaptive_controller.decide(
                batch_size=len(batch.reqs),
                context_len=max(1, int(round(statistics.fmean(context_lens)))),
            )
        if self.scheduler.ps.tp_size > 1:
            payload = broadcast_pyobj(
                payload,
                self.scheduler.tp_group.rank,
                self.scheduler.tp_cpu_group,
                src=self.scheduler.tp_group.ranks[0],
            )
        if payload is None:
            raise RuntimeError(
                "Adaptive verifier decision broadcast returned no payload."
            )
        new_steps = int(payload["new_steps"])
        if new_steps != self.verify_worker.active_verify_steps:
            # With overlap, result r may be processed after r+1 launched. The
            # pending state is consumed by prepare_batch at the r+2 boundary.
            self.verify_worker.queue_verify_steps(new_steps)

    @staticmethod
    def _record_integer_histogram(
        counts: list[int],
        outliers: list[int],
        *,
        offset: int,
        value: int,
    ) -> None:
        index = int(value) - int(offset)
        if index < 0:
            outliers[0] += 1
        elif index >= len(counts):
            outliers[1] += 1
        else:
            counts[index] += 1

    @staticmethod
    def _take_integer_histogram(
        counts: list[int],
        outliers: list[int],
        *,
        offset: int,
    ) -> IntegerHistogram:
        histogram = IntegerHistogram(
            offset=offset,
            counts=list(counts),
            underflow_count=outliers[0],
            overflow_count=outliers[1],
        )
        counts[:] = [0] * len(counts)
        outliers[:] = [0, 0]
        return histogram

    def _record_tail_select_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        row_valid = getattr(result, "decoupled_rebase_valid", None)
        row_valid_cpu = None if row_valid is None else row_valid.tolist()
        selected_lens = getattr(result, "decoupled_selected_draft_lens", None)
        selected_lens_cpu = getattr(result, "num_proposed_drafts_per_req_cpu", None)
        if row_valid_cpu is not None:
            self.rebase_row_ct += len(row_valid_cpu)
            self.rebase_hit_ct += sum(int(value) for value in row_valid_cpu)
        if selected_lens is not None:
            if selected_lens_cpu is None:
                raise RuntimeError(
                    "Decoupled selected draft lengths were not resolved on CPU."
                )
            self.selected_draft_tokens_ct += sum(selected_lens_cpu)

        debug_rows = getattr(result, "decoupled_tail_select_debug", None)
        if debug_rows is None:
            return
        if not self.is_entry_rank:
            raise RuntimeError(
                "Only verifier TP0 may receive GPU tail-selector debug rows."
            )
        if row_valid_cpu is None or selected_lens is None or selected_lens_cpu is None:
            raise RuntimeError(
                "Decoupled tail-selector debug requires row-valid and "
                "selected-length tensors."
            )
        if debug_rows.ndim != 2 or debug_rows.shape[1] < 7:
            raise RuntimeError(
                "Decoupled tail-selector debug tensor has an invalid shape: "
                f"shape={tuple(debug_rows.shape)}"
            )
        mirror_request_ids = batch.decoupled_result_mirror_ids
        if debug_rows.shape[0] != len(mirror_request_ids):
            raise RuntimeError(
                "Decoupled tail-selector debug rows do not match launch identities: "
                f"rows={debug_rows.shape[0]} mirrors={len(mirror_request_ids)}"
            )

        tail_capacity = 2 * self.num_draft_tokens + 1
        for valid, selected_len, debug, mirror_request_id in zip(
            row_valid_cpu,
            selected_lens_cpu,
            debug_rows.tolist(),
            mirror_request_ids,
        ):
            reason = int(debug[0])
            publish_seq = int(debug[1])
            logical_delta = int(debug[2])
            raw_draft_tail_length = int(debug[3])
            consumable_draft_tail_length = int(debug[4])
            pending_prefix_length = int(debug[5])
            update_error = int(debug[7]) if len(debug) >= 8 else 0
            update_error_op_seq = int(debug[8]) if len(debug) >= 9 else -1
            pending_prefix_fast_forward_ct = int(debug[9]) if len(debug) >= 10 else None
            seqlock_retries = int(debug[10]) if len(debug) >= 11 else 0
            selected_len = int(selected_len)
            valid = int(valid)

            if reason < 0 or reason >= len(self._tail_select_reason_ct):
                raise RuntimeError(
                    f"Decoupled tail selector returned an unknown reason: {reason}"
                )
            if selected_len < 0 or selected_len > self.num_draft_tokens:
                raise RuntimeError(
                    "Decoupled tail selector returned an invalid selected length: "
                    f"selected_len={selected_len}"
                )
            if seqlock_retries < 0:
                raise RuntimeError(
                    "Decoupled tail selector returned a negative seqlock retry "
                    f"count: retries={seqlock_retries}"
                )
            reason_is_valid = GPU_DRAFT_TAIL_SELECT_REASON_NAMES[reason] in (
                "direct",
                "rebased",
            )
            if bool(valid) != reason_is_valid:
                raise RuntimeError(
                    "Decoupled tail-selector reason disagrees with row-valid: "
                    f"reason={GPU_DRAFT_TAIL_SELECT_REASON_NAMES[reason]} "
                    f"row_valid={valid}"
                )
            if update_error != 0:
                self._protocol_error_ct += 1
                raise RuntimeError(
                    "GPU draft-tail update protocol failed: "
                    f"request_id={mirror_request_id} error_code={update_error} "
                    f"error_op_seq={update_error_op_seq} "
                    f"selector_reason={GPU_DRAFT_TAIL_SELECT_REASON_NAMES[reason]}"
                )
            fast_forward_counter_is_stable = GPU_DRAFT_TAIL_SELECT_REASON_NAMES[
                reason
            ] in (
                "direct",
                "logical_behind",
                "delta_too_large",
                "delta_beyond_consumable",
                "bonus_mismatch",
                "rebased",
            )
            if (
                pending_prefix_fast_forward_ct is not None
                and fast_forward_counter_is_stable
            ):
                if pending_prefix_fast_forward_ct < -1:
                    raise RuntimeError(
                        "GPU pending-prefix fast-forward counter is negative: "
                        f"request_id={mirror_request_id} "
                        f"counter={pending_prefix_fast_forward_ct}"
                    )
                previous_fast_forward_ct = (
                    self._last_pending_prefix_fast_forward_ct_by_mirror.get(
                        mirror_request_id
                    )
                )
                if pending_prefix_fast_forward_ct == -1:
                    previous_fast_forward_ct = None
                elif (
                    previous_fast_forward_ct is not None
                    and pending_prefix_fast_forward_ct < previous_fast_forward_ct
                ):
                    raise RuntimeError(
                        "GPU pending-prefix fast-forward counter regressed: "
                        f"request_id={mirror_request_id} "
                        f"previous={previous_fast_forward_ct} "
                        f"current={pending_prefix_fast_forward_ct}"
                    )
                if pending_prefix_fast_forward_ct >= 0:
                    self._pending_prefix_fast_forward_ct += (
                        pending_prefix_fast_forward_ct
                        if previous_fast_forward_ct is None
                        else pending_prefix_fast_forward_ct - previous_fast_forward_ct
                    )
                    self._last_pending_prefix_fast_forward_ct_by_mirror[
                        mirror_request_id
                    ] = pending_prefix_fast_forward_ct

            self._tail_select_row_ct += 1
            self._tail_select_valid_row_ct += valid
            self._tail_select_reason_ct[reason] += 1
            self._seqlock_retry_row_ct += int(seqlock_retries > 0)
            self._seqlock_retry_ct += seqlock_retries
            self._max_seqlock_retries = max(
                self._max_seqlock_retries, seqlock_retries
            )
            self._selected_draft_length_ct[selected_len] += 1
            self._record_integer_histogram(
                self._raw_draft_tail_length_ct,
                self._raw_draft_tail_length_outliers,
                offset=-1,
                value=raw_draft_tail_length,
            )
            self._record_integer_histogram(
                self._consumable_draft_tail_length_ct,
                self._consumable_draft_tail_length_outliers,
                offset=-1,
                value=consumable_draft_tail_length,
            )
            self._record_integer_histogram(
                self._logical_delta_ct,
                self._logical_delta_outliers,
                offset=-tail_capacity,
                value=logical_delta,
            )
            self._record_integer_histogram(
                self._pending_prefix_length_ct,
                self._pending_prefix_length_outliers,
                offset=-1,
                value=pending_prefix_length,
            )
            if publish_seq < 0:
                continue
            previous_publish_seq = self._last_publish_seq_by_mirror.get(
                mirror_request_id
            )
            if previous_publish_seq is None:
                self._publish_seq_initial_ct += 1
            elif publish_seq == previous_publish_seq:
                self._publish_seq_same_ct += 1
            elif publish_seq > previous_publish_seq:
                self._publish_seq_advance_ct += 1
            else:
                raise RuntimeError(
                    "GPU draft-tail publish sequence regressed: "
                    f"request_id={mirror_request_id} "
                    f"previous={previous_publish_seq} current={publish_seq}"
                )
            self._last_publish_seq_by_mirror[mirror_request_id] = publish_seq

    def take_decode_metrics_window(self) -> DecoupledSpecDecodeMetrics | None:
        """Freeze and reset counters aligned with the scheduler decode window."""

        if not self.is_entry_rank:
            return None
        transport = DraftTransportMetrics.from_dict(
            self.data_plane.take_transport_metrics()
        )
        reason_counts = {
            name: count
            for name, count in zip(
                GPU_DRAFT_TAIL_SELECT_REASON_NAMES, self._tail_select_reason_ct
            )
            if count
        }
        tail_capacity = 2 * self.num_draft_tokens + 1
        selected_draft_length_histogram = IntegerHistogram(
            offset=0,
            counts=list(self._selected_draft_length_ct),
        )
        self._selected_draft_length_ct[:] = [0] * len(self._selected_draft_length_ct)
        if (
            self._seqlock_retry_row_ct > self._tail_select_row_ct
            or self._seqlock_retry_ct < self._seqlock_retry_row_ct
            or self._max_seqlock_retries > self._seqlock_retry_ct
            or (
                self._seqlock_retry_row_ct > 0
                and self._max_seqlock_retries == 0
            )
            or (
                self._seqlock_retry_row_ct == 0
                and (self._seqlock_retry_ct or self._max_seqlock_retries)
            )
        ):
            raise RuntimeError(
                "Decoupled tail-selector seqlock retry counters disagree: "
                f"rows={self._tail_select_row_ct} "
                f"retry_rows={self._seqlock_retry_row_ct} "
                f"retries={self._seqlock_retry_ct} "
                f"max_retries={self._max_seqlock_retries}"
            )
        tail_select = DraftTailSelectMetrics(
            num_select_rows=self._tail_select_row_ct,
            num_select_valid_rows=self._tail_select_valid_row_ct,
            reason_counts=reason_counts,
            selected_draft_length_histogram=selected_draft_length_histogram,
            raw_draft_tail_length_histogram=self._take_integer_histogram(
                self._raw_draft_tail_length_ct,
                self._raw_draft_tail_length_outliers,
                offset=-1,
            ),
            consumable_draft_tail_length_histogram=self._take_integer_histogram(
                self._consumable_draft_tail_length_ct,
                self._consumable_draft_tail_length_outliers,
                offset=-1,
            ),
            logical_delta_histogram=self._take_integer_histogram(
                self._logical_delta_ct,
                self._logical_delta_outliers,
                offset=-tail_capacity,
            ),
            pending_prefix_length_histogram=self._take_integer_histogram(
                self._pending_prefix_length_ct,
                self._pending_prefix_length_outliers,
                offset=-1,
            ),
            num_publish_seq_initial=self._publish_seq_initial_ct,
            num_publish_seq_same=self._publish_seq_same_ct,
            num_publish_seq_advance=self._publish_seq_advance_ct,
            num_pending_prefix_fast_forwards=(self._pending_prefix_fast_forward_ct),
            num_protocol_errors=self._protocol_error_ct,
            num_seqlock_retry_rows=self._seqlock_retry_row_ct,
            num_seqlock_retries=self._seqlock_retry_ct,
            max_seqlock_retries=self._max_seqlock_retries,
        )
        self._tail_select_row_ct = 0
        self._tail_select_valid_row_ct = 0
        self._tail_select_reason_ct[:] = [0] * len(self._tail_select_reason_ct)
        self._publish_seq_initial_ct = 0
        self._publish_seq_same_ct = 0
        self._publish_seq_advance_ct = 0
        self._pending_prefix_fast_forward_ct = 0
        self._protocol_error_ct = 0
        self._seqlock_retry_row_ct = 0
        self._seqlock_retry_ct = 0
        self._max_seqlock_retries = 0
        return DecoupledSpecDecodeMetrics(
            tail_select=tail_select,
            transport=transport,
            adaptive_verify=(
                None
                if self.adaptive_controller is None
                else self.adaptive_controller.take_metrics_window()
            ),
        )

    def abort_request(self, req: Req, reason: str = "abort") -> None:
        mirror_request_id = self._open_mirror_by_req.get(req.rid)
        lifecycle = self._open_lifecycle_by_req.get(req.rid)
        drafter_rank = self._open_drafter_rank_by_req.get(req.rid)
        if not (
            (mirror_request_id is not None)
            == (lifecycle is not None)
            == (drafter_rank is not None)
        ):
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
        dst_drafter_rank = self._open_drafter_rank_by_req.get(req.rid)
        if dst_drafter_rank is None:
            raise RuntimeError(
                "Decoupled verifier live mirror has no drafter route: "
                f"request_id={req.rid} mirror_request_id={mirror_request_id}"
            )
        if self.data_plane is not None:
            self.data_plane.close_request(
                DraftClose(
                    request_id=mirror_request_id,
                    src_verifier_rank=self.src_verifier_rank,
                    dst_drafter_rank=dst_drafter_rank,
                    reason=reason,
                )
            )
            self._has_pending_lifecycle_landing_fence = True
        self._forget_open_request(req.rid, mirror_request_id)

    def _forget_open_request(
        self,
        request_id: str,
        mirror_request_id: str,
    ) -> None:
        if self._open_mirror_by_req.get(request_id) == mirror_request_id:
            self._last_publish_seq_by_mirror.pop(mirror_request_id, None)
            self._last_pending_prefix_fast_forward_ct_by_mirror.pop(
                mirror_request_id, None
            )
            self._open_mirror_by_req.pop(request_id, None)
            self._open_lifecycle_by_req.pop(request_id, None)
            self._open_drafter_rank_by_req.pop(request_id, None)

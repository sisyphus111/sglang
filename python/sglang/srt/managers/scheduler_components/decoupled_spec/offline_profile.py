"""Real-Scheduler offline profiler for decoupled verifier cycle ITL."""

from __future__ import annotations

import logging
import math
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_verify_profile import (
    DECOUPLED_VERIFY_PROFILE_ABI_VERSION,
    DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE,
    DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE,
    DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR,
    DECOUPLED_VERIFY_PROFILE_COST_SCOPE,
    DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER,
    DECOUPLED_VERIFY_PROFILE_KIND,
    DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK,
    DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION,
    DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE,
    atomic_write_json,
    build_decoupled_verify_profile_fingerprint,
    estimate_scheduler_cycle_cost,
    load_decoupled_verify_profile,
    load_decoupled_verify_profile_job,
    validate_profile_fingerprint,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


class DecoupledVerifyOfflineProfiler:
    """Measure completion gaps after normal result processing and local commit."""

    def __init__(self, scheduler: Scheduler, spec_path: str) -> None:
        self.scheduler = scheduler
        self.spec_path = str(Path(spec_path).expanduser().resolve())
        self.job = load_decoupled_verify_profile_job(self.spec_path)
        self.output_path = str(Path(self.job["output_path"]).expanduser().resolve())
        self.request_prefix = str(self.job["request_prefix"])
        self.warmup_iters = int(self.job["warmup_iters"])
        self.measure_iters = int(self.job["measure_iters"])
        if self.warmup_iters < 0 or self.measure_iters <= 0:
            raise ValueError("Offline profile warmup/measure counts are invalid.")
        self.points = [self._normalize_point(point) for point in self.job["points"]]
        self.requested_points = [
            self._normalize_point(point)
            for point in self.job.get("requested_points", self.job["points"])
        ]
        self.is_output_rank = scheduler.ps.tp_rank == 0 and scheduler.ps.pp_rank == 0
        self.runtime_fingerprint = build_decoupled_verify_profile_fingerprint(
            server_args=scheduler.server_args,
            model_config=scheduler.model_config,
        )
        self._profile = self._initialize_profile()
        self._point_index = 0
        self._previous_completion_ns: int | None = None
        self._num_gaps = 0
        self._costs_ms: list[float] = []
        self._measured_context_lens: list[float] = []
        self._measured_accept_lengths: list[float] = []
        self._measured_selected_draft_lens: list[float] = []
        self._measured_graph_replays = 0
        self._invalid_rounds = 0
        if self.is_output_rank:
            self._publish()

    @classmethod
    def maybe_create(
        cls, scheduler: Scheduler
    ) -> DecoupledVerifyOfflineProfiler | None:
        spec_path = envs.SGLANG_DECOUPLED_VERIFY_OFFLINE_PROFILE_SPEC_PATH.get()
        return None if not spec_path else cls(scheduler, spec_path)

    @staticmethod
    def configured_steps() -> list[int]:
        spec_path = envs.SGLANG_DECOUPLED_VERIFY_OFFLINE_PROFILE_SPEC_PATH.get()
        if not spec_path:
            return []
        job = load_decoupled_verify_profile_job(spec_path)
        return sorted({int(point["step"]) for point in job["points"]})

    @property
    def active(self) -> bool:
        return self._point_index < len(self.points)

    def step_for_batch(self, batch: ScheduleBatch) -> int | None:
        point_index = self._profile_point_index(batch)
        if point_index is None:
            return None
        point = self.points[point_index]
        return int(point["step"])

    def record_decode_completion(
        self,
        *,
        completion_ns: int,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        point_index = self._profile_point_index(batch)
        if point_index is None or point_index < self._point_index:
            return
        if point_index != self._point_index:
            raise RuntimeError(
                "Offline profile points reached the scheduler out of order: "
                f"expected={self._point_index} actual={point_index}."
            )
        point = self.points[point_index]
        batch_size = len(batch.reqs)
        applied_steps = int(
            getattr(result, "decoupled_verify_steps", None)
            if getattr(result, "decoupled_verify_steps", None) is not None
            else int(result.speculative_num_draft_tokens or 1) - 1
        )
        selected = [
            int(value) for value in (result.num_proposed_drafts_per_req_cpu or [])
        ]
        correct = list(result.num_correct_drafts_per_req_cpu or [])
        is_valid = (
            batch_size == int(point["batch_size"])
            and applied_steps == int(point["step"])
            and len(selected) == batch_size
            and all(value == applied_steps for value in selected)
            and len(correct) == batch_size
            and all(int(value) == applied_steps for value in correct)
            and bool(result.can_run_cuda_graph)
            and not any(getattr(req, "is_retracted", False) for req in batch.reqs)
            and not any(req.finished() for req in batch.reqs)
        )
        if not is_valid:
            self._invalid_rounds += 1
            self._previous_completion_ns = None
            self._num_gaps = 0
            self._costs_ms.clear()
            self._measured_context_lens.clear()
            self._measured_accept_lengths.clear()
            self._measured_selected_draft_lens.clear()
            self._measured_graph_replays = 0
            return

        if self._previous_completion_ns is None:
            self._previous_completion_ns = int(completion_ns)
            return
        gap_ms = (int(completion_ns) - self._previous_completion_ns) / 1e6
        self._previous_completion_ns = int(completion_ns)
        if not math.isfinite(gap_ms) or gap_ms <= 0:
            raise RuntimeError(f"Invalid scheduler-cycle profile gap: {gap_ms} ms.")
        self._num_gaps += 1
        if self._num_gaps <= self.warmup_iters:
            return

        pre_output_lens = list(getattr(batch, "decoupled_pre_output_lens", []) or [])
        context_lens = [
            len(req.origin_input_ids)
            + (
                pre_output_lens[row]
                if row < len(pre_output_lens)
                else len(req.output_ids)
            )
            for row, req in enumerate(batch.reqs)
        ]
        self._costs_ms.append(gap_ms)
        self._measured_context_lens.append(statistics.fmean(context_lens))
        self._measured_accept_lengths.append(
            statistics.fmean(int(value) + 1 for value in correct)
        )
        self._measured_selected_draft_lens.append(statistics.fmean(selected))
        self._measured_graph_replays += int(bool(result.can_run_cuda_graph))
        if len(self._costs_ms) == self.measure_iters:
            self._complete_point(point)

    @staticmethod
    def _normalize_point(point: dict[str, Any]) -> dict[str, int]:
        normalized = {
            "step": int(point["step"]),
            "batch_size": int(point["batch_size"]),
            "context_len": int(point["context_len"]),
            "input_context_len": int(point["input_context_len"]),
        }
        if (
            normalized["step"] < 0
            or normalized["batch_size"] <= 0
            or normalized["context_len"] <= 0
            or normalized["input_context_len"] <= 0
        ):
            raise ValueError(f"Invalid offline profile point: {point!r}.")
        return normalized

    def _profile_point_index(self, batch: ScheduleBatch) -> int | None:
        matching = [
            req.rid for req in batch.reqs if req.rid.startswith(self.request_prefix)
        ]
        if not matching:
            return None
        if len(matching) != len(batch.reqs):
            raise RuntimeError(
                "Offline profile requests mixed with normal serving traffic."
            )
        suffixes = [rid[len(self.request_prefix) :] for rid in matching]
        try:
            indices = {int(suffix.split("-", 1)[0][1:]) for suffix in suffixes}
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed offline profile request ids: {matching}."
            ) from exc
        if len(indices) != 1:
            raise RuntimeError(
                "One scheduler batch mixed multiple offline profile points."
            )
        point_index = indices.pop()
        if not 0 <= point_index < len(self.points):
            raise RuntimeError(
                f"Offline profile point index is out of range: {point_index}."
            )
        return point_index

    def _initialize_profile(self) -> dict[str, Any]:
        output = Path(self.output_path)
        if output.is_file():
            profile = load_decoupled_verify_profile(output, require_complete=False)
            validate_profile_fingerprint(
                profile["fingerprint"], self.runtime_fingerprint
            )
            expected_request = {
                "warmup_iters": self.warmup_iters,
                "measure_iters": self.measure_iters,
                "random_seed": int(self.job.get("random_seed", 42)),
            }
            if profile.get("request") != expected_request:
                raise ValueError(
                    "Cannot mix offline profile points measured with different "
                    f"request settings: existing={profile.get('request')!r} "
                    f"requested={expected_request!r}."
                )
        else:
            profile = {
                "schema_version": DECOUPLED_VERIFY_PROFILE_SCHEMA_VERSION,
                "kind": DECOUPLED_VERIFY_PROFILE_KIND,
                "status": "partial",
                "profile_abi_version": DECOUPLED_VERIFY_PROFILE_ABI_VERSION,
                "measurement_clock": DECOUPLED_VERIFY_PROFILE_MEASUREMENT_CLOCK,
                "cost_scope": DECOUPLED_VERIFY_PROFILE_COST_SCOPE,
                "cost_estimator": DECOUPLED_VERIFY_PROFILE_COST_ESTIMATOR,
                "draft_provider": DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER,
                "profile_control_plane": DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE,
                "context_anchor_mode": DECOUPLED_VERIFY_PROFILE_CONTEXT_ANCHOR_MODE,
                "trajectory_acceptance": DECOUPLED_VERIFY_PROFILE_TRAJECTORY_ACCEPTANCE,
                "fingerprint": self.runtime_fingerprint,
                "request": {
                    "warmup_iters": self.warmup_iters,
                    "measure_iters": self.measure_iters,
                    "random_seed": int(self.job.get("random_seed", 42)),
                },
                "points": [],
                "coverage": {},
                "provenance": dict(self.job.get("provenance", {})),
            }
        profile["draft_provider"] = DECOUPLED_VERIFY_PROFILE_DRAFT_PROVIDER
        profile["profile_control_plane"] = DECOUPLED_VERIFY_PROFILE_CONTROL_PLANE
        existing = {
            (int(point["step"]), int(point["batch_size"]), int(point["context_len"]))
            for point in profile["points"]
        }
        missing_job_points = [
            point
            for point in self.points
            if (point["step"], point["batch_size"], point["context_len"]) in existing
        ]
        if missing_job_points:
            raise ValueError(
                "Offline profile job includes points already present in its output: "
                f"{missing_job_points}."
            )
        return profile

    def _complete_point(self, point: dict[str, int]) -> None:
        ordered_costs = sorted(self._costs_ms)
        measurement = {
            **point,
            "cost_ms": estimate_scheduler_cycle_cost(self._costs_ms),
            "iteration_latency_ms": {
                "count": len(self._costs_ms),
                "min": ordered_costs[0],
                "p50": statistics.median(ordered_costs),
                "p95": ordered_costs[
                    min(
                        len(ordered_costs) - 1, math.ceil(0.95 * len(ordered_costs)) - 1
                    )
                ],
                "max": ordered_costs[-1],
                "mean": statistics.fmean(ordered_costs),
            },
            "mean_accept_length": statistics.fmean(self._measured_accept_lengths),
            "mean_selected_draft_length": statistics.fmean(
                self._measured_selected_draft_lens
            ),
            "mean_context_len": statistics.fmean(self._measured_context_lens),
            "cuda_graph_fraction": self._measured_graph_replays / len(self._costs_ms),
            "invalid_rounds_before_measurement": self._invalid_rounds,
        }
        if self.is_output_rank:
            self._profile["points"].append(measurement)
            self._profile["points"].sort(
                key=lambda value: (
                    int(value["step"]),
                    int(value["batch_size"]),
                    int(value["context_len"]),
                )
            )
        self._point_index += 1
        self._previous_completion_ns = None
        self._num_gaps = 0
        self._costs_ms = []
        self._measured_context_lens = []
        self._measured_accept_lengths = []
        self._measured_selected_draft_lens = []
        self._measured_graph_replays = 0
        self._invalid_rounds = 0
        if self.is_output_rank:
            self._publish()
        logger.info(
            "Profiled decoupled verifier scheduler cycle: step=%s bs=%s ctx=%s "
            "cost_ms=%.4f point=%s/%s",
            point["step"],
            point["batch_size"],
            point["context_len"],
            measurement["cost_ms"],
            self._point_index,
            len(self.points),
        )

    def _publish(self) -> None:
        requested_keys = {
            (point["step"], point["batch_size"], point["context_len"])
            for point in self.requested_points
        }
        completed_keys = {
            (int(point["step"]), int(point["batch_size"]), int(point["context_len"]))
            for point in self._profile["points"]
        }
        self._profile["coverage"] = {
            "requested": len(requested_keys),
            "reused": int(self.job.get("reused_point_count", 0)),
            "profiled": self._point_index,
            "completed": len(requested_keys & completed_keys),
            "missing": [list(key) for key in sorted(requested_keys - completed_keys)],
        }
        self._profile["status"] = (
            "complete" if requested_keys.issubset(completed_keys) else "partial"
        )
        atomic_write_json(self.output_path, self._profile)


__all__ = ["DecoupledVerifyOfflineProfiler"]

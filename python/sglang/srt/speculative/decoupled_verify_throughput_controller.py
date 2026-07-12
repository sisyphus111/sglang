"""Throughput-aware adaptive controller for decoupled verifier workers."""

from __future__ import annotations

import logging
import math
from typing import Optional

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveSpecWorker,
    _SpecAdaptiveBase,
)
from sglang.srt.speculative.decoupled_verify_cost_model import (
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_UPDATE_INTERVAL,
    DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES,
    DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS,
    BatchSizeCostTable,
    DraftPositionStatsTracker,
    _format_optional_score,
    _score_for_step,
    _with_runtime_cpu_overhead_ms,
    format_score_rows,
    pick_best_step,
    pick_best_step_with_hysteresis,
    score_decoupled_verify_candidates,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


def parse_decoupled_verify_throughput_profile_ctx_lens(
    raw: Optional[str],
) -> Optional[list[int]]:
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None

    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise ValueError(
                "decoupled verifier throughput profile ctx lens must be a "
                f"comma-separated list of positive integers, got {raw!r}."
            )
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(
                "decoupled verifier throughput profile ctx lens must be a "
                f"comma-separated list of positive integers, got {raw!r}."
            ) from exc
        if value <= 0:
            raise ValueError(
                "decoupled verifier throughput profile ctx lens must be positive, "
                f"got {value} in {raw!r}."
            )
        values.append(value)

    return sorted(set(values))


class DecoupledVerifyThroughputAwareController(_SpecAdaptiveBase):
    """Select verify steps from global draft supply/accept EMAs and profile cost."""

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        *,
        candidate_steps: list[int],
        initial_steps: int,
        ema_alpha: float = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA,
        warmup_batches: int = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES,
        update_interval: int = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_UPDATE_INTERVAL,
        switch_hysteresis: float = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS,
    ) -> None:
        super().__init__(worker)
        self._candidate_steps = sorted({int(step) for step in candidate_steps})
        if not self._candidate_steps:
            raise ValueError("candidate_steps must not be empty.")
        if any(step < 0 for step in self._candidate_steps):
            raise ValueError(
                f"candidate_steps must be non-negative, got {self._candidate_steps}."
            )
        positive_steps = self._positive_candidate_steps()
        if positive_steps and positive_steps != list(
            range(positive_steps[0], positive_steps[-1] + 1)
        ):
            raise ValueError(
                "positive candidate_steps must be contiguous so position EMA "
                f"exploration cannot skip unobserved draft positions, got "
                f"{self._candidate_steps}."
            )

        initial_steps = int(initial_steps)
        if initial_steps > 0 and initial_steps in self._candidate_steps:
            self._current_steps = initial_steps
        else:
            if not positive_steps:
                raise ValueError("candidate_steps must include a positive step.")
            self._current_steps = positive_steps[0]
        self._update_interval = max(1, int(update_interval))
        self._switch_hysteresis = float(switch_hysteresis)
        self._batch_count = 0
        self._probe_steps: Optional[int] = self._current_steps
        self._tracker = DraftPositionStatsTracker(
            max_steps=max(self._candidate_steps),
            ema_alpha=ema_alpha,
            warmup_batches=warmup_batches,
        )
        self._cost_table = BatchSizeCostTable()

    @property
    def candidate_steps(self) -> list[int]:
        return list(self._candidate_steps)

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        missing_steps = [
            steps for steps in self._candidate_steps if steps not in self._states
        ]
        if missing_steps:
            raise ValueError(
                "Missing decoupled verifier throughput-aware runtime states for "
                f"steps={missing_steps}."
            )
        self._activate(self._current_steps)

    def set_profile_cost(
        self, *, batch_size: int, steps: int, ctx_len: int, cost_ms: float
    ) -> None:
        self._cost_table.set(batch_size, steps, ctx_len, cost_ms)

    def cost_table_summary(self) -> str:
        return self._cost_table.summary()

    def cost_table_items(self) -> list[tuple[int, int, int, float]]:
        return self._cost_table.items()

    def get_modeled_throughput(
        self, *, batch_size: int, ctx_len: int, accept_length: float
    ) -> Optional[dict]:
        ctx_len = max(1, int(ctx_len))
        profile_cost_ms, matched_batch_size, matched_ctx_len = (
            self._cost_table.lookup_with_match(batch_size, self._current_steps, ctx_len)
        )
        cost_ms = _with_runtime_cpu_overhead_ms(profile_cost_ms)
        accept_length = float(accept_length)
        if (
            cost_ms is None
            or cost_ms <= 0
            or not math.isfinite(accept_length)
            or accept_length < 0
        ):
            return None
        return {
            "modeled_throughput": int(batch_size) * accept_length * 1000.0 / cost_ms,
            "batch_size": int(batch_size),
            "steps": int(self._current_steps),
            "ctx_len": ctx_len,
            "matched_batch_size": matched_batch_size,
            "matched_ctx_len": matched_ctx_len,
            "cost_ms": cost_ms,
            "profile_cost_ms": profile_cost_ms,
            "runtime_cpu_overhead_ms": DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS,
        }

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        *,
        num_consumable_drafts_per_req: list[int],
        verified_steps: int,
        batch_size: int = 0,
    ) -> None:
        if not num_correct_drafts_per_req:
            return
        if batch_size and int(batch_size) != len(num_correct_drafts_per_req):
            raise ValueError(
                "Decoupled verifier batch_size must match the number of per-request "
                f"observations, got batch_size={batch_size}, "
                f"observations={len(num_correct_drafts_per_req)}."
            )
        self._batch_count += 1
        self._tracker.update(
            num_correct_drafts_per_req,
            num_consumable_drafts_per_req,
            verified_steps,
        )

    def activate_step_by_batch(self, batch_size: int, ctx_len: int = 1) -> None:
        if self._should_reevaluate():
            self._reevaluate_and_switch(batch_size, ctx_len)
        if self._current_steps != self.worker.speculative_num_steps:
            self._activate(self._current_steps)

    def _should_reevaluate(self) -> bool:
        if (
            self._batch_count <= 0
            or self._batch_count % self._update_interval != 0
            or self._cost_table.is_empty()
        ):
            return False
        if self._current_steps == 0 and self._positive_candidate_steps():
            return True
        return True

    def _reevaluate_and_switch(self, batch_size: int, ctx_len: int) -> None:
        ctx_len = max(1, int(ctx_len))
        switch_reason = "score_hysteresis"
        if self._probe_steps is not None:
            if not self._tracker.all_positions_warmed(self._probe_steps):
                return
            self._probe_steps = None

        rows = score_decoupled_verify_candidates(
            self._tracker,
            self._cost_table,
            self._candidate_steps,
            batch_size,
            ctx_len,
        )
        if self._current_steps == 0 and not self._has_positive_score(
            batch_size, ctx_len
        ):
            best_steps = self._positive_candidate_steps()[0]
            switch_reason = "positive_probe"
            self._probe_steps = best_steps
        else:
            eligible_rows = [row for row in rows if row["eligible"]]
            best_steps = pick_best_step_with_hysteresis(
                eligible_rows,
                current_steps=self._current_steps,
                hysteresis=self._switch_hysteresis,
            )
            if best_steps == self._current_steps:
                probe_steps = self._next_cold_probe_steps()
                probe_row = self._row_for_step(rows, probe_steps)
                current_score = _score_for_step(rows, self._current_steps)
                if probe_row is not None and current_score is not None:
                    optimistic_expected = self._tracker.get_optimistic_expected_tokens(
                        probe_steps
                    )
                    probe_cost_ms = probe_row.get("cost_ms")
                    optimistic_score = (
                        optimistic_expected / probe_cost_ms
                        if optimistic_expected is not None
                        and probe_cost_ms is not None
                        and probe_cost_ms > 0
                        else None
                    )
                    if (
                        optimistic_score is not None
                        and optimistic_score
                        > current_score * (1.0 + self._switch_hysteresis)
                    ):
                        probe_row["expected"] = optimistic_expected
                        probe_row["score"] = optimistic_score
                        probe_row["eligible"] = True
                        probe_row["expected_source"] = "optimistic_probe"
                        best_steps = probe_steps
                        self._probe_steps = probe_steps
                        switch_reason = "cold_position_probe"

        if best_steps == self._current_steps:
            return

        old_steps = self._current_steps
        current_score = _score_for_step(rows, old_steps)
        best_score = _score_for_step(rows, best_steps)
        score_ratio = (
            best_score / current_score
            if best_score is not None and current_score is not None and current_score > 0
            else None
        )
        self._current_steps = best_steps
        log_info_on_rank0(
            logger,
            "Decoupled verifier throughput-aware step switch: "
            f"steps {old_steps} -> {best_steps}, bs={int(batch_size)}, "
            f"avg_ctx_len={ctx_len}, "
            f"batch_count={self._batch_count}, "
            f"current_score={_format_optional_score(current_score)}, "
            f"best_score={_format_optional_score(best_score)}, "
            f"best_over_current={_format_optional_score(score_ratio)}, "
            f"reason={switch_reason}, "
            f"scores={format_score_rows(rows, best_steps)}",
        )

    def _positive_candidate_steps(self) -> list[int]:
        return [steps for steps in self._candidate_steps if steps > 0]

    def _next_cold_probe_steps(self) -> Optional[int]:
        next_steps = self._current_steps + 1
        if next_steps not in self._candidate_steps:
            return None
        if self._tracker.all_positions_warmed(next_steps):
            return None
        return next_steps

    @staticmethod
    def _row_for_step(rows: list[dict], steps: Optional[int]) -> Optional[dict]:
        if steps is None:
            return None
        return next(
            (row for row in rows if int(row["steps"]) == int(steps)), None
        )

    def _has_positive_score(self, batch_size: int, ctx_len: int) -> bool:
        rows = score_decoupled_verify_candidates(
            self._tracker,
            self._cost_table,
            self._positive_candidate_steps(),
            batch_size,
            ctx_len,
        )
        return any(row["score"] is not None for row in rows)

"""Throughput-aware adaptive controller for decoupled verifier workers."""

from __future__ import annotations

import bisect
import logging
import math
import os
from collections import deque
from typing import Optional

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveSpecWorker,
    _SpecAdaptiveBase,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WINDOW_SIZE = 50
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_UPDATE_INTERVAL = 5
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS = 0.1


def _ta_debug_enabled() -> bool:
    return os.environ.get("SGLANG_TA_DEBUG") == "1"


class PositionAcceptanceTracker:
    """Tracks per-draft-position acceptance rates from verifier results."""

    def __init__(self, *, max_steps: int, window_size: int):
        self.max_steps = int(max_steps)
        self.window_size = int(window_size)
        self._windows: list[deque[float]] = [
            deque(maxlen=self.window_size) for _ in range(max(0, self.max_steps))
        ]

    def update(self, num_correct_drafts_per_req: list[int], current_steps: int) -> None:
        num_reqs = len(num_correct_drafts_per_req)
        steps = min(max(0, int(current_steps)), self.max_steps)
        if num_reqs == 0 or steps == 0:
            return

        counts = [0] * (steps + 1)
        for raw_accept_len in num_correct_drafts_per_req:
            accept_len = max(0, min(int(raw_accept_len), steps))
            counts[accept_len] += 1

        cumulative_rejected_or_stopped = 0
        for position in range(steps):
            cumulative_rejected_or_stopped += counts[position]
            accepted = num_reqs - cumulative_rejected_or_stopped
            self._windows[position].append(accepted / num_reqs)

    def all_positions_warmed(self, target_steps: int) -> bool:
        target_steps = min(max(0, int(target_steps)), self.max_steps)
        return all(
            len(self._windows[position]) >= self.window_size
            for position in range(target_steps)
        )

    def get_expected_tokens(self, target_steps: int) -> Optional[float]:
        target_steps = int(target_steps)
        if target_steps < 0:
            return None
        if target_steps == 0:
            return 1.0

        rates: list[float] = []
        for position in range(target_steps):
            rate = self._rate_or_extrapolate(position, rates)
            if rate is None:
                return None
            rates.append(rate)
        return 1.0 + sum(rates)

    def snapshot_position_rates(self, num_positions: int) -> list[Optional[float]]:
        rates: list[Optional[float]] = []
        known_rates: list[float] = []
        for position in range(max(0, int(num_positions))):
            rate = self._rate_or_extrapolate(position, known_rates)
            rates.append(rate)
            if rate is not None:
                known_rates.append(rate)
        return rates

    def is_position_extrapolated(self, position: int) -> bool:
        return 0 <= int(position) < self.max_steps and not self._windows[int(position)]

    def _window_rate(self, position: int) -> Optional[float]:
        if position >= self.max_steps:
            return None
        window = self._windows[position]
        return sum(window) / len(window) if window else None

    def _rate_or_extrapolate(
        self, position: int, known_rates: list[float]
    ) -> Optional[float]:
        rate = self._window_rate(position)
        if rate is not None:
            return rate
        if not known_rates:
            return None
        if len(known_rates) == 1:
            return known_rates[0]

        first_rate = known_rates[0]
        last_rate = known_rates[-1]
        if first_rate <= 0:
            return 0.0
        alpha = (last_rate / first_rate) ** (1.0 / (len(known_rates) - 1))
        return max(0.0, min(last_rate, last_rate * alpha))


class BatchSizeCostTable:
    """Stores verifier decode cost in milliseconds for (batch_size, steps)."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int], float] = {}
        self._batch_sizes_by_step: dict[int, list[int]] = {}

    def set(self, batch_size: int, steps: int, cost_ms: float) -> None:
        batch_size = int(batch_size)
        steps = int(steps)
        cost_ms = float(cost_ms)
        if batch_size <= 0 or steps < 0 or not math.isfinite(cost_ms) or cost_ms <= 0:
            raise ValueError(
                "Cost table entries require positive batch size, non-negative "
                f"steps, and positive finite cost, got bs={batch_size}, "
                f"steps={steps}, cost_ms={cost_ms}."
            )

        self._data[(batch_size, steps)] = cost_ms
        self._batch_sizes_by_step[steps] = sorted(
            set(self._batch_sizes_by_step.get(steps, [])) | {batch_size}
        )

    def lookup(self, batch_size: int, steps: int) -> Optional[float]:
        batch_sizes = self._batch_sizes_by_step.get(int(steps))
        if not batch_sizes:
            return None
        index = bisect.bisect_left(batch_sizes, int(batch_size))
        if index >= len(batch_sizes):
            index = len(batch_sizes) - 1
        return self._data.get((batch_sizes[index], int(steps)))

    def has_exact(self, batch_size: int, steps: int) -> bool:
        return (int(batch_size), int(steps)) in self._data

    def items(self) -> list[tuple[int, int, float]]:
        return [
            (batch_size, steps, cost_ms)
            for (batch_size, steps), cost_ms in sorted(self._data.items())
        ]

    def is_empty(self) -> bool:
        return not self._data

    def summary(self) -> str:
        if not self._data:
            return "{}"
        return (
            "{"
            + ", ".join(
                f"(bs={bs}, steps={steps}): {cost:.4f}ms"
                for (bs, steps), cost in sorted(self._data.items())
            )
            + "}"
        )


def score_decoupled_verify_candidates(
    tracker: PositionAcceptanceTracker,
    cost_table: BatchSizeCostTable,
    candidate_steps: list[int],
    batch_size: int,
) -> list[dict]:
    rows = []
    for raw_steps in candidate_steps:
        steps = int(raw_steps)
        position_accept_rates = tracker.snapshot_position_rates(steps)
        expected = tracker.get_expected_tokens(steps)
        cost_ms = cost_table.lookup(batch_size, steps)
        score = (
            expected / cost_ms
            if expected is not None and cost_ms is not None and cost_ms > 0
            else None
        )
        rows.append(
            {
                "steps": steps,
                "expected": expected,
                "cost_ms": cost_ms,
                "score": score,
                "position_accept_rates": position_accept_rates,
                "position_accept_sources": [
                    (
                        "unknown"
                        if rate is None
                        else (
                            "extra"
                            if tracker.is_position_extrapolated(position)
                            else "win"
                        )
                    )
                    for position, rate in enumerate(position_accept_rates)
                ],
            }
        )
    return rows


def pick_best_step(rows: list[dict], fallback: int) -> int:
    best_step = int(fallback)
    best_score = -math.inf
    for row in rows:
        score = row.get("score")
        if score is not None and score > best_score:
            best_score = float(score)
            best_step = int(row["steps"])
    return best_step


def pick_best_step_with_hysteresis(
    rows: list[dict], *, current_steps: int, hysteresis: float
) -> int:
    best_step = pick_best_step(rows, fallback=current_steps)
    if best_step == int(current_steps):
        return int(current_steps)

    score_by_step = {int(row["steps"]): row.get("score") for row in rows}
    best_score = score_by_step.get(best_step)
    current_score = score_by_step.get(int(current_steps))
    if best_score is None:
        return int(current_steps)
    if current_score is None or current_score <= 0:
        return best_step
    return (
        best_step
        if best_score > current_score * (1.0 + float(hysteresis))
        else int(current_steps)
    )


def format_score_rows(rows: list[dict], best_steps: int) -> str:
    parts = []
    for row in rows:
        marker = "*" if int(row["steps"]) == int(best_steps) else ""
        expected = row["expected"]
        cost_ms = row["cost_ms"]
        score = row["score"]
        accept_rates = row.get("position_accept_rates") or []
        accept_sources = row.get("position_accept_sources") or []
        rate_parts = []
        for position, rate in enumerate(accept_rates):
            source = accept_sources[position] if position < len(accept_sources) else "?"
            if rate is None:
                rate_parts.append(f"p{position + 1}=?")
            else:
                rate_parts.append(f"p{position + 1}={rate:.3f}:{source}")
        rate_text = f",rates=[{','.join(rate_parts)}]" if rate_parts else ""
        if score is None:
            parts.append(f"S={row['steps']}:score=?{rate_text}{marker}")
        else:
            parts.append(
                f"S={row['steps']}:E={expected:.3f}/cost={cost_ms:.4f}ms"
                f"={score:.6f}{rate_text}{marker}"
            )
    return "[" + ", ".join(parts) + "]"


def _score_for_step(rows: list[dict], steps: int) -> Optional[float]:
    for row in rows:
        if int(row["steps"]) == int(steps):
            score = row.get("score")
            return None if score is None else float(score)
    return None


def _format_optional_score(score: Optional[float]) -> str:
    return "?" if score is None else f"{score:.6f}"


class DecoupledVerifyThroughputAwareController(_SpecAdaptiveBase):
    """Select decoupled verifier steps from accept rates and profiled cost."""

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        *,
        candidate_steps: list[int],
        initial_steps: int,
        window_size: int = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WINDOW_SIZE,
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

        initial_steps = int(initial_steps)
        if initial_steps > 0 and initial_steps in self._candidate_steps:
            self._current_steps = initial_steps
        else:
            positive_steps = self._positive_candidate_steps()
            if not positive_steps:
                raise ValueError("candidate_steps must include a positive step.")
            self._current_steps = positive_steps[0]
        self._update_interval = max(1, int(update_interval))
        self._switch_hysteresis = float(switch_hysteresis)
        self._batch_count = 0
        self._cuda_graph_bs: list[int] | None = None
        self._tracker = PositionAcceptanceTracker(
            max_steps=max(self._candidate_steps),
            window_size=max(1, int(window_size)),
        )
        self._cost_table = BatchSizeCostTable()

    @property
    def candidate_steps(self) -> list[int]:
        return list(self._candidate_steps)

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        self._cuda_graph_bs = sorted({int(bs) for bs in cuda_graph_bs or []}) or None
        missing_steps = [
            steps for steps in self._candidate_steps if steps not in self._states
        ]
        if missing_steps:
            raise ValueError(
                "Missing decoupled verifier throughput-aware runtime states for "
                f"steps={missing_steps}."
            )
        self._activate(self._current_steps)

    def set_profile_cost(self, *, batch_size: int, steps: int, cost_ms: float) -> None:
        self._cost_table.set(batch_size, steps, cost_ms)

    def cost_table_summary(self) -> str:
        return self._cost_table.summary()

    def cost_table_items(self) -> list[tuple[int, int, float]]:
        return self._cost_table.items()

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if not num_correct_drafts_per_req:
            return
        self._tracker.update(num_correct_drafts_per_req, self._current_steps)
        self._batch_count += 1

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self._should_reevaluate():
            self._reevaluate_and_switch(batch_size)
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
        return self._tracker.all_positions_warmed(self._current_steps)

    def _reevaluate_and_switch(self, batch_size: int) -> None:
        if self._current_steps == 0 and not self._has_positive_score(batch_size):
            best_steps = self._positive_candidate_steps()[0]
            rows = []
        else:
            rows = score_decoupled_verify_candidates(
                self._tracker, self._cost_table, self._candidate_steps, batch_size
            )
            best_steps = pick_best_step_with_hysteresis(
                rows,
                current_steps=self._current_steps,
                hysteresis=self._switch_hysteresis,
            )

        if _ta_debug_enabled():
            current_score = _score_for_step(rows, self._current_steps)
            best_score = _score_for_step(rows, best_steps)
            score_ratio = (
                best_score / current_score
                if best_score is not None
                and current_score is not None
                and current_score > 0
                else None
            )
            log_info_on_rank0(
                logger,
                "[TA-VERIFY-SCORE] "
                f"batch_count={self._batch_count}, bs={int(batch_size)}, "
                f"active_steps={self._current_steps}, "
                f"scores={format_score_rows(rows, best_steps)}, "
                f"best_steps={best_steps}, "
                f"current_score={_format_optional_score(current_score)}, "
                f"best_score={_format_optional_score(best_score)}, "
                f"best_over_current={_format_optional_score(score_ratio)}, "
                f"switch_hysteresis={self._switch_hysteresis:.4f}",
            )

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
        if _ta_debug_enabled():
            log_info_on_rank0(
                logger,
                "[TA-VERIFY-SWITCH] "
                f"batch_count={self._batch_count}, bs={int(batch_size)}, "
                f"from_steps={old_steps}, to_steps={best_steps}, "
                f"current_score={_format_optional_score(current_score)}, "
                f"best_score={_format_optional_score(best_score)}, "
                f"best_over_current={_format_optional_score(score_ratio)}, "
                f"switch_hysteresis={self._switch_hysteresis:.4f}, "
                f"reason=score_hysteresis",
            )
        log_info_on_rank0(
            logger,
            "Decoupled verifier throughput-aware step switch: "
            f"steps {old_steps} -> {best_steps}, bs={int(batch_size)}, "
            f"batch_count={self._batch_count}, "
            f"current_score={_format_optional_score(current_score)}, "
            f"best_score={_format_optional_score(best_score)}, "
            f"best_over_current={_format_optional_score(score_ratio)}, "
            f"scores={format_score_rows(rows, best_steps)}",
        )

    def _positive_candidate_steps(self) -> list[int]:
        return [steps for steps in self._candidate_steps if steps > 0]

    def _has_positive_score(self, batch_size: int) -> bool:
        rows = score_decoupled_verify_candidates(
            self._tracker,
            self._cost_table,
            self._positive_candidate_steps(),
            batch_size,
        )
        return any(row["score"] is not None for row in rows)

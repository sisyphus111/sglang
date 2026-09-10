"""Acceptance/supply EMA and scheduler-cycle cost model for decoupled verify."""

from __future__ import annotations

import bisect
import math
from typing import Any


class DraftPositionStatsTracker:
    """Track per-position draft supply and conditional acceptance EMAs."""

    def __init__(self, *, max_steps: int, ema_alpha: float):
        self.max_steps = int(max_steps)
        self.ema_alpha = float(ema_alpha)
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        self._supply_ema: list[float | None] = [None] * self.max_steps
        self._accept_ema: list[float | None] = [None] * self.max_steps
        self._accept_updates = [0] * self.max_steps
        self._accept_samples = [0] * self.max_steps

    def update(
        self,
        *,
        correct_drafts: list[int],
        consumable_drafts: list[int],
        applied_steps: int,
    ) -> None:
        if len(correct_drafts) != len(consumable_drafts):
            raise ValueError("Correct and consumable draft rows must align.")
        if not correct_drafts:
            return
        applied_steps = int(applied_steps)
        if not 0 <= applied_steps <= self.max_steps:
            raise ValueError("applied_steps is outside the configured range.")
        correct = [int(value) for value in correct_drafts]
        consumable = [min(self.max_steps, int(value)) for value in consumable_drafts]
        for accepted, supplied in zip(correct, consumable, strict=True):
            if (
                supplied < 0
                or accepted < 0
                or accepted > supplied
                or accepted > applied_steps
            ):
                raise ValueError(
                    "Correct drafts must not exceed supply or applied steps: "
                    f"correct={accepted}, consumable={supplied}, steps={applied_steps}."
                )
        num_rows = len(correct)
        # ``consumable_drafts`` is the length actually verified this round and
        # is therefore capped by active K. Positions beyond active K were not
        # observed; keep their previous EMA instead of treating them as zero
        # supply and biasing a future K increase downward.
        for position in range(applied_steps):
            num_supplied = sum(value > position for value in consumable)
            self._supply_ema[position] = self._update_ema(
                self._supply_ema[position], num_supplied / num_rows
            )
            if position >= applied_steps or num_supplied == 0:
                continue
            conditional_accept = (
                sum(value > position for value in correct) / num_supplied
            )
            self._accept_ema[position] = self._update_ema(
                self._accept_ema[position], conditional_accept
            )
            self._accept_updates[position] += 1
            self._accept_samples[position] += num_supplied

    def _update_ema(self, current: float | None, value: float) -> float:
        if current is None:
            return float(value)
        return (1.0 - self.ema_alpha) * current + self.ema_alpha * float(value)

    def expected_accept_length(self, steps: int) -> float | None:
        steps = int(steps)
        if steps < 0 or steps > self.max_steps:
            return None
        expected = 1.0
        for position in range(steps):
            supply = self._supply_ema[position]
            acceptance = self._accept_ema[position]
            if supply is None:
                return None
            # Starting from Kmax observes every candidate position immediately.
            # No supply is a complete zero-contribution observation even though
            # conditional acceptance is undefined for that position.
            if supply == 0.0:
                continue
            if acceptance is None:
                return None
            expected += supply * acceptance
        return expected

    def snapshot(self) -> dict[str, Any]:
        return {
            "supply_ema": list(self._supply_ema),
            "conditional_accept_ema": list(self._accept_ema),
            "accept_update_counts": list(self._accept_updates),
            "accept_sample_counts": list(self._accept_samples),
        }


class BatchSizeCostTable:
    """CUDA-Graph ceiling-BS/nearest-context scheduler-cycle cost lookup."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int, int], float] = {}

    def set(
        self, *, batch_size: int, steps: int, context_len: int, cost_ms: float
    ) -> None:
        key = (int(batch_size), int(steps), int(context_len))
        value = float(cost_ms)
        if (
            key[0] <= 0
            or key[1] < 0
            or key[2] <= 0
            or value <= 0
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Invalid verifier cost point: key={key}, cost_ms={value}."
            )
        self._data[key] = value

    def lookup(
        self, *, batch_size: int, steps: int, context_len: int
    ) -> tuple[float | None, int | None, int | None]:
        batch_sizes = sorted({bs for bs, step, _ in self._data if step == int(steps)})
        if not batch_sizes:
            return None, None, None
        bs_index = bisect.bisect_left(batch_sizes, int(batch_size))
        matched_bs = batch_sizes[min(bs_index, len(batch_sizes) - 1)]
        context_lens = sorted(
            ctx
            for bs, step, ctx in self._data
            if bs == matched_bs and step == int(steps)
        )
        ctx_index = bisect.bisect_left(context_lens, int(context_len))
        if ctx_index == 0:
            matched_ctx = context_lens[0]
        elif ctx_index == len(context_lens):
            matched_ctx = context_lens[-1]
        else:
            lower, upper = context_lens[ctx_index - 1 : ctx_index + 1]
            matched_ctx = (
                lower if int(context_len) - lower <= upper - int(context_len) else upper
            )
        return (
            self._data[(matched_bs, int(steps), matched_ctx)],
            matched_bs,
            matched_ctx,
        )

    def items(self) -> list[tuple[int, int, int, float]]:
        return [(*key, value) for key, value in sorted(self._data.items())]


def score_verify_candidates(
    *,
    tracker: DraftPositionStatsTracker,
    cost_table: BatchSizeCostTable,
    candidate_steps: list[int],
    batch_size: int,
    context_len: int,
) -> list[dict[str, Any]]:
    rows = []
    for steps in candidate_steps:
        expected = tracker.expected_accept_length(steps)
        cost_ms, matched_bs, matched_ctx = cost_table.lookup(
            batch_size=batch_size, steps=steps, context_len=context_len
        )
        modeled_tps = (
            int(batch_size) * expected * 1000.0 / cost_ms
            if expected is not None and cost_ms is not None
            else None
        )
        rows.append(
            {
                "steps": int(steps),
                "expected_accept_length": expected,
                "cost_ms": cost_ms,
                "modeled_tps": modeled_tps,
                "matched_batch_size": matched_bs,
                "matched_context_len": matched_ctx,
            }
        )
    return rows


def pick_step_with_hysteresis(
    rows: list[dict[str, Any]], *, current_steps: int, hysteresis: float
) -> int:
    valid_rows = [row for row in rows if row.get("modeled_tps") is not None]
    if not valid_rows:
        return int(current_steps)
    best = min(
        valid_rows, key=lambda row: (-float(row["modeled_tps"]), int(row["steps"]))
    )
    current = next(
        (row for row in valid_rows if int(row["steps"]) == int(current_steps)), None
    )
    if current is not None and float(best["modeled_tps"]) <= float(
        current["modeled_tps"]
    ) * (1.0 + max(0.0, float(hysteresis))):
        return int(current_steps)
    return int(best["steps"])


__all__ = [
    "BatchSizeCostTable",
    "DraftPositionStatsTracker",
    "pick_step_with_hysteresis",
    "score_verify_candidates",
]

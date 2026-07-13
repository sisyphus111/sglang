"""Pure acceptance and cost model for decoupled verifier step selection."""

from __future__ import annotations

import bisect
import math
from typing import Optional

DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA = 0.2
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES = 10
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_UPDATE_INTERVAL = 5
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS = 0.05
DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS = 2.0


def _with_runtime_cpu_overhead_ms(profile_cost_ms: Optional[float]) -> Optional[float]:
    if profile_cost_ms is None:
        return None
    return float(profile_cost_ms) + DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS


class DraftPositionStatsTracker:
    """Tracks global per-position draft supply and conditional accept EMAs."""

    def __init__(
        self,
        *,
        max_steps: int,
        ema_alpha: float = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA,
        warmup_batches: int = DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES,
    ):
        self.max_steps = int(max_steps)
        self.ema_alpha = float(ema_alpha)
        self.warmup_batches = int(warmup_batches)
        if self.max_steps < 0:
            raise ValueError(f"max_steps must be non-negative, got {max_steps}.")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {self.ema_alpha}."
            )
        if self.warmup_batches <= 0:
            raise ValueError(
                f"warmup_batches must be positive, got {self.warmup_batches}."
            )
        self._supply_ema_rates: list[Optional[float]] = [None] * self.max_steps
        self._accept_ema_rates: list[Optional[float]] = [None] * self.max_steps
        self._accept_update_counts: list[int] = [0] * self.max_steps
        self._accept_sample_counts: list[int] = [0] * self.max_steps

    def update(
        self,
        num_correct_drafts_per_req: list[int],
        num_consumable_drafts_per_req: list[int],
        verified_steps: int,
    ) -> None:
        if len(num_correct_drafts_per_req) != len(
            num_consumable_drafts_per_req
        ):
            raise ValueError(
                "Correct-draft and consumable-draft observations must have the "
                "same number of requests."
            )
        num_reqs = len(num_correct_drafts_per_req)
        if num_reqs == 0:
            return

        steps = int(verified_steps)
        if not 0 <= steps <= self.max_steps:
            raise ValueError(
                f"verified_steps must be in [0, {self.max_steps}], got {steps}."
            )
        if any(int(value) < 0 for value in num_consumable_drafts_per_req):
            raise ValueError("Consumable-draft counts must be non-negative.")
        num_consumable = [
            min(int(value), self.max_steps)
            for value in num_consumable_drafts_per_req
        ]
        num_correct = [int(value) for value in num_correct_drafts_per_req]
        for correct, consumable in zip(num_correct, num_consumable):
            if correct < 0 or correct > steps or correct > consumable:
                raise ValueError(
                    "Each correct-draft count must be non-negative and no larger "
                    "than both verified_steps and the matching snapshot supply, "
                    f"got correct={correct}, consumable={consumable}, "
                    f"verified_steps={steps}."
                )

        for position in range(self.max_steps):
            num_supplied = sum(value > position for value in num_consumable)
            batch_supply_rate = num_supplied / num_reqs
            self._supply_ema_rates[position] = self._update_ema(
                self._supply_ema_rates[position], batch_supply_rate
            )

            if position >= steps or num_supplied == 0:
                continue
            batch_accept_rate = (
                sum(value > position for value in num_correct) / num_supplied
            )
            self._accept_ema_rates[position] = self._update_ema(
                self._accept_ema_rates[position], batch_accept_rate
            )
            self._accept_update_counts[position] += 1
            self._accept_sample_counts[position] += num_supplied

    def _update_ema(self, old_value: Optional[float], batch_value: float) -> float:
        return (
            float(batch_value)
            if old_value is None
            else (1.0 - self.ema_alpha) * old_value
            + self.ema_alpha * float(batch_value)
        )

    def all_positions_warmed(self, target_steps: int) -> bool:
        target_steps = min(max(0, int(target_steps)), self.max_steps)
        return all(
            self._accept_update_counts[position] >= self.warmup_batches
            for position in range(target_steps)
        )

    def get_expected_tokens(self, target_steps: int) -> Optional[float]:
        target_steps = int(target_steps)
        if target_steps < 0:
            return None
        if target_steps == 0:
            return 1.0

        expected = 1.0
        for position in range(target_steps):
            supply_rate = self.supply_rate(position)
            accept_rate = self.accept_rate(position)
            if supply_rate is None or accept_rate is None:
                return None
            expected += supply_rate * accept_rate
        return expected

    def get_optimistic_expected_tokens(self, target_steps: int) -> Optional[float]:
        target_steps = int(target_steps)
        if target_steps < 0:
            return None
        expected = 1.0
        for position in range(target_steps):
            supply_rate = self.supply_rate(position)
            if supply_rate is None:
                return None
            accept_rate = self.accept_rate(position)
            expected += supply_rate * (1.0 if accept_rate is None else accept_rate)
        return expected

    def supply_rate(self, position: int) -> Optional[float]:
        if not 0 <= int(position) < self.max_steps:
            return None
        return self._supply_ema_rates[int(position)]

    def accept_rate(self, position: int) -> Optional[float]:
        if not 0 <= int(position) < self.max_steps:
            return None
        return self._accept_ema_rates[int(position)]

    def accept_update_count(self, position: int) -> int:
        if not 0 <= int(position) < self.max_steps:
            return 0
        return self._accept_update_counts[int(position)]

    def accept_sample_count(self, position: int) -> int:
        if not 0 <= int(position) < self.max_steps:
            return 0
        return self._accept_sample_counts[int(position)]

    def position_source(self, position: int) -> str:
        if self.accept_rate(position) is None:
            return "unobserved"
        return (
            "ema"
            if self.accept_update_count(position) >= self.warmup_batches
            else "warming"
        )


class BatchSizeCostTable:
    """Stores verifier decode cost in milliseconds for (batch_size, steps, ctx)."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, int, int], float] = {}
        self._batch_sizes_by_step: dict[int, list[int]] = {}
        self._ctx_lens_by_step_bs: dict[tuple[int, int], list[int]] = {}

    def set(self, batch_size: int, steps: int, ctx_len: int, cost_ms: float) -> None:
        batch_size = int(batch_size)
        steps = int(steps)
        ctx_len = int(ctx_len)
        cost_ms = float(cost_ms)
        if (
            batch_size <= 0
            or steps < 0
            or ctx_len <= 0
            or not math.isfinite(cost_ms)
            or cost_ms <= 0
        ):
            raise ValueError(
                "Cost table entries require positive batch size, non-negative "
                "steps, positive ctx_len, and positive finite cost, got "
                f"bs={batch_size}, steps={steps}, ctx_len={ctx_len}, "
                f"cost_ms={cost_ms}."
            )

        self._data[(batch_size, steps, ctx_len)] = cost_ms
        self._batch_sizes_by_step[steps] = sorted(
            set(self._batch_sizes_by_step.get(steps, [])) | {batch_size}
        )
        self._ctx_lens_by_step_bs[(steps, batch_size)] = sorted(
            set(self._ctx_lens_by_step_bs.get((steps, batch_size), [])) | {ctx_len}
        )

    def lookup(self, batch_size: int, steps: int, ctx_len: int) -> Optional[float]:
        cost_ms, _, _ = self.lookup_with_match(batch_size, steps, ctx_len)
        return cost_ms

    def lookup_with_match(
        self, batch_size: int, steps: int, ctx_len: int
    ) -> tuple[Optional[float], Optional[int], Optional[int]]:
        steps = int(steps)
        batch_sizes = self._batch_sizes_by_step.get(int(steps))
        if not batch_sizes:
            return None, None, None
        index = bisect.bisect_left(batch_sizes, int(batch_size))
        if index >= len(batch_sizes):
            index = len(batch_sizes) - 1
        matched_batch_size = batch_sizes[index]

        ctx_lens = self._ctx_lens_by_step_bs.get((steps, matched_batch_size))
        if not ctx_lens:
            return None, matched_batch_size, None
        ctx_index = bisect.bisect_left(ctx_lens, int(ctx_len))
        if ctx_index <= 0:
            matched_ctx_len = ctx_lens[0]
        elif ctx_index >= len(ctx_lens):
            matched_ctx_len = ctx_lens[-1]
        else:
            lower = ctx_lens[ctx_index - 1]
            upper = ctx_lens[ctx_index]
            matched_ctx_len = (
                lower if int(ctx_len) - lower <= upper - int(ctx_len) else upper
            )
        return (
            self._data.get((matched_batch_size, steps, matched_ctx_len)),
            matched_batch_size,
            matched_ctx_len,
        )

    def has_exact(self, batch_size: int, steps: int, ctx_len: int) -> bool:
        return (int(batch_size), int(steps), int(ctx_len)) in self._data

    def items(self) -> list[tuple[int, int, int, float]]:
        return [
            (batch_size, steps, ctx_len, cost_ms)
            for (batch_size, steps, ctx_len), cost_ms in sorted(self._data.items())
        ]

    def is_empty(self) -> bool:
        return not self._data

    def summary(self) -> str:
        if not self._data:
            return "{}"
        return (
            "{"
            + ", ".join(
                f"(bs={bs}, steps={steps}, ctx={ctx_len}): {cost:.4f}ms"
                for (bs, steps, ctx_len), cost in sorted(self._data.items())
            )
            + "}"
        )


def score_decoupled_verify_candidates(
    tracker: DraftPositionStatsTracker,
    cost_table: BatchSizeCostTable,
    candidate_steps: list[int],
    batch_size: int,
    ctx_len: int,
) -> list[dict]:
    rows = []
    ctx_len = max(1, int(ctx_len))
    for raw_steps in candidate_steps:
        steps = int(raw_steps)
        position_supply_rates = [
            tracker.supply_rate(position) for position in range(steps)
        ]
        position_accept_rates = [
            tracker.accept_rate(position) for position in range(steps)
        ]
        expected = tracker.get_expected_tokens(steps)
        profile_cost_ms, matched_batch_size, matched_ctx_len = (
            cost_table.lookup_with_match(batch_size, steps, ctx_len)
        )
        cost_ms = _with_runtime_cpu_overhead_ms(profile_cost_ms)
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
                "profile_cost_ms": profile_cost_ms,
                "runtime_cpu_overhead_ms": (
                    DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS
                    if profile_cost_ms is not None
                    else None
                ),
                "ctx_len": ctx_len,
                "matched_batch_size": matched_batch_size,
                "matched_ctx_len": matched_ctx_len,
                "score": score,
                "eligible": steps == 0 or tracker.all_positions_warmed(steps),
                "expected_source": (
                    "fixed"
                    if steps == 0
                    else "global_position_ema"
                    if expected is not None
                    else "unobserved"
                ),
                "position_supply_rates": position_supply_rates,
                "position_accept_rates": position_accept_rates,
                "position_accept_sources": [
                    tracker.position_source(position)
                    for position in range(len(position_accept_rates))
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
        eligibility = "" if row.get("eligible", True) else ":blocked"
        expected = row["expected"]
        cost_ms = row["cost_ms"]
        profile_cost_ms = row.get("profile_cost_ms")
        runtime_cpu_overhead_ms = row.get("runtime_cpu_overhead_ms")
        score = row["score"]
        expected_source = row.get("expected_source")
        ctx_len = row.get("ctx_len")
        matched_ctx_len = row.get("matched_ctx_len")
        ctx_text = ""
        if ctx_len is not None:
            matched_text = "?" if matched_ctx_len is None else str(int(matched_ctx_len))
            ctx_text = f",ctx={int(ctx_len)}->{matched_text}"
        supply_rates = row.get("position_supply_rates") or []
        accept_rates = row.get("position_accept_rates") or []
        accept_sources = row.get("position_accept_sources") or []
        rate_parts = []
        for position, accept_rate in enumerate(accept_rates):
            supply_rate = (
                supply_rates[position] if position < len(supply_rates) else None
            )
            source = accept_sources[position] if position < len(accept_sources) else "?"
            supply_text = "?" if supply_rate is None else f"{supply_rate:.3f}"
            if accept_rate is None:
                rate_parts.append(
                    f"p{position + 1}=supply:{supply_text}/accept:?:{source}"
                )
            else:
                rate_parts.append(
                    f"p{position + 1}=supply:{supply_text}/"
                    f"accept:{accept_rate:.3f}:{source}"
                )
        rate_text = f",rates=[{','.join(rate_parts)}]" if rate_parts else ""
        expected_source_text = (
            f",expected_source={expected_source}"
            if expected_source is not None
            else ""
        )
        if score is None:
            parts.append(
                f"S={row['steps']}:score=?{ctx_text}{rate_text}"
                f"{expected_source_text}{eligibility}{marker}"
            )
        else:
            cost_source_text = ""
            if profile_cost_ms is not None and runtime_cpu_overhead_ms is not None:
                cost_source_text = (
                    f",profile_cost={profile_cost_ms:.4f}ms"
                    f",cpu_overhead={runtime_cpu_overhead_ms:.4f}ms"
                )
            parts.append(
                f"S={row['steps']}:E={expected:.3f}/cost={cost_ms:.4f}ms"
                f"={score:.6f}{cost_source_text}{ctx_text}{rate_text}"
                f"{expected_source_text}{eligibility}{marker}"
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

"""Throughput-aware adaptive controller for decoupled verifier workers."""

from __future__ import annotations

import bisect
import logging
import math
import os
from typing import Optional

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveSpecWorker,
    _SpecAdaptiveBase,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

DEFAULT_DECOUPLED_VERIFY_TP_AWARE_EMA_ALPHA = 0.2
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_WARMUP_BATCHES = 10
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_UPDATE_INTERVAL = 5
DEFAULT_DECOUPLED_VERIFY_TP_AWARE_SWITCH_HYSTERESIS = 0.05
DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS = 3.0


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


def _ta_debug_enabled() -> bool:
    return os.environ.get("SGLANG_TA_DEBUG") == "1"


def _with_runtime_cpu_overhead_ms(profile_cost_ms: Optional[float]) -> Optional[float]:
    if profile_cost_ms is None:
        return None
    return float(profile_cost_ms) + DECOUPLED_VERIFY_RUNTIME_CPU_OVERHEAD_MS


class PositionAcceptanceTracker:
    """Tracks per-draft-position acceptance rates with an EMA.

    A position is observed only while the active verify width reaches it.  The
    first unseen position receives a conservative geometric estimate so the
    controller can probe one position deeper; positions beyond that remain
    unscored until the probe supplies real verifier evidence.
    """

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
        self._ema_rates: list[Optional[float]] = [None] * self.max_steps
        self._update_counts: list[int] = [0] * self.max_steps
        self._projected: list[bool] = [False] * self.max_steps

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
            batch_rate = accepted / num_reqs
            old_rate = self._ema_rates[position]
            self._ema_rates[position] = (
                batch_rate
                if old_rate is None
                else (1.0 - self.ema_alpha) * old_rate
                + self.ema_alpha * batch_rate
            )
            self._update_counts[position] += 1
            self._projected[position] = False

        # Higher positions may be stale after a step reduction.  Acceptance is
        # cumulative by construction (p[k + 1] <= p[k]), so conservatively
        # clamp preserved estimates whenever a newly updated lower position
        # drops below them.
        invalidate_higher = False
        for position in range(1, self.max_steps):
            previous_rate = self._ema_rates[position - 1]
            rate = self._ema_rates[position]
            if previous_rate is None:
                break
            if rate is not None and rate > previous_rate:
                self._ema_rates[position] = previous_rate
                invalidate_higher = True
            if invalidate_higher and rate is not None:
                self._update_counts[position] = 0
                self._projected[position] = True

    def all_positions_warmed(self, target_steps: int) -> bool:
        target_steps = min(max(0, int(target_steps)), self.max_steps)
        return all(
            not self._projected[position]
            and self._update_counts[position] >= self.warmup_batches
            for position in range(target_steps)
        )

    def observed_depth(self) -> int:
        depth = 0
        while depth < self.max_steps and self._ema_rates[depth] is not None:
            depth += 1
        return depth

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
        return (
            0 <= int(position) < self.max_steps
            and self._ema_rates[int(position)] is None
        )

    def position_update_count(self, position: int) -> int:
        if not 0 <= int(position) < self.max_steps:
            return 0
        return self._update_counts[int(position)]

    def position_source(self, position: int) -> str:
        if not 0 <= int(position) < self.max_steps:
            return "unknown"
        position = int(position)
        if self._ema_rates[position] is None:
            return "probe"
        return "projected" if self._projected[position] else "ema"

    def _ema_rate(self, position: int) -> Optional[float]:
        if position >= self.max_steps:
            return None
        return self._ema_rates[position]

    def _rate_or_extrapolate(
        self, position: int, known_rates: list[float]
    ) -> Optional[float]:
        rate = self._ema_rate(position)
        if rate is not None:
            return rate
        # At most one unseen position can be estimated.  This prevents the old
        # p2=p3=p1 cold-start path from jumping directly from DL1 to DL3.
        if not known_rates or position > self.observed_depth():
            return None
        if len(known_rates) == 1:
            return known_rates[0] * known_rates[0]

        previous_rate = known_rates[-2]
        last_rate = known_rates[-1]
        if previous_rate <= 0:
            return 0.0
        decay = last_rate / previous_rate
        return max(0.0, min(last_rate, last_rate * decay))


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
    tracker: PositionAcceptanceTracker,
    cost_table: BatchSizeCostTable,
    candidate_steps: list[int],
    batch_size: int,
    ctx_len: int,
) -> list[dict]:
    rows = []
    ctx_len = max(1, int(ctx_len))
    for raw_steps in candidate_steps:
        steps = int(raw_steps)
        position_accept_rates = tracker.snapshot_position_rates(steps)
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
                "position_accept_rates": position_accept_rates,
                "position_accept_sources": [
                    "unknown" if rate is None else tracker.position_source(position)
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
        eligibility = "" if row.get("eligible", True) else ":blocked"
        expected = row["expected"]
        cost_ms = row["cost_ms"]
        profile_cost_ms = row.get("profile_cost_ms")
        runtime_cpu_overhead_ms = row.get("runtime_cpu_overhead_ms")
        score = row["score"]
        expected_source = row.get("expected_source")
        tier_update_count = row.get("tier_update_count")
        tier_age_batches = row.get("tier_age_batches")
        ctx_len = row.get("ctx_len")
        matched_ctx_len = row.get("matched_ctx_len")
        ctx_text = ""
        if ctx_len is not None:
            matched_text = "?" if matched_ctx_len is None else str(int(matched_ctx_len))
            ctx_text = f",ctx={int(ctx_len)}->{matched_text}"
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
        expected_source_text = (
            f",expected_source={expected_source},tier_updates={tier_update_count}"
            + (
                f",tier_age={tier_age_batches}"
                if tier_age_batches is not None
                else ""
            )
            if expected_source is not None and tier_update_count is not None
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


class DecoupledVerifyThroughputAwareController(_SpecAdaptiveBase):
    """Select decoupled verifier steps from accept rates and profiled cost."""

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
        self._ema_alpha = float(ema_alpha)
        self._warmup_batches = int(warmup_batches)
        self._batch_count = 0
        self._cuda_graph_bs: list[int] | None = None
        # With partial draft tails, changing verify width also changes draft
        # availability.  Preserve each tier's measured output instead of using
        # one tier's position rates as a counterfactual estimate for another.
        self._tier_expected_tokens_ema: dict[tuple[int, int], float] = {}
        self._tier_update_counts: dict[tuple[int, int], int] = {}
        self._tier_last_update_batch: dict[tuple[int, int], int] = {}
        self._tracker = PositionAcceptanceTracker(
            max_steps=max(self._candidate_steps),
            ema_alpha=ema_alpha,
            warmup_batches=warmup_batches,
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
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if not num_correct_drafts_per_req:
            return
        self._batch_count += 1
        self._tracker.update(num_correct_drafts_per_req, self._current_steps)
        slot = self._batch_slot(batch_size)
        key = (slot, self._current_steps)
        batch_expected_tokens = 1.0 + sum(num_correct_drafts_per_req) / len(
            num_correct_drafts_per_req
        )
        old_expected_tokens = self._tier_expected_tokens_ema.get(key)
        self._tier_expected_tokens_ema[key] = (
            batch_expected_tokens
            if old_expected_tokens is None
            else (1.0 - self._ema_alpha) * old_expected_tokens
            + self._ema_alpha * batch_expected_tokens
        )
        self._tier_update_counts[key] = self._tier_update_counts.get(key, 0) + 1
        self._tier_last_update_batch[key] = self._batch_count

    def activate_step_by_batch(self, batch_size: int, ctx_len: int) -> None:
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
        if self._current_steps == 0 and not self._has_positive_score(
            batch_size, ctx_len
        ):
            best_steps = self._positive_candidate_steps()[0]
            rows = []
            switch_reason = "positive_probe"
        else:
            rows = score_decoupled_verify_candidates(
                self._tracker,
                self._cost_table,
                self._candidate_steps,
                batch_size,
                ctx_len,
            )
            slot = self._batch_slot(batch_size)
            for row in rows:
                steps = int(row["steps"])
                key = (slot, steps)
                update_count = self._tier_update_counts.get(key, 0)
                row["tier_update_count"] = update_count
                row["tier_age_batches"] = None
                if steps == 0:
                    row["expected"] = 1.0
                    row["expected_source"] = "fixed"
                elif update_count >= self._warmup_batches:
                    tier_expected = self._tier_expected_tokens_ema[key]
                    tier_age_batches = max(
                        0,
                        self._batch_count - self._tier_last_update_batch[key],
                    )
                    row["tier_age_batches"] = tier_age_batches
                    position_expected = row.get("expected")
                    if position_expected is None or tier_age_batches == 0:
                        row["expected"] = tier_expected
                        row["expected_source"] = "tier_ema"
                    else:
                        # A width-specific sample is valuable for partial draft
                        # tails, but it should not stay authoritative while the
                        # workload moves through a different context regime.
                        # Reuse EMA's own decay instead of a separate hard age
                        # threshold, and fall back continuously to the current
                        # position estimate.
                        tier_confidence = (1.0 - self._ema_alpha) ** tier_age_batches
                        row["expected"] = (
                            tier_confidence * tier_expected
                            + (1.0 - tier_confidence) * position_expected
                        )
                        row["expected_source"] = "tier_ema_decay"
                else:
                    row["expected_source"] = "position_ema"
                cost_ms = row.get("cost_ms")
                expected = row.get("expected")
                row["score"] = (
                    expected / cost_ms
                    if expected is not None and cost_ms is not None and cost_ms > 0
                    else None
                )
                # A new CUDA-graph BS slot should not force every tier to run.
                # The global position EMA is already a counterfactual estimate;
                # let a cold tier collect width-specific samples only after its
                # estimated throughput wins the normal score comparison.
                row["eligible"] = steps == 0 or expected is not None
            eligible_rows = [row for row in rows if row["eligible"]]
            best_steps = pick_best_step_with_hysteresis(
                eligible_rows,
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
                f"avg_ctx_len={ctx_len}, "
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
                f"avg_ctx_len={ctx_len}, "
                f"from_steps={old_steps}, to_steps={best_steps}, "
                f"current_score={_format_optional_score(current_score)}, "
                f"best_score={_format_optional_score(best_score)}, "
                f"best_over_current={_format_optional_score(score_ratio)}, "
                f"switch_hysteresis={self._switch_hysteresis:.4f}, "
                f"reason={switch_reason}",
            )
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

    def _batch_slot(self, batch_size: int) -> int:
        batch_size = max(1, int(batch_size))
        if not self._cuda_graph_bs:
            return batch_size
        index = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[index]
            if index < len(self._cuda_graph_bs)
            else batch_size
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

"""Throughput-aware active-K controller for the decoupled verifier."""

from __future__ import annotations

import bisect
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from sglang.srt.speculative.decoupled_verify_cost_model import (
    BatchSizeCostTable,
    DraftPositionStatsTracker,
    pick_step_with_hysteresis,
    score_verify_candidates,
)

logger = logging.getLogger(__name__)

DEFAULT_EMA_ALPHA = 0.2
DEFAULT_UPDATE_INTERVAL_ROUNDS = 20
DEFAULT_SWITCH_HYSTERESIS = 0.05
MAX_SWITCH_EVENTS = 32


def load_decoupled_verify_adaptive_config(
    path: str | None, *, max_steps: int
) -> dict[str, Any]:
    if path is None:
        raw: dict[str, Any] = {
            "candidate_steps": list(range(int(max_steps) + 1)),
        }
    else:
        value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Decoupled verify adaptive config must be a JSON mapping.")
        raw = value
    obsolete_keys = sorted(
        {"initial_steps", "warmup_updates", "warmup_batches"}.intersection(raw)
    )
    if obsolete_keys:
        raise ValueError(
            "Decoupled verify throughput-aware selection always starts from the "
            "maximum candidate step and has no runtime warmup; remove obsolete "
            f"config keys {obsolete_keys}."
        )

    global_candidates = raw.get("candidate_steps")
    slots: dict[int, dict[str, Any]] = {}
    for key, entry in raw.items():
        if not isinstance(key, str) or not key.isdigit():
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Adaptive BS slot {key} must be a mapping.")
        slots[int(key)] = dict(entry)
    if not slots:
        slots[1] = {"candidate_steps": global_candidates}

    resolved_slots: dict[int, dict[str, Any]] = {}
    candidate_union: set[int] = set()
    for batch_size, slot in sorted(slots.items()):
        candidates = slot.get("candidate_steps", global_candidates)
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f"Adaptive BS slot {batch_size} requires non-empty candidate_steps."
            )
        normalized = sorted({int(step) for step in candidates})
        if normalized != list(range(normalized[0], normalized[-1] + 1)):
            raise ValueError(
                "Decoupled verify candidate_steps must be contiguous, got "
                f"{normalized} for BS {batch_size}."
            )
        if normalized[0] < 0 or normalized[-1] > int(max_steps):
            raise ValueError(
                f"Adaptive candidates {normalized} exceed Kmax={max_steps}."
            )
        resolved_slots[batch_size] = {**raw, **slot, "candidate_steps": normalized}
        candidate_union.update(normalized)

    return {
        "candidate_steps": sorted(candidate_union),
        "slots": resolved_slots,
        "ema_alpha": float(raw.get("ema_alpha", DEFAULT_EMA_ALPHA)),
        "update_interval_rounds": int(
            raw.get(
                "update_interval_rounds",
                raw.get("update_interval", DEFAULT_UPDATE_INTERVAL_ROUNDS),
            )
        ),
        "switch_hysteresis": float(
            raw.get("switch_hysteresis", DEFAULT_SWITCH_HYSTERESIS)
        ),
    }


def resolve_decoupled_verify_candidate_steps(
    path: str | None, *, max_steps: int
) -> list[int]:
    return list(
        load_decoupled_verify_adaptive_config(path, max_steps=max_steps)[
            "candidate_steps"
        ]
    )


class DecoupledVerifyThroughputController:
    """Choose verifier K from real supply/accept EMAs and profiled cycle ITL."""

    def __init__(
        self,
        *,
        max_steps: int,
        config_path: str | None,
        profile: dict[str, Any],
        profile_sha256: str,
    ) -> None:
        self.max_steps = int(max_steps)
        self.config = load_decoupled_verify_adaptive_config(
            config_path, max_steps=self.max_steps
        )
        self.candidate_steps = list(self.config["candidate_steps"])
        self.current_steps = max(self.candidate_steps)
        self.tracker = DraftPositionStatsTracker(
            max_steps=self.max_steps,
            ema_alpha=float(self.config["ema_alpha"]),
        )
        self.cost_table = BatchSizeCostTable()
        for point in profile["points"]:
            self.cost_table.set(
                batch_size=int(point["batch_size"]),
                steps=int(point["step"]),
                context_len=int(point["context_len"]),
                cost_ms=float(point["cost_ms"]),
            )
        missing = sorted(
            set(self.candidate_steps)
            - {steps for _, steps, _, _ in self.cost_table.items()}
        )
        if missing:
            raise ValueError(
                "Decoupled verify profile has no costs for adaptive steps "
                f"{missing}."
            )
        self.profile_sha256 = str(profile_sha256)
        self.profile_status = str(profile["status"])
        self.round_count = 0
        self.decision_seq = 0
        self.reevaluation_count = 0
        self.switch_count = 0
        self._window_reevaluation_count = 0
        self._window_switch_count = 0
        self._step_residency = [0] * (self.max_steps + 1)
        self._window_step_residency = [0] * (self.max_steps + 1)
        self._last_scores: list[dict[str, Any]] = []
        self._last_reason = "initial"
        self._last_decision_latency_ms = 0.0
        self._switch_events: deque[dict[str, Any]] = deque(maxlen=MAX_SWITCH_EVENTS)

    def _slot(self, batch_size: int) -> dict[str, Any]:
        batch_sizes = sorted(self.config["slots"])
        index = bisect.bisect_right(batch_sizes, int(batch_size)) - 1
        return self.config["slots"][batch_sizes[max(0, index)]]

    def observe(
        self,
        *,
        correct_drafts: list[int],
        consumable_drafts: list[int],
        applied_steps: int,
    ) -> None:
        applied_steps = int(applied_steps)
        self.round_count += 1
        self._step_residency[applied_steps] += 1
        self._window_step_residency[applied_steps] += 1
        self.tracker.update(
            correct_drafts=correct_drafts,
            consumable_drafts=consumable_drafts,
            applied_steps=applied_steps,
        )

    def decision_due(self) -> bool:
        interval = max(1, int(self.config["update_interval_rounds"]))
        return self.round_count > 0 and self.round_count % interval == 0

    def decide(self, *, batch_size: int, context_len: int) -> dict[str, Any]:
        started = time.perf_counter()
        slot = self._slot(batch_size)
        candidates = list(slot["candidate_steps"])
        rows = score_verify_candidates(
            tracker=self.tracker,
            cost_table=self.cost_table,
            candidate_steps=candidates,
            batch_size=batch_size,
            context_len=context_len,
        )
        old_steps = self.current_steps
        new_steps = pick_step_with_hysteresis(
            rows,
            current_steps=old_steps,
            hysteresis=float(
                slot.get("switch_hysteresis", self.config["switch_hysteresis"])
            ),
        )
        reason = "score_hysteresis" if new_steps != old_steps else "keep_current"
        self.decision_seq += 1
        self.reevaluation_count += 1
        self._window_reevaluation_count += 1
        self._last_scores = rows
        self._last_reason = reason
        self._last_decision_latency_ms = (time.perf_counter() - started) * 1000.0
        if new_steps != old_steps:
            self.current_steps = new_steps
            self.switch_count += 1
            self._window_switch_count += 1
            event = {
                "decision_seq": self.decision_seq,
                "round_count": self.round_count,
                "old_steps": old_steps,
                "new_steps": new_steps,
                "batch_size": int(batch_size),
                "context_len": int(context_len),
                "reason": reason,
            }
            self._switch_events.append(event)
            logger.info(
                "Decoupled verifier adaptive K switch: %s -> %s, bs=%s, "
                "mean_ctx_len=%s, decision_seq=%s, scores=%s",
                old_steps,
                new_steps,
                batch_size,
                context_len,
                self.decision_seq,
                rows,
            )
        return {
            "decision_seq": self.decision_seq,
            "old_steps": old_steps,
            "new_steps": new_steps,
            "reason": reason,
            "scores": rows,
            "decision_latency_ms": self._last_decision_latency_ms,
        }

    def take_metrics_window(self) -> dict[str, Any]:
        metrics = {
            "active_verify_steps": self.current_steps,
            "max_verify_steps": self.max_steps,
            "decision_seq": self.decision_seq,
            "round_count": self.round_count,
            "reevaluation_count": self.reevaluation_count,
            "switch_count": self.switch_count,
            "window_reevaluation_count": self._window_reevaluation_count,
            "window_switch_count": self._window_switch_count,
            "step_residency": list(self._step_residency),
            "window_step_residency": list(self._window_step_residency),
            "last_reason": self._last_reason,
            "last_decision_latency_ms": self._last_decision_latency_ms,
            "candidate_scores": list(self._last_scores),
            "switch_events": list(self._switch_events),
            "profile_status": self.profile_status,
            "profile_sha256": self.profile_sha256,
            **self.tracker.snapshot(),
        }
        self._window_reevaluation_count = 0
        self._window_switch_count = 0
        self._window_step_residency[:] = [0] * len(self._window_step_residency)
        return metrics


__all__ = [
    "DecoupledVerifyThroughputController",
    "load_decoupled_verify_adaptive_config",
    "resolve_decoupled_verify_candidate_steps",
]

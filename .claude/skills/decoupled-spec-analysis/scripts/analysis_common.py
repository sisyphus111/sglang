#!/usr/bin/env python3
"""Shared parsing and profile lookup for decoupled-spec experiment analysis."""

from __future__ import annotations

import bisect
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


CASE_RE = re.compile(
    r"(?:^|_)(?P<partial>ap[01])_(?P<mode>static|dynamic)_dl(?P<step>\d+)(?:_|$)"
)
TIMESTAMP_RE = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"(?:\s+TP(?P<tp_rank>\d+))?.*?\]"
)
SWITCH_RE = re.compile(
    r"throughput-aware step switch: steps (?P<from_steps>\d+) -> (?P<to_steps>\d+), "
    r"bs=(?P<batch_size>\d+), avg_ctx_len=(?P<ctx_len>\d+), "
    r"batch_count=(?P<batch_count>\d+), current_score=(?P<current_score>[0-9.?]+), "
    r"best_score=(?P<best_score>[0-9.?]+), "
    r"best_over_current=(?P<best_over_current>[0-9.?]+), scores=(?P<scores>.*)$"
)
CANDIDATE_RE = re.compile(
    r"S=(?P<steps>\d+):E=(?P<expected>[0-9.]+)"
    r"/cost=(?P<cost_ms>[0-9.]+)ms=(?P<score>[0-9.]+),"
    r"profile_cost=(?P<profile_cost_ms>[0-9.]+)ms,"
    r"cpu_overhead=(?P<cpu_overhead_ms>[0-9.]+)ms,"
    r"ctx=(?P<ctx_len>\d+)->(?P<matched_ctx_len>\d+)"
    r"(?:,rates=\[(?P<rates>[^]]*)\])?(?P<selected>\*)?"
)
RATE_RE = re.compile(r"p(?P<position>\d+)=(?P<rate>[0-9.]+):(?P<source>\w+)")


@dataclass(frozen=True)
class Case:
    label: str
    log_path: Path
    allow_partial: bool
    dynamic: bool
    max_step: int


@dataclass(frozen=True)
class ProfileMatch:
    cost_ms: float
    batch_size: int
    ctx_len: int


def _number(line: str, pattern: str, cast: type = float) -> Any | None:
    match = re.search(pattern, line)
    return cast(match.group(1)) if match else None


def discover_cases(logs_dir: Path) -> list[Case]:
    """Discover runner-style case logs and ignore profile probes or unrelated logs."""
    cases: list[Case] = []
    for path in sorted(logs_dir.glob("*.log")):
        match = CASE_RE.search(path.stem)
        if not match:
            continue
        cases.append(
            Case(
                label=path.stem,
                log_path=path,
                allow_partial=match.group("partial") == "ap1",
                dynamic=match.group("mode") == "dynamic",
                max_step=int(match.group("step")),
            )
        )
    return cases


class ProfileTable:
    """Mirror BatchSizeCostTable lookup: BS ceil/clamp, then nearest ctx."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.data: dict[tuple[int, int, int], float] = {}
        self.batch_sizes_by_step: dict[int, list[int]] = {}
        self.ctx_lens_by_step_bs: dict[tuple[int, int], list[int]] = {}
        batch_sets: dict[int, set[int]] = {}
        ctx_sets: dict[tuple[int, int], set[int]] = {}
        for row in rows:
            batch_size = int(row["batch_size"])
            steps = int(row["steps"])
            ctx_len = int(row["ctx_len"])
            cost_ms = float(row["cost_ms"])
            self.data[(batch_size, steps, ctx_len)] = cost_ms
            batch_sets.setdefault(steps, set()).add(batch_size)
            ctx_sets.setdefault((steps, batch_size), set()).add(ctx_len)
        self.batch_sizes_by_step = {
            step: sorted(values) for step, values in batch_sets.items()
        }
        self.ctx_lens_by_step_bs = {
            key: sorted(values) for key, values in ctx_sets.items()
        }

    @classmethod
    def load(cls, path: Path) -> "ProfileTable":
        payload = json.loads(path.read_text())
        rows = payload["costs"] if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"profile contains no costs: {path}")
        return cls(rows)

    def rows(self) -> Iterator[dict[str, Any]]:
        for (batch_size, steps, ctx_len), cost_ms in sorted(self.data.items()):
            yield {
                "batch_size": batch_size,
                "steps": steps,
                "ctx_len": ctx_len,
                "cost_ms": cost_ms,
            }

    def lookup(self, batch_size: int, steps: int, ctx_len: float) -> ProfileMatch:
        batch_sizes = self.batch_sizes_by_step.get(int(steps))
        if not batch_sizes:
            raise KeyError(f"profile has no points for steps={steps}")
        index = bisect.bisect_left(batch_sizes, int(batch_size))
        matched_bs = batch_sizes[min(index, len(batch_sizes) - 1)]

        ctx_lens = self.ctx_lens_by_step_bs[(int(steps), matched_bs)]
        target = int(ctx_len)
        index = bisect.bisect_left(ctx_lens, target)
        if index <= 0:
            matched_ctx = ctx_lens[0]
        elif index >= len(ctx_lens):
            matched_ctx = ctx_lens[-1]
        else:
            lower, upper = ctx_lens[index - 1], ctx_lens[index]
            matched_ctx = lower if target - lower <= upper - target else upper
        return ProfileMatch(
            cost_ms=self.data[(matched_bs, int(steps), matched_ctx)],
            batch_size=matched_bs,
            ctx_len=matched_ctx,
        )

    def missing_points(
        self, batch_sizes: Iterable[int], steps: Iterable[int], ctx_lens: Iterable[int]
    ) -> list[tuple[int, int, int]]:
        return [
            (int(bs), int(step), int(ctx))
            for step in steps
            for bs in batch_sizes
            for ctx in ctx_lens
            if (int(bs), int(step), int(ctx)) not in self.data
        ]


def parse_candidate_scores(raw_scores: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in CANDIDATE_RE.finditer(raw_scores):
        rates = [
            {
                "position": int(rate.group("position")),
                "rate": float(rate.group("rate")),
                "source": rate.group("source"),
            }
            for rate in RATE_RE.finditer(match.group("rates") or "")
        ]
        candidates.append(
            {
                "steps": int(match.group("steps")),
                "expected": float(match.group("expected")),
                "cost_ms": float(match.group("cost_ms")),
                "score": float(match.group("score")),
                "profile_cost_ms": float(match.group("profile_cost_ms")),
                "cpu_overhead_ms": float(match.group("cpu_overhead_ms")),
                "ctx_len": int(match.group("ctx_len")),
                "matched_ctx_len": int(match.group("matched_ctx_len")),
                "position_accept_rates": rates,
                "selected": bool(match.group("selected")),
            }
        )
    return candidates


def parse_switches(case: Case) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in case.log_path.read_text(errors="ignore").splitlines():
        match = SWITCH_RE.search(line)
        if not match:
            continue
        row: dict[str, Any] = {
            "label": case.label,
            "allow_partial": case.allow_partial,
            "max_step": case.max_step,
            "from_steps": int(match.group("from_steps")),
            "to_steps": int(match.group("to_steps")),
            "batch_size": int(match.group("batch_size")),
            "avg_ctx_len": int(match.group("ctx_len")),
            "batch_count": int(match.group("batch_count")),
            "current_score": match.group("current_score"),
            "best_score": match.group("best_score"),
            "best_over_current": match.group("best_over_current"),
            "scores": match.group("scores"),
        }
        row["candidate_scores_json"] = json.dumps(
            parse_candidate_scores(row["scores"]), separators=(",", ":")
        )
        rows.append(row)
    return rows


def parse_decode_points(
    case: Case, profile: ProfileTable, runtime_cpu_overhead_ms: float
) -> list[dict[str, Any]]:
    """Normalize scheduler INFO and derive modeled values without double-counting CPU."""
    rows: list[dict[str, Any]] = []
    reconstructed_elapsed_s = 0.0
    first_timestamp: datetime | None = None
    for line in case.log_path.read_text(errors="ignore").splitlines():
        if "Decode batch" not in line:
            continue
        timestamp_match = TIMESTAMP_RE.search(line)
        batch_size = _number(line, r"#running-req:\s*(\d+)", int)
        full_tokens = _number(line, r"#(?:full )?token:\s*(\d+)", int)
        latency_ms = _number(line, r"iter latency \(ms\):\s*([0-9.]+)")
        observed_throughput = _number(
            line, r"gen throughput \(token/s\):\s*([0-9.]+)"
        )
        queue_req = _number(line, r"#queue-req:\s*(\d+)", int)
        if None in (batch_size, full_tokens, latency_ms, observed_throughput, queue_req):
            continue

        timestamp: datetime | None = None
        if timestamp_match:
            timestamp = datetime.strptime(
                timestamp_match.group("time"), "%Y-%m-%d %H:%M:%S"
            )
            first_timestamp = first_timestamp or timestamp
        accept_len = _number(line, r"accept len:\s*([0-9.]+)")
        modeled_throughput = _number(
            line, r"modeled throughput \(token/s\):\s*([0-9.]+)"
        )
        ctx_per_req = float(full_tokens) / max(int(batch_size), 1)
        profile_match = profile.lookup(batch_size, case.max_step, ctx_per_req)

        if accept_len is not None:
            output_tokens = float(batch_size) * accept_len
        else:
            output_tokens = float(observed_throughput) * float(latency_ms) / 1000.0

        if case.dynamic and modeled_throughput is not None and modeled_throughput > 0:
            modeled_itl_ms = output_tokens * 1000.0 / modeled_throughput
            model_source = "scheduler_modeled_throughput"
            # The scheduler value reflects the selected runtime step, while lookup below
            # uses max_step only to expose a reference profile match.
        else:
            modeled_itl_ms = profile_match.cost_ms + runtime_cpu_overhead_ms
            modeled_throughput = output_tokens * 1000.0 / modeled_itl_ms
            model_source = "profile_lookup_plus_overhead"

        rows.append(
            {
                "label": case.label,
                "allow_partial": case.allow_partial,
                "dynamic": case.dynamic,
                "max_step": case.max_step,
                "point_index": len(rows) + 1,
                "timestamp": timestamp.isoformat(sep=" ") if timestamp else "",
                "wall_elapsed_s": (
                    (timestamp - first_timestamp).total_seconds()
                    if timestamp is not None and first_timestamp is not None
                    else ""
                ),
                "elapsed_s": reconstructed_elapsed_s,
                "tp_rank": (
                    int(timestamp_match.group("tp_rank"))
                    if timestamp_match and timestamp_match.group("tp_rank")
                    else 0
                ),
                "batch_size": int(batch_size),
                "full_tokens": int(full_tokens),
                "ctx_per_req": ctx_per_req,
                "observed_itl_ms": float(latency_ms),
                "accept_len": accept_len if accept_len is not None else "",
                "valid_draft_len": _number(line, r"valid draft len:\s*([0-9.]+)") or "",
                "accept_rate": _number(line, r"accept rate:\s*([0-9.]+)") or "",
                "output_tokens": output_tokens,
                "observed_throughput_tok_s": float(observed_throughput),
                "modeled_itl_ms": modeled_itl_ms,
                "modeled_throughput_tok_s": float(modeled_throughput),
                "model_source": model_source,
                "profile_cost_ms": profile_match.cost_ms,
                "runtime_cpu_overhead_ms": runtime_cpu_overhead_ms,
                "matched_profile_batch_size": profile_match.batch_size,
                "matched_profile_ctx_len": profile_match.ctx_len,
                "observed_minus_modeled_itl_ms": float(latency_ms) - modeled_itl_ms,
                "observed_over_modeled_throughput": (
                    float(observed_throughput) / float(modeled_throughput)
                    if modeled_throughput > 0
                    else ""
                ),
                "queue_req": int(queue_req),
            }
        )
        reconstructed_elapsed_s += float(latency_ms) / 1000.0
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    left = (window - 1) // 2
    right = window - left
    return [
        sum(values[max(0, i - left) : min(len(values), i + right)])
        / (min(len(values), i + right) - max(0, i - left))
        for i in range(len(values))
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def case_dict(case: Case) -> dict[str, Any]:
    row = asdict(case)
    row["log_path"] = str(case.log_path)
    return row


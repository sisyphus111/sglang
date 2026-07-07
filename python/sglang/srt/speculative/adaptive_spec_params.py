"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING, Iterable

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DECOUPLED_VERIFY_ROOFLINE_TOKEN_BUDGET = "roofline"
DEFAULT_DECOUPLED_VERIFY_ROOFLINE_BS_CANDIDATES = list(range(64, 513, 64))


def is_decoupled_verify_roofline_budget(value) -> bool:
    return (
        isinstance(value, str)
        and value.strip().lower() == DECOUPLED_VERIFY_ROOFLINE_TOKEN_BUDGET
    )


def normalize_decoupled_verify_roofline_bs_candidates(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "--verifier-roofline-profile-bs-candidates must be a list of "
            "positive integers."
        )
    try:
        candidates = sorted({int(bs) for bs in value if int(bs) > 0})
    except TypeError as exc:
        raise ValueError(
            "--verifier-roofline-profile-bs-candidates must be a list of "
            "positive integers."
        ) from exc
    if not candidates:
        raise ValueError(
            "--verifier-roofline-profile-bs-candidates must contain at least "
            "one positive integer."
        )
    return candidates

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 3, 7],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    is_decoupled_verify = server_args.speculative_algorithm == "DECOUPLED_VERIFY"
    if is_decoupled_verify:
        return None
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3/DECOUPLED_VERIFY are supported)"
        )
    if (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk != 1
    ):
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if server_args.enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if server_args.enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
        )
    if server_args.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if server_args.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def _validate_adaptive_config(cfg: dict) -> tuple[dict, dict[int, dict]]:
    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config must contain at least one integer-string "
            'BS key, e.g. {"1": {"candidate_steps": [1,3,7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def _load_adaptive_config(cfg_path: str | None) -> tuple[dict, dict[int, dict]]:
    """Load and validate adaptive config from a file, or the EAGLE default."""
    if cfg_path is not None:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_ADAPTIVE_CONFIG
    return _validate_adaptive_config(cfg)


def _resolve_adaptive_config(
    cfg_path: str | None = None,
    config: dict | None = None,
) -> tuple[dict, dict[int, dict]]:
    if cfg_path is not None and config is not None:
        raise ValueError("Only one of cfg_path or config can be provided.")
    if config is not None:
        return _validate_adaptive_config(config)
    return _load_adaptive_config(cfg_path)


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
    config: dict | None = None,
) -> list[int]:
    """Union of every BS slot's candidate steps; sizes the runtime buffers."""
    _, bs_entries = _resolve_adaptive_config(cfg_path=cfg_path, config=config)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


def build_decoupled_verify_adaptive_config(
    *,
    max_running_requests: int,
    target_verify_token_budget: int,
    max_speculative_steps: int | None = None,
    cuda_graph_bs: list[int] | None = None,
) -> dict[str, dict]:
    """Build static per-BS verify steps from the decoupled target budget."""
    max_running_requests = int(max_running_requests)
    if max_running_requests <= 0:
        raise ValueError(
            "max_running_requests must be positive for decoupled verifier "
            f"adaptive config, got {max_running_requests}."
        )

    target_verify_token_budget = int(target_verify_token_budget)
    if target_verify_token_budget <= 0:
        raise ValueError(
            "target_verify_token_budget must be positive for decoupled verifier "
            f"adaptive config, got {target_verify_token_budget}."
        )
    if max_speculative_steps is not None:
        max_speculative_steps = int(max_speculative_steps)
        if max_speculative_steps < 0:
            raise ValueError(
                "max_speculative_steps must be non-negative for decoupled verifier "
                f"adaptive config, got {max_speculative_steps}."
            )

    if cuda_graph_bs is None:
        bs_values = [max_running_requests]
    else:
        bs_values = sorted({int(bs) for bs in cuda_graph_bs if int(bs) > 0})
        if not bs_values:
            raise ValueError(
                "cuda_graph_bs must contain at least one positive batch size for "
                "decoupled verifier adaptive config."
            )

    config: dict[str, dict] = {}
    for bs in bs_values:
        if target_verify_token_budget <= bs:
            raise ValueError(
                "decoupled verifier target verify token budget is too small: "
                f"budget={target_verify_token_budget}, cuda_graph_bs={bs}. "
                "Even zero-step verify requires budget > cuda_graph_bs."
            )
        budget_step_cap = max(0, (target_verify_token_budget - 1) // bs - 1)
        max_step = (
            budget_step_cap
            if max_speculative_steps is None
            else min(max_speculative_steps, budget_step_cap)
        )
        config[str(bs)] = {"candidate_steps": [max_step]}

    return config


def build_decoupled_verify_roofline_adaptive_config(
    *,
    max_running_requests: int,
    roofline_bs: int,
    max_speculative_steps: int | None = None,
    cuda_graph_bs: list[int] | None = None,
) -> dict[str, dict]:
    """Build per-BS verify steps from a profiled roofline token batch."""
    max_running_requests = int(max_running_requests)
    if max_running_requests <= 0:
        raise ValueError(
            "max_running_requests must be positive for decoupled verifier "
            f"roofline config, got {max_running_requests}."
        )

    roofline_bs = int(roofline_bs)
    if roofline_bs <= 0:
        raise ValueError(
            f"roofline_bs must be positive for decoupled verifier, got {roofline_bs}."
        )

    if max_speculative_steps is not None:
        max_speculative_steps = int(max_speculative_steps)
        if max_speculative_steps < 0:
            raise ValueError(
                "max_speculative_steps must be non-negative for decoupled verifier "
                f"roofline config, got {max_speculative_steps}."
            )

    if cuda_graph_bs is None:
        bs_values = [max_running_requests]
    else:
        bs_values = sorted({int(bs) for bs in cuda_graph_bs if int(bs) > 0})
        if not bs_values:
            raise ValueError(
                "cuda_graph_bs must contain at least one positive batch size for "
                "decoupled verifier roofline config."
            )

    config: dict[str, dict] = {}
    for bs in bs_values:
        # num of speculative steps when reaching the roofline
        roofline_step_cap = max(0, roofline_bs // bs - 1)
        max_step = (
            roofline_step_cap
            if max_speculative_steps is None
            else min(max_speculative_steps, roofline_step_cap)
        )
        config[str(bs)] = {"candidate_steps": [max_step]}

    return config


def select_decoupled_verify_roofline_bs(
    profile_rows: Iterable[tuple[int, float]],
    plateau_ratio: float = 0.95,
) -> tuple[int, int, float, float]:
    """Return (roofline_bs, peak_bs, peak_throughput, threshold)."""
    if not (0 < plateau_ratio <= 1.0):
        raise ValueError(f"plateau_ratio must be in (0, 1], got {plateau_ratio}.")

    rows = [(int(bs), float(throughput)) for bs, throughput in profile_rows]
    if not rows:
        raise ValueError("Cannot select roofline batch size from an empty profile.")
    for bs, throughput in rows:
        if bs <= 0 or throughput <= 0:
            raise ValueError(
                "Profile rows must contain positive batch sizes and throughputs, "
                f"got bs={bs}, throughput={throughput}."
            )

    peak_bs, peak_throughput = max(rows, key=lambda item: item[1])
    threshold = peak_throughput * plateau_ratio
    roofline_bs = min(bs for bs, throughput in rows if throughput >= threshold)
    return roofline_bs, peak_bs, peak_throughput, threshold


def select_decoupled_verify_roofline_steps_by_bs(
    profile_rows: Iterable[tuple[int, int, float]],
    plateau_ratio: float = 0.95,
) -> tuple[
    dict[int, int], dict[int, dict[str, float | int]], tuple[int, int, float]
]:
    """Select a per-BS step from measured ``(bs, steps, throughput)`` rows."""
    if not (0 < plateau_ratio <= 1.0):
        raise ValueError(f"plateau_ratio must be in (0, 1], got {plateau_ratio}.")

    grouped: dict[int, list[tuple[int, float]]] = {}
    global_rows: list[tuple[int, int, float]] = []
    for raw_bs, raw_steps, raw_throughput in profile_rows:
        bs = int(raw_bs)
        steps = int(raw_steps)
        throughput = float(raw_throughput)
        if bs <= 0 or steps < 0 or throughput <= 0:
            raise ValueError(
                "Profile rows must contain positive batch sizes, non-negative "
                f"steps, and positive throughputs, got bs={bs}, steps={steps}, "
                f"throughput={throughput}."
            )
        grouped.setdefault(bs, []).append((steps, throughput))
        global_rows.append((bs, steps, throughput))

    if not global_rows:
        raise ValueError("Cannot select roofline steps from an empty profile.")

    selected_steps: dict[int, int] = {}
    summaries: dict[int, dict[str, float | int]] = {}
    for bs, points in sorted(grouped.items()):
        peak_step, peak_throughput = max(points, key=lambda item: item[1])
        threshold = peak_throughput * plateau_ratio
        selected_step = min(
            steps for steps, throughput in points if throughput >= threshold
        )
        selected_steps[bs] = selected_step
        summaries[bs] = {
            "selected_step": selected_step,
            "peak_step": peak_step,
            "peak_throughput": peak_throughput,
            "threshold": threshold,
        }

    global_peak = max(global_rows, key=lambda item: item[2])
    return selected_steps, summaries, global_peak


def build_decoupled_verify_profiled_adaptive_config(
    selected_steps_by_bs: dict[int, int],
) -> dict[str, dict]:
    """Build adaptive config from per-BS profiled roofline step limits."""
    if not selected_steps_by_bs:
        raise ValueError("selected_steps_by_bs must contain at least one entry.")

    config: dict[str, dict] = {}
    for raw_bs, raw_selected_step in sorted(selected_steps_by_bs.items()):
        bs = int(raw_bs)
        selected_step = int(raw_selected_step)
        if bs <= 0 or selected_step < 0:
            raise ValueError(
                "selected_steps_by_bs must map positive batch sizes to "
                f"non-negative steps, got bs={bs}, steps={selected_step}."
            )
        config[str(bs)] = {"candidate_steps": list(range(selected_step + 1))}
    return config


def _resolve_server_args_decode_cuda_graph_bs(
    server_args: ServerArgs,
) -> list[int] | None:
    cuda_graph_config = getattr(server_args, "cuda_graph_config", None)
    decode_config = getattr(cuda_graph_config, "decode", None)
    capture_bs = getattr(decode_config, "bs", None)
    if not capture_bs:
        capture_bs = getattr(server_args, "cuda_graph_bs_decode", None)
    if not capture_bs:
        return None

    max_running_requests = int(getattr(server_args, "max_running_requests", 0) or 0)
    if max_running_requests <= 0:
        return sorted({int(bs) for bs in capture_bs if int(bs) > 0})

    raw_bs = [int(bs) for bs in capture_bs if int(bs) > 0]
    bs_values = [bs for bs in raw_bs if bs <= max_running_requests]
    if raw_bs and max(raw_bs) > max_running_requests:
        bs_values.append(max_running_requests)
    return sorted(set(bs_values)) or [max_running_requests]


def resolve_decoupled_verify_roofline_profile_bs_candidates(
    server_args: ServerArgs,
    cuda_graph_bs: list[int] | None = None,
) -> list[int]:
    return (
        normalize_decoupled_verify_roofline_bs_candidates(
            getattr(server_args, "verifier_roofline_profile_bs_candidates", None)
        )
        or list(DEFAULT_DECOUPLED_VERIFY_ROOFLINE_BS_CANDIDATES)
    )


def resolve_decoupled_verify_roofline_capture_bs_candidates(
    server_args: ServerArgs,
    profile_bs_candidates: list[int],
    cuda_graph_bs: list[int] | None = None,
) -> list[int]:
    capture_bs = list(int(bs) for bs in profile_bs_candidates if int(bs) > 0)
    if cuda_graph_bs is None:
        cuda_graph_bs = _resolve_server_args_decode_cuda_graph_bs(server_args)
    if cuda_graph_bs:
        capture_bs.extend(int(bs) for bs in cuda_graph_bs if int(bs) > 0)
    capture_bs = sorted(set(capture_bs))
    if not capture_bs:
        raise ValueError(
            "Decoupled verifier roofline capture bs candidates must contain at "
            "least one positive batch size."
        )
    return capture_bs

def resolve_decoupled_verify_adaptive_config_from_server_args(
    server_args: ServerArgs,
    cuda_graph_bs: list[int] | None = None,
) -> dict[str, dict]:
    if getattr(server_args, "speculative_adaptive_config", None) is not None:
        cfg, _ = _load_adaptive_config(server_args.speculative_adaptive_config)
        return cfg

    if cuda_graph_bs is None:
        cuda_graph_bs = _resolve_server_args_decode_cuda_graph_bs(server_args)

    if is_decoupled_verify_roofline_budget(
        getattr(server_args, "decoupled_spec_target_verify_token_budget", None)
    ):
        adaptive_config = getattr(
            server_args, "_decoupled_verify_roofline_adaptive_config", None
        )
        if adaptive_config is not None:
            return adaptive_config

        roofline_bs = getattr(server_args, "_decoupled_verify_roofline_bs", None)
        if roofline_bs is None:
            profile_candidates = resolve_decoupled_verify_roofline_profile_bs_candidates(
                server_args, cuda_graph_bs=cuda_graph_bs
            )
            roofline_bs = max(profile_candidates)
        return build_decoupled_verify_roofline_adaptive_config(
            max_running_requests=server_args.max_running_requests,
            roofline_bs=roofline_bs,
            max_speculative_steps=getattr(
                server_args,
                "_decoupled_verify_max_speculative_steps",
                getattr(server_args, "speculative_num_steps", None),
            ),
            cuda_graph_bs=cuda_graph_bs,
        )

    return build_decoupled_verify_adaptive_config(
        max_running_requests=server_args.max_running_requests,
        target_verify_token_budget=server_args.decoupled_spec_target_verify_token_budget,
        max_speculative_steps=getattr(
            server_args,
            "_decoupled_verify_max_speculative_steps",
            getattr(server_args, "speculative_num_steps", None),
        ),
        cuda_graph_bs=cuda_graph_bs,
    )


def resolve_decoupled_verify_candidate_steps_from_server_args(
    server_args: ServerArgs,
    cuda_graph_bs: list[int] | None = None,
) -> list[int]:
    config = resolve_decoupled_verify_adaptive_config_from_server_args(
        server_args, cuda_graph_bs=cuda_graph_bs
    )
    return resolve_candidate_steps_from_config(config=config)


class AdaptiveStepSlot:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - num_steps can be selected from different candidate sets on different batch_sizes
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self.ema_accept_len = (
                1 - self.ema_alpha
            ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # EMA ceiling: only caps downward — never blocks step-ups, so the
        # system can explore higher steps and let the EMA catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False


class AdaptiveSpeculativeParams:
    """Routes ``batch_size`` to the correct per-BS slot.

    A slot is a per-BS configuration of adaptive step selection.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
        config: dict | None = None,
    ):
        cfg, bs_entries = _resolve_adaptive_config(cfg_path=cfg_path, config=config)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = AdaptiveStepSlot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        return self._route(batch_size).current_steps

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        """Feed verify results to the matching BS slot's EMA.

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``).
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """Map *batch_size* → pad to CUDA-graph BS → closest slot."""
        return self._slots[
            self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]

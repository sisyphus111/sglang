import bisect
import dataclasses
import json
import logging
import os
import tempfile
import time
from array import array
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    BatchSizeCostTable,
    DecoupledVerifyThroughputAwareController,
    parse_decoupled_verify_throughput_profile_ctx_lens,
)
from sglang.srt.speculative.decoupled_verify_input import (
    build_next_draft_input_stub,
    get_req_tail_token_id,
)
from sglang.srt.speculative.decoupled_verify_state import (
    prepare_decoupled_verify_snapshot,
)
from sglang.srt.utils import get_device_name, log_info_on_rank0

logger = logging.getLogger(__name__)

_THROUGHPUT_PROFILE_WARMUP_ITERS = 5
_THROUGHPUT_PROFILE_MEASURE_ITERS = 1000


class DecoupledVerifyProfilerMixin:
    def _capture_throughput_profile_states(
        self, capture_bs: Optional[List[int]] = None
    ) -> None:
        profile_capture_bs = self._resolve_throughput_profile_capture_bs(capture_bs)
        self._throughput_profile_states_by_step = {}
        self._throughput_profile_capture_bs_by_step = {}
        self._throughput_profile_bs_by_step = {}
        for steps in self._throughput_profile_steps():
            state = self.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=profile_capture_bs,
            )
            capture_bs = self._get_capture_bs(state.target_graph_runner)
            self._throughput_profile_states_by_step[steps] = state
            self._throughput_profile_capture_bs_by_step[steps] = capture_bs
            self._throughput_profile_bs_by_step[steps] = list(capture_bs)

        empty_profile_steps = [
            steps
            for steps, profile_bs in self._throughput_profile_bs_by_step.items()
            if not profile_bs
        ]
        if empty_profile_steps:
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling has no replayable "
                "profile "
                f"batch sizes for steps={empty_profile_steps}; "
                f"capture_bs_by_step={self._throughput_profile_capture_bs_by_step}"
            )

        profile_capture_bs_union = sorted(
            {
                int(bs)
                for capture_bs in self._throughput_profile_capture_bs_by_step.values()
                for bs in capture_bs
            }
        )
        if not profile_capture_bs_union:
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling has no captured "
                "CUDA Graph batch sizes."
            )
        self._throughput_profile_capture_bs = profile_capture_bs_union

    def _resolve_throughput_profile_capture_bs(
        self, capture_bs: Optional[List[int]] = None
    ) -> List[int]:
        if capture_bs is None:
            decode_config = getattr(self.server_args.cuda_graph_config, "decode", None)
            capture_bs = getattr(decode_config, "bs", None)
        if capture_bs is None:
            capture_bs = getattr(self.server_args, "cuda_graph_bs_decode", None)
        resolved = sorted({int(bs) for bs in capture_bs or [] if int(bs) > 0})
        if not resolved:
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling requires captured "
                "decode CUDA Graph batch sizes."
            )
        return resolved

    def _throughput_profile_padded_graph_bs(
        self, raw_bs: int, capture_bs: List[int]
    ) -> int:
        sorted_capture_bs = sorted(int(bs) for bs in capture_bs)
        index = bisect.bisect_left(sorted_capture_bs, int(raw_bs))
        if index >= len(sorted_capture_bs):
            raise RuntimeError(
                "Decoupled verifier throughput-aware profile batch size cannot be "
                f"replayed by captured CUDA Graphs: raw_bs={raw_bs}, "
                f"capture_bs={capture_bs}"
            )
        return sorted_capture_bs[index]

    def _throughput_profile_steps(self) -> List[int]:
        return list(self.adaptive_config.candidate_steps)

    def _throughput_profile_cache_path(self) -> Optional[str]:
        path = getattr(
            self.server_args,
            "decoupled_verify_throughput_profile_path",
            None,
        )
        return None if path is None else str(path)

    def _throughput_profile_fingerprint(self) -> dict:
        target_model_path = str(getattr(self.server_args, "model_path", ""))
        target_dp_size = int(getattr(self.server_args, "dp_size", 1) or 1)
        enable_dp_attention = bool(
            getattr(self.server_args, "enable_dp_attention", False)
        )
        engine_tp_size = int(getattr(self.server_args, "tp_size", 1) or 1)
        target_tp_size = (
            engine_tp_size // max(1, target_dp_size)
            if enable_dp_attention
            else engine_tp_size
        )
        return {
            "target_model_path": target_model_path,
            "target_tp_size": target_tp_size,
            "target_dp_size": target_dp_size,
            "enable_dp_attention": enable_dp_attention,
            "gpu_name": str(get_device_name(getattr(self, "gpu_id", 0)) or ""),
        }

    def _throughput_profile_required_points(
        self,
        profile_bs_by_step: dict[int, List[int]],
        profile_ctx_lens: List[int],
    ) -> list[tuple[int, int, int]]:
        return [
            (int(bs), int(steps), int(ctx_len))
            for steps in sorted(profile_bs_by_step)
            for bs in profile_bs_by_step.get(steps, [])
            for ctx_len in profile_ctx_lens
        ]

    def _load_throughput_profile_cache(
        self,
        *,
        controller: DecoupledVerifyThroughputAwareController,
        profile_path: str,
        profile_bs_by_step: dict[int, List[int]],
        profile_ctx_lens: List[int],
    ) -> list[tuple[int, int, int]]:
        required_points = self._throughput_profile_required_points(
            profile_bs_by_step,
            profile_ctx_lens,
        )
        if not os.path.exists(profile_path):
            log_info_on_rank0(
                logger,
                "Decoupled verifier throughput-aware profile cache miss: "
                f"path does not exist, path={profile_path}",
            )
            return required_points

        try:
            with open(profile_path) as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"cache payload must be an object, got {type(payload).__name__}"
                )
            expected_fingerprint = self._throughput_profile_fingerprint()
            actual_fingerprint = payload.get("fingerprint")
            if not isinstance(actual_fingerprint, dict):
                raise ValueError("fingerprint must be an object")
            fingerprint_mismatches = {
                key: {
                    "expected": expected_value,
                    "actual": actual_fingerprint.get(key),
                }
                for key, expected_value in expected_fingerprint.items()
                if actual_fingerprint.get(key) != expected_value
            }
            if fingerprint_mismatches:
                log_info_on_rank0(
                    logger,
                    "Decoupled verifier throughput-aware profile cache miss: "
                    f"fingerprint mismatch, mismatches={fingerprint_mismatches}, "
                    f"expected={expected_fingerprint}, "
                    f"actual={actual_fingerprint}, path={profile_path}",
                )
                return required_points

            costs = payload.get("costs")
            if not isinstance(costs, list):
                raise ValueError("costs must be a list")

            cost_table = BatchSizeCostTable()
            skipped_invalid_entries = 0
            skipped_reasons: list[str] = []
            for index, entry in enumerate(costs):
                try:
                    if not isinstance(entry, dict):
                        raise ValueError(
                            f"cost entry must be an object, got {entry!r}"
                        )
                    cost_table.set(
                        batch_size=entry["batch_size"],
                        steps=entry["steps"],
                        ctx_len=entry["ctx_len"],
                        cost_ms=entry["cost_ms"],
                    )
                except Exception as exc:
                    skipped_invalid_entries += 1
                    if len(skipped_reasons) < 3:
                        skipped_reasons.append(f"index={index}: {exc}")

            missing = [
                point
                for point in required_points
                if not cost_table.has_exact(
                    batch_size=point[0],
                    steps=point[1],
                    ctx_len=point[2],
                )
            ]

            for batch_size, steps, ctx_len, cost_ms in cost_table.items():
                controller.set_profile_cost(
                    batch_size=batch_size,
                    steps=steps,
                    ctx_len=ctx_len,
                    cost_ms=cost_ms,
                )
        except Exception as exc:
            log_info_on_rank0(
                logger,
                "Decoupled verifier throughput-aware profile cache miss: "
                f"failed to load {profile_path}: {exc}",
            )
            return required_points

        log_info_on_rank0(
            logger,
            "Loaded decoupled verifier throughput-aware profile data from "
            f"{profile_path}: loaded_points={len(cost_table.items())}, "
            f"missing_points={len(missing)}, "
            f"skipped_invalid_entries={skipped_invalid_entries}, "
            f"invalid_reasons={skipped_reasons}, "
            f"summary={payload.get('summary')!r}, "
            f"cost_table={controller.cost_table_summary()}",
        )
        return missing

    def _write_throughput_profile_cache(
        self,
        *,
        profile_path: str,
        controller: DecoupledVerifyThroughputAwareController,
    ) -> None:
        payload = {
            "summary": controller.cost_table_summary(),
            "fingerprint": self._throughput_profile_fingerprint(),
            "costs": [
                {
                    "batch_size": batch_size,
                    "steps": steps,
                    "ctx_len": ctx_len,
                    "cost_ms": cost_ms,
                }
                for batch_size, steps, ctx_len, cost_ms in controller.cost_table_items()
            ],
        }
        output_dir = os.path.dirname(os.path.abspath(profile_path))
        os.makedirs(output_dir, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".tmp",
                prefix=os.path.basename(profile_path) + ".",
                dir=output_dir,
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, profile_path)
        except Exception:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    def run_startup_spec_profiling(self, tree_cache) -> None:
        if not self._use_throughput_aware_adaptive_verify():
            return
        if self._throughput_profile_done:
            return

        if not self._throughput_profile_states_by_step:
            self._capture_throughput_profile_states()

        profile_capture_bs_by_step = self._throughput_profile_capture_bs_by_step
        profile_bs_by_step = self._throughput_profile_bs_by_step
        profile_capture_bs = self._throughput_profile_capture_bs or sorted(
            {
                int(bs)
                for capture_bs in profile_capture_bs_by_step.values()
                for bs in capture_bs
            }
        )
        if not profile_capture_bs or not profile_capture_bs_by_step:
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling has no captured "
                "CUDA Graph batch sizes."
            )
        profile_ctx_lens = self._throughput_profile_ctx_lens()

        log_info_on_rank0(
            logger,
            "Profile decoupled verifier throughput-aware cost table begin: "
            f"profile_bs_by_step={profile_bs_by_step}, "
            f"profile_ctx_lens={profile_ctx_lens}, "
            f"profile_capture_bs_by_step={profile_capture_bs_by_step}, "
            f"warmup_iters={_THROUGHPUT_PROFILE_WARMUP_ITERS}, "
            f"measure_iters={_THROUGHPUT_PROFILE_MEASURE_ITERS}",
        )

        initial_steps = int(self.speculative_num_steps or 0)
        controller = DecoupledVerifyThroughputAwareController(
            self,
            candidate_steps=sorted(self._throughput_profile_states_by_step),
            initial_steps=initial_steps,
        )

        profile_path = self._throughput_profile_cache_path()
        missing_profile_points = self._throughput_profile_required_points(
            profile_bs_by_step,
            profile_ctx_lens,
        )
        if profile_path is not None:
            missing_profile_points = self._load_throughput_profile_cache(
                controller=controller,
                profile_path=profile_path,
                profile_bs_by_step=profile_bs_by_step,
                profile_ctx_lens=profile_ctx_lens,
            )

        profile_rows: list[tuple[int, int, int, float, float]] = []
        if missing_profile_points:
            for bs, steps, ctx_len in missing_profile_points:
                state = self._throughput_profile_states_by_step[steps]
                capture_bs = profile_capture_bs_by_step.get(steps, [])
                padded_graph_bs = self._throughput_profile_padded_graph_bs(
                    bs, capture_bs
                )
                avg_decode_ms = self._profile_throughput_shape(
                    batch_size=int(bs),
                    steps=int(steps),
                    ctx_len=int(ctx_len),
                    state=state,
                    tree_cache=tree_cache,
                )
                avg_decode_ms = self._max_reduce_profile_ms(avg_decode_ms)
                throughput = int(bs) * (int(steps) + 1) * 1000.0 / avg_decode_ms
                controller.set_profile_cost(
                    batch_size=int(bs),
                    steps=int(steps),
                    ctx_len=int(ctx_len),
                    cost_ms=avg_decode_ms,
                )
                profile_rows.append(
                    (
                        int(bs),
                        int(steps),
                        int(ctx_len),
                        avg_decode_ms,
                        throughput,
                    )
                )
                log_info_on_rank0(
                    logger,
                    "Decoupled verifier throughput-aware profile point: "
                    f"bs={int(bs)}, steps={int(steps)}, "
                    f"ctx_len={int(ctx_len)}, "
                    f"padded_graph_bs={padded_graph_bs}, "
                    f"verify_tokens_per_req={int(steps) + 1}, "
                    f"avg_decode_ms={avg_decode_ms:.4f}, "
                    f"throughput={throughput:.2f} tok/s",
                )

            if profile_path is not None:
                self._write_throughput_profile_cache(
                    profile_path=profile_path,
                    controller=controller,
                )

        for steps, state in self._throughput_profile_states_by_step.items():
            controller.register(state, steps=steps)
        self.adaptive_controller = controller
        controller.init_states(cuda_graph_bs=profile_capture_bs)
        self.server_args._decoupled_verify_throughput_cost_table_summary = (
            controller.cost_table_summary()
        )
        self._throughput_profile_done = True
        log_info_on_rank0(
            logger,
            "Profile decoupled verifier throughput-aware cost table end: "
            f"points={len(profile_rows)}, "
            f"cost_table={controller.cost_table_summary()}",
        )

    def _throughput_max_speculative_steps(self) -> Optional[int]:
        if self.adaptive_config is None:
            return self.speculative_num_steps
        return max(self.adaptive_config.candidate_steps)

    def _profile_throughput_shape(
        self,
        batch_size: int,
        steps: int,
        ctx_len: int,
        state: SpecRuntimeState,
        tree_cache,
    ) -> float:
        if self.device != "cuda":
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling currently "
                "requires CUDA event timing."
            )
        self.apply_runtime_state(state)

        reqs, batch = self._build_throughput_profile_batch(
            batch_size=batch_size,
            seq_len=ctx_len,
            tree_cache=tree_cache,
        )

        try:
            keep_alive_refs = []
            self._prepare_throughput_profile_decode_state(
                batch=batch,
                seq_len=ctx_len,
            )
            for _ in range(_THROUGHPUT_PROFILE_WARMUP_ITERS):
                self._prepare_throughput_profile_draft_snapshots(batch, steps)
                keep_alive_refs.extend(
                    self._run_throughput_profile_decode(batch).extra_keep_alive_refs
                    or []
                )
            torch.cuda.current_stream().synchronize()
            keep_alive_refs.clear()

            events = []
            for _ in range(_THROUGHPUT_PROFILE_MEASURE_ITERS):
                self._prepare_throughput_profile_draft_snapshots(batch, steps)
                batch.prepare_for_decode()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                result = self._run_throughput_profile_forward(batch)
                end_event.record()
                self._apply_throughput_decode_result(batch, result)
                keep_alive_refs.extend(result.extra_keep_alive_refs or [])
                events.append((start_event, end_event))

            torch.cuda.current_stream().synchronize()
            total_ms = sum(start.elapsed_time(end) for start, end in events)
            return total_ms / max(1, _THROUGHPUT_PROFILE_MEASURE_ITERS)
        finally:
            self._teardown_throughput_profile_reqs(reqs, tree_cache)

    def _build_throughput_profile_batch(
        self,
        *,
        batch_size: int,
        seq_len: int,
        tree_cache,
    ) -> tuple[list[Req], ScheduleBatch]:
        profile_seq_len = int(seq_len)
        self._validate_throughput_profile_prompt_len(
            seq_len=profile_seq_len,
            batch_size=batch_size,
        )
        vocab_size = int(getattr(self.model_config, "vocab_size", 32000) or 32000)
        token_mod = max(1, vocab_size - 1)
        max_new_tokens = self._throughput_profile_decode_headroom(extra_iters=4)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            ignore_eos=True,
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        reqs = []
        for i in range(int(batch_size)):
            token_ids = array(
                "q",
                [((i + j) % token_mod) + 1 for j in range(1)],
            )
            req = Req(
                rid=(
                    f"decoupled-throughput-profile-{batch_size}-{profile_seq_len}-"
                    f"{i}-{time.time_ns()}"
                ),
                origin_input_text="",
                origin_input_ids=token_ids,
                sampling_params=sampling_params,
                extra_key=(
                    f"decoupled-throughput-profile-{batch_size}-{profile_seq_len}-"
                    f"{i}-{time.time_ns()}"
                ),
            )
            req.skip_radix_cache_insert = True
            req.init_next_round_input(tree_cache)
            req.fill_len = len(req.full_untruncated_fill_ids)
            reqs.append(req)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_config,
            enable_overlap=False,
            spec_algorithm=self.speculative_algorithm,
        )
        return reqs, batch

    def _prepare_throughput_profile_decode_state(
        self,
        *,
        batch: ScheduleBatch,
        seq_len: int,
    ) -> None:
        """Build a decode-shaped profile batch without profiling full prefill.

        Throughput-aware profile measures verifier decode latency. Running an
        eager full prefill for every (bs, ctx_len) point makes high-bs/high-ctx
        profile points dominated by synthetic setup memory instead of verifier
        decode cost. This initializes the scheduler/KV metadata needed by the
        normal verify path while avoiding that full prefill forward.
        """
        if self.page_size != 1:
            raise RuntimeError(
                "Decoupled verifier throughput-aware synthetic profile state "
                "currently requires page_size == 1."
            )

        seq_len = int(seq_len)
        batch.prepare_for_extend()
        bs = batch.batch_size()
        if bs == 0:
            return

        if seq_len > 1:
            extra_len = seq_len - 1
            extra_locs = alloc_token_slots(batch.tree_cache, bs * extra_len)
            extra_locs = extra_locs.reshape(bs, extra_len).to(torch.int32)
            for req_index, req in enumerate(batch.reqs):
                batch.req_to_token_pool.write(
                    (req.req_pool_idx, slice(1, seq_len)),
                    extra_locs[req_index],
                )

        seq_lens_cpu = torch.full((bs,), seq_len, dtype=torch.int64)
        batch.seq_lens_cpu = seq_lens_cpu
        batch.seq_lens = seq_lens_cpu.to(batch.device, non_blocking=True)
        batch.orig_seq_lens = torch.full(
            (bs,), seq_len, dtype=torch.int32, device=batch.device
        )
        batch.seq_lens_sum = int(seq_len * bs)
        batch.input_ids = None
        batch.prefill_input_ids_cpu = None

        for req in batch.reqs:
            req.fill_len = seq_len
            req.kv_committed_len = seq_len
            req.kv_allocated_len = seq_len

        bonus_tokens = torch.tensor(
            [get_req_tail_token_id(req) for req in batch.reqs],
            dtype=torch.int32,
            device=batch.device,
        )
        batch.spec_info = build_next_draft_input_stub(bonus_tokens, self.topk)

    def _throughput_profile_decode_headroom(self, *, extra_iters: int) -> int:
        max_steps = self._throughput_max_speculative_steps()
        if max_steps is None:
            max_steps = int(self.speculative_num_steps or 0)
        tokens_per_decode = max(1, int(max_steps) + 1)
        return (
            _THROUGHPUT_PROFILE_WARMUP_ITERS
            + _THROUGHPUT_PROFILE_MEASURE_ITERS
            + int(extra_iters)
        ) * tokens_per_decode

    def _throughput_profile_ctx_lens(self) -> List[int]:
        raw = getattr(
            self.server_args,
            "decoupled_verify_throughput_profile_ctx_lens",
            None,
        )
        parsed = parse_decoupled_verify_throughput_profile_ctx_lens(raw)
        if parsed is None:
            return [self._throughput_profile_prompt_len()]
        return parsed

    def _throughput_profile_prompt_limit_details(
        self, *, batch_size: int
    ) -> tuple[int, dict[str, Optional[int]]]:
        batch_size = max(1, int(batch_size))
        context_len = int(getattr(self.model_config, "context_len", 4096) or 4096)
        decode_headroom = self._throughput_profile_decode_headroom(extra_iters=8)
        context_limited_len = context_len - decode_headroom

        available_size = getattr(
            self.token_to_kv_pool_allocator,
            "available_size",
            None,
        )
        available_tokens = int(available_size()) if callable(available_size) else None
        pool_limited_len = (
            int(available_tokens) // batch_size - decode_headroom
            if available_tokens is not None
            else None
        )

        details = {
            "context_limit": context_limited_len,
            "kv_pool_limit": pool_limited_len,
        }
        return min(value for value in details.values() if value is not None), details

    def _validate_throughput_profile_prompt_len(
        self, *, seq_len: int, batch_size: int
    ) -> None:
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise RuntimeError(
                "Decoupled verifier throughput-aware profile ctx_len must be "
                f"positive, got ctx_len={seq_len}."
            )
        max_profile_len, details = self._throughput_profile_prompt_limit_details(
            batch_size=batch_size
        )
        if seq_len <= max_profile_len:
            return
        limits = ", ".join(
            f"{name}={value}" for name, value in details.items() if value is not None
        )
        raise RuntimeError(
            "Decoupled verifier throughput-aware profile ctx_len exceeds the "
            "safe profiling range: "
            f"ctx_len={seq_len}, batch_size={int(batch_size)}, "
            f"max_profile_ctx_len={max_profile_len}, {limits}."
        )

    def _throughput_profile_prompt_len(self) -> int:
        max_profile_bs = max(self._throughput_profile_capture_bs or [1])
        max_profile_len, details = self._throughput_profile_prompt_limit_details(
            batch_size=max_profile_bs
        )
        if max_profile_len <= 0:
            limits = ", ".join(
                f"{name}={value}"
                for name, value in details.items()
                if value is not None
            )
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling cannot build a "
                "dummy batch with decode headroom: "
                f"batch_size={int(max_profile_bs)}, {limits}."
            )
        return max(1, min(256, max_profile_len))

    def _run_throughput_profile_prefill(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        batch.prepare_for_extend()
        if batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        result = self._run_throughput_profile_forward(batch)
        self._apply_throughput_prefill_result(batch, result)
        return result

    def _run_throughput_profile_decode(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        batch.prepare_for_decode()
        result = self._run_throughput_profile_forward(batch)
        self._apply_throughput_decode_result(batch, result)
        return result

    def _prepare_throughput_profile_draft_snapshots(
        self, batch: ScheduleBatch, steps: int
    ) -> None:
        steps = int(steps)
        for req_idx, req in enumerate(batch.reqs):
            snapshot = prepare_decoupled_verify_snapshot(req, len(req.output_ids))
            if steps <= 0:
                continue
            snapshot.draft_tokens.extend(
                self._throughput_dummy_token_id(req_idx, len(req.output_ids) + i)
                for i in range(steps)
            )

    def _run_throughput_profile_forward(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            batch.sampling_info = sampling_info.copy_for_forward()
        try:
            result = self.forward_batch_generation(batch)
            if result.extra_keep_alive_refs is None:
                result.extra_keep_alive_refs = []
            result.extra_keep_alive_refs.append(snapshot)
            return result
        finally:
            for name, value in snapshot.items():
                setattr(batch, name, value)

    def _apply_throughput_prefill_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        batch.spec_info = result.next_draft_input
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
        batch.input_ids = None
        for i, req in enumerate(batch.reqs):
            req.output_ids.append(
                self._throughput_dummy_token_id(i, len(req.output_ids))
            )

    def _throughput_decode_accept_lens(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> List[int]:
        if result.accept_lens is not None:
            return [int(x) for x in result.accept_lens.to("cpu").tolist()]
        if result.new_seq_lens is not None and batch.seq_lens is not None:
            accept_lens = (
                result.new_seq_lens.to(batch.seq_lens.device) - batch.seq_lens
            )
            return [int(x) for x in accept_lens.to("cpu").tolist()]
        if result.num_correct_drafts_per_req_cpu is not None:
            return [int(x) + 1 for x in result.num_correct_drafts_per_req_cpu]
        return [1 for _ in batch.reqs]

    def _apply_throughput_decode_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        if not result.can_run_cuda_graph:
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling expected target "
                "verify CUDA Graph replay, but the measured decode ran eagerly."
            )
        accept_lens = self._throughput_decode_accept_lens(batch, result)
        if len(accept_lens) != len(batch.reqs):
            raise RuntimeError(
                "Decoupled verifier throughput-aware profiling decode result has "
                f"{len(accept_lens)} accept lengths for {len(batch.reqs)} requests."
            )
        batch.spec_info = result.next_draft_input
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = result.new_seq_lens.to("cpu")
        else:
            if batch.seq_lens is not None:
                batch.seq_lens = batch.seq_lens + torch.tensor(
                    accept_lens,
                    dtype=batch.seq_lens.dtype,
                    device=batch.seq_lens.device,
                )
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = batch.seq_lens_cpu + torch.tensor(
                    accept_lens,
                    dtype=batch.seq_lens_cpu.dtype,
                    device=batch.seq_lens_cpu.device,
                )
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        elif batch.seq_lens is not None:
            batch.seq_lens_sum = int(torch.sum(batch.seq_lens).item())
        batch.input_ids = None
        for i, req in enumerate(batch.reqs):
            for _ in range(max(0, int(accept_lens[i]))):
                req.output_ids.append(
                    self._throughput_dummy_token_id(i, len(req.output_ids))
                )

    def _throughput_dummy_token_id(self, req_idx: int, output_idx: int) -> int:
        vocab_size = int(getattr(self.model_config, "vocab_size", 32000) or 32000)
        return ((int(req_idx) + int(output_idx)) % max(1, vocab_size - 1)) + 1

    def _teardown_throughput_profile_reqs(
        self, reqs: list[Req], tree_cache
    ) -> None:
        for req in reqs:
            if req.req_pool_idx is None and getattr(req, "mamba_pool_idx", None) is None:
                continue
            try:
                if len(req.prefix_indices) != 0:
                    raise RuntimeError(
                        "Decoupled verifier throughput-aware profile request "
                        "unexpectedly matched prefix cache; profile teardown "
                        f"cannot safely own prefix KV slots. rid={req.rid}, "
                        f"prefix_len={len(req.prefix_indices)}"
                    )
                if req.req_pool_idx is not None:
                    allocated_len = int(req.kv_allocated_len)
                    if allocated_len > 0:
                        kv_indices = tree_cache.req_to_token_pool.req_to_token[
                            req.req_pool_idx, :allocated_len
                        ]
                        tree_cache.token_to_kv_pool_allocator.free(kv_indices)
                    tree_cache.req_to_token_pool.free(req)
                if getattr(req, "mamba_pool_idx", None) is not None:
                    tree_cache.req_to_token_pool.free_mamba_cache(req)
            except Exception:
                logger.exception(
                    "Failed to release decoupled verifier throughput-aware "
                    f"profile request {req.rid}."
                )

    def _max_reduce_profile_ms(self, avg_ms: float) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return float(avg_ms)
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return float(avg_ms)
        value = torch.tensor([float(avg_ms)], dtype=torch.float32, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=tp_group.device_group)
        return float(value.item())

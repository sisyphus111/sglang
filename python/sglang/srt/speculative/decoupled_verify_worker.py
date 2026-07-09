import bisect
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from array import array
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.server_args import ServerArgs
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
    _SpecAdaptiveBase,
)
from sglang.srt.speculative.adaptive_spec_params import (
    resolve_decoupled_verify_throughput_aware_candidate_steps,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.decoupled_verify_throughput_controller import (
    BatchSizeCostTable,
    DecoupledVerifyThroughputAwareController,
    parse_decoupled_verify_throughput_profile_ctx_lens,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
)
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    dynamic_verify_enabled,
)
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    generate_token_bitmask,
    record_stream_each,
    record_stream_for_v2_verify,
)
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.async_probe import maybe_detect_inf, maybe_detect_nan
from sglang.srt.utils.common import get_available_gpu_memory

logger = logging.getLogger(__name__)

_THROUGHPUT_PROFILE_WARMUP_ITERS = 5
_THROUGHPUT_PROFILE_MEASURE_ITERS = 1000
_THROUGHPUT_PROFILE_CACHE_SCHEMA_VERSION = 2


def _get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external "
        "draft verification."
    )


def _build_bonus_tokens_from_accepts(
    predict: torch.Tensor,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
) -> torch.Tensor:
    accept_tokens = predict[accept_index]
    row_indices = torch.arange(accept_lens.numel(), device=accept_lens.device)
    bonus_indices = torch.clamp(accept_lens.to(torch.long) - 1, min=0)
    return accept_tokens[row_indices, bonus_indices].to(dtype=torch.int32)


def _build_next_draft_input_stub(
    bonus_tokens: torch.Tensor,
    topk: int,
) -> EagleDraftInput:
    bonus_tokens = bonus_tokens.to(dtype=torch.int32)
    batch_size = int(bonus_tokens.numel())
    device = bonus_tokens.device
    return EagleDraftInput(
        bonus_tokens=bonus_tokens,
        topk_p=torch.zeros(
            (batch_size, int(topk)), device=device, dtype=torch.float32
        ),
        topk_index=torch.zeros(
            (batch_size, int(topk)), device=device, dtype=torch.int64
        ),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _normalize_token_id(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = _normalize_token_id(item)
            if normalized is not None:
                return normalized
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_linear_topk1_tree_metadata(
    batch_size: int,
    spec_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected_index = (
        torch.arange(spec_steps, dtype=torch.long, device=device)
        .expand(batch_size, -1)
        .contiguous()
    )

    if spec_steps <= 1:
        parent_list = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    else:
        parent_list = (
            torch.arange(-1, spec_steps - 1, dtype=torch.long, device=device)
            .expand(batch_size, -1)
            .contiguous()
        )

    return selected_index, parent_list


class VerifyWorker(BaseSpecWorker):
    """Spec-v2 worker for target-side decoupled verification.

    Decoupled verifier has no local draft model.  Its ``draft()`` method turns
    scheduler-bound external draft-tail snapshots into an ``EagleVerifyInput``;
    ``verify()`` then follows the EAGLE v2 target-verify/result contract.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self._decoupled_verify_max_speculative_steps = (
            None
            if server_args.speculative_num_steps is None
            else int(server_args.speculative_num_steps)
        )
        if not hasattr(server_args, "_decoupled_verify_max_speculative_steps"):
            server_args._decoupled_verify_max_speculative_steps = (
                self._decoupled_verify_max_speculative_steps
            )
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.pp_rank = getattr(target_worker, "pp_rank", 0)
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_config
        self.enable_adaptive_verify = dynamic_verify_enabled(server_args)
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.total_accept_length = 0
        self.total_num_verified_reqs = 0
        self.adaptive_controller: Optional[_SpecAdaptiveBase] = None
        self._throughput_profile_states_by_step: dict[int, SpecRuntimeState] = {}
        self._throughput_profile_capture_bs_by_step: dict[int, List[int]] = {}
        self._throughput_profile_bs_by_step: dict[int, List[int]] = {}
        self._throughput_profile_capture_bs: Optional[List[int]] = None
        self._throughput_profile_done = False

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase]:
        return None

    def clear_cache_pool(self):
        return

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        return

    def _use_throughput_aware_adaptive_verify(self) -> bool:
        return (
            self.enable_adaptive_verify
            and getattr(self.server_args, "speculative_adaptive_strategy", "ema")
            == "throughput_aware"
        )

    def init_cuda_graphs(self):
        if self.enable_adaptive_verify:
            capture_bs = (
                None
                if self.server_args.disable_cuda_graph
                else self.server_args.cuda_graph_bs_decode
            )
            if capture_bs is None and not self.server_args.disable_cuda_graph:
                capture_bs, _ = get_batch_sizes_to_capture(
                    self.target_worker.model_runner, num_tokens_per_bs=1
                )

            if self._use_throughput_aware_adaptive_verify():
                # Capture profile states here; profiling needs scheduler-owned tree_cache.
                self._capture_throughput_profile_states(capture_bs=capture_bs)
                log_info_on_rank0(
                    logger,
                    "Captured decoupled verifier throughput-aware profile runtime states; "
                    "startup profiling will run after scheduler KV cache init. "
                    f"profile_capture_bs_by_step="
                    f"{self._throughput_profile_capture_bs_by_step}",
                )
                return

            self.adaptive_controller = AdaptiveController(
                self,
                config_path=self.server_args.speculative_adaptive_config,
            )
            log_info_on_rank0(
                logger,
                "Capture decoupled verifier adaptive runtime states begin: "
                f"candidate_steps={self.adaptive_controller.candidate_steps}, "
                f"global_max_steps={self.speculative_num_steps}, "
                f"cuda_graph_bs={capture_bs}",
            )

            model_runner = self.target_worker.model_runner
            initial_graph_runner = model_runner.decode_cuda_graph_runner
            if initial_graph_runner is None:
                raise RuntimeError(
                    "Decoupled verifier adaptive initial state requires the target "
                    "worker's captured decode CUDA Graph."
                )
            initial_state = SpecRuntimeState.for_decoupled_verify(
                speculative_num_steps=self.speculative_num_steps,
                speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                target_attn_backend=model_runner.attn_backend,
                target_graph_runner=initial_graph_runner,
            )
            self._validate_decoupled_runtime_state(initial_state)
            self.adaptive_controller.register(initial_state)
            self.adaptive_controller.init_states(cuda_graph_bs=capture_bs)
            log_info_on_rank0(
                logger,
                "Capture decoupled verifier adaptive runtime states end.",
            )

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
        return resolve_decoupled_verify_throughput_aware_candidate_steps(
            self._throughput_max_speculative_steps()
        )

    def _throughput_profile_cache_path(self) -> Optional[str]:
        path = getattr(
            self.server_args,
            "decoupled_verify_throughput_profile_path",
            None,
        )
        return None if path is None else str(path)

    def _throughput_profile_model_identity(self, path: object) -> tuple[str, str]:
        raw_path = "none" if path is None else str(path)
        name = os.path.basename(raw_path.rstrip(os.sep)) or raw_path
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
        if not slug:
            slug = "model"
        if len(slug) > 48:
            slug = slug[:48].rstrip("-._")
        digest = hashlib.sha256(raw_path.encode("utf-8")).hexdigest()[:12]
        return slug, digest

    def _throughput_profile_capture_hash(self, capture_bs: List[int]) -> str:
        encoded = ",".join(str(int(bs)) for bs in sorted(set(capture_bs)))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def _throughput_profile_ctx_hash(self, profile_ctx_lens: List[int]) -> str:
        encoded = ",".join(str(int(ctx_len)) for ctx_len in sorted(set(profile_ctx_lens)))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def _throughput_profile_fingerprint(
        self, capture_bs: List[int], profile_ctx_lens: List[int]
    ) -> dict:
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
            "max_steps": int(self._throughput_max_speculative_steps() or 0),
            "capture_bs": sorted({int(bs) for bs in capture_bs}),
            "profile_ctx_lens": sorted({int(ctx_len) for ctx_len in profile_ctx_lens}),
        }

    def _expected_throughput_profile_cache_basename(
        self, capture_bs: List[int], profile_ctx_lens: List[int]
    ) -> str:
        fingerprint = self._throughput_profile_fingerprint(
            capture_bs, profile_ctx_lens
        )
        target_slug, target_hash = self._throughput_profile_model_identity(
            fingerprint["target_model_path"]
        )
        capture_hash = self._throughput_profile_capture_hash(
            fingerprint["capture_bs"]
        )
        ctx_hash = self._throughput_profile_ctx_hash(
            fingerprint["profile_ctx_lens"]
        )
        dp_attention = 1 if fingerprint["enable_dp_attention"] else 0
        return (
            "decoupled_verify_throughput"
            f"__target-{target_slug}-{target_hash}"
            f"__targettp-{fingerprint['target_tp_size']}"
            f"__targetdp-{fingerprint['target_dp_size']}"
            f"__dpa-{dp_attention}"
            f"__maxsteps-{fingerprint['max_steps']}"
            f"__cgraph-{capture_hash}"
            f"__ctx-{ctx_hash}.json"
        )

    def _load_throughput_profile_cache(
        self,
        *,
        controller: DecoupledVerifyThroughputAwareController,
        profile_path: str,
        expected_basename: str,
        profile_bs_by_step: dict[int, List[int]],
        profile_ctx_lens: List[int],
    ) -> bool:
        actual_basename = os.path.basename(profile_path)
        if actual_basename != expected_basename:
            log_info_on_rank0(
                logger,
                "Decoupled verifier throughput-aware profile cache miss: "
                f"basename mismatch, expected={expected_basename}, "
                f"actual={actual_basename}, path={profile_path}",
            )
            return False
        if not os.path.exists(profile_path):
            log_info_on_rank0(
                logger,
                "Decoupled verifier throughput-aware profile cache miss: "
                f"path does not exist, path={profile_path}",
            )
            return False

        try:
            with open(profile_path) as f:
                payload = json.load(f)
            if payload.get("schema_version") != _THROUGHPUT_PROFILE_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version={payload.get('schema_version')!r}"
                )
            costs = payload.get("costs")
            if not isinstance(costs, list):
                raise ValueError("costs must be a list")

            cost_table = BatchSizeCostTable()
            for entry in costs:
                if not isinstance(entry, dict):
                    raise ValueError(f"cost entry must be an object, got {entry!r}")
                cost_table.set(
                    batch_size=entry["batch_size"],
                    steps=entry["steps"],
                    ctx_len=entry["ctx_len"],
                    cost_ms=entry["cost_ms"],
                )

            missing = [
                (int(bs), int(steps), int(ctx_len))
                for steps, batch_sizes in profile_bs_by_step.items()
                for bs in batch_sizes
                for ctx_len in profile_ctx_lens
                if not cost_table.has_exact(
                    batch_size=bs, steps=steps, ctx_len=ctx_len
                )
            ]
            if missing:
                raise ValueError(f"missing cost entries: {missing}")

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
            return False

        log_info_on_rank0(
            logger,
            "Loaded decoupled verifier throughput-aware profile data from "
            f"{profile_path}: cost_table={controller.cost_table_summary()}",
        )
        return True

    def _write_throughput_profile_cache(
        self,
        *,
        profile_path: str,
        capture_bs: List[int],
        profile_ctx_lens: List[int],
        controller: DecoupledVerifyThroughputAwareController,
    ) -> None:
        payload = {
            "schema_version": _THROUGHPUT_PROFILE_CACHE_SCHEMA_VERSION,
            "summary": controller.cost_table_summary(),
            "fingerprint": self._throughput_profile_fingerprint(
                capture_bs, profile_ctx_lens
            ),
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
        loaded_from_cache = False
        if profile_path is not None:
            expected_basename = self._expected_throughput_profile_cache_basename(
                profile_capture_bs,
                profile_ctx_lens,
            )
            loaded_from_cache = self._load_throughput_profile_cache(
                controller=controller,
                profile_path=profile_path,
                expected_basename=expected_basename,
                profile_bs_by_step=profile_bs_by_step,
                profile_ctx_lens=profile_ctx_lens,
            )

        profile_rows: list[tuple[int, int, int, float, float]] = []
        if not loaded_from_cache:
            for steps in sorted(self._throughput_profile_states_by_step):
                state = self._throughput_profile_states_by_step[steps]
                capture_bs = profile_capture_bs_by_step.get(steps, [])
                for bs in profile_bs_by_step.get(steps, []):
                    padded_graph_bs = self._throughput_profile_padded_graph_bs(
                        bs, capture_bs
                    )
                    for ctx_len in profile_ctx_lens:
                        avg_decode_ms = self._profile_throughput_shape(
                            batch_size=int(bs),
                            steps=int(steps),
                            ctx_len=int(ctx_len),
                            state=state,
                            tree_cache=tree_cache,
                        )
                        avg_decode_ms = self._max_reduce_profile_ms(avg_decode_ms)
                        throughput = (
                            int(bs) * (int(steps) + 1) * 1000.0 / avg_decode_ms
                        )
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
                    capture_bs=profile_capture_bs,
                    profile_ctx_lens=profile_ctx_lens,
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
        max_steps = getattr(
            self.server_args,
            "_decoupled_verify_max_speculative_steps",
            None,
        )
        if max_steps is None:
            max_steps = getattr(self, "_decoupled_verify_max_speculative_steps", None)
        if max_steps is None:
            max_steps = getattr(self.server_args, "speculative_num_steps", None)
        return None if max_steps is None else int(max_steps)

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
            keep_alive_refs.extend(
                self._run_throughput_profile_prefill(batch).extra_keep_alive_refs
                or []
            )
            for _ in range(_THROUGHPUT_PROFILE_WARMUP_ITERS):
                self._prepare_throughput_profile_draft_buffers(batch, steps)
                keep_alive_refs.extend(
                    self._run_throughput_profile_decode(batch).extra_keep_alive_refs
                    or []
                )
            torch.cuda.current_stream().synchronize()
            keep_alive_refs.clear()

            events = []
            for _ in range(_THROUGHPUT_PROFILE_MEASURE_ITERS):
                self._prepare_throughput_profile_draft_buffers(batch, steps)
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
        seq_len = int(seq_len)
        self._validate_throughput_profile_prompt_len(
            seq_len=seq_len,
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
                [((i + j) % token_mod) + 1 for j in range(seq_len)],
            )
            req = Req(
                rid=(
                    f"decoupled-throughput-profile-{batch_size}-{seq_len}-"
                    f"{i}-{time.time_ns()}"
                ),
                origin_input_text="",
                origin_input_ids=token_ids,
                sampling_params=sampling_params,
                extra_key=(
                    f"decoupled-throughput-profile-{batch_size}-{seq_len}-"
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

        prefill_budget_candidates = [
            int(value)
            for value in (
                getattr(self.server_args, "chunked_prefill_size", None),
                getattr(self.server_args, "max_prefill_tokens", None),
            )
            if value is not None and int(value) > 0
        ]
        prefill_limited_len = (
            min(prefill_budget_candidates) // batch_size
            if prefill_budget_candidates
            else None
        )
        details = {
            "context_limit": context_limited_len,
            "kv_pool_limit": pool_limited_len,
            "prefill_limit": prefill_limited_len,
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

    def _prepare_throughput_profile_draft_buffers(
        self, batch: ScheduleBatch, steps: int
    ) -> None:
        steps = int(steps)
        for req_idx, req in enumerate(batch.reqs):
            if steps <= 0:
                req.draft_buffer = []
                continue
            req.draft_buffer = [
                self._throughput_dummy_token_id(req_idx, len(req.output_ids) + i)
                for i in range(steps)
            ]

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
                release_kv_cache(req, tree_cache, is_insert=False)
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

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: List[int], batch_size: int = 0
    ) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int, ctx_len: int = 1) -> None:
        if self.adaptive_controller is not None:
            if isinstance(
                self.adaptive_controller, DecoupledVerifyThroughputAwareController
            ):
                self.adaptive_controller.activate_step_by_batch(batch_size, ctx_len)
            else:
                self.adaptive_controller.activate_step_by_batch(batch_size)

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: Optional[List[int]] = None,
    ) -> SpecRuntimeState:
        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            raise RuntimeError(
                "Decoupled verifier adaptive runtime states require target verify "
                "CUDA Graph."
            )
        speculative_num_steps = int(speculative_num_steps)
        speculative_num_draft_tokens = int(speculative_num_draft_tokens)
        verify_tokens_per_req = speculative_num_steps + 1
        if speculative_num_draft_tokens != verify_tokens_per_req:
            speculative_num_draft_tokens = verify_tokens_per_req
        capture_bs = (
            None if cuda_graph_bs is None else [int(bs) for bs in cuda_graph_bs]
        )
        if capture_bs is not None and not capture_bs:
            raise RuntimeError(
                "Decoupled verifier adaptive runtime state got an empty CUDA "
                f"Graph batch-size list for steps={speculative_num_steps}."
            )

        model_runner = self.target_worker.model_runner
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(
            self.device, getattr(model_runner, "gpu_id", 0), empty_cache=False
        )
        log_info_on_rank0(
            logger,
            "Capture decoupled verifier adaptive runtime state: "
            f"step={speculative_num_steps}, "
            f"verify_tokens_per_req={verify_tokens_per_req}, "
            f"capture_bs={capture_bs}",
        )

        with self._override_worker_state(
            speculative_num_steps,
            verify_tokens_per_req,
            cuda_graph_bs=capture_bs,
        ):
            backup_init = model_runner.init_new_workspace
            try:
                target_attn_backend = model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                model_runner.init_new_workspace = backup_init
            target_graph_runner = DecodeCudaGraphRunner(
                model_runner,
                attn_backend=target_attn_backend,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=verify_tokens_per_req,
            )
        after_mem = get_available_gpu_memory(
            self.device, getattr(model_runner, "gpu_id", 0), empty_cache=False
        )
        log_info_on_rank0(
            logger,
            "Captured decoupled verifier adaptive runtime state: "
            f"step={speculative_num_steps}, "
            f"verify_tokens_per_req={verify_tokens_per_req}, "
            f"capture_bs={target_graph_runner.capture_bs}, "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )
        state = SpecRuntimeState.for_decoupled_verify(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=verify_tokens_per_req,
            target_attn_backend=target_attn_backend,
            target_graph_runner=target_graph_runner,
        )
        self._validate_decoupled_runtime_state(state)
        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        self._validate_decoupled_runtime_state(state)
        if state.target_attn_backend is None or state.target_graph_runner is None:
            raise RuntimeError(
                "Decoupled verifier dynamic runtime state is missing target "
                "CUDA Graph resources."
            )
        if (
            self.speculative_num_steps == state.speculative_num_steps
            and self.speculative_num_draft_tokens == state.speculative_num_draft_tokens
            and self.target_worker.model_runner.attn_backend
            is state.target_attn_backend
            and self.target_worker.model_runner.decode_cuda_graph_runner
            is state.target_graph_runner
        ):
            return

        old_steps = self.speculative_num_steps
        old_draft_tokens = self.speculative_num_draft_tokens
        self.speculative_num_steps = int(state.speculative_num_steps)
        self.speculative_num_draft_tokens = int(state.speculative_num_draft_tokens)
        self.target_worker.model_runner.attn_backend = state.target_attn_backend
        self.target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )
        self.server_args.speculative_num_steps = self.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            self.speculative_num_draft_tokens
        )
        if (
            old_steps != self.speculative_num_steps
            or old_draft_tokens != self.speculative_num_draft_tokens
        ):
            log_info_on_rank0(
                logger,
                "Switch decoupled verifier adaptive state: "
                f"steps {old_steps} -> {self.speculative_num_steps}, "
                f"draft_tokens {old_draft_tokens} -> "
                f"{self.speculative_num_draft_tokens}",
            )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: Optional[List[int]] = None,
    ):
        """Temporarily override target-only adaptive state for graph capture."""
        sa = self.server_args
        decode_config = sa.cuda_graph_config.decode
        decode_config_bs = decode_config.bs
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs_decode,
            None if decode_config_bs is None else list(decode_config_bs),
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        if cuda_graph_bs is not None:
            sa.cuda_graph_bs_decode = cuda_graph_bs
            decode_config.bs = cuda_graph_bs

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
                sa.cuda_graph_bs_decode,
                decode_config.bs,
            ) = backup

    def _validate_decoupled_runtime_state(self, state: SpecRuntimeState) -> None:
        if (
            state.draft_attn_backend is not None
            or state.cuda_graph_runner is not None
            or state.draft_extend_attn_backend is not None
            or state.cuda_graph_runner_for_draft_extend is not None
        ):
            raise ValueError(
                "Decoupled verifier runtime state must not carry draft resources."
            )
        expected_draft_tokens = int(state.speculative_num_steps) + 1
        if int(state.speculative_num_draft_tokens) != expected_draft_tokens:
            raise RuntimeError(
                "Decoupled verifier runtime state has inconsistent verify width: "
                f"steps={state.speculative_num_steps}, "
                f"draft_tokens={state.speculative_num_draft_tokens}, "
                f"expected={expected_draft_tokens}"
            )
        graph_runner = state.target_graph_runner
        if graph_runner is None:
            raise RuntimeError(
                "Decoupled verifier adaptive runtime state has no target graph runner."
            )
        capture_bs = self._get_capture_bs(graph_runner)
        if not capture_bs:
            raise RuntimeError(
                "Decoupled verifier adaptive runtime state has no captured "
                f"batch sizes for steps={state.speculative_num_steps}."
            )
        adaptive_controller = getattr(self, "adaptive_controller", None)
        if isinstance(adaptive_controller, DecoupledVerifyThroughputAwareController):
            if int(state.speculative_num_steps) not in [
                int(step) for step in adaptive_controller.candidate_steps
            ]:
                raise RuntimeError(
                    "Decoupled verifier throughput-aware runtime state "
                    f"steps={state.speculative_num_steps} is not selected "
                    "by the controller."
                )

    def _get_capture_bs(self, graph_runner) -> List[int]:
        if graph_runner is not None and getattr(graph_runner, "capture_bs", None):
            return [int(bs) for bs in graph_runner.capture_bs]
        cuda_graph_config = getattr(self.server_args, "cuda_graph_config", None)
        decode_config = getattr(cuda_graph_config, "decode", None)
        capture_bs = getattr(decode_config, "bs", None)
        if capture_bs:
            return [int(bs) for bs in capture_bs]
        max_bs = getattr(decode_config, "max_bs", None)
        if max_bs is not None:
            return [int(max_bs)]
        capture_bs = getattr(self.server_args, "cuda_graph_bs", None)
        if capture_bs:
            return [int(bs) for bs in capture_bs]
        max_bs = getattr(self.server_args, "cuda_graph_max_bs", None)
        if max_bs is not None:
            return [int(max_bs)]
        raise RuntimeError(
            "Cannot determine decoupled verifier CUDA graph batch sizes for "
            "dynamic verify length."
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        return self.target_worker.update_weights_from_disk(recv_req)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        return self.target_worker.update_weights_from_ipc(recv_req)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def _get_verify_buffers(
        self, draft_token_num: int, batch: Optional[ScheduleBatch]
    ):
        expected_draft_tokens = int(self.speculative_num_draft_tokens)
        if int(draft_token_num) != expected_draft_tokens:
            raise RuntimeError(
                "Decoupled verifier verify buffer width mismatch: "
                f"draft_token_num={draft_token_num}, "
                f"expected={expected_draft_tokens}"
            )
        attn_backend = getattr(self.target_worker.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None, None

        get_buffers = getattr(
            attn_backend, "get_verify_buffers_to_fill_after_draft", None
        )
        if get_buffers is None:
            return None, None

        return get_buffers()

    def _get_pad_token_id(self) -> int:
        """Return an EOS token id used to pad short external draft tails."""
        hf_generation_config = getattr(self.model_config, "hf_generation_config", None)
        eos_token_id = _normalize_token_id(
            getattr(hf_generation_config, "eos_token_id", None)
        )
        if eos_token_id is not None:
            return eos_token_id

        hf_config = getattr(self.model_config, "hf_config", None)
        eos_token_id = _normalize_token_id(getattr(hf_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        get_text_config = getattr(hf_config, "get_text_config", None)
        text_config = (
            get_text_config()
            if callable(get_text_config)
            else getattr(hf_config, "text_config", None)
        )
        eos_token_id = _normalize_token_id(getattr(text_config, "eos_token_id", None))
        if eos_token_id is not None:
            return eos_token_id

        eos_token_ids = getattr(self.model_config, "hf_eos_token_id", None)
        if eos_token_ids:
            return min(int(token_id) for token_id in eos_token_ids)

        raise RuntimeError("External draft verification requires an EOS token id.")

    def _build_req_verify_tokens(
        self, req, pad_token_id: int, spec_depth: int
    ) -> List[int]:
        tail_token = _get_req_tail_token_id(req)
        draft_buffer = list(getattr(req, "draft_buffer", []) or [])
        draft_tokens = list(draft_buffer[:spec_depth])
        if len(draft_tokens) < spec_depth:
            draft_tokens.extend([int(pad_token_id)] * (spec_depth - len(draft_tokens)))
        return [tail_token, *draft_tokens]

    def _get_snapshot_tail_lens(
        self, batch: ScheduleBatch, spec_depth: int
    ) -> List[int]:
        return [
            min(
                len(list(getattr(req, "draft_buffer", []) or [])),
                spec_depth,
            )
            for req in batch.reqs
        ]

    def _assert_num_correct_within_snapshot_tail(
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: List[int]
    ) -> List[int]:
        # req.draft_buffer is a per-forward snapshot bound before verify. Any
        # concurrent drafter appends belong to later verify rounds.
        spec_steps = int(self.speculative_num_steps)
        real_tail_lens = self._get_snapshot_tail_lens(batch, spec_steps)
        raw_accept_lens = [int(x) for x in num_correct_drafts_per_req_cpu]
        for req, raw_accept_len, real_tail_len in zip(
            batch.reqs, raw_accept_lens, real_tail_lens
        ):
            assert raw_accept_len <= real_tail_len, (
                "Decoupled verify has accepted padded draft tokens: "
                f"request_id={req.rid} "
                f"raw_accept_len={raw_accept_len} "
                f"snapshot_tail_len={real_tail_len}"
            )

        return raw_accept_lens

    def _record_valid_draft_metrics(
        self, batch: ScheduleBatch, num_correct_drafts_per_req_cpu: List[int]
    ) -> int:
        spec_steps = int(self.speculative_num_steps)
        total_valid_draft_tokens = 0
        for req, accepted_drafts in zip(batch.reqs, num_correct_drafts_per_req_cpu):
            valid_draft_tokens = min(
                len(list(getattr(req, "draft_buffer", []) or [])), spec_steps
            )
            valid_accepted_tokens = min(int(accepted_drafts), valid_draft_tokens)
            total_valid_draft_tokens += valid_draft_tokens
            req.spec_valid_draft_tokens += valid_draft_tokens
            req.spec_valid_accepted_tokens += valid_accepted_tokens
            metric_len = max(
                spec_steps,
                len(req.spec_valid_draft_tokens_by_position),
                len(req.spec_valid_accepted_tokens_by_position),
            )
            req.spec_valid_draft_tokens_by_position = (
                req.spec_valid_draft_tokens_by_position + [0] * metric_len
            )[:metric_len]
            req.spec_valid_accepted_tokens_by_position = (
                req.spec_valid_accepted_tokens_by_position + [0] * metric_len
            )[:metric_len]
            for pos in range(valid_draft_tokens):
                req.spec_valid_draft_tokens_by_position[pos] += 1
            for pos in range(valid_accepted_tokens):
                req.spec_valid_accepted_tokens_by_position[pos] += 1
        return total_valid_draft_tokens

    def _build_trivial_verify_input(
        self, batch: ScheduleBatch, seq_lens_sum: int
    ) -> EagleVerifyInput:
        batch_size = batch.batch_size()
        device = batch.device
        draft_token = torch.tensor(
            [_get_req_tail_token_id(req) for req in batch.reqs],
            dtype=torch.long,
            device=device,
        )
        retrieve_index = torch.arange(
            batch_size, dtype=torch.long, device=device
        ).unsqueeze(1)
        retrieve_next_token = torch.full(
            (batch_size, 1), -1, dtype=torch.long, device=device
        )
        retrieve_next_sibling = torch.full(
            (batch_size, 1), -1, dtype=torch.long, device=device
        )

        tree_mask_buf, position_buf = self._get_verify_buffers(1, batch=batch)
        if tree_mask_buf is not None:
            custom_mask = tree_mask_buf
            custom_mask.fill_(True)
        else:
            custom_mask = torch.ones(
                seq_lens_sum + batch_size, dtype=torch.bool, device=device
            )

        if position_buf is not None:
            positions = position_buf
            positions[:batch_size].copy_(batch.seq_lens)
        else:
            positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_token,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        spec_steps = int(self.speculative_num_steps)
        draft_token_num = int(self.speculative_num_draft_tokens)
        if draft_token_num < 1:
            raise RuntimeError(
                "External draft verification requires at least one verify token "
                "per request."
            )

        if batch.forward_mode.is_idle():
            spec_info = EagleVerifyInput.create_idle_input(
                self.topk,
                spec_steps,
                draft_token_num,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL
            return spec_info

        seq_lens_sum = (
            int(batch.seq_lens_cpu.sum())
            if batch.seq_lens_cpu is not None
            else int(torch.sum(batch.seq_lens).item())
        )
        batch.seq_lens_sum = seq_lens_sum

        sampling_info = getattr(batch, "sampling_info", None)
        penalizer_orchestrator = getattr(sampling_info, "penalizer_orchestrator", None)
        if (
            penalizer_orchestrator is not None
            and penalizer_orchestrator.is_required
            and batch.reqs
        ):
            penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(
                    [_get_req_tail_token_id(req) for req in batch.reqs],
                    dtype=torch.int64,
                    device=batch.device,
                )
            )

        if spec_steps == 0:
            return self._build_trivial_verify_input(batch, seq_lens_sum)

        pad_token_id = self._get_pad_token_id()

        full_draft_tokens_by_req = [
            self._build_req_verify_tokens(req, pad_token_id, spec_steps)
            for req in batch.reqs
        ]
        bonus_tokens = torch.tensor(
            [tokens[0] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens = torch.tensor(
            [tokens[1:] for tokens in full_draft_tokens_by_req],
            dtype=torch.long,
            device=batch.device,
        )
        batch_size = batch.batch_size()
        selected_index, parent_list = _build_linear_topk1_tree_metadata(
            batch_size,
            spec_steps,
            batch.device,
        )

        tree_mask_buf, position_buf = self._get_verify_buffers(
            draft_token_num, batch=batch
        )

        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            flat_draft_tokens,
        ) = build_tree_kernel_efficient(
            bonus_tokens=bonus_tokens,
            parent_list=parent_list,
            top_scores_index=selected_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=seq_lens_sum,
            topk=1,
            spec_steps=spec_steps,
            num_verify_tokens=draft_token_num,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            tree_mask_buf=tree_mask_buf,
            position_buf=position_buf,
        )

        terminal_indices = torch.tensor(
            self._get_snapshot_tail_lens(batch, spec_steps),
            dtype=torch.long,
            device=batch.device,
        )
        row_indices = torch.arange(batch_size, dtype=torch.long, device=batch.device)
        terminal_indices = torch.clamp(terminal_indices, max=draft_token_num - 1)
        retrieve_next_token[row_indices, terminal_indices] = -1

        return EagleVerifyInput(
            draft_token=flat_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=spec_steps,
            topk=1,
            draft_token_num=draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(self, batch: ScheduleBatch) -> GenerationBatchResult:
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        assert verify_input is not None
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        was_idle = batch.forward_mode.is_idle()
        seq_lens_pre_verify = batch.seq_lens.clone()

        verify_input.num_tokens_per_req = verify_input.draft_token_num
        batch.return_hidden_states = False

        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            self.req_to_token_pool,
            batch,
            self.target_worker,
        )
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)
        assert (
            verify_forward_batch.capture_hidden_mode
            == verify_input.capture_hidden_mode
        )

        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output

        vocab_mask = None
        if batch.has_grammar:
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(
            logits_output.next_token_logits, "decoupled_verify: target model logits"
        )
        maybe_detect_inf(
            logits_output.next_token_logits, "decoupled_verify: target model logits"
        )

        predict, accept_lens, accept_index = eagle_sample(
            verify_input, batch, logits_output, vocab_mask
        )
        num_correct_drafts_per_req_cpu = (accept_lens - 1).cpu().tolist()
        self._assert_num_correct_within_snapshot_tail(
            batch, num_correct_drafts_per_req_cpu
        )
        valid_draft_tokens = self._record_valid_draft_metrics(
            batch, num_correct_drafts_per_req_cpu
        )
        new_seq_lens = seq_lens_pre_verify + accept_lens

        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_index,
            verify_input.draft_token_num,
        )

        if not was_idle:
            if self.page_size != 1:
                raise RuntimeError(
                    "Decoupled verifier currently requires page_size == 1."
                )

        if not was_idle:
            bonus_tokens = _build_bonus_tokens_from_accepts(
                predict, accept_lens, accept_index
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not was_idle:
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, verify_input.spec_steps
            )

        batch.forward_mode = ForwardMode.IDLE if was_idle else ForwardMode.DECODE
        batch.spec_info = None
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            num_correct_drafts=sum(num_correct_drafts_per_req_cpu),
            num_correct_drafts_per_req_cpu=num_correct_drafts_per_req_cpu,
            spec_valid_draft_tokens=valid_draft_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=verify_input.draft_token_num,
            next_draft_input=_build_next_draft_input_stub(
                bonus_tokens, verify_input.topk
            ),
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            batch.capture_hidden_mode = CaptureHiddenMode.NULL
            batch_output = self.target_worker.forward_batch_generation(batch)
            if isinstance(batch_output.next_token_ids, torch.Tensor):
                batch_output.next_draft_input = _build_next_draft_input_stub(
                    bonus_tokens=batch_output.next_token_ids.flatten().to(
                        dtype=torch.int32
                    ),
                    topk=self.topk,
                )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        spec_info = self.draft(batch)
        assert spec_info.is_verify_input()
        batch.spec_info = spec_info
        batch_output = self.verify(batch)

        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if self.enable_adaptive_verify and not batch_output.can_run_cuda_graph:
            raise RuntimeError(
                "Decoupled verifier dynamic verify length requires full CUDA "
                "Graph replay for target verify, but this forward ran without it: "
                f"speculative_num_steps={self.speculative_num_steps}, "
                f"speculative_num_draft_tokens={self.speculative_num_draft_tokens}, "
                f"draft_token_num={spec_info.draft_token_num}"
            )

        num_verified_reqs = len(batch_output.num_correct_drafts_per_req_cpu or [])
        self.total_accept_length += int(batch_output.num_correct_drafts)
        self.total_num_verified_reqs += num_verified_reqs
        return batch_output

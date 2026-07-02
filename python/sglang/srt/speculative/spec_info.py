from __future__ import annotations

import bisect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Type, Union

import msgspec
import torch

from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    ServerArgsValidator,
    WorkerFactory,
)
from sglang.srt.speculative.spec_registry import get_spec as _get_registered_spec
from sglang.srt.speculative.spec_registry import (
    register_algorithm as _register_algorithm,
)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker

logger = logging.getLogger(__name__)


class DecoupledVerifyShape(msgspec.Struct, frozen=True, kw_only=True):
    raw_batch_size: int
    captured_batch_size: int
    budget: int
    max_speculative_num_steps: int
    num_speculative_steps: int
    verify_tokens_per_req: int

    @property
    def padded_verify_tokens(self) -> int:
        return self.captured_batch_size * self.verify_tokens_per_req

    @property
    def raw_verify_tokens(self) -> int:
        return self.raw_batch_size * self.verify_tokens_per_req

    @property
    def uses_decode_graph(self) -> bool:
        return self.num_speculative_steps == 0

    def validate(self) -> None:
        if self.raw_batch_size <= 0:
            raise ValueError(f"raw_batch_size must be positive: {self.raw_batch_size}")
        if self.captured_batch_size < self.raw_batch_size:
            raise ValueError(
                "captured_batch_size must be >= raw_batch_size: "
                f"captured={self.captured_batch_size}, raw={self.raw_batch_size}"
            )
        if self.verify_tokens_per_req != self.num_speculative_steps + 1:
            raise ValueError(
                "verify_tokens_per_req must equal num_speculative_steps + 1: "
                f"tokens={self.verify_tokens_per_req}, "
                f"steps={self.num_speculative_steps}"
            )
        if self.num_speculative_steps < 0:
            raise ValueError(
                f"num_speculative_steps must be non-negative: {self.num_speculative_steps}"
            )
        if self.num_speculative_steps > self.max_speculative_num_steps:
            raise ValueError(
                "num_speculative_steps exceeds max_speculative_num_steps: "
                f"steps={self.num_speculative_steps}, "
                f"max={self.max_speculative_num_steps}"
            )
        if self.padded_verify_tokens >= self.budget:
            raise ValueError(
                "decoupled verifier target verify budget violated after CUDA graph "
                "batch padding: "
                f"captured_bs={self.captured_batch_size}, "
                f"verify_tokens_per_req={self.verify_tokens_per_req}, "
                f"padded_verify_tokens={self.padded_verify_tokens}, "
                f"budget={self.budget}"
            )


@dataclass(frozen=True)
class DecoupledVerifyRuntimeState:
    """Runtime resources selected for one decoupled verifier target-verify pass.

    This is the decoupled-verifier analogue of ``SpecRuntimeState``.  It is
    scoped to a single verify batch and only describes target-side resources;
    the decoupled drafter and draft-extend paths live in separate workers.
    """

    shape: DecoupledVerifyShape
    target_attn_backend: object | None = None
    target_graph_runner: object | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def raw_batch_size(self) -> int:
        return self.shape.raw_batch_size

    @property
    def captured_batch_size(self) -> int:
        return self.shape.captured_batch_size

    @property
    def budget(self) -> int:
        return self.shape.budget

    @property
    def max_speculative_num_steps(self) -> int:
        return self.shape.max_speculative_num_steps

    @property
    def num_speculative_steps(self) -> int:
        return self.shape.num_speculative_steps

    @property
    def verify_tokens_per_req(self) -> int:
        return self.shape.verify_tokens_per_req

    @property
    def padded_verify_tokens(self) -> int:
        return self.shape.padded_verify_tokens

    @property
    def raw_verify_tokens(self) -> int:
        return self.shape.raw_verify_tokens

    @property
    def uses_decode_graph(self) -> bool:
        return self.shape.uses_decode_graph

    def validate(self) -> None:
        self.shape.validate()


def dynamic_verify_enabled(server_args) -> bool:
    return bool(getattr(server_args, "decoupled_spec_dynamic_verify_length", False))


def select_captured_batch_size(
    *,
    raw_batch_size: int,
    capture_bs: Sequence[int],
) -> int:
    if raw_batch_size <= 0:
        raise ValueError(f"raw_batch_size must be positive: {raw_batch_size}")
    if not capture_bs:
        raise ValueError("No CUDA graph batch sizes are available for target verify.")
    ordered_bs = sorted(int(bs) for bs in capture_bs)
    index = bisect.bisect_left(ordered_bs, int(raw_batch_size))
    if index >= len(ordered_bs):
        raise ValueError(
            "No captured CUDA graph batch size can cover decoupled verifier batch: "
            f"raw_batch_size={raw_batch_size}, max_captured_bs={ordered_bs[-1]}"
        )
    return ordered_bs[index]


def compute_decoupled_verify_shape(
    *,
    raw_batch_size: int,
    captured_batch_size: int,
    budget: int,
    max_speculative_num_steps: int,
) -> DecoupledVerifyShape:
    raw_batch_size = int(raw_batch_size)
    captured_batch_size = int(captured_batch_size)
    budget = int(budget)
    max_speculative_num_steps = int(max_speculative_num_steps)
    if raw_batch_size <= 0:
        raise ValueError(f"raw_batch_size must be positive: {raw_batch_size}")
    if captured_batch_size < raw_batch_size:
        raise ValueError(
            "captured_batch_size must cover raw_batch_size: "
            f"captured={captured_batch_size}, raw={raw_batch_size}"
        )
    if budget <= 0:
        raise ValueError(f"target verify token budget must be positive: {budget}")
    if max_speculative_num_steps < 0:
        raise ValueError(
            "max_speculative_num_steps must be non-negative: "
            f"{max_speculative_num_steps}"
        )

    budget_step_cap = (budget - 1) // captured_batch_size - 1
    if budget_step_cap < 0:
        raise ValueError(
            "No legal decoupled verifier target verify length fits the token "
            "budget after CUDA graph batch padding: "
            f"captured_bs={captured_batch_size}, budget={budget}. "
            "Need budget > captured_bs to verify at least one token per request."
        )
    num_speculative_steps = min(max_speculative_num_steps, budget_step_cap)
    shape = DecoupledVerifyShape(
        raw_batch_size=raw_batch_size,
        captured_batch_size=captured_batch_size,
        budget=budget,
        max_speculative_num_steps=max_speculative_num_steps,
        num_speculative_steps=num_speculative_steps,
        verify_tokens_per_req=num_speculative_steps + 1,
    )
    shape.validate()
    return shape


def compute_shape_from_capture_bs(
    *,
    raw_batch_size: int,
    capture_bs: Sequence[int],
    budget: int,
    max_speculative_num_steps: int,
) -> DecoupledVerifyShape:
    captured_batch_size = select_captured_batch_size(
        raw_batch_size=raw_batch_size,
        capture_bs=capture_bs,
    )
    return compute_decoupled_verify_shape(
        raw_batch_size=raw_batch_size,
        captured_batch_size=captured_batch_size,
        budget=budget,
        max_speculative_num_steps=max_speculative_num_steps,
    )


def compute_decoupled_verify_runtime_state(
    *,
    raw_batch_size: int,
    captured_batch_size: int,
    budget: int,
    max_speculative_num_steps: int,
    target_attn_backend: object | None = None,
    target_graph_runner: object | None = None,
) -> DecoupledVerifyRuntimeState:
    shape = compute_decoupled_verify_shape(
        raw_batch_size=raw_batch_size,
        captured_batch_size=captured_batch_size,
        budget=budget,
        max_speculative_num_steps=max_speculative_num_steps,
    )
    return DecoupledVerifyRuntimeState(
        shape=shape,
        target_attn_backend=target_attn_backend,
        target_graph_runner=target_graph_runner,
    )


def compute_runtime_state_from_capture_bs(
    *,
    raw_batch_size: int,
    capture_bs: Sequence[int],
    budget: int,
    max_speculative_num_steps: int,
    target_attn_backend: object | None = None,
    target_graph_runner: object | None = None,
) -> DecoupledVerifyRuntimeState:
    captured_batch_size = select_captured_batch_size(
        raw_batch_size=raw_batch_size,
        capture_bs=capture_bs,
    )
    return compute_decoupled_verify_runtime_state(
        raw_batch_size=raw_batch_size,
        captured_batch_size=captured_batch_size,
        budget=budget,
        max_speculative_num_steps=max_speculative_num_steps,
        target_attn_backend=target_attn_backend,
        target_graph_runner=target_graph_runner,
    )


class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")

    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:
        """Decorator to register a plugin speculative algorithm. The factory
        takes ``server_args`` and returns the worker class. Pass a
        ``CustomSpecAlgo`` subclass via ``spec_class`` to override any
        ``is_*()`` / ``create_worker`` method.

        Example:
            @SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
            def _factory(server_args):
                return MySpecWorker
        """
        return _register_algorithm(
            name,
            supports_overlap=supports_overlap,
            validate_server_args=validate_server_args,
            spec_class=spec_class,
        )

    def is_some(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_speculative(self) -> bool:
        return self != SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # FIXME(kpham_sgl): Remove FROZEN_KV_MTP here once we
        # have established support for it in the scheduler.
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        )

    def is_eagle3(self) -> bool:
        return self == SpeculativeAlgorithm.EAGLE3

    def is_frozen_kv_mtp(self) -> bool:
        return self == SpeculativeAlgorithm.FROZEN_KV_MTP

    def is_dflash(self) -> bool:
        return self == SpeculativeAlgorithm.DFLASH

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def is_decoupled_verify(self) -> bool:
        return False

    def is_decoupled_draft(self) -> bool:
        return False

    def supports_spec_v2(self) -> bool:
        return self.is_speculative()

    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash()

    def has_draft_kv(self) -> bool:
        """Whether the draft phase writes KV chains. NGRAM does not (its tree
        lives only in the verify mask), so per-decode KV sizing needs no
        per-topk page rounding; see get_alloc_len_per_decode."""
        return not self.is_ngram()

    def carries_draft_hidden_states(self) -> bool:
        """Whether the disagg prefill->decode transfer carries draft hidden
        states (EAGLE-family only; STANDALONE's vanilla draft ignores them)."""
        return self.is_eagle()

    def create_future_map(
        self,
        device: torch.device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ) -> FutureMap:
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

    def build_disagg_draft_input(
        self,
        batch: ScheduleBatch,
        server_args: ServerArgs,
        last_tokens_tensor: torch.Tensor,
        future_map: FutureMap,
    ) -> Optional[SpecInput]:
        if self.is_eagle():
            from sglang.srt.speculative.eagle_disaggregation import (
                build_eagle_disagg_draft_input,
            )

            return build_eagle_disagg_draft_input(
                batch, server_args, last_tokens_tensor, future_map
            )
        return None

    def need_topk(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def handle_server_args(self, server_args: ServerArgs) -> None:
        """Hook for per-algorithm server args mutation.

        In-place updated.
        """
        from sglang.srt.arg_groups.speculative_hook import (
            _handle_dflash,
            _handle_eagle_family,
            _handle_frozen_kv_mtp,
            _handle_ngram,
        )

        if self.is_dflash():
            _handle_dflash(server_args)
        elif self.is_frozen_kv_mtp():
            _handle_frozen_kv_mtp(server_args)
        elif self.is_eagle() or self.is_standalone():
            _handle_eagle_family(server_args)
        elif self.is_ngram():
            _handle_ngram(server_args)

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens: int, is_draft_worker: bool
    ) -> int:
        # FIXME: Remove this after the forward mode refactor. Target verify is
        # essentially a fixed sequence length prefill/extend with full cuda
        # graph support. We can use it for target verify, or we can use it for
        # other cases which is not target verify but fixed length prefill.
        # Here, we expose this interface to allow the other use cases.
        return num_draft_tokens

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        if self.is_dflash():
            # V2 worker drives both overlap and non-overlap (scheduler runs it
            # synchronously when overlap is disabled), same as EAGLE.
            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

            return DFlashWorkerV2

        if self.is_frozen_kv_mtp():
            # V2 worker drives both overlap and non-overlap (scheduler runs it
            # synchronously when overlap is disabled), same as EAGLE.
            from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
                FrozenKVMTPWorkerV2,
            )

            return FrozenKVMTPWorkerV2

        # EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER always use the V2 worker,
        # even with overlap disabled (scheduler drives it synchronously).
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2

        elif self.is_eagle():
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            return EAGLEWorkerV2
        elif self.is_standalone():
            from sglang.srt.speculative.standalone_worker_v2 import (
                StandaloneWorkerV2,
            )

            return StandaloneWorkerV2
        elif self.is_ngram():
            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")


class DecoupledVerifySpecAlgo(CustomSpecAlgo):
    def is_decoupled_verify(self) -> bool:
        return True

    def supports_spec_v2(self) -> bool:
        return False

    @staticmethod
    def validate_server_args(server_args: ServerArgs) -> None:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        if server_args.pp_size != 1:
            raise ValueError("decoupled verifier currently requires pp_size == 1.")
        if server_args.decoupled_spec_rank_base < 0:
            raise ValueError("--decoupled-spec-rank-base must be non-negative.")
        if server_args.page_size is not None and server_args.page_size > 1:
            raise ValueError(
                "decoupled verifier currently requires page_size == 1 because "
                "token rollback is token-granular."
            )

        if server_args.max_running_requests is None:
            server_args.max_running_requests = 64
            logger.warning(
                "Max running requests is reset to 64 for decoupled speculative decoding. "
                "You can override this by explicitly setting --max-running-requests."
            )

        if not server_args.disable_overlap_schedule:
            server_args.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled for decoupled speculative decoding."
            )

        if server_args.enable_mixed_chunk:
            server_args.enable_mixed_chunk = False
            logger.warning(
                "Mixed chunked prefill is disabled for decoupled speculative decoding."
            )

        if server_args.cuda_graph_config.prefill.backend != Backend.DISABLED:
            server_args.cuda_graph_config.prefill.backend = Backend.DISABLED
            logger.warning(
                "Prefill CUDA graph is disabled for decoupled speculative decoding."
            )
        server_args.disable_piecewise_cuda_graph = True

        if server_args.speculative_num_steps is None:
            raise ValueError(
                "decoupled speculative decoding requires speculative_num_steps to be set."
            )

        if (
            server_args.speculative_eagle_topk is not None
            and server_args.speculative_eagle_topk != 1
        ):
            raise ValueError(
                "decoupled speculative decoding currently only supports "
                "speculative_eagle_topk == 1."
            )
        server_args.speculative_eagle_topk = 1

        expected_num_draft_tokens = int(server_args.speculative_num_steps) + 1
        if server_args.speculative_num_draft_tokens != expected_num_draft_tokens:
            logger.warning(
                "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 "
                "for decoupled speculative decoding."
            )
            server_args.speculative_num_draft_tokens = expected_num_draft_tokens

        if getattr(server_args, "decoupled_spec_dynamic_verify_length", False):
            budget = getattr(
                server_args, "decoupled_spec_target_verify_token_budget", None
            )
            if budget is None or int(budget) <= 0:
                raise ValueError(
                    "--decoupled-spec-target-verify-token-budget must be a "
                    "positive integer when "
                    "--decoupled-spec-dynamic-verify-length is enabled."
                )
            server_args.decoupled_spec_target_verify_token_budget = int(budget)
            if server_args.cuda_graph_config.decode.backend == Backend.DISABLED:
                raise ValueError(
                    "decoupled verifier dynamic verify length requires full CUDA "
                    "Graph; enable decode CUDA graph."
                )


class DecoupledDraftSpecAlgo(CustomSpecAlgo):
    def is_decoupled_draft(self) -> bool:
        return True

    def supports_spec_v2(self) -> bool:
        return False

    @staticmethod
    def validate_server_args(server_args: ServerArgs) -> None:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        if server_args.pp_size != 1:
            raise ValueError("decoupled drafter currently requires pp_size == 1.")
        if server_args.decoupled_spec_rank_base < 0:
            raise ValueError("--decoupled-spec-rank-base must be non-negative.")

        if server_args.max_running_requests is None:
            server_args.max_running_requests = 64
            logger.warning(
                "Max running requests is reset to 64 for decoupled speculative decoding. "
                "You can override this by explicitly setting --max-running-requests."
            )

        if not server_args.disable_overlap_schedule:
            server_args.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled for decoupled speculative decoding."
            )

        if not server_args.disable_radix_cache:
            server_args.disable_radix_cache = True
            logger.warning("Radix cache is disabled for decoupled drafter.")

        if server_args.mamba_radix_cache_strategy != "no_buffer":
            server_args.mamba_radix_cache_strategy = "no_buffer"
            logger.warning(
                "Mamba extra buffer is disabled for decoupled drafter engine. "
                "Falling back to --mamba-radix-cache-strategy no_buffer."
            )

        if server_args.enable_mixed_chunk:
            server_args.enable_mixed_chunk = False
            logger.warning(
                "Mixed chunked prefill is disabled for decoupled speculative decoding."
            )

        if server_args.cuda_graph_config.prefill.backend != Backend.DISABLED:
            server_args.cuda_graph_config.prefill.backend = Backend.DISABLED
            logger.warning(
                "Prefill CUDA graph is disabled for decoupled speculative decoding."
            )
        server_args.disable_piecewise_cuda_graph = True

        if server_args.speculative_num_steps is None:
            raise ValueError(
                "decoupled speculative decoding requires speculative_num_steps to be set."
            )

        if (
            server_args.speculative_eagle_topk is not None
            and server_args.speculative_eagle_topk != 1
        ):
            raise ValueError(
                "decoupled speculative decoding currently only supports "
                "speculative_eagle_topk == 1."
            )
        server_args.speculative_eagle_topk = 1

        expected_num_draft_tokens = int(server_args.speculative_num_steps) + 1
        if server_args.speculative_num_draft_tokens != expected_num_draft_tokens:
            logger.warning(
                "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 "
                "for decoupled speculative decoding."
            )
            server_args.speculative_num_draft_tokens = expected_num_draft_tokens
        if getattr(server_args, "decoupled_spec_dynamic_verify_length", False):
            raise ValueError(
                "--decoupled-spec-dynamic-verify-length is only supported on "
                "DECOUPLED_VERIFY target workers, not DECOUPLED_DRAFT."
            )

    def create_worker(self, server_args: ServerArgs) -> Type:
        raise ValueError(
            "decoupled_draft uses the normal TP worker instead of create_worker()."
        )


@SpeculativeAlgorithm.register(
    "DECOUPLED_VERIFY",
    validate_server_args=DecoupledVerifySpecAlgo.validate_server_args,
    spec_class=DecoupledVerifySpecAlgo,
)
def _decoupled_verify_worker_factory(server_args: ServerArgs) -> Type:
    from sglang.srt.speculative.decoupled_verify_worker import VerifyWorker

    return VerifyWorker


@SpeculativeAlgorithm.register(
    "DECOUPLED_DRAFT",
    validate_server_args=DecoupledDraftSpecAlgo.validate_server_args,
    spec_class=DecoupledDraftSpecAlgo,
)
def _decoupled_draft_worker_factory(server_args: ServerArgs) -> Type:
    raise ValueError(
        "decoupled_draft uses the normal TP worker instead of create_worker()."
    )


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_DRAFT_EXTEND = auto()
    EAGLE_VERIFY = auto()
    FROZEN_KV_MTP_DRAFT = auto()
    FROZEN_KV_MTP_VERIFY = auto()
    DFLASH_DRAFT = auto()
    DFLASH_VERIFY = auto()
    NGRAM_VERIFY = auto()


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    # Cross-algorithm phase guards. Used by attention backends and
    # ForwardBatch padding logic to dispatch on phase without hardcoding the
    # specific algo class (EAGLE / FROZEN_KV_MTP / DFLASH / NGRAM each have
    # their own draft / verify SpecInput subclasses).
    def is_draft_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_DRAFT,
            SpecInputType.EAGLE_DRAFT_EXTEND,
            SpecInputType.FROZEN_KV_MTP_DRAFT,
            SpecInputType.DFLASH_DRAFT,
        }

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.FROZEN_KV_MTP_VERIFY,
            SpecInputType.DFLASH_VERIFY,
            SpecInputType.NGRAM_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, batch: ScheduleBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob


def create_dummy_verify_input(
    spec_algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
    custom_mask: torch.Tensor,
    num_tokens_per_bs: int,
    is_draft_worker: bool,
) -> Optional[SpecInput]:
    """Dummy verify ``SpecInput`` for CUDA-graph capture (per-algorithm dispatch)."""
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    spec_info = None
    if spec_algorithm.is_eagle() or spec_algorithm.is_standalone():
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        if is_draft_worker:
            raise RuntimeError("This should not happen.")
        else:
            spec_info = EagleVerifyInput(
                draft_token=None,
                custom_mask=custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_cum_len=None,
                spec_steps=server_args.speculative_num_steps,
                topk=server_args.speculative_eagle_topk,
                draft_token_num=server_args.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )
    elif spec_algorithm.is_dflash():
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
        spec_info = DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=server_args.speculative_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if is_draft_worker else CaptureHiddenMode.FULL
            ),
        )

    elif spec_algorithm.is_ngram():
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        spec_info = NgramVerifyInput(
            draft_token=None,
            custom_mask=custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            draft_token_num=num_tokens_per_bs,
        )
        spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

    return spec_info

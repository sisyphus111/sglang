from __future__ import annotations

import glob
import logging
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

GPU_DRAFT_TAIL_DEBUG_WIDTH = 7
GPU_DRAFT_TAIL_DEBUG_FIELD_NAMES = (
    "reason",
    # Current-epoch drafter APPEND arrival sequence; verifier-local updates do
    # not advance it, and OPEN/CLOSE expose -1 until the next arrival.
    "publish_seq",
    "delta",
    "raw_len",
    "consumable_len",
    "pending_len",
    "committed_len",
    "error_code",
    "error_op_seq",
    "pending_prefix_fast_forward_ct",
    "seqlock_retries",
)
GPU_DRAFT_TAIL_SELECT_REASON_NAMES = (
    "unset",
    "invalid_seat",
    "writer_in_progress",
    "identity_mismatch",
    "metadata_invalid",
    "direct",
    "logical_behind",
    "delta_too_large",
    "delta_beyond_consumable",
    "bonus_mismatch",
    "rebased",
    "version_changed",
    "logical_cursor_mismatch",
    "pending_prefix",
)
GPU_DRAFT_TAIL_UPDATE_ERROR_NAMES = (
    "none",
    "invalid_op",
    "op_sequence_regression",
    "commit_prefix_mismatch",
    "pending_tail_invariant",
    "pending_capacity_exceeded",
    "draft_base_ahead",
    "draft_token_conflict",
    "draft_token_skip",
    "tail_capacity_exceeded",
    "invalid_metadata",
    "critical_commit_pending_prefix",
    "critical_commit_invalid_length",
    "critical_commit_logical_cursor_mismatch",
    "commit_ack_ahead",
    "drafter_commit_requires_replay",
    "drafter_model_state_invalid",
    "drafter_checkpoint_unavailable",
)


@lru_cache(maxsize=1)
def _load_decoupled_spec_cpp_module():
    _ensure_zmq_lib_path()
    source_dir = Path(__file__).resolve().parent / "csrc" / "decoupled_spec"
    sources = [
        source_dir / "decoupled_spec_pybind.cpp",
        source_dir / "gpu_draft_tail.cu",
    ]
    logger.info(
        "Loading the decoupled-spec GPU backend and wire codec; JIT compilation may be "
        "triggered: source=%s",
        sources,
    )
    return load(
        name="sglang_decoupled_spec_pybind",
        sources=[str(source) for source in sources],
        extra_cflags=["-O3", "-std=c++17", "-pthread"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-ldl", "-pthread"],
        verbose=False,
    )


def _ensure_zmq_lib_path() -> None:
    if envs.SGLANG_DECOUPLED_SPEC_ZMQ_LIB.get():
        return
    try:
        import zmq
    except Exception:
        return

    zmq_dir = Path(zmq.__file__).resolve().parent
    for parent in (zmq_dir, *zmq_dir.parents):
        candidates = glob.glob(str(parent / "pyzmq.libs" / "libzmq*.so*"))
        if candidates:
            envs.SGLANG_DECOUPLED_SPEC_ZMQ_LIB.set(candidates[0])
            return


class CppGpuDraftTailBuffer:
    """One rolling GPU tail row per request-pool seat.

    The verifier role materializes direct snapshots from remote draft arrivals.
    In drafter-authoritative mode, network controls and forward completion
    instead share the same writer lock around branch/KV/state reconciliation.
    """

    def __init__(
        self,
        *,
        device: str | torch.device,
        num_seats: int,
        num_draft_tokens: int,
        landing_stream: torch.cuda.Stream,
        mock_profile: bool = False,
        drafter_authoritative: bool = False,
        pending_token_capacity: int | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"GPU draft-tail buffer requires CUDA, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if int(num_seats) <= 0 or int(num_draft_tokens) <= 0:
            raise ValueError(
                "GPU draft-tail dimensions must be positive: "
                f"num_seats={num_seats} num_draft_tokens={num_draft_tokens}"
            )
        if torch.device(landing_stream.device) != device:
            raise ValueError(
                "GPU draft-tail landing stream is on a different device: "
                f"buffer_device={device} stream_device={landing_stream.device}"
            )

        self.device = device
        self.num_seats = int(num_seats)
        self.num_draft_tokens = int(num_draft_tokens)
        self.allow_partial = envs.SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL.get()
        self.tail_capacity = 2 * self.num_draft_tokens + 1
        self.pending_token_capacity = (
            self.tail_capacity
            if pending_token_capacity is None
            else int(pending_token_capacity)
        )
        if self.pending_token_capacity < self.tail_capacity:
            raise ValueError(
                "GPU pending-token capacity must cover the draft tail: "
                f"pending={self.pending_token_capacity} "
                f"tail={self.tail_capacity}"
            )
        self.landing_stream = landing_stream
        self.mock_profile = bool(mock_profile)
        self.drafter_authoritative = bool(drafter_authoritative)
        with torch.cuda.device(device):
            self.versions = torch.zeros(
                self.num_seats, dtype=torch.int64, device=device
            )
            self.publish_seqs = torch.zeros_like(self.versions)
            self.request_epochs = torch.full_like(self.versions, -1)
            self.active_request_epochs = torch.full_like(self.versions, -1)
            self.prompt_lens = torch.full_like(self.versions, -1)
            self.committed_lens = torch.full_like(self.versions, -1)
            self.can_accept_prefix_lens = torch.full_like(self.versions, -1)
            self.raw_tail_lens = torch.zeros_like(self.versions)
            self.consumable_tail_lens = torch.zeros_like(self.versions)
            self.pending_expected_lens = torch.zeros_like(self.versions)
            # Pending verifier tokens use absolute output position modulo the
            # fixed row capacity, retaining the newest bounded suffix on lag.
            self.pending_expected_tokens = torch.zeros(
                (self.num_seats, self.pending_token_capacity),
                dtype=torch.int64,
                device=device,
            )
            self.tail_tokens = torch.full(
                (self.num_seats, self.tail_capacity),
                100 if self.mock_profile else 0,
                dtype=torch.int64,
                device=device,
            )
            self.last_op_seqs = torch.zeros_like(self.versions)
            self.error_codes = torch.zeros_like(self.versions)
            self.error_op_seqs = torch.zeros_like(self.versions)
            self.pending_prefix_fast_forward_cts = torch.zeros_like(self.versions)
            self.model_output_lens = torch.full_like(self.versions, -1)
            self.model_state_positions = torch.full_like(self.versions, -1)
            self.model_input_tokens = torch.full_like(self.versions, -1)
            # Exact logical-position tags prevent modulo-ring ABA when a
            # checkpoint offset is reused after a branch rewind.
            self.checkpoint_positions = torch.full(
                (self.num_seats, self.tail_capacity),
                -1,
                dtype=torch.int64,
                device=device,
            )
            # A bounded cumulative snapshot is the only production token
            # egress from an authoritative drafter GPU row.
            self.egress_seqs = torch.zeros_like(self.versions)
            self.last_commit_tokens = torch.full_like(self.versions, -1)
            # Per-seat [request epoch, exclusive KV ownership high-water].
            # Request-table cells past this bound may belong to an old lifetime.
            self._decode_kv_ownership = torch.full(
                (self.num_seats, 2), -1, dtype=torch.int64, device=device
            )
            self._init_event = torch.cuda.Event()
            self._init_event.record(torch.cuda.current_stream(device))
            self.landing_stream.wait_event(self._init_event)

        self._cpp = _load_decoupled_spec_cpp_module().GpuDraftTailBuffer(
            int(device.index),
            self.num_seats,
            self.num_draft_tokens,
            self.pending_token_capacity,
            int(self.landing_stream.cuda_stream),
            int(self.versions.data_ptr()),
            int(self.publish_seqs.data_ptr()),
            int(self.request_epochs.data_ptr()),
            int(self.prompt_lens.data_ptr()),
            int(self.committed_lens.data_ptr()),
            int(self.can_accept_prefix_lens.data_ptr()),
            int(self.raw_tail_lens.data_ptr()),
            int(self.consumable_tail_lens.data_ptr()),
            int(self.pending_expected_lens.data_ptr()),
            int(self.pending_expected_tokens.data_ptr()),
            int(self.tail_tokens.data_ptr()),
            int(self.last_op_seqs.data_ptr()),
            int(self.error_codes.data_ptr()),
            int(self.error_op_seqs.data_ptr()),
            int(self.pending_prefix_fast_forward_cts.data_ptr()),
            int(self.model_output_lens.data_ptr()),
            int(self.model_state_positions.data_ptr()),
            int(self.model_input_tokens.data_ptr()),
            int(self.checkpoint_positions.data_ptr()),
            int(self.egress_seqs.data_ptr()),
            int(self.last_commit_tokens.data_ptr()),
            self.drafter_authoritative,
            self.allow_partial,
        )
        self._closed = False
        # Retain stream objects as well as handles: initialization is immutable,
        # so each decode stream needs this dependency only once.
        self._decode_initialized_streams = {}
        self._decode_checkpoint_table = None
        self._decode_req_table = None

    @property
    def staging_slot_count(self) -> int:
        """Allocated landing slots; strict verification preallocates the full ring."""

        return int(self._cpp.staging_slot_count)

    @property
    def max_staging_slots(self) -> int:
        """Hard bound for in-flight landing publications."""

        return int(self._cpp.max_staging_slots)

    def bind_request(self, request_id: str, gpu_seat: int, request_epoch: int) -> None:
        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        gpu_seat = int(gpu_seat)
        request_epoch = int(request_epoch)
        self._cpp.bind_request(str(request_id), gpu_seat, request_epoch)
        # This scheduler-stream tensor is the source for immutable per-launch
        # epochs. GPU row ownership itself linearizes at the landing OPEN lock.
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).wait_event(self._init_event)
            self.active_request_epochs[gpu_seat] = request_epoch

    def lookup_binding(
        self, request_id: str, src_verifier_rank: int
    ) -> tuple[int, int] | None:
        """Return the native mirror seat and lifetime epoch for one control key."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        seat, request_epoch = self._cpp.lookup_binding_native(
            str(request_id), int(src_verifier_rank)
        )
        if int(seat) < 0:
            return None
        return int(seat), int(request_epoch)

    def wait_for_landing(self) -> None:
        """Fence first-use lifecycle state without synchronizing the host."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._init_event)
        self._cpp.wait_for_landing_native(int(current_stream.cuda_stream))

    def capture_active_request_epochs(
        self,
        gpu_seats: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Freeze each batch row's lifecycle identity on the caller's stream."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        batch_size = int(gpu_seats.numel())
        self._validate_vector(gpu_seats, "gpu_seats", batch_size, (torch.int64,))
        self._validate_vector(out, "out", batch_size, (torch.int64,))
        torch.index_select(self.active_request_epochs, 0, gpu_seats, out=out)
        return out

    def wait_for_landing_event(self, event: torch.cuda.Event) -> None:
        """Order the caller's stream after one recorded landing watermark."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        torch.cuda.current_stream(self.device).wait_event(event)

    def apply_verify_commit_from_device(
        self,
        gpu_seats: torch.Tensor,
        expected_request_epochs: torch.Tensor,
        pre_verify_seq_lens: torch.Tensor | None,
        accept_tokens: torch.Tensor,
        num_accept_tokens: torch.Tensor,
        *,
        accept_token_stride: int,
        commit_mask: torch.Tensor | None = None,
    ) -> None:
        """Apply one target-accepted run per row before the next GPU snapshot.

        Decode passes its pre-verify sequence lengths. Extend passes ``None``
        because its OPEN cursor is already the authoritative pre-commit cursor.
        """

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        batch_size = int(gpu_seats.numel())
        self._validate_vector(gpu_seats, "gpu_seats", batch_size, (torch.int64,))
        self._validate_vector(
            expected_request_epochs,
            "expected_request_epochs",
            batch_size,
            (torch.int64,),
        )
        if pre_verify_seq_lens is not None:
            self._validate_vector(
                pre_verify_seq_lens,
                "pre_verify_seq_lens",
                batch_size,
                (torch.int64,),
            )
        self._validate_vector(
            num_accept_tokens,
            "num_accept_tokens",
            batch_size,
            (torch.int32,),
        )
        accept_token_stride = int(accept_token_stride)
        if not 0 < accept_token_stride <= self.tail_capacity:
            raise ValueError(
                "accept_token_stride must be inside the GPU tail capacity: "
                f"stride={accept_token_stride} capacity={self.tail_capacity}"
            )
        if (
            accept_tokens.device != self.device
            or accept_tokens.dtype != torch.int32
            or not accept_tokens.is_contiguous()
            or accept_tokens.ndim != 1
            or int(accept_tokens.numel()) != batch_size * accept_token_stride
        ):
            raise ValueError(
                "accept_tokens must be a contiguous int32 CUDA vector with "
                f"batch_size * stride values on {self.device}, got "
                f"shape={tuple(accept_tokens.shape)} dtype={accept_tokens.dtype} "
                f"device={accept_tokens.device}"
            )
        if commit_mask is not None:
            self._validate_vector(
                commit_mask,
                "commit_mask",
                batch_size,
                (torch.bool,),
            )

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._init_event)
        self._cpp.apply_verify_commit_from_device_native(
            int(gpu_seats.data_ptr()),
            int(expected_request_epochs.data_ptr()),
            (0 if pre_verify_seq_lens is None else int(pre_verify_seq_lens.data_ptr())),
            int(accept_tokens.data_ptr()),
            int(num_accept_tokens.data_ptr()),
            0 if commit_mask is None else int(commit_mask.data_ptr()),
            accept_token_stride,
            batch_size,
            int(current_stream.cuda_stream),
        )

    def select_snapshot(
        self,
        gpu_seats: torch.Tensor,
        seq_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        *,
        request_epochs: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        out_cursor: torch.Tensor | None = None,
        debug_out: torch.Tensor | None = None,
        required_tail_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select tails; strict mode waits on GPU for the requested live-row length."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        batch_size = int(gpu_seats.numel())
        required_tail_len = (
            self.num_draft_tokens if required_tail_len is None else required_tail_len
        )
        if not 0 <= required_tail_len <= self.num_draft_tokens:
            raise ValueError("required_tail_len is outside the configured draft width")
        self._validate_vector(gpu_seats, "gpu_seats", batch_size, (torch.int64,))
        self._validate_vector(seq_lens, "seq_lens", batch_size, (torch.int64,))
        self._validate_vector(
            bonus_tokens,
            "bonus_tokens",
            batch_size,
            (torch.int32, torch.int64),
        )
        if not self.mock_profile:
            if request_epochs is None:
                raise ValueError(
                    "Real GPU draft-tail selection requires request epochs"
                )
            self._validate_vector(
                request_epochs,
                "request_epochs",
                batch_size,
                (torch.int64,),
            )
        if out is None:
            out = torch.empty(
                (batch_size, self.num_draft_tokens + 2),
                dtype=torch.int64,
                device=self.device,
            )
        self._validate_matrix(
            out,
            "out",
            (batch_size, self.num_draft_tokens + 2),
        )
        if out_cursor is None:
            out_cursor = torch.empty(batch_size, dtype=torch.int64, device=self.device)
        self._validate_vector(out_cursor, "out_cursor", batch_size, (torch.int64,))
        debug_width = 0
        if debug_out is not None:
            if debug_out.ndim != 2 or int(debug_out.shape[0]) != batch_size:
                raise ValueError(
                    "debug_out must have shape [batch_size, width], got "
                    f"shape={tuple(debug_out.shape)} batch_size={batch_size}"
                )
            debug_width = int(debug_out.shape[1])
            if debug_width < GPU_DRAFT_TAIL_DEBUG_WIDTH:
                raise ValueError(
                    "debug_out width must be at least "
                    f"{GPU_DRAFT_TAIL_DEBUG_WIDTH}, got {debug_width}"
                )
            self._validate_matrix(
                debug_out,
                "debug_out",
                (batch_size, debug_width),
            )

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._init_event)
        if self.mock_profile:
            self._cpp.select_mock_snapshot_native(
                int(gpu_seats.data_ptr()),
                int(seq_lens.data_ptr()),
                int(bonus_tokens.data_ptr()),
                bonus_tokens.dtype == torch.int32,
                int(out.data_ptr()),
                int(out_cursor.data_ptr()),
                0 if debug_out is None else int(debug_out.data_ptr()),
                debug_width,
                batch_size,
                int(current_stream.cuda_stream),
            )
        else:
            assert request_epochs is not None
            self._cpp.select_snapshot_native(
                int(gpu_seats.data_ptr()),
                int(request_epochs.data_ptr()),
                int(seq_lens.data_ptr()),
                int(bonus_tokens.data_ptr()),
                bonus_tokens.dtype == torch.int32,
                int(out.data_ptr()),
                int(out_cursor.data_ptr()),
                0 if debug_out is None else int(debug_out.data_ptr()),
                debug_width,
                batch_size,
                int(current_stream.cuda_stream),
                self.allow_partial,
                required_tail_len,
            )
        return out, out_cursor

    def prepare_decode(
        self,
        mirror_seats: torch.Tensor,
        request_epochs: torch.Tensor,
        req_pool_indices: torch.Tensor,
        candidate_out_cache_locs: torch.Tensor,
        checkpoint_slot_table: torch.Tensor | None,
        req_to_token: torch.Tensor,
        *,
        resolved_input_ids: torch.Tensor,
        resolved_seq_lens: torch.Tensor,
        resolved_orig_seq_lens: torch.Tensor,
        mamba_src_indices: torch.Tensor | None,
        mamba_dst_indices: torch.Tensor | None,
        captured_state_positions: torch.Tensor,
        old_cache_locs: torch.Tensor,
        validate_inputs: bool = True,
    ) -> None:
        """Resolve and bind one drafter decode batch on the caller stream."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        if not self.drafter_authoritative:
            raise RuntimeError("prepare_decode requires a drafter GPU control mirror")
        batch_size = int(mirror_seats.numel())
        if not (
            (checkpoint_slot_table is None)
            == (mamba_src_indices is None)
            == (mamba_dst_indices is None)
        ):
            raise ValueError(
                "Recurrent checkpoint table and routes must be supplied together"
            )
        if validate_inputs:
            for tensor, name in (
                (mirror_seats, "mirror_seats"),
                (request_epochs, "request_epochs"),
                (req_pool_indices, "req_pool_indices"),
                (candidate_out_cache_locs, "candidate_out_cache_locs"),
                (resolved_input_ids, "resolved_input_ids"),
                (resolved_seq_lens, "resolved_seq_lens"),
                (captured_state_positions, "captured_state_positions"),
                (old_cache_locs, "old_cache_locs"),
            ):
                self._validate_vector(tensor, name, batch_size, (torch.int64,))
            if checkpoint_slot_table is not None:
                self._validate_vector(
                    mamba_src_indices, "mamba_src_indices", batch_size, (torch.int64,)
                )
                self._validate_vector(
                    mamba_dst_indices, "mamba_dst_indices", batch_size, (torch.int64,)
                )
            self._validate_vector(
                resolved_orig_seq_lens,
                "resolved_orig_seq_lens",
                batch_size,
                (torch.int32,),
            )
        # Pool tables have fixed storage throughout an engine lifetime. Retain
        # their owners and resolve their layout once; per-batch vectors above
        # still require validation on every call.
        if (
            checkpoint_slot_table is not None
            and checkpoint_slot_table is not self._decode_checkpoint_table
        ):
            self._validate_matrix(
                checkpoint_slot_table,
                "checkpoint_slot_table",
                (self.num_seats, self.tail_capacity),
            )
            self._decode_checkpoint_table = checkpoint_slot_table
            self._decode_checkpoint_ptr = checkpoint_slot_table.data_ptr()
        if req_to_token is not self._decode_req_table:
            self._validate_req_to_token(req_to_token)
            self._decode_req_table = req_to_token
            self._decode_req_layout = (req_to_token.data_ptr(), *req_to_token.shape)
        current_stream = torch.cuda.current_stream(self.device)
        stream_handle = current_stream.cuda_stream
        if stream_handle not in self._decode_initialized_streams:
            current_stream.wait_event(self._init_event)
            self._decode_initialized_streams[stream_handle] = current_stream
        self._cpp.prepare_decode_native(
            int(mirror_seats.data_ptr()),
            int(request_epochs.data_ptr()),
            int(req_pool_indices.data_ptr()),
            int(candidate_out_cache_locs.data_ptr()),
            0 if checkpoint_slot_table is None else self._decode_checkpoint_ptr,
            self.tail_capacity,
            *self._decode_req_layout,
            int(resolved_input_ids.data_ptr()),
            int(resolved_seq_lens.data_ptr()),
            int(resolved_orig_seq_lens.data_ptr()),
            0 if mamba_src_indices is None else int(mamba_src_indices.data_ptr()),
            0 if mamba_dst_indices is None else int(mamba_dst_indices.data_ptr()),
            int(captured_state_positions.data_ptr()),
            int(old_cache_locs.data_ptr()),
            int(self._decode_kv_ownership.data_ptr()),
            batch_size,
            int(current_stream.cuda_stream),
        )

    def finish_decode(
        self,
        mirror_seats: torch.Tensor,
        request_epochs: torch.Tensor,
        req_pool_indices: torch.Tensor,
        candidate_out_cache_locs: torch.Tensor,
        sampled_tokens: torch.Tensor,
        req_to_token: torch.Tensor,
        *,
        resolved_input_tokens: torch.Tensor,
        captured_state_positions: torch.Tensor,
        old_cache_locs: torch.Tensor,
        kv_outcomes: torch.Tensor,
        future_output_tokens: torch.Tensor,
        validate_inputs: bool = True,
    ) -> None:
        """Linearize a sample against its prepared position and input token.

        resolved_input_tokens must retain the prepare output until this call
        executes. A BS1 FutureMap view is allowed: finish reads it before
        updating the relay slot on the same forward stream.
        """

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        if not self.drafter_authoritative:
            raise RuntimeError("finish_decode requires a drafter GPU control mirror")
        batch_size = int(mirror_seats.numel())
        if validate_inputs:
            for tensor, name in (
                (mirror_seats, "mirror_seats"),
                (request_epochs, "request_epochs"),
                (req_pool_indices, "req_pool_indices"),
                (candidate_out_cache_locs, "candidate_out_cache_locs"),
                (resolved_input_tokens, "resolved_input_tokens"),
                (captured_state_positions, "captured_state_positions"),
                (old_cache_locs, "old_cache_locs"),
            ):
                self._validate_vector(tensor, name, batch_size, (torch.int64,))
            self._validate_vector(
                sampled_tokens,
                "sampled_tokens",
                batch_size,
                (torch.int32, torch.int64),
            )
            self._validate_matrix(
                kv_outcomes,
                "kv_outcomes",
                (batch_size, 3),
            )
            self._validate_vector(
                future_output_tokens,
                "future_output_tokens",
                int(future_output_tokens.numel()),
                (torch.int64,),
            )
        if req_to_token is not self._decode_req_table:
            self._validate_req_to_token(req_to_token)
            self._decode_req_table = req_to_token
            self._decode_req_layout = (req_to_token.data_ptr(), *req_to_token.shape)
        current_stream = torch.cuda.current_stream(self.device)
        stream_handle = current_stream.cuda_stream
        if stream_handle not in self._decode_initialized_streams:
            current_stream.wait_event(self._init_event)
            self._decode_initialized_streams[stream_handle] = current_stream
        self._cpp.finish_decode_native(
            int(mirror_seats.data_ptr()),
            int(request_epochs.data_ptr()),
            int(req_pool_indices.data_ptr()),
            int(candidate_out_cache_locs.data_ptr()),
            int(sampled_tokens.data_ptr()),
            sampled_tokens.dtype == torch.int32,
            *self._decode_req_layout,
            int(resolved_input_tokens.data_ptr()),
            int(captured_state_positions.data_ptr()),
            int(old_cache_locs.data_ptr()),
            int(kv_outcomes.data_ptr()),
            int(future_output_tokens.data_ptr()),
            int(future_output_tokens.numel()),
            batch_size,
            int(current_stream.cuda_stream),
        )

    def append_prefill_sample(
        self,
        mirror_seats: torch.Tensor,
        request_epochs: torch.Tensor,
        sampled_tokens: torch.Tensor,
        *,
        accept_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Initialize model input and checkpoint tag after final prefill.

        If verifier commits landed after the prefill launch, ``sampled_tokens``
        is overwritten in place with the first forced token so the ordinary
        final-prefill FutureMap/result path relays the authoritative value.
        """

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        if not self.drafter_authoritative:
            raise RuntimeError(
                "append_prefill_sample requires a drafter GPU control mirror"
            )
        batch_size = int(mirror_seats.numel())
        self._validate_vector(mirror_seats, "mirror_seats", batch_size, (torch.int64,))
        self._validate_vector(
            request_epochs, "request_epochs", batch_size, (torch.int64,)
        )
        self._validate_vector(
            sampled_tokens,
            "sampled_tokens",
            batch_size,
            (torch.int32, torch.int64),
        )
        if accept_out is None:
            accept_out = torch.empty(batch_size, dtype=torch.bool, device=self.device)
        self._validate_vector(accept_out, "accept_out", batch_size, (torch.bool,))
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._init_event)
        self._cpp.append_prefill_sample_native(
            int(mirror_seats.data_ptr()),
            int(request_epochs.data_ptr()),
            int(sampled_tokens.data_ptr()),
            sampled_tokens.dtype == torch.int32,
            int(accept_out.data_ptr()),
            batch_size,
            int(current_stream.cuda_stream),
        )
        return accept_out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cpp.close()

    def _validate_vector(
        self,
        tensor: torch.Tensor,
        name: str,
        size: int,
        dtypes: tuple[torch.dtype, ...],
    ) -> None:
        if (
            tensor.device != self.device
            or tensor.dtype not in dtypes
            or not tensor.is_contiguous()
            or tensor.ndim != 1
            or int(tensor.numel()) != size
        ):
            raise ValueError(
                f"{name} must be a contiguous {dtypes} CUDA vector on "
                f"{self.device} with {size} values, got "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device}"
            )

    def _validate_matrix(
        self,
        tensor: torch.Tensor,
        name: str,
        shape: tuple[int, int],
    ) -> None:
        if (
            tensor.device != self.device
            or tensor.dtype != torch.int64
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(
                f"{name} must be a contiguous int64 CUDA tensor on "
                f"{self.device} with shape {shape}, got "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device}"
            )

    def _validate_req_to_token(self, req_to_token: torch.Tensor) -> None:
        if (
            req_to_token.device != self.device
            or req_to_token.dtype != torch.int32
            or not req_to_token.is_contiguous()
            or req_to_token.ndim != 2
            or int(req_to_token.shape[0]) <= 0
            or int(req_to_token.shape[1]) <= 0
        ):
            raise ValueError(
                "req_to_token must be a non-empty contiguous int32 CUDA matrix "
                f"on {self.device}, got shape={tuple(req_to_token.shape)} "
                f"dtype={req_to_token.dtype} device={req_to_token.device}"
            )


__all__ = [
    "GPU_DRAFT_TAIL_DEBUG_WIDTH",
    "GPU_DRAFT_TAIL_DEBUG_FIELD_NAMES",
    "GPU_DRAFT_TAIL_SELECT_REASON_NAMES",
    "GPU_DRAFT_TAIL_UPDATE_ERROR_NAMES",
    "CppGpuDraftTailBuffer",
]

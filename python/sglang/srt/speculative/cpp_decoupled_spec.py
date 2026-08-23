from __future__ import annotations

import glob
import logging
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
    VerifierCommitSegment,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailSnapshot

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.0005


@lru_cache(maxsize=1)
def _load_decoupled_spec_cpp_module():
    _ensure_zmq_lib_path()
    source_dir = Path(__file__).resolve().parent / "csrc" / "decoupled_spec"
    sources = [
        source_dir / "decoupled_spec_pybind.cpp",
        source_dir / "gpu_draft_tail.cu",
    ]
    logger.info(
        "Loading the decoupled-spec C++ data plane; JIT compilation may be "
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


def _sync_row(message: DraftSync) -> tuple[str, int, int, list[int], list[int]]:
    return (
        str(message.request_id),
        int(message.src_verifier_rank),
        int(message.dst_drafter_rank),
        [int(token) for token in message.prompt_token_ids],
        [int(token) for token in message.committed_outputs],
    )


def _commit_row(message: VerifyCommit) -> tuple[str, int, int, int, list[int]]:
    return (
        str(message.request_id),
        int(message.src_verifier_rank),
        int(message.dst_drafter_rank),
        int(message.pre_verify_committed_len),
        [int(token) for token in message.committed_tokens],
    )


def _close_row(message: DraftClose) -> tuple[str, int, int, str]:
    return (
        str(message.request_id),
        int(message.src_verifier_rank),
        int(message.dst_drafter_rank),
        str(message.reason),
    )


def _tail_row(
    output: DraftTailStreamOutput,
) -> tuple[int, int, str, int, int, int]:
    return (
        int(output.src_drafter_rank),
        int(output.dst_verifier_rank),
        str(output.request_id),
        int(output.base_committed_len),
        int(output.new_token_pos),
        int(output.new_token),
    )


def _sync_from_native(row: Sequence[Any]) -> DraftSync:
    return DraftSync(
        request_id=str(row[0]),
        src_verifier_rank=int(row[1]),
        dst_drafter_rank=int(row[2]),
        prompt_token_ids=[int(token) for token in row[3]],
        committed_outputs=[int(token) for token in row[4]],
    )


def _commit_from_native(row: Sequence[Any]) -> VerifyCommit:
    return VerifyCommit(
        request_id=str(row[0]),
        src_verifier_rank=int(row[1]),
        dst_drafter_rank=int(row[2]),
        pre_verify_committed_len=int(row[3]),
        committed_tokens=[int(token) for token in row[4]],
    )


def _close_from_native(row: Sequence[Any]) -> DraftClose:
    return DraftClose(
        request_id=str(row[0]),
        src_verifier_rank=int(row[1]),
        dst_drafter_rank=int(row[2]),
        reason=str(row[3]),
    )


def _segment_from_native(row: Sequence[Any]) -> VerifierCommitSegment:
    return VerifierCommitSegment(
        draft_key=DraftReqKey(
            src_verifier_rank=int(row[1]),
            request_id=str(row[0]),
        ),
        dst_drafter_rank=int(row[2]),
        pre_verify_committed_len=int(row[3]),
        committed_tokens=[int(token) for token in row[4]],
    )


class CppGpuDraftTailBuffer:
    """One rolling GPU tail row per request-pool seat.

    The native verifier daemon is the only tail writer. The verify stream
    calls :meth:`select_snapshot` immediately before target verification and
    materializes a fixed-shape, forward-local snapshot without a host read.
    """

    def __init__(
        self,
        *,
        device: str | torch.device,
        num_seats: int,
        num_draft_tokens: int,
        landing_stream: torch.cuda.Stream,
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
        self.tail_capacity = 2 * self.num_draft_tokens + 1
        self.landing_stream = landing_stream
        with torch.cuda.device(device):
            self.versions = torch.zeros(
                self.num_seats, dtype=torch.int64, device=device
            )
            self.publish_seqs = torch.zeros_like(self.versions)
            self.request_epochs = torch.full_like(self.versions, -1)
            self.active_request_epochs = torch.full_like(self.versions, -1)
            self.prompt_lens = torch.full_like(self.versions, -1)
            self.committed_lens = torch.full_like(self.versions, -1)
            self.raw_tail_lens = torch.zeros_like(self.versions)
            self.consumable_tail_lens = torch.zeros_like(self.versions)
            self.tail_tokens = torch.zeros(
                (self.num_seats, self.tail_capacity),
                dtype=torch.int64,
                device=device,
            )
            self._init_event = torch.cuda.Event()
            self._init_event.record(torch.cuda.current_stream(device))
            self.landing_stream.wait_event(self._init_event)

        self._cpp = _load_decoupled_spec_cpp_module().GpuDraftTailBuffer(
            int(device.index),
            self.num_seats,
            self.num_draft_tokens,
            int(self.landing_stream.cuda_stream),
            int(self.versions.data_ptr()),
            int(self.publish_seqs.data_ptr()),
            int(self.request_epochs.data_ptr()),
            int(self.active_request_epochs.data_ptr()),
            int(self.prompt_lens.data_ptr()),
            int(self.committed_lens.data_ptr()),
            int(self.raw_tail_lens.data_ptr()),
            int(self.consumable_tail_lens.data_ptr()),
            int(self.tail_tokens.data_ptr()),
        )
        self._closed = False

    @property
    def staging_slot_count(self) -> int:
        """Number of landing slots allocated so far (initially eight)."""

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
        # Lifecycle metadata is updated once per seat assignment on the
        # scheduler stream. This is not a per-round host-to-device dependency.
        with torch.cuda.device(self.device):
            self.active_request_epochs[gpu_seat] = request_epoch

    def select_snapshot(
        self,
        gpu_seats: torch.Tensor,
        seq_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        *,
        request_epochs: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        out_cursor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and materialize tails on the caller's current CUDA stream."""

        if self._closed:
            raise RuntimeError("GPU draft-tail buffer is closed")
        batch_size = int(gpu_seats.numel())
        self._validate_vector(gpu_seats, "gpu_seats", batch_size, (torch.int64,))
        self._validate_vector(seq_lens, "seq_lens", batch_size, (torch.int64,))
        self._validate_vector(
            bonus_tokens,
            "bonus_tokens",
            batch_size,
            (torch.int32, torch.int64),
        )
        if request_epochs is not None:
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

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(self._init_event)
        self._cpp.select_snapshot_native(
            int(gpu_seats.data_ptr()),
            0 if request_epochs is None else int(request_epochs.data_ptr()),
            int(seq_lens.data_ptr()),
            int(bonus_tokens.data_ptr()),
            bonus_tokens.dtype == torch.int32,
            int(out.data_ptr()),
            int(out_cursor.data_ptr()),
            batch_size,
            int(current_stream.cuda_stream),
        )
        return out, out_cursor

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


class CppDraftTailBuffer:
    """Python-contract facade over the linearizable native rolling tail."""

    def __init__(self, *, verifier_rank: int, required_tail_len: int = 0) -> None:
        self.verifier_rank = int(verifier_rank)
        self.required_tail_len = max(0, int(required_tail_len))
        self._cpp = _load_decoupled_spec_cpp_module().DraftTailBuffer(
            self.verifier_rank, self.required_tail_len
        )
        self.last_draft_wait_ns = 0

    def close(self) -> None:
        self._cpp.close()

    def has_request(self, request_id: str) -> bool:
        return bool(self._cpp.has_request(str(request_id)))

    def get_committed_len(self, request_id: str) -> int | None:
        value = int(self._cpp.get_committed_len(str(request_id)))
        return None if value < 0 else value

    def open_request(self, message: DraftSync) -> None:
        self.open_requests([message])

    def open_requests(self, messages: Sequence[DraftSync]) -> None:
        if not messages:
            return
        dst_drafter_rank = int(messages[0].dst_drafter_rank)
        batch = DraftControlBatch(
            dst_drafter_rank=dst_drafter_rank,
            sync_messages=list(messages),
        )
        self.apply_control_batch(batch)

    def apply_verify_commit(self, message: VerifyCommit) -> None:
        self.apply_verify_commits([message])

    def apply_verify_commits(self, messages: Sequence[VerifyCommit]) -> None:
        if not messages:
            return
        dst_drafter_rank = int(messages[0].dst_drafter_rank)
        batch = DraftControlBatch(
            dst_drafter_rank=dst_drafter_rank,
            verify_commit_messages=list(messages),
        )
        self.apply_control_batch(batch)

    def close_request(self, message: DraftClose) -> None:
        self.close_requests([message])

    def close_requests(self, messages: Sequence[DraftClose]) -> None:
        if not messages:
            return
        dst_drafter_rank = int(messages[0].dst_drafter_rank)
        batch = DraftControlBatch(
            dst_drafter_rank=dst_drafter_rank,
            close_messages=list(messages),
        )
        self.apply_control_batch(batch)

    def apply_control_batch(self, batch: DraftControlBatch) -> None:
        self._validate_control_batch(batch)
        self._cpp.apply_control_batch_native(
            int(batch.dst_drafter_rank),
            [_sync_row(message) for message in batch.sync_messages],
            [_commit_row(message) for message in batch.verify_commit_messages],
            [_close_row(message) for message in batch.close_messages],
        )

    def append_draft_stream_batch(self, batch: DraftTailStreamOutputBatch) -> None:
        for output in batch.outputs:
            if int(output.dst_verifier_rank) != self.verifier_rank:
                raise RuntimeError(
                    "Draft stream output targets a different verifier: "
                    f"expected_verifier_rank={self.verifier_rank} "
                    f"dst_verifier_rank={output.dst_verifier_rank}"
                )
        if batch.outputs:
            self._cpp.append_draft_stream_batch_native(
                [_tail_row(output) for output in batch.outputs]
            )

    def snapshot(
        self,
        request_id: str,
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> DraftTailSnapshot:
        return self.get_draft_snapshots(
            [request_id],
            allow_partial=allow_partial,
            max_tail_len=max_tail_len,
            timeout_s=timeout_s,
        )[0]

    def get_draft_snapshots(
        self,
        request_ids: Sequence[str],
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> list[DraftTailSnapshot]:
        request_ids = [str(request_id) for request_id in request_ids]
        tail_cap = -1 if max_tail_len is None else max(0, int(max_tail_len))

        if allow_partial or timeout_s is None:
            rows, wait_ns = self._get_draft_snapshots_native(
                request_ids,
                allow_partial=bool(allow_partial),
                tail_cap=tail_cap,
            )
            self.last_draft_wait_ns = int(wait_ns)
            return self._snapshots_from_native(rows)

        deadline = time.monotonic() + float(timeout_s)
        required_tail_len = self.required_tail_len
        if tail_cap >= 0:
            required_tail_len = min(required_tail_len, tail_cap)
        min_tail_len = max(0 if tail_cap == 0 else 1, required_tail_len)
        wait_start_ns = time.perf_counter_ns()
        while True:
            rows, _ = self._get_draft_snapshots_native(
                request_ids,
                allow_partial=True,
                tail_cap=tail_cap,
            )
            if all(int(row[4]) >= min_tail_len for row in rows):
                self.last_draft_wait_ns = time.perf_counter_ns() - wait_start_ns
                return self._snapshots_from_native(rows)
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                self.last_draft_wait_ns = time.perf_counter_ns() - wait_start_ns
                raise TimeoutError(
                    "Timed out waiting for decoupled draft tails: "
                    f"request_ids={request_ids} min_tail_len={min_tail_len}"
                )
            time.sleep(min(_IDLE_WAIT_S, remaining_s))

    def _get_draft_snapshots_native(
        self,
        request_ids: Sequence[str],
        *,
        allow_partial: bool,
        tail_cap: int,
    ) -> tuple[Any, int]:
        try:
            return self._cpp.get_draft_snapshots_native(
                request_ids, allow_partial, tail_cap
            )
        except RuntimeError as exc:
            if "unexpected request_id=" not in str(exc):
                raise
            missing_request_ids = [
                request_id
                for request_id in request_ids
                if not self.has_request(request_id)
            ]
            raise KeyError(
                "Missing decoupled draft-tail requests: "
                f"request_ids={missing_request_ids}"
            ) from exc

    @staticmethod
    def _snapshots_from_native(
        rows: Sequence[Sequence[Any]],
    ) -> list[DraftTailSnapshot]:
        return [
            DraftTailSnapshot(
                request_id=str(row[0]),
                committed_len=int(row[1]),
                tail_tokens=tuple(int(token) for token in row[2]),
                raw_tail_len=int(row[3]),
                num_consumable_drafts=int(row[4]),
            )
            for row in rows
        ]

    def _validate_control_batch(self, batch: DraftControlBatch) -> None:
        dst_drafter_rank = int(batch.dst_drafter_rank)
        for message in (
            *batch.sync_messages,
            *batch.verify_commit_messages,
            *batch.close_messages,
        ):
            if int(message.src_verifier_rank) != self.verifier_rank:
                raise RuntimeError(
                    "Verifier control source rank mismatch: "
                    f"expected_verifier_rank={self.verifier_rank} "
                    f"src_verifier_rank={message.src_verifier_rank}"
                )
            if int(message.dst_drafter_rank) != dst_drafter_rank:
                raise RuntimeError(
                    "Verifier control destination rank mismatch: "
                    f"batch_drafter_rank={dst_drafter_rank} "
                    f"message_drafter_rank={message.dst_drafter_rank}"
                )


class CppVerifierDecoupledSpecDataPlane:
    """Native verifier transport with synchronous local control application."""

    def __init__(
        self,
        config: DecoupledSpecIpcConfig,
        *,
        required_tail_len: int = 0,
        context: Any | None = None,
        device: str | torch.device | None = None,
        num_gpu_seats: int | None = None,
        num_draft_tokens: int | None = None,
        landing_stream: torch.cuda.Stream | None = None,
    ) -> None:
        # Keep an injected pyzmq context alive and let native sockets share its
        # underlying libzmq context. This preserves inproc test/deployment
        # semantics without transferring socket ownership across languages.
        self._context = context
        external_context = 0 if context is None else int(context.underlying)
        self.config = config
        self._peer_ranks = frozenset(int(peer.rank) for peer in config.peers)
        self.draft_tail_buffer = CppDraftTailBuffer(
            verifier_rank=int(config.rank),
            required_tail_len=required_tail_len,
        )
        gpu_config = (device, num_gpu_seats, num_draft_tokens, landing_stream)
        if any(value is not None for value in gpu_config) and not all(
            value is not None for value in gpu_config
        ):
            raise ValueError(
                "device, num_gpu_seats, num_draft_tokens, and landing_stream "
                "must be supplied together for the GPU draft-tail path"
            )
        self.gpu_tail_buffer = (
            CppGpuDraftTailBuffer(
                device=device,
                num_seats=int(num_gpu_seats),
                num_draft_tokens=int(num_draft_tokens),
                landing_stream=landing_stream,
            )
            if device is not None
            else None
        )
        if self.gpu_tail_buffer is not None:
            self.draft_tail_buffer._cpp.attach_gpu_tail_buffer(
                self.gpu_tail_buffer._cpp
            )
        self._transport = _load_decoupled_spec_cpp_module().DraftProxyThread(
            int(config.rank),
            str(config.bind_endpoint),
            _peer_rows(config),
            self.draft_tail_buffer._cpp,
            external_context,
        )
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("A closed decoupled-spec data plane cannot be restarted")
        if not self._started:
            self._transport.start()
            self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._transport.close()
        finally:
            try:
                self.draft_tail_buffer.close()
            finally:
                if self.gpu_tail_buffer is not None:
                    self.gpu_tail_buffer.close()

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        self._ensure_running()
        self._validate_peer_rank(int(batch.dst_drafter_rank))
        self.draft_tail_buffer._validate_control_batch(batch)
        # The native method applies the batch to DraftTailBuffer under its lock
        # before placing the same encoded batch on the network send queue.
        self._transport.submit_control_batch_native(
            int(batch.dst_drafter_rank),
            [_sync_row(message) for message in batch.sync_messages],
            [_commit_row(message) for message in batch.verify_commit_messages],
            [_close_row(message) for message in batch.close_messages],
        )

    def open_request(
        self,
        message: DraftSync,
        *,
        gpu_seat: int | None = None,
        request_epoch: int | None = None,
    ) -> None:
        self.open_requests([(message, gpu_seat, request_epoch)])

    def open_requests(
        self,
        requests: Sequence[tuple[DraftSync, int | None, int | None]],
    ) -> None:
        """Bind and publish one destination-homogeneous request-open batch."""

        if not requests:
            return
        dst_drafter_rank = int(requests[0][0].dst_drafter_rank)
        control_batch = DraftControlBatch(
            dst_drafter_rank=dst_drafter_rank,
            sync_messages=[message for message, _, _ in requests],
        )
        self._validate_peer_rank(dst_drafter_rank)
        self.draft_tail_buffer._validate_control_batch(control_batch)
        if self.gpu_tail_buffer is not None:
            for message, gpu_seat, request_epoch in requests:
                if gpu_seat is None or request_epoch is None:
                    raise ValueError(
                        "GPU draft-tail open requires gpu_seat and request_epoch"
                    )
                self.gpu_tail_buffer.bind_request(
                    str(message.request_id), int(gpu_seat), int(request_epoch)
                )
        elif any(
            gpu_seat is not None or request_epoch is not None
            for _, gpu_seat, request_epoch in requests
        ):
            raise ValueError(
                "gpu_seat/request_epoch require an enabled GPU draft-tail buffer"
            )
        self.submit_control_batch(control_batch)

    def commit(self, message: VerifyCommit) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                verify_commit_messages=[message],
            )
        )

    def close_request(self, message: DraftClose) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                close_messages=[message],
            )
        )

    def snapshot(
        self,
        request_ids: Sequence[str] | str,
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> list[DraftTailSnapshot]:
        self._ensure_running()
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        return self.draft_tail_buffer.get_draft_snapshots(
            request_ids,
            allow_partial=allow_partial,
            max_tail_len=max_tail_len,
            timeout_s=timeout_s,
        )

    def snapshot_one(
        self,
        request_id: str,
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> DraftTailSnapshot:
        return self.snapshot(
            [request_id],
            allow_partial=allow_partial,
            max_tail_len=max_tail_len,
            timeout_s=timeout_s,
        )[0]

    def snapshot_many(
        self,
        request_ids: Sequence[str],
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> list[DraftTailSnapshot]:
        return self.snapshot(
            request_ids,
            allow_partial=allow_partial,
            max_tail_len=max_tail_len,
            timeout_s=timeout_s,
        )

    def _ensure_running(self) -> None:
        if not self._started:
            raise RuntimeError("Decoupled-spec transport has not been started")
        if self._closed:
            raise RuntimeError("Decoupled-spec transport is closed")

    def _validate_peer_rank(self, rank: int) -> None:
        if rank not in self._peer_ranks:
            raise RuntimeError(
                "Missing decoupled-spec peer endpoint: "
                f"peer_rank={rank} "
                f"configured_peer_ranks={sorted(self._peer_ranks)}"
            )


class CppDrafterDecoupledSpecDataPlane:
    """Native drafter inbox and tail publisher with the Python role contract."""

    def __init__(
        self,
        config: DecoupledSpecIpcConfig,
        *,
        context: Any | None = None,
    ) -> None:
        self._context = context
        external_context = 0 if context is None else int(context.underlying)
        self.config = config
        self._peer_ranks = frozenset(int(peer.rank) for peer in config.peers)
        self._transport = _load_decoupled_spec_cpp_module().TokenSyncThread(
            int(config.rank),
            str(config.bind_endpoint),
            _peer_rows(config),
            external_context,
        )
        self._started = False
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("A closed decoupled-spec data plane cannot be restarted")
        if not self._started:
            self._transport.start()
            self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._transport.close()

    def drain_controls(self, max_batches: int | None = None) -> list[DraftControlBatch]:
        self._ensure_running()
        if max_batches is not None and int(max_batches) < 0:
            raise ValueError(f"max_batches must be non-negative, got {max_batches}")
        limit = -1 if max_batches is None else int(max_batches)
        batches = []
        for (
            dst_drafter_rank,
            sync_rows,
            commit_rows,
            close_rows,
        ) in self._transport.drain_control_batches_native(limit):
            batches.append(
                DraftControlBatch(
                    dst_drafter_rank=int(dst_drafter_rank),
                    sync_messages=[_sync_from_native(row) for row in sync_rows],
                    verify_commit_messages=[
                        _commit_from_native(row) for row in commit_rows
                    ],
                    close_messages=[_close_from_native(row) for row in close_rows],
                )
            )
        return batches

    def pending_control_count(self) -> int:
        return int(self._transport.pending_control_count())

    def wait_for_control(self, timeout_s: float) -> bool:
        self._ensure_running()
        timeout_us = max(0, int(float(timeout_s) * 1_000_000))
        return bool(self._transport.wait_for_pending_control(timeout_us))

    def collect_ready_controls(
        self,
        consumable_commit_len: Callable[[VerifierCommitSegment], int],
    ) -> ReadyDraftControls:
        """Consume independent ready prefixes from the native per-key inbox."""

        self._ensure_running()
        # Keep the legacy raw-batch compatibility queue bounded. Network ingress
        # already inserted these controls into the native segmented inbox.
        self._transport.drain_control_batches_native(-1)
        pending_segments = [
            _segment_from_native(row)
            for row in self._transport.snapshot_pending_commit_segments_native()
        ]
        decisions = [
            (
                segment.draft_key.request_id,
                int(segment.draft_key.src_verifier_rank),
                int(segment.dst_drafter_rank),
                int(segment.pre_verify_committed_len),
                int(consumable_commit_len(segment)),
            )
            for segment in pending_segments
        ]
        sync_rows, close_rows, segment_rows = (
            self._transport.extract_ready_controls_native(decisions)
        )
        return ReadyDraftControls(
            sync_messages=[_sync_from_native(row) for row in sync_rows],
            close_keys={
                DraftReqKey(
                    src_verifier_rank=int(row[1]),
                    request_id=str(row[0]),
                )
                for row in close_rows
            },
            ready_commit_segments=[_segment_from_native(row) for row in segment_rows],
        )

    def publish_tail(self, output: DraftTailStreamOutput) -> None:
        self.publish_tails(DraftTailStreamOutputBatch(outputs=[output]))

    def publish_tails(self, batch: DraftTailStreamOutputBatch) -> None:
        self._ensure_running()
        for output in batch.outputs:
            if int(output.src_drafter_rank) != int(self.config.rank):
                raise RuntimeError(
                    "Draft tail source rank mismatch: "
                    f"expected_drafter_rank={self.config.rank} "
                    f"src_drafter_rank={output.src_drafter_rank}"
                )
            self._validate_peer_rank(int(output.dst_verifier_rank))
        if batch.outputs:
            self._transport.submit_draft_results_native(
                [_tail_row(output) for output in batch.outputs]
            )

    def _ensure_running(self) -> None:
        if not self._started:
            raise RuntimeError("Decoupled-spec transport has not been started")
        if self._closed:
            raise RuntimeError("Decoupled-spec transport is closed")

    def _validate_peer_rank(self, rank: int) -> None:
        if rank not in self._peer_ranks:
            raise RuntimeError(
                "Missing decoupled-spec peer endpoint: "
                f"peer_rank={rank} "
                f"configured_peer_ranks={sorted(self._peer_ranks)}"
            )


def _peer_rows(config: DecoupledSpecIpcConfig) -> list[tuple[int, str]]:
    return [(int(peer.rank), str(peer.endpoint)) for peer in config.peers]


__all__ = [
    "CppGpuDraftTailBuffer",
    "CppDraftTailBuffer",
    "CppDrafterDecoupledSpecDataPlane",
    "CppVerifierDecoupledSpecDataPlane",
]

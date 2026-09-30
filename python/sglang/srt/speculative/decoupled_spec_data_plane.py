"""Role APIs and socket transports for the shared decoupled-spec GPU backend.

The backend owns wire encoding, bounded queues, GPU landing/egress and request
semantics. Each role can use native threads/libzmq or Python threads/pyzmq.
Neither transport maintains a second token transcript or GPU state machine.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import torch
import zmq

from sglang.srt.environ import envs
from sglang.srt.speculative.cpp_decoupled_spec import (
    CppGpuDraftTailBuffer,
    _load_decoupled_spec_cpp_module,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
    VerifyCommit,
)


class PythonZmqTransport:
    """Own sockets on one Python thread; share the native GPU/codec backend."""

    def __init__(
        self,
        backend,
        config: DecoupledSpecIpcConfig,
        context: zmq.Context | None,
        *,
        role: str,
    ) -> None:
        self._backend = backend
        self._config = config
        self._context = context
        if role not in ("verifier", "drafter"):
            raise ValueError(f"Unknown decoupled-spec transport role: {role}")
        self._role = role
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._send_sockets: dict[int, zmq.Socket] = {}

    def __getattr__(self, name):
        # Role APIs operate on the same backend queues as the socket thread.
        self.raise_if_failed()
        return getattr(self._backend, name)

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"Decoupled-spec Python transport failed: {self._error}"
            ) from self._error

    def start(self) -> None:
        if self._closed.is_set():
            raise RuntimeError("A closed decoupled-spec transport cannot be restarted")
        if self._thread is not None:
            self.raise_if_failed()
            return
        self._backend.start()  # External-I/O mode starts no native thread/socket.
        self._thread = threading.Thread(
            target=self._run,
            name=f"dspec-python-{self._role}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(5):
            self.close()
            raise TimeoutError("Timed out starting decoupled-spec Python transport")
        self.raise_if_failed()

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Decoupled-spec Python transport did not stop")
        self._backend.close()

    def _send(self, rank: int, frame: bytes) -> bool:
        try:
            self._send_sockets[rank].send(frame, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def _run(self) -> None:
        context = self._context if self._context is not None else zmq.Context()
        recv = None
        try:
            recv = context.socket(zmq.PULL)
            recv.setsockopt(zmq.LINGER, 0)
            recv.setsockopt(zmq.RCVHWM, 8192)
            recv.setsockopt(zmq.RCVBUF, 512 * 1024 * 1024)
            if "[" in self._config.bind_endpoint:
                recv.setsockopt(zmq.IPV6, 1)
            recv.bind(self._config.bind_endpoint)
            for peer in self._config.peers:
                socket = context.socket(zmq.PUSH)
                self._send_sockets[int(peer.rank)] = socket
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.IMMEDIATE, 1)
                socket.setsockopt(zmq.SNDHWM, 8192)
                socket.setsockopt(zmq.SNDBUF, 512 * 1024 * 1024)
                if "[" in peer.endpoint:
                    socket.setsockopt(zmq.IPV6, 1)
                socket.connect(peer.endpoint)
            self._ready.set()
            backend = self._backend
            send = self._send
            while not self._closed.is_set():
                did_work = False
                if self._role == "drafter":
                    did_work = backend.poll_gpu_egress()
                # Keep the queue front until send succeeds. A backpressured peer
                # must not prevent incoming commits or GPU egress from progressing.
                did_work = backend.send_pending(send) or did_work
                for _ in range(64):
                    try:
                        frame = recv.recv(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if self._role == "verifier":
                        backend.receive_frame(frame)
                    else:
                        backend.receive_frame(frame, send)
                    did_work = True
                if self._role == "verifier":
                    did_work = backend.send_clock_probe(send) or did_work
                if not did_work:
                    self._closed.wait(0.00005)
        except BaseException as error:
            self._error = error
        finally:
            self._ready.set()
            for socket in self._send_sockets.values():
                socket.close(linger=0)
            if recv is not None:
                recv.close(linger=0)
            if self._context is None:
                context.destroy(linger=0)


def _sync_row(
    message: DraftSync,
) -> tuple[str, int, int, int, list[int], list[int]]:
    return (
        str(message.request_id),
        int(message.src_verifier_rank),
        int(message.dst_drafter_rank),
        int(message.max_new_tokens),
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
) -> tuple[int, int, str, int, int, list[int], bool]:
    return (
        int(output.src_drafter_rank),
        int(output.dst_verifier_rank),
        str(output.request_id),
        int(output.base_committed_len),
        int(output.start_token_pos),
        [int(token) for token in output.tokens],
        bool(output.is_commit_echo),
    )


def _validate_verifier_control_batch(
    batch: DraftControlBatch, *, verifier_rank: int
) -> None:
    dst_drafter_rank = int(batch.dst_drafter_rank)
    for message in (
        *batch.sync_messages,
        *batch.verify_commit_messages,
        *batch.close_messages,
    ):
        if int(message.src_verifier_rank) != int(verifier_rank):
            raise RuntimeError(
                "Verifier control source rank mismatch: "
                f"expected_verifier_rank={verifier_rank} "
                f"src_verifier_rank={message.src_verifier_rank}"
            )
        if int(message.dst_drafter_rank) != dst_drafter_rank:
            raise RuntimeError(
                "Verifier control destination rank mismatch: "
                f"batch_drafter_rank={dst_drafter_rank} "
                f"message_drafter_rank={message.dst_drafter_rank}"
            )


def _sync_from_native(row: Sequence[Any]) -> DraftSync:
    return DraftSync(
        request_id=str(row[0]),
        src_verifier_rank=int(row[1]),
        dst_drafter_rank=int(row[2]),
        max_new_tokens=int(row[3]),
        prompt_token_ids=[int(token) for token in row[4]],
        committed_outputs=[int(token) for token in row[5]],
    )


class _DecoupledSpecDataPlane:
    """Own one GPU state buffer and its transport lifecycle on either role."""

    def __init__(self, config: DecoupledSpecIpcConfig, context) -> None:
        # Retain injected pyzmq contexts for native inproc socket ownership.
        self._context = context
        self.config = config
        self._peer_ranks = frozenset(peer.rank for peer in config.peers)
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
            self.gpu_tail_buffer.close()

    def take_transport_metrics(self) -> dict[str, Any]:
        """Atomically drain this role's transport-metric window."""

        return dict(self._transport.take_transport_metrics())

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


class VerifierDecoupledSpecDataPlane(_DecoupledSpecDataPlane):
    """Verifier GPU backend and role API, with selectable socket transport."""

    def __init__(
        self,
        config: DecoupledSpecIpcConfig,
        *,
        context: Any | None = None,
        python_transport: bool = False,
        device: str | torch.device,
        num_gpu_seats: int,
        num_draft_tokens: int,
        landing_stream: torch.cuda.Stream,
        mock_profile: bool = False,
    ) -> None:
        # Keep an injected pyzmq context alive and let native sockets share its
        # underlying libzmq context. This preserves inproc test/deployment
        # semantics without transferring socket ownership across languages.
        super().__init__(config, context)
        external_context = 0 if context is None else int(context.underlying)
        self.gpu_tail_buffer = CppGpuDraftTailBuffer(
            device=device,
            num_seats=num_gpu_seats,
            num_draft_tokens=num_draft_tokens,
            landing_stream=landing_stream,
            mock_profile=mock_profile,
        )
        self._transport = _load_decoupled_spec_cpp_module().DraftProxyThread(
            int(config.rank),
            str(config.bind_endpoint),
            _peer_rows(config),
            self.gpu_tail_buffer._cpp,
            external_context,
            bool(mock_profile),
            python_transport,
        )
        if python_transport:
            self._transport = PythonZmqTransport(
                self._transport, config, context, role="verifier"
            )
        self.mock_profile = bool(mock_profile)

    def submit_control_batch(
        self,
        batch: DraftControlBatch,
        *,
        apply_local_verify_commits: bool = True,
    ) -> None:
        self._ensure_running()
        self._validate_peer_rank(int(batch.dst_drafter_rank))
        _validate_verifier_control_batch(batch, verifier_rank=int(self.config.rank))
        self._transport.submit_control_batch_native(
            int(batch.dst_drafter_rank),
            [_sync_row(message) for message in batch.sync_messages],
            [_commit_row(message) for message in batch.verify_commit_messages],
            [_close_row(message) for message in batch.close_messages],
            bool(apply_local_verify_commits),
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
        _validate_verifier_control_batch(
            control_batch, verifier_rank=int(self.config.rank)
        )
        if not self.mock_profile:
            for message, gpu_seat, request_epoch in requests:
                if gpu_seat is None or request_epoch is None:
                    raise ValueError(
                        "GPU draft-tail open requires gpu_seat and request_epoch"
                    )
                self.gpu_tail_buffer.bind_request(
                    message.request_id, gpu_seat, request_epoch
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


class DrafterDecoupledSpecDataPlane(_DecoupledSpecDataPlane):
    """Drafter backend and role API, with selectable socket transport."""

    def __init__(
        self,
        config: DecoupledSpecIpcConfig,
        *,
        context: Any | None = None,
        python_transport: bool = False,
        device: str | torch.device,
        num_gpu_seats: int,
        num_draft_tokens: int,
        pending_token_capacity: int | None = None,
        landing_stream: torch.cuda.Stream,
        schedule_ahead_limit: int | None = None,
    ) -> None:
        super().__init__(config, context)
        external_context = 0 if context is None else int(context.underlying)
        self.gpu_tail_buffer = CppGpuDraftTailBuffer(
            device=device,
            num_seats=num_gpu_seats,
            num_draft_tokens=num_draft_tokens,
            pending_token_capacity=pending_token_capacity,
            landing_stream=landing_stream,
            drafter_authoritative=True,
        )
        self._transport = _load_decoupled_spec_cpp_module().TokenSyncThread(
            int(config.rank),
            str(config.bind_endpoint),
            _peer_rows(config),
            external_context,
            self.gpu_tail_buffer._cpp,
            python_transport,
            -1 if schedule_ahead_limit is None else int(schedule_ahead_limit),
        )
        if python_transport:
            self._transport = PythonZmqTransport(
                self._transport, config, context, role="drafter"
            )

    def lookup_gpu_binding(
        self, request_id: str, src_verifier_rank: int
    ) -> tuple[int, int] | None:
        """Return the auto-assigned GPU mirror identity for one DraftSync."""

        self._ensure_running()
        seat, request_epoch = self._transport.lookup_gpu_binding_native(
            str(request_id), int(src_verifier_rank)
        )
        if int(seat) < 0:
            return None
        return int(seat), int(request_epoch)

    def pending_control_count(self) -> int:
        return int(self._transport.pending_control_count())

    def wait_for_control(self, timeout_s: float) -> bool:
        self._ensure_running()
        timeout_us = max(0, int(float(timeout_s) * 1_000_000))
        return bool(self._transport.wait_for_pending_control(timeout_us))

    def drain_gpu_progress(
        self,
    ) -> list[tuple[str, int, int, int, int, int]]:
        """Drain ``(logical_output, committed, raw_tail)`` scheduling cursors."""

        self._ensure_running()
        return [
            (
                str(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
            )
            for row in self._transport.drain_gpu_progress_native()
        ]

    def collect_lifecycle_controls(self) -> ReadyDraftControls:
        """Consume OPEN/CLOSE; verifier commits are applied directly on GPU."""

        self._ensure_running()
        sync_rows, close_rows = self._transport.extract_lifecycle_controls_native()
        return ReadyDraftControls(
            sync_messages=[_sync_from_native(row) for row in sync_rows],
            close_keys={
                DraftReqKey(
                    src_verifier_rank=int(row[1]),
                    request_id=str(row[0]),
                )
                for row in close_rows
            },
        )

    def publish_tail(
        self,
        output: DraftTailStreamOutput,
    ) -> None:
        self.publish_tails(DraftTailStreamOutputBatch(outputs=[output]))

    def publish_tails(
        self,
        batch: DraftTailStreamOutputBatch,
    ) -> None:
        self._ensure_running()
        for output in batch.outputs:
            output.validate()
            if int(output.src_drafter_rank) != int(self.config.rank):
                raise RuntimeError(
                    "Draft tail source rank mismatch: "
                    f"expected_drafter_rank={self.config.rank} "
                    f"src_drafter_rank={output.src_drafter_rank}"
                )
            self._validate_peer_rank(int(output.dst_verifier_rank))
        if batch.outputs:
            self._transport.submit_draft_results_native(
                [_tail_row(output) for output in batch.outputs],
            )


def _peer_rows(config: DecoupledSpecIpcConfig) -> list[tuple[int, str]]:
    return [(int(peer.rank), str(peer.endpoint)) for peer in config.peers]


def create_verifier_decoupled_spec_data_plane(
    config: DecoupledSpecIpcConfig,
    *,
    context: zmq.Context | None = None,
    device,
    num_gpu_seats: int,
    num_draft_tokens: int,
    landing_stream,
    mock_profile: bool = False,
):
    """Create the configured verifier data plane behind one role contract."""

    return VerifierDecoupledSpecDataPlane(
        config,
        python_transport=not envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get(),
        context=context,
        device=device,
        num_gpu_seats=num_gpu_seats,
        num_draft_tokens=num_draft_tokens,
        landing_stream=landing_stream,
        mock_profile=mock_profile,
    )


def create_drafter_decoupled_spec_data_plane(
    config: DecoupledSpecIpcConfig,
    *,
    context: zmq.Context | None = None,
    device,
    num_gpu_seats: int,
    num_draft_tokens: int,
    pending_token_capacity: int | None = None,
    landing_stream,
    schedule_ahead_limit: int | None = None,
):
    """Create the configured drafter data plane behind one role contract."""

    return DrafterDecoupledSpecDataPlane(
        config,
        python_transport=not envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get(),
        context=context,
        device=device,
        num_gpu_seats=num_gpu_seats,
        num_draft_tokens=num_draft_tokens,
        pending_token_capacity=pending_token_capacity,
        landing_stream=landing_stream,
        schedule_ahead_limit=schedule_ahead_limit,
    )

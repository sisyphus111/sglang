from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable, Sequence
from itertools import chain

import zmq

from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftControlInbox,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
    VerifierCommitSegment,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import (
    DraftTailBuffer,
    DraftTailSnapshot,
)

logger = logging.getLogger(__name__)

_IDLE_WAIT_S = 0.0005
_START_TIMEOUT_S = 5.0


class _BackgroundZmqTransport:
    """Thread-owned ZMQ sockets with fail-fast lifecycle reporting."""

    def __init__(
        self,
        *,
        config: DecoupledSpecIpcConfig,
        context: zmq.Context | None,
        thread_name: str,
    ) -> None:
        self.config = config
        self._owns_context = context is None
        self._context = context if context is not None else zmq.Context()
        self._closed = threading.Event()
        self._wakeup = threading.Event()
        self._ready = threading.Event()
        self._thread_error: BaseException | None = None
        self._started = False
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=thread_name,
            daemon=True,
        )

    @property
    def has_peers(self) -> bool:
        return bool(self.config.connect_endpoints)

    def start(self) -> None:
        if self._started:
            return
        if self._closed.is_set():
            raise RuntimeError("A closed decoupled-spec transport cannot be restarted")
        self._started = True
        self._thread.start()
        if not self._ready.wait(timeout=_START_TIMEOUT_S):
            raise TimeoutError(
                "Timed out starting decoupled-spec transport: "
                f"bind_endpoint={self.config.bind_endpoint}"
            )
        self.raise_if_failed()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError(
                "Decoupled-spec transport thread did not stop: "
                f"thread_name={self._thread.name}"
            )
        if self._owns_context:
            self._context.destroy(linger=0)

    def ensure_running(self) -> None:
        if not self._started:
            raise RuntimeError("Decoupled-spec transport has not been started")
        if self._closed.is_set():
            raise RuntimeError("Decoupled-spec transport is closed")
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError(
                "Decoupled-spec transport thread failed: "
                f"thread_name={self._thread.name}"
            ) from self._thread_error

    def _run_guarded(self) -> None:
        try:
            self._run()
        except BaseException as exc:
            self._thread_error = exc
            self._ready.set()
            logger.exception(
                "Decoupled-spec transport thread failed: thread_name=%s",
                self._thread.name,
            )

    def _run(self) -> None:
        raise NotImplementedError

    def _peer_endpoint(self, rank: int) -> str:
        rank = int(rank)
        if rank < 0 or rank >= len(self.config.connect_endpoints):
            raise RuntimeError(
                "Missing decoupled-spec peer endpoint: "
                f"peer_rank={rank} num_peers={len(self.config.connect_endpoints)}"
            )
        return str(self.config.connect_endpoints[rank])

    @staticmethod
    def _new_socket(context: zmq.Context, socket_type: int) -> zmq.Socket:
        socket = context.socket(socket_type)
        socket.setsockopt(zmq.LINGER, 0)
        return socket


class _VerifierTransport(_BackgroundZmqTransport):
    def __init__(
        self,
        *,
        config: DecoupledSpecIpcConfig,
        draft_tail_buffer: DraftTailBuffer,
        context: zmq.Context | None,
    ) -> None:
        super().__init__(
            config=config,
            context=context,
            thread_name="sglang-decoupled-verifier-transport",
        )
        self._draft_tail_buffer = draft_tail_buffer
        self._outgoing: queue.SimpleQueue[DraftControlBatch] = queue.SimpleQueue()

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        self.ensure_running()
        self._peer_endpoint(batch.dst_drafter_rank)
        self._outgoing.put(batch)
        self._wakeup.set()

    def _run(self) -> None:
        recv_socket = self._new_socket(self._context, zmq.PULL)
        send_sockets: dict[int, zmq.Socket] = {}
        try:
            recv_socket.bind(str(self.config.bind_endpoint))
            for rank, endpoint in enumerate(self.config.connect_endpoints):
                socket = self._new_socket(self._context, zmq.PUSH)
                socket.connect(str(endpoint))
                send_sockets[rank] = socket
            self._ready.set()

            pending: deque[DraftControlBatch] = deque()
            while not self._closed.is_set():
                did_work = False
                while True:
                    try:
                        pending.append(self._outgoing.get_nowait())
                    except queue.Empty:
                        break

                while pending:
                    batch = pending[0]
                    socket = send_sockets[int(batch.dst_drafter_rank)]
                    try:
                        socket.send_pyobj(
                            DraftMeshMessage.from_control_batch(batch),
                            flags=zmq.NOBLOCK,
                        )
                    except zmq.Again:
                        break
                    pending.popleft()
                    did_work = True

                while True:
                    try:
                        message = recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if not isinstance(message, DraftMeshMessage):
                        raise RuntimeError(
                            f"Unexpected verifier transport message: {message!r}"
                        )
                    if (
                        message.message_type
                        != DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
                        or message.tail_stream_output_batch is None
                    ):
                        raise RuntimeError(
                            f"Unexpected verifier transport message: {message!r}"
                        )
                    self._draft_tail_buffer.append_draft_stream_batch(
                        message.tail_stream_output_batch
                    )
                    did_work = True

                if not did_work:
                    self._wakeup.wait(timeout=_IDLE_WAIT_S)
                    self._wakeup.clear()
        finally:
            recv_socket.close(linger=0)
            for socket in send_sockets.values():
                socket.close(linger=0)


class _DrafterTransport(_BackgroundZmqTransport):
    def __init__(
        self,
        *,
        config: DecoupledSpecIpcConfig,
        context: zmq.Context | None,
    ) -> None:
        super().__init__(
            config=config,
            context=context,
            thread_name="sglang-decoupled-drafter-transport",
        )
        self._outgoing: queue.SimpleQueue[tuple[int, DraftTailStreamOutputBatch]] = (
            queue.SimpleQueue()
        )
        self._pending_lock = threading.Lock()
        self._pending_controls: deque[DraftControlBatch] = deque()
        self._control_ready = threading.Event()

    def submit_tail_batch(
        self, dst_verifier_rank: int, batch: DraftTailStreamOutputBatch
    ) -> None:
        self.ensure_running()
        self._peer_endpoint(dst_verifier_rank)
        self._outgoing.put((int(dst_verifier_rank), batch))
        self._wakeup.set()

    def drain_controls(self, max_batches: int | None = None) -> list[DraftControlBatch]:
        self.ensure_running()
        if max_batches is not None and int(max_batches) < 0:
            raise ValueError(f"max_batches must be non-negative, got {max_batches}")
        limit = None if max_batches is None else int(max_batches)
        batches: list[DraftControlBatch] = []
        with self._pending_lock:
            while self._pending_controls and (limit is None or len(batches) < limit):
                batches.append(self._pending_controls.popleft())
            if not self._pending_controls:
                self._control_ready.clear()
        return batches

    def pending_control_count(self) -> int:
        self.raise_if_failed()
        with self._pending_lock:
            return len(self._pending_controls)

    def wait_for_control(self, timeout_s: float) -> bool:
        self.ensure_running()
        if self.pending_control_count() > 0:
            return True
        return self._control_ready.wait(timeout=max(0.0, float(timeout_s)))

    def _run(self) -> None:
        recv_socket = self._new_socket(self._context, zmq.PULL)
        send_sockets: dict[int, zmq.Socket] = {}
        try:
            recv_socket.bind(str(self.config.bind_endpoint))
            for rank, endpoint in enumerate(self.config.connect_endpoints):
                socket = self._new_socket(self._context, zmq.PUSH)
                socket.connect(str(endpoint))
                send_sockets[rank] = socket
            self._ready.set()

            pending: deque[tuple[int, DraftTailStreamOutputBatch]] = deque()
            while not self._closed.is_set():
                did_work = False
                while True:
                    try:
                        pending.append(self._outgoing.get_nowait())
                    except queue.Empty:
                        break

                while pending:
                    dst_verifier_rank, batch = pending[0]
                    socket = send_sockets[dst_verifier_rank]
                    try:
                        socket.send_pyobj(
                            DraftMeshMessage.from_tail_stream_output_batch(batch),
                            flags=zmq.NOBLOCK,
                        )
                    except zmq.Again:
                        break
                    pending.popleft()
                    did_work = True

                while True:
                    try:
                        message = recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    if not isinstance(message, DraftMeshMessage):
                        raise RuntimeError(
                            f"Unexpected drafter transport message: {message!r}"
                        )
                    if (
                        message.message_type != DraftMeshMessageType.CONTROL_BATCH
                        or message.control_batch is None
                    ):
                        raise RuntimeError(
                            f"Unexpected drafter transport message: {message!r}"
                        )
                    if int(message.control_batch.dst_drafter_rank) != int(
                        self.config.rank
                    ):
                        raise RuntimeError(
                            "Draft control batch targets a different drafter: "
                            f"expected_drafter_rank={self.config.rank} "
                            f"dst_drafter_rank="
                            f"{message.control_batch.dst_drafter_rank}"
                        )
                    with self._pending_lock:
                        self._pending_controls.append(message.control_batch)
                        self._control_ready.set()
                    did_work = True

                if not did_work:
                    self._wakeup.wait(timeout=_IDLE_WAIT_S)
                    self._wakeup.clear()
        finally:
            recv_socket.close(linger=0)
            for socket in send_sockets.values():
                socket.close(linger=0)


class VerifierDecoupledSpecDataPlane:
    """Composition-facing verifier data plane.

    Control submission updates the local authoritative tail synchronously before
    the same batch is queued for the background ZMQ sender.
    """

    def __init__(
        self,
        config: DecoupledSpecIpcConfig,
        *,
        required_tail_len: int = 0,
        context: zmq.Context | None = None,
    ) -> None:
        self.config = config
        self.draft_tail_buffer = DraftTailBuffer(
            verifier_rank=int(config.rank), required_tail_len=required_tail_len
        )
        self._transport = _VerifierTransport(
            config=config,
            draft_tail_buffer=self.draft_tail_buffer,
            context=context,
        )

    def start(self) -> None:
        self._transport.start()

    def close(self) -> None:
        try:
            self._transport.close()
        finally:
            self.draft_tail_buffer.close()

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        self._transport.ensure_running()
        self._validate_control_batch(batch)
        # This ordering is the verifier-side linearization point used by both
        # non-overlap and overlap snapshot selection.
        self.draft_tail_buffer.apply_control_batch(batch)
        self._transport.submit_control_batch(batch)

    def open_request(self, message: DraftSync) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                sync_messages=[message],
            )
        )

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
        """Return one immutable snapshot row per request, in input order."""

        self._transport.ensure_running()
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

    def _validate_control_batch(self, batch: DraftControlBatch) -> None:
        dst_drafter_rank = int(batch.dst_drafter_rank)
        self._transport._peer_endpoint(dst_drafter_rank)
        messages = chain(
            batch.sync_messages,
            batch.verify_commit_messages,
            batch.close_messages,
        )
        for message in messages:
            if int(message.src_verifier_rank) != int(self.config.rank):
                raise RuntimeError(
                    "Verifier control source rank mismatch: "
                    f"expected_verifier_rank={self.config.rank} "
                    f"src_verifier_rank={message.src_verifier_rank}"
                )
            if int(message.dst_drafter_rank) != dst_drafter_rank:
                raise RuntimeError(
                    "Verifier control destination rank mismatch: "
                    f"batch_drafter_rank={dst_drafter_rank} "
                    f"message_drafter_rank={message.dst_drafter_rank}"
                )


class DrafterDecoupledSpecDataPlane:
    """Composition-facing drafter control inbox and tail publisher."""

    def __init__(
        self,
        config: DecoupledSpecIpcConfig,
        *,
        context: zmq.Context | None = None,
    ) -> None:
        self.config = config
        self._transport = _DrafterTransport(config=config, context=context)
        self._control_inbox = DraftControlInbox()

    def start(self) -> None:
        self._transport.start()

    def close(self) -> None:
        self._transport.close()

    def drain_controls(self, max_batches: int | None = None) -> list[DraftControlBatch]:
        """Remove received control batches in their original wire order."""

        return self._transport.drain_controls(max_batches=max_batches)

    def pending_control_count(self) -> int:
        return (
            self._transport.pending_control_count()
            + self._control_inbox.pending_control_count()
        )

    def wait_for_control(self, timeout_s: float) -> bool:
        if self._control_inbox.pending_control_count() > 0:
            return True
        return self._transport.wait_for_control(timeout_s)

    def collect_ready_controls(
        self,
        consumable_commit_len: Callable[[VerifierCommitSegment], int],
    ) -> ReadyDraftControls:
        """Coalesce wire commits per request and consume each ready prefix."""

        for batch in self._transport.drain_controls():
            self._control_inbox.add_control_batch_locked(batch)
        return self._control_inbox.extract_ready_controls_locked(consumable_commit_len)

    def publish_tail(self, output: DraftTailStreamOutput) -> None:
        self.publish_tails(DraftTailStreamOutputBatch(outputs=[output]))

    def publish_tails(self, batch: DraftTailStreamOutputBatch) -> None:
        self._transport.ensure_running()
        batches_by_verifier: dict[int, DraftTailStreamOutputBatch] = {}
        for output in batch.outputs:
            if int(output.src_drafter_rank) != int(self.config.rank):
                raise RuntimeError(
                    "Draft tail source rank mismatch: "
                    f"expected_drafter_rank={self.config.rank} "
                    f"src_drafter_rank={output.src_drafter_rank}"
                )
            dst_verifier_rank = int(output.dst_verifier_rank)
            self._transport._peer_endpoint(dst_verifier_rank)
            batches_by_verifier.setdefault(
                dst_verifier_rank, DraftTailStreamOutputBatch()
            ).outputs.append(output)

        for dst_verifier_rank, verifier_batch in batches_by_verifier.items():
            self._transport.submit_tail_batch(dst_verifier_rank, verifier_batch)


def create_verifier_decoupled_spec_data_plane(
    config: DecoupledSpecIpcConfig,
    *,
    required_tail_len: int = 0,
    context: zmq.Context | None = None,
    device=None,
    num_gpu_seats: int | None = None,
    num_draft_tokens: int | None = None,
    landing_stream=None,
):
    """Create the configured verifier data plane behind one role contract."""

    if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get():
        from sglang.srt.speculative.cpp_decoupled_spec import (
            CppVerifierDecoupledSpecDataPlane,
        )

        return CppVerifierDecoupledSpecDataPlane(
            config,
            required_tail_len=required_tail_len,
            context=context,
            device=device,
            num_gpu_seats=num_gpu_seats,
            num_draft_tokens=num_draft_tokens,
            landing_stream=landing_stream,
        )
    return VerifierDecoupledSpecDataPlane(
        config,
        required_tail_len=required_tail_len,
        context=context,
    )


def create_drafter_decoupled_spec_data_plane(
    config: DecoupledSpecIpcConfig,
    *,
    context: zmq.Context | None = None,
):
    """Create the configured drafter data plane behind one role contract."""

    if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get():
        from sglang.srt.speculative.cpp_decoupled_spec import (
            CppDrafterDecoupledSpecDataPlane,
        )

        return CppDrafterDecoupledSpecDataPlane(config, context=context)
    return DrafterDecoupledSpecDataPlane(config, context=context)

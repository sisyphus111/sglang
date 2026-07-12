from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftControlInbox,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)

TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S = 0.0005  # 0.5ms


@dataclass
class TokenSyncThread:
    """Drafter-side token sync thread for decoupled speculation IPC."""

    context: zmq.Context | None = None
    drafter_rank: int = 0
    control_bind_endpoint: str | None = field(default=None, init=False)
    verifier_result_endpoints: list[str] = field(default_factory=list, init=False)
    _pending_control_inbox: DraftControlInbox = field(
        default_factory=DraftControlInbox
    )
    # verifier -> drafter controls
    control_recv_socket: zmq.Socket | None = None
    # drafter -> verifier draft tokens
    result_send_sockets: dict[int, zmq.Socket] = field(default_factory=dict)
    # protects _pending_control_inbox
    _pending_lock: threading.Lock = field(default_factory=threading.Lock)
    _outgoing_results: queue.SimpleQueue[DraftTailStreamOutputBatch] = field(
        default_factory=queue.SimpleQueue
    )
    _closed: threading.Event = field(default_factory=threading.Event)
    _wakeup: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        if self.context is None:
            self._thread = threading.Thread(
                target=self._run,
                name="sglang-token-sync-thread",
                daemon=True,
            )
            return
        bind_host = get_local_ip_auto("127.0.0.1")
        port, self.control_recv_socket = get_zmq_socket_on_host(
            self.context, zmq.PULL, host=bind_host
        )
        self.control_bind_endpoint = NetworkAddress(bind_host, port).to_tcp()
        logger.info(
            "Bound decoupled-spec drafter control endpoint: "
            "drafter_rank=%s endpoint=%s",
            self.drafter_rank,
            self.control_bind_endpoint,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-token-sync-thread",
            daemon=True,
        )

    def start(self) -> None:
        if not self.result_send_sockets:
            return
        if self._thread is None:
            return
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.control_recv_socket is not None:
            self.control_recv_socket.close(linger=0)
        for socket in self.result_send_sockets.values():
            socket.close(linger=0)

    def configure_peer_endpoints(self, verifier_result_endpoints: list[str]) -> None:
        endpoints = list(verifier_result_endpoints)
        if not endpoints:
            raise RuntimeError(
                "Decoupled drafter requires at least one verifier result endpoint"
            )
        if self.result_send_sockets:
            if endpoints == self.verifier_result_endpoints:
                return
            raise RuntimeError("Decoupled drafter peer endpoints are already configured")
        if self.context is None:
            raise RuntimeError("Decoupled drafter ZMQ context is not initialized")
        self.result_send_sockets = {
            verifier_rank: get_zmq_socket(
                self.context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for verifier_rank, endpoint in enumerate(endpoints)
        }
        self.verifier_result_endpoints = endpoints
        logger.info(
            "Configured decoupled-spec drafter peers: "
            "drafter_rank=%s verifier_result_endpoints=%s",
            self.drafter_rank,
            endpoints,
        )

    def _drain_control_socket(self) -> bool:
        did_work = False
        if self.control_recv_socket is None:
            return did_work

        while True:
            try:
                control_batch = self._recv_control_batch_from_socket()
            except zmq.error.ContextTerminated:
                raise
            except zmq.ZMQError:
                break
            did_work = True
            if control_batch is None:
                continue
            with self._pending_lock:
                self._pending_control_inbox.add_control_batch_locked(control_batch)
        return did_work

    def _recv_control_batch_from_socket(self) -> DraftControlBatch | None:
        message = self.control_recv_socket.recv_pyobj(zmq.NOBLOCK)
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(f"Unexpected draft control message: {message}")
        if (
            message.message_type != DraftMeshMessageType.CONTROL_BATCH
            or message.control_batch is None
        ):
            raise RuntimeError(f"Unexpected draft control message: {message}")
        control_batch = message.control_batch
        if int(control_batch.dst_drafter_rank) != int(self.drafter_rank):
            return None
        return control_batch

    def collect_ready_draft_controls(
        self,
        collector: Callable[[DraftControlInbox], ReadyDraftControls],
    ) -> ReadyDraftControls:
        """Extract ready controls from the live inbox under the inbox lock."""
        with self._pending_lock:
            return collector(self._pending_control_inbox)

    def submit_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return
        if not self.result_send_sockets:
            raise RuntimeError("Decoupled drafter peer endpoints are not configured")
        queued_batch = DraftTailStreamOutputBatch(outputs=list(result_batch.outputs))
        self._outgoing_results.put(queued_batch)
        self._wakeup.set()

    def _drain_outgoing_results(self) -> bool:
        did_work = False
        while True:
            try:
                result_batch = self._outgoing_results.get_nowait()
            except queue.Empty:
                break
            did_work = True
            self._send_draft_results(result_batch)
        return did_work

    def _send_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return

        batches_by_verifier: dict[int, DraftTailStreamOutputBatch] = {}
        for output in result_batch.outputs:
            dst_verifier_rank = int(output.dst_verifier_rank)
            batches_by_verifier.setdefault(
                dst_verifier_rank,
                DraftTailStreamOutputBatch(),
            ).outputs.append(output)

        for dst_verifier_rank, send_batch in batches_by_verifier.items():
            self._send_result_batch(dst_verifier_rank, send_batch)

    def _send_result_batch(
        self,
        dst_verifier_rank: int,
        send_batch: DraftTailStreamOutputBatch,
    ) -> None:
        socket = self.result_send_sockets.get(dst_verifier_rank)
        if socket is None:
            raise RuntimeError(
                f"Missing result socket for dst_verifier_rank={dst_verifier_rank}"
            )
        socket.send_pyobj(DraftMeshMessage.from_tail_stream_output_batch(send_batch))

    def _idle_wait(self) -> None:
        self._wakeup.wait(timeout=TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S)
        self._wakeup.clear()

    def _run(self) -> None:
        while not self._closed.is_set():
            did_work = False
            try:
                did_work = bool(self._drain_outgoing_results()) or did_work
                did_work = bool(self._drain_control_socket()) or did_work
            except zmq.error.ContextTerminated:
                break

            if not did_work:
                self._idle_wait()

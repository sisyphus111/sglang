from __future__ import annotations

import logging
import queue
import threading
import time
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
    decoupled_spec_print_buffer,
    decoupled_spec_print_thread,
    decoupled_spec_print_timing,
    decoupled_spec_timing_start,
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
        start_ns = decoupled_spec_timing_start()
        item_count = 0
        try:
            message = self.control_recv_socket.recv_pyobj(zmq.NOBLOCK)
            if not isinstance(message, DraftMeshMessage):
                raise RuntimeError(f"Unexpected draft control message: {message}")
            if (
                message.message_type != DraftMeshMessageType.CONTROL_BATCH
                or message.control_batch is None
            ):
                raise RuntimeError(f"Unexpected draft control message: {message}")
            control_batch = message.control_batch
            item_count = (
                len(control_batch.sync_messages)
                + len(control_batch.verify_commit_messages)
                + len(control_batch.close_messages)
            )
            recv_ns = time.perf_counter_ns()
            send_ns = getattr(message, "_decoupled_debug_send_ns", 0)
            enqueue_ns = getattr(message, "_decoupled_debug_enqueue_ns", 0)
            decoupled_spec_print_thread(
                component="token_sync_thread",
                op="recv_control_batch",
                drafter_rank=int(self.drafter_rank),
                items=item_count,
                send_to_recv_us=(
                    (recv_ns - int(send_ns)) / 1_000.0 if send_ns else 0.0
                ),
                enqueue_to_recv_us=(
                    (recv_ns - int(enqueue_ns)) / 1_000.0 if enqueue_ns else 0.0
                ),
            )
            if int(control_batch.dst_drafter_rank) != int(self.drafter_rank):
                return None
            return control_batch
        finally:
            if item_count > 0:
                decoupled_spec_print_timing(
                    component="token_sync_thread",
                    op="recv_control_batch",
                    start_ns=start_ns,
                    items=item_count,
                )

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
        queue_size_before = self._outgoing_results_size()
        queued_batch = DraftTailStreamOutputBatch(outputs=list(result_batch.outputs))
        setattr(queued_batch, "_decoupled_debug_enqueue_ns", time.perf_counter_ns())
        self._outgoing_results.put(queued_batch)
        self._wakeup.set()
        decoupled_spec_print_buffer(
            component="token_sync_thread",
            op="enqueue_draft_results",
            drafter_rank=int(self.drafter_rank),
            items=len(result_batch.outputs),
            outgoing_queue_before=queue_size_before,
            outgoing_queue_after=self._outgoing_results_size(),
        )

    def _drain_outgoing_results(self) -> bool:
        queue_size_before = self._outgoing_results_size()
        did_work = False
        num_result_batches = 0
        num_stream_outputs = 0
        queue_delay_total_us = 0.0
        queue_delay_max_us = 0.0
        while True:
            try:
                result_batch = self._outgoing_results.get_nowait()
            except queue.Empty:
                break
            did_work = True
            drain_start_ns = time.perf_counter_ns()
            enqueue_ns = getattr(result_batch, "_decoupled_debug_enqueue_ns", 0)
            queue_delay_us = (
                (drain_start_ns - int(enqueue_ns)) / 1_000.0 if enqueue_ns else 0.0
            )
            queue_delay_total_us += queue_delay_us
            queue_delay_max_us = max(queue_delay_max_us, queue_delay_us)
            setattr(result_batch, "_decoupled_debug_drain_ns", drain_start_ns)
            num_result_batches += 1
            num_stream_outputs += len(result_batch.outputs)
            self._send_draft_results(result_batch)
        if did_work:
            decoupled_spec_print_thread(
                component="token_sync_thread",
                op="draft_result_queue_delay",
                drafter_rank=int(self.drafter_rank),
                queue_size_before=queue_size_before,
                queue_size_after=self._outgoing_results_size(),
                num_result_batches=num_result_batches,
                num_stream_outputs=num_stream_outputs,
                queue_delay_avg_us=(
                    queue_delay_total_us / num_result_batches
                    if num_result_batches
                    else 0.0
                ),
                queue_delay_max_us=queue_delay_max_us,
            )
            decoupled_spec_print_buffer(
                component="token_sync_thread",
                op="drain_outgoing_results",
                drafter_rank=int(self.drafter_rank),
                queue_size_before=queue_size_before,
                queue_size_after=self._outgoing_results_size(),
                num_result_batches=num_result_batches,
                num_stream_outputs=num_stream_outputs,
            )
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
            for attr_name in (
                "_decoupled_debug_enqueue_ns",
                "_decoupled_debug_drain_ns",
            ):
                if hasattr(result_batch, attr_name):
                    setattr(send_batch, attr_name, getattr(result_batch, attr_name))
            self._send_result_batch(dst_verifier_rank, send_batch)

    def _send_result_batch(
        self,
        dst_verifier_rank: int,
        send_batch: DraftTailStreamOutputBatch,
    ) -> None:
        start_ns = decoupled_spec_timing_start()
        try:
            socket = self.result_send_sockets.get(dst_verifier_rank)
            if socket is None:
                raise RuntimeError(
                    f"Missing result socket for dst_verifier_rank={dst_verifier_rank}"
                )
            send_ns = time.perf_counter_ns()
            setattr(send_batch, "_decoupled_debug_send_ns", send_ns)
            message = DraftMeshMessage.from_tail_stream_output_batch(send_batch)
            setattr(message, "_decoupled_debug_send_ns", send_ns)
            socket.send_pyobj(message)
        finally:
            send_done_ns = time.perf_counter_ns()
            enqueue_ns = getattr(send_batch, "_decoupled_debug_enqueue_ns", 0)
            drain_ns = getattr(send_batch, "_decoupled_debug_drain_ns", 0)
            decoupled_spec_print_thread(
                component="token_sync_thread",
                op="send_result_batch",
                drafter_rank=int(self.drafter_rank),
                dst_verifier_rank=int(dst_verifier_rank),
                items=len(send_batch.outputs),
                enqueue_to_send_us=(
                    (send_done_ns - int(enqueue_ns)) / 1_000.0 if enqueue_ns else 0.0
                ),
                drain_to_send_us=(
                    (send_done_ns - int(drain_ns)) / 1_000.0 if drain_ns else 0.0
                ),
            )
            decoupled_spec_print_timing(
                component="token_sync_thread",
                op="send_result_batch",
                start_ns=start_ns,
                items=len(send_batch.outputs),
            )
            decoupled_spec_print_buffer(
                component="token_sync_thread",
                op="send_result_batch",
                drafter_rank=int(self.drafter_rank),
                dst_verifier_rank=int(dst_verifier_rank),
                items=len(send_batch.outputs),
            )

    def _outgoing_results_size(self) -> int:
        try:
            return int(self._outgoing_results.qsize())
        except (AttributeError, NotImplementedError):
            return -1

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

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftTailStreamOutputBatch,
    decoupled_spec_print_timing,
    decoupled_spec_timing_start,
)
from sglang.srt.speculative.tracer import (
    SpecTraceEvent,
    NullSpecTracer,
    trace_speculative,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.srt.utils.network import (
    NetworkAddress,
    get_local_ip_auto,
    get_zmq_socket,
    get_zmq_socket_on_host,
)

logger = logging.getLogger(__name__)


class DraftProxyThread:
    """
    Verifier-side proxy thread for decoupled speculation.

    Control batches from the verifier are first applied to the local
    DraftTailBuffer, then forwarded to the drafter. Draft tail stream batches
    from the drafter are appended to the same buffer.
    """

    def __init__(
        self,
        *,
        context: zmq.Context,
        verifier_rank: int,
        draft_tail_buffer: DraftTailBuffer,
        tracer: Any = None,
    ) -> None:
        self.context = context
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self.tracer = tracer or NullSpecTracer()
        # verifier -> drafter send control messages
        self.control_send_sockets: dict[int, zmq.Socket] = {}
        self.drafter_control_endpoints: list[str] = []
        bind_host = get_local_ip_auto("127.0.0.1")
        port, self.result_recv_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=bind_host
        )
        self.result_bind_endpoint = NetworkAddress(bind_host, port).to_tcp()
        logger.info(
            "Bound decoupled-spec verifier result endpoint: "
            "verifier_rank=%s endpoint=%s",
            self.verifier_rank,
            self.result_bind_endpoint,
        )
        self._send_queue: queue.SimpleQueue[DraftControlBatch] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-draft-proxy",
            daemon=True,
        )

    def start(self) -> None:
        if not self.control_send_sockets:
            return
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self.draft_tail_buffer.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for socket in self.control_send_sockets.values():
            socket.close(linger=0)
        self.result_recv_socket.close(linger=0)

    def configure_peer_endpoints(self, drafter_control_endpoints: list[str]) -> None:
        endpoints = list(drafter_control_endpoints)
        if not endpoints:
            raise RuntimeError(
                "Decoupled verify requires at least one drafter control endpoint"
            )
        if self.control_send_sockets:
            if endpoints == self.drafter_control_endpoints:
                return
            raise RuntimeError("Decoupled verify peer endpoints are already configured")
        self.control_send_sockets = {
            drafter_rank: get_zmq_socket(
                self.context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for drafter_rank, endpoint in enumerate(endpoints)
        }
        self.drafter_control_endpoints = endpoints
        logger.info(
            "Configured decoupled-spec verifier peers: "
            "verifier_rank=%s drafter_control_endpoints=%s",
            self.verifier_rank,
            endpoints,
        )

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        if not self.control_send_sockets:
            raise RuntimeError("Decoupled verify peer endpoints are not configured")
        self._apply_control_batch(batch)
        self._send_queue.put(batch)

    @trace_speculative(
        SpecTraceEvent.DRAFT_PROXY_APPLY_CONTROL_BATCH,
        inject_trace_enabled="collect_trace_stats",
    )
    def _apply_control_batch(
        self,
        batch: DraftControlBatch,
        *,
        collect_trace_stats: bool = False,
    ) -> dict[str, Any] | None:
        return self.draft_tail_buffer.apply_control_batch(
            batch,
            collect_stats=collect_trace_stats,
        )

    def _recv_tail_stream_output_batch(self) -> None:
        output_batch = self._recv_tail_stream_output_batch_from_socket()
        self._append_tail_stream_output_batch(output_batch)

    @trace_speculative(SpecTraceEvent.DRAFT_PROXY_RECV_TAIL_STREAM_BATCH)
    def _recv_tail_stream_output_batch_from_socket(
        self,
    ) -> DraftTailStreamOutputBatch:
        start_ns = decoupled_spec_timing_start()
        output_count = 0
        try:
            message = self.result_recv_socket.recv_pyobj()
            if not isinstance(message, DraftMeshMessage):
                raise RuntimeError(f"Unexpected draft proxy message: {message}")
            if (
                message.message_type != DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
                or message.tail_stream_output_batch is None
            ):
                raise RuntimeError(f"Unexpected draft proxy message: {message}")

            output_batch = message.tail_stream_output_batch
            output_count = len(output_batch.outputs)
            mismatched_outputs = [
                output
                for output in output_batch.outputs
                if int(output.dst_verifier_rank) != self.verifier_rank
            ]
            if mismatched_outputs:
                raise RuntimeError(
                    "Draft proxy received a tail stream batch for the wrong verifier: "
                    f"verifier_rank={self.verifier_rank} "
                    f"dst_verifier_ranks={[int(output.dst_verifier_rank) for output in output_batch.outputs]} "
                    f"request_ids={[output.request_id for output in output_batch.outputs]}"
                )
            return output_batch
        finally:
            decoupled_spec_print_timing(
                component="draft_proxy",
                op="recv_tail_stream_batch",
                start_ns=start_ns,
                items=output_count,
            )

    @trace_speculative(
        SpecTraceEvent.DRAFT_PROXY_APPEND_TAIL_STREAM_BATCH,
        inject_trace_enabled="collect_trace_stats",
    )
    def _append_tail_stream_output_batch(
        self,
        output_batch: DraftTailStreamOutputBatch,
        *,
        collect_trace_stats: bool = False,
    ) -> dict[str, Any] | None:
        return self.draft_tail_buffer.append_draft_stream_batch(
            output_batch,
            collect_stats=collect_trace_stats,
        )

    @trace_speculative(SpecTraceEvent.DRAFT_PROXY_SEND_CONTROL_BATCH)
    def _send_control_batch(self, batch: DraftControlBatch) -> None:
        start_ns = decoupled_spec_timing_start()
        dst_drafter_rank = int(batch.dst_drafter_rank)
        try:
            socket = self.control_send_sockets.get(dst_drafter_rank)
            if socket is None:
                raise RuntimeError(
                    f"Missing control socket for dst_drafter_rank={dst_drafter_rank}"
                )
            socket.send_pyobj(DraftMeshMessage.from_control_batch(batch))
        finally:
            decoupled_spec_print_timing(
                component="draft_proxy",
                op="send_control_batch",
                start_ns=start_ns,
                items=(
                    len(batch.sync_messages)
                    + len(batch.verify_commit_messages)
                    + len(batch.close_messages)
                ),
            )

    def _run(self) -> None:
        while not self._closed.is_set():
            while True:
                try:
                    batch = self._send_queue.get_nowait()
                except queue.Empty:
                    break
                self._send_control_batch(batch)

            try:
                if self.result_recv_socket.poll(timeout=1):
                    self._recv_tail_stream_output_batch()
            except zmq.error.ContextTerminated:
                break

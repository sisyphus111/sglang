"""Python socket transport for the shared decoupled-spec GPU backend.

The backend owns wire encoding, bounded queues, GPU landing/egress and request
semantics. Python owns this transport's thread and pyzmq sockets. No Python token
transcript or second GPU state machine is maintained here.
"""
from __future__ import annotations

import threading

import zmq

from sglang.srt.environ import envs
from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDrafterDecoupledSpecDataPlane,
    CppVerifierDecoupledSpecDataPlane,
)
from sglang.srt.speculative.decoupled_spec_io import DecoupledSpecIpcConfig


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


class VerifierDecoupledSpecDataPlane(CppVerifierDecoupledSpecDataPlane):
    """Shared verifier GPU backend driven by Python sockets."""

    _python_transport = True


class DrafterDecoupledSpecDataPlane(CppDrafterDecoupledSpecDataPlane):
    """Shared drafter backend driven by Python sockets."""

    _python_transport = True


def create_verifier_decoupled_spec_data_plane(
    config: DecoupledSpecIpcConfig,
    *,
    required_tail_len: int = 0,
    context: zmq.Context | None = None,
    device=None,
    num_gpu_seats: int | None = None,
    num_draft_tokens: int | None = None,
    landing_stream=None,
    mock_profile: bool = False,
):
    """Create the configured verifier data plane behind one role contract."""

    plane_type = (
        CppVerifierDecoupledSpecDataPlane
        if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get()
        else VerifierDecoupledSpecDataPlane
    )
    return plane_type(
        config,
        required_tail_len=required_tail_len,
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
    device=None,
    num_gpu_seats: int | None = None,
    num_draft_tokens: int | None = None,
    pending_token_capacity: int | None = None,
    landing_stream=None,
):
    """Create the configured drafter data plane behind one role contract."""

    plane_type = (
        CppDrafterDecoupledSpecDataPlane
        if envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.get()
        else DrafterDecoupledSpecDataPlane
    )
    return plane_type(
        config,
        context=context,
        device=device,
        num_gpu_seats=num_gpu_seats,
        num_draft_tokens=num_draft_tokens,
        pending_token_capacity=pending_token_capacity,
        landing_stream=landing_stream,
    )

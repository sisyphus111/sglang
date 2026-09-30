"""Host transport contracts using the production CUDA-built native extension."""

import socket
import struct
import threading
import time
import unittest
import uuid

import zmq
from decoupled_spec_test_utils import (
    CppDrafterTestPeer as CppDrafterDecoupledSpecDataPlane,
)
from decoupled_spec_test_utils import (
    CppVerifierTestPeer as CppVerifierDecoupledSpecDataPlane,
)
from decoupled_spec_test_utils import (
    PythonDrafterTestPeer as DrafterDecoupledSpecDataPlane,
)
from decoupled_spec_test_utils import (
    PythonVerifierTestPeer as VerifierDecoupledSpecDataPlane,
)

from sglang.srt.environ import envs
from sglang.srt.speculative.decoupled_spec_data_plane import (
    _peer_rows,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DecoupledSpecPeerConfig,
    DraftClose,
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Both socket implementations JIT the same extension, including its CUDA source.
register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


class TestRankedCppPeerRows(CustomTestCase):
    def test_sparse_peer_ranks_are_not_reenumerated(self):
        config = DecoupledSpecIpcConfig(
            bind_endpoint="tcp://verifier:30005",
            connect_endpoints=(),
            rank=5,
            peers=(
                DecoupledSpecPeerConfig(
                    rank=3, endpoint="tcp://drafter-a:31003", quota=2
                ),
                DecoupledSpecPeerConfig(
                    rank=9, endpoint="tcp://drafter-b:31009", quota=1
                ),
            ),
        )

        self.assertEqual(
            _peer_rows(config),
            [
                (3, "tcp://drafter-a:31003"),
                (9, "tcp://drafter-b:31009"),
            ],
        )


def _sync(request_id: str = "req") -> DraftSync:
    return DraftSync(
        request_id=request_id,
        src_verifier_rank=0,
        dst_drafter_rank=0,
        prompt_token_ids=[7, 8],
        committed_outputs=[],
    )


def _tail(
    start_token_pos: int,
    tokens: int | tuple[int, ...],
    *,
    request_id: str = "req",
    base_committed_len: int = 0,
    is_commit_echo: bool = False,
) -> DraftTailStreamOutput:
    if isinstance(tokens, int):
        tokens = (tokens,)
    return DraftTailStreamOutput(
        src_drafter_rank=0,
        dst_verifier_rank=0,
        request_id=request_id,
        base_committed_len=base_committed_len,
        start_token_pos=start_token_pos,
        tokens=tuple(tokens),
        is_commit_echo=is_commit_echo,
    )


def _snapshot_tuple(snapshot):
    return (
        snapshot.request_id,
        snapshot.committed_len,
        tuple(snapshot.tail_tokens),
        snapshot.raw_tail_len,
        snapshot.num_consumable_drafts,
    )


def _tcp_endpoint() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"tcp://127.0.0.1:{port}"


def _wait_for_value(callback, timeout_s: float = 2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = callback()
        if value:
            return value
        time.sleep(0.001)
    raise AssertionError("Timed out waiting for decoupled-spec transport")


def _enqueue_cpp_sender_frame(sender, direction: str, request_id: str) -> None:
    if direction == "verifier_control":
        sender.close_request(
            DraftClose(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                reason="queue-probe",
            )
        )
        return
    if direction == "drafter_tail":
        sender.publish_tails(
            DraftTailStreamOutputBatch(outputs=[_tail(0, 101, request_id=request_id)])
        )
        return
    raise AssertionError(f"Unknown C++ sender direction: {direction}")


def _bind_raw_pull(endpoint: str):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.bind(endpoint)
    return context, socket


class TestGpuDataPlaneFactory(CustomTestCase):
    def test_backend_choice_preserves_gpu_state_owner(self):
        from decoupled_spec_test_utils import gpu_config

        from sglang.srt.speculative.decoupled_spec_data_plane import (
            PythonZmqTransport,
            create_drafter_decoupled_spec_data_plane,
            create_verifier_decoupled_spec_data_plane,
        )

        for use_cpp in (True, False):
            for factory in (
                create_verifier_decoupled_spec_data_plane,
                create_drafter_decoupled_spec_data_plane,
            ):
                with self.subTest(use_cpp=use_cpp, factory=factory.__name__):
                    with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(use_cpp):
                        plane = factory(
                            DecoupledSpecIpcConfig(
                                bind_endpoint=_tcp_endpoint(),
                                connect_endpoints=(_tcp_endpoint(),),
                                rank=0,
                            ),
                            **gpu_config(),
                        )
                    try:
                        self.assertEqual(plane.gpu_tail_buffer.device.type, "cuda")
                        self.assertEqual(
                            isinstance(plane._transport, PythonZmqTransport),
                            not use_cpp,
                        )
                        plane.start()
                    finally:
                        plane.close()


class TestCppDecoupledSpecDataPlane(CustomTestCase):
    verifier_type = CppVerifierDecoupledSpecDataPlane
    drafter_type = CppDrafterDecoupledSpecDataPlane

    def _make_sender(self, direction, own_endpoint, peer_endpoint):
        config = DecoupledSpecIpcConfig(
            bind_endpoint=own_endpoint,
            connect_endpoints=(peer_endpoint,),
            rank=0,
        )
        if direction == "verifier_control":
            return self.verifier_type(config, required_tail_len=0)
        if direction == "drafter_tail":
            return self.drafter_type(config)
        raise AssertionError(f"Unknown sender direction: {direction}")

    def test_network_wire_version_mismatch_fails_fast(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://wire-verifier-{suffix}"
        drafter_endpoint = f"inproc://wire-drafter-{suffix}"
        control_pull = context.socket(zmq.PULL)
        result_push = context.socket(zmq.PUSH)
        control_pull.bind(drafter_endpoint)
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            context=context,
        )
        try:
            verifier.start()
            result_push.connect(verifier_endpoint)
            # Version 1 is intentionally rejected by the current wire contract.
            result_push.send(b"DSC1\x01\x02")
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    verifier.open_request(_sync(f"wire-{time.monotonic_ns()}"))
                except RuntimeError as exc:
                    self.assertRegex(
                        str(exc),
                        "Unsupported " "decoupled-spec frame version",
                    )
                    break
                if time.monotonic() >= deadline:
                    self.fail("Wire-version mismatch did not fail the native proxy")
                time.sleep(0.001)
        finally:
            verifier.close()
            result_push.close(linger=0)
            control_pull.close(linger=0)
            context.destroy(linger=0)

    def test_transport_metrics_exchange_and_clock_calibration(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://metrics-verifier-{suffix}"
        drafter_endpoint = f"inproc://metrics-drafter-{suffix}"
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            required_tail_len=0,
            context=context,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=context,
        )
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(_sync("metrics"))
            _wait_for_value(drafter.drain_controls, timeout_s=2.0)
            _wait_for_value(
                lambda: (
                    metrics
                    if (metrics := verifier.take_transport_metrics())[
                        "clock_sync_valid"
                    ]
                    else None
                ),
                timeout_s=3.0,
            )

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(0, (10, 11), request_id="metrics"),
                    ]
                )
            )
            _wait_for_value(
                lambda: verifier.snapshot_one("metrics").tail_tokens,
                timeout_s=2.0,
            )
            drafter_metrics = _wait_for_value(
                lambda: (
                    metrics
                    if (metrics := drafter.take_transport_metrics())[
                        "num_draft_result_frames"
                    ]
                    else None
                )
            )
            verifier_metrics = _wait_for_value(
                lambda: (
                    metrics
                    if (metrics := verifier.take_transport_metrics())[
                        "num_draft_result_frames"
                    ]
                    else None
                )
            )

            self.assertEqual(drafter_metrics["num_draft_result_frames"], 1)
            self.assertEqual(drafter_metrics["num_draft_result_tokens"], 2)
            self.assertGreaterEqual(drafter_metrics["draft_send_queue_depth_max"], 1)
            self._assert_latency_histogram(
                drafter_metrics["draft_send_queue_latency_us"], expected_count=1
            )
            self.assertEqual(verifier_metrics["num_draft_result_frames"], 1)
            self.assertEqual(verifier_metrics["num_draft_result_tokens"], 2)
            self._assert_latency_histogram(
                verifier_metrics["draft_receive_to_gpu_publish_enqueue_latency_us"],
                expected_count=1,
            )
            self.assertTrue(verifier_metrics["clock_sync_valid"])
            self.assertEqual(verifier_metrics["num_clock_sync_valid_peers"], 1)
            self.assertEqual(verifier_metrics["num_clock_sync_invalid_peers"], 0)
            self.assertIsNotNone(verifier_metrics["clock_error_bound_us"])
            self._assert_latency_histogram(
                verifier_metrics["draft_transport_one_way_latency_us"],
                expected_count=1,
            )
            self._assert_latency_histogram(
                verifier_metrics["draft_result_ready_to_receive_latency_us"],
                expected_count=1,
            )

            empty_drafter = drafter.take_transport_metrics()
            empty_verifier = verifier.take_transport_metrics()
            self.assertEqual(empty_drafter["num_draft_result_frames"], 0)
            self.assertEqual(empty_verifier["num_draft_result_frames"], 0)
            self._assert_latency_histogram(
                empty_drafter["draft_send_queue_latency_us"], expected_count=0
            )
        finally:
            verifier.close()
            drafter.close()
            context.destroy(linger=0)

    def test_calibration_missing_peer_does_not_block_healthy_business_io(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://calibration-verifier-{suffix}"
        healthy_drafter_endpoint = f"inproc://calibration-healthy-{suffix}"
        missing_drafter_endpoint = f"inproc://calibration-missing-{suffix}"
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(),
                rank=0,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=0, endpoint=healthy_drafter_endpoint, quota=1
                    ),
                    DecoupledSpecPeerConfig(
                        rank=1, endpoint=missing_drafter_endpoint, quota=1
                    ),
                ),
            ),
            required_tail_len=0,
            context=context,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=healthy_drafter_endpoint,
                connect_endpoints=(),
                rank=0,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=0, endpoint=verifier_endpoint, quota=1
                    ),
                ),
            ),
            context=context,
        )
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(_sync("before-calibration"))
            _wait_for_value(drafter.drain_controls)

            # The first calibration cycle starts after one second. Peer rank 1
            # has no bound PULL socket, so every best-effort probe sees EAGAIN.
            time.sleep(1.1)

            verifier.open_request(_sync("after-calibration"))
            controls = []

            def collect_after_calibration_open():
                controls.extend(drafter.drain_controls())
                return next(
                    (
                        batch
                        for batch in controls
                        if batch.sync_messages
                        and batch.sync_messages[0].request_id == "after-calibration"
                    ),
                    None,
                )

            _wait_for_value(collect_after_calibration_open, timeout_s=2.0)
            drafter.publish_tail(_tail(0, 17, request_id="after-calibration"))
            snapshot = _wait_for_value(
                lambda: (
                    current
                    if (
                        current := verifier.snapshot_one("after-calibration")
                    ).tail_tokens
                    else None
                ),
                timeout_s=2.0,
            )
            self.assertEqual(snapshot.tail_tokens, (17,))
        finally:
            verifier.close()
            drafter.close()
            context.destroy(linger=0)

    def test_calibration_reply_backpressure_does_not_block_control_receive(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://reply-verifier-{suffix}"
        drafter_endpoint = f"inproc://reply-drafter-{suffix}"
        missing_verifier_endpoint = _tcp_endpoint()
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(),
                rank=0,
                peers=(
                    DecoupledSpecPeerConfig(rank=0, endpoint=drafter_endpoint, quota=1),
                ),
            ),
            required_tail_len=0,
            context=context,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(),
                rank=0,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=0, endpoint=verifier_endpoint, quota=1
                    ),
                    DecoupledSpecPeerConfig(
                        rank=1, endpoint=missing_verifier_endpoint, quota=1
                    ),
                ),
            ),
            context=context,
        )
        control_push = context.socket(zmq.PUSH)
        control_push.setsockopt(zmq.LINGER, 0)
        try:
            drafter.start()
            verifier.start()
            control_push.connect(drafter_endpoint)

            probe_seq = 17
            epoch = 3
            verifier_send_ns = time.monotonic_ns()
            control_push.send(
                struct.pack(
                    "=4sBBqqqqiiqqq",
                    b"DSC1",
                    5,
                    3,
                    probe_seq,
                    0,
                    0,
                    epoch,
                    1,
                    0,
                    probe_seq,
                    verifier_send_ns,
                    epoch,
                )
            )
            time.sleep(0.05)

            verifier.open_request(_sync("control-after-dropped-reply"))
            controls = _wait_for_value(drafter.drain_controls, timeout_s=2.0)
            self.assertEqual(
                controls[0].sync_messages[0].request_id,
                "control-after-dropped-reply",
            )
        finally:
            verifier.close()
            drafter.close()
            control_push.close(linger=0)
            context.destroy(linger=0)

    def test_send_start_timestamp_excludes_local_retry_backpressure(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        drafter_endpoint = f"inproc://retry-drafter-{suffix}"
        # TCP plus ZMQ_IMMEDIATE guarantees EAGAIN until the late PULL peer has
        # completed a real connection handshake. Inproc may queue pre-bind.
        late_verifier_endpoint = _tcp_endpoint()
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(late_verifier_endpoint,),
                rank=0,
            ),
            context=context,
        )
        result_pull = context.socket(zmq.PULL)
        result_pull.setsockopt(zmq.LINGER, 0)
        result_pull.setsockopt(zmq.RCVTIMEO, 3000)
        try:
            drafter.start()
            drafter.publish_tail(_tail(0, 19, request_id="retry-timestamp"))
            time.sleep(0.05)

            bind_start_ns = time.monotonic_ns()
            result_pull.bind(late_verifier_endpoint)
            frame = result_pull.recv()
            _, result_ready_ns, send_start_ns, _ = struct.unpack_from("=qqqq", frame, 6)
            self.assertGreaterEqual(send_start_ns, bind_start_ns - 5_000_000)
            self.assertGreaterEqual(send_start_ns - result_ready_ns, 30_000_000)

            metrics = _wait_for_value(
                lambda: (
                    current
                    if (current := drafter.take_transport_metrics())[
                        "num_draft_result_frames"
                    ]
                    else None
                )
            )
            self.assertGreaterEqual(
                metrics["draft_send_queue_latency_us"]["sum_us"], 30_000
            )
        finally:
            drafter.close()
            result_pull.close(linger=0)
            context.destroy(linger=0)

    def test_healthy_peer_latency_survives_an_invalid_peer(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://partial-clock-verifier-{suffix}"
        healthy_drafter_endpoint = f"inproc://partial-clock-healthy-{suffix}"
        silent_drafter_endpoint = f"inproc://partial-clock-silent-{suffix}"
        silent_pull = context.socket(zmq.PULL)
        silent_pull.bind(silent_drafter_endpoint)
        stop_silent_peer = threading.Event()

        def drain_silent_peer():
            while not stop_silent_peer.is_set():
                try:
                    silent_pull.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.0005)

        silent_thread = threading.Thread(target=drain_silent_peer)
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(),
                rank=5,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=3, endpoint=healthy_drafter_endpoint, quota=1
                    ),
                    DecoupledSpecPeerConfig(
                        rank=9, endpoint=silent_drafter_endpoint, quota=1
                    ),
                ),
            ),
            context=context,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=healthy_drafter_endpoint,
                connect_endpoints=(),
                rank=3,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=5, endpoint=verifier_endpoint, quota=1
                    ),
                ),
            ),
            context=context,
        )
        try:
            silent_thread.start()
            drafter.start()
            verifier.start()
            verifier.open_request(
                DraftSync(
                    request_id="healthy-peer",
                    src_verifier_rank=5,
                    dst_drafter_rank=3,
                    prompt_token_ids=[7, 8],
                    committed_outputs=[],
                )
            )
            _wait_for_value(drafter.drain_controls)
            _wait_for_value(
                lambda: (
                    metrics
                    if (metrics := verifier.take_transport_metrics())[
                        "num_clock_sync_valid_peers"
                    ]
                    == 1
                    and metrics["num_clock_sync_invalid_peers"] == 1
                    else None
                ),
                timeout_s=3.0,
            )

            drafter.publish_tail(
                DraftTailStreamOutput(
                    src_drafter_rank=3,
                    dst_verifier_rank=5,
                    request_id="healthy-peer",
                    base_committed_len=0,
                    start_token_pos=0,
                    tokens=(10,),
                )
            )
            _wait_for_value(lambda: verifier.snapshot_one("healthy-peer").tail_tokens)
            metrics = _wait_for_value(
                lambda: (
                    current
                    if (current := verifier.take_transport_metrics())[
                        "num_draft_result_frames"
                    ]
                    else None
                )
            )
            self.assertFalse(metrics["clock_sync_valid"])
            self.assertEqual(metrics["num_clock_sync_valid_peers"], 1)
            self.assertEqual(metrics["num_clock_sync_invalid_peers"], 1)
            self._assert_latency_histogram(
                metrics["draft_transport_one_way_latency_us"], expected_count=1
            )
            self._assert_latency_histogram(
                metrics["draft_result_ready_to_receive_latency_us"],
                expected_count=1,
            )
        finally:
            verifier.close()
            drafter.close()
            stop_silent_peer.set()
            if silent_thread.is_alive():
                silent_thread.join(timeout=2.0)
            silent_pull.close(linger=0)
            context.destroy(linger=0)

    def test_transport_histogram_concurrent_take_is_coherent(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://hist-verifier-{suffix}"
        drafter_endpoint = f"inproc://hist-drafter-{suffix}"
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            context=context,
            num_draft_tokens=64,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=context,
        )
        stop = threading.Event()
        windows = []

        def take_windows():
            while not stop.is_set():
                windows.append(drafter.take_transport_metrics())
                time.sleep(0)

        taker = threading.Thread(target=take_windows)
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(_sync("hist"))
            _wait_for_value(drafter.drain_controls)
            taker.start()
            for position in range(128):
                drafter.publish_tail(
                    _tail(position, 1000 + position, request_id="hist")
                )
            _wait_for_value(
                lambda: (
                    snapshot
                    if (snapshot := verifier.snapshot_one("hist")).raw_tail_len == 128
                    else None
                ),
                timeout_s=3.0,
            )
        finally:
            stop.set()
            if taker.is_alive():
                taker.join(timeout=2.0)
            windows.append(drafter.take_transport_metrics())
            verifier.close()
            drafter.close()
            context.destroy(linger=0)

        total_frames = 0
        total_histogram_samples = 0
        for metrics in windows:
            histogram = metrics["draft_send_queue_latency_us"]
            self._assert_latency_histogram(
                histogram,
                expected_count=histogram["count"],
            )
            self.assertEqual(metrics["num_draft_result_frames"], histogram["count"])
            total_frames += metrics["num_draft_result_frames"]
            total_histogram_samples += histogram["count"]
        self.assertEqual(total_frames, 128)
        self.assertEqual(total_histogram_samples, 128)

    def _assert_latency_histogram(self, histogram, *, expected_count):
        self.assertEqual(histogram["count"], expected_count)
        self.assertEqual(len(histogram["bucket_upper_bounds_us"]), 17)
        self.assertEqual(len(histogram["bucket_counts"]), 18)
        self.assertEqual(sum(histogram["bucket_counts"]), expected_count)

    def _assert_close_is_interruptible(self, sender, peer_endpoint: str) -> None:
        errors = []

        def close_sender():
            try:
                sender.close()
            except BaseException as exc:
                errors.append(exc)

        close_thread = threading.Thread(target=close_sender, daemon=True)
        close_thread.start()
        close_thread.join(timeout=2.0)
        missed_deadline = close_thread.is_alive()

        # A blocking-send regression must not wedge the complete test process.
        # Binding the missing peer lets the old send path finish so the failed
        # assertion below remains actionable instead of hanging CI forever.
        rescue_context = None
        rescue_socket = None
        if missed_deadline:
            rescue_context, rescue_socket = _bind_raw_pull(peer_endpoint)
            close_thread.join(timeout=3.0)
        if rescue_socket is not None:
            rescue_socket.close(linger=0)
        if rescue_context is not None:
            rescue_context.term()

        self.assertFalse(
            missed_deadline,
            "C++ decoupled-spec close waited on an unavailable ZMQ peer",
        )
        self.assertFalse(
            close_thread.is_alive(),
            "C++ decoupled-spec close remained stuck after peer rescue",
        )
        if errors:
            raise errors[0]

    def test_lifecycle_controls_do_not_wait_for_commit_consumption(self):
        verifier_endpoint = _tcp_endpoint()
        drafter_endpoint = _tcp_endpoint()
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            )
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            )
        )
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(_sync("closed"))
            opened = _wait_for_value(
                lambda: drafter.collect_lifecycle_controls().sync_messages
            )
            self.assertEqual([message.request_id for message in opened], ["closed"])
            verifier.gpu_tail_buffer.bind_request("ready", 1, 2)
            verifier.gpu_tail_buffer.bind_request("cancelled", 2, 3)
            verifier.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[_sync("ready"), _sync("cancelled")],
                    verify_commit_messages=[
                        VerifyCommit(
                            request_id="closed",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            pre_verify_committed_len=0,
                            committed_tokens=[10, 11],
                        )
                    ],
                    close_messages=[
                        DraftClose(
                            request_id=request_id,
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            reason="finished",
                        )
                        for request_id in ("closed", "cancelled")
                    ],
                )
            )
            controls = _wait_for_value(
                lambda: (
                    ready
                    if not (ready := drafter.collect_lifecycle_controls()).is_empty()
                    else None
                )
            )
            self.assertEqual(
                [message.request_id for message in controls.sync_messages], ["ready"]
            )
            self.assertEqual(
                {key.request_id for key in controls.close_keys}, {"closed", "cancelled"}
            )
            self.assertEqual(drafter.pending_control_count(), 0)
            self.assertTrue(drafter.collect_lifecycle_controls().is_empty())
        finally:
            verifier.close()
            drafter.close()

    def test_cpp_mismatch_rewrite_accepts_contiguous_next_publication(self):
        verifier_endpoint = _tcp_endpoint()
        drafter_endpoint = _tcp_endpoint()
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            required_tail_len=0,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            )
        )
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(
                DraftSync(
                    request_id="rewrite-continuation",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    prompt_token_ids=[7, 8],
                    committed_outputs=[10],
                )
            )
            _wait_for_value(
                lambda: (
                    controls
                    if (controls := drafter.collect_lifecycle_controls()).sync_messages
                    else None
                )
            )
            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(
                            position,
                            token,
                            request_id="rewrite-continuation",
                            base_committed_len=1,
                        )
                        for position, token in ((1, 11), (2, 12), (3, 13))
                    ]
                )
            )
            _wait_for_value(
                lambda: verifier.snapshot_one("rewrite-continuation").tail_tokens
            )
            verifier.commit(
                VerifyCommit(
                    request_id="rewrite-continuation",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=1,
                    committed_tokens=[11, 99],
                )
            )
            drafter.wait_for_commit("rewrite-continuation", 3)

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(
                            2,
                            99,
                            request_id="rewrite-continuation",
                            base_committed_len=3,
                            is_commit_echo=True,
                        )
                    ]
                )
            )
            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(
                            3,
                            33,
                            request_id="rewrite-continuation",
                            base_committed_len=3,
                        ),
                    ]
                )
            )
            snapshot = _wait_for_value(
                lambda: (
                    current
                    if (
                        current := verifier.snapshot_one("rewrite-continuation")
                    ).tail_tokens
                    else None
                )
            )
            self.assertEqual(snapshot.committed_len, 3)
            self.assertEqual(snapshot.tail_tokens, (33,))
        finally:
            verifier.close()
            drafter.close()

    def test_cpp_one_verifier_routes_to_two_sparse_drafters(self):
        verifier_endpoint = _tcp_endpoint()
        drafter_endpoints = {3: _tcp_endpoint(), 9: _tcp_endpoint()}
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(),
                rank=5,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=3, endpoint=drafter_endpoints[3], quota=2
                    ),
                    DecoupledSpecPeerConfig(
                        rank=9, endpoint=drafter_endpoints[9], quota=1
                    ),
                ),
            ),
            required_tail_len=0,
        )
        drafters = {
            drafter_rank: self.drafter_type(
                DecoupledSpecIpcConfig(
                    bind_endpoint=drafter_endpoint,
                    connect_endpoints=(),
                    rank=drafter_rank,
                    peers=(
                        DecoupledSpecPeerConfig(
                            rank=5, endpoint=verifier_endpoint, quota=1
                        ),
                    ),
                )
            )
            for drafter_rank, drafter_endpoint in drafter_endpoints.items()
        }
        requests = {
            3: ("to-drafter-3", 103),
            9: ("to-drafter-9", 109),
        }
        try:
            for drafter in drafters.values():
                drafter.start()
            verifier.start()
            for drafter_rank, (request_id, _) in requests.items():
                verifier.open_requests(
                    [
                        (
                            DraftSync(
                                request_id=request_id,
                                src_verifier_rank=5,
                                dst_drafter_rank=drafter_rank,
                                prompt_token_ids=[7, 8],
                                committed_outputs=[],
                            ),
                            None,
                            None,
                        )
                    ]
                )

            for drafter_rank, drafter in drafters.items():
                opened = _wait_for_value(drafter.drain_controls)
                self.assertEqual(len(opened), 1)
                self.assertEqual(opened[0].dst_drafter_rank, drafter_rank)
                self.assertEqual(opened[0].sync_messages[0].src_verifier_rank, 5)
                self.assertEqual(
                    opened[0].sync_messages[0].request_id,
                    requests[drafter_rank][0],
                )

            for drafter_rank, drafter in drafters.items():
                request_id, token = requests[drafter_rank]
                drafter.publish_tails(
                    DraftTailStreamOutputBatch(
                        outputs=[
                            DraftTailStreamOutput(
                                src_drafter_rank=drafter_rank,
                                dst_verifier_rank=5,
                                request_id=request_id,
                                base_committed_len=0,
                                start_token_pos=0,
                                tokens=(token,),
                            )
                        ]
                    )
                )

            for request_id, token in requests.values():
                snapshot = _wait_for_value(
                    lambda request_id=request_id: (
                        current
                        if (current := verifier.snapshot_one(request_id)).tail_tokens
                        else None
                    )
                )
                self.assertEqual(snapshot.tail_tokens, (token,))

            for drafter_rank, (request_id, token) in requests.items():
                verifier.commit(
                    VerifyCommit(
                        request_id=request_id,
                        src_verifier_rank=5,
                        dst_drafter_rank=drafter_rank,
                        pre_verify_committed_len=0,
                        committed_tokens=[token],
                    )
                )
            for drafter_rank, drafter in drafters.items():
                drafter.wait_for_commit(requests[drafter_rank][0], 1, verifier_rank=5)
        finally:
            verifier.close()
            for drafter in drafters.values():
                drafter.close()

    def test_cpp_two_sparse_verifiers_share_one_drafter(self):
        drafter_endpoint = _tcp_endpoint()
        verifier_endpoints = {5: _tcp_endpoint(), 11: _tcp_endpoint()}
        verifiers = {
            verifier_rank: self.verifier_type(
                DecoupledSpecIpcConfig(
                    bind_endpoint=verifier_endpoint,
                    connect_endpoints=(),
                    rank=verifier_rank,
                    peers=(
                        DecoupledSpecPeerConfig(
                            rank=9, endpoint=drafter_endpoint, quota=1
                        ),
                    ),
                ),
                required_tail_len=0,
            )
            for verifier_rank, verifier_endpoint in verifier_endpoints.items()
        }
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(),
                rank=9,
                peers=tuple(
                    DecoupledSpecPeerConfig(
                        rank=verifier_rank,
                        endpoint=verifier_endpoint,
                        quota=1,
                    )
                    for verifier_rank, verifier_endpoint in verifier_endpoints.items()
                ),
            )
        )
        try:
            drafter.start()
            for verifier in verifiers.values():
                verifier.start()
            for verifier_rank, verifier in verifiers.items():
                verifier.open_request(
                    DraftSync(
                        request_id="shared-request-id",
                        src_verifier_rank=verifier_rank,
                        dst_drafter_rank=9,
                        prompt_token_ids=[7, 8],
                        committed_outputs=[],
                    )
                )

            opened = []

            def collect_opens():
                opened.extend(drafter.drain_controls())
                return opened if len(opened) == 2 else None

            _wait_for_value(collect_opens)
            self.assertEqual(
                {
                    (
                        batch.sync_messages[0].src_verifier_rank,
                        batch.sync_messages[0].request_id,
                    )
                    for batch in opened
                },
                {(5, "shared-request-id"), (11, "shared-request-id")},
            )

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        DraftTailStreamOutput(
                            src_drafter_rank=9,
                            dst_verifier_rank=verifier_rank,
                            request_id="shared-request-id",
                            base_committed_len=0,
                            start_token_pos=0,
                            tokens=(100 + verifier_rank,),
                        )
                        for verifier_rank in verifiers
                    ]
                )
            )
            for verifier_rank, verifier in verifiers.items():
                snapshot = _wait_for_value(
                    lambda verifier=verifier: (
                        current
                        if (
                            current := verifier.snapshot_one("shared-request-id")
                        ).tail_tokens
                        else None
                    )
                )
                self.assertEqual(snapshot.tail_tokens, (100 + verifier_rank,))
                verifier.commit(
                    VerifyCommit(
                        request_id="shared-request-id",
                        src_verifier_rank=verifier_rank,
                        dst_drafter_rank=9,
                        pre_verify_committed_len=0,
                        committed_tokens=[100 + verifier_rank],
                    )
                )
            for verifier_rank in verifiers:
                seat = drafter.wait_for_commit("shared-request-id", 1, verifier_rank)
                self.assertEqual(
                    drafter.gpu_tail_buffer.pending_expected_tokens[seat, 0].item(),
                    100 + verifier_rank,
                )
        finally:
            for verifier in verifiers.values():
                verifier.close()
            drafter.close()

    def test_cpp_no_peer_enqueue_close_is_interruptible(self):
        for direction in ("verifier_control", "drafter_tail"):
            with self.subTest(direction=direction):
                own_endpoint = _tcp_endpoint()
                peer_endpoint = _tcp_endpoint()
                sender = self._make_sender(direction, own_endpoint, peer_endpoint)
                try:
                    sender.start()
                    _enqueue_cpp_sender_frame(sender, direction, "no-peer")
                    time.sleep(0.05)
                    self._assert_close_is_interruptible(sender, peer_endpoint)
                finally:
                    sender.close()

    def test_cpp_no_peer_queue_saturation_fails_fast(self):
        for direction in ("verifier_control", "drafter_tail"):
            with self.subTest(direction=direction):
                own_endpoint = _tcp_endpoint()
                peer_endpoint = _tcp_endpoint()
                sender = self._make_sender(direction, own_endpoint, peer_endpoint)
                failed_request_id = None
                try:
                    sender.start()
                    # The native owner queue is capped at 8192 frames. One
                    # additional attempt covers either scheduling outcome for
                    # the queue-front frame (still queued or already in retry).
                    for index in range(8194):
                        request_id = f"{direction}-saturated-{index}"
                        try:
                            _enqueue_cpp_sender_frame(sender, direction, request_id)
                        except RuntimeError as exc:
                            self.assertIn(
                                "outbound queue reached its hard capacity",
                                str(exc),
                            )
                            failed_request_id = request_id
                            break

                    self.assertIsNotNone(failed_request_id)
                    self._assert_close_is_interruptible(sender, peer_endpoint)
                finally:
                    sender.close()

    def test_cpp_peer_late_connect_preserves_frame_order(self):
        for direction in ("verifier_control", "drafter_tail"):
            with self.subTest(direction=direction):
                own_endpoint = _tcp_endpoint()
                peer_endpoint = _tcp_endpoint()
                sender = self._make_sender(direction, own_endpoint, peer_endpoint)
                peer_context = None
                peer_socket = None
                try:
                    sender.start()
                    request_ids = (
                        f"{direction}-late-first",
                        f"{direction}-late-second",
                    )
                    for request_id in request_ids:
                        _enqueue_cpp_sender_frame(sender, direction, request_id)
                    time.sleep(0.05)

                    peer_context, peer_socket = _bind_raw_pull(peer_endpoint)
                    frames = [peer_socket.recv(), peer_socket.recv()]
                    self.assertIn(request_ids[0].encode(), frames[0])
                    self.assertIn(request_ids[1].encode(), frames[1])
                finally:
                    if peer_socket is not None:
                        peer_socket.close(linger=0)
                    if peer_context is not None:
                        peer_context.term()
                    sender.close()

    def test_cpp_peer_crash_does_not_block_close(self):
        for direction in ("verifier_control", "drafter_tail"):
            with self.subTest(direction=direction):
                own_endpoint = _tcp_endpoint()
                peer_endpoint = _tcp_endpoint()
                peer_context, peer_socket = _bind_raw_pull(peer_endpoint)
                sender = self._make_sender(direction, own_endpoint, peer_endpoint)
                try:
                    sender.start()
                    _enqueue_cpp_sender_frame(sender, direction, "before-crash")
                    self.assertIn(b"before-crash", peer_socket.recv())

                    peer_socket.close(linger=0)
                    peer_context.term()
                    peer_socket = None
                    peer_context = None
                    time.sleep(0.1)

                    _enqueue_cpp_sender_frame(sender, direction, "after-crash")
                    time.sleep(0.05)
                    self._assert_close_is_interruptible(sender, peer_endpoint)
                finally:
                    if peer_socket is not None:
                        peer_socket.close(linger=0)
                    if peer_context is not None:
                        peer_context.term()
                    sender.close()

    def _exercise_transport(
        self,
        verifier_cls,
        drafter_cls,
        *,
        context=None,
        use_inproc: bool = False,
    ):
        if use_inproc:
            suffix = uuid.uuid4().hex
            verifier_endpoint = f"inproc://decoupled-verifier-{suffix}"
            drafter_endpoint = f"inproc://decoupled-drafter-{suffix}"
        else:
            verifier_endpoint = _tcp_endpoint()
            drafter_endpoint = _tcp_endpoint()
        verifier = verifier_cls(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            required_tail_len=2,
            context=context,
        )
        drafter = drafter_cls(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=context,
        )
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(_sync())
            open_batch = _wait_for_value(drafter.drain_controls)[0]

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[_tail(0, 101), _tail(1, 102), _tail(2, 103)]
                )
            )
            before_commit = _wait_for_value(
                lambda: (
                    snapshot
                    if (snapshot := verifier.snapshot_one("req")).tail_tokens
                    else None
                )
            )
            verifier.commit(
                VerifyCommit(
                    request_id="req",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=0,
                    committed_tokens=[101, 102],
                )
            )
            # This read happens before the drafter is allowed to drain the wire
            # update and pins the verifier's local-apply-before-send contract.
            after_commit = verifier.snapshot_one("req")
            seat = drafter.wait_for_commit("req", 2)

            verifier.close_request(
                DraftClose(
                    request_id="req",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason="finished",
                )
            )
            local_closed = (
                not verifier.gpu_tail_buffer.lookup_binding("req", verifier.config.rank)
                is not None
            )
            close_batch = _wait_for_value(drafter.drain_controls)[0]

            return (
                open_batch.sync_messages[0].committed_outputs,
                _snapshot_tuple(before_commit),
                _snapshot_tuple(after_commit),
                drafter.gpu_tail_buffer.pending_expected_tokens[seat, :2].tolist(),
                local_closed,
                close_batch.close_messages[0].reason,
            )
        finally:
            drafter.close()
            verifier.close()

    def _exercise_same_rid_reopen_transport(self, verifier_cls, drafter_cls):
        verifier_endpoint = _tcp_endpoint()
        drafter_endpoint = _tcp_endpoint()
        verifier = verifier_cls(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            context=None,
        )
        drafter = drafter_cls(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=None,
        )
        old_request_id = "same-rid::draft-epoch::1"
        new_request_id = "same-rid::draft-epoch::2"
        try:
            drafter.start()
            verifier.start()
            verifier.open_request(_sync(old_request_id))
            _wait_for_value(drafter.drain_controls)

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[_tail(0, 10, request_id=old_request_id)]
                )
            )
            _wait_for_value(
                lambda: (
                    snapshot
                    if (snapshot := verifier.snapshot_one(old_request_id)).tail_tokens
                    else None
                )
            )

            verifier.close_request(
                DraftClose(
                    request_id=old_request_id,
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason="finished",
                )
            )
            verifier.open_request(_sync(new_request_id))

            received_controls = []

            def drain_reopen_controls():
                received_controls.extend(drafter.drain_controls())
                has_close = any(
                    message.request_id == old_request_id
                    for batch in received_controls
                    for message in batch.close_messages
                )
                has_sync = any(
                    message.request_id == new_request_id
                    for batch in received_controls
                    for message in batch.sync_messages
                )
                return received_controls if has_close and has_sync else None

            _wait_for_value(drain_reopen_controls)

            # This is a real late wire frame for the closed lifecycle. The
            # frame also carries a token for the replacement, so a repeated
            # user rid would immediately expose request-id aliasing.
            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(1, 11, request_id=old_request_id),
                        _tail(0, 20, request_id=new_request_id),
                    ]
                )
            )
            reopened_snapshot = _wait_for_value(
                lambda: (
                    snapshot
                    if (snapshot := verifier.snapshot_one(new_request_id)).tail_tokens
                    else None
                )
            )
            old_is_closed = (
                not verifier.gpu_tail_buffer.lookup_binding(
                    old_request_id, verifier.config.rank
                )
                is not None
            )
            control_events = []
            for batch in received_controls:
                control_events.extend(
                    ("close", message.request_id) for message in batch.close_messages
                )
                control_events.extend(
                    ("sync", message.request_id) for message in batch.sync_messages
                )
            return control_events, old_is_closed, _snapshot_tuple(reopened_snapshot)
        finally:
            drafter.close()
            verifier.close()

    def test_python_cpp_transport_contract_is_differential(self):
        python_outcome = self._exercise_transport(
            VerifierDecoupledSpecDataPlane,
            DrafterDecoupledSpecDataPlane,
        )
        cpp_outcome = self._exercise_transport(
            CppVerifierDecoupledSpecDataPlane,
            CppDrafterDecoupledSpecDataPlane,
        )
        self.assertEqual(python_outcome, cpp_outcome)
        self.assertEqual(cpp_outcome[1][1:3], (0, (101, 102, 103)))
        self.assertEqual(cpp_outcome[2][1:3], (2, (103,)))
        self.assertTrue(cpp_outcome[4])

    def test_same_rid_reopen_drops_late_old_frame_python_cpp_differential(self):
        python_outcome = self._exercise_same_rid_reopen_transport(
            VerifierDecoupledSpecDataPlane,
            DrafterDecoupledSpecDataPlane,
        )
        cpp_outcome = self._exercise_same_rid_reopen_transport(
            CppVerifierDecoupledSpecDataPlane,
            CppDrafterDecoupledSpecDataPlane,
        )

        self.assertEqual(python_outcome, cpp_outcome)
        self.assertEqual(
            cpp_outcome[0],
            [
                ("close", "same-rid::draft-epoch::1"),
                ("sync", "same-rid::draft-epoch::2"),
            ],
        )
        self.assertTrue(cpp_outcome[1])
        self.assertEqual(
            cpp_outcome[2][0:3],
            ("same-rid::draft-epoch::2", 0, (20,)),
        )

    def test_cpp_transport_shares_an_injected_pyzmq_context(self):
        context = zmq.Context()
        try:
            outcome = self._exercise_transport(
                CppVerifierDecoupledSpecDataPlane,
                CppDrafterDecoupledSpecDataPlane,
                context=context,
                use_inproc=True,
            )
            self.assertEqual(outcome[1][1:3], (0, (101, 102, 103)))
            self.assertEqual(outcome[2][1:3], (2, (103,)))
        finally:
            context.destroy(linger=0)


class TestPythonSocketDataPlane(TestCppDecoupledSpecDataPlane):
    verifier_type = VerifierDecoupledSpecDataPlane
    drafter_type = DrafterDecoupledSpecDataPlane


if __name__ == "__main__":
    unittest.main()

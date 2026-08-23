"""Differential CPU tests for the Python and C++ decoupled-spec data planes."""

import socket
import threading
import time
import unittest
import uuid

import zmq

from sglang.srt.environ import envs
from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDrafterDecoupledSpecDataPlane,
    CppDraftTailBuffer,
    CppVerifierDecoupledSpecDataPlane,
    _peer_rows,
)
from sglang.srt.speculative.decoupled_spec_data_plane import (
    DrafterDecoupledSpecDataPlane,
    VerifierDecoupledSpecDataPlane,
    create_drafter_decoupled_spec_data_plane,
    create_verifier_decoupled_spec_data_plane,
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
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


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
    token_pos: int,
    token: int,
    *,
    request_id: str = "req",
    base_committed_len: int = 0,
) -> DraftTailStreamOutput:
    return DraftTailStreamOutput(
        src_drafter_rank=0,
        dst_verifier_rank=0,
        request_id=request_id,
        base_committed_len=base_committed_len,
        new_token_pos=token_pos,
        new_token=token,
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


def _make_cpp_sender(direction: str, own_endpoint: str, peer_endpoint: str):
    config = DecoupledSpecIpcConfig(
        bind_endpoint=own_endpoint,
        connect_endpoints=(peer_endpoint,),
        rank=0,
    )
    if direction == "verifier_control":
        return CppVerifierDecoupledSpecDataPlane(config, required_tail_len=0)
    if direction == "drafter_tail":
        return CppDrafterDecoupledSpecDataPlane(config)
    raise AssertionError(f"Unknown C++ sender direction: {direction}")


def _enqueue_cpp_sender_frame(sender, direction: str, request_id: str) -> None:
    if direction == "verifier_control":
        sender.open_request(_sync(request_id))
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


class TestCppDraftTailBuffer(CustomTestCase):
    def test_python_cpp_state_machine_is_differential(self):
        outcomes = []
        for buffer_cls in (DraftTailBuffer, CppDraftTailBuffer):
            buffer = buffer_cls(verifier_rank=0, required_tail_len=2)
            try:
                buffer.open_request(_sync())
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[_tail(0, 10), _tail(1, 11), _tail(2, 12)]
                    )
                )
                immutable_snapshot = buffer.snapshot("req", max_tail_len=3)
                buffer.apply_verify_commit(
                    VerifyCommit(
                        request_id="req",
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=0,
                        committed_tokens=[10, 11],
                    )
                )
                full_match_snapshot = buffer.snapshot("req")

                buffer.apply_verify_commit(
                    VerifyCommit(
                        request_id="req",
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=2,
                        committed_tokens=[99],
                    )
                )
                pending_snapshot = buffer.snapshot("req")
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[
                            _tail(2, 99, base_committed_len=2),
                            _tail(3, 13, base_committed_len=3),
                        ]
                    )
                )
                realigned_snapshot = buffer.snapshot("req")
                outcomes.append(
                    (
                        _snapshot_tuple(immutable_snapshot),
                        _snapshot_tuple(full_match_snapshot),
                        _snapshot_tuple(pending_snapshot),
                        _snapshot_tuple(realigned_snapshot),
                        _snapshot_tuple(immutable_snapshot),
                    )
                )
            finally:
                buffer.close()

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0][2], (10, 11, 12))
        self.assertEqual(outcomes[0][1][1:3], (2, (12,)))
        self.assertEqual(outcomes[0][2][1:3], (2, ()))
        self.assertEqual(outcomes[0][3][1:3], (3, (13,)))
        self.assertEqual(outcomes[0][4][2], (10, 11, 12))

    def test_short_tail_pending_confirmation_accepts_older_valid_base(self):
        outcomes = []
        for buffer_cls in (DraftTailBuffer, CppDraftTailBuffer):
            buffer = buffer_cls(verifier_rank=0, required_tail_len=2)
            try:
                buffer.open_request(_sync())
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(outputs=[_tail(0, 10)])
                )
                buffer.apply_verify_commit(
                    VerifyCommit(
                        request_id="req",
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=0,
                        committed_tokens=[10, 11],
                    )
                )
                pending_snapshot = buffer.snapshot("req")

                # The drafter emitted position 1 from base 0 before observing
                # the commit. That base remains valid because every token up to
                # this pending position matched and can_accept_prefix_len did
                # not advance.
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[_tail(1, 11, base_committed_len=0)]
                    )
                )
                confirmed_snapshot = buffer.snapshot("req")
                outcomes.append(
                    (
                        _snapshot_tuple(pending_snapshot),
                        _snapshot_tuple(confirmed_snapshot),
                    )
                )
            finally:
                buffer.close()

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0][1:3], (1, ()))
        self.assertEqual(outcomes[0][1][1:3], (2, ()))

    def test_snapshot_commit_race_has_only_linearized_outcomes(self):
        expected = {(0, (10, 11, 12)), (2, (12,))}
        for buffer_cls in (DraftTailBuffer, CppDraftTailBuffer):
            observed = set()
            for _ in range(20):
                buffer = buffer_cls(verifier_rank=0, required_tail_len=2)
                try:
                    buffer.open_request(_sync())
                    buffer.append_draft_stream_batch(
                        DraftTailStreamOutputBatch(
                            outputs=[_tail(0, 10), _tail(1, 11), _tail(2, 12)]
                        )
                    )
                    barrier = threading.Barrier(3)
                    snapshots = []
                    errors = []

                    def take_snapshot():
                        try:
                            barrier.wait()
                            snapshots.append(buffer.snapshot("req"))
                        except BaseException as exc:
                            errors.append(exc)

                    def apply_commit():
                        try:
                            barrier.wait()
                            buffer.apply_verify_commit(
                                VerifyCommit(
                                    request_id="req",
                                    src_verifier_rank=0,
                                    dst_drafter_rank=0,
                                    pre_verify_committed_len=0,
                                    committed_tokens=[10, 11],
                                )
                            )
                        except BaseException as exc:
                            errors.append(exc)

                    snapshot_thread = threading.Thread(target=take_snapshot)
                    commit_thread = threading.Thread(target=apply_commit)
                    snapshot_thread.start()
                    commit_thread.start()
                    barrier.wait()
                    snapshot_thread.join()
                    commit_thread.join()

                    self.assertEqual(errors, [])
                    observed.add(
                        (
                            snapshots[0].committed_len,
                            tuple(snapshots[0].tail_tokens),
                        )
                    )
                    self.assertEqual(buffer.snapshot("req").committed_len, 2)
                finally:
                    buffer.close()
            self.assertTrue(observed.issubset(expected))

    def test_python_cpp_match_for_pending_tokens_and_same_rid_reopen(self):
        outcomes = []
        for buffer_cls in (DraftTailBuffer, CppDraftTailBuffer):
            buffer = buffer_cls(verifier_rank=0, required_tail_len=2)
            old_request_id = "req::draft-epoch::1"
            new_request_id = "req::draft-epoch::2"
            try:
                buffer.open_request(_sync(old_request_id))
                buffer.apply_verify_commit(
                    VerifyCommit(
                        request_id=old_request_id,
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=0,
                        committed_tokens=[20],
                    )
                )
                buffer.apply_verify_commit(
                    VerifyCommit(
                        request_id=old_request_id,
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=1,
                        committed_tokens=[21],
                    )
                )
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[
                            _tail(
                                1,
                                21,
                                request_id=old_request_id,
                                base_committed_len=0,
                            )
                        ]
                    )
                )
                before_confirmation = buffer.snapshot(old_request_id)
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[
                            _tail(
                                0,
                                20,
                                request_id=old_request_id,
                                base_committed_len=0,
                            ),
                            _tail(
                                1,
                                21,
                                request_id=old_request_id,
                                base_committed_len=0,
                            ),
                        ]
                    )
                )
                after_confirmation = buffer.snapshot(old_request_id)

                buffer.close_request(
                    DraftClose(
                        request_id=old_request_id,
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        reason="finished",
                    )
                )
                buffer.open_request(_sync(new_request_id))
                buffer.append_draft_stream_batch(
                    DraftTailStreamOutputBatch(
                        outputs=[
                            _tail(2, 22, request_id=old_request_id),
                            _tail(0, 30, request_id=new_request_id),
                        ]
                    )
                )
                reseated = buffer.snapshot(new_request_id)
                outcomes.append(
                    (
                        _snapshot_tuple(before_confirmation),
                        _snapshot_tuple(after_confirmation),
                        _snapshot_tuple(reseated),
                    )
                )
            finally:
                buffer.close()

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0][0][1:3], (0, ()))
        self.assertEqual(outcomes[0][1][1:3], (2, ()))
        self.assertEqual(outcomes[0][2][0:3], (new_request_id, 0, (30,)))


class TestCppDecoupledSpecDataPlane(CustomTestCase):
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

    def test_cpp_one_verifier_routes_to_two_sparse_drafters(self):
        verifier_endpoint = _tcp_endpoint()
        drafter_endpoints = {3: _tcp_endpoint(), 9: _tcp_endpoint()}
        verifier = CppVerifierDecoupledSpecDataPlane(
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
            drafter_rank: CppDrafterDecoupledSpecDataPlane(
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
                                new_token_pos=0,
                                new_token=token,
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
                committed = _wait_for_value(drafter.drain_controls)
                self.assertEqual(len(committed), 1)
                commit = committed[0].verify_commit_messages[0]
                self.assertEqual(commit.dst_drafter_rank, drafter_rank)
                self.assertEqual(commit.request_id, requests[drafter_rank][0])
        finally:
            verifier.close()
            for drafter in drafters.values():
                drafter.close()

    def test_cpp_two_sparse_verifiers_share_one_drafter(self):
        drafter_endpoint = _tcp_endpoint()
        verifier_endpoints = {5: _tcp_endpoint(), 11: _tcp_endpoint()}
        verifiers = {
            verifier_rank: CppVerifierDecoupledSpecDataPlane(
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
        drafter = CppDrafterDecoupledSpecDataPlane(
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
                            new_token_pos=0,
                            new_token=100 + verifier_rank,
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
            committed = []

            def collect_commits():
                committed.extend(drafter.drain_controls())
                return committed if len(committed) == 2 else None

            _wait_for_value(collect_commits)
            self.assertEqual(
                {
                    (
                        batch.verify_commit_messages[0].src_verifier_rank,
                        batch.verify_commit_messages[0].committed_tokens[0],
                    )
                    for batch in committed
                },
                {(5, 105), (11, 111)},
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
                sender = _make_cpp_sender(direction, own_endpoint, peer_endpoint)
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
                sender = _make_cpp_sender(direction, own_endpoint, peer_endpoint)
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
                    if direction == "verifier_control":
                        # Capacity is checked before local apply, so a frame
                        # that never entered the wire queue cannot open local
                        # verifier tail state.
                        self.assertFalse(
                            sender.draft_tail_buffer.has_request(failed_request_id)
                        )
                    self._assert_close_is_interruptible(sender, peer_endpoint)
                finally:
                    sender.close()

    def test_cpp_peer_late_connect_preserves_frame_order(self):
        for direction in ("verifier_control", "drafter_tail"):
            with self.subTest(direction=direction):
                own_endpoint = _tcp_endpoint()
                peer_endpoint = _tcp_endpoint()
                sender = _make_cpp_sender(direction, own_endpoint, peer_endpoint)
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
                sender = _make_cpp_sender(direction, own_endpoint, peer_endpoint)
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

    def _exercise_segmented_drafter_inbox(self, verifier_cls, drafter_cls):
        verifier_endpoint = _tcp_endpoint()
        drafter_endpoint = _tcp_endpoint()
        verifier = verifier_cls(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            required_tail_len=0,
        )
        drafter = drafter_cls(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            )
        )

        def sync(request_id):
            return DraftSync(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=[10],
            )

        def commit(request_id, pre, tokens):
            return VerifyCommit(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=pre,
                committed_tokens=tokens,
            )

        try:
            drafter.start()
            verifier.start()
            request_ids = ("blocked", "ready", "mismatch")
            verifier.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    sync_messages=[sync(request_id) for request_id in request_ids],
                )
            )
            opened = _wait_for_value(
                lambda: (
                    controls
                    if len(
                        controls := drafter.collect_ready_controls(
                            lambda _segment: 0
                        ).sync_messages
                    )
                    == 3
                    else None
                )
            )
            self.assertEqual(
                {message.request_id for message in opened}, set(request_ids)
            )

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(1, 11, request_id="blocked", base_committed_len=1),
                        _tail(1, 21, request_id="ready", base_committed_len=1),
                        _tail(1, 11, request_id="mismatch", base_committed_len=1),
                        _tail(2, 12, request_id="mismatch", base_committed_len=1),
                        _tail(3, 13, request_id="mismatch", base_committed_len=1),
                    ]
                )
            )
            for request_id in request_ids:
                _wait_for_value(
                    lambda request_id=request_id: (
                        snapshot
                        if (snapshot := verifier.snapshot_one(request_id)).tail_tokens
                        else None
                    )
                )

            # The two blocked messages must coalesce into one contiguous segment.
            # A separate ready key and a mismatch key remain independently
            # consumable in the same collection pass.
            verifier.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    verify_commit_messages=[
                        commit("blocked", 1, [11]),
                        commit("blocked", 2, [12]),
                        commit("ready", 1, [21]),
                        commit("mismatch", 1, [11, 99, 100]),
                    ],
                )
            )

            def first_consumable(segment):
                return {
                    "blocked": 0,
                    "ready": 1,
                    # One matching token plus the first rewrite token. The
                    # following authoritative token belongs to a later replay.
                    "mismatch": 2,
                }[segment.draft_key.request_id]

            first = _wait_for_value(
                lambda: (
                    controls
                    if len(
                        controls := drafter.collect_ready_controls(
                            first_consumable
                        ).ready_commit_segments
                    )
                    == 2
                    else None
                )
            )
            first_rows = sorted(
                (
                    segment.draft_key.request_id,
                    segment.pre_verify_committed_len,
                    tuple(segment.committed_tokens),
                )
                for segment in first
            )

            second = drafter.collect_ready_controls(
                lambda segment: 1 if segment.draft_key.request_id == "blocked" else 0
            ).ready_commit_segments
            second_rows = [
                (
                    segment.draft_key.request_id,
                    segment.pre_verify_committed_len,
                    tuple(segment.committed_tokens),
                )
                for segment in second
            ]

            pending_rows = []

            def record_pending(segment):
                pending_rows.append(
                    (
                        segment.draft_key.request_id,
                        segment.pre_verify_committed_len,
                        tuple(segment.committed_tokens),
                    )
                )
                return 0

            drafter.collect_ready_controls(record_pending)
            pending_rows.sort()

            verifier.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    close_messages=[
                        DraftClose(
                            request_id=request_id,
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            reason="finished",
                        )
                        for request_id in ("blocked", "mismatch")
                    ],
                )
            )
            closed = _wait_for_value(
                lambda: (
                    keys
                    if len(
                        keys := drafter.collect_ready_controls(
                            lambda _segment: 0
                        ).close_keys
                    )
                    == 2
                    else None
                )
            )
            closed_rows = sorted(
                (key.src_verifier_rank, key.request_id) for key in closed
            )
            final = drafter.collect_ready_controls(lambda _segment: 0)
            return (
                first_rows,
                second_rows,
                pending_rows,
                closed_rows,
                final.is_empty(),
                drafter.pending_control_count(),
            )
        finally:
            drafter.close()
            verifier.close()

    def test_python_cpp_segmented_inbox_match_after_real_zmq_ingress(self):
        python_result = self._exercise_segmented_drafter_inbox(
            VerifierDecoupledSpecDataPlane,
            DrafterDecoupledSpecDataPlane,
        )
        cpp_result = self._exercise_segmented_drafter_inbox(
            CppVerifierDecoupledSpecDataPlane,
            CppDrafterDecoupledSpecDataPlane,
        )

        self.assertEqual(cpp_result, python_result)
        self.assertEqual(
            python_result[0],
            [
                ("mismatch", 1, (11, 99)),
                ("ready", 1, (21,)),
            ],
        )
        self.assertEqual(python_result[1], [("blocked", 1, (11,))])
        self.assertEqual(
            python_result[2],
            [
                ("blocked", 2, (12,)),
                ("mismatch", 3, (100,)),
            ],
        )
        self.assertEqual(
            python_result[3],
            [(0, "blocked"), (0, "mismatch")],
        )
        self.assertTrue(python_result[4])
        self.assertEqual(python_result[5], 0)

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
            commit_batch = _wait_for_value(drafter.drain_controls)[0]

            verifier.close_request(
                DraftClose(
                    request_id="req",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason="finished",
                )
            )
            local_closed = not verifier.draft_tail_buffer.has_request("req")
            close_batch = _wait_for_value(drafter.drain_controls)[0]

            return (
                open_batch.sync_messages[0].committed_outputs,
                _snapshot_tuple(before_commit),
                _snapshot_tuple(after_commit),
                commit_batch.verify_commit_messages[0].committed_tokens,
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
            old_is_closed = not verifier.draft_tail_buffer.has_request(old_request_id)
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

    def test_factory_selects_one_contract_for_both_roles(self):
        for use_cpp, verifier_cls, drafter_cls in (
            (
                False,
                VerifierDecoupledSpecDataPlane,
                DrafterDecoupledSpecDataPlane,
            ),
            (
                True,
                CppVerifierDecoupledSpecDataPlane,
                CppDrafterDecoupledSpecDataPlane,
            ),
        ):
            verifier_endpoint = _tcp_endpoint()
            drafter_endpoint = _tcp_endpoint()
            with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(use_cpp):
                verifier = create_verifier_decoupled_spec_data_plane(
                    DecoupledSpecIpcConfig(
                        bind_endpoint=verifier_endpoint,
                        connect_endpoints=(drafter_endpoint,),
                        rank=0,
                    )
                )
                drafter = create_drafter_decoupled_spec_data_plane(
                    DecoupledSpecIpcConfig(
                        bind_endpoint=drafter_endpoint,
                        connect_endpoints=(verifier_endpoint,),
                        rank=0,
                    )
                )
                try:
                    self.assertIsInstance(verifier, verifier_cls)
                    self.assertIsInstance(drafter, drafter_cls)
                finally:
                    drafter.close()
                    verifier.close()


if __name__ == "__main__":
    unittest.main()

"""CPU tests for the Python decoupled-spec data plane."""

import threading
import time
import unittest
import uuid

import zmq

from sglang.srt.speculative.decoupled_spec_data_plane import (
    DrafterDecoupledSpecDataPlane,
    VerifierDecoupledSpecDataPlane,
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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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


def _wait_for_value(callback, timeout_s: float = 1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = callback()
        if value:
            return value
        time.sleep(0.001)
    raise AssertionError("Timed out waiting for decoupled-spec background transport")


class TestDraftTailBuffer(CustomTestCase):
    def test_append_accepts_one_contiguous_token_span(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())

        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(0, (10, 11, 12))])
        )

        self.assertEqual(buffer.snapshot("req").tail_tokens, (10, 11, 12))

    def test_conflicting_overlap_rejects_entire_span(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(0, (10, 11, 12))])
        )

        with self.assertRaises(RuntimeError):
            buffer.append_draft_stream_batch(
                DraftTailStreamOutputBatch(outputs=[_tail(1, (11, 99))])
            )

        self.assertEqual(buffer.snapshot("req").tail_tokens, (10, 11, 12))

    def test_commit_echo_then_retained_span_apply_in_one_batch(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(0, (10, 11))])
        )
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[10, 11, 12],
            )
        )

        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[
                    _tail(2, 12, base_committed_len=3, is_commit_echo=True),
                    _tail(3, (13, 14), base_committed_len=3),
                ]
            )
        )

        snapshot = buffer.snapshot("req")
        self.assertEqual(snapshot.committed_len, 3)
        self.assertEqual(snapshot.tail_tokens, (13, 14))

    def test_pending_prefix_state_is_request_local(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_requests([_sync("blocked"), _sync("ready")])
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="blocked",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[20],
            )
        )

        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(0, 30, request_id="ready")])
        )

        self.assertEqual(buffer.snapshot("blocked").tail_tokens, ())
        self.assertEqual(buffer.snapshot("ready").tail_tokens, (30,))

    def test_mismatching_short_tail_waits_for_cumulative_commit_echo(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(0, 10, base_committed_len=0)])
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

        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[
                    _tail(1, 999, base_committed_len=0),
                    _tail(
                        1,
                        11,
                        base_committed_len=2,
                        is_commit_echo=True,
                    ),
                ]
            )
        )

        snapshot = buffer.snapshot("req")
        self.assertEqual(snapshot.committed_len, 2)
        self.assertEqual(snapshot.tail_tokens, ())

    def test_snapshot_and_commit_have_only_linearized_outcomes(self):
        for _ in range(50):
            buffer = DraftTailBuffer(verifier_rank=0)
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
            self.assertEqual(len(snapshots), 1)
            outcome = (snapshots[0].committed_len, snapshots[0].tail_tokens)
            self.assertIn(outcome, ((0, (10, 11, 12)), (2, (12,))))
            self.assertEqual(buffer.snapshot("req").committed_len, 2)

    def test_full_match_preserves_suffix_and_snapshot_is_immutable(self):
        buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        buffer.open_request(_sync())
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[_tail(0, 10), _tail(1, 11), _tail(2, 12)]
            )
        )

        snapshot_before = buffer.snapshot("req", max_tail_len=3)
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[10, 11],
            )
        )
        snapshot_after = buffer.snapshot("req", max_tail_len=3)

        self.assertEqual(snapshot_before.tail_tokens, (10, 11, 12))
        self.assertEqual(snapshot_after.committed_len, 2)
        self.assertEqual(snapshot_after.tail_tokens, (12,))
        self.assertEqual(snapshot_before.tail_tokens, (10, 11, 12))

    def test_mismatch_waits_for_rewrite_and_rejects_stale_base(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[_tail(0, 10), _tail(1, 99), _tail(2, 77)]
            )
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
        self.assertEqual(pending_snapshot.committed_len, 2)
        self.assertEqual(pending_snapshot.tail_tokens, ())

        # The mismatching stream was based before the rewrite boundary.
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(1, 11)])
        )
        self.assertEqual(buffer.snapshot("req").committed_len, 2)

        # The aligned drafter confirms the verifier-owned token, then publishes
        # a new tail from the rewritten prefix.
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[
                    _tail(
                        1,
                        11,
                        base_committed_len=2,
                        is_commit_echo=True,
                    ),
                    _tail(2, 12, base_committed_len=2),
                ]
            )
        )
        aligned_snapshot = buffer.snapshot("req")
        self.assertEqual(aligned_snapshot.committed_len, 2)
        self.assertEqual(aligned_snapshot.tail_tokens, (12,))

    def test_short_tail_accumulates_expected_commits_until_confirmed(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[20, 21],
            )
        )
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(0, 20)])
        )
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=2,
                committed_tokens=[22],
            )
        )
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[
                    _tail(
                        1,
                        21,
                        base_committed_len=2,
                        is_commit_echo=True,
                    ),
                    _tail(
                        2,
                        22,
                        base_committed_len=3,
                        is_commit_echo=True,
                    ),
                    _tail(3, 23, base_committed_len=3),
                ]
            )
        )

        snapshot = buffer.snapshot("req")
        self.assertEqual(snapshot.committed_len, 3)
        self.assertEqual(snapshot.tail_tokens, (23,))

    def test_cumulative_commit_echo_is_idempotent(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        buffer.open_request(_sync())
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[20],
            )
        )
        buffer.apply_verify_commit(
            VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=1,
                committed_tokens=[21],
            )
        )

        # An out-of-order output cannot skip the first pending verifier token.
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(outputs=[_tail(1, 21, base_committed_len=0)])
        )
        self.assertEqual(buffer.snapshot("req").committed_len, 2)

        ack_one = _tail(
            0,
            20,
            base_committed_len=1,
            is_commit_echo=True,
        )
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[
                    ack_one,
                    ack_one,
                    _tail(
                        1,
                        21,
                        base_committed_len=2,
                        is_commit_echo=True,
                    ),
                    _tail(2, 22, base_committed_len=2),
                ]
            )
        )
        snapshot = buffer.snapshot("req")
        self.assertEqual(snapshot.committed_len, 2)
        self.assertEqual(snapshot.tail_tokens, (22,))

        with self.assertRaisesRegex(RuntimeError, "ACK is ahead"):
            buffer.append_draft_stream_batch(
                DraftTailStreamOutputBatch(
                    outputs=[
                        _tail(
                            2,
                            22,
                            base_committed_len=3,
                            is_commit_echo=True,
                        )
                    ]
                )
            )

    def test_close_and_same_rid_reopen_use_distinct_wire_epochs(self):
        buffer = DraftTailBuffer(verifier_rank=0)
        old_request_id = "req::draft-epoch::1"
        new_request_id = "req::draft-epoch::2"
        buffer.open_request(_sync(old_request_id))
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[_tail(0, 10, request_id=old_request_id)]
            )
        )
        buffer.close_request(
            DraftClose(
                request_id=old_request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                reason="finished",
            )
        )
        buffer.open_request(_sync(new_request_id))

        # In-flight data from the closed generation is ignored and cannot
        # mutate the new row, even though both generations share the user rid.
        buffer.append_draft_stream_batch(
            DraftTailStreamOutputBatch(
                outputs=[
                    _tail(1, 11, request_id=old_request_id),
                    _tail(0, 20, request_id=new_request_id),
                ]
            )
        )

        with self.assertRaises(KeyError):
            buffer.snapshot(old_request_id)
        snapshot = buffer.snapshot(new_request_id)
        self.assertEqual(snapshot.committed_len, 0)
        self.assertEqual(snapshot.tail_tokens, (20,))


class TestPythonDecoupledSpecDataPlane(CustomTestCase):
    def test_sparse_ranked_bidirectional_transport(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://decoupled-verifier-{suffix}"
        drafter_endpoint = f"inproc://decoupled-drafter-{suffix}"
        verifier = VerifierDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(),
                rank=5,
                peers=(
                    DecoupledSpecPeerConfig(rank=9, endpoint=drafter_endpoint, quota=1),
                ),
            ),
            context=context,
        )
        drafter = DrafterDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(),
                rank=9,
                peers=(
                    DecoupledSpecPeerConfig(
                        rank=5, endpoint=verifier_endpoint, quota=1
                    ),
                ),
            ),
            context=context,
        )

        try:
            verifier.start()
            drafter.start()
            verifier.open_request(
                DraftSync(
                    request_id="sparse",
                    src_verifier_rank=5,
                    dst_drafter_rank=9,
                    prompt_token_ids=[7, 8],
                    committed_outputs=[],
                )
            )
            open_batches = _wait_for_value(drafter.drain_controls)
            self.assertEqual(open_batches[0].dst_drafter_rank, 9)
            self.assertEqual(open_batches[0].sync_messages[0].src_verifier_rank, 5)

            drafter.publish_tails(
                DraftTailStreamOutputBatch(
                    outputs=[
                        DraftTailStreamOutput(
                            src_drafter_rank=9,
                            dst_verifier_rank=5,
                            request_id="sparse",
                            base_committed_len=0,
                            start_token_pos=0,
                            tokens=(101,),
                        )
                    ]
                )
            )
            snapshot = _wait_for_value(
                lambda: (
                    current
                    if (current := verifier.snapshot_one("sparse")).tail_tokens
                    else None
                )
            )
            self.assertEqual(snapshot.tail_tokens, (101,))
        finally:
            drafter.close()
            verifier.close()
            context.destroy(linger=0)

    def test_bidirectional_zmq_transport(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://decoupled-verifier-{suffix}"
        drafter_endpoint = f"inproc://decoupled-drafter-{suffix}"
        verifier = VerifierDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            context=context,
        )
        drafter = DrafterDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=context,
        )

        try:
            verifier.start()
            drafter.start()
            verifier.open_request(_sync())
            open_batches = _wait_for_value(drafter.drain_controls)
            self.assertEqual(open_batches[0].sync_messages[0].request_id, "req")

            drafter.publish_tails(
                DraftTailStreamOutputBatch(outputs=[_tail(0, 101), _tail(1, 102)])
            )
            snapshot = _wait_for_value(
                lambda: (
                    current
                    if (current := verifier.snapshot_one("req")).tail_tokens
                    else None
                )
            )
            self.assertEqual(snapshot.tail_tokens, (101, 102))

            verifier.commit(
                VerifyCommit(
                    request_id="req",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=0,
                    committed_tokens=[101],
                )
            )
            locally_committed = verifier.snapshot_one("req")
            self.assertEqual(locally_committed.committed_len, 1)
            self.assertEqual(locally_committed.tail_tokens, (102,))
            commit_batches = _wait_for_value(drafter.drain_controls)
            self.assertEqual(
                commit_batches[0].verify_commit_messages[0].committed_tokens,
                [101],
            )

            verifier.close_request(
                DraftClose(
                    request_id="req",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason="finished",
                )
            )
            self.assertFalse(verifier.draft_tail_buffer.has_request("req"))
            close_batches = _wait_for_value(drafter.drain_controls)
            self.assertEqual(close_batches[0].close_messages[0].reason, "finished")
        finally:
            drafter.close()
            verifier.close()
            context.destroy(linger=0)

    def test_remote_only_verify_commit_does_not_mutate_local_tail(self):
        context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://decoupled-verifier-{suffix}"
        drafter_endpoint = f"inproc://decoupled-drafter-{suffix}"
        verifier = VerifierDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            context=context,
        )
        drafter = DrafterDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=context,
        )

        try:
            verifier.start()
            drafter.start()
            verifier.open_request(_sync())
            _wait_for_value(drafter.drain_controls)
            verifier.draft_tail_buffer.append_draft_stream_batch(
                DraftTailStreamOutputBatch(outputs=[_tail(0, 101)])
            )
            commit = VerifyCommit(
                request_id="req",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[101],
            )

            verifier.submit_control_batch(
                DraftControlBatch(
                    dst_drafter_rank=0,
                    verify_commit_messages=[commit],
                ),
                apply_local_verify_commits=False,
            )

            local = verifier.snapshot_one("req")
            self.assertEqual(local.committed_len, 0)
            self.assertEqual(local.tail_tokens, (101,))
            remote = _wait_for_value(drafter.drain_controls)
            self.assertEqual(remote[0].verify_commit_messages, [commit])
        finally:
            drafter.close()
            verifier.close()
            context.destroy(linger=0)


if __name__ == "__main__":
    unittest.main()

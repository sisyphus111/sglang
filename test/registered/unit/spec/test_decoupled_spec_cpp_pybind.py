import time
import unittest

from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDraftProxyThread,
    CppDraftTailBuffer,
    CppTokenSyncThread,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-test-cpu")


class _Req:
    def __init__(self, rid: str):
        self.rid = rid


def _snapshot(buffer, rid: str):
    return buffer.get_draft_snapshots(
        [_Req(rid)], allow_partial=True, include_raw_tail_tokens=True
    )[0]


def _exercise_tail_buffer(buffer):
    rid = "req-a"
    buffer.apply_control_batch(
        DraftControlBatch(
            0,
            sync_messages=[
                DraftSync(
                    request_id=rid,
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    prompt_token_ids=[1, 2],
                    committed_output_ids=[3],
                )
            ],
        )
    )
    buffer.append_draft_stream_batch(
        DraftTailStreamOutputBatch(
            [
                DraftTailStreamOutput(0, 0, rid, 1, 1, 10),
                DraftTailStreamOutput(0, 0, rid, 1, 2, 11),
            ]
        )
    )
    first = _snapshot(buffer, rid)

    buffer.apply_control_batch(
        DraftControlBatch(
            0,
            verify_commit_messages=[VerifyCommit(rid, 0, 0, 1, [10])],
        )
    )
    second = _snapshot(buffer, rid)

    buffer.apply_control_batch(
        DraftControlBatch(
            0,
            verify_commit_messages=[VerifyCommit(rid, 0, 0, 2, [99])],
        )
    )
    third = _snapshot(buffer, rid)

    buffer.append_draft_stream_batch(
        DraftTailStreamOutputBatch([DraftTailStreamOutput(0, 0, rid, 2, 2, 99)])
    )
    fourth = _snapshot(buffer, rid)

    buffer.append_draft_stream_batch(
        DraftTailStreamOutputBatch([DraftTailStreamOutput(0, 0, rid, 3, 3, 12)])
    )
    fifth = _snapshot(buffer, rid)
    return [
        (
            item.committed_len,
            item.tail_tokens,
            item.raw_tail_len,
            item.raw_tail_tokens,
        )
        for item in [first, second, third, fourth, fifth]
    ]


class TestDecoupledSpecCppPybind(unittest.TestCase):
    def test_draft_tail_buffer_matches_python_state_machine(self):
        py_buffer = DraftTailBuffer(verifier_rank=0, required_tail_len=2)
        cpp_buffer = CppDraftTailBuffer(verifier_rank=0, required_tail_len=2)
        try:
            self.assertEqual(
                _exercise_tail_buffer(cpp_buffer),
                _exercise_tail_buffer(py_buffer),
            )
        finally:
            py_buffer.close()
            cpp_buffer.close()

    def test_cpp_proxy_token_sync_roundtrip(self):
        buffer = CppDraftTailBuffer(verifier_rank=0, required_tail_len=2)
        proxy = CppDraftProxyThread(
            context=None,
            verifier_rank=0,
            draft_tail_buffer=buffer,
        )
        token_sync = CppTokenSyncThread(context=None, drafter_rank=0)
        try:
            proxy.configure_peer_endpoints([token_sync.control_bind_endpoint])
            token_sync.configure_peer_endpoints([proxy.result_bind_endpoint])
            proxy.start()
            token_sync.start()

            proxy.submit_control_batch(
                DraftControlBatch(
                    0,
                    sync_messages=[
                        DraftSync(
                            request_id="req-b",
                            src_verifier_rank=0,
                            dst_drafter_rank=0,
                            prompt_token_ids=[1, 2],
                            committed_output_ids=[3],
                        )
                    ],
                    verify_commit_messages=[
                        VerifyCommit("req-b", 0, 0, 1, [10]),
                    ],
                )
            )

            def collector(inbox):
                return inbox.extract_ready_controls_locked(
                    lambda segment: len(segment.committed_token_ids)
                )

            ready = None
            for _ in range(2000):
                ready = token_sync.collect_ready_draft_controls(collector)
                if ready.sync_messages and ready.ready_commit_segments:
                    break
                time.sleep(0.001)

            self.assertIsNotNone(ready)
            self.assertEqual([msg.request_id for msg in ready.sync_messages], ["req-b"])
            self.assertEqual(
                [seg.committed_token_ids for seg in ready.ready_commit_segments],
                [[10]],
            )

            token_sync.submit_draft_results(
                DraftTailStreamOutputBatch(
                    [DraftTailStreamOutput(0, 0, "req-b", 1, 1, 10)]
                )
            )
            for _ in range(2000):
                snapshot = _snapshot(buffer, "req-b")
                if snapshot.committed_len == 2:
                    break
                time.sleep(0.001)
            self.assertEqual(snapshot.committed_len, 2)
            self.assertEqual(snapshot.raw_tail_tokens, [])
        finally:
            token_sync.close()
            proxy.close()
            buffer.close()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for srt/speculative/decoupled_spec_io.

decoupled_spec_io is the schema-only IPC layer for decoupled speculative
decoding: protocol message dataclasses, the cross-process request id codec, and
token-span validation. These tests drive the request-id codec and wire-message
validation on CPU; there is no GPU or transport here.
"""

import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftReqKey,
    DraftTailStreamOutput,
    VerifyCommit,
    build_draft_scheduler_rid,
    parse_draft_scheduler_rid,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _commit(rid, *, pre, tokens, src_verifier_rank=0, drafter_rank=0) -> VerifyCommit:
    return VerifyCommit(
        request_id=rid,
        src_verifier_rank=src_verifier_rank,
        dst_drafter_rank=drafter_rank,
        pre_verify_committed_len=pre,
        committed_tokens=list(tokens),
    )


class TestDraftSchedulerRid(CustomTestCase):
    def test_round_trip(self):
        key = DraftReqKey(src_verifier_rank=3, request_id="req-1")
        rid = build_draft_scheduler_rid(key)
        self.assertEqual(rid, "draft:3:req-1")
        self.assertEqual(parse_draft_scheduler_rid(rid), key)

    def test_request_id_containing_colon_round_trips(self):
        # request_id may contain ':' — parse splits on the first ':' only.
        key = DraftReqKey(src_verifier_rank=1, request_id="a:b:c")
        self.assertEqual(parse_draft_scheduler_rid(build_draft_scheduler_rid(key)), key)

    def test_parse_invalid_rid_raises(self):
        for bad in ["no-prefix", "draft:", "draft:0:", "draft:notint:r"]:
            with self.subTest(rid=bad):
                with self.assertRaises(ValueError):
                    parse_draft_scheduler_rid(bad)


class TestVerifyCommitValidation(CustomTestCase):
    def test_empty_tokens_raises(self):
        with self.assertRaises(ValueError):
            _commit("r", pre=0, tokens=[]).validate_committed_tokens()

    def test_negative_pre_len_raises(self):
        with self.assertRaises(ValueError):
            _commit("r", pre=-1, tokens=[1]).validate_committed_tokens()

    def test_valid_commit_passes(self):
        # Should not raise.
        _commit("r", pre=0, tokens=[1, 2]).validate_committed_tokens()


class TestDraftTailStreamOutputValidation(CustomTestCase):
    @staticmethod
    def _output(
        *,
        base_committed_len=0,
        start_token_pos=0,
        tokens=(10,),
        is_commit_echo=False,
    ):
        return DraftTailStreamOutput(
            src_drafter_rank=0,
            dst_verifier_rank=0,
            request_id="r",
            base_committed_len=base_committed_len,
            start_token_pos=start_token_pos,
            tokens=tokens,
            is_commit_echo=is_commit_echo,
        )

    def test_contiguous_append_span_is_valid(self):
        self._output(tokens=(10, 11, 12)).validate()

    def test_append_span_must_be_nonempty_tuple(self):
        with self.assertRaises(ValueError):
            self._output(tokens=()).validate()
        with self.assertRaises(TypeError):
            self._output(tokens=[10]).validate()
        with self.assertRaises(ValueError):
            self._output(base_committed_len=2, start_token_pos=1).validate()

    def test_commit_echo_is_singleton_at_cumulative_ack_boundary(self):
        self._output(
            base_committed_len=3,
            start_token_pos=2,
            tokens=(12,),
            is_commit_echo=True,
        ).validate()
        with self.assertRaises(ValueError):
            self._output(
                base_committed_len=3,
                start_token_pos=2,
                tokens=(12, 13),
                is_commit_echo=True,
            ).validate()
        with self.assertRaises(ValueError):
            self._output(
                base_committed_len=2,
                start_token_pos=2,
                tokens=(12,),
                is_commit_echo=True,
            ).validate()


if __name__ == "__main__":
    unittest.main(verbosity=3)

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_decoupled_verify_mixin import (
    SchedulerDecoupledVerifyMixin,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    get_decoupled_spec_transport,
)
from sglang.srt.speculative.decoupled_verify_state import DecoupledVerifySnapshot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, *, extend: bool = False, decode: bool = False):
        self._extend = extend
        self._decode = decode

    def is_extend(self):
        return self._extend

    def is_decode(self):
        return self._decode

    def __str__(self):
        if self._extend:
            return "EXTEND"
        if self._decode:
            return "DECODE"
        return "OTHER"


class _Req:
    def __init__(
        self,
        rid,
        *,
        output_ids,
        origin_input_ids=None,
        finished=False,
        is_retracted=False,
    ):
        self.rid = rid
        self.output_ids = list(output_ids)
        self.origin_input_ids = list(origin_input_ids or [])
        self._finished = finished
        self.is_retracted = is_retracted
        self.decoupled_verify_snapshot = None

    def finished(self):
        return self._finished


class _Batch:
    def __init__(self, reqs, forward_mode):
        self.reqs = reqs
        self.forward_mode = forward_mode

    def is_dllm(self):
        return False


class _DraftTailBuffer:
    def __init__(self, active_rids=None, committed_lens=None):
        self.active_rids = set(active_rids or [])
        self.committed_lens = dict(committed_lens or {})

    def has_request(self, rid):
        return rid in self.active_rids

    def get_committed_len(self, rid):
        return self.committed_lens.get(rid)


class _DraftProxy:
    def __init__(self):
        self.rows = []

    def submit_control_rows(self, dst_drafter_rank, sync_rows, commit_rows, close_rows):
        self.rows.append(
            (
                dst_drafter_rank,
                list(sync_rows),
                list(commit_rows),
                list(close_rows),
            )
        )

    def submit_control_batch(self, _batch):
        raise AssertionError("C++ env path must use native rows")


class _VerifyScheduler(SchedulerDecoupledVerifyMixin):
    def __init__(self, draft_tail_buffer):
        self.draft_tail_buffer = draft_tail_buffer
        self.draft_proxy_thread = _DraftProxy()
        self.released = []

    def is_verify_entry_rank(self):
        return True

    def get_decoupled_spec_rank(self):
        return 5

    def assign_drafter_rank(self, _rid):
        return 2

    def get_drafter_rank(self, _rid):
        return 2

    def release_drafter_rank(self, rid):
        self.released.append(rid)


class TestDecoupledSpecCppDataPlane(unittest.TestCase):
    def test_transport_selection_reuses_backend_descriptors(self):
        with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(False):
            python_transport = get_decoupled_spec_transport()
            self.assertIs(python_transport, get_decoupled_spec_transport())
            self.assertFalse(python_transport.supports_native_rows)

        with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(True):
            cpp_transport = get_decoupled_spec_transport()
            self.assertIs(cpp_transport, get_decoupled_spec_transport())
            self.assertTrue(cpp_transport.supports_native_rows)

        self.assertIsNot(python_transport, cpp_transport)

    def test_sync_verify_requests_cpp_env_submits_native_rows(self):
        scheduler = _VerifyScheduler(_DraftTailBuffer())
        batch = _Batch(
            [
                _Req("req-a", origin_input_ids=[1, 2], output_ids=[10, 11]),
            ],
            _ForwardMode(extend=True),
        )

        with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(True):
            result = SchedulerDecoupledVerifyMixin._sync_verify_requests(
                scheduler, batch
            )

        self.assertIsNone(result)
        self.assertEqual(
            scheduler.draft_proxy_thread.rows,
            [
                (
                    2,
                    [("req-a", 5, 2, [1, 2], [10, 11])],
                    [],
                    [],
                )
            ],
        )

    def test_submit_verify_updates_cpp_env_submits_native_rows(self):
        scheduler = _VerifyScheduler(
            _DraftTailBuffer(
                active_rids={"req-commit", "req-close"},
                committed_lens={"req-commit": 2},
            )
        )
        req_commit = _Req("req-commit", output_ids=[10, 11, 12])
        verify_snapshot = DecoupledVerifySnapshot(
            pre_committed_len=2, draft_tokens=[99]
        )
        draft_tokens = verify_snapshot.draft_tokens
        req_commit.decoupled_verify_snapshot = verify_snapshot
        req_close = _Req("req-close", output_ids=[20], finished=True)
        batch = _Batch(
            [req_commit, req_close],
            _ForwardMode(decode=True),
        )

        with envs.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND.override(True):
            result = SchedulerDecoupledVerifyMixin.submit_verify_updates(
                scheduler, batch
            )

        self.assertIsNone(result)
        self.assertEqual(scheduler.released, ["req-close"])
        self.assertIs(req_commit.decoupled_verify_snapshot, verify_snapshot)
        self.assertIs(verify_snapshot.draft_tokens, draft_tokens)
        self.assertEqual(verify_snapshot.pre_committed_len, 3)
        self.assertEqual(verify_snapshot.draft_tokens, [])
        self.assertIsNone(req_close.decoupled_verify_snapshot)
        self.assertEqual(
            scheduler.draft_proxy_thread.rows,
            [
                (
                    2,
                    [],
                    [("req-commit", 5, 2, 2, [12])],
                    [("req-close", 5, 2, "finished")],
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()

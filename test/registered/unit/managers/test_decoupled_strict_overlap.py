"""Strict draft waiting drains prefill bootstrap without serializing decode."""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDecoupledStrictOverlap(CustomTestCase):
    def test_only_strict_prefill_to_decode_boundary_drains(self):
        scheduler = object.__new__(Scheduler)
        scheduler.require_mlp_sync = False
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.DECOUPLED_VERIFY,
            grammar_needs_sync=lambda: False,
        )
        for partial, previous, algorithm, expected in (
            (False, ForwardMode.EXTEND, SpeculativeAlgorithm.DECOUPLED_VERIFY, True),
            (False, ForwardMode.DECODE, SpeculativeAlgorithm.DECOUPLED_VERIFY, False),
            (True, ForwardMode.EXTEND, SpeculativeAlgorithm.DECOUPLED_VERIFY, False),
            (False, ForwardMode.EXTEND, SpeculativeAlgorithm.NONE, False),
            (False, None, SpeculativeAlgorithm.DECOUPLED_VERIFY, False),
        ):
            with self.subTest(partial=partial, previous=previous, algorithm=algorithm):
                batch.spec_algorithm = algorithm
                last_batch = (
                    None if previous is None else SimpleNamespace(forward_mode=previous)
                )
                scheduler.result_queue = (
                    [] if last_batch is None else [(last_batch, object())]
                )
                # The planner can alias/reuse last_batch for the next decode;
                # the queued forward snapshot still carries the prefill mode.
                last_batch = batch if last_batch is not None else None
                with envs.SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL.override(partial):
                    self.assertEqual(
                        bool(scheduler.is_disable_overlap_for_batch(batch, last_batch)),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()

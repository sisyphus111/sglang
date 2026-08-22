"""Output-stream ownership tests for internal decoupled drafter requests."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecoupledDraftOutputOwnership(CustomTestCase):
    def test_internal_draft_request_never_enters_generic_output_path(self):
        accumulator = _GenerationStreamAccumulator(
            return_logprob=False,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=1,
            get_cached_tokens_details=lambda req: None,
        )
        internal_req = SimpleNamespace(decoupled_draft_generation=object())

        accumulator.accept(req=internal_req)

        self.assertEqual(accumulator.rids, [])
        self.assertIsNone(accumulator.to_payload(dp_rank=0, is_idle_batch=False))

    def test_actual_proposed_drafts_are_carried_in_the_output_payload(self):
        accumulator = _GenerationStreamAccumulator(
            return_logprob=False,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.DECOUPLED_VERIFY,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=1,
            get_cached_tokens_details=lambda req: None,
        )
        accumulator.rids.append("req")
        accumulator.spec_verify_ct.append(3)
        accumulator.spec_num_proposed_drafts.append(5)
        accumulator.spec_num_correct_drafts.append(4)
        accumulator.spec_proposed_drafts_histogram.append([0, 1, 2])

        payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)

        self.assertEqual(payload.spec_verify_ct, [3])
        self.assertEqual(payload.spec_num_proposed_drafts, [5])
        self.assertEqual(payload.spec_num_correct_drafts, [4])
        self.assertEqual(payload.spec_proposed_drafts_histogram, [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()

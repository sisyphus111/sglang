"""CPU tests for request-level speculative metric denominators."""

from collections import deque
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.managers.scheduler_components.metrics_reporter import (  # noqa: E402
    SchedulerMetricsReporter,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _manager() -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(speculative_num_draft_tokens=4)
    return manager


def _output(*, proposed=..., correct: int = 1):
    values = dict(
        spec_verify_ct=[2],
        spec_num_correct_drafts=[correct],
        spec_proposed_drafts_histogram=None,
        completion_tokens=[4],
        spec_num_cap_tokens=None,
        spec_num_block_accept_tokens=None,
        spec_correct_drafts_histogram=[],
        spec_cap_lens_histogram=[],
    )
    if proposed is not ...:
        values["spec_num_proposed_drafts"] = [proposed]
    if proposed == 2 and correct == 1:
        values["spec_proposed_drafts_histogram"] = [[0, 2]]
        values["spec_correct_drafts_histogram"] = [[1, 1]]
    return SimpleNamespace(**values)


class TestSpeculativeResponseMetrics(CustomTestCase):
    def test_decode_metrics_window_uses_fixed_iteration_denominators(self):
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.decode_metrics_window_id = 4
        reporter.decode_metrics_windows = deque(maxlen=64)

        reporter._record_decode_metrics_window(
            elapsed_s=0.4,
            num_decode_iters=40,
            num_decode_rows=320,
            sum_context_lens=3_200_000,
            num_verify_rows=320,
            num_accept_tokens=640,
            num_proposed_drafts=480,
        )

        window = reporter.decode_metrics_windows[0]
        self.assertEqual(window.window_id, 5)
        self.assertEqual(window.num_decode_iters, 40)
        self.assertAlmostEqual(window.iter_latency_ms, 10.0)
        self.assertAlmostEqual(window.mean_batch_size, 8.0)
        self.assertAlmostEqual(window.mean_context_length, 10_000.0)
        self.assertAlmostEqual(window.accept_length, 2.0)
        self.assertAlmostEqual(window.proposed_draft_length, 1.5)

    def test_non_spec_decode_window_has_no_speculative_ratios(self):
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.decode_metrics_window_id = 0
        reporter.decode_metrics_windows = deque(maxlen=64)

        reporter._record_decode_metrics_window(
            elapsed_s=0.2,
            num_decode_iters=20,
            num_decode_rows=160,
            sum_context_lens=160_000,
            num_verify_rows=0,
            num_accept_tokens=0,
            num_proposed_drafts=0,
        )

        window = reporter.decode_metrics_windows[0]
        self.assertIsNone(window.accept_length)
        self.assertIsNone(window.proposed_draft_length)

    def test_service_reporter_keeps_actual_and_nominal_denominators(self):
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.spec_num_accept_tokens = 0
        reporter.spec_num_forward_ct = 0
        reporter.spec_proposed_drafts_ct = 0
        reporter.spec_nominal_drafts_ct = 0
        reporter.spec_num_block_accept_tokens = 0
        reporter.spec_num_cap_tokens = 0
        reporter.num_generated_tokens = 0

        reporter.update_spec_metrics(
            bs=2,
            num_correct_drafts=1,
            num_proposed_drafts=2,
            num_nominal_drafts=6,
        )

        self.assertEqual(reporter.spec_num_accept_tokens, 3)
        self.assertEqual(reporter.spec_num_forward_ct, 2)
        self.assertEqual(reporter.spec_proposed_drafts_ct, 2)
        self.assertEqual(reporter.spec_nominal_drafts_ct, 6)

    def test_decoupled_accept_rate_uses_actual_proposals(self):
        meta_info = {}

        _manager()._calculate_spec_decoding_metrics(meta_info, _output(proposed=2), 0)

        self.assertEqual(meta_info["spec_num_proposed_drafts"], 2)
        self.assertEqual(meta_info["spec_num_correct_drafts"], 1)
        self.assertEqual(meta_info["spec_accept_rate"], 0.5)
        self.assertEqual(meta_info["spec_accept_length"], 2.0)
        self.assertAlmostEqual(meta_info["spec_draft_occupancy_rate"], 1 / 3)
        self.assertEqual(meta_info["spec_proposed_draft_length"], 1.0)
        self.assertEqual(meta_info["spec_num_proposed_drafts_by_position"], [2, 0, 0])
        self.assertEqual(meta_info["spec_num_correct_drafts_by_position"], [1, 0, 0])
        self.assertEqual(meta_info["spec_accept_rate_by_position"], [0.5, None, None])
        self.assertEqual(meta_info["spec_proposed_drafts_histogram"], [0, 2])

    def test_other_spec_falls_back_to_fixed_k(self):
        meta_info = {}

        _manager()._calculate_spec_decoding_metrics(meta_info, _output(), 0)

        self.assertEqual(meta_info["spec_num_proposed_drafts"], 6)
        self.assertAlmostEqual(meta_info["spec_accept_rate"], 1 / 6)
        self.assertEqual(meta_info["spec_draft_occupancy_rate"], 1.0)

    def test_zero_actual_proposals_are_reported_without_fake_acceptance(self):
        meta_info = {}

        _manager()._calculate_spec_decoding_metrics(
            meta_info, _output(proposed=0, correct=0), 0
        )

        self.assertEqual(meta_info["spec_num_proposed_drafts"], 0)
        self.assertIsNone(meta_info["spec_accept_rate"])
        self.assertEqual(meta_info["spec_draft_occupancy_rate"], 0.0)

import torch

from sglang.srt.utils.nvtx_utils import profile_range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNvtxUtils(CustomTestCase):
    def test_profile_range_records_with_profile_all_threads(self):
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            experimental_config=torch.profiler._ExperimentalConfig(
                profile_all_threads=True
            ),
        )
        profiler.start()
        with profile_range("sglang.test.profile_all_threads"):
            torch.ones(1).add_(1)
        profiler.stop()

        event_names = {event.key for event in profiler.key_averages()}
        self.assertIn("sglang.test.profile_all_threads", event_names)


if __name__ == "__main__":
    import unittest

    unittest.main()

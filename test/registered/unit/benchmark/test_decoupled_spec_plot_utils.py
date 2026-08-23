"""CPU tests for decoupled-spec plot artifact loading."""

import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PLOT_ROOT = _REPO_ROOT / "benchmark" / "decoupled_spec" / "plot"
sys.path.insert(0, str(_PLOT_ROOT))

from plot_utils import load_csv  # noqa: E402


class TestDecoupledSpecPlotUtils(CustomTestCase):
    def test_load_csv_accepts_long_generated_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request_metrics.csv"
            generated_text = "x" * 200_000
            path.write_text(
                f"request_id,generated_text\nreq-0,{generated_text}\n",
                encoding="utf-8",
            )
            rows = load_csv(path)
            self.assertEqual(rows[0]["request_id"], "req-0")
            self.assertEqual(rows[0]["generated_text"], generated_text)


if __name__ == "__main__":
    unittest.main()

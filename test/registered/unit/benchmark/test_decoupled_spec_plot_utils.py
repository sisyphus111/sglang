"""CPU tests for fixed requests.csv plot loading."""

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
    def test_load_csv_accepts_long_json_array_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            array_text = "[" + ",".join("1" for _ in range(100_000)) + "]"
            path.write_text(
                "batch_row_index,spec_num_proposed_drafts_by_position\n"
                f'0,"{array_text}"\n',
                encoding="utf-8",
            )
            rows = load_csv(path)
            self.assertEqual(rows[0]["batch_row_index"], "0")
            self.assertEqual(
                rows[0]["spec_num_proposed_drafts_by_position"], array_text
            )


if __name__ == "__main__":
    unittest.main()

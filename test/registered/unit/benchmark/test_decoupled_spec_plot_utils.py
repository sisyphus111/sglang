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

from plot_utils import load_csv, select_formal_decode_windows  # noqa: E402


class TestDecoupledSpecPlotUtils(CustomTestCase):
    def test_formal_windows_exclude_one_boundary_per_engine_and_rank(self):
        rows = [
            dict(target_id=target, dp_rank=rank, window_id=wid, end_time=end)
            for target, rank in (("verifier-0", 0), ("verifier-0", 1), ("drafter-0", 0))
            for wid, end in ((3, 103.0), (0, 99.0), (2, 102.0), (1, 101.0))
        ]
        retained, excluded = select_formal_decode_windows(rows, 100.0, 102.5)
        self.assertEqual(len(retained), 3)
        self.assertEqual(len(excluded), 3)
        self.assertTrue(all(row["window_id"] == 2 for row in retained))
        self.assertTrue(all(row["window_id"] == 1 for row in excluded))
        self.assertEqual(len(rows), 12)

    def test_single_window_is_not_restored_for_short_request(self):
        row = dict(target_id="verifier-0", dp_rank=0, window_id=7, end_time=10.5)
        retained, excluded = select_formal_decode_windows([row], 10.0, 11.0)
        self.assertEqual(retained, [])
        self.assertEqual(excluded, [row])

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

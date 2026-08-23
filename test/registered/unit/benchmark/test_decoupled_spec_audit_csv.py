"""CPU tests for long-field decoupled-spec artifact auditing."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_AUDIT_PATH = (
    _REPO_ROOT
    / "benchmark"
    / "decoupled_spec"
    / "skills"
    / "audit-decoupled-spec-artifacts"
    / "scripts"
    / "audit_run.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "decoupled_spec_audit_csv_test", _AUDIT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


class TestDecoupledSpecAuditCsv(CustomTestCase):
    def test_read_csv_accepts_long_generated_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request_metrics.csv"
            generated_text = "x" * 200_000
            path.write_text(
                f"request_id,generated_text\nreq-0,{generated_text}\n",
                encoding="utf-8",
            )
            errors = []
            rows = _AUDIT._read_csv(path, errors)
            self.assertEqual(errors, [])
            self.assertEqual(rows[0]["request_id"], "req-0")
            self.assertEqual(rows[0]["generated_text"], generated_text)


if __name__ == "__main__":
    unittest.main()

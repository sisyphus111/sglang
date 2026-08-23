"""CPU tests for unified decoupled-spec manifest artifact contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4] / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_ROOT))

from common.collector import _summarize_decode_metric_windows


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_audit = _load_script(
    "decoupled_spec_unified_audit",
    _ROOT / "skills" / "audit-decoupled-spec-artifacts" / "scripts" / "audit_run.py",
)
_wait = _load_script(
    "decoupled_spec_unified_wait",
    _ROOT
    / "skills"
    / "operate-decoupled-spec-servers"
    / "scripts"
    / "wait_for_roles.py",
)


def _engines() -> list[dict]:
    values = []
    for role, count in (("verifier", 2), ("drafter", 1)):
        for rank in range(count):
            engine_id = f"{role}-{rank}"
            values.append(
                {
                    "engine_id": engine_id,
                    "role": role,
                    "rank": rank,
                    "http_url": f"http://127.0.0.1:{30000 + len(values)}",
                    "transport_endpoint": f"tcp://127.0.0.1:{31000 + len(values)}",
                    "resolved_config_path": (
                        f"server/engines/{engine_id}/resolved_config.json"
                    ),
                    "status_path": f"server/engines/{engine_id}/status.json",
                    "log_path": f"logs/server/{engine_id}.log",
                }
            )
    return values


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TestUnifiedArtifactIdentity(CustomTestCase):
    def _write_identity_artifacts(self, run_dir: Path, state: str) -> list[dict]:
        engines = _engines()
        _write_json(
            run_dir / "server" / "manifest.json",
            {
                "schema_version": 1,
                "state": state,
                "topology": {"num_verifiers": 2, "num_drafters": 1},
                "engines": engines,
            },
        )
        selected = engines[1]
        _write_json(
            run_dir / "client" / "resolved_config.json",
            {
                "server": {
                    "target_id": selected["engine_id"],
                    "role": selected["role"],
                    "rank": selected["rank"],
                    "base_url": selected["http_url"],
                }
            },
        )
        _write_json(
            run_dir / "observability" / "resolved_config.json",
            {
                "targets": {
                    engine["engine_id"]: {
                        "target_id": engine["engine_id"],
                        "role": engine["role"],
                        "rank": engine["rank"],
                        "base_url": engine["http_url"],
                    }
                    for engine in engines
                }
            },
        )
        return engines

    def test_audit_closes_manifest_client_collector_identity_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._write_identity_artifacts(run_dir, "stopped")
            errors = []
            report = _audit._check_unified_consumer_identity(run_dir, errors)
            self.assertEqual(errors, [])
            self.assertEqual(report["engine_count"], 3)

            client_path = run_dir / "client" / "resolved_config.json"
            client = json.loads(client_path.read_text(encoding="utf-8"))
            client["server"]["base_url"] = "http://wrong:1"
            _write_json(client_path, client)
            errors = []
            _audit._check_unified_consumer_identity(run_dir, errors)
            self.assertTrue(
                any("selected verifier identity disagrees" in error for error in errors)
            )

    def test_wait_gate_rejects_duplicate_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            engines = self._write_identity_artifacts(run_dir, "ready")
            manifest_path = run_dir / "server" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["engines"].append(dict(engines[0]))
            _write_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "duplicate.*engine identity"):
                _wait.wait_for_roles(
                    run_dir,
                    ["verifier", "drafter"],
                    timeout_s=1.0,
                    poll_interval_s=0.01,
                    check_http=False,
                    http_timeout_s=0.1,
                )

    def test_zero_window_summary_keeps_every_engine(self):
        targets = [
            {
                "target_id": engine["engine_id"],
                "role": engine["role"],
                "rank": engine["rank"],
                "base_url": engine["http_url"],
            }
            for engine in _engines()
        ]

        by_target, by_role = _summarize_decode_metric_windows({}, targets)

        self.assertEqual(set(by_target), {"verifier-0", "verifier-1", "drafter-0"})
        self.assertTrue(
            all(metrics["window_count"] == 0 for metrics in by_target.values())
        )
        self.assertEqual(
            by_role["verifier"]["target_ids"], ["verifier-0", "verifier-1"]
        )
        self.assertIsNone(by_role["drafter"]["scheduler_cycle_ms"]["mean"])

    def test_missing_legacy_logs_warn_but_unified_engine_logs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            errors = []
            warnings = []
            _audit._check_zero_waiting_queues(run_dir, errors, warnings)
            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 2)

            self._write_identity_artifacts(run_dir, "stopped")
            errors = []
            warnings = []
            _audit._check_zero_waiting_queues(run_dir, errors, warnings)
            self.assertEqual(len(errors), 3)
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""CPU tests for verifier profile configuration and output validation."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from profile_common import (
    build_engine_kwargs,
    detect_gpu_name,
    load_spec,
    validate_profile,
)
from profile_verifier import _run_parent

CONFIG = """
[paths]
repo = "{repo}"
target_model = "/models/target"
output_dir = "{output}"

[target]
gpus = ["0", "1", "2", "3"]
tp_size = 4
dp_size = 1
enable_dp_attention = false
dtype = "auto"
trust_remote_code = false
mem_fraction_static = 0.8

[profile]
batch_sizes = [1, 8]
ctx_lens = [1024, 4096]
steps = [0, 1]

[run]
timeout_s = 60
"""


class ProfileVerifierTest(unittest.TestCase):
    def make_spec(self, root: Path, replacement: tuple[str, str] | None = None):
        text = CONFIG.format(repo=root, output=root / "output")
        if replacement is not None:
            text = text.replace(*replacement)
        path = root / "config.toml"
        path.write_text(text)
        return load_spec(path)

    def write_valid_profile(self, spec) -> None:
        spec.output_dir.mkdir(exist_ok=True)
        spec.profile_path.write_text(
            json.dumps(
                {
                    "fingerprint": {
                        "target_model_path": spec.target_model,
                        "target_tp_size": spec.tp_size,
                        "target_dp_size": spec.dp_size,
                        "enable_dp_attention": spec.enable_dp_attention,
                        "gpu_name": "NVIDIA H20",
                    },
                    "costs": [
                        {
                            "batch_size": bs,
                            "steps": step,
                            "ctx_len": ctx,
                            "cost_ms": 1.0,
                        }
                        for step in spec.steps
                        for bs in spec.batch_sizes
                        for ctx in spec.ctx_lens
                    ],
                    "summary": {},
                }
            )
        )

    def test_loads_grid_and_builds_runtime_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
        kwargs = build_engine_kwargs(spec, "127.0.0.1:12345")
        self.assertEqual(spec.required_points.__len__(), 8)
        self.assertEqual(kwargs["speculative_algorithm"], "DECOUPLED_VERIFY")
        self.assertEqual(kwargs["speculative_num_steps"], 1)
        self.assertEqual(kwargs["cuda_graph_bs_decode"], [1, 8])
        self.assertEqual(
            kwargs["decoupled_verify_throughput_profile_ctx_lens"], "1024,4096"
        )
        self.assertEqual(kwargs["tp_size"], 4)
        self.assertNotIn("enable_deterministic_inference", kwargs)

    def test_check_cli_requires_no_gpu_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("profile_verifier.py")),
                    "--config",
                    str(spec.config_path),
                    "--check",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["required_points"], 8)
        self.assertEqual(
            payload["engine_kwargs"]["speculative_algorithm"], "DECOUPLED_VERIFY"
        )

    def test_rejects_sparse_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "contiguous"):
                self.make_spec(Path(directory), ("steps = [0, 1]", "steps = [0, 2]"))

    def test_rejects_deterministic_target_option(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                self.make_spec(
                    Path(directory),
                    (
                        "trust_remote_code = false",
                        "trust_remote_code = false\ndeterministic = true",
                    ),
                )

    def test_rejects_duplicate_grid_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                self.make_spec(
                    Path(directory),
                    ("batch_sizes = [1, 8]", "batch_sizes = [1, 1]"),
                )

    def test_rejects_boolean_integer_fields(self) -> None:
        replacements = (
            ("batch_sizes = [1, 8]", "batch_sizes = [true, 8]"),
            ("steps = [0, 1]", "steps = [0, true]"),
            ("tp_size = 4", "tp_size = true"),
        )
        for replacement in replacements:
            with self.subTest(
                replacement=replacement
            ), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "integer"):
                    self.make_spec(Path(directory), replacement)

    def test_rejects_gpu_topology_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected 2"):
                self.make_spec(Path(directory), ("tp_size = 4", "tp_size = 2"))

    def test_dp_attention_uses_tp_times_dp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = CONFIG.format(repo=root, output=root / "output")
            text = text.replace("tp_size = 4", "tp_size = 2")
            text = text.replace("dp_size = 1", "dp_size = 2")
            text = text.replace(
                "enable_dp_attention = false", "enable_dp_attention = true"
            )
            path = root / "config.toml"
            path.write_text(text)
            spec = load_spec(path)
        self.assertEqual(spec.engine_tp_size, 4)
        self.assertEqual(build_engine_kwargs(spec, "x")["tp_size"], 4)

    def test_validates_exact_profile_grid_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            spec.output_dir.mkdir()
            rows = [
                {
                    "batch_size": bs,
                    "steps": step,
                    "ctx_len": ctx,
                    "cost_ms": float(bs + step + ctx / 1000),
                }
                for step in spec.steps
                for bs in spec.batch_sizes
                for ctx in spec.ctx_lens
            ]
            spec.profile_path.write_text(
                json.dumps(
                    {
                        "fingerprint": {
                            "target_model_path": spec.target_model,
                            "target_tp_size": spec.tp_size,
                            "target_dp_size": spec.dp_size,
                            "enable_dp_attention": spec.enable_dp_attention,
                            "gpu_name": "NVIDIA H20",
                        },
                        "costs": rows,
                        "summary": {},
                    }
                )
            )
            result = validate_profile(spec, expected_gpu_name="NVIDIA H20")
        self.assertEqual(result["actual_points"], 8)
        self.assertEqual(result["duplicate_points"], 0)

    def test_rejects_missing_extra_duplicate_and_nonpositive_costs(self) -> None:
        mutations = {
            "missing": lambda rows: rows.pop(),
            "extra": lambda rows: rows.append(
                {"batch_size": 16, "steps": 0, "ctx_len": 1024, "cost_ms": 1.0}
            ),
            "duplicate": lambda rows: rows.append(dict(rows[0])),
            "nonpositive": lambda rows: rows[0].update(cost_ms=0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                spec = self.make_spec(Path(directory))
                spec.output_dir.mkdir()
                rows = [
                    {
                        "batch_size": bs,
                        "steps": step,
                        "ctx_len": ctx,
                        "cost_ms": 1.0,
                    }
                    for step in spec.steps
                    for bs in spec.batch_sizes
                    for ctx in spec.ctx_lens
                ]
                mutate(rows)
                spec.profile_path.write_text(
                    json.dumps(
                        {
                            "fingerprint": {
                                "target_model_path": spec.target_model,
                                "target_tp_size": 4,
                                "target_dp_size": 1,
                                "enable_dp_attention": False,
                            },
                            "costs": rows,
                        }
                    )
                )
                with self.assertRaises(ValueError):
                    validate_profile(spec)

    def test_detects_one_homogeneous_gpu_name(self) -> None:
        with mock.patch(
            "profile_common.subprocess.check_output",
            return_value="NVIDIA H20\nNVIDIA H20\n",
        ):
            self.assertEqual(detect_gpu_name(("0", "1")), "NVIDIA H20")
        with mock.patch(
            "profile_common.subprocess.check_output",
            return_value="NVIDIA H20\nNVIDIA A100\n",
        ):
            with self.assertRaisesRegex(ValueError, "homogeneous"):
                detect_gpu_name(("0", "1"))

    def test_interrupt_stops_worker_group_and_records_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            process = mock.Mock()
            process.wait.side_effect = KeyboardInterrupt
            with mock.patch(
                "profile_verifier.detect_gpu_name", return_value="NVIDIA H20"
            ), mock.patch(
                "profile_verifier._git_provenance", return_value={"commit": "abc"}
            ), mock.patch(
                "profile_verifier.subprocess.Popen", return_value=process
            ), mock.patch(
                "profile_verifier._stop_process_group"
            ) as stop:
                with self.assertRaises(KeyboardInterrupt):
                    _run_parent(spec, force=False)
            manifest = json.loads((spec.output_dir / "manifest.json").read_text())
        stop.assert_called_once_with(process)
        self.assertEqual(manifest["status"], "interrupted")

    def test_zero_exit_without_profile_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            process = mock.Mock()
            process.wait.return_value = 0
            with mock.patch(
                "profile_verifier.detect_gpu_name", return_value="NVIDIA H20"
            ), mock.patch(
                "profile_verifier._git_provenance", return_value={"commit": "abc"}
            ), mock.patch(
                "profile_verifier.subprocess.Popen", return_value=process
            ):
                with self.assertRaises(FileNotFoundError):
                    _run_parent(spec, force=False)
            manifest = json.loads((spec.output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "invalid_profile")

    def test_success_records_validated_profile_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            process = mock.Mock()

            def finish(timeout):
                self.write_valid_profile(spec)
                return 0

            process.wait.side_effect = finish
            with mock.patch(
                "profile_verifier.detect_gpu_name", return_value="NVIDIA H20"
            ), mock.patch(
                "profile_verifier._git_provenance", return_value={"commit": "abc"}
            ), mock.patch(
                "profile_verifier.subprocess.Popen", return_value=process
            ):
                with redirect_stdout(io.StringIO()):
                    _run_parent(spec, force=False)
            manifest = json.loads((spec.output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["validation"]["actual_points"], 8)
        self.assertEqual(len(manifest["profile_sha256"]), 64)

    def test_config_lock_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            spec.output_dir.mkdir()
            (spec.output_dir / "profile-config.lock.toml").write_text("old config")
            with mock.patch(
                "profile_verifier.detect_gpu_name", return_value="NVIDIA H20"
            ):
                with self.assertRaisesRegex(ValueError, "immutable lock"):
                    _run_parent(spec, force=False)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused tests for the reproducible decoupled-spec matrix runner."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from profile_cache import ProfileTable
from run_matrix import (
    base_command,
    build_env,
    main,
    load_config,
    profile_complete,
    run_process,
    run_matrix as execute_matrix,
)


class ProfileCoverageTest(unittest.TestCase):
    def test_deterministic_workload_option_is_rejected(self) -> None:
        template = (
            Path(__file__).resolve().parents[1]
            / "references"
            / "qwen35-27b-08b-default.toml"
        ).read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                template.replace(
                    "sampling_seed = 42",
                    "sampling_seed = 42\ndeterministic = true",
                )
            )
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                load_config(path)

    def test_boolean_sampling_seed_is_rejected(self) -> None:
        template = (
            Path(__file__).resolve().parents[1]
            / "references"
            / "qwen35-27b-08b-default.toml"
        ).read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                template.replace("sampling_seed = 42", "sampling_seed = true")
            )
            with self.assertRaisesRegex(ValueError, "sampling_seed"):
                load_config(path)

    def test_exact_matrix_coverage_is_required(self) -> None:
        table = ProfileTable(
            [
                {"batch_size": 64, "steps": 1, "ctx_len": 1024, "cost_ms": 10},
                {"batch_size": 64, "steps": 2, "ctx_len": 1024, "cost_ms": 12},
            ]
        )
        self.assertEqual(table.missing_points([64], [1, 2], [1024]), [])
        self.assertEqual(
            table.missing_points([64], [1, 2], [1024, 2048]),
            [(64, 1, 2048), (64, 2, 2048)],
        )

    def test_profile_complete_reports_missing_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "fingerprint": {
                            "target_model_path": "/models/target",
                            "target_tp_size": 4,
                            "target_dp_size": 1,
                            "enable_dp_attention": False,
                            "gpu_name": "NVIDIA H20",
                        },
                        "costs": [
                            {
                                "batch_size": 64,
                                "steps": 1,
                                "ctx_len": 1024,
                                "cost_ms": 10,
                            }
                        ],
                    }
                )
            )
            config = {
                "paths": {"profile": str(path), "target_model": "/models/target"},
                "parallelism": {"target_tp_size": 4},
                "profile": {
                    "capture_bs": [64],
                    "steps": [1, 2],
                    "ctx_lens": [1024],
                    "expected_gpu_name": "NVIDIA H20",
                },
            }
            complete, reason = profile_complete(config)
        self.assertFalse(complete)
        self.assertIn("(64, 2, 1024)", reason)

    def test_existing_run_rejects_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.toml"
            source.write_text("new config")
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "config.toml").write_text("old config")
            args = SimpleNamespace(
                config=source,
                run_dir=run_dir,
                action="run",
                force=False,
                timeout_s=1.0,
            )
            with mock.patch("run_matrix.parse_args", return_value=args), mock.patch(
                "run_matrix.load_config", return_value={"paths": {"repo": str(root)}}
            ), mock.patch(
                "run_matrix.subprocess.check_output", side_effect=["deadbeef\n", ""]
            ):
                with self.assertRaisesRegex(ValueError, "run config differs"):
                    main()

    def test_invocation_records_checkout_and_config_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.toml"
            source.write_text("stable config")
            run_dir = root / "run"
            args = SimpleNamespace(
                config=source,
                run_dir=run_dir,
                action="run",
                force=False,
                timeout_s=1.0,
            )
            config = {
                "paths": {
                    "repo": str(root),
                    "profile": str(root / "missing-profile.json"),
                }
            }
            with mock.patch("run_matrix.parse_args", return_value=args), mock.patch(
                "run_matrix.load_config", return_value=config
            ), mock.patch(
                "run_matrix.subprocess.check_output",
                side_effect=["deadbeef\n", ""],
            ), mock.patch(
                "run_matrix.run_matrix"
            ):
                main()
            row = json.loads((run_dir / "status.jsonl").read_text())
        self.assertEqual(row["repo_commit"], "deadbeef")
        self.assertFalse(row["repo_dirty"])
        self.assertEqual(len(row["config_sha256"]), 64)

    def test_run_dir_config_is_frozen_by_separate_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            config_path = run_dir / "config.toml"
            config_path.write_text("first config")
            args = SimpleNamespace(
                config=config_path,
                run_dir=run_dir,
                action="run",
                force=False,
                timeout_s=1.0,
            )
            config = {
                "paths": {
                    "repo": str(root),
                    "profile": str(root / "missing-profile.json"),
                }
            }
            with mock.patch("run_matrix.parse_args", return_value=args), mock.patch(
                "run_matrix.load_config", return_value=config
            ), mock.patch(
                "run_matrix.subprocess.check_output", side_effect=["deadbeef\n", ""]
            ), mock.patch(
                "run_matrix.run_matrix"
            ):
                main()
            config_path.write_text("changed config")
            with mock.patch("run_matrix.parse_args", return_value=args), mock.patch(
                "run_matrix.load_config", return_value=config
            ), mock.patch(
                "run_matrix.subprocess.check_output", side_effect=["deadbeef\n", ""]
            ):
                with self.assertRaisesRegex(ValueError, "immutable copy"):
                    main()

    def test_force_profile_accepts_invalid_or_legacy_cache_for_rebuild(self) -> None:
        for payload in ("{invalid", "[]"):
            with self.subTest(
                payload=payload
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.toml"
                source.write_text("stable config")
                profile = root / "profile.json"
                profile.write_text(payload)
                args = SimpleNamespace(
                    config=source,
                    run_dir=root / "run",
                    action="profile",
                    force=True,
                    timeout_s=1.0,
                    poll_s=0.1,
                )
                config = {"paths": {"repo": str(root), "profile": str(profile)}}
                with mock.patch("run_matrix.parse_args", return_value=args), mock.patch(
                    "run_matrix.load_config", return_value=config
                ), mock.patch(
                    "run_matrix.subprocess.check_output",
                    side_effect=["deadbeef\n", ""],
                ), mock.patch(
                    "run_matrix.generate_profile"
                ) as generate:
                    main()
                generate.assert_called_once()

    def test_dirty_checkout_fails_before_creating_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.toml"
            source.write_text("stable config")
            run_dir = root / "run"
            args = SimpleNamespace(config=source, run_dir=run_dir, action="run")
            config = {"paths": {"repo": str(root)}}
            with mock.patch("run_matrix.parse_args", return_value=args), mock.patch(
                "run_matrix.load_config", return_value=config
            ), mock.patch(
                "run_matrix.subprocess.check_output",
                side_effect=["deadbeef\n", " M changed.py\n"],
            ):
                with self.assertRaisesRegex(RuntimeError, "dirty"):
                    main()
            self.assertFalse(run_dir.exists())


class MatrixCommandTest(unittest.TestCase):
    @staticmethod
    def _config(strategy: str) -> dict:
        return {
            "paths": {
                "repo": "/repo",
                "entrypoint": "single-node.py",
                "dataset": "/data/input.parquet",
                "target_model": "/models/target",
                "draft_model": "/models/draft",
                "profile": "/profiles/cost.json",
            },
            "parallelism": {
                "target_tp_size": 4,
                "draft_tp_size": 1,
                "n_gpu_per_node": 5,
                "verify_ngpus": 4,
                "draft_ngpus": 1,
            },
            "workload": {
                "dataset_format": "dapo_math_17k",
                "batch_size": 64,
                "max_running_requests": 65,
                "max_new_tokens": 32768,
                "temperature": 1.0,
                "sampling_seed": 42,
            },
            "profile": {
                "capture_bs": [64],
                "ctx_lens": [32768],
                "steps": [0, 1, 2, 3],
            },
            "adaptive": {"strategy": strategy, "config": "/configs/ema.json"},
        }

    def test_ema_dynamic_command_uses_accept_length_config(self) -> None:
        command = base_command(self._config("ema"), Path("/output"), 3, True)
        self.assertIn("ema", command)
        self.assertIn("--speculative-adaptive-config", command)
        self.assertNotIn("--decoupled-verify-throughput-profile-path", command)

    def test_profile_command_can_force_throughput_aware(self) -> None:
        command = base_command(
            self._config("ema"),
            Path("/output"),
            3,
            True,
            adaptive_strategy="throughput_aware",
        )
        self.assertIn("throughput_aware", command)
        self.assertIn("--decoupled-verify-throughput-profile-path", command)
        self.assertNotIn("--speculative-adaptive-config", command)

    def test_mem_fraction_is_identical_for_profile_and_formal_cases(self) -> None:
        config = self._config("ema")
        config["profile"]["mem_fraction_static"] = 0.74
        command = base_command(config, Path("/output"), 2, False)
        index = command.index("--mem-fraction-static")
        self.assertEqual(command[index + 1], "0.74")

    def test_force_failure_removes_stale_summary_and_manifest_survives(self) -> None:
        config = self._config("ema")
        config["matrix"] = {
            "allow_partial": [True],
            "static_steps": [1],
            "dynamic_max_steps": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = run_dir / "runs" / "ap1_static_dl1" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text("stale")
            with mock.patch("run_matrix.run_process", return_value=1):
                execute_matrix(config, run_dir, force=True, timeout_s=1.0)
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertFalse(summary.exists())
        self.assertEqual(manifest["cases"][0]["label"], "ap1_static_dl1")

    def test_zero_exit_without_summary_is_not_success(self) -> None:
        config = self._config("ema")
        config["matrix"] = {
            "allow_partial": [True],
            "static_steps": [1],
            "dynamic_max_steps": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with mock.patch("run_matrix.run_process", return_value=0):
                execute_matrix(config, run_dir, force=False, timeout_s=1.0)
            rows = [
                json.loads(line)
                for line in (run_dir / "status.jsonl").read_text().splitlines()
            ]
        self.assertEqual(rows[-1]["status"], "missing_summary")

    def test_keyboard_interrupt_stops_process_group(self) -> None:
        config = self._config("ema")
        process = mock.Mock()
        process.wait.side_effect = KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "run_matrix.subprocess.Popen", return_value=process
        ), mock.patch("run_matrix.stop_process_group") as stop:
            with self.assertRaises(KeyboardInterrupt):
                run_process(
                    config=config,
                    command=["python", "probe.py"],
                    log_path=Path(directory) / "probe.log",
                    allow_partial=True,
                    timeout_s=1.0,
                )
        stop.assert_called_once_with(process)

    def test_formal_commands_lock_sampling_seed_without_determinism(self) -> None:
        for steps, dynamic in ((0, False), (3, False), (3, True)):
            with self.subTest(steps=steps, dynamic=dynamic):
                command = base_command(
                    self._config("ema"), Path("/output"), steps, dynamic
                )
                seed_index = command.index("--sampling-seed")
                self.assertEqual(command[seed_index + 1], "42")
                self.assertNotIn("--deterministic", command)

    def test_child_environment_prefers_configured_worktree(self) -> None:
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/ambient/python"}):
            env = build_env(self._config("ema"), allow_partial=True)
        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep), ["/repo/python", "/ambient/python"]
        )


if __name__ == "__main__":
    unittest.main()

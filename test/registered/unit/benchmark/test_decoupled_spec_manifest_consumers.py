"""CPU tests for decoupled-spec server-manifest consumers."""

import argparse
import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from aiohttp import web
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4] / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "client-side"))
sys.path.insert(0, str(_ROOT / "plot"))

from client import add_cli_args as add_client_cli_args
from client import apply_cli_overrides as apply_client_cli_overrides
from common.collector import add_cli_args as add_collector_cli_args
from common.collector import apply_cli_overrides as apply_collector_cli_overrides
from common.collector import collect
from common.collector import validate_config as validate_collector_config
from plot_observability import render_observability


def _write_manifest(path: Path, base_url: str, state: str = "ready") -> None:
    engines = []
    for role, count in (("verifier", 2), ("drafter", 1)):
        for rank in range(count):
            target_id = f"{role}-{rank}"
            engines.append(
                {
                    "engine_id": target_id,
                    "role": role,
                    "rank": rank,
                    "node_id": "mock-node",
                    "node_ip": "127.0.0.1",
                    "http_url": f"{base_url}/{target_id}",
                    "transport_endpoint": f"tcp://127.0.0.1:{32000 + len(engines)}",
                    "tp_size": 1,
                    "gpu_ids": [str(len(engines))],
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": state,
                "engines": engines,
                "topology": {"quota_edges": []},
            }
        ),
        encoding="utf-8",
    )


class TestDecoupledSpecManifestConsumers(CustomTestCase):
    def test_client_selects_one_ready_verifier_and_rejects_url_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            _write_manifest(manifest_path, "http://127.0.0.1:31000")
            parser = argparse.ArgumentParser()
            add_client_cli_args(parser)
            config = {"server": {"base_url": "http://stale:30000"}}
            apply_client_cli_overrides(
                config,
                parser.parse_args(
                    [
                        "--server-manifest",
                        str(manifest_path),
                        "--verifier-rank",
                        "1",
                    ]
                ),
            )

            self.assertEqual(config["server"]["target_id"], "verifier-1")
            self.assertEqual(config["server"]["rank"], 1)
            self.assertEqual(
                config["server"]["base_url"],
                "http://127.0.0.1:31000/verifier-1",
            )
            self.assertEqual(
                config["server_manifest"]["path"], str(manifest_path.resolve())
            )

            with self.assertRaisesRegex(ValueError, "conflicts with --base-url"):
                apply_client_cli_overrides(
                    {},
                    parser.parse_args(
                        [
                            "--server-manifest",
                            str(manifest_path),
                            "--base-url",
                            "http://explicit:30000",
                        ]
                    ),
                )

            with self.assertRaisesRegex(ValueError, "--verifier-rank is required"):
                apply_client_cli_overrides(
                    {},
                    parser.parse_args(["--server-manifest", str(manifest_path)]),
                )

            _write_manifest(manifest_path, "http://127.0.0.1:31000", state="starting")
            with self.assertRaisesRegex(ValueError, "not ready"):
                apply_client_cli_overrides(
                    {},
                    parser.parse_args(["--server-manifest", str(manifest_path)]),
                )

    def test_collector_and_plot_keep_same_role_targets_separate(self):
        async def exercise():
            window_end_times = {}
            cycle_ms = {"verifier-0": 10.0, "verifier-1": 12.0, "drafter-0": 7.0}

            async def metadata(_request):
                return web.json_response({"model_path": "mock"})

            async def loads(request):
                target_id = request.match_info["target_id"]
                is_verifier = target_id.startswith("verifier")
                end_time = window_end_times.setdefault(target_id, time.time())
                return web.json_response(
                    {
                        "loads": [
                            {
                                "dp_rank": 0,
                                "num_running_reqs": 8,
                                "num_waiting_reqs": 0,
                                "gen_throughput": 100.0,
                                "token_usage": 0.2,
                                "speculative": (
                                    {
                                        "accept_length": 2.0,
                                        "accept_rate": 0.8,
                                        "proposed_draft_length": 1.5,
                                        "draft_occupancy_rate": 0.5,
                                    }
                                    if is_verifier
                                    else None
                                ),
                                "decode_metrics_windows": [
                                    {
                                        "window_id": 9,
                                        "end_time": end_time,
                                        "num_decode_iters": 40,
                                        "iter_latency_ms": cycle_ms[target_id],
                                        "num_decode_rows": 320,
                                        "sum_context_lens": 3_200_000,
                                        "mean_batch_size": 8.0,
                                        "mean_context_length": 10_000.0,
                                        "num_verify_rows": 320 if is_verifier else 0,
                                        "num_accept_tokens": (
                                            640 if is_verifier else 0
                                        ),
                                        "num_proposed_drafts": (
                                            480 if is_verifier else 0
                                        ),
                                        "accept_length": (2.0 if is_verifier else None),
                                        "proposed_draft_length": (
                                            1.5 if is_verifier else None
                                        ),
                                    }
                                ],
                            }
                        ]
                    }
                )

            app = web.Application()
            app.router.add_get("/{target_id}/model_info", metadata)
            app.router.add_get("/{target_id}/server_info", metadata)
            app.router.add_get("/{target_id}/v1/loads", loads)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = Path(directory)
                    manifest_path = run_dir / "server" / "manifest.json"
                    manifest_path.parent.mkdir()
                    _write_manifest(manifest_path, f"http://127.0.0.1:{port}")
                    parser = argparse.ArgumentParser()
                    add_collector_cli_args(parser)
                    config = {
                        "schema_version": 1,
                        "interval_s": 0.01,
                        "request_timeout_s": 0.5,
                        "targets": {
                            "verifier": {"base_url": "http://stale:30000"},
                            "drafter": {"base_url": "http://stale:30001"},
                        },
                    }
                    apply_collector_cli_overrides(
                        config,
                        parser.parse_args(["--server-manifest", str(manifest_path)]),
                    )
                    validate_collector_config(config)
                    summary = await collect(config, run_dir, duration_s=0.025)
                    records = [
                        json.loads(line)
                        for line in (run_dir / "observability" / "samples.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
                    plot_manifest = render_observability(run_dir)
            finally:
                await runner.cleanup()

            self.assertEqual(summary["target_ct"], 3)
            self.assertEqual(
                set(summary["decode_metrics_by_target"]),
                {"verifier-0", "verifier-1", "drafter-0"},
            )
            self.assertEqual(summary["decode_metrics"]["verifier"]["window_count"], 2)
            self.assertEqual(
                summary["decode_metrics"]["verifier"]["scheduler_cycle_ms"]["mean"],
                11.0,
            )
            self.assertEqual(
                summary["decode_metrics_by_target"]["verifier-1"]["scheduler_cycle_ms"][
                    "mean"
                ],
                12.0,
            )
            self.assertEqual(
                {record["target_id"] for record in records},
                {"verifier-0", "verifier-1", "drafter-0"},
            )
            for record in records:
                self.assertIn(record["role"], {"verifier", "drafter"})
                self.assertIsInstance(record["rank"], int)
                self.assertTrue(record["base_url"].startswith("http://127.0.0.1:"))
            self.assertEqual(
                plot_manifest["decode_metrics_window_ct_by_target"],
                {"drafter-0": 1, "verifier-0": 1, "verifier-1": 1},
            )
            self.assertEqual(
                plot_manifest["target_ct_by_role"], {"drafter": 1, "verifier": 2}
            )
            self.assertEqual(plot_manifest["target_ct"], 3)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()

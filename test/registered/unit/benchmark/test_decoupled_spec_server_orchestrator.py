"""CPU tests for the unified decoupled-spec Ray server launcher."""

from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_BENCHMARK_ROOT = _REPOSITORY_ROOT / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_BENCHMARK_ROOT))

_orchestrator = importlib.import_module("server-side.orchestrator")
EngineActor = _orchestrator.EngineActor
RayServerOrchestrator = _orchestrator.RayServerOrchestrator
build_peer_configs = _orchestrator.build_peer_configs
load_config = _orchestrator.load_config
plan_decoupled_spec_quota_edges = _orchestrator.plan_decoupled_spec_quota_edges


def _minimal_config() -> str:
    return """
schema_version: 1
ray:
  address: auto
verifier:
  replicas: 2
  runtime:
    env:
      SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"
  server_args:
    model_path: mock-target
    tp_size: 2
    speculative_algorithm: DECOUPLED_VERIFY
    speculative_num_steps: 3
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 4
drafter:
  replicas: 3
  runtime:
    env:
      SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"
  server_args:
    model_path: mock-draft
    tp_size: 1
    page_size: 1
    disable_overlap_schedule: true
    speculative_algorithm: null
    speculative_num_steps: 3
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 4
"""


class TestUnifiedServerConfig(CustomTestCase):
    def test_ray_actor_uses_distributable_package_identity(self):
        self.assertEqual(EngineActor.__module__, "server-side.orchestrator")
        self.assertTrue(
            Path(_orchestrator.__file__).with_name("role_server.py").is_file()
        )

    def test_check_config_has_no_static_topology_or_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.yaml"
            path.write_text(_minimal_config(), encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.verifier.replicas, 2)
        self.assertEqual(config.drafter.replicas, 3)
        for role in (config.verifier, config.drafter):
            self.assertFalse(
                {
                    "host",
                    "port",
                    "nccl_port",
                    "base_gpu_id",
                    "gpu_id_step",
                    "use_ray",
                    "decoupled_spec_rank",
                    "decoupled_spec_bind_endpoint",
                    "decoupled_spec_peer_configs",
                }
                & set(role.server_args)
            )

    def test_rejects_launcher_owned_server_args(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.yaml"
            path.write_text(
                _minimal_config().replace(
                    "    tp_size: 2\n", "    tp_size: 2\n    port: 30000\n"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "launcher-owned fields"):
                load_config(path)


class TestQuotaTopology(CustomTestCase):
    def test_quota_graph_balances_arbitrary_replica_ratios(self):
        for num_verifiers in range(1, 7):
            for num_drafters in range(1, 7):
                with self.subTest(
                    num_verifiers=num_verifiers, num_drafters=num_drafters
                ):
                    edges = plan_decoupled_spec_quota_edges(num_verifiers, num_drafters)
                    for rank in range(num_verifiers):
                        self.assertEqual(
                            sum(
                                edge["quota"]
                                for edge in edges
                                if edge["verifier_rank"] == rank
                            ),
                            num_drafters,
                        )
                    for rank in range(num_drafters):
                        self.assertEqual(
                            sum(
                                edge["quota"]
                                for edge in edges
                                if edge["drafter_rank"] == rank
                            ),
                            num_verifiers,
                        )

    def test_v2_d3_uses_sparse_smooth_weighted_edges(self):
        edges = plan_decoupled_spec_quota_edges(2, 3)
        self.assertEqual(
            edges,
            [
                {"verifier_rank": 0, "drafter_rank": 0, "quota": 2},
                {"verifier_rank": 0, "drafter_rank": 1, "quota": 1},
                {"verifier_rank": 1, "drafter_rank": 1, "quota": 1},
                {"verifier_rank": 1, "drafter_rank": 2, "quota": 2},
            ],
        )

        engines = [
            {
                "engine_id": f"verifier-{rank}",
                "role": "verifier",
                "rank": rank,
                "transport_endpoint": f"tcp://10.0.0.1:{31000 + rank}",
            }
            for rank in range(2)
        ] + [
            {
                "engine_id": f"drafter-{rank}",
                "role": "drafter",
                "rank": rank,
                "transport_endpoint": f"tcp://10.0.0.2:{32000 + rank}",
            }
            for rank in range(3)
        ]
        peers = build_peer_configs(engines, edges)

        self.assertEqual(
            peers["verifier-0"],
            [
                {"rank": 0, "endpoint": "tcp://10.0.0.2:32000", "quota": 2},
                {"rank": 1, "endpoint": "tcp://10.0.0.2:32001", "quota": 1},
            ],
        )
        self.assertEqual(
            peers["drafter-1"],
            [
                {"rank": 0, "endpoint": "tcp://10.0.0.1:31000", "quota": 1},
                {"rank": 1, "endpoint": "tcp://10.0.0.1:31001", "quota": 1},
            ],
        )


class TestNodeLocalPortLeases(CustomTestCase):
    def test_actor_normalizes_bracketed_ipv6_from_ray(self):
        actor = EngineActor(
            engine_id="drafter-0",
            role="drafter",
            rank=0,
            tp_size=1,
            runtime_env={},
            node_info_override={"node_id": "local", "node_ip": "[::1]"},
            gpu_ids_override=["0"],
        )
        self.assertEqual(actor._runtime_identity()[1], "::1")
        actor.stop(0.1)

    def test_actor_holds_three_real_sockets_and_cleanup_is_idempotent(self):
        actor = EngineActor(
            engine_id="verifier-0",
            role="verifier",
            rank=0,
            tp_size=1,
            runtime_env={},
            node_info_override={"node_id": "local", "node_ip": "127.0.0.1"},
            gpu_ids_override=["0"],
        )
        info = actor.reserve_ports()
        ports = {
            int(info["http_url"].rsplit(":", 1)[1]),
            int(info["transport_endpoint"].rsplit(":", 1)[1]),
            int(info["nccl_port"]),
        }
        self.assertEqual(len(ports), 3)
        for port in ports:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with probe:
                with self.assertRaises(OSError):
                    probe.bind(("127.0.0.1", port))

        first = actor.stop(0.1)
        second = actor.stop(0.1)
        self.assertEqual(first["return_code"], None)
        self.assertEqual(second["return_code"], None)
        for port in ports:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with probe:
                probe.bind(("127.0.0.1", port))

    def test_actor_stop_cleans_grandchildren_after_leader_exits(self):
        actor = EngineActor(
            engine_id="verifier-0",
            role="verifier",
            rank=0,
            tp_size=1,
            runtime_env={},
            node_info_override={"node_id": "local", "node_ip": "127.0.0.1"},
            gpu_ids_override=["0"],
        )
        child_pid_path = actor.local_dir / "child.pid"
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import pathlib,subprocess,sys,time; "
                    "child=subprocess.Popen([sys.executable,'-c',"
                    "'import signal,time;signal.signal(signal.SIGTERM,"
                    "signal.SIG_IGN);time.sleep(60)']);"
                    f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid));"
                    "time.sleep(60)"
                ),
            ],
            start_new_session=True,
        )
        actor._process = leader
        deadline = time.monotonic() + 5.0
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(child_pid_path.exists())
        process_group_id = leader.pid

        result = actor.stop(0.1)

        self.assertTrue(result["sigkill_escalated"])
        self.assertFalse(result["process_group_alive"])
        with self.assertRaises(ProcessLookupError):
            os.killpg(process_group_id, 0)


class TestManifestContract(CustomTestCase):
    def test_startup_wait_is_sigterm_interruptible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "server.yaml"
            config_path.write_text(_minimal_config(), encoding="utf-8")
            orchestrator = RayServerOrchestrator(
                load_config(config_path),
                config_path=config_path,
                run_dir=root / "run",
            )
            stop = threading.Event()
            stop.set()
            with self.assertRaisesRegex(InterruptedError, "startup interrupted"):
                orchestrator._get_refs([object()], timeout_s=1.0, stop_event=stop)

    def test_ready_manifest_contains_every_engine_and_quota_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "server.yaml"
            config_path.write_text(_minimal_config(), encoding="utf-8")
            config = load_config(config_path)
            orchestrator = RayServerOrchestrator(
                config,
                config_path=config_path,
                run_dir=root / "run",
            )
            orchestrator.engine_infos = [
                {
                    "engine_id": f"{role}-{rank}",
                    "role": role,
                    "rank": rank,
                    "node_id": "node-a",
                    "node_ip": "10.0.0.1",
                    "http_url": f"http://10.0.0.1:{30000 + rank}",
                    "transport_endpoint": f"tcp://10.0.0.1:{31000 + rank}",
                    "gpu_ids": [str(rank)],
                }
                for role, count in (("verifier", 2), ("drafter", 3))
                for rank in range(count)
            ]
            manifest = orchestrator._write_manifest("ready")
            persisted = json.loads(
                (root / "run" / "server" / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest, persisted)
        self.assertEqual(persisted["state"], "ready")
        self.assertEqual(len(persisted["engines"]), 5)
        self.assertEqual(len(persisted["topology"]["quota_edges"]), 4)


if __name__ == "__main__":
    unittest.main()

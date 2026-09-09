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
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_BENCHMARK_ROOT = _REPOSITORY_ROOT / "benchmark" / "decoupled_spec"
sys.path.insert(0, str(_BENCHMARK_ROOT.parent))

_orchestrator = importlib.import_module("decoupled_spec.server.orchestrator")
_placement = importlib.import_module("decoupled_spec.server.placement")
_role_server = importlib.import_module("decoupled_spec.server.role_server")
EngineActor = _orchestrator.EngineActor
RayServerOrchestrator = _orchestrator.RayServerOrchestrator
build_peer_configs = _orchestrator.build_peer_configs
load_config = _orchestrator.load_config
plan_decoupled_spec_quota_edges = _orchestrator.plan_decoupled_spec_quota_edges
CandidateNode = _placement.CandidateNode
NodeAllocation = _placement.NodeAllocation
PlacementPlan = _placement.PlacementPlan
plan_decoupled_spec_placement = _placement.plan_decoupled_spec_placement
plan_spec_engine_placement = _placement.plan_spec_engine_placement


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
    disable_radix_cache: true
    disable_overlap_schedule: true
    speculative_algorithm: null
    speculative_num_steps: 3
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 4
"""


def _minimal_coupled_config() -> str:
    return """
schema_version: 2
deployment: coupled_spec
ray:
  address: auto
target:
  replicas: 1
  runtime:
    env: {}
  server_args:
    model_path: mock-target
    tp_size: 4
    speculative_algorithm: EAGLE
    speculative_draft_model_path: mock-target
    speculative_num_steps: 3
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 4
"""


class TestUnifiedServerConfig(CustomTestCase):
    def test_coupled_config_uses_one_target_role(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.yaml"
            path.write_text(_minimal_coupled_config(), encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.deployment_kind, "coupled_spec")
        self.assertEqual(config.primary_role, "target")
        self.assertEqual(config.target.server_args["tp_size"], 4)
        self.assertIsNone(config.verifier)
        self.assertIsNone(config.drafter)

    def test_coupled_config_rejects_decoupled_target_algorithm(self):
        config_text = _minimal_coupled_config().replace("EAGLE", "DECOUPLED_VERIFY")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.yaml"
            path.write_text(config_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be DECOUPLED_VERIFY"):
                load_config(path)

    def test_coupled_config_accepts_ordinary_decode_target(self):
        config_text = _minimal_coupled_config().replace(
            "    speculative_algorithm: EAGLE\n", ""
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.yaml"
            path.write_text(config_text, encoding="utf-8")
            config = load_config(path)

        self.assertIsNone(config.target.server_args.get("speculative_algorithm"))

    def test_ray_actor_uses_distributable_package_identity(self):
        self.assertEqual(EngineActor.__module__, "decoupled_spec.server.orchestrator")
        self.assertTrue(
            Path(_orchestrator.__file__).with_name("role_server.py").is_file()
        )

    def test_ray_runtime_upload_keeps_server_as_a_package(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "server.yaml"
            config_path.write_text(_minimal_config(), encoding="utf-8")
            orchestrator = RayServerOrchestrator(
                load_config(config_path),
                config_path=config_path,
                run_dir=Path(directory) / "run",
            )
            with patch("ray.init") as ray_init:
                orchestrator._connect_ray()

        self.assertTrue(ray_init.call_args.kwargs["log_to_driver"])
        runtime_env = ray_init.call_args.kwargs["runtime_env"]
        benchmark_package_root = Path(runtime_env["py_modules"][0])
        self.assertEqual(benchmark_package_root, _BENCHMARK_ROOT)
        self.assertTrue((benchmark_package_root / "server" / "__init__.py").is_file())
        self.assertFalse((benchmark_package_root / "server.py").exists())
        self.assertIn("results/**", runtime_env["excludes"])

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

    def test_drafter_supports_both_schedule_modes(self):
        for disable_overlap_schedule in (True, False):
            with self.subTest(disable_overlap_schedule=disable_overlap_schedule):
                config_text = _minimal_config().replace(
                    "    disable_overlap_schedule: true\n",
                    "    disable_overlap_schedule: "
                    f"{str(disable_overlap_schedule).lower()}\n",
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "server.yaml"
                    path.write_text(config_text, encoding="utf-8")
                    config = load_config(path)

                self.assertEqual(
                    config.drafter.server_args["disable_overlap_schedule"],
                    disable_overlap_schedule,
                )

    def test_drafter_overlap_keeps_phase_one_guards(self):
        overlap_config = _minimal_config().replace(
            "    disable_overlap_schedule: true\n",
            "    disable_overlap_schedule: false\n",
        )
        cases = [
            (
                overlap_config.replace(
                    "    model_path: mock-draft\n    tp_size: 1\n",
                    "    model_path: mock-draft\n    tp_size: 2\n",
                ),
                "requires tp_size=1",
            ),
            (
                overlap_config.replace("    page_size: 1\n", "    page_size: 64\n"),
                "requires page_size=1",
            ),
            (
                overlap_config.replace(
                    "    speculative_eagle_topk: 1\n",
                    "    speculative_eagle_topk: 2\n",
                ),
                "requires topk=1",
            ),
        ]
        for config_text, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "server.yaml"
                path.write_text(config_text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    load_config(path)

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

    def test_adaptive_artifacts_are_staged_to_actor_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "sglang_decoupled_verify_scheduler_cycle_profile",
                        "status": "complete",
                        "measurement_clock": "scheduler_verify_commit_gap",
                        "cost_scope": "decoupled_verifier_scheduler_cycle",
                        "cost_estimator": "trimmed_mean_10pct",
                        "draft_provider": "forward_stream_mock_gpu_tail_selector",
                        "profile_control_plane": "daemon_drop",
                        "context_anchor_mode": "measurement_midpoint",
                        "trajectory_acceptance": "full",
                        "points": [
                            {
                                "step": 0,
                                "batch_size": 1,
                                "context_len": 512,
                                "cost_ms": 7.0,
                                "cuda_graph_fraction": 1.0,
                                "mean_selected_draft_length": 0.0,
                                "mean_accept_length": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            adaptive_path = root / "adaptive.json"
            adaptive_path.write_text(
                json.dumps({"candidate_steps": [0, 1, 2, 3]}), encoding="utf-8"
            )
            config_path = root / "server.yaml"
            config_text = (
                _minimal_config()
                .replace(
                    '      SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"\n',
                    '      SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"\n'
                    f"      SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH: {profile_path}\n",
                    1,
                )
                .replace(
                    "    speculative_num_draft_tokens: 4\n",
                    "    speculative_num_draft_tokens: 4\n"
                    f"    speculative_adaptive_config: {adaptive_path}\n",
                    1,
                )
            )
            config_path.write_text(config_text, encoding="utf-8")
            config = load_config(config_path)
            orchestrator = RayServerOrchestrator(
                config,
                config_path=config_path,
                run_dir=root / "run",
            )

            actor = EngineActor(
                engine_id="verifier-0",
                role="verifier",
                rank=0,
                tp_size=2,
                node_rank=1,
                nnodes=2,
                expected_num_gpus=1,
                runtime_env=config.verifier.runtime.env,
                staged_files=orchestrator.verifier_staged_files,
                node_info_override={"node_id": "node-b", "node_ip": "10.0.0.2"},
                gpu_ids_override=["3"],
            )
            actor.allocation()
            resolved = {
                "server_args": {
                    "model_path": "dummy",
                    "tp_size": 2,
                    "nnodes": 2,
                    "node_rank": 1,
                    "dist_init_addr": "10.0.0.1:30001",
                    "speculative_adaptive_config": str(adaptive_path),
                }
            }
            engine = MagicMock()
            with patch.dict(os.environ, {}, clear=False), patch(
                "sglang.Engine", return_value=engine
            ) as engine_cls:
                actor.start(resolved)
                profile_local = Path(
                    actor.runtime_env["SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH"]
                )
                adaptive_local = Path(
                    engine_cls.call_args.kwargs["speculative_adaptive_config"]
                )
                self.assertNotEqual(profile_local, profile_path)
                self.assertNotEqual(adaptive_local, adaptive_path)
                self.assertEqual(profile_local.read_bytes(), profile_path.read_bytes())
                self.assertEqual(
                    adaptive_local.read_bytes(), adaptive_path.read_bytes()
                )
                actor.stop(0.1)


class TestQuotaTopology(CustomTestCase):
    @staticmethod
    def _nodes(*capacities):
        return [
            CandidateNode(
                node_id=f"node-{index}",
                node_ip=f"10.0.0.{index + 1}",
                available_gpus=capacity,
            )
            for index, capacity in enumerate(capacities)
        ]

    def test_placement_returns_typed_plan_and_serializes(self):
        plan = plan_decoupled_spec_placement(
            self._nodes(8, 8),
            num_verifiers=2,
            target_tp_size=2,
            num_drafters=2,
            draft_tp_size=1,
        )

        self.assertIsInstance(plan, PlacementPlan)
        self.assertTrue(
            all(isinstance(item, NodeAllocation) for item in plan.node_allocations)
        )
        self.assertEqual(plan.verifier_nodes_per_replica, 1)
        self.assertEqual(plan.verifier_gpus_per_node, 2)
        self.assertIn("verifier_gpus", plan.to_dict()["node_allocations"][0])

    def test_coupled_tp4_uses_exactly_four_gpus(self):
        plan = plan_spec_engine_placement(
            self._nodes(8), num_replicas=1, tp_size=4
        )

        self.assertEqual(plan.verifier_bundle_node_ids, [["node-0"]])
        self.assertEqual(plan.verifier_gpus_per_node, 4)
        self.assertEqual(plan.drafter_node_ids, [])
        self.assertEqual(plan.node_allocations[0].free_gpus, 4)

    def test_joint_planner_handles_fragmentation(self):
        plan = plan_decoupled_spec_placement(
            self._nodes(1, 2, 3),
            num_verifiers=1,
            target_tp_size=2,
            num_drafters=2,
            draft_tp_size=2,
        )
        self.assertEqual(plan.verifier_bundle_node_ids, [["node-0", "node-2"]])
        with self.assertRaisesRegex(ValueError, "unable to jointly place"):
            plan_decoupled_spec_placement(
                self._nodes(3, 3),
                num_verifiers=1,
                target_tp_size=4,
                num_drafters=1,
                draft_tp_size=2,
            )

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

    def test_tp16_verifier_and_tp1_drafter_use_three_8_gpu_nodes(self):
        nodes = self._nodes(8, 8, 8)

        plan = plan_decoupled_spec_placement(
            nodes,
            num_verifiers=1,
            target_tp_size=16,
            num_drafters=1,
            draft_tp_size=1,
        )

        self.assertEqual(plan.verifier_nodes_per_replica, 2)
        self.assertEqual(plan.verifier_gpus_per_node, 8)
        self.assertEqual(len(plan.verifier_bundle_node_ids[0]), 2)
        self.assertEqual(len(plan.drafter_node_ids), 1)
        self.assertEqual(
            sum(
                item.verifier_gpus + item.drafter_gpus for item in plan.node_allocations
            ),
            17,
        )

        with self.assertRaisesRegex(ValueError, "unable to jointly place"):
            plan_decoupled_spec_placement(
                nodes[:2],
                num_verifiers=1,
                target_tp_size=16,
                num_drafters=1,
                draft_tp_size=1,
            )


class TestNodeLocalPortLeases(CustomTestCase):
    def test_role_server_takes_noninheritable_socket_ownership(self):
        listener = _orchestrator._reserve_tcp_socket("127.0.0.1")
        child_fd = os.dup(listener.fileno())
        os.set_inheritable(child_fd, True)
        child_socket = _role_server._take_http_socket(
            child_fd, listener.getsockname()[1]
        )
        try:
            self.assertFalse(child_socket.get_inheritable())
        finally:
            child_socket.close()
            listener.close()
        with self.assertRaises(OSError):
            os.fstat(child_fd)

    def test_actor_prefers_and_holds_numbered_environment_ports(self):
        candidates = []
        while len(candidates) < 2:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
            probe.close()
            if port not in candidates:
                candidates.append(port)

        with patch.dict(
            os.environ,
            {"PORT10": str(candidates[1]), "PORT2": str(candidates[0])},
            clear=True,
        ):
            first = _orchestrator._reserve_tcp_socket("127.0.0.1")
            second = _orchestrator._reserve_tcp_socket("127.0.0.1")
        try:
            self.assertEqual(first.getsockname()[1], candidates[0])
            self.assertEqual(second.getsockname()[1], candidates[1])
        finally:
            first.close()
            second.close()

    def test_actor_rejects_invalid_or_duplicate_environment_ports(self):
        for env, message in (
            ({"PORT1": "bad"}, "invalid reserved port"),
            ({"PORT1": "12345", "PORT2": "12345"}, "must be unique"),
        ):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, message):
                    _orchestrator._reserve_tcp_socket("127.0.0.1")

    def test_leader_process_inherits_actor_info_log_streams(self):
        actor = EngineActor(
            engine_id="verifier-0",
            role="verifier",
            rank=0,
            tp_size=1,
            node_rank=0,
            nnodes=1,
            expected_num_gpus=1,
            runtime_env={},
            node_info_override={"node_id": "local", "node_ip": "127.0.0.1"},
            gpu_ids_override=["0"],
        )
        info = actor.reserve_ports()
        resolved = {
            "server_args": {
                "model_path": "dummy",
                "host": "127.0.0.1",
                "port": int(info["http_port"]),
                "nccl_port": int(info["nccl_port"]),
            }
        }
        process = MagicMock(pid=12345)
        observed = {}

        def launch_process(*args, **kwargs):
            http_socket_fd = kwargs["pass_fds"][0]
            os.fstat(http_socket_fd)
            observed["fd"] = http_socket_fd
            observed["command"] = args[0]
            competing = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with competing, self.assertRaises(OSError):
                competing.bind(("127.0.0.1", int(info["http_port"])))
            return process

        with patch("subprocess.Popen", side_effect=launch_process) as popen:
            actor.start(resolved)

        self.assertIsNone(popen.call_args.kwargs["stdout"])
        self.assertIsNone(popen.call_args.kwargs["stderr"])
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")
        self.assertIn("--http-socket-fd", observed["command"])
        with self.assertRaises(OSError):
            os.fstat(observed["fd"])
        actor._process = None
        actor.stop(0.1)

    def test_actor_normalizes_bracketed_ipv6_from_ray(self):
        actor = EngineActor(
            engine_id="drafter-0",
            role="drafter",
            rank=0,
            tp_size=1,
            node_rank=0,
            nnodes=1,
            expected_num_gpus=1,
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
            node_rank=0,
            nnodes=1,
            expected_num_gpus=1,
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
            node_rank=0,
            nnodes=1,
            expected_num_gpus=1,
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

    def test_nonzero_node_actor_owns_native_engine(self):
        actor = EngineActor(
            engine_id="verifier-0",
            role="verifier",
            rank=0,
            tp_size=2,
            node_rank=1,
            nnodes=2,
            expected_num_gpus=1,
            runtime_env={"SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND": "1"},
            node_info_override={"node_id": "node-b", "node_ip": "10.0.0.2"},
            gpu_ids_override=["3"],
        )
        engine = MagicMock()
        resolved = {
            "server_args": {
                "model_path": "dummy",
                "tp_size": 2,
                "nnodes": 2,
                "node_rank": 1,
                "dist_init_addr": "10.0.0.1:30001",
            }
        }

        with patch.dict(os.environ, {}, clear=False), patch(
            "sglang.Engine", return_value=engine
        ) as engine_cls:
            allocation = actor.allocation()
            start_info = actor.start(resolved)
            result = actor.stop(0.1)

        self.assertEqual(allocation["gpu_ids"], ["3"])
        self.assertEqual(start_info["state"], "engine_ready")
        engine_cls.assert_called_once_with(**resolved["server_args"])
        engine.shutdown.assert_called_once_with()
        self.assertEqual(result["node_rank"], 1)
        self.assertEqual(result["return_code"], 0)


class TestManifestContract(CustomTestCase):
    def test_coupled_manifest_has_one_target_and_no_transport_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "server.yaml"
            config_path.write_text(_minimal_coupled_config(), encoding="utf-8")
            orchestrator = RayServerOrchestrator(
                load_config(config_path),
                config_path=config_path,
                run_dir=root / "run",
            )
            orchestrator.engine_infos = [
                {
                    "engine_id": "target-0",
                    "role": "target",
                    "rank": 0,
                    "node_id": "node-a",
                    "node_ip": "10.0.0.1",
                    "http_url": "http://10.0.0.1:30000",
                    "gpu_ids": ["0", "1", "2", "3"],
                }
            ]
            manifest = orchestrator._write_manifest("ready")

        self.assertEqual(manifest["deployment"], "coupled_spec")
        self.assertEqual(manifest["topology"]["primary_role"], "target")
        self.assertEqual(manifest["topology"]["num_verifiers"], 0)
        self.assertEqual(manifest["topology"]["num_drafters"], 0)
        self.assertEqual(manifest["topology"]["quota_edges"], [])
        self.assertNotIn("transport_endpoint", manifest["engines"][0])

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

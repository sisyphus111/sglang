"""Unit tests for LoadSnapshot SHM and ZMQ backends."""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace

from sglang.srt.managers.load_snapshot import (
    DecoupledSpecDecodeMetrics,
    DecodeMetricsWindow,
    DraftTailSelectMetrics,
    DraftTransportMetrics,
    IntegerHistogram,
    LatencyHistogram,
    LoadSnapshot,
    SLOT_LEN_STRUCT,
    SLOT_SIZE,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
    SpeculativeMetrics,
    ZmqLoadSnapshotWriter,
    ZmqShmLoadSnapshotReader,
    _zmq_addr_for,
    create_load_snapshot_reader,
    create_load_snapshot_writer,
    snapshot_encoder,
    should_use_zmq,
    zmq_reader_owner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _temp_path() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(path)
    return path


def _ipc_addr() -> str:
    fd, path = tempfile.mkstemp(prefix="sglang_test_zmq_", suffix=".sock")
    os.close(fd)
    os.unlink(path)
    return f"ipc://{path}"


def _warmup_zmq(writers, reader, attempts=20, interval=0.05):
    """Send warmup messages until the reader receives from all writers."""
    expected = {w.dp_rank for w in writers}
    received = set()
    for _ in range(attempts):
        for w in writers:
            w.write(LoadSnapshot(dp_rank=w.dp_rank, timestamp=-1.0, num_running_reqs=0))
        time.sleep(interval)
        for rank in expected:
            load = reader.read(rank)
            if load is not None:
                received.add(rank)
        if received >= expected:
            return
    raise RuntimeError(f"warmup failed: expected {expected}, received {received}")


def _decoupled_metrics() -> DecoupledSpecDecodeMetrics:
    empty_latency = LatencyHistogram(
        count=0,
        sum_us=0,
        bucket_upper_bounds_us=[
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1_000,
            2_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
            250_000,
            500_000,
            1_000_000,
        ],
        bucket_counts=[0] * 18,
    )
    return DecoupledSpecDecodeMetrics(
        tail_select=DraftTailSelectMetrics(
            num_select_rows=320,
            num_select_valid_rows=160,
            reason_counts={"direct": 160, "bonus_mismatch": 160},
            selected_draft_length_histogram=IntegerHistogram(
                offset=0, counts=[160, 0, 0, 0, 0, 160]
            ),
            raw_draft_tail_length_histogram=IntegerHistogram(
                offset=-1, counts=[0, 0, 0, 0, 0, 0, 320, 0, 0, 0, 0, 0, 0]
            ),
            consumable_draft_tail_length_histogram=IntegerHistogram(
                offset=-1, counts=[0, 0, 0, 0, 0, 0, 320, 0, 0, 0, 0, 0, 0]
            ),
            logical_delta_histogram=IntegerHistogram(
                offset=-11,
                counts=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 320] + [0] * 11,
            ),
            pending_prefix_length_histogram=IntegerHistogram(
                offset=-1, counts=[0, 320] + [0] * 11
            ),
            num_publish_seq_initial=8,
            num_publish_seq_same=80,
            num_publish_seq_advance=232,
            num_seqlock_retry_rows=12,
            num_seqlock_retries=19,
            max_seqlock_retries=4,
        ),
        transport=DraftTransportMetrics(
            num_draft_result_frames=0,
            num_draft_result_tokens=0,
            draft_receive_to_gpu_publish_enqueue_latency_us=empty_latency,
            draft_gpu_publish_completion_latency_us=empty_latency,
            draft_transport_one_way_latency_us=empty_latency,
            draft_result_ready_to_receive_latency_us=empty_latency,
            gpu_publish_staging_slots_max=8,
            clock_sync_valid=False,
            num_clock_sync_valid_peers=1,
            num_clock_sync_invalid_peers=1,
        ),
        adaptive_verify={
            "active_verify_steps": 2,
            "max_verify_steps": 5,
            "decision_seq": 255,
            "round_count": 1277,
            "reevaluation_count": 255,
            "switch_count": 32,
            "window_reevaluation_count": 8,
            "window_switch_count": 2,
            "step_residency": [0, 0, 1246, 25, 0, 6],
            "window_step_residency": [0, 0, 40, 0, 0, 0],
            "last_reason": "keep_current",
            "last_decision_latency_ms": 0.1,
            "candidate_scores": [
                {
                    "steps": step,
                    "expected_accept_length": 1.0 + step / 5,
                    "cost_ms": 7.0 + step / 10,
                    "modeled_tps": 500.0 + step * 100,
                    "matched_batch_size": 4,
                    "matched_context_len": 4096,
                }
                for step in range(6)
            ],
            "switch_events": [
                {
                    "decision_seq": index,
                    "round_count": index * 5,
                    "old_steps": 5 - index % 4,
                    "new_steps": 4 - index % 4,
                    "batch_size": 4,
                    "context_len": 4096,
                    "reason": "score_hysteresis",
                }
                for index in range(32)
            ],
            "profile_status": "complete",
            "profile_sha256": "a" * 64,
            "supply_ema": [0.8] * 5,
            "conditional_accept_ema": [0.9] * 5,
            "accept_update_counts": [1200] * 5,
            "accept_sample_counts": [3000] * 5,
        },
    )


class TestShmRoundTrip(CustomTestCase):
    def test_partial_clock_sync_keeps_healthy_peer_latency_samples(self):
        histogram = {
            "count": 1,
            "sum_us": 25,
            "bucket_upper_bounds_us": [10, 100],
            "bucket_counts": [0, 1, 0],
        }

        metrics = DraftTransportMetrics.from_dict(
            {
                "num_draft_result_frames": 1,
                "num_draft_result_tokens": 2,
                "draft_transport_one_way_latency_us": histogram,
                "draft_result_ready_to_receive_latency_us": histogram,
                "clock_sync_valid": False,
                "clock_error_bound_us": 5.5,
                "num_clock_sync_valid_peers": 1,
                "num_clock_sync_invalid_peers": 1,
            }
        )

        self.assertFalse(metrics.clock_sync_valid)
        self.assertEqual(metrics.draft_transport_one_way_latency_us.count, 1)
        self.assertEqual(metrics.draft_result_ready_to_receive_latency_us.count, 1)
        self.assertEqual(metrics.num_clock_sync_valid_peers, 1)
        self.assertEqual(metrics.num_clock_sync_invalid_peers, 1)

    def test_http_dict_keeps_ordinary_decode_window_shape(self):
        window = DecodeMetricsWindow(
            window_id=1,
            end_time=1.0,
            num_decode_iters=40,
            iter_latency_ms=1.0,
        )

        value = LoadSnapshot(decode_metrics_windows=[window]).to_dict()[
            "decode_metrics_windows"
        ][0]

        self.assertNotIn("decoupled_spec", value)

    def test_http_dict_omits_transport_fields_owned_by_the_other_role(self):
        drafter_window = DecodeMetricsWindow(
            window_id=1,
            end_time=1.0,
            num_decode_iters=40,
            iter_latency_ms=1.0,
            decoupled_spec=DecoupledSpecDecodeMetrics(
                transport=DraftTransportMetrics(
                    num_draft_result_frames=0,
                    num_draft_result_tokens=0,
                    draft_send_queue_latency_us=LatencyHistogram(
                        count=0,
                        sum_us=0,
                        bucket_upper_bounds_us=[10],
                        bucket_counts=[0, 0],
                    ),
                    draft_send_queue_depth_max=0,
                )
            ),
        )

        transport = LoadSnapshot(decode_metrics_windows=[drafter_window]).to_dict()[
            "decode_metrics_windows"
        ][0]["decoupled_spec"]["transport"]

        self.assertIn("draft_send_queue_latency_us", transport)
        self.assertNotIn("draft_receive_to_gpu_publish_enqueue_latency_us", transport)
        self.assertNotIn("num_clock_sync_valid_peers", transport)
        self.assertNotIn("clock_sync_valid", transport)

    def test_full_decode_metrics_history_fits_one_snapshot_slot(self):
        windows = [
            DecodeMetricsWindow(
                window_id=index,
                end_time=1000.0 + index,
                num_decode_iters=40,
                iter_latency_ms=10.0,
                num_decode_rows=320,
                sum_context_lens=3_200_000,
                mean_batch_size=8.0,
                mean_context_length=10_000.0,
                num_verify_rows=320,
                num_accept_tokens=640,
                num_proposed_drafts=480,
                accept_length=2.0,
                proposed_draft_length=1.5,
                decoupled_spec=_decoupled_metrics(),
            )
            for index in range(64)
        ]

        payload = snapshot_encoder.encode(
            LoadSnapshot(dp_rank=0, decode_metrics_windows=windows)
        )

        self.assertLess(len(payload), SLOT_SIZE - SLOT_LEN_STRUCT.size)

    def test_single_rank_write_read(self):
        path = _temp_path()
        writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
        reader = ShmLoadSnapshotReader(path, dp_size=1)
        try:
            writer.write(
                LoadSnapshot(
                    dp_rank=0,
                    num_running_reqs=5,
                    timestamp=1.0,
                    num_active_tokens=4096,
                    total_prefill_uncached_tokens=1000,
                    total_prefill_busy_us=250_000,
                    decode_moments=[2, 30, 3000, 500, 50_000, 60],
                    decode_metrics_windows=[
                        DecodeMetricsWindow(
                            window_id=7,
                            end_time=12.5,
                            num_decode_iters=40,
                            iter_latency_ms=10.0,
                            num_decode_rows=320,
                            sum_context_lens=3_200_000,
                            mean_batch_size=8.0,
                            mean_context_length=10_000.0,
                            num_verify_rows=320,
                            num_accept_tokens=640,
                            num_proposed_drafts=480,
                            accept_length=2.0,
                            proposed_draft_length=1.5,
                            decoupled_spec=_decoupled_metrics(),
                        )
                    ],
                    speculative=SpeculativeMetrics(
                        accept_length=2.0,
                        accept_rate=0.5,
                        draft_occupancy_rate=0.25,
                        proposed_draft_length=0.75,
                    ),
                )
            )
            load = reader.read(0)
            self.assertIsNotNone(load)
            self.assertEqual(load.num_running_reqs, 5)
            self.assertEqual(load.timestamp, 1.0)
            self.assertEqual(load.num_active_tokens, 4096)
            # The cumulative total_* counters round-trip like any other
            # core scalar.
            self.assertEqual(load.total_prefill_uncached_tokens, 1000)
            self.assertEqual(load.total_prefill_busy_us, 250_000)
            self.assertEqual(load.decode_moments[0], 2)
            self.assertEqual(load.decode_moments[5], 60)
            self.assertEqual(load.decode_metrics_windows[0].window_id, 7)
            self.assertEqual(load.decode_metrics_windows[0].iter_latency_ms, 10.0)
            self.assertEqual(load.decode_metrics_windows[0].mean_batch_size, 8.0)
            self.assertEqual(
                load.decode_metrics_windows[0].mean_context_length, 10_000.0
            )
            self.assertEqual(
                load.to_dict()["decode_metrics_windows"][0]["accept_length"], 2.0
            )
            decoupled = load.to_dict()["decode_metrics_windows"][0]["decoupled_spec"]
            self.assertEqual(decoupled["tail_select"]["num_select_rows"], 320)
            self.assertEqual(
                decoupled["tail_select"]["reason_counts"],
                {"direct": 160, "bonus_mismatch": 160},
            )
            self.assertEqual(
                decoupled["tail_select"]["num_seqlock_retry_rows"], 12
            )
            self.assertEqual(decoupled["tail_select"]["num_seqlock_retries"], 19)
            self.assertEqual(decoupled["tail_select"]["max_seqlock_retries"], 4)
            self.assertEqual(
                decoupled["transport"][
                    "draft_receive_to_gpu_publish_enqueue_latency_us"
                ]["bucket_counts"],
                [0] * 18,
            )
            self.assertEqual(load.speculative.accept_rate, 0.5)
            self.assertEqual(load.speculative.draft_occupancy_rate, 0.25)
            self.assertEqual(load.speculative.proposed_draft_length, 0.75)
        finally:
            reader.close()
            writer.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_multi_rank_write_read_all(self):
        path = _temp_path()
        writers = []
        try:
            for rank in range(4):
                w = ShmLoadSnapshotWriter(path, dp_size=4, dp_rank=rank)
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        num_running_reqs=rank * 10,
                        timestamp=1.0,
                    )
                )
                writers.append(w)

            reader = ShmLoadSnapshotReader(path, dp_size=4)
            loads = reader.read_all()
            self.assertEqual(len(loads), 4)
            for i, load in enumerate(loads):
                self.assertEqual(load.dp_rank, i)
                self.assertEqual(load.num_running_reqs, i * 10)
            reader.close()
        finally:
            for w in writers:
                w.close()
            if os.path.exists(path):
                os.unlink(path)

    def test_reader_empty_before_writer(self):
        path = _temp_path()
        reader = ShmLoadSnapshotReader(path, dp_size=2)
        self.assertEqual(reader.read_all(), [])
        self.assertIsNone(reader.read(0))
        reader.close()


class TestZmqRoundTrip(CustomTestCase):
    def test_single_rank_zmq_to_shm(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=2)
        writer = ZmqLoadSnapshotWriter(addr, dp_size=2, dp_rank=0)
        try:
            _warmup_zmq([writer], reader)

            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=7, timestamp=2.0))
            time.sleep(0.05)

            load = reader.read(0)
            self.assertIsNotNone(load)
            self.assertEqual(load.num_running_reqs, 7)
            self.assertEqual(load.timestamp, 2.0)
        finally:
            writer.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_multi_rank_zmq(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        dp_size = 4
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size)
        writers = []
        try:
            for rank in range(dp_size):
                w = ZmqLoadSnapshotWriter(addr, dp_size, dp_rank=rank)
                writers.append(w)

            _warmup_zmq(writers, reader)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(dp_rank=rank, num_running_reqs=rank + 1, timestamp=3.0)
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(len(loads), dp_size)
            for load in loads:
                self.assertEqual(load.num_running_reqs, load.dp_rank + 1)
        finally:
            for w in writers:
                w.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_read_returns_latest(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=1)
        writer = ZmqLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            _warmup_zmq([writer], reader)

            for i in range(10):
                writer.write(
                    LoadSnapshot(dp_rank=0, num_running_reqs=i, timestamp=float(i))
                )
            time.sleep(0.05)

            load = reader.read(0)
            self.assertIsNotNone(load)
            self.assertEqual(load.num_running_reqs, 9)
            self.assertEqual(load.timestamp, 9.0)
        finally:
            writer.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def test_zmq_writer_noblock_without_reader(self):
        addr = _ipc_addr()
        writer = ZmqLoadSnapshotWriter(addr, dp_size=1, dp_rank=0)
        try:
            writer.write(LoadSnapshot(dp_rank=0, num_running_reqs=1, timestamp=1.0))
        finally:
            writer.close()
            ipc_path = addr[len("ipc://") :]
            if os.path.exists(ipc_path):
                os.unlink(ipc_path)

    def test_reader_ipc_cleanup(self):
        addr = _ipc_addr()
        shm_path = _temp_path()
        ipc_path = addr[len("ipc://") :]
        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size=1)
        self.assertTrue(os.path.exists(ipc_path))
        reader.close()
        self.assertFalse(os.path.exists(ipc_path))
        if os.path.exists(shm_path):
            os.unlink(shm_path)


class TestFactoryFunctions(CustomTestCase):
    def test_shm_mode(self):
        server_args = SimpleNamespace(
            enable_dp_attention=False,
            nnodes=1,
            dp_size=1,
            load_balance_method="round_robin",
            node_rank=0,
            tokenizer_worker_num=1,
        )
        port_args = SimpleNamespace(instance_id="test_shm_factory")
        writer = create_load_snapshot_writer(
            server_args, port_args, dp_size=1, dp_rank=0
        )
        self.assertIsInstance(writer, ShmLoadSnapshotWriter)
        reader = create_load_snapshot_reader(
            server_args, port_args, caller="TokenizerManager"
        )
        self.assertIsInstance(reader, ShmLoadSnapshotReader)
        reader.close()
        writer.close()
        from sglang.srt.managers.load_snapshot import shm_path_for

        path = shm_path_for("test_shm_factory")
        if os.path.exists(path):
            os.unlink(path)

    def test_zmq_mode_via_env(self):
        server_args = SimpleNamespace(
            enable_dp_attention=False,
            nnodes=1,
            dp_size=1,
            load_balance_method="round_robin",
            node_rank=0,
            tokenizer_worker_num=1,
        )
        port_args = SimpleNamespace(instance_id="test_zmq_factory")
        os.environ["SGLANG_LOAD_SNAPSHOT_USE_ZMQ"] = "1"
        try:
            writer = create_load_snapshot_writer(
                server_args, port_args, dp_size=1, dp_rank=0
            )
            self.assertIsInstance(writer, ZmqLoadSnapshotWriter)
            reader = create_load_snapshot_reader(
                server_args, port_args, caller="TokenizerManager"
            )
            self.assertIsInstance(reader, ZmqShmLoadSnapshotReader)
            reader.close()
            writer.close()
        finally:
            del os.environ["SGLANG_LOAD_SNAPSHOT_USE_ZMQ"]

    def test_should_use_zmq_multinode_dp_attention(self):
        args = SimpleNamespace(enable_dp_attention=True, nnodes=2)
        self.assertTrue(should_use_zmq(args))


class TestZmqReaderOwner(CustomTestCase):
    """At most one process binds the zmq PULL socket across all callers."""

    CALLERS = ("TokenizerManager", "MultiTokenizerRouter", "DataParallelController")

    @staticmethod
    def _args(**overrides):
        base = dict(
            enable_dp_attention=True,
            nnodes=2,
            node_rank=0,
            dp_size=1,
            load_balance_method="round_robin",
            tokenizer_worker_num=1,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _owners(self, args):
        return {c for c in self.CALLERS if zmq_reader_owner(args, c)}

    def test_zmq_disabled_no_owner(self):
        args = self._args(enable_dp_attention=False, nnodes=1)
        self.assertEqual(self._owners(args), set())

    def test_non_zero_node_rank_no_owner(self):
        args = self._args(node_rank=1, dp_size=4, tokenizer_worker_num=8)
        self.assertEqual(self._owners(args), set())

    def test_tokenizer_manager_owns_when_dp1(self):
        self.assertEqual(self._owners(self._args(dp_size=1)), {"TokenizerManager"})

    def test_multi_tokenizer_router_owns_in_multi_tokenizer_dp1(self):
        args = self._args(dp_size=1, tokenizer_worker_num=8)
        self.assertEqual(self._owners(args), {"MultiTokenizerRouter"})

    def test_multi_tokenizer_router_owns_in_multi_tokenizer_round_robin(self):
        args = self._args(dp_size=4, tokenizer_worker_num=8)
        self.assertEqual(self._owners(args), {"MultiTokenizerRouter"})

    def test_data_parallel_controller_owns_load_aware(self):
        for method in ("total_tokens", "total_requests"):
            args = self._args(
                dp_size=4, tokenizer_worker_num=8, load_balance_method=method
            )
            self.assertEqual(self._owners(args), {"DataParallelController"})

    def test_tokenizer_manager_owns_dp4_round_robin(self):
        args = self._args(dp_size=4, tokenizer_worker_num=1)
        self.assertEqual(self._owners(args), {"TokenizerManager"})

    def test_at_most_one_owner_across_configs(self):
        for dp_size in (1, 4):
            for tw in (1, 8):
                for method in ("round_robin", "total_tokens", "total_requests"):
                    for node_rank in (0, 1):
                        args = self._args(
                            dp_size=dp_size,
                            tokenizer_worker_num=tw,
                            load_balance_method=method,
                            node_rank=node_rank,
                        )
                        self.assertLessEqual(len(self._owners(args)), 1, args)


class TestZmqAddr(CustomTestCase):
    def test_ipc_for_single_node(self):
        port_args = SimpleNamespace(instance_id="myinstance")
        addr = _zmq_addr_for(port_args)
        self.assertTrue(addr.startswith("ipc://"))
        self.assertIn("myinstance", addr)

    def test_tcp_from_port_args(self):
        from sglang.srt.utils.network import NetworkAddress

        port_args = SimpleNamespace(
            instance_id="myinstance",
            load_collector_ipc_name=NetworkAddress("10.0.0.1", 29506).to_tcp(),
        )
        addr = _zmq_addr_for(port_args)
        self.assertTrue(addr.startswith("tcp://"))
        self.assertIn("10.0.0.1", addr)


class TestEndToEndZmqSimulation(CustomTestCase):
    """Simulate multi-node DP attention on single machine using IPC."""

    def test_full_flow_dp_size_2(self):
        shm_path = _temp_path()
        addr = _ipc_addr()
        dp_size = 2

        reader = ZmqShmLoadSnapshotReader(addr, shm_path, dp_size)

        writers = []
        for rank in range(dp_size):
            w = ZmqLoadSnapshotWriter(addr, dp_size, dp_rank=rank)
            writers.append(w)

        try:
            _warmup_zmq(writers, reader)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        timestamp=1.0,
                        num_running_reqs=10 + rank,
                        num_waiting_reqs=5 + rank,
                        num_total_tokens=100 + rank * 50,
                    )
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(len(loads), dp_size)
            self.assertEqual(loads[0].num_running_reqs, 10)
            self.assertEqual(loads[1].num_running_reqs, 11)
            self.assertEqual(loads[0].num_total_tokens, 100)
            self.assertEqual(loads[1].num_total_tokens, 150)

            for rank, w in enumerate(writers):
                w.write(
                    LoadSnapshot(
                        dp_rank=rank,
                        timestamp=2.0,
                        num_running_reqs=20 + rank,
                        num_waiting_reqs=0,
                        num_total_tokens=200 + rank * 50,
                    )
                )
            time.sleep(0.05)

            loads = reader.read_all()
            self.assertEqual(loads[0].num_running_reqs, 20)
            self.assertEqual(loads[1].num_running_reqs, 21)
            self.assertEqual(loads[0].num_total_tokens, 200)
            self.assertEqual(loads[1].num_total_tokens, 250)
        finally:
            for w in writers:
                w.close()
            reader.close()
            if os.path.exists(shm_path):
                os.unlink(shm_path)


if __name__ == "__main__":
    unittest.main()

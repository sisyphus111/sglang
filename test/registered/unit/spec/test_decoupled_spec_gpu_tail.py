"""CUDA tests for the verifier's native rolling draft-tail landing path."""

import time
import threading
import unittest
import uuid

import torch
import zmq

from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDrafterDecoupledSpecDataPlane,
    CppVerifierDecoupledSpecDataPlane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCppGpuDraftTailBuffer(CustomTestCase):
    def setUp(self):
        self.context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://gpu-tail-verifier-{suffix}"
        drafter_endpoint = f"inproc://gpu-tail-drafter-{suffix}"
        self.landing_stream = torch.cuda.Stream(device="cuda:0")
        self.verifier = CppVerifierDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            required_tail_len=3,
            context=self.context,
            device="cuda:0",
            num_gpu_seats=4,
            num_draft_tokens=3,
            landing_stream=self.landing_stream,
        )
        self.drafter = CppDrafterDecoupledSpecDataPlane(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=self.context,
        )
        self.verifier.start()
        self.drafter.start()

    def tearDown(self):
        self.drafter.close()
        self.verifier.close()
        self.context.destroy(linger=0)

    def test_stream_commit_pending_and_reseat_are_device_selected(self):
        self._open("old", request_epoch=5)
        self._publish("old", range(7), token_base=10)

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        seq_lens = torch.tensor([2], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([10], dtype=torch.int32, device="cuda:0")

        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[11, 12, 13, 3, 1],
            expected_cursor=1,
        )

        self.verifier.commit(
            VerifyCommit(
                request_id="old",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[10, 11],
            )
        )
        seq_lens.fill_(3)
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[12, 13, 14, 3, 1],
            expected_cursor=2,
        )

        self.verifier.commit(
            VerifyCommit(
                request_id="old",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=2,
                committed_tokens=[99],
            )
        )
        seq_lens.fill_(4)
        bonus_tokens.fill_(99)
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[0, 0, 0, 0, 0],
            expected_cursor=3,
        )

        self._publish(
            "old",
            [2, 3],
            token_base=97,
            base_committed_lens=[2, 3],
        )
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[100, 0, 0, 1, 1],
            expected_cursor=3,
        )

        # Rebinding the seat changes persistent lifecycle metadata before the
        # daemon can publish the new row. The old row must immediately miss,
        # including its commit cursor.
        self.verifier.gpu_tail_buffer.bind_request("new", 1, 6)
        compact, cursor = self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats, seq_lens, bonus_tokens
        )
        self.assertEqual(compact.cpu().tolist(), [[0, 0, 0, 0, 0]])
        self.assertEqual(cursor.cpu().tolist(), [-1])

    def test_tail_capacity_is_a_fail_fast_invariant(self):
        self._open("overflow", request_epoch=1)
        self._publish("overflow", range(8), token_base=20)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.verifier.snapshot_one("overflow").raw_tail_len == 8:
                break
            time.sleep(0.001)
        else:
            self.fail("Overflow tail did not reach the verifier CPU mirror")
        time.sleep(0.01)
        with self.assertRaisesRegex(
            RuntimeError,
            "CppDraftProxyThread failed: GPU draft-tail capacity exceeded",
        ):
            self.verifier.commit(
                VerifyCommit(
                    request_id="overflow",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=0,
                    committed_tokens=[20],
                )
            )

    def test_concurrent_landing_select_never_exposes_a_torn_tail(self):
        self._open("race", request_epoch=1)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        # prompt_len=2 and seq_len=1 gives logical output length zero.
        seq_lens = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([0], dtype=torch.int32, device="cuda:0")
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[0, 0, 0, 0, 1],
            expected_cursor=0,
        )

        errors = []

        def publish_tokens():
            try:
                for position in range(7):
                    self._publish("race", [position], token_base=100)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        publisher = threading.Thread(target=publish_tokens)
        publisher.start()
        for _ in range(256):
            compact, cursor = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats, seq_lens, bonus_tokens
            )
            values = compact.cpu().tolist()[0]
            cursor_value = cursor.cpu().tolist()[0]
            selected_len, row_valid = values[-2:]
            if row_valid:
                self.assertEqual(cursor_value, 0)
                self.assertIn(selected_len, (0, 1, 2, 3))
                self.assertEqual(values[:selected_len], [100, 101, 102][:selected_len])
                self.assertEqual(values[selected_len:3], [0] * (3 - selected_len))
            else:
                # A genuine seqlock collision invalidates both the selected
                # tail and its cursor. A semantic miss from one stable row may
                # retain the row's cursor for host-side fallback bookkeeping.
                self.assertIn(cursor_value, (-1, 0))
                self.assertEqual(values, [0, 0, 0, 0, 0])
        publisher.join()
        self.assertEqual(errors, [])
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[100, 101, 102, 3, 1],
            expected_cursor=0,
        )

    def test_landing_staging_ring_grows_without_waiting_for_inflight_slots(self):
        # Keep every initial staging event incomplete while the daemon drains
        # control messages. The ninth publication must allocate a new slot;
        # it must not synchronize an existing slot or fail at the initial size.
        blocker_stream = torch.cuda.Stream(device="cuda:0")
        landing_gate = torch.cuda.Event()
        with torch.cuda.stream(blocker_stream):
            torch.cuda._sleep(1_000_000_000)
            landing_gate.record()
        self.landing_stream.wait_event(landing_gate)

        self._open("growth", request_epoch=1)
        gpu_tail_buffer = self.verifier.gpu_tail_buffer
        self.assertEqual(gpu_tail_buffer.staging_slot_count, 8)
        for committed_len in range(24):
            self.verifier.commit(
                VerifyCommit(
                    request_id="growth",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=committed_len,
                    committed_tokens=[1000 + committed_len],
                )
            )

        deadline = time.monotonic() + 3.0
        while gpu_tail_buffer.staging_slot_count == 8 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertGreater(gpu_tail_buffer.staging_slot_count, 8)
        self.assertLessEqual(
            gpu_tail_buffer.staging_slot_count,
            gpu_tail_buffer.max_staging_slots,
        )

    def test_close_synchronizes_landing_stream_before_releasing_staging(self):
        blocker_stream = torch.cuda.Stream(device="cuda:0")
        landing_gate = torch.cuda.Event()
        with torch.cuda.stream(blocker_stream):
            torch.cuda._sleep(200_000_000)
            landing_gate.record()
        self.landing_stream.wait_event(landing_gate)

        self._open("shutdown", request_epoch=1)
        landing_done = torch.cuda.Event()
        landing_done.record(self.landing_stream)
        self.verifier.close()

        # GpuDraftTailBufferCore.close() must synchronize the landing stream
        # before destroying its events or freeing pinned/device staging.
        self.assertTrue(landing_done.query())

    def _open(self, request_id: str, *, request_epoch: int) -> None:
        self.verifier.open_request(
            DraftSync(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=[],
            ),
            gpu_seat=1,
            request_epoch=request_epoch,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.drafter.drain_controls():
                return
            time.sleep(0.001)
        self.fail("DraftSync did not reach the drafter")

    def _publish(
        self,
        request_id: str,
        positions,
        *,
        token_base: int,
        base_committed_lens=None,
    ) -> None:
        positions = list(positions)
        if base_committed_lens is None:
            base_committed_lens = [0] * len(positions)
        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id=request_id,
                        base_committed_len=int(base),
                        new_token_pos=int(position),
                        new_token=token_base + int(position),
                    )
                    for position, base in zip(positions, base_committed_lens)
                ]
            )
        )

    def _wait_select(
        self,
        gpu_seats: torch.Tensor,
        seq_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        *,
        expected_compact: list[int],
        expected_cursor: int,
    ) -> None:
        deadline = time.monotonic() + 3.0
        while True:
            compact, cursor = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats, seq_lens, bonus_tokens
            )
            compact_values = compact.cpu().tolist()
            cursor_values = cursor.cpu().tolist()
            if compact_values == [expected_compact] and cursor_values == [
                expected_cursor
            ]:
                return
            if time.monotonic() >= deadline:
                self.fail(
                    "GPU draft-tail select did not converge: "
                    f"compact={compact_values} cursor={cursor_values}"
                )
            time.sleep(0.001)


if __name__ == "__main__":
    unittest.main()

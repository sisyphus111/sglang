"""CUDA tests for the verifier's native rolling draft-tail landing path."""

import random
import struct
import threading
import time
import unittest
import uuid

import torch
import zmq

from sglang.srt.speculative.cpp_decoupled_spec import (
    CppDrafterDecoupledSpecDataPlane,
    CppVerifierDecoupledSpecDataPlane,
    GPU_DRAFT_TAIL_DEBUG_FIELD_NAMES,
    GPU_DRAFT_TAIL_SELECT_REASON_NAMES,
)
from sglang.srt.speculative.decoupled_spec_data_plane import (
    DrafterDecoupledSpecDataPlane,
    VerifierDecoupledSpecDataPlane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DecoupledSpecIpcConfig,
    DraftClose,
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCppGpuDraftTailBuffer(CustomTestCase):
    verifier_type = CppVerifierDecoupledSpecDataPlane
    drafter_type = CppDrafterDecoupledSpecDataPlane

    def setUp(self):
        self.context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://gpu-tail-verifier-{suffix}"
        drafter_endpoint = f"inproc://gpu-tail-drafter-{suffix}"
        self.landing_stream = torch.cuda.Stream(device="cuda:0")
        self.verifier_endpoint = verifier_endpoint
        self.verifier = self.verifier_type(
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
        self.drafter = self.drafter_type(
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

    def test_mock_profile_selector_reads_static_gpu_tail(self):
        suffix = uuid.uuid4().hex
        mock_verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=f"inproc://mock-gpu-tail-verifier-{suffix}",
                connect_endpoints=(f"inproc://mock-gpu-tail-drafter-{suffix}",),
                rank=0,
            ),
            required_tail_len=0,
            context=self.context,
            device="cuda:0",
            num_gpu_seats=4,
            num_draft_tokens=3,
            landing_stream=torch.cuda.Stream(device="cuda:0"),
            mock_profile=True,
        )
        try:
            gpu_seats = torch.tensor([0, 2], dtype=torch.int64, device="cuda:0")
            seq_lens = torch.tensor([511, 1023], dtype=torch.int64, device="cuda:0")
            bonus_tokens = torch.tensor([17, 18], dtype=torch.int32, device="cuda:0")
            debug_out = torch.empty(
                (2, len(GPU_DRAFT_TAIL_DEBUG_FIELD_NAMES)),
                dtype=torch.int64,
                device="cuda:0",
            )
            compact, cursor = mock_verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=torch.zeros_like(gpu_seats),
                debug_out=debug_out,
            )
            self.assertEqual(
                compact.cpu().tolist(),
                [[100, 100, 100, 3, 1], [100, 100, 100, 3, 1]],
            )
            self.assertEqual(cursor.cpu().tolist(), [-1, -1])
            debug = debug_out.cpu().tolist()
            self.assertEqual(
                [GPU_DRAFT_TAIL_SELECT_REASON_NAMES[row[0]] for row in debug],
                ["direct", "direct"],
            )
            self.assertEqual([row[3:6] for row in debug], [[3, 3, 0], [3, 3, 0]])
            self.assertEqual([row[7] for row in debug], [0, 0])
        finally:
            mock_verifier.close()

    def test_critical_commit_and_reseat_are_device_selected(self):
        self._open("old", request_epoch=5)
        self._publish("old", range(7), token_base=10)

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.full_like(gpu_seats, 5)
        seq_lens = torch.tensor([2], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([10], dtype=torch.int32, device="cuda:0")

        def apply_commit(pre_verify_output_len, accept_tokens):
            stride = 4
            self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
                gpu_seats,
                expected_request_epochs,
                torch.tensor(
                    [pre_verify_output_len + 1],
                    dtype=torch.int64,
                    device="cuda:0",
                ),
                torch.tensor(
                    [*accept_tokens, *([0] * (stride - len(accept_tokens)))],
                    dtype=torch.int32,
                    device="cuda:0",
                ),
                torch.tensor([len(accept_tokens)], dtype=torch.int32, device="cuda:0"),
                accept_token_stride=stride,
            )

        apply_commit(0, [10])
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[11, 12, 13, 3, 1],
            expected_cursor=1,
        )

        apply_commit(1, [11])
        seq_lens.fill_(3)
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[12, 13, 14, 3, 1],
            expected_cursor=2,
        )

        apply_commit(2, [99])
        seq_lens.fill_(4)
        bonus_tokens.fill_(99)
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[0, 0, 0, 0, 0],
            expected_cursor=3,
        )

        self._send_raw_tail(
            request_id="old",
            base_committed_len=3,
            start_token_pos=2,
            tokens=99,
            frame_seq=450,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)
        self._send_raw_tail(
            request_id="old",
            base_committed_len=3,
            start_token_pos=3,
            tokens=100,
            frame_seq=451,
        )
        self._wait_for_result_frames(1)
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
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=self._capture_request_epochs(gpu_seats),
        )
        self.assertEqual(compact.cpu().tolist(), [[0, 0, 0, 0, 0]])
        self.assertEqual(cursor.cpu().tolist(), [-1])

    def test_first_extend_commit_waits_for_landing_open(self):
        gate_stream = torch.cuda.Stream(device="cuda:0")
        landing_gate = torch.cuda.Event()
        with torch.cuda.stream(gate_stream):
            torch.cuda._sleep(500_000_000)
            landing_gate.record()
        self.landing_stream.wait_event(landing_gate)

        self._open("gated-open", request_epoch=6)
        self.assertFalse(landing_gate.query())
        tail = self.verifier.gpu_tail_buffer
        open_event = torch.cuda.Event()
        open_event.record(self.landing_stream)
        forward_stream = torch.cuda.Stream(device="cuda:0")
        done = torch.cuda.Event()
        with torch.cuda.stream(forward_stream):
            tail.wait_for_landing_event(open_event)
            gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
            tail.apply_verify_commit_from_device(
                gpu_seats,
                torch.full_like(gpu_seats, 6),
                None,
                torch.tensor([90, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
                torch.ones(1, dtype=torch.int32, device="cuda:0"),
                accept_token_stride=4,
            )
            done.record()

        done.synchronize()
        self.assertEqual(int(tail.request_epochs[1].cpu().item()), 6)
        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.error_codes[1].cpu().item()), 0)

    def test_landing_event_orders_remote_append_before_selector(self):
        self._open("landing-order", request_epoch=19, committed_outputs=(10,))
        torch.cuda.synchronize()

        # Hold the landing stream after transport ingress. The forward-stream
        # event wait must be the only reason the selector observes the append.
        gate_stream = torch.cuda.Stream(device="cuda:0")
        landing_gate = torch.cuda.Event()
        with torch.cuda.stream(gate_stream):
            torch.cuda._sleep(500_000_000)
            landing_gate.record()
        self.landing_stream.wait_event(landing_gate)
        self._publish(
            "landing-order",
            range(1, 5),
            token_base=10,
            base_committed_lens=[1] * 4,
        )

        self._wait_for_result_frames(1)
        self.assertFalse(landing_gate.query())

        tail = self.verifier.gpu_tail_buffer
        landing_event = torch.cuda.Event()
        landing_event.record(self.landing_stream)
        forward_stream = torch.cuda.Stream(device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        with torch.cuda.stream(forward_stream):
            tail.wait_for_landing_event(landing_event)
            gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
            compact, cursor = tail.select_snapshot(
                gpu_seats,
                torch.tensor([2], dtype=torch.int64, device="cuda:0"),
                torch.tensor([10], dtype=torch.int32, device="cuda:0"),
                request_epochs=torch.full_like(gpu_seats, 19),
                debug_out=debug_out,
            )
        forward_stream.synchronize()

        self.assertEqual(
            compact.cpu().tolist(),
            [[11, 12, 13, 3, 1]],
            debug_out.cpu().tolist(),
        )
        self.assertEqual(cursor.cpu().tolist(), [1])

    def test_stale_critical_commit_does_not_mutate_reseated_row(self):
        self._open("stale-critical-old", request_epoch=7)
        torch.cuda.synchronize()

        old_gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        old_expected_request_epochs = torch.full_like(old_gpu_seats, 7)
        old_accept_tokens = torch.tensor(
            [90, 0, 0, 0], dtype=torch.int32, device="cuda:0"
        )
        old_num_accept_tokens = torch.ones(1, dtype=torch.int32, device="cuda:0")

        self.verifier.close_request(
            DraftClose(
                request_id="stale-critical-old",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                reason="reseated",
            )
        )
        self._open("stale-critical-new", request_epoch=8)
        tail = self.verifier.gpu_tail_buffer
        open_event = torch.cuda.Event()
        open_event.record(self.landing_stream)
        tail.wait_for_landing_event(open_event)
        torch.cuda.current_stream().synchronize()

        stale_compact, stale_cursor = tail.select_snapshot(
            old_gpu_seats,
            torch.tensor([1], dtype=torch.int64, device="cuda:0"),
            torch.tensor([0], dtype=torch.int32, device="cuda:0"),
            request_epochs=old_expected_request_epochs,
        )
        self.assertEqual(stale_compact.cpu().tolist(), [[0, 0, 0, 0, 0]])
        self.assertEqual(stale_cursor.cpu().tolist(), [-1])

        tail.apply_verify_commit_from_device(
            old_gpu_seats,
            old_expected_request_epochs,
            None,
            old_accept_tokens,
            old_num_accept_tokens,
            accept_token_stride=4,
        )
        torch.cuda.synchronize()

        self.assertEqual(int(tail.request_epochs[1].cpu().item()), 8)
        self.assertEqual(int(tail.prompt_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.can_accept_prefix_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.raw_tail_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.consumable_tail_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.error_codes[1].cpu().item()), 0)
        self.assertEqual(int(tail.versions[1].cpu().item()) % 2, 0)
        fresh_compact, fresh_cursor = tail.select_snapshot(
            old_gpu_seats,
            torch.tensor([1], dtype=torch.int64, device="cuda:0"),
            torch.tensor([0], dtype=torch.int32, device="cuda:0"),
            request_epochs=torch.full_like(old_gpu_seats, 8),
        )
        self.assertEqual(fresh_compact.cpu().tolist(), [[0, 0, 0, 0, 1]])
        self.assertEqual(fresh_cursor.cpu().tolist(), [0])

    def test_critical_commit_stream_handoff_orders_single_snapshot(self):
        self._open("stream-handoff", request_epoch=7)
        self._publish("stream-handoff", range(7), token_base=10)
        self._wait_for_result_frames(1)
        # Only setup is synchronized. The critical section below has no host
        # barrier between producing its inputs, applying the commit, and select.
        torch.cuda.synchronize()

        forward_stream = torch.cuda.Stream(device="cuda:0")
        gpu_seats = torch.empty(1, dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.empty(1, dtype=torch.int64, device="cuda:0")
        pre_verify_seq_lens = torch.empty(1, dtype=torch.int64, device="cuda:0")
        accept_tokens = torch.empty(4, dtype=torch.int32, device="cuda:0")
        num_accept_tokens = torch.empty(1, dtype=torch.int32, device="cuda:0")
        seq_lens = torch.empty(1, dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.empty(1, dtype=torch.int32, device="cuda:0")
        compact_out = torch.empty((1, 5), dtype=torch.int64, device="cuda:0")
        cursor_out = torch.empty(1, dtype=torch.int64, device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        compact_cpu = torch.empty((1, 5), dtype=torch.int64, pin_memory=True)
        cursor_cpu = torch.empty(1, dtype=torch.int64, pin_memory=True)
        debug_cpu = torch.empty((1, 10), dtype=torch.int64, pin_memory=True)
        done = torch.cuda.Event()

        with torch.cuda.stream(forward_stream):
            gpu_seats.fill_(1)
            expected_request_epochs.fill_(7)
            # prompt_len=2, so pre-verify seq_len=1 is output cursor zero.
            pre_verify_seq_lens.fill_(1)
            torch.arange(10, 14, out=accept_tokens)
            num_accept_tokens.fill_(2)
            # After committing [10, 11], seq_len=3 is logical output cursor two.
            seq_lens.fill_(3)
            bonus_tokens.zero_()

            self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
                gpu_seats,
                expected_request_epochs,
                pre_verify_seq_lens,
                accept_tokens,
                num_accept_tokens,
                accept_token_stride=4,
            )
            compact, cursor = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=expected_request_epochs,
                out=compact_out,
                out_cursor=cursor_out,
                debug_out=debug_out,
            )
            compact_cpu.copy_(compact, non_blocking=True)
            cursor_cpu.copy_(cursor, non_blocking=True)
            debug_cpu.copy_(debug_out, non_blocking=True)
            done.record()

        done.synchronize()
        debug = debug_cpu.tolist()[0]
        self.assertEqual(compact_cpu.tolist(), [[12, 13, 14, 3, 1]])
        self.assertEqual(cursor_cpu.tolist(), [2])
        self.assertEqual(GPU_DRAFT_TAIL_SELECT_REASON_NAMES[debug[0]], "direct")
        self.assertEqual(debug[2], 0)
        self.assertEqual(debug[6], 2)

    def test_landing_updates_and_forward_commit_linearize(self):
        self._open("late-binding", request_epoch=8)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.full_like(gpu_seats, 8)

        # First target-only commit creates one pending token.
        self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
            gpu_seats,
            expected_request_epochs,
            None,
            torch.tensor([90, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
            torch.tensor([1], dtype=torch.int32, device="cuda:0"),
            accept_token_stride=4,
        )
        torch.cuda.synchronize()

        forward_stream = torch.cuda.Stream(device="cuda:0")
        done = torch.cuda.Event()
        compact_cpu = torch.empty((1, 5), dtype=torch.int64, pin_memory=True)
        debug_cpu = torch.empty((1, 10), dtype=torch.int64, pin_memory=True)
        with torch.cuda.stream(forward_stream):
            # Stand in for a target verify long enough for landing-stream ACK
            # and draft updates to execute before the commit boundary.
            torch.cuda._sleep(200_000_000)
            self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
                gpu_seats,
                expected_request_epochs,
                torch.tensor([2], dtype=torch.int64, device="cuda:0"),
                torch.tensor([91, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
                torch.tensor([1], dtype=torch.int32, device="cuda:0"),
                accept_token_stride=4,
            )
            debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
            compact, _ = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats,
                torch.tensor([3], dtype=torch.int64, device="cuda:0"),
                torch.tensor([91], dtype=torch.int32, device="cuda:0"),
                request_epochs=expected_request_epochs,
                debug_out=debug_out,
            )
            compact_cpu.copy_(compact, non_blocking=True)
            debug_cpu.copy_(debug_out, non_blocking=True)
            done.record()

        self._send_raw_tail(
            request_id="late-binding",
            base_committed_len=1,
            start_token_pos=0,
            tokens=90,
            frame_seq=460,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)
        self._send_raw_tail(
            request_id="late-binding",
            base_committed_len=1,
            start_token_pos=1,
            tokens=91,
            frame_seq=461,
        )
        self._wait_for_result_frames(1)

        done.synchronize()
        debug = debug_cpu.tolist()[0]
        self.assertEqual(debug[6], 2)
        self.assertEqual(debug[7], 0)
        self.assertIn(
            GPU_DRAFT_TAIL_SELECT_REASON_NAMES[debug[0]],
            ("direct", "pending_prefix"),
        )

        # Whichever writer won the boundary, a cumulative ACK through the
        # current verifier cursor must settle it, after which new-base draft
        # output is consumable again.
        self._send_raw_tail(
            request_id="late-binding",
            base_committed_len=2,
            start_token_pos=1,
            tokens=91,
            frame_seq=462,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)
        self._send_raw_tail(
            request_id="late-binding",
            base_committed_len=2,
            start_token_pos=2,
            tokens=92,
            frame_seq=463,
        )
        self._wait_for_result_frames(1)
        self._wait_select(
            gpu_seats,
            torch.tensor([3], dtype=torch.int64, device="cuda:0"),
            torch.tensor([91], dtype=torch.int32, device="cuda:0"),
            expected_compact=[92, 0, 0, 1, 1],
            expected_cursor=2,
        )

    def test_direct_only_selector_rejects_unsettled_logical_cursor(self):
        self._open("unsettled", request_epoch=7)
        self._publish("unsettled", range(7), token_base=10)
        self._wait_for_result_frames(1)

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        compact, cursor = self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats,
            # prompt_len=2, so seq_len=4 means logical output length three.
            torch.tensor([4], dtype=torch.int64, device="cuda:0"),
            torch.tensor([12], dtype=torch.int32, device="cuda:0"),
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        compact_values = compact.cpu().tolist()
        cursor_values = cursor.cpu().tolist()
        debug = debug_out.cpu().tolist()[0]

        self.assertEqual(compact_values, [[0, 0, 0, 0, 0]])
        self.assertEqual(cursor_values, [3])
        self.assertEqual(
            GPU_DRAFT_TAIL_SELECT_REASON_NAMES[debug[0]],
            "logical_cursor_mismatch",
        )
        self.assertEqual(debug[2], 3)

    def test_critical_extend_commit_fences_old_tail(self):
        self._open("bootstrap", request_epoch=8)
        self._publish("bootstrap", range(4), token_base=10)
        self._wait_for_result_frames(1)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.full_like(gpu_seats, 8)

        # Extend has no pre-verify seq-lens input. A mismatching accepted token
        # becomes pending until the drafter confirms it at the current cursor.
        self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
            gpu_seats,
            expected_request_epochs,
            None,
            torch.tensor([99, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
            torch.tensor([1], dtype=torch.int32, device="cuda:0"),
            accept_token_stride=4,
        )
        # A real peer can ACK only after the verifier result was resolved and
        # sent. Preserve that causal order for this synthetic immediate reply.
        torch.cuda.current_stream().synchronize()
        self._send_raw_tail(
            request_id="bootstrap",
            base_committed_len=1,
            start_token_pos=0,
            tokens=99,
            frame_seq=401,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)
        self._send_raw_tail(
            request_id="bootstrap",
            base_committed_len=1,
            start_token_pos=1,
            tokens=101,
            frame_seq=402,
        )
        self._wait_for_result_frames(1)

        self._wait_select(
            gpu_seats,
            torch.tensor([2], dtype=torch.int64, device="cuda:0"),
            torch.tensor([99], dtype=torch.int32, device="cuda:0"),
            expected_compact=[101, 0, 0, 1, 1],
            expected_cursor=1,
        )

    def test_critical_commit_pending_count_uses_cumulative_echo(self):
        self._open("critical-pending", request_epoch=9)
        self._publish("critical-pending", range(3), token_base=10)
        self._wait_for_result_frames(1)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.full_like(gpu_seats, 9)

        def commit(pre_verify_output_len, token):
            self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
                gpu_seats,
                expected_request_epochs,
                torch.tensor(
                    [pre_verify_output_len + 1],
                    dtype=torch.int64,
                    device="cuda:0",
                ),
                torch.tensor([token, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
                torch.tensor([1], dtype=torch.int32, device="cuda:0"),
                accept_token_stride=4,
            )

        # Both target tokens mismatch the resident draft prefix. The verifier
        # cursor advances authoritatively while the pending count grows.
        commit(0, 90)
        commit(1, 91)
        torch.cuda.synchronize()
        tail = self.verifier.gpu_tail_buffer
        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 2)

        # An output starting after the confirmed cursor cannot clear pending.
        self._send_raw_tail(
            request_id="critical-pending",
            base_committed_len=2,
            start_token_pos=2,
            tokens=92,
            frame_seq=501,
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 2)

        # Cumulative echoes are idempotent and do not advance committed_len.
        for frame_seq, ack_len, token in (
            (502, 1, 90),
            (503, 1, 90),
            (504, 2, 91),
        ):
            self._send_raw_tail(
                request_id="critical-pending",
                base_committed_len=ack_len,
                start_token_pos=ack_len - 1,
                tokens=token,
                frame_seq=frame_seq,
                is_commit_echo=True,
            )
            self._wait_for_result_frames(1)

        self._send_raw_tail(
            request_id="critical-pending",
            base_committed_len=2,
            start_token_pos=2,
            tokens=92,
            frame_seq=505,
        )
        self._wait_for_result_frames(1)
        self._wait_select(
            gpu_seats,
            # prompt_len=2, so seq_len=3 is logical output cursor two.
            torch.tensor([3], dtype=torch.int64, device="cuda:0"),
            torch.tensor([91], dtype=torch.int32, device="cuda:0"),
            expected_compact=[92, 0, 0, 1, 1],
            expected_cursor=2,
        )

    def test_echo_and_retained_span_land_from_one_frame(self):
        self._open("echo-span", request_epoch=12)
        self._publish("echo-span", range(2), token_base=10)
        self._wait_for_result_frames(1)
        self.verifier.commit(
            VerifyCommit(
                request_id="echo-span",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[10, 11, 12],
            )
        )

        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="echo-span",
                        base_committed_len=3,
                        start_token_pos=2,
                        tokens=(12,),
                        is_commit_echo=True,
                    ),
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="echo-span",
                        base_committed_len=3,
                        start_token_pos=3,
                        tokens=(13, 14),
                    ),
                ]
            )
        )
        self._wait_for_result_frames(1)

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        self._wait_select(
            gpu_seats,
            torch.tensor([4], dtype=torch.int64, device="cuda:0"),
            torch.tensor([12], dtype=torch.int32, device="cuda:0"),
            expected_compact=[13, 14, 0, 2, 1],
            expected_cursor=3,
        )

    def test_stale_echo_span_reconciles_residual_pending_prefix(self):
        self._open("stale-echo-span", request_epoch=13)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.full_like(gpu_seats, 13)
        for pre_verify_output_len, token in ((0, 10), (1, 11)):
            self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
                gpu_seats,
                expected_request_epochs,
                torch.tensor(
                    [pre_verify_output_len + 1],
                    dtype=torch.int64,
                    device="cuda:0",
                ),
                torch.tensor([token, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
                torch.tensor([1], dtype=torch.int32, device="cuda:0"),
                accept_token_stride=4,
            )
        torch.cuda.synchronize()

        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="stale-echo-span",
                        base_committed_len=1,
                        start_token_pos=0,
                        tokens=(10,),
                        is_commit_echo=True,
                    ),
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="stale-echo-span",
                        base_committed_len=1,
                        start_token_pos=1,
                        tokens=(11, 12),
                    ),
                ]
            )
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()

        tail = self.verifier.gpu_tail_buffer
        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.raw_tail_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.consumable_tail_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.can_accept_prefix_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.tail_tokens[1, 0].cpu().item()), 12)
        self.assertEqual(
            int(tail.pending_prefix_fast_forward_cts[1].cpu().item()), 1
        )

        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        compact, cursor = tail.select_snapshot(
            gpu_seats,
            torch.tensor([3], dtype=torch.int64, device="cuda:0"),
            torch.tensor([11], dtype=torch.int32, device="cuda:0"),
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        self.assertEqual(compact.cpu().tolist(), [[12, 0, 0, 1, 1]])
        self.assertEqual(cursor.cpu().tolist(), [2])
        debug = debug_out.cpu().tolist()[0]
        self.assertEqual(GPU_DRAFT_TAIL_SELECT_REASON_NAMES[debug[0]], "direct")
        self.assertEqual(debug[3:7], [1, 1, 0, 2])
        self.assertEqual(debug[7], 0)
        self.assertEqual(debug[9], 1)

    def test_stale_echo_span_must_match_residual_pending_value(self):
        self._open("stale-echo-mismatch", request_epoch=14)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        expected_request_epochs = torch.full_like(gpu_seats, 14)
        for pre_verify_output_len, token in ((0, 10), (1, 11)):
            self.verifier.gpu_tail_buffer.apply_verify_commit_from_device(
                gpu_seats,
                expected_request_epochs,
                torch.tensor(
                    [pre_verify_output_len + 1],
                    dtype=torch.int64,
                    device="cuda:0",
                ),
                torch.tensor([token, 0, 0, 0], dtype=torch.int32, device="cuda:0"),
                torch.tensor([1], dtype=torch.int32, device="cuda:0"),
                accept_token_stride=4,
            )
        torch.cuda.synchronize()

        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="stale-echo-mismatch",
                        base_committed_len=1,
                        start_token_pos=0,
                        tokens=(10,),
                        is_commit_echo=True,
                    ),
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="stale-echo-mismatch",
                        base_committed_len=1,
                        start_token_pos=1,
                        tokens=(99, 12),
                    ),
                ]
            )
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()

        tail = self.verifier.gpu_tail_buffer
        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.raw_tail_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.consumable_tail_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.pending_expected_tokens[1, 1].cpu().item()), 11)
        self.assertEqual(
            int(tail.pending_prefix_fast_forward_cts[1].cpu().item()), 0
        )

    def test_partial_pending_match_recovers_after_corrected_echo_span(self):
        self._open("partial-pending-recovery", request_epoch=16)
        for committed_len, token in enumerate((10, 11, 12)):
            self.verifier.commit(
                VerifyCommit(
                    request_id="partial-pending-recovery",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=committed_len,
                    committed_tokens=[token],
                )
            )

        tail = self.verifier.gpu_tail_buffer
        deadline = time.monotonic() + 3.0
        while int(tail.pending_expected_lens[1].cpu().item()) != 3:
            if time.monotonic() >= deadline:
                self.fail("Pending prefix did not reach three tokens")
            time.sleep(0.001)

        # Confirm position zero, then hit a real mismatch at position one.
        self._send_raw_tail(
            request_id="partial-pending-recovery",
            base_committed_len=0,
            start_token_pos=0,
            tokens=(10, 99, 13),
            frame_seq=601,
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.can_accept_prefix_lens[1].cpu().item()), 2)
        self.assertEqual(int(tail.raw_tail_lens[1].cpu().item()), 0)

        # The corrected cumulative ACK and its retained span land together.
        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="partial-pending-recovery",
                        base_committed_len=2,
                        start_token_pos=1,
                        tokens=(11,),
                        is_commit_echo=True,
                    ),
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="partial-pending-recovery",
                        base_committed_len=2,
                        start_token_pos=2,
                        tokens=(12, 13),
                    ),
                ]
            )
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()

        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 3)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.raw_tail_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.consumable_tail_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.can_accept_prefix_lens[1].cpu().item()), 3)
        self.assertEqual(int(tail.tail_tokens[1, 0].cpu().item()), 13)
        self.assertEqual(
            int(tail.pending_prefix_fast_forward_cts[1].cpu().item()), 2
        )
        self.assertEqual(int(tail.error_codes[1].cpu().item()), 0)

    def test_tail_capacity_is_a_fail_fast_invariant(self):
        self._open("overflow", request_epoch=1)
        self._publish("overflow", range(6), token_base=20)
        self._wait_for_result_frames(1)
        # The second span crosses the row capacity. The landing kernel must
        # reject the whole span rather than publishing its first token.
        self._publish("overflow", range(6, 8), token_base=20)

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        seq_lens = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([0], dtype=torch.int32, device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            compact, _ = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=self._capture_request_epochs(gpu_seats),
                debug_out=debug_out,
            )
            debug_values = debug_out.cpu().tolist()[0]
            if debug_values[7] != 0:
                break
            time.sleep(0.001)
        else:
            self.fail("Overflow tail did not latch a GPU protocol error")

        self.assertEqual(compact.cpu().tolist(), [[0, 0, 0, 0, 0]])
        self.assertEqual(
            GPU_DRAFT_TAIL_SELECT_REASON_NAMES[debug_values[0]],
            "metadata_invalid",
        )
        self.assertEqual(debug_values[3:5], [6, 6])
        self.assertEqual(debug_values[7], 9)  # tail_capacity_exceeded
        self.assertGreater(debug_values[8], 0)
        self.assertGreater(
            self.verifier.gpu_tail_buffer.error_op_seqs[1].cpu().item(), 0
        )

    def test_future_commit_echo_is_a_protocol_error(self):
        self._open("ack-ahead", request_epoch=10)
        self.verifier.commit(
            VerifyCommit(
                request_id="ack-ahead",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[90],
            )
        )
        self._send_raw_tail(
            request_id="ack-ahead",
            base_committed_len=2,
            start_token_pos=1,
            tokens=91,
            frame_seq=201,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        deadline = time.monotonic() + 3.0
        while True:
            compact, _ = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats,
                torch.tensor([2], dtype=torch.int64, device="cuda:0"),
                torch.tensor([90], dtype=torch.int32, device="cuda:0"),
                request_epochs=self._capture_request_epochs(gpu_seats),
                debug_out=debug_out,
            )
            debug = debug_out.cpu().tolist()[0]
            if debug[7] != 0:
                break
            if time.monotonic() >= deadline:
                self.fail(f"Future commit echo was not rejected: {debug}")
        self.assertEqual(compact.cpu().tolist(), [[0, 0, 0, 0, 0]])
        self.assertEqual(debug[7], 14)  # commit_ack_ahead

    def test_pending_count_can_exceed_tail_capacity(self):
        self._open("lagging", request_epoch=1)
        for committed_len in range(8):
            self.verifier.commit(
                VerifyCommit(
                    request_id="lagging",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=committed_len,
                    committed_tokens=[100 + committed_len],
                )
            )

        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        # prompt_len=2 and seq_len=9 gives logical output length eight.
        seq_lens = torch.tensor([9], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([107], dtype=torch.int32, device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")
        deadline = time.monotonic() + 3.0
        while True:
            compact, cursor = self.verifier.gpu_tail_buffer.select_snapshot(
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=self._capture_request_epochs(gpu_seats),
                debug_out=debug_out,
            )
            compact_values = compact.cpu().tolist()
            cursor_values = cursor.cpu().tolist()
            debug_values = debug_out.cpu().tolist()[0]
            if debug_values[5] == 8:
                break
            if time.monotonic() >= deadline:
                self.fail(f"Pending count did not reach eight: {debug_values}")
            time.sleep(0.001)

        self.assertEqual(compact_values, [[0, 0, 0, 0, 0]])
        self.assertEqual(cursor_values, [8])
        self.assertEqual(debug_values[5], 8)
        self.assertEqual(debug_values[7], 0)
        self.assertEqual(debug_values[9], 0)
        self.assertEqual(
            GPU_DRAFT_TAIL_SELECT_REASON_NAMES[debug_values[0]],
            "pending_prefix",
        )

        # One cumulative echo clears all eight pending tokens. The bounded ring
        # retains only their newest tail-capacity suffix.
        self._send_raw_tail(
            request_id="lagging",
            base_committed_len=8,
            start_token_pos=7,
            tokens=107,
            frame_seq=100,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)
        self._send_raw_tail(
            request_id="lagging",
            base_committed_len=8,
            start_token_pos=8,
            tokens=108,
            frame_seq=101,
        )
        self._wait_for_result_frames(1)
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[108, 0, 0, 1, 1],
            expected_cursor=8,
        )

    def test_overflowed_pending_ring_reconciles_after_partial_ack(self):
        self._open("lagging-reconcile", request_epoch=15)
        for committed_len in range(8):
            self.verifier.commit(
                VerifyCommit(
                    request_id="lagging-reconcile",
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    pre_verify_committed_len=committed_len,
                    committed_tokens=[100 + committed_len],
                )
            )

        tail = self.verifier.gpu_tail_buffer
        deadline = time.monotonic() + 3.0
        while int(tail.pending_expected_lens[1].cpu().item()) != 8:
            if time.monotonic() >= deadline:
                self.fail("Pending prefix did not exceed the ring capacity")
            time.sleep(0.001)

        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="lagging-reconcile",
                        base_committed_len=2,
                        start_token_pos=1,
                        tokens=(101,),
                        is_commit_echo=True,
                    ),
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id="lagging-reconcile",
                        base_committed_len=2,
                        start_token_pos=2,
                        tokens=(102, 103, 104, 105, 106, 107, 108),
                    ),
                ]
            )
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()

        self.assertEqual(int(tail.committed_lens[1].cpu().item()), 8)
        self.assertEqual(int(tail.pending_expected_lens[1].cpu().item()), 0)
        self.assertEqual(int(tail.raw_tail_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.consumable_tail_lens[1].cpu().item()), 1)
        self.assertEqual(int(tail.tail_tokens[1, 0].cpu().item()), 108)
        self.assertEqual(
            int(tail.pending_prefix_fast_forward_cts[1].cpu().item()), 1
        )

    def test_publish_sequence_tracks_drafter_arrivals_not_verifier_commits(self):
        self._open("arrival-seq", request_epoch=10)
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        seq_lens = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([0], dtype=torch.int32, device="cuda:0")
        debug_out = torch.empty((1, 10), dtype=torch.int64, device="cuda:0")

        torch.cuda.synchronize()
        self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        self.assertEqual(debug_out.cpu().tolist()[0][1], -1)

        self._publish("arrival-seq", [0], token_base=100)
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        first_arrival_seq = debug_out.cpu().tolist()[0][1]
        self.assertGreater(first_arrival_seq, 0)

        self.verifier.commit(
            VerifyCommit(
                request_id="arrival-seq",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                pre_verify_committed_len=0,
                committed_tokens=[100],
            )
        )
        seq_lens.fill_(2)
        bonus_tokens.fill_(100)
        torch.cuda.synchronize()
        self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        self.assertEqual(debug_out.cpu().tolist()[0][1], first_arrival_seq)

        self._send_raw_tail(
            request_id="arrival-seq",
            base_committed_len=1,
            start_token_pos=0,
            tokens=100,
            frame_seq=101,
            is_commit_echo=True,
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        self.assertEqual(debug_out.cpu().tolist()[0][1], first_arrival_seq)

        self._send_raw_tail(
            request_id="arrival-seq",
            base_committed_len=1,
            start_token_pos=1,
            tokens=101,
            frame_seq=102,
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.verifier.gpu_tail_buffer.select_snapshot(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=self._capture_request_epochs(gpu_seats),
            debug_out=debug_out,
        )
        self.assertGreater(debug_out.cpu().tolist()[0][1], first_arrival_seq)

    def test_remote_only_commit_does_not_double_apply_gpu_tail(self):
        self._open("remote-only", request_epoch=11)
        self._publish("remote-only", [0], token_base=100)
        self._wait_for_result_frames(1)
        commit = VerifyCommit(
            request_id="remote-only",
            src_verifier_rank=0,
            dst_drafter_rank=0,
            pre_verify_committed_len=0,
            committed_tokens=[100],
        )

        self.verifier.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=0,
                verify_commit_messages=[commit],
            ),
            apply_local_verify_commits=False,
        )
        torch.cuda.synchronize()

        self.assertEqual(
            int(self.verifier.gpu_tail_buffer.committed_lens[1].cpu().item()), 0
        )
        self.assertEqual(
            self.verifier.gpu_tail_buffer.tail_tokens[1, :1].cpu().tolist(), [100]
        )
        deadline = time.monotonic() + 3.0
        remote_batches = []
        while not remote_batches and time.monotonic() < deadline:
            remote_batches = self.drafter.drain_controls()
            if not remote_batches:
                time.sleep(0.001)
        self.assertTrue(remote_batches, "Remote-only VerifyCommit was not delivered")
        self.assertEqual(remote_batches[0].verify_commit_messages, [commit])

    def test_open_and_append_have_both_linearized_terminal_states(self):
        observed_raw_lens = []
        for append_first in (True, False):
            request_epoch = 20 + int(append_first)
            request_id = f"open-append-{request_epoch}"
            self.verifier.gpu_tail_buffer.bind_request(request_id, 1, request_epoch)
            sync = DraftSync(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=[],
            )
            first_done = threading.Event()
            errors = []

            def apply_open():
                try:
                    if append_first:
                        first_done.wait(timeout=2.0)
                    self.verifier.submit_control_batch(
                        DraftControlBatch(dst_drafter_rank=0, sync_messages=[sync])
                    )
                    if not append_first:
                        first_done.set()
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            def apply_append():
                try:
                    if not append_first:
                        first_done.wait(timeout=2.0)
                    self._send_raw_tail(
                        request_id=request_id,
                        base_committed_len=0,
                        start_token_pos=0,
                        tokens=100 + request_epoch,
                        frame_seq=200 + request_epoch,
                    )
                    self._wait_for_result_frames(1)
                    if append_first:
                        first_done.set()
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            open_thread = threading.Thread(target=apply_open)
            append_thread = threading.Thread(target=apply_append)
            open_thread.start()
            append_thread.start()
            open_thread.join(timeout=3.0)
            append_thread.join(timeout=3.0)
            self.assertFalse(open_thread.is_alive())
            self.assertFalse(append_thread.is_alive())
            self.assertEqual(errors, [])
            torch.cuda.synchronize()
            raw_tail_len = int(
                self.verifier.gpu_tail_buffer.raw_tail_lens[1].cpu().item()
            )
            observed_raw_lens.append(raw_tail_len)
            self.assertEqual(
                int(self.verifier.gpu_tail_buffer.error_codes[1].cpu().item()), 0
            )
            self.verifier.close_request(
                DraftClose(
                    request_id=request_id,
                    src_verifier_rank=0,
                    dst_drafter_rank=0,
                    reason="test",
                )
            )

        self.assertEqual(observed_raw_lens, [0, 1])

    def test_reseat_drops_late_old_epoch_and_accepts_new_epoch(self):
        self._open("old-epoch", request_epoch=30)
        self.verifier.close_request(
            DraftClose(
                request_id="old-epoch",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                reason="reseat",
            )
        )
        self.verifier.open_request(
            DraftSync(
                request_id="new-epoch",
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=[],
            ),
            gpu_seat=1,
            request_epoch=31,
        )
        self._send_raw_tail(
            request_id="old-epoch",
            base_committed_len=0,
            start_token_pos=0,
            tokens=999,
            frame_seq=300,
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.assertEqual(
            int(self.verifier.gpu_tail_buffer.request_epochs[1].cpu().item()), 31
        )
        self.assertEqual(
            int(self.verifier.gpu_tail_buffer.raw_tail_lens[1].cpu().item()), 0
        )

        self._send_raw_tail(
            request_id="new-epoch",
            base_committed_len=0,
            start_token_pos=0,
            tokens=31,
            frame_seq=301,
        )
        self._wait_for_result_frames(1)
        torch.cuda.synchronize()
        self.assertEqual(
            self.verifier.gpu_tail_buffer.tail_tokens[1, :1].cpu().tolist(), [31]
        )

    def test_gpu_state_machine_matches_cpu_reference_random_sequence(self):
        rng = random.Random(20260824)
        cpu_reference = DraftTailBuffer(verifier_rank=0)
        request_epoch = 40
        request_id = f"random-{request_epoch}"
        pending_echo = None

        def open_request():
            message = DraftSync(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=[],
            )
            cpu_reference.open_request(message)
            self.verifier.open_request(message, gpu_seat=1, request_epoch=request_epoch)

        def assert_matches_reference():
            torch.cuda.synchronize()
            snapshot = cpu_reference.snapshot(request_id, max_tail_len=32)
            gpu_tail = self.verifier.gpu_tail_buffer
            self.assertEqual(
                int(gpu_tail.committed_lens[1].cpu().item()),
                snapshot.committed_len,
            )
            self.assertEqual(
                int(gpu_tail.raw_tail_lens[1].cpu().item()),
                snapshot.raw_tail_len,
            )
            self.assertEqual(
                int(gpu_tail.consumable_tail_lens[1].cpu().item()),
                snapshot.num_consumable_drafts,
            )
            self.assertEqual(
                int(gpu_tail.pending_expected_lens[1].cpu().item()),
                0 if pending_echo is None else 1,
            )
            self.assertEqual(
                gpu_tail.tail_tokens[1, : snapshot.raw_tail_len].cpu().tolist(),
                list(snapshot.tail_tokens),
            )
            self.assertEqual(int(gpu_tail.error_codes[1].cpu().item()), 0)

        try:
            open_request()
            for step in range(200):
                if step > 0 and step % 50 == 0:
                    close = DraftClose(
                        request_id=request_id,
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        reason="random-reseat",
                    )
                    cpu_reference.close_request(close)
                    self.verifier.close_request(close)
                    request_epoch += 1
                    request_id = f"random-{request_epoch}"
                    pending_echo = None
                    open_request()

                snapshot = cpu_reference.snapshot(request_id, max_tail_len=32)
                if pending_echo is not None:
                    ack_len, ack_token = pending_echo
                    output = DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id=request_id,
                        base_committed_len=ack_len,
                        start_token_pos=ack_len - 1,
                        tokens=(ack_token,),
                        is_commit_echo=True,
                    )
                    pending_echo = None
                    cpu_reference.append_draft_stream_batch(
                        DraftTailStreamOutputBatch(outputs=[output])
                    )
                    self._send_raw_tail_output(output, frame_seq=400 + step)
                    self._wait_for_result_frames(1)
                elif snapshot.raw_tail_len == 0 or (
                    snapshot.raw_tail_len < 5 and rng.random() < 0.6
                ):
                    token = rng.randrange(100, 10000)
                    output = DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id=request_id,
                        base_committed_len=snapshot.committed_len,
                        start_token_pos=(
                            snapshot.committed_len + snapshot.raw_tail_len
                        ),
                        tokens=(token,),
                    )
                    cpu_reference.append_draft_stream_batch(
                        DraftTailStreamOutputBatch(outputs=[output])
                    )
                    self._send_raw_tail_output(output, frame_seq=400 + step)
                    self._wait_for_result_frames(1)
                else:
                    if rng.random() < 0.75:
                        num_commit_tokens = rng.randint(
                            1, min(2, snapshot.raw_tail_len)
                        )
                        committed_tokens = list(
                            snapshot.tail_tokens[:num_commit_tokens]
                        )
                    else:
                        mismatch = (int(snapshot.tail_tokens[0]) + 1) % 10000
                        if mismatch == int(snapshot.tail_tokens[0]):
                            mismatch += 1
                        committed_tokens = [mismatch]
                        pending_echo = (
                            snapshot.committed_len + len(committed_tokens),
                            mismatch,
                        )
                    commit = VerifyCommit(
                        request_id=request_id,
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=snapshot.committed_len,
                        committed_tokens=committed_tokens,
                    )
                    cpu_reference.apply_verify_commit(commit)
                    self.verifier.commit(commit)

                assert_matches_reference()
        finally:
            cpu_reference.close()

    def test_selector_spins_until_per_request_writer_releases(self):
        self._open("spin", request_epoch=1)
        torch.cuda.synchronize()
        tail = self.verifier.gpu_tail_buffer
        gpu_seats = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        request_epochs = self._capture_request_epochs(gpu_seats)
        seq_lens = torch.tensor([1], dtype=torch.int64, device="cuda:0")
        bonus_tokens = torch.tensor([0], dtype=torch.int32, device="cuda:0")
        self._wait_select(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            expected_compact=[0, 0, 0, 0, 1],
            expected_cursor=0,
        )
        version_before = int(tail.versions[1].cpu().item())
        self.assertEqual(version_before % 2, 0)

        writer_stream = torch.cuda.Stream(device="cuda:0")
        writer_active = torch.cuda.Event()
        with torch.cuda.stream(writer_stream):
            tail.versions[1].fill_(version_before + 1)
            writer_active.record()
            torch.cuda._sleep(500_000_000)
            tail.versions[1].fill_(version_before + 2)
        writer_active.synchronize()

        debug_out = torch.empty(
            (1, len(GPU_DRAFT_TAIL_DEBUG_FIELD_NAMES)),
            dtype=torch.int64,
            device="cuda:0",
        )
        compact, cursor = tail.select_snapshot(
            gpu_seats,
            seq_lens,
            bonus_tokens,
            request_epochs=request_epochs,
            debug_out=debug_out,
        )

        self.assertEqual(compact.cpu().tolist(), [[0, 0, 0, 0, 1]])
        self.assertEqual(cursor.cpu().tolist(), [0])
        retry_index = GPU_DRAFT_TAIL_DEBUG_FIELD_NAMES.index("seqlock_retries")
        self.assertGreater(int(debug_out[0, retry_index].cpu().item()), 0)
        self.assertEqual(int(tail.versions[1].cpu().item()), version_before + 2)

    def test_concurrent_landing_select_spins_without_exposing_a_torn_tail(self):
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
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=self._capture_request_epochs(gpu_seats),
            )
            values = compact.cpu().tolist()[0]
            cursor_value = cursor.cpu().tolist()[0]
            selected_len, row_valid = values[-2:]
            self.assertEqual(row_valid, 1)
            self.assertEqual(cursor_value, 0)
            self.assertIn(selected_len, (0, 1, 2, 3))
            self.assertEqual(values[:selected_len], [100, 101, 102][:selected_len])
            self.assertEqual(values[selected_len:3], [0] * (3 - selected_len))
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

    def _open(
        self,
        request_id: str,
        *,
        request_epoch: int,
        committed_outputs: tuple[int, ...] = (),
    ) -> None:
        self.verifier.open_request(
            DraftSync(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=list(committed_outputs),
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
        self.assertTrue(positions)
        self.assertEqual(
            positions,
            list(range(positions[0], positions[0] + len(positions))),
        )
        self.assertEqual(len(set(base_committed_lens)), 1)
        self.drafter.publish_tails(
            DraftTailStreamOutputBatch(
                outputs=[
                    DraftTailStreamOutput(
                        src_drafter_rank=0,
                        dst_verifier_rank=0,
                        request_id=request_id,
                        base_committed_len=int(base_committed_lens[0]),
                        start_token_pos=int(positions[0]),
                        tokens=tuple(
                            token_base + int(position) for position in positions
                        ),
                    )
                ]
            )
        )

    def _send_raw_tail(
        self,
        *,
        request_id: str,
        base_committed_len: int,
        start_token_pos: int,
        tokens: int | tuple[int, ...],
        frame_seq: int,
        is_commit_echo: bool = False,
    ) -> None:
        if isinstance(tokens, int):
            tokens = (tokens,)
        request_bytes = request_id.encode()
        frame = b"".join(
            (
                struct.pack("<4sBBqqqq", b"DSC1", 5, 2, frame_seq, 0, 0, 0),
                struct.pack("<IiiI", 1, 0, 0, len(request_bytes)),
                request_bytes,
                struct.pack(
                    "<qqI", base_committed_len, start_token_pos, len(tokens)
                ),
                struct.pack(f"<{len(tokens)}i", *tokens),
                struct.pack("<B", int(is_commit_echo)),
            )
        )
        socket = self.context.socket(zmq.PUSH)
        try:
            socket.connect(self.verifier_endpoint)
            socket.send(frame)
        finally:
            socket.close(linger=0)

    def _send_raw_tail_output(
        self, output: DraftTailStreamOutput, *, frame_seq: int
    ) -> None:
        self._send_raw_tail(
            request_id=str(output.request_id),
            base_committed_len=int(output.base_committed_len),
            start_token_pos=int(output.start_token_pos),
            tokens=tuple(output.tokens),
            frame_seq=frame_seq,
            is_commit_echo=bool(output.is_commit_echo),
        )

    def _wait_for_result_frames(self, num_frames: int) -> None:
        deadline = time.monotonic() + 3.0
        observed = 0
        while observed < num_frames:
            observed += int(
                self.verifier.take_transport_metrics()["num_draft_result_frames"]
            )
            if observed >= num_frames:
                return
            if time.monotonic() >= deadline:
                self.fail(
                    "Timed out waiting for verifier draft-result frames: "
                    f"expected={num_frames} observed={observed}"
                )
            time.sleep(0.001)

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
                gpu_seats,
                seq_lens,
                bonus_tokens,
                request_epochs=self._capture_request_epochs(gpu_seats),
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

    def _capture_request_epochs(self, gpu_seats: torch.Tensor) -> torch.Tensor:
        return self.verifier.gpu_tail_buffer.capture_active_request_epochs(
            gpu_seats,
            out=torch.empty_like(gpu_seats),
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCppGpuDrafterAuthoritativeTail(CustomTestCase):
    verifier_type = CppVerifierDecoupledSpecDataPlane
    drafter_type = CppDrafterDecoupledSpecDataPlane

    """Direct CUDA coverage for drafter-side forced-token reconciliation."""

    def setUp(self):
        self.context = zmq.Context()
        suffix = uuid.uuid4().hex
        verifier_endpoint = f"inproc://gpu-authoritative-verifier-{suffix}"
        drafter_endpoint = f"inproc://gpu-authoritative-drafter-{suffix}"
        self.landing_stream = torch.cuda.Stream(device="cuda:0")
        self.verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=verifier_endpoint,
                connect_endpoints=(drafter_endpoint,),
                rank=0,
            ),
            required_tail_len=0,
            context=self.context,
        )
        self.drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=drafter_endpoint,
                connect_endpoints=(verifier_endpoint,),
                rank=0,
            ),
            context=self.context,
            device="cuda:0",
            num_gpu_seats=1,
            num_draft_tokens=3,
            landing_stream=self.landing_stream,
        )
        self.verifier.start()
        self.drafter.start()
        self.tail = self.drafter.gpu_tail_buffer
        self.checkpoint_slots = torch.arange(
            self.tail.num_seats * self.tail.tail_capacity,
            dtype=torch.int64,
            device="cuda:0",
        ).reshape(self.tail.num_seats, self.tail.tail_capacity)
        self.req_to_token = torch.zeros(
            (self.tail.num_seats, 128), dtype=torch.int32, device="cuda:0"
        )
        self.future_output_tokens = torch.full(
            (self.tail.num_seats,), -1, dtype=torch.int64, device="cuda:0"
        )
        self.next_cache_loc = 100

    def tearDown(self):
        self.drafter.close()
        self.verifier.close()
        self.context.destroy(linger=0)

    def test_prefill_commit_race_replays_without_cpu_token_shadow(self):
        seat, request_epoch = self._open("prefill-race")
        self._commit_batch(
            "prefill-race",
            [(0, [90]), (1, [91])],
        )
        self._wait_state(seat, committed_lens=2, pending_expected_lens=2)

        sampled = torch.tensor([77], dtype=torch.int32, device="cuda:0")
        accepted = self.tail.append_prefill_sample(
            self._vector(seat),
            self._vector(request_epoch),
            sampled,
        )
        torch.cuda.synchronize()
        self.assertEqual(sampled.cpu().tolist(), [90])
        self.assertEqual(accepted.cpu().tolist(), [True])
        self._assert_state(
            seat,
            committed_lens=2,
            model_output_lens=1,
            pending_expected_lens=1,
            raw_tail_lens=0,
            model_input_tokens=90,
            can_accept_prefix_lens=0,
            error_codes=0,
        )

        first = self._forward(seat, request_epoch, sampled_token=700)
        self.assertEqual((first["input"], first["relay"]), (90, 91))
        second = self._forward(seat, request_epoch, sampled_token=701)
        self.assertEqual((second["input"], second["relay"]), (91, 701))
        self._assert_state(
            seat,
            committed_lens=2,
            model_output_lens=3,
            pending_expected_lens=0,
            raw_tail_lens=1,
            can_accept_prefix_lens=2,
            error_codes=0,
        )
        self.assertEqual(
            self.tail.checkpoint_positions[seat, 2:5].cpu().tolist(),
            [2, 3, 4],
        )

    def test_gpu_progress_exposes_only_compact_scheduling_cursors(self):
        seat, request_epoch = self._open("compact-progress")
        self._append_prefill(seat, request_epoch, 10)
        for token in range(11, 16):
            self._forward(seat, request_epoch, sampled_token=token)

        deadline = time.monotonic() + 3.0
        observed = None
        while time.monotonic() < deadline:
            for row in self.drafter.drain_gpu_progress():
                if row[0] == "compact-progress" and row[5] == 6:
                    observed = row
                    break
            if observed is not None:
                break
            time.sleep(0.001)

        self.assertEqual(observed, ("compact-progress", 0, request_epoch, 6, 0, 6))

    def test_gpu_progress_wakes_for_forced_replay_without_token_shadow(self):
        seat, request_epoch = self._open("forced-progress")
        self._append_prefill(seat, request_epoch, 10)
        for token in range(11, 16):
            self._forward(seat, request_epoch, sampled_token=token)
        # Consume the complete raw tail and queue one verifier bonus. This
        # crosses non-runnable -> runnable without exposing token history.
        self._commit_batch(
            "forced-progress",
            [(0, [10, 11, 12, 13, 14, 15, 99])],
        )
        self._wait_state(
            seat,
            committed_lens=7,
            model_output_lens=6,
            pending_expected_lens=1,
            raw_tail_lens=0,
        )

        deadline = time.monotonic() + 3.0
        observed = None
        while time.monotonic() < deadline:
            for row in self.drafter.drain_gpu_progress():
                if row[0] == "forced-progress" and row[4:] == (7, 0):
                    observed = row
                    break
            if observed is not None:
                break
            time.sleep(0.001)

        self.assertEqual(observed, ("forced-progress", 0, request_epoch, 7, 7, 0))

    def test_one_token_commit_between_decode_prepare_and_finish_is_valid(self):
        seat, request_epoch = self._open("one-token-health")
        self._append_prefill(seat, request_epoch, 10)
        prepared = self._prepare(seat, request_epoch)

        self._commit_batch("one-token-health", [(0, [10])])
        self._wait_state(
            seat,
            committed_lens=1,
            model_output_lens=1,
            raw_tail_lens=0,
            error_codes=0,
        )
        finished = self._finish_prepared(
            seat,
            request_epoch,
            prepared,
            sampled_token=11,
        )

        self.assertTrue(finished["accepted"])
        self.assertTrue(finished["owned"])
        self.assertEqual(finished["relay"], 11)
        self._assert_state(
            seat,
            committed_lens=1,
            model_output_lens=2,
            raw_tail_lens=1,
            can_accept_prefix_lens=1,
            error_codes=0,
        )

    def test_mismatch_rebinds_candidate_and_reclaims_rewound_kv_cell(self):
        seat, request_epoch = self._open("candidate-rebind")
        self._append_prefill(seat, request_epoch, 10)
        first = self._forward(seat, request_epoch, sampled_token=11)
        self.assertEqual(first["reclaim"], -1)
        self.assertEqual(int(self.req_to_token[seat, 2].item()), 100)

        self._commit_batch("candidate-rebind", [(0, [99])])
        self._wait_state(
            seat,
            committed_lens=1,
            model_state_positions=2,
            model_input_tokens=99,
        )
        prepared = self._prepare(seat, request_epoch)
        self.assertEqual(int(prepared["old"].item()), 100)
        self.assertEqual(int(prepared["candidate"].item()), 101)
        self.assertEqual(int(self.req_to_token[seat, 2].item()), 101)

        resumed = self._finish_prepared(
            seat,
            request_epoch,
            prepared,
            sampled_token=100,
        )
        self.assertTrue(resumed["accepted"])
        self.assertTrue(resumed["owned"])
        self.assertEqual(resumed["reclaim"], 100)
        self.assertEqual(resumed["relay"], 100)

    def test_direct_mismatch_then_cross_batch_target_commit_replays_in_order(self):
        seat, request_epoch = self._open("cross-batch")
        self._append_prefill(seat, request_epoch, 10)
        self._forward(seat, request_epoch, sampled_token=11)

        # [10, X] rewinds to X through a resident mismatch. A separately
        # landed target-only Y must queue behind X rather than advancing the
        # model cursor through an unmaterialized checkpoint.
        self._commit_batch("cross-batch", [(0, [10, 99])])
        self._wait_state(seat, committed_lens=2, model_input_tokens=99)
        self._assert_state(
            seat,
            model_output_lens=2,
            pending_expected_lens=0,
            raw_tail_lens=0,
            error_codes=0,
        )
        self._commit_batch("cross-batch", [(2, [100])])
        self._wait_state(seat, committed_lens=3, pending_expected_lens=1)

        replay_x = self._forward(seat, request_epoch, sampled_token=700)
        self.assertEqual((replay_x["input"], replay_x["relay"]), (99, 100))
        replay_y = self._forward(seat, request_epoch, sampled_token=701)
        self.assertEqual((replay_y["input"], replay_y["relay"]), (100, 701))
        self._assert_state(
            seat,
            committed_lens=3,
            model_output_lens=4,
            pending_expected_lens=0,
            raw_tail_lens=1,
            can_accept_prefix_lens=3,
            error_codes=0,
        )

    def test_prepared_candidate_survives_all_match_bonus_landing(self):
        seat, request_epoch = self._open("prepared-all-match")
        self._append_prefill(seat, request_epoch, 10)
        self._forward(seat, request_epoch, sampled_token=11)

        # The candidate was prepared from current input 11 before landing.
        # Matching all resident raw tokens and queuing bonus 99 changes C/F,
        # but not the physical model branch or its current input/state.
        prepared = self._prepare(seat, request_epoch)
        self.assertEqual(
            (
                int(prepared["input"].item()),
                int(prepared["state"].item()),
            ),
            (11, 3),
        )
        self._commit_batch("prepared-all-match", [(0, [10, 11, 99])])
        self._wait_state(seat, committed_lens=3, pending_expected_lens=1)
        self._assert_state(
            seat,
            model_output_lens=2,
            model_state_positions=3,
            model_input_tokens=11,
            pending_expected_lens=1,
            raw_tail_lens=0,
            can_accept_prefix_lens=0,
            error_codes=0,
        )

        finished = self._finish_prepared(
            seat,
            request_epoch,
            prepared,
            sampled_token=777,
        )
        self.assertTrue(finished["accepted"])
        self.assertTrue(finished["owned"])
        self.assertEqual(finished["relay"], 99)
        self._assert_state(
            seat,
            committed_lens=3,
            model_output_lens=3,
            model_state_positions=4,
            model_input_tokens=99,
            pending_expected_lens=0,
            raw_tail_lens=0,
            can_accept_prefix_lens=0,
            error_codes=0,
        )
        self.assertEqual(
            int(self.tail.checkpoint_positions[seat, 4].cpu()),
            4,
        )

        lookahead = self._forward(seat, request_epoch, sampled_token=100)
        self.assertEqual((lookahead["input"], lookahead["relay"]), (99, 100))
        self._assert_state(
            seat,
            committed_lens=3,
            model_output_lens=4,
            pending_expected_lens=0,
            raw_tail_lens=1,
            can_accept_prefix_lens=3,
            error_codes=0,
        )

    def test_finish_identity_position_input_across_commit_orderings(self):
        # The repeated-token case deliberately preserves input while moving
        # the state backward; the boundary mismatch does the converse.
        cases = (
            ("match-prefix", [10, 11, 12], [(0, [10])], True),
            ("match-bonus", [10, 11], [(0, [10, 11, 99])], True),
            ("replace-input", [10, 11], [(0, [10, 99])], False),
            ("rewind-same-input", [10, 11, 12], [(0, [12])], False),
            ("rewrite-then-force", [10, 11], [(0, [10, 99]), (2, [11])], False),
            ("two-matches", [10, 11, 12], [(0, [10]), (1, [11])], True),
        )
        for name, raw, commits, survives_commit in cases:
            for ordering in ("before-prepare", "before-finish", "after-finish"):
                for relay_view in (False, True):
                    with self.subTest(
                        case=name, ordering=ordering, relay_view=relay_view
                    ):
                        request_id = f"{name}-{ordering}-{relay_view}"
                        seat, epoch = self._open(request_id)
                        self._append_prefill(seat, epoch, raw[0])
                        for token in raw[1:]:
                            self._forward(seat, epoch, sampled_token=token)
                        end = commits[-1][0] + len(commits[-1][1])
                        if ordering == "before-prepare":
                            self._commit_batch(request_id, commits)
                            self._wait_state(seat, committed_lens=end)
                        prepared = self._prepare(seat, epoch, relay_view=relay_view)
                        if ordering == "before-finish":
                            self._commit_batch(request_id, commits)
                            self._wait_state(seat, committed_lens=end)
                        result = self._finish_prepared(
                            seat, epoch, prepared, sampled_token=77
                        )
                        self.assertEqual(
                            result["accepted"],
                            survives_commit if ordering == "before-finish" else True,
                        )
                        if ordering == "after-finish":
                            self._commit_batch(request_id, commits)
                            self._wait_state(seat, committed_lens=end)
                        self._assert_state(seat, error_codes=0)
                        self._close(request_id)

    def test_finish_races_native_control_landing_with_relay_view(self):
        for iteration in range(20):
            with self.subTest(iteration=iteration):
                seat, epoch = self._open("finish-race")
                self._append_prefill(seat, epoch, 10)
                self._forward(seat, epoch, sampled_token=11)
                prepared = self._prepare(seat, epoch, relay_view=True)
                # Do not wait for landing: the RX daemon's stream and finish
                # contend for the seat. Either ordering must converge to the
                # same corrected input and forced backlog, without a branch ID.
                self._commit_batch("finish-race", [(0, [10, 99]), (2, [11])])
                self._finish_prepared(
                    seat, epoch, prepared, sampled_token=77, check_relay=False
                )
                self._wait_state(seat, committed_lens=3)
                self._assert_state(
                    seat, model_output_lens=2, model_input_tokens=99,
                    raw_tail_lens=0, pending_expected_lens=1, error_codes=0,
                )
                resumed = self._forward(seat, epoch, sampled_token=88)
                self.assertTrue(resumed["accepted"])
                self.assertEqual((resumed["input"], resumed["relay"]), (99, 11))
                self._close("finish-race")

    def test_decode_does_not_reclaim_stale_unowned_request_table_cell(self):
        seat, epoch = self._open("stale-kv")
        self._append_prefill(seat, epoch, 10)
        # A recycled request-pool row retains old indices beyond its prefill.
        self.req_to_token[seat, 2] = 777
        prepared = self._prepare(seat, epoch)
        result = self._finish_prepared(seat, epoch, prepared, sampled_token=11)
        self.assertEqual(result["reclaim"], -1)

    def test_close_reopen_rejects_decode_with_identical_position_and_input(self):
        seat, epoch = self._open("decode-reused")
        self._append_prefill(seat, epoch, 10)
        prepared = self._prepare(seat, epoch)
        self._close("decode-reused")
        new_seat, new_epoch = self._open("decode-reused")
        self.assertEqual(new_seat, seat)
        self.assertGreater(new_epoch, epoch)
        self._append_prefill(seat, new_epoch, 10)
        self.future_output_tokens[seat] = 10
        result = self._finish_prepared(seat, epoch, prepared, sampled_token=99)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["owned"])
        self.assertEqual(result["reclaim"], int(prepared["candidate"].item()))
        self._assert_state(
            seat, model_input_tokens=10, model_output_lens=1,
            raw_tail_lens=1, error_codes=0,
        )

    def test_forced_ring_capacity_plus_one_latches_error(self):
        seat, _ = self._open("forced-overflow")
        capacity = self.tail.tail_capacity
        self._commit_batch(
            "forced-overflow",
            [(index, [100 + index]) for index in range(capacity + 1)],
        )
        self._wait_state(seat, error_codes=5)
        self._assert_state(
            seat,
            committed_lens=capacity,
            model_output_lens=0,
            pending_expected_lens=capacity,
            raw_tail_lens=0,
            error_codes=5,
        )

    def test_expanded_pending_capacity_absorbs_drafter_stall(self):
        suffix = uuid.uuid4().hex
        verifier = self.verifier_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=f"inproc://pending-cap-verifier-{suffix}",
                connect_endpoints=(f"inproc://pending-cap-drafter-{suffix}",),
                rank=0,
            ),
            required_tail_len=0,
            context=self.context,
        )
        drafter = self.drafter_type(
            DecoupledSpecIpcConfig(
                bind_endpoint=f"inproc://pending-cap-drafter-{suffix}",
                connect_endpoints=(f"inproc://pending-cap-verifier-{suffix}",),
                rank=0,
            ),
            context=self.context,
            device="cuda:0",
            num_gpu_seats=1,
            num_draft_tokens=3,
            pending_token_capacity=32,
            landing_stream=torch.cuda.Stream(device="cuda:0"),
        )
        verifier.start()
        drafter.start()
        original = self.verifier, self.drafter, self.tail
        self.verifier, self.drafter, self.tail = (
            verifier,
            drafter,
            drafter.gpu_tail_buffer,
        )
        try:
            seat, _ = self._open("expanded-pending")
            num_commits = self.tail.tail_capacity + 1
            self._commit_batch(
                "expanded-pending",
                [(index, [100 + index]) for index in range(num_commits)],
            )
            self._wait_state(
                seat,
                pending_expected_lens=num_commits,
                error_codes=0,
            )
            self.assertEqual(
                self.tail.pending_expected_tokens[seat, :num_commits]
                .cpu()
                .tolist(),
                list(range(100, 100 + num_commits)),
            )
        finally:
            self.verifier, self.drafter, self.tail = original
            drafter.close()
            verifier.close()

    def test_all_raw_match_bonus_and_consecutive_commits_wrap_forced_ring(self):
        seat, request_epoch = self._open("wrap")
        self._append_prefill(seat, request_epoch, 10)
        for token in range(11, 16):
            self._forward(seat, request_epoch, sampled_token=token)
        self._assert_state(
            seat,
            committed_lens=0,
            model_output_lens=6,
            raw_tail_lens=6,
            pending_expected_lens=0,
            error_codes=0,
        )

        # M=6 and capacity=7: the bonus and two immediately following commits
        # occupy forced-ring offsets 6, 0 and 1. No intermediate ACK or Python
        # commit staging is allowed to serialize this sequence.
        self._commit_batch(
            "wrap",
            [
                (0, [10, 11, 12, 13, 14, 15, 90]),
                (7, [91]),
                (8, [92]),
            ],
        )
        self._wait_state(seat, committed_lens=9, pending_expected_lens=3)
        self._assert_state(
            seat,
            model_output_lens=6,
            raw_tail_lens=0,
            model_input_tokens=15,
            can_accept_prefix_lens=0,
            error_codes=0,
        )
        self.assertEqual(
            self.tail.pending_expected_tokens[seat, [6, 0, 1]].cpu().tolist(),
            [90, 91, 92],
        )

        relays = []
        for sampled_token in (800, 801, 802, 803):
            result = self._forward(
                seat, request_epoch, sampled_token=sampled_token
            )
            relays.append(result["relay"])
        self.assertEqual(relays, [90, 91, 92, 803])
        # A second sample distinguishes full-tail publication from the old
        # conservative K-1 egress policy.
        self._forward(seat, request_epoch, sampled_token=804)
        self._assert_state(
            seat,
            committed_lens=9,
            model_output_lens=11,
            pending_expected_lens=0,
            raw_tail_lens=2,
            can_accept_prefix_lens=9,
            error_codes=0,
        )
        snapshot = self._wait_snapshot("wrap", committed_len=9, tail_len=2)
        self.assertEqual(snapshot.tail_tokens, (803, 804))

    def test_capacity_edge_stale_forward_cannot_alias_mismatch_checkpoint(self):
        for mismatch_index in (0, self.tail.tail_capacity - 2):
            with self.subTest(mismatch_index=mismatch_index):
                request_id = f"capacity-mismatch-{mismatch_index}"
                seat, request_epoch = self._open(request_id)
                self._append_prefill(seat, request_epoch, 10)
                for token in range(11, 16):
                    self._forward(seat, request_epoch, sampled_token=token)

                prepared = self._prepare(seat, request_epoch)
                restore_position = 2 + mismatch_index
                stale_dst_position = int(prepared["state"].item()) + 1
                self.assertEqual(
                    (stale_dst_position - restore_position)
                    % self.tail.tail_capacity,
                    self.tail.tail_capacity - mismatch_index - 1,
                )
                self.assertNotEqual(
                    stale_dst_position % self.tail.tail_capacity,
                    restore_position % self.tail.tail_capacity,
                )

                committed_tokens = list(range(10, 10 + mismatch_index)) + [99]
                self._commit_batch(request_id, [(0, committed_tokens)])
                self._wait_state(
                    seat,
                    committed_lens=len(committed_tokens),
                )
                stale = self._finish_prepared(
                    seat,
                    request_epoch,
                    prepared,
                    sampled_token=777,
                )
                self.assertFalse(stale["accepted"])
                self.assertTrue(stale["owned"])
                self.assertEqual(stale["relay"], 99)
                self.assertEqual(
                    int(
                        self.tail.checkpoint_positions[
                            seat,
                            restore_position % self.tail.tail_capacity,
                        ].cpu()
                    ),
                    restore_position,
                )

                resumed = self._forward(
                    seat, request_epoch, sampled_token=100
                )
                self.assertEqual(resumed["input"], 99)
                self._assert_state(
                    seat,
                    raw_tail_lens=1,
                    pending_expected_lens=0,
                    can_accept_prefix_lens=len(committed_tokens),
                    error_codes=0,
                )
                self._close(request_id)

    def test_close_reopen_rejects_old_prefill_epilogue(self):
        old_seat, old_epoch = self._open("reused")
        self._close("reused")
        new_seat, new_epoch = self._open("reused")
        self.assertEqual(new_seat, old_seat)
        self.assertGreater(new_epoch, old_epoch)

        old_sample = torch.tensor([77], dtype=torch.int32, device="cuda:0")
        old_accept = self.tail.append_prefill_sample(
            self._vector(old_seat),
            self._vector(old_epoch),
            old_sample,
        )
        torch.cuda.synchronize()
        self.assertEqual(old_accept.cpu().tolist(), [False])
        self.assertEqual(old_sample.cpu().tolist(), [77])
        self._assert_state(
            new_seat,
            committed_lens=0,
            model_output_lens=0,
            raw_tail_lens=0,
            pending_expected_lens=0,
            model_input_tokens=-1,
            error_codes=0,
        )
        self._append_prefill(new_seat, new_epoch, 88)
        self._assert_state(
            new_seat,
            model_output_lens=1,
            raw_tail_lens=1,
            model_input_tokens=88,
            error_codes=0,
        )

    def _open(self, request_id: str) -> tuple[int, int]:
        self.verifier.open_request(
            DraftSync(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                prompt_token_ids=[7, 8],
                committed_outputs=[],
            )
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            binding = self.drafter.lookup_gpu_binding(request_id, 0)
            if binding is not None:
                self.tail.wait_for_landing()
                torch.cuda.synchronize()
                return binding
            time.sleep(0.001)
        self.fail(f"GPU drafter did not bind request {request_id}")

    def _close(self, request_id: str) -> None:
        self.verifier.close_request(
            DraftClose(
                request_id=request_id,
                src_verifier_rank=0,
                dst_drafter_rank=0,
                reason="test",
            )
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.drafter.lookup_gpu_binding(request_id, 0) is None:
                self.tail.wait_for_landing()
                torch.cuda.synchronize()
                return
            time.sleep(0.001)
        self.fail(f"GPU drafter did not close request {request_id}")

    def _commit_batch(
        self,
        request_id: str,
        segments: list[tuple[int, list[int]]],
    ) -> None:
        self.verifier.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=0,
                verify_commit_messages=[
                    VerifyCommit(
                        request_id=request_id,
                        src_verifier_rank=0,
                        dst_drafter_rank=0,
                        pre_verify_committed_len=pre_verify_committed_len,
                        committed_tokens=committed_tokens,
                    )
                    for pre_verify_committed_len, committed_tokens in segments
                ],
            )
        )

    def _append_prefill(
        self, seat: int, request_epoch: int, sampled_token: int
    ) -> None:
        sampled = torch.tensor(
            [sampled_token], dtype=torch.int32, device="cuda:0"
        )
        accepted = self.tail.append_prefill_sample(
            self._vector(seat), self._vector(request_epoch), sampled
        )
        torch.cuda.synchronize()
        self.assertEqual(accepted.cpu().tolist(), [True])

    def _prepare(
        self, seat: int, request_epoch: int, *, relay_view: bool = False
    ) -> dict[str, torch.Tensor]:
        tensors = {
            name: torch.empty(1, dtype=torch.int64, device="cuda:0")
            for name in ("input", "seq_len", "src", "dst", "state", "old")
        }
        if relay_view:
            tensors["input"] = self.future_output_tokens[seat : seat + 1]
        tensors["orig_seq_len"] = torch.empty(
            1, dtype=torch.int32, device="cuda:0"
        )
        candidate = torch.tensor(
            [self.next_cache_loc], dtype=torch.int64, device="cuda:0"
        )
        self.next_cache_loc += 1
        self.tail.prepare_decode(
            self._vector(seat),
            self._vector(request_epoch),
            self._vector(seat),
            candidate,
            self.checkpoint_slots,
            self.req_to_token,
            resolved_input_ids=tensors["input"],
            resolved_seq_lens=tensors["seq_len"],
            resolved_orig_seq_lens=tensors["orig_seq_len"],
            mamba_src_indices=None if self.checkpoint_slots is None else tensors["src"],
            mamba_dst_indices=None if self.checkpoint_slots is None else tensors["dst"],
            captured_state_positions=tensors["state"],
            old_cache_locs=tensors["old"],
        )
        tensors["candidate"] = candidate
        torch.cuda.synchronize()
        self.assertEqual(tensors["orig_seq_len"].item(), tensors["seq_len"].item())
        return tensors

    def _finish_prepared(
        self,
        seat: int,
        request_epoch: int,
        prepared: dict[str, torch.Tensor],
        *,
        sampled_token: int,
        check_relay: bool = True,
    ) -> dict[str, int | bool]:
        kv_outcomes = torch.empty((1, 3), dtype=torch.int64, device="cuda:0")
        self.tail.finish_decode(
            self._vector(seat),
            self._vector(request_epoch),
            self._vector(seat),
            prepared["candidate"],
            torch.tensor([sampled_token], dtype=torch.int32, device="cuda:0"),
            self.req_to_token,
            resolved_input_tokens=prepared["input"],
            captured_state_positions=prepared["state"],
            old_cache_locs=prepared["old"],
            kv_outcomes=kv_outcomes,
            future_output_tokens=self.future_output_tokens,
        )
        torch.cuda.synchronize()
        if check_relay:
            self.assertEqual(
                int(self.future_output_tokens[seat].item()),
                int(self.tail.model_input_tokens[seat].item()),
            )
        return {
            "accepted": bool(kv_outcomes[0, 0].item()),
            "owned": int(kv_outcomes[0, 1].item()) >= 0,
            "relay": int(self.future_output_tokens[seat].item()),
            "reclaim": int(kv_outcomes[0, 2].item()),
        }

    def _forward(
        self, seat: int, request_epoch: int, *, sampled_token: int
    ) -> dict[str, int | bool]:
        prepared = self._prepare(seat, request_epoch)
        result = self._finish_prepared(
            seat,
            request_epoch,
            prepared,
            sampled_token=sampled_token,
        )
        result["input"] = int(prepared["input"].item())
        result["state"] = int(prepared["state"].item())
        self.assertTrue(result["accepted"])
        self.assertTrue(result["owned"])
        return result

    def _wait_state(self, seat: int, **expected: int) -> None:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            torch.cuda.synchronize()
            if all(
                int(getattr(self.tail, name)[seat].cpu()) == value
                for name, value in expected.items()
            ):
                return
            time.sleep(0.001)
        actual = {
            name: int(getattr(self.tail, name)[seat].cpu()) for name in expected
        }
        self.fail(f"GPU drafter state did not converge: {actual=} {expected=}")

    def _assert_state(self, seat: int, **expected: int) -> None:
        torch.cuda.synchronize()
        self.assertEqual(
            {
                name: int(getattr(self.tail, name)[seat].cpu())
                for name in expected
            },
            expected,
        )

    def _wait_snapshot(self, request_id: str, *, committed_len: int, tail_len: int):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            snapshot = self.verifier.snapshot_one(request_id)
            if (
                snapshot.committed_len == committed_len
                and len(snapshot.tail_tokens) == tail_len
            ):
                return snapshot
            time.sleep(0.001)
        self.fail(
            f"Verifier snapshot did not converge: request_id={request_id} "
            f"snapshot={snapshot}"
        )

    @staticmethod
    def _vector(value: int) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.int64, device="cuda:0")


class TestPythonGpuDraftTail(TestCppGpuDraftTailBuffer):
    verifier_type = VerifierDecoupledSpecDataPlane
    drafter_type = DrafterDecoupledSpecDataPlane


class TestPythonVerifierGpuDraftTail(TestCppGpuDraftTailBuffer):
    verifier_type = VerifierDecoupledSpecDataPlane
    drafter_type = CppDrafterDecoupledSpecDataPlane


class TestPythonDrafterGpuDraftTail(TestCppGpuDraftTailBuffer):
    verifier_type = CppVerifierDecoupledSpecDataPlane
    drafter_type = DrafterDecoupledSpecDataPlane


class TestPythonGpuDrafter(TestCppGpuDrafterAuthoritativeTail):
    verifier_type = VerifierDecoupledSpecDataPlane
    drafter_type = DrafterDecoupledSpecDataPlane


class TestPythonVerifierGpuDrafter(TestCppGpuDrafterAuthoritativeTail):
    verifier_type = VerifierDecoupledSpecDataPlane
    drafter_type = CppDrafterDecoupledSpecDataPlane


class TestPythonDrafterGpuDrafter(TestCppGpuDrafterAuthoritativeTail):
    verifier_type = CppVerifierDecoupledSpecDataPlane
    drafter_type = DrafterDecoupledSpecDataPlane


class TestDenseGpuDrafterAuthoritativeTail(TestCppGpuDrafterAuthoritativeTail):
    """Reconcile/rollback, stale forwards and KV reclaim without recurrent slots."""

    def setUp(self):
        super().setUp()
        self.checkpoint_slots = None


class TestPythonDenseGpuDrafterAuthoritativeTail(TestDenseGpuDrafterAuthoritativeTail):
    verifier_type = VerifierDecoupledSpecDataPlane
    drafter_type = DrafterDecoupledSpecDataPlane


if __name__ == "__main__":
    unittest.main()

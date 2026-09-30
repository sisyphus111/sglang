"""GPU-backed transport peers and synchronized inspection for unit tests only."""

import time

import torch
from draft_tail_reference import DraftTailSnapshot

from sglang.srt.speculative.decoupled_spec_data_plane import (
    DrafterDecoupledSpecDataPlane,
    VerifierDecoupledSpecDataPlane,
)
from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
)


def gpu_config():
    return dict(
        device="cuda:0",
        num_gpu_seats=16,
        num_draft_tokens=3,
        landing_stream=torch.cuda.Stream(),
    )


class VerifierGpuTestPeer:
    python_transport = False

    def __init__(self, config, *, required_tail_len=0, **kwargs):
        if "device" not in kwargs:
            kwargs = gpu_config() | kwargs
        super().__init__(config, python_transport=self.python_transport, **kwargs)
        self._test_epoch = 0
        self._test_seats = {}

    def open_requests(self, requests):
        bound = []
        for message, gpu_seat, request_epoch in requests:
            if gpu_seat is None:
                occupied = set(self._test_seats.values())
                gpu_seat = next(
                    i
                    for i in range(self.gpu_tail_buffer.num_seats)
                    if i not in occupied
                )
                self._test_epoch += 1
                request_epoch = self._test_epoch
            self._test_seats[message.request_id] = gpu_seat
            bound.append((message, gpu_seat, request_epoch))
        super().open_requests(bound)

    def snapshot_one(self, request_id):
        tail = self.gpu_tail_buffer
        binding = tail.lookup_binding(request_id, self.config.rank)
        if binding is None:
            raise KeyError(request_id)
        seat, _ = binding
        tail.wait_for_landing()
        torch.cuda.synchronize()
        raw_len = int(tail.raw_tail_lens[seat].item())
        return DraftTailSnapshot(
            request_id=request_id,
            committed_len=int(tail.committed_lens[seat].item()),
            tail_tokens=tuple(tail.tail_tokens[seat, :raw_len].tolist()),
            raw_tail_len=raw_len,
            num_consumable_drafts=int(tail.consumable_tail_lens[seat].item()),
        )

    def close_request(self, message):
        super().close_request(message)
        self._test_seats.pop(message.request_id, None)


class DrafterGpuTestPeer:
    python_transport = False

    def __init__(self, config, **kwargs):
        if "device" not in kwargs:
            # These transport probes inject tails without model forwards. Keep
            # their unconsumed commits in a bounded test-sized replay buffer.
            kwargs = gpu_config() | {"pending_token_capacity": 1024} | kwargs
        super().__init__(config, python_transport=self.python_transport, **kwargs)

    def drain_controls(self):
        controls = self.collect_lifecycle_controls()
        if controls.is_empty():
            return []
        return [
            DraftControlBatch(
                dst_drafter_rank=self.config.rank,
                close_messages=[
                    DraftClose(
                        request_id=k.request_id,
                        src_verifier_rank=k.src_verifier_rank,
                        dst_drafter_rank=self.config.rank,
                        reason="test-observation",
                    )
                ],
            )
            for k in controls.close_keys
        ] + [
            DraftControlBatch(
                dst_drafter_rank=self.config.rank,
                sync_messages=[message],
            )
            for message in controls.sync_messages
        ]

    def wait_for_commit(self, request_id, committed_len, verifier_rank=0):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            binding = self.lookup_gpu_binding(request_id, verifier_rank)
            if binding is not None:
                seat, _ = binding
                self.gpu_tail_buffer.wait_for_landing()
                if self.gpu_tail_buffer.committed_lens[seat].item() == committed_len:
                    return seat
            time.sleep(0.001)
        raise AssertionError(
            f"Commit did not reach GPU: {request_id=} {committed_len=}"
        )


class CppVerifierTestPeer(VerifierGpuTestPeer, VerifierDecoupledSpecDataPlane):
    pass


class CppDrafterTestPeer(DrafterGpuTestPeer, DrafterDecoupledSpecDataPlane):
    pass


class PythonVerifierTestPeer(VerifierGpuTestPeer, VerifierDecoupledSpecDataPlane):
    python_transport = True


class PythonDrafterTestPeer(DrafterGpuTestPeer, DrafterDecoupledSpecDataPlane):
    python_transport = True

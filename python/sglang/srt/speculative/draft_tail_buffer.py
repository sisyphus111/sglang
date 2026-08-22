from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence

import msgspec

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    VerifyCommit,
)


class DraftTailSnapshot(msgspec.Struct, frozen=True):
    """Immutable verifier view of one linearized draft-tail state."""

    request_id: str
    committed_len: int
    tail_tokens: tuple[int, ...] = ()
    raw_tail_len: int = 0
    num_consumable_drafts: int = 0


class _RequestDraftTailState(msgspec.Struct):
    """Verifier-owned rolling state for one drafter request."""

    drafter_rank: int
    committed_len: int = 0
    can_accept_prefix_len: int = 0
    tail_tokens: list[int] = []
    pending_expected_tokens: deque[int] = msgspec.field(default_factory=deque)

    def consumable_tail(self) -> tuple[int, ...]:
        if self.pending_expected_tokens:
            return ()
        return tuple(self.tail_tokens)


class DraftTailBuffer:
    """Authoritative verifier-side rolling draft tail.

    Snapshot, commit, and stream append all linearize under the same condition
    lock. A commit preserves a fully matching draft suffix. A short or
    mismatching tail is cleared and converted into ``pending_expected_tokens``
    until the drafter confirms the verifier-owned prefix token by token.
    """

    def __init__(self, *, verifier_rank: int, required_tail_len: int = 0) -> None:
        self.verifier_rank = int(verifier_rank)
        self.required_tail_len = max(0, int(required_tail_len))
        self._condition = threading.Condition()
        self._closed = False
        self._states: dict[str, _RequestDraftTailState] = {}
        self.last_draft_wait_ns = 0

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._states.clear()
            self._condition.notify_all()

    def has_request(self, request_id: str) -> bool:
        with self._condition:
            return request_id in self._states

    def get_committed_len(self, request_id: str) -> int | None:
        with self._condition:
            state = self._states.get(request_id)
            return None if state is None else int(state.committed_len)

    def open_request(self, message: DraftSync) -> None:
        self.open_requests([message])

    def open_requests(self, messages: Sequence[DraftSync]) -> None:
        if not messages:
            return
        with self._condition:
            self._ensure_open_locked()
            for message in messages:
                self._open_request_locked(message)
            self._condition.notify_all()

    def _open_request_locked(self, message: DraftSync) -> None:
        if int(message.src_verifier_rank) != self.verifier_rank:
            raise RuntimeError(
                "DraftSync belongs to a different verifier: "
                f"request_id={message.request_id} "
                f"expected_verifier_rank={self.verifier_rank} "
                f"src_verifier_rank={message.src_verifier_rank}"
            )
        committed_len = len(message.committed_outputs)
        self._states[message.request_id] = _RequestDraftTailState(
            drafter_rank=int(message.dst_drafter_rank),
            committed_len=committed_len,
            can_accept_prefix_len=committed_len,
        )

    def apply_verify_commit(self, message: VerifyCommit) -> None:
        self.apply_verify_commits([message])

    def apply_verify_commits(self, messages: Sequence[VerifyCommit]) -> None:
        if not messages:
            return
        with self._condition:
            self._ensure_open_locked()
            for message in messages:
                self._apply_commit_locked(message)
            self._condition.notify_all()

    def _apply_commit_locked(self, message: VerifyCommit) -> None:
        message.validate_committed_tokens()
        if int(message.src_verifier_rank) != self.verifier_rank:
            raise RuntimeError(
                "VerifyCommit belongs to a different verifier: "
                f"request_id={message.request_id} "
                f"expected_verifier_rank={self.verifier_rank} "
                f"src_verifier_rank={message.src_verifier_rank}"
            )

        state = self._states.get(message.request_id)
        if state is None:
            # A close may race with an already queued verifier result.
            return
        if int(message.dst_drafter_rank) != int(state.drafter_rank):
            raise RuntimeError(
                "VerifyCommit targets a different drafter: "
                f"request_id={message.request_id} "
                f"expected_drafter_rank={state.drafter_rank} "
                f"dst_drafter_rank={message.dst_drafter_rank}"
            )

        pre_verify_committed_len = int(message.pre_verify_committed_len)
        expected_pre_verify_len = int(state.committed_len) + len(
            state.pending_expected_tokens
        )
        if pre_verify_committed_len != expected_pre_verify_len:
            raise RuntimeError(
                "VerifyCommit prefix does not match the draft-tail state: "
                f"request_id={message.request_id} "
                f"pre_verify_committed_len={pre_verify_committed_len} "
                f"state_committed_len={state.committed_len} "
                f"pending_expected_len={len(state.pending_expected_tokens)}"
            )

        commit_tokens = [int(token) for token in message.committed_tokens]
        if state.pending_expected_tokens:
            if state.tail_tokens:
                raise RuntimeError(
                    "Draft tail must be empty while committed tokens are pending: "
                    f"request_id={message.request_id} "
                    f"tail_tokens={state.tail_tokens}"
                )
            state.pending_expected_tokens.extend(commit_tokens)
            return

        raw_tail_len = len(state.tail_tokens)
        max_match_len = min(len(commit_tokens), raw_tail_len)
        match_len = 0
        while (
            match_len < max_match_len
            and int(state.tail_tokens[match_len]) == commit_tokens[match_len]
        ):
            match_len += 1

        if match_len:
            del state.tail_tokens[:match_len]
            state.committed_len += match_len

        if match_len == len(commit_tokens):
            return

        # A present mismatch invalidates every stream based before this point.
        # A merely short tail keeps the previous stale-base boundary.
        if match_len < raw_tail_len:
            state.can_accept_prefix_len = int(state.committed_len)
        state.tail_tokens.clear()
        state.pending_expected_tokens.extend(commit_tokens[match_len:])

    def close_request(self, message: DraftClose) -> None:
        self.close_requests([message])

    def close_requests(self, messages: Sequence[DraftClose]) -> None:
        if not messages:
            return
        with self._condition:
            self._ensure_open_locked()
            for message in messages:
                self._close_request_locked(message)
            self._condition.notify_all()

    def _close_request_locked(self, message: DraftClose) -> None:
        if int(message.src_verifier_rank) != self.verifier_rank:
            raise RuntimeError(
                "DraftClose belongs to a different verifier: "
                f"request_id={message.request_id} "
                f"expected_verifier_rank={self.verifier_rank} "
                f"src_verifier_rank={message.src_verifier_rank}"
            )
        state = self._states.get(message.request_id)
        if state is not None and int(message.dst_drafter_rank) != int(
            state.drafter_rank
        ):
            raise RuntimeError(
                "DraftClose targets a different drafter: "
                f"request_id={message.request_id} "
                f"expected_drafter_rank={state.drafter_rank} "
                f"dst_drafter_rank={message.dst_drafter_rank}"
            )
        self._states.pop(message.request_id, None)

    def apply_control_batch(self, batch: DraftControlBatch) -> None:
        with self._condition:
            self._ensure_open_locked()
            for message in batch.sync_messages:
                self._open_request_locked(message)
            for message in batch.verify_commit_messages:
                self._apply_commit_locked(message)
            for message in batch.close_messages:
                self._close_request_locked(message)
            self._condition.notify_all()

    def append_draft_stream_batch(self, batch: DraftTailStreamOutputBatch) -> None:
        if not batch.outputs:
            return
        with self._condition:
            self._ensure_open_locked()
            for output in batch.outputs:
                self._append_one_locked(output)
            self._condition.notify_all()

    def _append_one_locked(self, output: DraftTailStreamOutput) -> None:
        request_id = output.request_id
        base_committed_len = int(output.base_committed_len)
        token_pos = int(output.new_token_pos)
        token = int(output.new_token)
        src_drafter_rank = int(output.src_drafter_rank)

        if int(output.dst_verifier_rank) != self.verifier_rank:
            raise RuntimeError(
                "Draft stream output targets a different verifier: "
                f"request_id={request_id} "
                f"expected_verifier_rank={self.verifier_rank} "
                f"dst_verifier_rank={output.dst_verifier_rank}"
            )

        state = self._states.get(request_id)
        if state is None:
            # Tail output can remain in flight after the verifier closes a row.
            return
        if src_drafter_rank != int(state.drafter_rank):
            raise RuntimeError(
                "Draft stream output came from a different drafter: "
                f"request_id={request_id} "
                f"expected_drafter_rank={state.drafter_rank} "
                f"src_drafter_rank={src_drafter_rank}"
            )

        state_committed_len = int(state.committed_len)
        if state.pending_expected_tokens:
            if state.tail_tokens:
                raise RuntimeError(
                    "Draft tail must be empty while committed tokens are pending: "
                    f"request_id={request_id} tail_tokens={state.tail_tokens}"
                )
            if base_committed_len < int(state.can_accept_prefix_len):
                return
            if token_pos < state_committed_len:
                return
            # A short-tail commit may arrive before the drafter's already-sent
            # continuation. That continuation legitimately keeps the older
            # generation base while targeting exactly the next absolute
            # position. Accept any base still inside the non-stale interval.
            if base_committed_len > state_committed_len:
                return
            if token_pos != state_committed_len:
                return

            expected_token = int(state.pending_expected_tokens[0])
            if token == expected_token:
                state.pending_expected_tokens.popleft()
                state.committed_len += 1
            else:
                state.can_accept_prefix_len = int(state.committed_len)
            return

        if base_committed_len > state_committed_len:
            raise RuntimeError(
                "Draft stream base is ahead of verifier state: "
                f"request_id={request_id} "
                f"base_committed_len={base_committed_len} "
                f"state_committed_len={state_committed_len}"
            )
        if base_committed_len < int(state.can_accept_prefix_len):
            return
        if token_pos < state_committed_len:
            return

        buffer_end_len = state_committed_len + len(state.tail_tokens)
        if token_pos < buffer_end_len:
            existing_token = int(state.tail_tokens[token_pos - state_committed_len])
            if existing_token != token:
                raise RuntimeError(
                    "Draft stream token conflicts with buffered tail: "
                    f"request_id={request_id} token_pos={token_pos} "
                    f"existing_token={existing_token} new_token={token}"
                )
            return
        if token_pos > buffer_end_len:
            if base_committed_len == state_committed_len:
                raise RuntimeError(
                    "Draft stream token skips the buffered tail: "
                    f"request_id={request_id} token_pos={token_pos} "
                    f"buffer_end_len={buffer_end_len}"
                )
            return

        state.tail_tokens.append(token)

    def snapshot(
        self,
        request_id: str,
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> DraftTailSnapshot:
        return self.get_draft_snapshots(
            [request_id],
            allow_partial=allow_partial,
            max_tail_len=max_tail_len,
            timeout_s=timeout_s,
        )[0]

    def get_draft_snapshots(
        self,
        request_ids: Sequence[str],
        *,
        allow_partial: bool = True,
        max_tail_len: int | None = None,
        timeout_s: float | None = None,
    ) -> list[DraftTailSnapshot]:
        tail_cap = None if max_tail_len is None else max(0, int(max_tail_len))
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)

        with self._condition:
            self._ensure_open_locked()
            self._require_requests_locked(request_ids)
            self.last_draft_wait_ns = 0
            if not allow_partial:
                required_tail_len = self.required_tail_len
                if tail_cap is not None:
                    required_tail_len = min(required_tail_len, tail_cap)
                min_tail_len = max(0 if tail_cap == 0 else 1, required_tail_len)
                wait_start_ns = time.perf_counter_ns()
                while not self._has_min_tail_locked(request_ids, min_tail_len):
                    remaining_s = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining_s is not None and remaining_s <= 0:
                        self.last_draft_wait_ns = time.perf_counter_ns() - wait_start_ns
                        raise TimeoutError(
                            "Timed out waiting for decoupled draft tails: "
                            f"request_ids={list(request_ids)} "
                            f"min_tail_len={min_tail_len}"
                        )
                    self._condition.wait(timeout=remaining_s)
                    self._ensure_open_locked()
                self.last_draft_wait_ns = time.perf_counter_ns() - wait_start_ns

            snapshots: list[DraftTailSnapshot] = []
            for request_id in request_ids:
                state = self._states[request_id]
                consumable_tail = state.consumable_tail()
                snapshot_tail = (
                    consumable_tail if tail_cap is None else consumable_tail[:tail_cap]
                )
                snapshots.append(
                    DraftTailSnapshot(
                        request_id=request_id,
                        committed_len=int(state.committed_len),
                        tail_tokens=snapshot_tail,
                        raw_tail_len=len(state.tail_tokens),
                        num_consumable_drafts=len(consumable_tail),
                    )
                )
            return snapshots

    def _has_min_tail_locked(
        self, request_ids: Sequence[str], min_tail_len: int
    ) -> bool:
        for request_id in request_ids:
            state = self._states[request_id]
            if state.pending_expected_tokens or len(state.tail_tokens) < min_tail_len:
                return False
        return True

    def _require_requests_locked(self, request_ids: Sequence[str]) -> None:
        missing_request_ids = [
            request_id for request_id in request_ids if request_id not in self._states
        ]
        if missing_request_ids:
            raise KeyError(
                "Missing decoupled draft-tail requests: "
                f"request_ids={missing_request_ids}"
            )

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("DraftTailBuffer is closed")

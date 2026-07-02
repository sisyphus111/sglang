from __future__ import annotations

import atexit
import csv
import glob
import json
import os
import struct
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from torch.utils.cpp_extension import load

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftControlInbox,
    DraftReqKey,
    DraftSync,
    DraftTailStreamOutput,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
    VerifierCommitSegment,
    VerifyCommit,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailSnapshot
from sglang.srt.speculative.tracer import (
    NullSpecTracer,
    SpecTraceEvent,
    trace_speculative,
)
from sglang.srt.utils.network import get_free_port

_MAGIC = b"DSC1"
_VERSION = 1
_KIND_CONTROL_BATCH = 1
_KIND_TAIL_STREAM_BATCH = 2
_KIND_STRING_LIST = 3
_KIND_EXTRACT_DECISIONS = 4
_PROFILE_FLUSH_INTERVAL_NS = 5_000_000_000


@lru_cache(maxsize=1)
def _load_decoupled_spec_cpp_module():
    csrc_dir = Path(__file__).resolve().parent / "csrc" / "decoupled_spec"
    return load(
        name="sglang_decoupled_spec_pybind",
        sources=[str(csrc_dir / "decoupled_spec_pybind.cpp")],
        extra_cflags=["-O3", "-std=c++17", "-pthread"],
        extra_ldflags=["-ldl", "-pthread"],
        verbose=False,
    )


def _ensure_zmq_lib_path() -> None:
    if os.environ.get("SGLANG_DECOUPLED_SPEC_ZMQ_LIB"):
        return
    try:
        import zmq
    except Exception:
        return
    zmq_dir = Path(zmq.__file__).resolve().parent
    candidates = []
    for parent in [zmq_dir, *zmq_dir.parents]:
        candidates.extend(glob.glob(str(parent / "pyzmq.libs" / "libzmq*.so*")))
    if candidates:
        os.environ["SGLANG_DECOUPLED_SPEC_ZMQ_LIB"] = candidates[0]


class _BinaryWriter:
    def __init__(self, kind: int) -> None:
        self.buf = bytearray()
        self.buf.extend(_MAGIC)
        self.u8(_VERSION)
        self.u8(kind)

    def u8(self, value: int) -> None:
        self.buf.extend(struct.pack("<B", int(value)))

    def u32(self, value: int) -> None:
        self.buf.extend(struct.pack("<I", int(value)))

    def i32(self, value: int) -> None:
        self.buf.extend(struct.pack("<i", int(value)))

    def i64(self, value: int) -> None:
        self.buf.extend(struct.pack("<q", int(value)))

    def string(self, value: str) -> None:
        data = str(value).encode("utf-8")
        self.u32(len(data))
        self.buf.extend(data)

    def int_list(self, values: list[int]) -> None:
        self.u32(len(values))
        for value in values:
            self.i32(value)

    def frame(self) -> bytes:
        return bytes(self.buf)


def _pack_control_batch(batch: DraftControlBatch) -> bytes:
    writer = _BinaryWriter(_KIND_CONTROL_BATCH)
    writer.i32(batch.dst_drafter_rank)
    writer.u32(len(batch.sync_messages))
    for message in batch.sync_messages:
        writer.string(message.request_id)
        writer.i32(message.src_verifier_rank)
        writer.i32(message.dst_drafter_rank)
        writer.int_list([int(x) for x in message.prompt_token_ids])
        writer.int_list([int(x) for x in message.committed_output_ids])
    writer.u32(len(batch.verify_commit_messages))
    for message in batch.verify_commit_messages:
        writer.string(message.request_id)
        writer.i32(message.src_verifier_rank)
        writer.i32(message.dst_drafter_rank)
        writer.i64(message.pre_verify_committed_len)
        writer.int_list([int(x) for x in message.committed_token_ids])
    writer.u32(len(batch.close_messages))
    for message in batch.close_messages:
        writer.string(message.request_id)
        writer.i32(message.src_verifier_rank)
        writer.i32(message.dst_drafter_rank)
        writer.string(message.reason)
    return writer.frame()


def _pack_tail_stream_batch(batch: DraftTailStreamOutputBatch) -> bytes:
    writer = _BinaryWriter(_KIND_TAIL_STREAM_BATCH)
    writer.u32(len(batch.outputs))
    for output in batch.outputs:
        writer.i32(output.src_drafter_rank)
        writer.i32(output.dst_verifier_rank)
        writer.string(output.request_id)
        writer.i64(output.base_committed_len)
        writer.i64(output.new_token_pos)
        writer.i32(output.new_token_id)
    return writer.frame()


def _pack_string_list(values: list[str]) -> bytes:
    writer = _BinaryWriter(_KIND_STRING_LIST)
    writer.u32(len(values))
    for value in values:
        writer.string(value)
    return writer.frame()


def _pack_extract_decisions(
    decisions: list[tuple[VerifierCommitSegment, int]]
) -> bytes:
    writer = _BinaryWriter(_KIND_EXTRACT_DECISIONS)
    writer.u32(len(decisions))
    for segment, consumable_len in decisions:
        writer.string(segment.draft_key.request_id)
        writer.i32(segment.draft_key.src_verifier_rank)
        writer.i32(segment.dst_drafter_rank)
        writer.i64(segment.pre_verify_committed_len)
        writer.i64(consumable_len)
    return writer.frame()


def _json_dict(payload: str | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return json.loads(payload)


def _draft_key_from_json(item: dict[str, Any]) -> DraftReqKey:
    return DraftReqKey(
        src_verifier_rank=int(item["src_verifier_rank"]),
        request_id=str(item["request_id"]),
    )


def _sync_from_json(item: dict[str, Any]) -> DraftSync:
    return DraftSync(
        request_id=str(item["request_id"]),
        src_verifier_rank=int(item["src_verifier_rank"]),
        dst_drafter_rank=int(item["dst_drafter_rank"]),
        prompt_token_ids=[int(x) for x in item.get("prompt_token_ids", [])],
        committed_output_ids=[int(x) for x in item.get("committed_output_ids", [])],
    )


def _segment_from_json(item: dict[str, Any]) -> VerifierCommitSegment:
    return VerifierCommitSegment(
        draft_key=_draft_key_from_json(item["draft_key"]),
        dst_drafter_rank=int(item["dst_drafter_rank"]),
        pre_verify_committed_len=int(item["pre_verify_committed_len"]),
        committed_token_ids=[int(x) for x in item.get("committed_token_ids", [])],
    )


def _ready_controls_from_json(payload: str | None) -> ReadyDraftControls:
    data = _json_dict(payload) or {}
    return ReadyDraftControls(
        sync_messages=[_sync_from_json(item) for item in data.get("sync_messages", [])],
        close_keys={
            _draft_key_from_json(item) for item in data.get("close_keys", [])
        },
        ready_commit_segments=[
            _segment_from_json(item) for item in data.get("ready_commit_segments", [])
        ],
    )


def _write_cpp_profile(tracer: Any, component: str, payload: str | None) -> None:
    if not payload:
        return
    output_dir = getattr(tracer, "output_dir", None)
    if output_dir is None:
        return

    profile_rows = json.loads(payload)
    if not profile_rows:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.time_ns()
    json_path = output_path / f"{component}__cpp_profile.json"
    csv_path = output_path / f"{component}__cpp_profile.csv"

    with json_path.open("w") as f:
        json.dump(profile_rows, f, indent=2)

    fieldnames = [
        "wall_time_ns",
        "component",
        "op",
        "count",
        "total_ns",
        "p50_ns",
        "p95_ns",
        "max_ns",
        "items",
        "total_items",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile_rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["wall_time_ns"] = timestamp
            out["component"] = component
            writer.writerow(out)


class CppDraftTailBuffer:
    def __init__(self, *, verifier_rank: int, required_tail_len: int) -> None:
        module = _load_decoupled_spec_cpp_module()
        self._cpp = module.DraftTailBuffer(int(verifier_rank), int(required_tail_len))
        self.verifier_rank = int(verifier_rank)
        self.required_tail_len = max(0, int(required_tail_len))

    def close(self) -> None:
        self._cpp.close()

    def has_request(self, request_id: str) -> bool:
        return bool(self._cpp.has_request(str(request_id)))

    def get_committed_len(self, request_id: str) -> int | None:
        value = int(self._cpp.get_committed_len(str(request_id)))
        return None if value < 0 else value

    def open_requests(self, messages: list[DraftSync]) -> None:
        self.apply_control_batch(DraftControlBatch(0, sync_messages=list(messages)))

    def apply_verify_commits(self, messages: list[VerifyCommit]) -> None:
        dst_rank = int(messages[0].dst_drafter_rank) if messages else 0
        self.apply_control_batch(
            DraftControlBatch(dst_rank, verify_commit_messages=list(messages))
        )

    def close_requests(self, messages: list[DraftClose]) -> None:
        dst_rank = int(messages[0].dst_drafter_rank) if messages else 0
        self.apply_control_batch(DraftControlBatch(dst_rank, close_messages=list(messages)))

    def apply_control_batch(
        self, batch: DraftControlBatch, *, collect_stats: bool = False
    ) -> dict[str, Any] | None:
        return _json_dict(
            self._cpp.apply_control_batch(
                _pack_control_batch(batch), bool(collect_stats)
            )
        )

    def append_draft_stream_batch(
        self, batch: DraftTailStreamOutputBatch, *, collect_stats: bool = False
    ) -> dict[str, Any] | None:
        if not batch.outputs:
            return None
        return _json_dict(
            self._cpp.append_draft_stream_batch(
                _pack_tail_stream_batch(batch), bool(collect_stats)
            )
        )

    def wait_for_draft_tokens(self, rids: list[str], min_draft_tokens: int) -> None:
        self._cpp.wait_for_draft_tokens(
            _pack_string_list([str(rid) for rid in rids]), int(min_draft_tokens)
        )

    def get_draft_snapshots(
        self,
        reqs: list,
        *,
        allow_partial: bool = True,
        include_raw_tail_tokens: bool = False,
        max_tail_len: int | None = None,
    ) -> list[DraftTailSnapshot]:
        rids = [str(req.rid) for req in reqs]
        tail_cap = -1 if max_tail_len is None else max(0, int(max_tail_len))
        payload = self._cpp.get_draft_snapshots(
            _pack_string_list(rids),
            bool(allow_partial),
            bool(include_raw_tail_tokens),
            int(tail_cap),
        )
        rows = json.loads(payload or "[]")
        return [
            DraftTailSnapshot(
                request_id=str(row["request_id"]),
                committed_len=int(row["committed_len"]),
                tail_tokens=[int(x) for x in row.get("tail_tokens", [])],
                raw_tail_len=int(row.get("raw_tail_len", 0)),
                raw_tail_tokens=[int(x) for x in row.get("raw_tail_tokens", [])],
            )
            for row in rows
        ]

    def profile_json(self) -> str:
        return self._cpp.profile_json()


class _CppDraftControlInboxProxy:
    def __init__(self, cpp_token_sync_thread: Any) -> None:
        self._cpp = cpp_token_sync_thread

    def is_empty(self) -> bool:
        return bool(self._cpp.pending_control_count() == 0)

    def pending_control_count(self) -> int:
        return int(self._cpp.pending_control_count())

    def extract_ready_controls_locked(
        self,
        consumable_commit_len: Callable[[VerifierCommitSegment], int],
    ) -> ReadyDraftControls:
        if int(self._cpp.pending_control_count()) <= 0:
            return ReadyDraftControls()
        pending = json.loads(self._cpp.snapshot_pending_commit_segments() or "[]")
        decisions: list[tuple[VerifierCommitSegment, int]] = []
        for item in pending:
            segment = _segment_from_json(item)
            consumable_len = int(consumable_commit_len(segment))
            if consumable_len > 0:
                decisions.append((segment, consumable_len))
        return _ready_controls_from_json(
            self._cpp.extract_ready_controls(_pack_extract_decisions(decisions))
        )


class CppTokenSyncThread:
    def __init__(
        self,
        context: Any | None = None,
        drafter_rank: int = 0,
        tracer: Any = None,
    ) -> None:
        _ensure_zmq_lib_path()
        module = _load_decoupled_spec_cpp_module()
        bind_endpoint = f"tcp://127.0.0.1:{get_free_port()}"
        self._cpp = module.TokenSyncThread(int(drafter_rank), bind_endpoint)
        self.context = context
        self.drafter_rank = int(drafter_rank)
        self.control_bind_endpoint = str(self._cpp.control_bind_endpoint())
        self.verifier_result_endpoints: list[str] = []
        self.tracer = tracer or NullSpecTracer()
        self._last_profile_flush_ns = 0
        atexit.register(self._flush_cpp_profile)

    def start(self) -> None:
        self._cpp.start()

    def close(self) -> None:
        try:
            self._cpp.close()
        finally:
            self._flush_cpp_profile()

    def _flush_cpp_profile(self) -> None:
        _write_cpp_profile(self.tracer, "token_sync_thread", self._cpp.profile_json())

    def _maybe_flush_cpp_profile(self) -> None:
        now = time.time_ns()
        if now - self._last_profile_flush_ns >= _PROFILE_FLUSH_INTERVAL_NS:
            self._last_profile_flush_ns = now
            self._flush_cpp_profile()

    def configure_peer_endpoints(self, verifier_result_endpoints: list[str]) -> None:
        endpoints = [str(endpoint) for endpoint in verifier_result_endpoints]
        self._cpp.configure_peer_endpoints(_pack_string_list(endpoints))
        self.verifier_result_endpoints = endpoints

    @trace_speculative(SpecTraceEvent.TOKEN_SYNC_DRAIN_CONTROL_BATCH)
    def collect_ready_draft_controls(
        self,
        collector: Callable[[DraftControlInbox], ReadyDraftControls],
    ) -> ReadyDraftControls:
        result = collector(_CppDraftControlInboxProxy(self._cpp))  # type: ignore[arg-type]
        self._maybe_flush_cpp_profile()
        return result

    @trace_speculative(SpecTraceEvent.TOKEN_SYNC_ENQUEUE_DRAFT_RESULT_BATCH)
    def submit_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return
        self._cpp.submit_draft_results(_pack_tail_stream_batch(result_batch))
        self._maybe_flush_cpp_profile()


class CppDraftProxyThread:
    def __init__(
        self,
        *,
        context: Any,
        verifier_rank: int,
        draft_tail_buffer: CppDraftTailBuffer,
        tracer: Any = None,
    ) -> None:
        _ensure_zmq_lib_path()
        module = _load_decoupled_spec_cpp_module()
        bind_endpoint = f"tcp://127.0.0.1:{get_free_port()}"
        self._cpp = module.DraftProxyThread(
            int(verifier_rank), bind_endpoint, draft_tail_buffer._cpp
        )
        self.context = context
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self.tracer = tracer or NullSpecTracer()
        self.result_bind_endpoint = str(self._cpp.result_bind_endpoint())
        self.drafter_control_endpoints: list[str] = []
        self._last_profile_flush_ns = 0
        atexit.register(self._flush_cpp_profile)

    def start(self) -> None:
        self._cpp.start()

    def close(self) -> None:
        try:
            self._cpp.close()
        finally:
            self._flush_cpp_profile()

    def _flush_cpp_profile(self) -> None:
        _write_cpp_profile(self.tracer, "draft_proxy", self._cpp.profile_json())
        _write_cpp_profile(
            self.tracer,
            "draft_tail_buffer",
            self.draft_tail_buffer.profile_json(),
        )

    def _maybe_flush_cpp_profile(self) -> None:
        now = time.time_ns()
        if now - self._last_profile_flush_ns >= _PROFILE_FLUSH_INTERVAL_NS:
            self._last_profile_flush_ns = now
            self._flush_cpp_profile()

    def configure_peer_endpoints(self, drafter_control_endpoints: list[str]) -> None:
        endpoints = [str(endpoint) for endpoint in drafter_control_endpoints]
        self._cpp.configure_peer_endpoints(_pack_string_list(endpoints))
        self.drafter_control_endpoints = endpoints

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        if not self.drafter_control_endpoints:
            raise RuntimeError("Decoupled verify peer endpoints are not configured")
        self._apply_control_batch(batch)
        self._enqueue_control_batch(batch)

    @trace_speculative(
        SpecTraceEvent.DRAFT_PROXY_APPLY_CONTROL_BATCH,
        inject_trace_enabled="collect_trace_stats",
    )
    def _apply_control_batch(
        self,
        batch: DraftControlBatch,
        *,
        collect_trace_stats: bool = False,
    ) -> dict[str, Any] | None:
        result = self.draft_tail_buffer.apply_control_batch(
            batch, collect_stats=collect_trace_stats
        )
        self._maybe_flush_cpp_profile()
        return result

    @trace_speculative(SpecTraceEvent.DRAFT_PROXY_SEND_CONTROL_BATCH)
    def _enqueue_control_batch(self, batch: DraftControlBatch) -> None:
        self._cpp.submit_control_batch(_pack_control_batch(batch))
        self._maybe_flush_cpp_profile()

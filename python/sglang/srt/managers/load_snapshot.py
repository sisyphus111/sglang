"""Load snapshot: publish scheduler load metrics for DP balancing and /v1/loads.

Architecture
------------

Each scheduler periodically publishes a ``LoadSnapshot`` containing its
current load metrics (running reqs, tokens, throughput, ...).  Two
transport backends are supported:

**SHM mode** (single-node, default)::

    Scheduler  ──ShmLoadSnapshotWriter──▶  /dev/shm mmap file
                                               ▲
    TokenizerManager  ──ShmLoadSnapshotReader───┘  (for /v1/loads)
    DataParallelController  ──ShmLoadSnapshotReader─┘  (for dispatch)

**ZMQ mode** (multi-node DP attention, or ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1``)::

    Scheduler (any node)  ──ZmqLoadSnapshotWriter (PUSH)──▶  network
                                                               │
    ZmqShmLoadSnapshotReader (PULL, node 0)  ◀─────────────────┘
        │  drains zmq, writes to SHM
        ▼
    /dev/shm mmap file (node 0)
        ▲
    TokenizerManager / DataParallelController  ──ShmLoadSnapshotReader──┘

Shared memory does not work across nodes, so multi-node DP attention
requires the ZMQ transport.  The ``ZmqShmLoadSnapshotReader`` on node 0
receives snapshots from all schedulers via zmq PUSH/PULL and writes them
into the local SHM file.  All readers (TokenizerManager,
DataParallelController) on
node 0 then read from SHM.

``zmq_reader_owner()`` decides which process on node 0 binds the zmq
PULL socket (only one can bind); the other reads plain SHM.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import math
import mmap
import os
import struct
from contextlib import contextmanager
from typing import Optional

import msgspec
import msgspec.msgpack
import msgspec.structs

from sglang.srt.environ import envs
from sglang.srt.utils.network import is_zmq_endpoint_ipv6

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def should_use_zmq(server_args) -> bool:
    """Whether to use zmq PUSH/PULL instead of shared memory for load snapshots.

    Shared memory (mmap) only works within a single node.  When schedulers
    run on multiple nodes (multi-node DP attention), they cannot write to
    the SHM file on node 0, so we fall back to zmq transport.  The env var
    ``SGLANG_LOAD_SNAPSHOT_USE_ZMQ`` forces zmq mode for testing.
    """
    return (
        server_args.enable_dp_attention and server_args.nnodes > 1
    ) or envs.SGLANG_LOAD_SNAPSHOT_USE_ZMQ.get()


_LOAD_AWARE_METHODS = frozenset({"total_requests", "total_tokens"})


def _tokenizer_load_snapshot_owner_caller(server_args) -> str:
    """The caller that plays the tokenizer-side zmq owner role.

    In multi-tokenizer mode (``tokenizer_worker_num > 1``) there are N
    independent ``TokenizerWorker`` processes that would all try to bind the
    same zmq PULL endpoint.  Instead, the single ``MultiTokenizerRouter``
    process owns the socket (polls zmq -> SHM) and every worker reads SHM.
    """
    if server_args.tokenizer_worker_num > 1:
        return "MultiTokenizerRouter"
    return "TokenizerManager"


def zmq_reader_owner(server_args, caller: str) -> bool:
    """Decide which process owns the zmq PULL socket.

    Exactly one of ``"DataParallelController"``, ``"TokenizerManager"``, or
    ``"MultiTokenizerRouter"`` must return True when zmq mode is active.  The
    owner polls zmq -> SHM; the others read SHM.

    Rules:
      - Non-zero node_rank: no TokenizerManager, DataParallelController only
        launches schedulers and waits -> nobody owns it.
      - dp_size == 1: no DataParallelController exists -> tokenizer-side owner
        owns it.
      - dp_size > 1, load-aware method: DataParallelController polls on every
        dispatch via refresh_load_budget() -> DataParallelController owns it.
      - dp_size > 1, round-robin / other: DataParallelController never reads
        load data -> tokenizer-side owner owns it (polls on /v1/loads calls).

    The tokenizer-side owner is the ``"MultiTokenizerRouter"`` caller in
    multi-tokenizer mode, otherwise the ``"TokenizerManager"`` caller.
    """
    if not should_use_zmq(server_args):
        return False
    if server_args.node_rank != 0:
        return False
    tokenizer_owner = _tokenizer_load_snapshot_owner_caller(server_args)
    if server_args.dp_size == 1:
        return caller == tokenizer_owner
    if server_args.load_balance_method.lower() in _LOAD_AWARE_METHODS:
        return caller == "DataParallelController"
    return caller == tokenizer_owner


# ---------------------------------------------------------------------------
# LoadSnapshot data class
# ---------------------------------------------------------------------------


class MemoryMetrics(msgspec.Struct, array_like=True):
    """Memory breakdown metrics."""

    weight_gb: float
    kv_cache_gb: float
    graph_gb: float
    token_capacity: int


class SpeculativeMetrics(msgspec.Struct, array_like=True):
    """Speculative decoding metrics."""

    accept_length: float
    accept_rate: float
    draft_occupancy_rate: float
    proposed_draft_length: float = 0.0


class IntegerHistogram(msgspec.Struct, array_like=True, frozen=True):
    """Compact histogram over consecutive integer values.

    ``counts[index]`` represents ``offset + index``. Values outside that
    range remain visible through the underflow/overflow counters.
    """

    offset: int
    counts: list[int]
    underflow_count: int = 0
    overflow_count: int = 0


class LatencyHistogram(msgspec.Struct, array_like=True, frozen=True):
    """Mergeable role-local latency histogram reported by the native plane.

    Bucket counts are non-cumulative. The final count is the overflow bucket,
    so its length is exactly one greater than ``bucket_upper_bounds_us``.
    """

    count: int
    sum_us: int
    bucket_upper_bounds_us: list[int]
    bucket_counts: list[int]

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> Optional[LatencyHistogram]:
        if value is None:
            return None
        histogram = cls(
            count=int(value.get("count", 0)),
            sum_us=int(value.get("sum_us", 0)),
            bucket_upper_bounds_us=[
                int(bound) for bound in value.get("bucket_upper_bounds_us", ())
            ],
            bucket_counts=[
                int(bucket_count) for bucket_count in value.get("bucket_counts", ())
            ],
        )
        if len(histogram.bucket_counts) != len(histogram.bucket_upper_bounds_us) + 1:
            raise RuntimeError(
                "Decoupled-spec latency histogram must include one overflow bucket: "
                f"bounds={len(histogram.bucket_upper_bounds_us)} "
                f"counts={len(histogram.bucket_counts)}"
            )
        if (
            histogram.count < 0
            or histogram.sum_us < 0
            or any(count < 0 for count in histogram.bucket_counts)
            or any(bound < 0 for bound in histogram.bucket_upper_bounds_us)
            or any(
                left >= right
                for left, right in zip(
                    histogram.bucket_upper_bounds_us,
                    histogram.bucket_upper_bounds_us[1:],
                )
            )
        ):
            raise RuntimeError(
                "Decoupled-spec latency histogram values must be non-negative "
                "with strictly increasing bounds."
            )
        if sum(histogram.bucket_counts) != histogram.count:
            raise RuntimeError(
                "Decoupled-spec latency histogram count disagrees with its buckets: "
                f"count={histogram.count} buckets={sum(histogram.bucket_counts)}"
            )
        return histogram


class DraftTailSelectMetrics(msgspec.Struct, array_like=True, frozen=True):
    """Verifier GPU-tail selector counters for one decode window."""

    num_select_rows: int
    num_select_valid_rows: int
    reason_counts: dict[str, int]
    selected_draft_length_histogram: IntegerHistogram
    raw_draft_tail_length_histogram: IntegerHistogram
    consumable_draft_tail_length_histogram: IntegerHistogram
    logical_delta_histogram: IntegerHistogram
    pending_prefix_length_histogram: IntegerHistogram
    num_publish_seq_initial: int = 0
    num_publish_seq_same: int = 0
    num_publish_seq_advance: int = 0
    num_pending_prefix_fast_forwards: int = 0
    num_protocol_errors: int = 0
    num_seqlock_retry_rows: int = 0
    num_seqlock_retries: int = 0
    max_seqlock_retries: int = 0


class DraftTransportMetrics(msgspec.Struct, array_like=True, frozen=True):
    """Role-local draft-result transport counters for one decode window."""

    num_draft_result_frames: int
    num_draft_result_tokens: int
    draft_send_queue_latency_us: Optional[LatencyHistogram] = None
    draft_send_queue_depth_max: Optional[int] = None
    draft_receive_to_gpu_publish_enqueue_latency_us: Optional[LatencyHistogram] = None
    draft_gpu_publish_completion_latency_us: Optional[LatencyHistogram] = None
    draft_transport_one_way_latency_us: Optional[LatencyHistogram] = None
    draft_result_ready_to_receive_latency_us: Optional[LatencyHistogram] = None
    gpu_publish_staging_slots_max: Optional[int] = None
    clock_sync_valid: Optional[bool] = None
    clock_error_bound_us: Optional[float] = None
    num_clock_sync_valid_peers: Optional[int] = None
    num_clock_sync_invalid_peers: Optional[int] = None

    @classmethod
    def from_dict(cls, value: dict) -> DraftTransportMetrics:
        """Validate and freeze one native exchange window without quantiles."""

        metrics = cls(
            num_draft_result_frames=int(value.get("num_draft_result_frames", 0)),
            num_draft_result_tokens=int(value.get("num_draft_result_tokens", 0)),
            draft_send_queue_latency_us=LatencyHistogram.from_dict(
                value.get("draft_send_queue_latency_us")
            ),
            draft_send_queue_depth_max=(
                None
                if value.get("draft_send_queue_depth_max") is None
                else int(value["draft_send_queue_depth_max"])
            ),
            draft_receive_to_gpu_publish_enqueue_latency_us=(
                LatencyHistogram.from_dict(
                    value.get("draft_receive_to_gpu_publish_enqueue_latency_us")
                )
            ),
            draft_gpu_publish_completion_latency_us=LatencyHistogram.from_dict(
                value.get("draft_gpu_publish_completion_latency_us")
            ),
            draft_transport_one_way_latency_us=LatencyHistogram.from_dict(
                value.get("draft_transport_one_way_latency_us")
            ),
            draft_result_ready_to_receive_latency_us=LatencyHistogram.from_dict(
                value.get("draft_result_ready_to_receive_latency_us")
            ),
            gpu_publish_staging_slots_max=(
                None
                if value.get("gpu_publish_staging_slots_max") is None
                else int(value["gpu_publish_staging_slots_max"])
            ),
            clock_sync_valid=(
                None
                if value.get("clock_sync_valid") is None
                else bool(value["clock_sync_valid"])
            ),
            clock_error_bound_us=(
                None
                if value.get("clock_error_bound_us") is None
                else float(value["clock_error_bound_us"])
            ),
            num_clock_sync_valid_peers=(
                None
                if value.get("num_clock_sync_valid_peers") is None
                else int(value["num_clock_sync_valid_peers"])
            ),
            num_clock_sync_invalid_peers=(
                None
                if value.get("num_clock_sync_invalid_peers") is None
                else int(value["num_clock_sync_invalid_peers"])
            ),
        )
        if (
            metrics.num_draft_result_frames < 0
            or metrics.num_draft_result_tokens < 0
            or (
                metrics.draft_send_queue_depth_max is not None
                and metrics.draft_send_queue_depth_max < 0
            )
            or (
                metrics.gpu_publish_staging_slots_max is not None
                and metrics.gpu_publish_staging_slots_max < 0
            )
            or (
                metrics.clock_error_bound_us is not None
                and (
                    not math.isfinite(metrics.clock_error_bound_us)
                    or metrics.clock_error_bound_us < 0
                )
            )
            or (
                metrics.num_clock_sync_valid_peers is not None
                and metrics.num_clock_sync_valid_peers < 0
            )
            or (
                metrics.num_clock_sync_invalid_peers is not None
                and metrics.num_clock_sync_invalid_peers < 0
            )
        ):
            raise RuntimeError(
                "Decoupled-spec transport counters must be non-negative."
            )
        return metrics


class DecoupledSpecDecodeMetrics(msgspec.Struct, array_like=True, frozen=True):
    """One aligned decoupled-spec observability window."""

    tail_select: Optional[DraftTailSelectMetrics] = None
    transport: Optional[DraftTransportMetrics] = None
    adaptive_verify: Optional[dict] = None


def _decode_metrics_window_to_dict(window: DecodeMetricsWindow) -> dict:
    value = msgspec.structs.asdict(window)
    decoupled_spec = window.decoupled_spec
    if decoupled_spec is None:
        value.pop("decoupled_spec", None)
        return value

    decoupled_value = {
        key: item
        for key, item in msgspec.structs.asdict(decoupled_spec).items()
        if item is not None
    }
    if decoupled_spec.tail_select is not None:
        tail_select = decoupled_spec.tail_select
        tail_value = msgspec.structs.asdict(tail_select)
        for field in (
            "selected_draft_length_histogram",
            "raw_draft_tail_length_histogram",
            "consumable_draft_tail_length_histogram",
            "logical_delta_histogram",
            "pending_prefix_length_histogram",
        ):
            tail_value[field] = msgspec.structs.asdict(getattr(tail_select, field))
        decoupled_value["tail_select"] = tail_value
    if decoupled_spec.transport is not None:
        transport = decoupled_spec.transport
        transport_value = {
            key: item
            for key, item in msgspec.structs.asdict(transport).items()
            if item is not None
        }
        for field in (
            "draft_send_queue_latency_us",
            "draft_receive_to_gpu_publish_enqueue_latency_us",
            "draft_gpu_publish_completion_latency_us",
            "draft_transport_one_way_latency_us",
            "draft_result_ready_to_receive_latency_us",
        ):
            histogram = getattr(transport, field)
            if histogram is not None:
                transport_value[field] = msgspec.structs.asdict(histogram)
        decoupled_value["transport"] = transport_value
    value["decoupled_spec"] = decoupled_value
    return value


class DecodeMetricsWindow(msgspec.Struct, array_like=True, frozen=True):
    """One fixed-iteration decode window exposed through ``/v1/loads``."""

    window_id: int
    end_time: float
    num_decode_iters: int
    iter_latency_ms: float
    num_decode_rows: int = 0
    sum_context_lens: int = 0
    mean_batch_size: Optional[float] = None
    mean_context_length: Optional[float] = None
    num_verify_rows: int = 0
    num_accept_tokens: int = 0
    num_proposed_drafts: int = 0
    accept_length: Optional[float] = None
    proposed_draft_length: Optional[float] = None
    decoupled_spec: Optional[DecoupledSpecDecodeMetrics] = None


class LoRAMetrics(msgspec.Struct, array_like=True):
    """LoRA adapter pool metrics."""

    slots_used: int
    slots_total: int
    utilization: float


class DisaggregationMetrics(msgspec.Struct, array_like=True):
    """PD disaggregation metrics."""

    mode: str  # "prefill", "decode", or "null"
    prefill_bootstrap_queue_reqs: int = 0
    prefill_inflight_queue_reqs: int = 0
    decode_prealloc_queue_reqs: int = 0
    decode_transfer_queue_reqs: int = 0
    decode_retracted_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0


class QueueMetrics(msgspec.Struct, array_like=True):
    """Detailed queue info breakdown."""

    waiting: int
    grammar: int
    paused: int
    retracted: int
    prealloc_ready: int


# LoadSnapshot's nested sub-struct fields; every other struct field is a flat
# scalar returned under "core".
_SECTION_FIELDS = frozenset(
    {"memory", "speculative", "lora", "disaggregation", "queues"}
)


class LoadSnapshot(msgspec.Struct, omit_defaults=True):
    """Per-DP-rank load metrics: the SHM/zmq wire format and the /v1/loads source."""

    timestamp: float = 0.0
    dp_rank: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_waiting_uncached_tokens: int = 0
    num_used_tokens: int = 0
    num_total_tokens: int = 0
    # num_total_tokens minus tokens still awaiting a KV transfer (equal to it
    # outside disaggregated decode).
    num_active_tokens: int = 0
    max_total_num_tokens: int = 0
    max_running_requests: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    utilization: float = 0.0
    # cumulative counters
    total_prefill_uncached_tokens: int = 0
    total_prefill_busy_us: int = 0
    # Decode step-time moment sums
    decode_moments: Optional[list[float]] = None
    # Completed fixed-iteration windows. The SHM representation is compact
    # array-like data; ``to_dict`` expands it into named HTTP fields.
    decode_metrics_windows: Optional[list[DecodeMetricsWindow]] = None

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None

    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    def to_dict(self, include: Optional[set[str]] = None) -> dict:
        load = {key: getattr(self, key) for key in _CORE_KEYS}
        if self.decode_metrics_windows is not None:
            load["decode_metrics_windows"] = [
                _decode_metrics_window_to_dict(window)
                for window in self.decode_metrics_windows
            ]

        if include is None or "all" in include:
            include_all = True
        else:
            if not (include <= self.VALID_SECTIONS):
                raise ValueError(
                    f"Invalid include sections: {include - self.VALID_SECTIONS}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )
            if include == {"core"}:
                return load
            include_all = False

        for field, include_name, section in (
            ("memory", "memory", self.memory),
            ("speculative", "spec", self.speculative),
            ("lora", "lora", self.lora),
            ("disaggregation", "disagg", self.disaggregation),
            ("queues", "queues", self.queues),
        ):
            if section is None or (not include_all and include_name not in include):
                continue
            load[field] = msgspec.structs.asdict(section)

        return load


# Flat scalar fields returned under "core": every struct field but the sections.
_CORE_KEYS = tuple(
    f for f in LoadSnapshot.__struct_fields__ if f not in _SECTION_FIELDS
)


def _enc_hook(obj):
    """Coerce numpy scalars to native Python; msgpack has no numpy types."""
    to_item = getattr(obj, "item", None)
    if to_item is not None:
        return to_item()
    raise NotImplementedError(f"cannot encode {type(obj).__name__} in load snapshot")


snapshot_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)
snapshot_decoder = msgspec.msgpack.Decoder(LoadSnapshot)


# ---------------------------------------------------------------------------
# SHM file layout utilities
# ---------------------------------------------------------------------------

MAGIC = b"SLNS"
VERSION = 2
HEADER_STRUCT = struct.Struct("<4sHHI")
SLOT_LEN_STRUCT = struct.Struct("<I")
# A full 64-window history includes selector, transport, and adaptive-decision
# telemetry. Keep one bounded slot per DP rank rather than dropping fields.
SLOT_SIZE = 512 * 1024


@contextmanager
def file_lock(fd: int, lock_type: int):
    fcntl.flock(fd, lock_type)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def shm_path_for(ipc_name: str) -> str:
    name = os.path.basename(ipc_name.rstrip("/")) or "default"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    digest = hashlib.blake2s(ipc_name.encode(), digest_size=4).hexdigest()
    return f"/dev/shm/sglang_loads_{safe_name}_{digest}.shm"


def file_size(dp_size: int, slot_size: int = SLOT_SIZE) -> int:
    return HEADER_STRUCT.size + dp_size * slot_size


def slot_offset(dp_rank: int, slot_size: int = SLOT_SIZE) -> int:
    return HEADER_STRUCT.size + dp_rank * slot_size


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


class ShmLoadSnapshotWriter:
    def __init__(
        self, path: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0

        self.path = path
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.slot_size = SLOT_SIZE
        self.fd = -1
        size = file_size(dp_size, self.slot_size)

        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            with file_lock(self.fd, fcntl.LOCK_EX):
                os.ftruncate(self.fd, size)
                self.mmap = mmap.mmap(self.fd, size, access=mmap.ACCESS_WRITE)
                HEADER_STRUCT.pack_into(
                    self.mmap, 0, MAGIC, VERSION, dp_size, self.slot_size
                )
                self._write_payload(LoadSnapshot(dp_rank=dp_rank))
        except Exception:
            if self.fd >= 0:
                os.close(self.fd)
            raise

    def write(self, snapshot: LoadSnapshot) -> None:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match writer dp_rank={self.dp_rank}"
            )

        with file_lock(self.fd, fcntl.LOCK_EX):
            self._write_payload(snapshot)

    def _write_payload(self, snapshot: LoadSnapshot) -> None:
        payload = snapshot_encoder.encode(snapshot)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if len(payload) > max_payload_size:
            raise ValueError(
                f"load snapshot payload size {len(payload)} exceeds slot payload "
                f"capacity {max_payload_size}"
            )

        offset = slot_offset(self.dp_rank, self.slot_size)
        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + len(payload)

        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, 0)
        self.mmap[payload_start:payload_end] = payload
        # The length prefix is authoritative. Do not clear the unused tail:
        # publication cost must scale with payload bytes, not slot capacity.
        SLOT_LEN_STRUCT.pack_into(self.mmap, offset, len(payload))

    def close(self) -> None:
        self.mmap.close()
        os.close(self.fd)


class ZmqLoadSnapshotWriter:
    """Sends load snapshots via zmq PUSH to a ZmqShmLoadSnapshotReader.

    CONFLATE is set so only the latest message is kept in the send
    buffer when the reader is slower than the writer.
    """

    def __init__(
        self, endpoint: str, dp_size: int, dp_rank: int, publish_interval: int = 1
    ):
        import zmq as _zmq

        if dp_rank < 0 or dp_rank >= dp_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_size={dp_size}")
        self.publish_interval = max(1, publish_interval)
        self.publish_counter = 0
        self.dp_size = dp_size
        self.dp_rank = dp_rank

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PUSH)
        if is_zmq_endpoint_ipv6(endpoint):
            self._socket.setsockopt(_zmq.IPV6, 1)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.connect(endpoint)

    def write(self, snapshot: LoadSnapshot) -> None:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError(
                f"snapshot dp_rank={snapshot.dp_rank} does not match "
                f"writer dp_rank={self.dp_rank}"
            )
        try:
            self._socket.send(snapshot_encoder.encode(snapshot), self._zmq.NOBLOCK)
        except self._zmq.Again:
            pass

    def close(self) -> None:
        self._socket.close()


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


class ShmLoadSnapshotReader:
    def __init__(self, path: str, dp_size: int):
        self.path = path
        self.dp_size = dp_size
        self.mmap: Optional[mmap.mmap] = None
        self.fd: Optional[int] = None
        self.slot_size = SLOT_SIZE
        self._header_warning_logged = False
        self._attach()

    def _attach(self) -> bool:
        if self.mmap is not None:
            return True

        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return False

        size = os.fstat(fd).st_size
        if size < HEADER_STRUCT.size:
            os.close(fd)
            return False

        try:
            with file_lock(fd, fcntl.LOCK_SH):
                mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                magic, version, dp_size, slot_size = HEADER_STRUCT.unpack_from(
                    mapped, 0
                )
        except (OSError, ValueError):
            os.close(fd)
            return False

        if (
            magic != MAGIC
            or version != VERSION
            or dp_size != self.dp_size
            or slot_size < SLOT_LEN_STRUCT.size
            or size < file_size(self.dp_size, slot_size)
        ):
            mapped.close()
            os.close(fd)
            if not self._header_warning_logged:
                logger.warning("load shm header mismatch at %s", self.path)
                self._header_warning_logged = True
            return False

        self.mmap = mapped
        self.fd = fd
        self.slot_size = slot_size
        return True

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        if dp_rank < 0 or dp_rank >= self.dp_size:
            return None
        if not self._attach():
            return None

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            return self._read_slot(dp_rank)

    def _read_slot(self, dp_rank: int) -> Optional[LoadSnapshot]:
        assert self.mmap is not None
        offset = slot_offset(dp_rank, self.slot_size)
        (payload_len,) = SLOT_LEN_STRUCT.unpack_from(self.mmap, offset)
        max_payload_size = self.slot_size - SLOT_LEN_STRUCT.size
        if payload_len == 0 or payload_len > max_payload_size:
            return None

        payload_start = offset + SLOT_LEN_STRUCT.size
        payload_end = payload_start + payload_len
        try:
            return snapshot_decoder.decode(self.mmap[payload_start:payload_end])
        except Exception as e:
            logger.debug("load snapshot decode failed for rank %s: %s", dp_rank, e)
            return None

    def read_all(self) -> list[LoadSnapshot]:
        if not self._attach():
            return []

        assert self.fd is not None
        with file_lock(self.fd, fcntl.LOCK_SH):
            loads = []
            for r in range(self.dp_size):
                load = self._read_slot(r)
                if load is not None:
                    loads.append(load)
            return loads

    def close(self) -> None:
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class ZmqShmLoadSnapshotReader:
    """Receives snapshots via zmq PULL from writers, writes to SHM, reads from SHM.

    Transparently wraps a ShmLoadSnapshotReader.  Every read() / read_all()
    first drains the PULL socket into SHM so callers always see fresh data.
    """

    def __init__(self, endpoint: str, shm_path: str, dp_size: int):
        import zmq as _zmq

        self._zmq = _zmq
        self._ctx = _zmq.Context.instance()
        self._socket = self._ctx.socket(_zmq.PULL)
        if is_zmq_endpoint_ipv6(endpoint):
            self._socket.setsockopt(_zmq.IPV6, 1)
        self._socket.setsockopt(_zmq.LINGER, 0)
        self._socket.setsockopt(_zmq.CONFLATE, 1)
        self._socket.bind(endpoint)

        self._endpoint = endpoint
        self._shm_path = shm_path
        self.dp_size = dp_size
        self._shm_reader = ShmLoadSnapshotReader(shm_path, dp_size)
        self._shm_writers: dict[int, ShmLoadSnapshotWriter] = {}

    def _poll(self) -> None:
        """Drain zmq messages and write latest per dp_rank to SHM."""
        latest: dict[int, LoadSnapshot] = {}
        while True:
            try:
                data = self._socket.recv(self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            try:
                snapshot = snapshot_decoder.decode(data)
                if 0 <= snapshot.dp_rank < self.dp_size:
                    latest[snapshot.dp_rank] = snapshot
            except Exception as e:
                logger.warning("load snapshot zmq decode failed: %s", e)

        for dp_rank, snapshot in latest.items():
            if dp_rank not in self._shm_writers:
                self._shm_writers[dp_rank] = ShmLoadSnapshotWriter(
                    self._shm_path, self.dp_size, dp_rank
                )
            try:
                self._shm_writers[dp_rank].write(snapshot)
            except Exception as e:
                logger.warning(
                    "load snapshot shm write failed for rank %d: %s", dp_rank, e
                )

    def fileno(self) -> int:
        """Edge-triggered fd that becomes readable when zmq messages arrive.

        Lets an owner process register the reader with an event loop and drain
        it via ``poll()`` instead of polling on a timer.
        """
        return self._socket.getsockopt(self._zmq.FD)

    def poll(self) -> None:
        """Drain the zmq PULL socket into SHM.

        Public entry point so an owner process (e.g. MultiTokenizerRouter) can
        keep SHM fresh without touching internals.
        """
        self._poll()

    def read(self, dp_rank: int) -> Optional[LoadSnapshot]:
        self._poll()
        return self._shm_reader.read(dp_rank)

    def read_all(self) -> list[LoadSnapshot]:
        self._poll()
        return self._shm_reader.read_all()

    def close(self) -> None:
        for w in self._shm_writers.values():
            w.close()
        self._shm_writers.clear()
        self._shm_reader.close()
        self._socket.close()
        if self._endpoint.startswith("ipc://"):
            try:
                os.unlink(self._endpoint[len("ipc://") :])
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _zmq_addr_for(port_args) -> str:
    """Return the zmq PUSH/PULL address from PortArgs.

    For dp_attention (TCP mode), uses the ``load_collector_ipc_name`` field
    stored in PortArgs.  For single-node IPC (env-var override), derives
    a deterministic IPC path from ``instance_id``.
    """
    ipc_name = getattr(port_args, "load_collector_ipc_name", "")
    if ipc_name:
        return ipc_name
    safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in port_args.instance_id
    )
    digest = hashlib.blake2s(port_args.instance_id.encode(), digest_size=4).hexdigest()
    return f"ipc:///tmp/sglang_load_collector_{safe}_{digest}.sock"


def create_load_snapshot_writer(
    server_args,
    port_args,
    dp_size: int,
    dp_rank: int,
    publish_interval: int = 1,
):
    """Return a SHM or ZMQ writer based on server configuration."""
    if should_use_zmq(server_args):
        return ZmqLoadSnapshotWriter(
            _zmq_addr_for(port_args), dp_size, dp_rank, publish_interval
        )
    return ShmLoadSnapshotWriter(
        shm_path_for(port_args.instance_id), dp_size, dp_rank, publish_interval
    )


def create_load_snapshot_reader(server_args, port_args, caller: str):
    """Create a load snapshot reader.

    Args:
        caller: ``"DataParallelController"``, ``"TokenizerManager"``, or
            ``"MultiTokenizerRouter"`` -- determines who binds the zmq PULL
            socket when zmq mode is active.
    """
    dp_size = server_args.dp_size
    if zmq_reader_owner(server_args, caller):
        return ZmqShmLoadSnapshotReader(
            _zmq_addr_for(port_args), shm_path_for(port_args.instance_id), dp_size
        )
    return ShmLoadSnapshotReader(shm_path_for(port_args.instance_id), dp_size)

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import msgspec

_MAX_TRANSPORT_RANK = 2**31 - 1


class DecoupledSpecPeerConfig(msgspec.Struct, frozen=True):
    """One explicitly ranked transport peer and its routing quota."""

    rank: int
    endpoint: str
    quota: int


def _validate_transport_rank(rank: Any, *, field_name: str) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError(f"{field_name} must be an integer, got {rank!r}.")
    if rank < 0 or rank > _MAX_TRANSPORT_RANK:
        raise ValueError(
            f"{field_name} must be in [0, {_MAX_TRANSPORT_RANK}], got {rank}."
        )
    return rank


def _validate_endpoint(endpoint: Any, *, field_name: str) -> str:
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError(f"{field_name} must be a non-empty string, got {endpoint!r}.")
    return endpoint


class DraftMeshMessageType(str, Enum):
    CONTROL_BATCH = "control_batch"
    TAIL_STREAM_OUTPUT_BATCH = "tail_stream_output_batch"


@dataclass(frozen=True)
class DraftReqKey:
    """Request identity on the drafter side.

    The original request_id is only unique within the verifier that owns it.
    src_verifier_rank keeps the drafter-side request table unambiguous when
    multiple verifier ranks send work to the same drafter rank.
    """

    src_verifier_rank: int
    request_id: str


def build_draft_scheduler_rid(draft_key: DraftReqKey) -> str:
    return f"draft:{int(draft_key.src_verifier_rank)}:{draft_key.request_id}"


def parse_draft_scheduler_rid(rid: str) -> DraftReqKey:
    if rid.startswith("draft:"):
        encoded = rid[len("draft:") :]
        rank_text, sep, request_id = encoded.partition(":")
        if sep and request_id:
            return DraftReqKey(
                src_verifier_rank=int(rank_text),
                request_id=request_id,
            )

    raise ValueError(f"Invalid decoupled draft scheduler rid: {rid}")


@dataclass
class DraftSync:
    """Open or re-open a drafter request from a verifier-owned prefix.

    The verifier is the source of truth for committed tokens. DraftSync gives
    the drafter the prompt and already committed output prefix that it must
    align to before it can emit draft tail tokens.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_outputs: list[int] = field(default_factory=list)

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class VerifyCommit:
    """
    Sent from verifier to drafter to commit a portion of the draft outputs.

    committed_tokens is the verifier-committed contiguous output segment:
    output_ids[
        pre_verify_committed_len:
        pre_verify_committed_len + len(committed_tokens)
    ].
    Drafter must align its reqs to these committed tokens,
    and sometimes needs to truncate tokens / reprefill.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    pre_verify_committed_len: int
    committed_tokens: list[int]

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )

    def validate_committed_tokens(self) -> None:
        if not self.committed_tokens:
            raise ValueError(
                "VerifyCommit committed_tokens must be non-empty: "
                f"request_id={self.request_id} "
                f"pre_verify_committed_len={self.pre_verify_committed_len}"
            )
        if int(self.pre_verify_committed_len) < 0:
            raise ValueError(
                "VerifyCommit pre_verify_committed_len must be non-negative: "
                f"request_id={self.request_id} "
                f"pre_verify_committed_len={self.pre_verify_committed_len}"
            )


@dataclass
class DraftClose:
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    reason: str

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class DraftTailStreamOutput:
    """
    Drafter sends one output token to the verifier-side DraftTailBuffer.

    base_committed_len records the verifier prefix length that the drafter used
    as the base when this token was emitted. The verifier compares it with its
    stale-base boundary before accepting the token as tail data or as
    pending-prefix confirmation.

    new_token_pos is the 0-based output token position for new_token. Normal
    decode streams send the latest generated token.
    """

    src_drafter_rank: int
    dst_verifier_rank: int
    request_id: str
    base_committed_len: int
    new_token_pos: int
    new_token: int


@dataclass
class DraftTailStreamOutputBatch:
    outputs: list[DraftTailStreamOutput] = field(default_factory=list)


@dataclass
class DraftControlBatch:
    dst_drafter_rank: int
    sync_messages: list[DraftSync] = field(default_factory=list)
    verify_commit_messages: list[VerifyCommit] = field(default_factory=list)
    close_messages: list[DraftClose] = field(default_factory=list)


@dataclass
class VerifierCommitSegment:
    """Contiguous VerifyCommit messages coalesced for one drafter request.

    When receiving contiguous VerifyCommit messages for the same draft req,
    the transport thread(TokenSync thread at drafter side) coalesces them into a single VerifierCommitSegment.

    VerifierCommitSegment represents a contiguous verifier-committed token segment for drafter,
    and drafter scheduler should align with these segments before emitting tail tokens
    """

    draft_key: DraftReqKey
    dst_drafter_rank: int
    pre_verify_committed_len: int
    committed_tokens: list[int] = field(default_factory=list)

    @property
    def end_committed_len(self) -> int:
        return int(self.pre_verify_committed_len) + len(self.committed_tokens)

    def append_message(self, message: VerifyCommit) -> None:
        """
        It runs on TokenSyncThread under _pending_lock. That loop only
        catches zmq.error.ContextTerminated, so a raise here escapes _run and
        silently kills the drafter control thread. It then stops applying
        ALL requests' controls while the verifier keeps pushing.

        TODO: 1. peer-data violations (non-contiguous / invalid len)
        should quarantine just that request (drop + add to close_keys), not
        crash the thread. 2. phase 5.c will handle the drafter failure by
        degrading the verifier into normal autoregressive decoding.
        """
        if message.draft_key != self.draft_key:
            raise RuntimeError(
                "Verifier commit segment received a commit for a different "
                f"request: segment_key={self.draft_key} message_key={message.draft_key}"
            )
        if int(message.dst_drafter_rank) != int(self.dst_drafter_rank):
            raise RuntimeError(
                "Verifier commit segment received a commit for a different "
                "drafter rank: "
                f"request_id={message.request_id} "
                f"segment_drafter_rank={self.dst_drafter_rank} "
                f"message_drafter_rank={message.dst_drafter_rank}"
            )
        message.validate_committed_tokens()
        pre_verify_committed_len = int(message.pre_verify_committed_len)
        if pre_verify_committed_len != self.end_committed_len:
            raise RuntimeError(
                "Verifier commit segment requires contiguous VerifyCommit "
                "messages: "
                f"request_id={message.request_id} "
                f"expected_pre_verify_committed_len={self.end_committed_len} "
                f"actual_pre_verify_committed_len={pre_verify_committed_len}"
            )

        token_ids = [int(token_id) for token_id in message.committed_tokens]
        self.committed_tokens.extend(token_ids)

    def extract_prefix(self, num_tokens: int) -> VerifierCommitSegment:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(
                "Verifier commit segment prefix length must be positive: "
                f"request_id={self.draft_key.request_id} num_tokens={num_tokens}"
            )
        if num_tokens > len(self.committed_tokens):
            raise ValueError(
                "Verifier commit segment prefix length exceeds segment length: "
                f"request_id={self.draft_key.request_id} "
                f"num_tokens={num_tokens} "
                f"segment_len={len(self.committed_tokens)}"
            )

        prefix_tokens = [
            int(token_id) for token_id in self.committed_tokens[:num_tokens]
        ]
        remaining_tokens = [
            int(token_id) for token_id in self.committed_tokens[num_tokens:]
        ]
        prefix_segment = VerifierCommitSegment(
            draft_key=self.draft_key,
            dst_drafter_rank=int(self.dst_drafter_rank),
            pre_verify_committed_len=int(self.pre_verify_committed_len),
            committed_tokens=prefix_tokens,
        )
        self.pre_verify_committed_len = int(self.pre_verify_committed_len) + num_tokens
        self.committed_tokens = remaining_tokens
        return prefix_segment


@dataclass
class DraftControlInbox:
    """Drafter-side inbox for verifier control messages.

    The TokenSync thread temporarily stores incoming control messages here.
    The drafter scheduler extracts and consumes them each time it finishes a decoding step.
    """

    sync_messages: list[DraftSync] = field(default_factory=list)
    verifier_commit_segments: dict[DraftReqKey, VerifierCommitSegment] = field(
        default_factory=dict
    )
    close_keys: set[DraftReqKey] = field(default_factory=set)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.verifier_commit_segments
            and not self.close_keys
        )

    def pending_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.verifier_commit_segments)
            + len(self.close_keys)
        )

    def add_control_batch_locked(self, batch: DraftControlBatch) -> None:
        for message in batch.close_messages:
            self.add_close_key_locked(message.draft_key)
        for message in batch.sync_messages:
            if message.draft_key not in self.close_keys:
                self.sync_messages.append(message)
        for message in batch.verify_commit_messages:
            self.add_verify_commit_locked(message)

    def add_close_key_locked(self, draft_key: DraftReqKey) -> None:
        self.close_keys.add(draft_key)
        self.verifier_commit_segments.pop(draft_key, None)
        self.sync_messages = [
            message for message in self.sync_messages if message.draft_key != draft_key
        ]

    def add_verify_commit_locked(self, message: VerifyCommit) -> None:
        if message.draft_key in self.close_keys:
            return
        segment = self.verifier_commit_segments.get(message.draft_key)
        if segment is None:
            segment = VerifierCommitSegment(
                draft_key=message.draft_key,
                dst_drafter_rank=int(message.dst_drafter_rank),
                pre_verify_committed_len=int(message.pre_verify_committed_len),
            )
            segment.append_message(message)
            self.verifier_commit_segments[message.draft_key] = segment
            return
        segment.append_message(message)

    def extract_ready_controls_locked(
        self,
        consumable_commit_len: Callable[[VerifierCommitSegment], int],
    ) -> ReadyDraftControls:
        ready_controls = ReadyDraftControls()

        if self.close_keys:
            ready_controls.close_keys = self.close_keys
            self.close_keys = set()

        if self.sync_messages:
            ready_controls.sync_messages = self.sync_messages
            self.sync_messages = []

        for draft_key, segment in list(self.verifier_commit_segments.items()):
            consumable_len = consumable_commit_len(segment)
            if consumable_len <= 0:
                continue

            ready_controls.ready_commit_segments.append(
                segment.extract_prefix(consumable_len)
            )
            if not segment.committed_tokens:
                self.verifier_commit_segments.pop(draft_key, None)

        return ready_controls


@dataclass
class ReadyDraftControls:
    sync_messages: list[DraftSync] = field(default_factory=list)
    close_keys: set[DraftReqKey] = field(default_factory=set)
    ready_commit_segments: list[VerifierCommitSegment] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.close_keys
            and not self.ready_commit_segments
        )

    def extracted_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.close_keys)
            + len(self.ready_commit_segments)
        )


@dataclass
class DraftMeshMessage:
    message_type: DraftMeshMessageType
    control_batch: Optional[DraftControlBatch] = None
    tail_stream_output_batch: Optional[DraftTailStreamOutputBatch] = None

    @staticmethod
    def from_control_batch(message: DraftControlBatch) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CONTROL_BATCH,
            control_batch=message,
        )

    @staticmethod
    def from_tail_stream_output_batch(
        message: DraftTailStreamOutputBatch,
    ) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH,
            tail_stream_output_batch=message,
        )


@dataclass(frozen=True)
class DecoupledSpecIpcConfig:
    bind_endpoint: str
    connect_endpoints: tuple[str, ...]
    rank: int
    peers: tuple[DecoupledSpecPeerConfig, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        rank = _validate_transport_rank(
            self.rank, field_name="Decoupled-spec local rank"
        )
        bind_endpoint = _validate_endpoint(
            self.bind_endpoint, field_name="Decoupled-spec bind endpoint"
        )
        connect_endpoints = tuple(self.connect_endpoints)
        peers = tuple(self.peers)
        if peers:
            if any(not isinstance(peer, DecoupledSpecPeerConfig) for peer in peers):
                raise ValueError(
                    "Decoupled-spec peers must contain DecoupledSpecPeerConfig values."
                )
            peer_endpoints = tuple(peer.endpoint for peer in peers)
            if connect_endpoints and connect_endpoints != peer_endpoints:
                raise ValueError(
                    "Decoupled-spec connect_endpoints and ranked peers disagree: "
                    f"connect_endpoints={connect_endpoints!r} "
                    f"peer_endpoints={peer_endpoints!r}"
                )
        else:
            peers = tuple(
                DecoupledSpecPeerConfig(
                    rank=peer_rank,
                    endpoint=_validate_endpoint(
                        endpoint,
                        field_name="Decoupled-spec peer endpoint",
                    ),
                    quota=1,
                )
                for peer_rank, endpoint in enumerate(connect_endpoints)
            )

        if not peers:
            raise ValueError("Decoupled-spec peer configs must be non-empty.")

        peer_ranks = []
        peer_endpoints = []
        normalized_peers = []
        for peer in peers:
            peer_rank = _validate_transport_rank(
                peer.rank, field_name="Decoupled-spec peer rank"
            )
            endpoint = _validate_endpoint(
                peer.endpoint, field_name="Decoupled-spec peer endpoint"
            )
            if isinstance(peer.quota, bool) or not isinstance(peer.quota, int):
                raise ValueError(
                    "Decoupled-spec peer quota must be an integer, "
                    f"got {peer.quota!r}."
                )
            if peer.quota <= 0:
                raise ValueError(
                    "Decoupled-spec peer quota must be positive, " f"got {peer.quota}."
                )
            peer_ranks.append(peer_rank)
            peer_endpoints.append(endpoint)
            normalized_peers.append(
                DecoupledSpecPeerConfig(
                    rank=peer_rank,
                    endpoint=endpoint,
                    quota=int(peer.quota),
                )
            )

        if len(set(peer_ranks)) != len(peer_ranks):
            raise ValueError(
                f"Decoupled-spec peer ranks must be unique, got {peer_ranks}."
            )
        if len(set(peer_endpoints)) != len(peer_endpoints):
            raise ValueError(
                "Decoupled-spec peer endpoints must be unique, "
                f"got {peer_endpoints}."
            )
        if bind_endpoint in peer_endpoints:
            raise ValueError(
                "Decoupled-spec bind endpoint must differ from every peer endpoint."
            )

        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "bind_endpoint", bind_endpoint)
        object.__setattr__(self, "peers", tuple(normalized_peers))
        object.__setattr__(self, "connect_endpoints", tuple(peer_endpoints))

    @classmethod
    def from_raw(
        cls,
        *,
        bind_endpoint: Any,
        rank: Any,
        peer_configs: Sequence[Mapping[str, Any]] | None,
        connect_endpoints: Sequence[str] | None,
    ) -> DecoupledSpecIpcConfig:
        """Build ranked peers, accepting the legacy ordered endpoint list."""

        if peer_configs is not None and connect_endpoints is not None:
            raise ValueError(
                "Specify only one of --decoupled-spec-peer-configs and "
                "--decoupled-spec-connect-endpoints."
            )

        peers = []
        if peer_configs is not None:
            if not isinstance(peer_configs, (list, tuple)) or not peer_configs:
                raise ValueError(
                    "--decoupled-spec-peer-configs must be a non-empty JSON list."
                )
            required_keys = {"rank", "endpoint", "quota"}
            for index, raw_peer in enumerate(peer_configs):
                if not isinstance(raw_peer, Mapping):
                    raise ValueError(
                        "Each --decoupled-spec-peer-configs item must be an "
                        f"object, got item {index}: {raw_peer!r}."
                    )
                actual_keys = set(raw_peer)
                if actual_keys != required_keys:
                    raise ValueError(
                        "Each --decoupled-spec-peer-configs item must contain "
                        f"exactly {sorted(required_keys)}, got item {index} keys "
                        f"{sorted(actual_keys)}."
                    )
                peers.append(
                    DecoupledSpecPeerConfig(
                        rank=raw_peer["rank"],
                        endpoint=raw_peer["endpoint"],
                        quota=raw_peer["quota"],
                    )
                )
            legacy_endpoints: tuple[str, ...] = ()
        else:
            if (
                not isinstance(connect_endpoints, (list, tuple))
                or not connect_endpoints
            ):
                raise ValueError(
                    "Decoupled speculation requires a non-empty "
                    "--decoupled-spec-peer-configs or legacy "
                    "--decoupled-spec-connect-endpoints value."
                )
            legacy_endpoints = tuple(connect_endpoints)

        return cls(
            bind_endpoint=bind_endpoint,
            connect_endpoints=legacy_endpoints,
            rank=rank,
            peers=tuple(peers),
        )

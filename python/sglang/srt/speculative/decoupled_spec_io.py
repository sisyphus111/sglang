from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    max_new_tokens: int = 0
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


@dataclass(frozen=True)
class DraftTailStreamOutput:
    """One contiguous drafter stream update for a verifier-side tail.

    Ordinary outputs append ``tokens`` at consecutive output positions starting
    at ``start_token_pos``. Commit echoes remain singleton outputs and
    cumulatively acknowledge the committed prefix through
    ``start_token_pos + 1``; the echoed boundary token preserves the shared wire
    shape but is not a value proof. Keeping an output single-purpose lets one
    transport batch carry an echo immediately followed by a retained-tail
    append.
    """

    src_drafter_rank: int
    dst_verifier_rank: int
    request_id: str
    base_committed_len: int
    start_token_pos: int
    tokens: tuple[int, ...]
    is_commit_echo: bool = False

    def validate(self) -> None:
        if int(self.base_committed_len) < 0:
            raise ValueError("Draft stream base must be non-negative")
        if int(self.start_token_pos) < 0:
            raise ValueError("Draft stream start position must be non-negative")
        if not isinstance(self.tokens, tuple):
            raise TypeError("Draft stream tokens must be an immutable tuple")
        if not self.tokens:
            raise ValueError("Draft stream output must contain at least one token")
        if any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in self.tokens
        ):
            raise TypeError("Draft stream tokens must contain only integers")
        if self.is_commit_echo:
            if len(self.tokens) != 1:
                raise ValueError("Draft commit echo must contain exactly one token")
            if int(self.base_committed_len) != int(self.start_token_pos) + 1:
                raise ValueError(
                    "Draft commit echo base must equal its cumulative ACK length"
                )
        elif int(self.start_token_pos) < int(self.base_committed_len):
            raise ValueError(
                "Draft append must not start before its committed-prefix base"
            )


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


class DraftCommitAction(msgspec.Struct, frozen=True):
    """Native/Python data-plane decision applied by the drafter scheduler."""

    draft_key: DraftReqKey
    dst_drafter_rank: int
    expected_output_len: int
    pre_verify_committed_len: int
    new_committed_len: int
    rewrite_position: int
    rewrite_token: int
    echo_position: int
    echo_token: int

    @property
    def is_rewrite(self) -> bool:
        return int(self.rewrite_position) >= 0


class DraftTranscriptMirror(msgspec.Struct):
    """Bounded protocol mirror of the drafter's published output suffix."""

    committed_len: int = 0
    draft_suffix: list[int] = msgspec.field(default_factory=list)

    @classmethod
    def from_sync(cls, message: DraftSync) -> DraftTranscriptMirror:
        return cls(committed_len=len(message.committed_outputs))

    @property
    def output_len(self) -> int:
        return int(self.committed_len) + len(self.draft_suffix)

    def append_output(
        self,
        output: DraftTailStreamOutput,
    ) -> None:
        output.validate()
        start_token_pos = int(output.start_token_pos)
        if output.is_commit_echo:
            if start_token_pos >= self.committed_len:
                raise RuntimeError(
                    "Draft commit echo must refer to an already committed token"
                )
            return
        if int(output.base_committed_len) < self.committed_len:
            return
        if int(output.base_committed_len) > self.committed_len:
            raise RuntimeError(
                "Draft output base is ahead of mirrored committed prefix"
            )

        output_end = start_token_pos + len(output.tokens)
        current_end = self.output_len
        if start_token_pos > current_end:
            raise RuntimeError("Draft output skips mirrored transcript suffix")

        overlap_end = min(output_end, current_end)
        for token_position in range(start_token_pos, overlap_end):
            existing_token = self.draft_suffix[
                token_position - int(self.committed_len)
            ]
            token = output.tokens[token_position - start_token_pos]
            if int(existing_token) != token:
                raise RuntimeError(
                    "Draft output conflicts with mirrored transcript suffix"
                )

        # Validate the full overlap before changing the mirror so a malformed
        # span cannot leave a partially appended suffix behind.
        if output_end > current_end:
            self.draft_suffix.extend(output.tokens[current_end - start_token_pos :])

    def plan_commit(
        self, segment: VerifierCommitSegment
    ) -> Optional[DraftCommitAction]:
        pre_verify_committed_len = int(segment.pre_verify_committed_len)
        if pre_verify_committed_len != self.committed_len:
            raise RuntimeError(
                "Verifier commit segment does not match mirrored committed prefix"
            )
        if not segment.committed_tokens or not self.draft_suffix:
            return None

        max_match = min(len(segment.committed_tokens), len(self.draft_suffix))
        num_match_tokens = 0
        while num_match_tokens < max_match and int(
            self.draft_suffix[num_match_tokens]
        ) == int(segment.committed_tokens[num_match_tokens]):
            num_match_tokens += 1

        rewrite_position = -1
        rewrite_token = -1
        if num_match_tokens == len(segment.committed_tokens):
            num_commit_tokens = num_match_tokens
        elif num_match_tokens < max_match:
            num_commit_tokens = num_match_tokens + 1
            rewrite_position = pre_verify_committed_len + num_match_tokens
            rewrite_token = int(segment.committed_tokens[num_match_tokens])
        else:
            num_commit_tokens = num_match_tokens
        if num_commit_tokens <= 0:
            return None

        new_committed_len = pre_verify_committed_len + num_commit_tokens
        return DraftCommitAction(
            draft_key=segment.draft_key,
            dst_drafter_rank=int(segment.dst_drafter_rank),
            expected_output_len=self.output_len,
            pre_verify_committed_len=pre_verify_committed_len,
            new_committed_len=new_committed_len,
            rewrite_position=rewrite_position,
            rewrite_token=rewrite_token,
            echo_position=new_committed_len - 1,
            echo_token=int(segment.committed_tokens[num_commit_tokens - 1]),
        )

    def apply_commit(self, action: DraftCommitAction) -> None:
        if int(action.pre_verify_committed_len) != self.committed_len:
            raise RuntimeError("Draft commit action does not match transcript cursor")
        if int(action.expected_output_len) != self.output_len:
            raise RuntimeError("Draft commit action does not match transcript length")
        num_commit_tokens = int(action.new_committed_len) - int(
            action.pre_verify_committed_len
        )
        if num_commit_tokens <= 0 or num_commit_tokens > len(self.draft_suffix):
            raise RuntimeError("Draft commit action consumes an invalid suffix prefix")
        if action.is_rewrite:
            num_match_tokens = int(action.rewrite_position) - int(
                action.pre_verify_committed_len
            )
            if num_match_tokens + 1 != num_commit_tokens:
                raise RuntimeError("Draft rewrite action has inconsistent positions")
            if int(self.draft_suffix[num_match_tokens]) == int(action.rewrite_token):
                raise RuntimeError(
                    "Draft rewrite token unexpectedly matches transcript suffix"
                )
            self.draft_suffix.clear()
        else:
            del self.draft_suffix[:num_commit_tokens]
        self.committed_len = int(action.new_committed_len)


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

    def extract_ready_actions_locked(
        self,
        model_ready: Callable[[DraftReqKey, int], bool],
        transcripts: dict[DraftReqKey, DraftTranscriptMirror],
    ) -> ReadyDraftControls:
        """Consume data-plane-planned actions for model-ready request rows."""

        ready_controls = ReadyDraftControls()
        if self.close_keys:
            ready_controls.close_keys = self.close_keys
            self.close_keys = set()
            for draft_key in ready_controls.close_keys:
                transcripts.pop(draft_key, None)

        preexisting_transcripts = set(transcripts)
        for draft_key, segment in list(self.verifier_commit_segments.items()):
            if draft_key not in preexisting_transcripts:
                continue
            transcript = transcripts[draft_key]
            if not model_ready(draft_key, transcript.output_len):
                continue
            action = transcript.plan_commit(segment)
            if action is None:
                continue
            num_commit_tokens = int(action.new_committed_len) - int(
                action.pre_verify_committed_len
            )
            segment.extract_prefix(num_commit_tokens)
            transcript.apply_commit(action)
            ready_controls.commit_actions.append(action)
            if not segment.committed_tokens:
                self.verifier_commit_segments.pop(draft_key, None)

        if self.sync_messages:
            ready_controls.sync_messages = self.sync_messages
            self.sync_messages = []
            for message in ready_controls.sync_messages:
                if message.draft_key in transcripts:
                    raise RuntimeError(
                        "DraftSync received for an existing transcript mirror"
                    )
                transcripts[message.draft_key] = DraftTranscriptMirror.from_sync(
                    message
                )
        return ready_controls


@dataclass
class ReadyDraftControls:
    sync_messages: list[DraftSync] = field(default_factory=list)
    close_keys: set[DraftReqKey] = field(default_factory=set)
    ready_commit_segments: list[VerifierCommitSegment] = field(default_factory=list)
    commit_actions: list[DraftCommitAction] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.close_keys
            and not self.ready_commit_segments
            and not self.commit_actions
        )

    def extracted_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.close_keys)
            + len(self.ready_commit_segments)
            + len(self.commit_actions)
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

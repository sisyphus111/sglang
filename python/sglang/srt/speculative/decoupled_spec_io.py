from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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


class DraftReqKey(msgspec.Struct, frozen=True):
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


class DraftSync(msgspec.Struct):
    """Open or re-open a drafter request from a verifier-owned prefix.

    The verifier is the source of truth for committed tokens. DraftSync gives
    the drafter the prompt and already committed output prefix that it must
    align to before it can emit draft tail tokens.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    max_new_tokens: int = 0
    prompt_token_ids: list[int] = msgspec.field(default_factory=list)
    committed_outputs: list[int] = msgspec.field(default_factory=list)

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


class VerifyCommit(msgspec.Struct):
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


class DraftClose(msgspec.Struct):
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


class DraftTailStreamOutput(msgspec.Struct, frozen=True):
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


class DraftTailStreamOutputBatch(msgspec.Struct):
    outputs: list[DraftTailStreamOutput] = msgspec.field(default_factory=list)


class DraftControlBatch(msgspec.Struct):
    dst_drafter_rank: int
    sync_messages: list[DraftSync] = msgspec.field(default_factory=list)
    verify_commit_messages: list[VerifyCommit] = msgspec.field(default_factory=list)
    close_messages: list[DraftClose] = msgspec.field(default_factory=list)


class ReadyDraftControls(msgspec.Struct):
    """Host-owned request lifecycle; verifier commits remain on the GPU."""

    sync_messages: list[DraftSync] = msgspec.field(default_factory=list)
    close_keys: set[DraftReqKey] = msgspec.field(default_factory=set)

    def is_empty(self) -> bool:
        return not self.sync_messages and not self.close_keys


class DecoupledSpecIpcConfig(msgspec.Struct, frozen=True):
    bind_endpoint: str
    connect_endpoints: tuple[str, ...]
    rank: int
    peers: tuple[DecoupledSpecPeerConfig, ...] = msgspec.field(default_factory=tuple)

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

        msgspec.structs.force_setattr(self, "rank", rank)
        msgspec.structs.force_setattr(self, "bind_endpoint", bind_endpoint)
        msgspec.structs.force_setattr(self, "peers", tuple(normalized_peers))
        msgspec.structs.force_setattr(self, "connect_endpoints", tuple(peer_endpoints))

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

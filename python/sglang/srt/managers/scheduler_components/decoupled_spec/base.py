from __future__ import annotations

_MIRROR_REQUEST_EPOCH_TAG = "::draft-epoch::"


def build_draft_mirror_request_id(request_id: str, request_epoch: int) -> str:
    """Build an opaque wire identity for one verifier-side request lifecycle."""

    request_epoch = int(request_epoch)
    if request_epoch <= 0:
        raise ValueError(
            "Decoupled draft mirror request epoch must be positive: "
            f"request_epoch={request_epoch}"
        )
    return f"{request_id}{_MIRROR_REQUEST_EPOCH_TAG}{request_epoch}"

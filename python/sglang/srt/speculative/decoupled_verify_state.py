from dataclasses import dataclass, field


@dataclass
class DecoupledVerifySnapshot:
    """Stable per-forward verifier state shared by scheduler and worker."""

    pre_committed_len: int
    draft_tokens: list[int] = field(default_factory=list)
    num_consumable_drafts: int = 0

    def reset(self, pre_committed_len: int) -> None:
        self.pre_committed_len = int(pre_committed_len)
        self.draft_tokens.clear()
        self.num_consumable_drafts = 0


def prepare_decoupled_verify_snapshot(req, pre_committed_len: int):
    snapshot = getattr(req, "decoupled_verify_snapshot", None)
    if snapshot is None:
        snapshot = DecoupledVerifySnapshot(
            pre_committed_len=int(pre_committed_len)
        )
        req.decoupled_verify_snapshot = snapshot
    else:
        snapshot.reset(pre_committed_len)
    return snapshot

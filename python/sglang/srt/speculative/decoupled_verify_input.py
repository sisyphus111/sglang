import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleDraftInput


def get_req_tail_token_id(req) -> int:
    if req.output_ids:
        return int(req.output_ids[-1])
    if req.origin_input_ids:
        return int(req.origin_input_ids[-1])
    raise RuntimeError(
        f"Request {req.rid} has no committed token to anchor external "
        "draft verification."
    )


def build_next_draft_input_stub(
    bonus_tokens: torch.Tensor,
    topk: int,
) -> EagleDraftInput:
    bonus_tokens = bonus_tokens.to(dtype=torch.int32)
    batch_size = int(bonus_tokens.numel())
    device = bonus_tokens.device
    return EagleDraftInput(
        bonus_tokens=bonus_tokens,
        topk_p=torch.zeros(
            (batch_size, int(topk)), device=device, dtype=torch.float32
        ),
        topk_index=torch.zeros(
            (batch_size, int(topk)), device=device, dtype=torch.int64
        ),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )

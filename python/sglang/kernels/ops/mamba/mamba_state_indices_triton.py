"""Fused replay-prep state-indices kernel for the mamba cuda-graph path.

``MambaAttnBackendBase._replay_metadata`` refreshes the captured per-bs
``state_indices_list`` buffer before every cuda-graph replay. The reference
form is a chain of dispatched aten ops whose host cost shows up in the bs=1
MTP inter-phase seam:

    req_pool_indices[valid_bs:] = 0             # zero padded rows (side effect)
    mamba_indices = mapping[req_pool_indices]   # get_mamba_indices gather
    mamba_indices = translate(mamba_indices)    # identity for the static pool
    mamba_indices[valid_bs:] = -1               # padding sentinel
    state_indices[:total_bs].copy_(mamba_indices)

This module fuses that chain into a single launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_replay_state_indices_kernel(
    req_pool_indices_ptr,  # (total_bs,) int64 — static replay buffer
    mamba_map_ptr,  # (req_pool_size,) int32 — req_index_to_mamba_index_mapping
    out_ptr,  # (total_bs,) int32 — state_indices_list[bs - 1]
    valid_bs,
    total_bs,
    BS_UPPER: tl.constexpr,
):
    offs = tl.arange(0, BS_UPPER)
    in_range = offs < total_bs
    valid = offs < valid_bs
    req = tl.load(req_pool_indices_ptr + offs, mask=valid, other=0)
    idx = tl.load(mamba_map_ptr + req, mask=valid, other=0)
    out_val = tl.where(valid, idx.to(tl.int32), -1)
    tl.store(out_ptr + offs, out_val, mask=in_range)
    # Preserve the reference chain's side effect: padded rows of the static
    # req_pool_indices buffer are zeroed so captured kernels that gather
    # with them stay in-bounds.
    zeros = tl.zeros([BS_UPPER], dtype=req.dtype)
    tl.store(req_pool_indices_ptr + offs, zeros, mask=in_range & (~valid))


def fused_replay_state_indices(
    *,
    req_pool_indices: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    out_state_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> torch.Tensor:
    """Fill the captured replay state-indices buffer in one launch.

    Mapping gather + padding sentinel + store into ``out_state_indices``, plus
    the reference chain's side effect of zeroing the padded rows of
    ``req_pool_indices``. Rows ``[valid_bs, total_bs)`` are padding: they get
    the ``-1`` sentinel (mamba kernels skip ``state_idx < 0``) and their
    ``req_pool_indices`` entries are zeroed.

    Callers must supply an identity virtual->physical mapping (the static
    hybrid pool); the unified pool's allocator translate is not a flat table
    gather and has to take the reference chain.

    Returns the filled ``out_state_indices[:total_bs]`` view.
    """
    _fused_replay_state_indices_kernel[(1,)](
        req_pool_indices,
        mamba_index_mapping,
        out_state_indices,
        valid_bs,
        total_bs,
        BS_UPPER=triton.next_power_of_2(total_bs),
    )
    return out_state_indices[:total_bs]


@triton.jit
def _fused_replay_routed_state_indices_kernel(
    req_pool_indices_ptr,  # (total_bs,) int64 -- static replay buffer
    route_src_ptr,  # (valid_bs,) int64 -- drafter checkpoint source slots
    route_dst_ptr,  # (valid_bs,) int64 -- drafter checkpoint destination slots
    out_src_ptr,  # (total_bs,) int32 -- captured source state indices
    out_dst_ptr,  # (total_bs,) int32 -- captured destination state indices
    valid_bs,
    total_bs,
    BS_UPPER: tl.constexpr,
):
    offs = tl.arange(0, BS_UPPER)
    in_range = offs < total_bs
    valid = offs < valid_bs
    route_src = tl.load(route_src_ptr + offs, mask=valid, other=-1)
    route_dst = tl.load(route_dst_ptr + offs, mask=valid, other=-1)
    tl.store(out_src_ptr + offs, route_src.to(tl.int32), mask=in_range)
    tl.store(out_dst_ptr + offs, route_dst.to(tl.int32), mask=in_range)
    zeros = tl.zeros([BS_UPPER], dtype=tl.int64)
    tl.store(req_pool_indices_ptr + offs, zeros, mask=in_range & (~valid))


def fused_replay_routed_state_indices(
    *,
    req_pool_indices: torch.Tensor,
    route_src_indices: torch.Tensor,
    route_dst_indices: torch.Tensor,
    out_src_indices: torch.Tensor,
    out_dst_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage decoupled-drafter src/dst routes in one launch.

    Valid rows are cast from the manager's int64 route ring into the int32
    buffers captured by the Mamba graph. Padded rows receive the ``-1`` state
    sentinel and zero their captured request-pool indices.
    """

    _fused_replay_routed_state_indices_kernel[(1,)](
        req_pool_indices,
        route_src_indices,
        route_dst_indices,
        out_src_indices,
        out_dst_indices,
        valid_bs,
        total_bs,
        BS_UPPER=triton.next_power_of_2(total_bs),
    )
    return out_src_indices[:total_bs], out_dst_indices[:total_bs]

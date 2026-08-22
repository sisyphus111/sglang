#pragma once

#include <cstddef>
#include <cstdint>

/**
 * Flat host/device publish rows contain seat, publish sequence, request epoch,
 * prompt length, committed length, raw/consumable tail lengths, and the fixed
 * capacity token payload.
 */
constexpr int64_t kGpuDraftTailPublishMetadataWidth = 7;

/** Publish CPU-mirrored rolling-tail rows on the daemon landing stream. */
void launch_publish_gpu_draft_tail(
    const int64_t* publish_rows,
    int64_t num_rows,
    int64_t tail_capacity,
    int64_t* versions,
    int64_t* publish_seqs,
    int64_t* request_epochs,
    int64_t* prompt_lens,
    int64_t* committed_lens,
    int64_t* raw_tail_lens,
    int64_t* consumable_tail_lens,
    int64_t* tail_tokens,
    void* stream);

/**
 * Materialize one immutable verify-local snapshot per batch row. compact_out
 * is int64 [batch_size, num_draft_tokens + 2]: draft tokens, selected length,
 * and row-valid bit. The kernel launches on the caller's verify stream.
 */
void launch_select_gpu_draft_tail(
    const int64_t* gpu_seats,
    const int64_t* expected_request_epochs,
    const int64_t* seq_lens,
    const void* bonus_tokens,
    bool bonus_tokens_are_int32,
    int64_t* compact_out,
    int64_t* logical_committed_lens_out,
    int64_t batch_size,
    int64_t num_seats,
    int64_t num_draft_tokens,
    int64_t tail_capacity,
    const int64_t* versions,
    const int64_t* request_epochs,
    const int64_t* active_request_epochs,
    const int64_t* prompt_lens,
    const int64_t* committed_lens,
    const int64_t* raw_tail_lens,
    const int64_t* consumable_tail_lens,
    const int64_t* tail_tokens,
    void* stream);

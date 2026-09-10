#pragma once

#include <cstddef>
#include <cstdint>

enum class GpuDraftTailUpdateOp : int64_t {
  kOpen = 1,
  kAppendDraft = 2,
  kVerifyCommit = 3,
  kClose = 4,
  kCommitAck = 5,
};

enum class GpuDraftTailUpdateError : int64_t {
  kNone = 0,
  kInvalidOp = 1,
  kOpSequenceRegression = 2,
  kCommitPrefixMismatch = 3,
  kPendingTailInvariant = 4,
  kPendingCapacityExceeded = 5,
  kDraftBaseAhead = 6,
  kDraftTokenConflict = 7,
  kDraftTokenSkip = 8,
  kTailCapacityExceeded = 9,
  kInvalidMetadata = 10,
  kCriticalCommitPendingPrefix = 11,
  kCriticalCommitInvalidLength = 12,
  kCriticalCommitLogicalCursorMismatch = 13,
  kCommitAckAhead = 14,
  kDrafterCommitRequiresReplay = 15,
  kDrafterModelStateInvalid = 16,
  kDrafterCheckpointUnavailable = 17,
};

// Keep these values synchronized with cpp_decoupled_spec.py. The selector
// writes them to a TP0-only debug tensor; the compact K+2 forward payload is
// intentionally unchanged.
enum class GpuDraftTailSelectReason : int64_t {
  kUnset = 0,
  kInvalidSeat = 1,
  kWriterInProgress = 2,
  kIdentityMismatch = 3,
  kMetadataInvalid = 4,
  kDirect = 5,
  kLogicalBehind = 6,
  kDeltaTooLarge = 7,
  kDeltaBeyondConsumable = 8,
  kBonusMismatch = 9,
  kRebased = 10,
  kVersionChanged = 11,
  kLogicalCursorMismatch = 12,
  kPendingPrefix = 13,
};

// seat, op sequence, request epoch, op, arg0, arg1, reserved, token count, then
// a fixed-capacity token payload. APPEND uses arg0/arg1 as base/start and the
// payload as one contiguous span; ACK uses arg0/arg1 as base/echo position and
// has no device payload. Rows are grouped by seat before launch so one CUDA
// block applies every operation for that seat under one writer critical section.
constexpr int64_t kGpuDraftTailUpdateMetadataWidth = 8;
constexpr int64_t kGpuDraftTailDebugWidth = 7;
// request epoch, egress sequence, committed length, logical model output
// length, ACK-ready committed length, ACK boundary token, raw-tail length,
// error code, then raw-tail tokens.
constexpr int64_t kGpuDrafterEgressMetadataWidth = 8;

void launch_update_gpu_draft_tail(
    const int64_t* update_rows,
    const int64_t* group_offsets,
    int64_t num_groups,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    int64_t* versions,
    int64_t* publish_seqs,
    int64_t* request_epochs,
    int64_t* prompt_lens,
    int64_t* committed_lens,
    int64_t* can_accept_prefix_lens,
    int64_t* raw_tail_lens,
    int64_t* consumable_tail_lens,
    int64_t* pending_expected_lens,
    int64_t* pending_expected_tokens,
    int64_t* tail_tokens,
    int64_t* last_op_seqs,
    int64_t* error_codes,
    int64_t* error_op_seqs,
    int64_t* pending_prefix_fast_forward_cts,
    bool drafter_authoritative,
    int64_t* model_output_lens,
    int64_t* model_state_positions,
    int64_t* model_input_tokens,
    int64_t* checkpoint_positions,
    int64_t* egress_seqs,
    int64_t* last_commit_tokens,
    void* stream);

/**
 * Resolve one drafter decode launch against the authoritative GPU transcript.
 * The writer lock makes the snapshot and req_to_token candidate binding one
 * transaction with landing-stream verifier controls.
 */
void launch_prepare_gpu_drafter_decode(
    const int64_t* mirror_seats,
    const int64_t* expected_request_epochs,
    const int64_t* req_pool_indices,
    const int64_t* candidate_out_cache_locs,
    const int64_t* checkpoint_slot_table,
    int64_t checkpoint_capacity,
    int32_t* req_to_token,
    int64_t req_to_token_num_rows,
    int64_t req_to_token_row_stride,
    int64_t* resolved_input_ids,
    int64_t* resolved_seq_lens,
    int32_t* resolved_orig_seq_lens,
    int64_t* mamba_src_indices,
    int64_t* mamba_dst_indices,
    int64_t* captured_state_positions,
    int64_t* old_cache_locs,
    int64_t* kv_ownership,
    int64_t batch_size,
    int64_t num_seats,
    int64_t* versions,
    const int64_t* request_epochs,
    const int64_t* prompt_lens,
    const int64_t* committed_lens,
    const int64_t* can_accept_prefix_lens,
    const int64_t* raw_tail_lens,
    const int64_t* pending_expected_lens,
    const int64_t* last_op_seqs,
    int64_t* error_codes,
    int64_t* error_op_seqs,
    const int64_t* model_output_lens,
    const int64_t* model_state_positions,
    const int64_t* model_input_tokens,
    const int64_t* checkpoint_positions,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    void* stream);

/**
 * Commit or invalidate one in-flight drafter decode. A same-epoch branch/state
 * change retains the candidate as the stale logical-cell owner; a lifecycle
 * epoch change restores the prior binding and returns the candidate.
 */
void launch_finish_gpu_drafter_decode(
    const int64_t* mirror_seats,
    const int64_t* expected_request_epochs,
    const int64_t* req_pool_indices,
    const int64_t* candidate_out_cache_locs,
    const void* sampled_tokens,
    bool sampled_tokens_are_int32,
    int32_t* req_to_token,
    int64_t req_to_token_num_rows,
    int64_t req_to_token_row_stride,
    const int64_t* resolved_input_tokens,
    const int64_t* captured_state_positions,
    const int64_t* old_cache_locs,
    int64_t* kv_outcomes,
    int64_t* future_output_tokens,
    int64_t future_output_tokens_size,
    int64_t batch_size,
    int64_t num_seats,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    int64_t* versions,
    const int64_t* request_epochs,
    const int64_t* prompt_lens,
    const int64_t* committed_lens,
    int64_t* can_accept_prefix_lens,
    int64_t* raw_tail_lens,
    int64_t* consumable_tail_lens,
    int64_t* pending_expected_lens,
    const int64_t* pending_expected_tokens,
    int64_t* tail_tokens,
    const int64_t* last_op_seqs,
    int64_t* error_codes,
    int64_t* error_op_seqs,
    int64_t* model_output_lens,
    int64_t* model_state_positions,
    int64_t* model_input_tokens,
    int64_t* checkpoint_positions,
    int64_t* egress_seqs,
    int64_t* last_commit_tokens,
    void* stream);

/** Initialize the model-side cursor with the token sampled by final prefill. */
void launch_append_gpu_drafter_prefill_sample(
    const int64_t* mirror_seats,
    const int64_t* expected_request_epochs,
    void* sampled_tokens,
    bool sampled_tokens_are_int32,
    bool* accept_out,
    int64_t batch_size,
    int64_t num_seats,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    int64_t* versions,
    const int64_t* request_epochs,
    const int64_t* prompt_lens,
    const int64_t* committed_lens,
    const int64_t* can_accept_prefix_lens,
    int64_t* raw_tail_lens,
    int64_t* consumable_tail_lens,
    int64_t* pending_expected_lens,
    const int64_t* pending_expected_tokens,
    int64_t* tail_tokens,
    const int64_t* last_op_seqs,
    int64_t* error_codes,
    int64_t* error_op_seqs,
    int64_t* model_output_lens,
    int64_t* model_state_positions,
    int64_t* model_input_tokens,
    int64_t* checkpoint_positions,
    int64_t* egress_seqs,
    void* stream);

/**
 * Snapshot the bound drafter seats into compact, seqlock-consistent egress
 * rows. The host filters rows by request epoch and egress sequence, then emits
 * a cumulative ACK plus the full bounded retained tail.
 */
void launch_snapshot_gpu_drafter_egress(
    int64_t* out,
    const int64_t* seat_indices,
    int64_t num_rows,
    int64_t num_seats,
    int64_t tail_capacity,
    const int64_t* versions,
    const int64_t* request_epochs,
    const int64_t* committed_lens,
    const int64_t* can_accept_prefix_lens,
    const int64_t* raw_tail_lens,
    const int64_t* pending_expected_lens,
    const int64_t* tail_tokens,
    const int64_t* error_codes,
    const int64_t* model_output_lens,
    const int64_t* egress_seqs,
    const int64_t* last_commit_tokens,
    void* stream);

/**
 * Apply the exact accepted token run produced by target verification. This
 * kernel runs on the forward stream immediately after target verification.
 * A per-seat GPU writer lock serializes it with landing-stream updates before
 * the next direct-only selector. expected_request_epochs is immutable launch
 * identity, so delayed work cannot commit into a reused request-pool seat.
 */
void launch_apply_gpu_draft_tail_verify_commit(
    const int64_t* gpu_seats,
    const int64_t* expected_request_epochs,
    const int64_t* pre_verify_seq_lens,
    const int32_t* accept_tokens,
    const int32_t* num_accept_tokens,
    const bool* commit_mask,
    int64_t accept_token_stride,
    int64_t batch_size,
    int64_t num_seats,
    int64_t tail_capacity,
    int64_t* versions,
    const int64_t* request_epochs,
    const int64_t* prompt_lens,
    int64_t* committed_lens,
    int64_t* can_accept_prefix_lens,
    int64_t* raw_tail_lens,
    int64_t* consumable_tail_lens,
    int64_t* pending_expected_lens,
    int64_t* pending_expected_tokens,
    int64_t* tail_tokens,
    const int64_t* last_op_seqs,
    int64_t* error_codes,
    int64_t* error_op_seqs,
    void* stream);

/**
 * Materialize one immutable verify-local snapshot per batch row. compact_out
 * remains int64 [batch_size, num_draft_tokens + 2]: draft tokens, selected
 * length, and row-valid bit. debug_out is optional int64 [batch_size,
 * debug_width], with the first seven fields defined by kGpuDraftTailDebugWidth.
 * A selector spins only on the requested seat until its seqlock snapshot is
 * stable; optional debug fields expose the number of read retries.
 */
void launch_select_gpu_draft_tail(
    const int64_t* gpu_seats,
    const int64_t* expected_request_epochs,
    const int64_t* seq_lens,
    const void* bonus_tokens,
    bool bonus_tokens_are_int32,
    int64_t* compact_out,
    int64_t* logical_committed_lens_out,
    int64_t* debug_out,
    int64_t debug_width,
    int64_t batch_size,
    int64_t num_seats,
    int64_t num_draft_tokens,
    int64_t tail_capacity,
    const int64_t* versions,
    const int64_t* publish_seqs,
    const int64_t* request_epochs,
    const int64_t* prompt_lens,
    const int64_t* committed_lens,
    const int64_t* raw_tail_lens,
    const int64_t* consumable_tail_lens,
    const int64_t* pending_expected_lens,
    const int64_t* tail_tokens,
    const int64_t* error_codes,
    const int64_t* error_op_seqs,
    const int64_t* pending_prefix_fast_forward_cts,
    void* stream,
    bool allow_partial,
    int64_t required_tail_len);

/**
 * Offline-profiler selector. It performs the same row-indexed GPU tail reads
 * and writes the same fixed compact/debug outputs as the production selector,
 * but ignores mutable lifecycle metadata and always exposes a full mock tail.
 */
void launch_select_mock_gpu_draft_tail(
    const int64_t* gpu_seats,
    const int64_t* seq_lens,
    const void* bonus_tokens,
    bool bonus_tokens_are_int32,
    int64_t* compact_out,
    int64_t* logical_committed_lens_out,
    int64_t* debug_out,
    int64_t debug_width,
    int64_t batch_size,
    int64_t num_seats,
    int64_t num_draft_tokens,
    int64_t tail_capacity,
    const int64_t* mock_tail_tokens,
    void* stream);

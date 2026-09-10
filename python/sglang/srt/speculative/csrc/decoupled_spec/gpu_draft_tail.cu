#include "gpu_draft_tail.h"

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

__device__ __forceinline__ int64_t atomic_load_relaxed(
    const int64_t* address) {
  // The owning tensors are int64_t, so keep atomic_ref's value type identical
  // to the storage type. In particular, do not reinterpret the storage as an
  // unsigned integer merely to perform the seqlock operations.
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> value(
      *const_cast<int64_t*>(address));
  return value.load(cuda::memory_order_relaxed);
}

__device__ __forceinline__ void atomic_store_relaxed(
    int64_t* address,
    int64_t new_value) {
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> value(*address);
  value.store(new_value, cuda::memory_order_relaxed);
}

__device__ __forceinline__ int64_t acquire_tail_writer(
    cuda::atomic_ref<int64_t, cuda::thread_scope_device>& version) {
  while (true) {
    int64_t expected = version.load(cuda::memory_order_acquire);
    if ((expected & int64_t{1}) != 0) {
      __nanosleep(64);
      continue;
    }
    if (version.compare_exchange_weak(
            expected,
            expected + 1,
            cuda::memory_order_acq_rel,
            cuda::memory_order_acquire)) {
      return expected;
    }
  }
}

__device__ __forceinline__ void release_tail_writer(
    cuda::atomic_ref<int64_t, cuda::thread_scope_device>& version,
    int64_t previous_even_version) {
  version.store(previous_even_version + 2, cuda::memory_order_release);
}

__device__ __forceinline__ void latch_update_error(
    int64_t seat,
    int64_t op_seq,
    GpuDraftTailUpdateError error,
    int64_t* error_codes,
    int64_t* error_op_seqs) {
  if (atomic_load_relaxed(error_codes + seat) != 0) return;
  atomic_store_relaxed(error_codes + seat, static_cast<int64_t>(error));
  atomic_store_relaxed(error_op_seqs + seat, op_seq);
}

__global__ void update_gpu_draft_tail_kernel(
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ group_offsets,
    int64_t num_groups,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    int64_t* __restrict__ versions,
    int64_t* __restrict__ publish_seqs,
    int64_t* __restrict__ request_epochs,
    int64_t* __restrict__ prompt_lens,
    int64_t* __restrict__ committed_lens,
    int64_t* __restrict__ can_accept_prefix_lens,
    int64_t* __restrict__ raw_tail_lens,
    int64_t* __restrict__ consumable_tail_lens,
    int64_t* __restrict__ pending_expected_lens,
    int64_t* __restrict__ pending_expected_tokens,
    int64_t* __restrict__ tail_tokens,
    int64_t* __restrict__ last_op_seqs,
    int64_t* __restrict__ error_codes,
    int64_t* __restrict__ error_op_seqs,
    int64_t* __restrict__ pending_prefix_fast_forward_cts,
    bool drafter_authoritative,
    int64_t* __restrict__ model_output_lens,
    int64_t* __restrict__ model_state_positions,
    int64_t* __restrict__ model_input_tokens,
    int64_t* __restrict__ checkpoint_positions,
    int64_t* __restrict__ egress_seqs,
    int64_t* __restrict__ last_commit_tokens) {
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= num_groups || threadIdx.x != 0) return;
  const int64_t begin = group_offsets[group];
  const int64_t end = group_offsets[group + 1];
  if (begin >= end) return;

  const int64_t row_width = kGpuDraftTailUpdateMetadataWidth + tail_capacity;
  const int64_t seat = rows[begin * row_width];
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(versions[seat]);
  const int64_t writer_version = acquire_tail_writer(version);

  for (int64_t row_index = begin; row_index < end; ++row_index) {
    const int64_t* row = rows + row_index * row_width;
    const int64_t op_seq = row[1];
    const int64_t request_epoch = row[2];
    const auto op = static_cast<GpuDraftTailUpdateOp>(row[3]);
    const int64_t arg0 = row[4];
    const int64_t arg1 = row[5];
    const int64_t num_tokens = row[7];
    const int64_t* op_tokens = row + kGpuDraftTailUpdateMetadataWidth;

    const int64_t last_op_seq = atomic_load_relaxed(last_op_seqs + seat);
    if (op_seq <= last_op_seq) {
      latch_update_error(
          seat,
          op_seq,
          GpuDraftTailUpdateError::kOpSequenceRegression,
          error_codes,
          error_op_seqs);
      continue;
    }
    atomic_store_relaxed(last_op_seqs + seat, op_seq);

    if (op == GpuDraftTailUpdateOp::kOpen) {
      if (arg0 < 0 || arg1 < 0 || num_tokens != 0) {
        latch_update_error(
            seat,
            op_seq,
            GpuDraftTailUpdateError::kInvalidMetadata,
            error_codes,
            error_op_seqs);
        continue;
      }
      atomic_store_relaxed(request_epochs + seat, request_epoch);
      atomic_store_relaxed(prompt_lens + seat, arg0);
      atomic_store_relaxed(committed_lens + seat, arg1);
      atomic_store_relaxed(can_accept_prefix_lens + seat, arg1);
      atomic_store_relaxed(raw_tail_lens + seat, 0);
      atomic_store_relaxed(consumable_tail_lens + seat, 0);
      atomic_store_relaxed(pending_expected_lens + seat, 0);
      atomic_store_relaxed(error_codes + seat, 0);
      atomic_store_relaxed(error_op_seqs + seat, 0);
      atomic_store_relaxed(pending_prefix_fast_forward_cts + seat, 0);
      atomic_store_relaxed(model_output_lens + seat, arg1);
      atomic_store_relaxed(model_state_positions + seat, arg0 + arg1);
      atomic_store_relaxed(model_input_tokens + seat, -1);
      atomic_store_relaxed(egress_seqs + seat, 0);
      atomic_store_relaxed(last_commit_tokens + seat, -1);
      for (int64_t index = 0; index < tail_capacity; ++index) {
        atomic_store_relaxed(
            checkpoint_positions + seat * tail_capacity + index, -1);
      }
      // Freshness is scoped to drafter arrivals for the current request epoch.
      // Verifier-local OPEN/COMMIT/CLOSE updates must not masquerade as new
      // draft supply at the next selector observation.
      atomic_store_relaxed(publish_seqs + seat, -1);
      continue;
    }

    // Late data for an earlier request generation is harmless. The host route
    // normally removes it before launch; the epoch guard keeps the device state
    // safe if an already-staged record retires after a seat has been reused.
    if (atomic_load_relaxed(request_epochs + seat) != request_epoch) continue;

    if (op == GpuDraftTailUpdateOp::kClose) {
      atomic_store_relaxed(request_epochs + seat, -1);
      atomic_store_relaxed(prompt_lens + seat, -1);
      atomic_store_relaxed(committed_lens + seat, -1);
      atomic_store_relaxed(can_accept_prefix_lens + seat, -1);
      atomic_store_relaxed(raw_tail_lens + seat, 0);
      atomic_store_relaxed(consumable_tail_lens + seat, 0);
      atomic_store_relaxed(pending_expected_lens + seat, 0);
      atomic_store_relaxed(publish_seqs + seat, -1);
      atomic_store_relaxed(model_output_lens + seat, -1);
      atomic_store_relaxed(model_state_positions + seat, -1);
      atomic_store_relaxed(model_input_tokens + seat, -1);
      atomic_store_relaxed(egress_seqs + seat, 0);
      atomic_store_relaxed(last_commit_tokens + seat, -1);
      for (int64_t index = 0; index < tail_capacity; ++index) {
        atomic_store_relaxed(
            checkpoint_positions + seat * tail_capacity + index, -1);
      }
      continue;
    }

    if (atomic_load_relaxed(error_codes + seat) != 0) continue;
    int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
    int64_t can_accept_prefix_len =
        atomic_load_relaxed(can_accept_prefix_lens + seat);
    int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
    int64_t pending_len = atomic_load_relaxed(pending_expected_lens + seat);
    int64_t* seat_pending =
        pending_expected_tokens + seat * pending_token_capacity;
    int64_t* seat_tail = tail_tokens + seat * tail_capacity;

    if (committed_len < 0 || can_accept_prefix_len < 0 ||
        can_accept_prefix_len > committed_len || raw_tail_len < 0 ||
        raw_tail_len > tail_capacity || pending_len < 0 ||
        pending_len > committed_len) {
      latch_update_error(
          seat,
          op_seq,
          GpuDraftTailUpdateError::kInvalidMetadata,
          error_codes,
          error_op_seqs);
      continue;
    }

    if (op == GpuDraftTailUpdateOp::kAppendDraft ||
        op == GpuDraftTailUpdateOp::kCommitAck) {
      const int64_t base_committed_len = arg0;
      const int64_t start_token_pos = arg1;

      if (op == GpuDraftTailUpdateOp::kCommitAck) {
        if (num_tokens != 0 || start_token_pos < 0 ||
            start_token_pos == int64_t{0x7fffffffffffffffLL} ||
            base_committed_len != start_token_pos + 1) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kInvalidMetadata,
              error_codes,
              error_op_seqs);
          continue;
        }
        if (pending_len > 0 && raw_tail_len != 0) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kPendingTailInvariant,
              error_codes,
              error_op_seqs);
          continue;
        }
        const int64_t confirmed_len = committed_len - pending_len;
        const int64_t ack_len = start_token_pos + 1;
        if (ack_len <= confirmed_len) continue;
        if (ack_len > committed_len) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kCommitAckAhead,
              error_codes,
              error_op_seqs);
          continue;
        }
        pending_len = committed_len - ack_len;
        atomic_store_relaxed(pending_expected_lens + seat, pending_len);
        if (pending_len == 0) {
          atomic_store_relaxed(
              can_accept_prefix_lens + seat, committed_len);
        }
        continue;
      }

      if (num_tokens <= 0 || num_tokens > tail_capacity ||
          base_committed_len < 0 || start_token_pos < base_committed_len ||
          start_token_pos > int64_t{0x7fffffffffffffffLL} - num_tokens) {
        latch_update_error(
            seat,
            op_seq,
            GpuDraftTailUpdateError::kInvalidMetadata,
            error_codes,
            error_op_seqs);
        continue;
      }
      // Freshness tracks a non-empty draft-token supply, not commit echoes.
      atomic_store_relaxed(publish_seqs + seat, op_seq);

      bool reconciled_pending = false;
      if (pending_len > 0) {
        if (raw_tail_len != 0) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kPendingTailInvariant,
              error_codes,
              error_op_seqs);
          continue;
        }
        if (base_committed_len > committed_len) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kDraftBaseAhead,
              error_codes,
              error_op_seqs);
          continue;
        }
        if (base_committed_len < can_accept_prefix_len) continue;
        const int64_t confirmed_len = committed_len - pending_len;
        const int64_t end_token_pos = start_token_pos + num_tokens;
        // Only the newest bounded suffix is stored when the drafter falls more
        // than one row capacity behind. A cumulative ACK must first bring the
        // residual prefix back inside this value-checked window.
        if (pending_len > pending_token_capacity ||
            start_token_pos > confirmed_len ||
            end_token_pos <= confirmed_len) {
          atomic_store_relaxed(consumable_tail_lens + seat, 0);
          continue;
        }

        const int64_t overlap_end =
            end_token_pos < committed_len ? end_token_pos : committed_len;
        int64_t match_len = 0;
        while (confirmed_len + match_len < overlap_end &&
               atomic_load_relaxed(
                   seat_pending +
                   (confirmed_len + match_len) % pending_token_capacity) ==
                   op_tokens[confirmed_len + match_len - start_token_pos]) {
          ++match_len;
        }
        if (match_len > 0) {
          pending_len -= match_len;
          atomic_store_relaxed(pending_expected_lens + seat, pending_len);
          atomic_store_relaxed(
              pending_prefix_fast_forward_cts + seat,
              atomic_load_relaxed(
                  pending_prefix_fast_forward_cts + seat) + 1);
        }
        if (confirmed_len + match_len < overlap_end) {
          const int64_t mismatch_fence = confirmed_len + match_len + 1;
          if (mismatch_fence > can_accept_prefix_len) {
            can_accept_prefix_len = mismatch_fence;
            atomic_store_relaxed(
                can_accept_prefix_lens + seat, can_accept_prefix_len);
          }
        }
        if (pending_len > 0) {
          atomic_store_relaxed(consumable_tail_lens + seat, 0);
          continue;
        }
        atomic_store_relaxed(
            can_accept_prefix_lens + seat, committed_len);
        reconciled_pending = true;
      }

      if (base_committed_len > committed_len) {
        latch_update_error(
            seat,
            op_seq,
            GpuDraftTailUpdateError::kDraftBaseAhead,
            error_codes,
            error_op_seqs);
        continue;
      }
      if (!reconciled_pending &&
          base_committed_len < can_accept_prefix_len) {
        continue;
      }
      const int64_t buffer_end_len = committed_len + raw_tail_len;
      const int64_t end_token_pos = start_token_pos + num_tokens;
      if (start_token_pos > buffer_end_len) {
        if (base_committed_len == committed_len) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kDraftTokenSkip,
              error_codes,
              error_op_seqs);
        }
        continue;
      }

      // Validate the complete span before publishing any new suffix tokens.
      const int64_t overlap_begin =
          start_token_pos > committed_len ? start_token_pos : committed_len;
      const int64_t overlap_end =
          end_token_pos < buffer_end_len ? end_token_pos : buffer_end_len;
      bool conflicts = false;
      for (int64_t token_pos = overlap_begin; token_pos < overlap_end;
           ++token_pos) {
        const int64_t token_index = token_pos - start_token_pos;
        if (atomic_load_relaxed(seat_tail + token_pos - committed_len) !=
            op_tokens[token_index]) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kDraftTokenConflict,
              error_codes,
              error_op_seqs);
          conflicts = true;
          break;
        }
      }
      if (conflicts || end_token_pos <= buffer_end_len) {
        continue;
      }

      const int64_t append_begin = buffer_end_len - start_token_pos;
      const int64_t num_append_tokens = num_tokens - append_begin;
      if (num_append_tokens > tail_capacity - raw_tail_len) {
        latch_update_error(
            seat,
            op_seq,
            GpuDraftTailUpdateError::kTailCapacityExceeded,
            error_codes,
            error_op_seqs);
        continue;
      }
      for (int64_t index = 0; index < num_append_tokens; ++index) {
        atomic_store_relaxed(
            seat_tail + raw_tail_len + index,
            op_tokens[append_begin + index]);
      }
      raw_tail_len += num_append_tokens;
      atomic_store_relaxed(raw_tail_lens + seat, raw_tail_len);
      atomic_store_relaxed(consumable_tail_lens + seat, raw_tail_len);
      continue;
    }

    if (op == GpuDraftTailUpdateOp::kVerifyCommit) {
      if (num_tokens <= 0 || num_tokens > tail_capacity ||
          arg0 != committed_len) {
        latch_update_error(
            seat,
            op_seq,
            arg0 != committed_len
                ? GpuDraftTailUpdateError::kCommitPrefixMismatch
                : GpuDraftTailUpdateError::kInvalidMetadata,
            error_codes,
            error_op_seqs);
        continue;
      }
      if (drafter_authoritative) {
        // pending_expected_tokens is an absolute-position modulo ring in this
        // mode. It holds verifier-forced tokens after the current model input;
        // no token transcript or matching work escapes to the CPU scheduler.
        const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
        int64_t model_output_len =
            atomic_load_relaxed(model_output_lens + seat);
        int64_t model_state_position =
            atomic_load_relaxed(model_state_positions + seat);
        int64_t model_input_token =
            atomic_load_relaxed(model_input_tokens + seat);
        const bool decode_state_valid =
            model_output_len > 0 &&
            model_state_position == prompt_len + model_output_len - 1 &&
            model_input_token >= 0;
        const bool prefill_state_valid =
            model_output_len >= 0 && raw_tail_len == 0 &&
            model_state_position == prompt_len + model_output_len &&
            model_input_token == -1;
        const bool model_valid =
            prompt_len >= 0 && committed_len >= 0 &&
            can_accept_prefix_len >= 0 &&
            can_accept_prefix_len <= committed_len && raw_tail_len >= 0 &&
            raw_tail_len <= tail_capacity && pending_len >= 0 &&
            pending_len <= pending_token_capacity &&
            (pending_len == 0 || raw_tail_len == 0) &&
            model_output_len + pending_len ==
                committed_len + raw_tail_len &&
            (decode_state_valid || prefill_state_valid);
        if (!model_valid) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kDrafterModelStateInvalid,
              error_codes,
              error_op_seqs);
          continue;
        }

        int64_t matched_tail_len = 0;
        if (pending_len == 0) {
          const int64_t max_match =
              num_tokens < raw_tail_len ? num_tokens : raw_tail_len;
          while (matched_tail_len < max_match &&
                 atomic_load_relaxed(seat_tail + matched_tail_len) ==
                     op_tokens[matched_tail_len]) {
            ++matched_tail_len;
          }
          if (matched_tail_len < max_match) {
            // A target mismatch rewrites the current raw branch. Exactly one
            // authoritative bonus token remains after the matching prefix;
            // its predecessor checkpoint was materialized before the
            // mismatching raw token was sampled. The position tag is not by
            // itself branch identity: safety also relies on one unretired
            // prepare per seat. A bound prepare has raw_tail_len<capacity, so
            // its dst is 1..capacity-1 slots after every resident mismatch
            // restore point and cannot alias it. If finish wins first, its
            // model write is already complete on the same forward stream.
            if (num_tokens - matched_tail_len != 1) {
              latch_update_error(
                  seat,
                  op_seq,
                  GpuDraftTailUpdateError::kDrafterCommitRequiresReplay,
                  error_codes,
                  error_op_seqs);
              continue;
            }
            const int64_t new_committed_len = committed_len + num_tokens;
            const int64_t new_state_position =
                prompt_len + new_committed_len - 1;
            if (atomic_load_relaxed(
                    checkpoint_positions +
                    seat * tail_capacity +
                    new_state_position % tail_capacity) !=
                new_state_position) {
              latch_update_error(
                  seat,
                  op_seq,
                  GpuDraftTailUpdateError::kDrafterCheckpointUnavailable,
                  error_codes,
                  error_op_seqs);
              continue;
            }
            committed_len = new_committed_len;
            model_output_len = committed_len;
            model_state_position = new_state_position;
            model_input_token = op_tokens[num_tokens - 1];
            raw_tail_len = 0;
            pending_len = 0;
            atomic_store_relaxed(committed_lens + seat, committed_len);
            atomic_store_relaxed(raw_tail_lens + seat, 0);
            atomic_store_relaxed(consumable_tail_lens + seat, 0);
            atomic_store_relaxed(pending_expected_lens + seat, 0);
            atomic_store_relaxed(
                model_output_lens + seat, model_output_len);
            atomic_store_relaxed(
                model_state_positions + seat, model_state_position);
            atomic_store_relaxed(
                model_input_tokens + seat, model_input_token);
            atomic_store_relaxed(
                egress_seqs + seat,
                atomic_load_relaxed(egress_seqs + seat) + 1);
            continue;
          }
        }

        // Either this commit matched a raw prefix or the model is already
        // behind target with forced tokens queued. Matching tokens retire from
        // raw; every target-ahead token is appended after the current input.
        if (matched_tail_len > 0) {
          for (int64_t index = matched_tail_len; index < raw_tail_len;
               ++index) {
            atomic_store_relaxed(
                seat_tail + index - matched_tail_len,
                atomic_load_relaxed(seat_tail + index));
          }
          raw_tail_len -= matched_tail_len;
        }
        const int64_t num_forced_tokens = num_tokens - matched_tail_len;
        if (num_forced_tokens > pending_token_capacity - pending_len) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kPendingCapacityExceeded,
              error_codes,
              error_op_seqs);
          continue;
        }
        for (int64_t index = 0; index < num_forced_tokens; ++index) {
          const int64_t forced_position =
              model_output_len + pending_len + index;
          atomic_store_relaxed(
              seat_pending + forced_position % pending_token_capacity,
              op_tokens[matched_tail_len + index]);
        }
        pending_len += num_forced_tokens;
        committed_len += num_tokens;
        atomic_store_relaxed(committed_lens + seat, committed_len);
        atomic_store_relaxed(raw_tail_lens + seat, raw_tail_len);
        atomic_store_relaxed(consumable_tail_lens + seat, raw_tail_len);
        atomic_store_relaxed(pending_expected_lens + seat, pending_len);
        if (raw_tail_len > 0) {
          // raw_tail_len>0 proves that the new committed boundary has a
          // materialized successor checkpoint/lookahead, so its cumulative
          // ACK can become visible to the verifier.
          atomic_store_relaxed(
              can_accept_prefix_lens + seat, committed_len);
          atomic_store_relaxed(
              last_commit_tokens + seat, op_tokens[num_tokens - 1]);
        }
        atomic_store_relaxed(
            egress_seqs + seat,
            atomic_load_relaxed(egress_seqs + seat) + 1);
        continue;
      }
      if (pending_len > 0) {
        if (raw_tail_len != 0) {
          latch_update_error(
              seat,
              op_seq,
              GpuDraftTailUpdateError::kPendingTailInvariant,
              error_codes,
              error_op_seqs);
          continue;
        }
        for (int64_t index = 0; index < num_tokens; ++index) {
          atomic_store_relaxed(
              seat_pending +
                  (committed_len + index) % pending_token_capacity,
              op_tokens[index]);
        }
        committed_len += num_tokens;
        pending_len += num_tokens;
        atomic_store_relaxed(committed_lens + seat, committed_len);
        atomic_store_relaxed(pending_expected_lens + seat, pending_len);
        atomic_store_relaxed(consumable_tail_lens + seat, 0);
        continue;
      }

      int64_t matched_tail_len = 0;
      const int64_t max_match =
          num_tokens < raw_tail_len ? num_tokens : raw_tail_len;
      while (matched_tail_len < max_match &&
             atomic_load_relaxed(seat_tail + matched_tail_len) ==
                 op_tokens[matched_tail_len]) {
        ++matched_tail_len;
      }
      const bool resident_mismatch = matched_tail_len < raw_tail_len;
      if (matched_tail_len > 0) {
        for (int64_t i = matched_tail_len; i < raw_tail_len; ++i) {
          atomic_store_relaxed(
              seat_tail + i - matched_tail_len,
              atomic_load_relaxed(seat_tail + i));
        }
        raw_tail_len -= matched_tail_len;
        atomic_store_relaxed(raw_tail_lens + seat, raw_tail_len);
      }
      committed_len += num_tokens;
      atomic_store_relaxed(committed_lens + seat, committed_len);
      if (matched_tail_len < num_tokens) {
        if (resident_mismatch) {
          const int64_t mismatch_fence =
              committed_len - num_tokens + matched_tail_len + 1;
          if (mismatch_fence > can_accept_prefix_len) {
            can_accept_prefix_len = mismatch_fence;
            atomic_store_relaxed(
                can_accept_prefix_lens + seat, can_accept_prefix_len);
          }
        }
        atomic_store_relaxed(raw_tail_lens + seat, 0);
        const int64_t remaining = num_tokens - matched_tail_len;
        for (int64_t index = 0; index < remaining; ++index) {
          atomic_store_relaxed(
              seat_pending +
                  (committed_len - num_tokens + matched_tail_len + index) %
                      pending_token_capacity,
              op_tokens[matched_tail_len + index]);
        }
        atomic_store_relaxed(pending_expected_lens + seat, remaining);
        atomic_store_relaxed(consumable_tail_lens + seat, 0);
      } else {
        atomic_store_relaxed(consumable_tail_lens + seat, raw_tail_len);
      }
      continue;
    }

    latch_update_error(
        seat,
        op_seq,
        GpuDraftTailUpdateError::kInvalidOp,
        error_codes,
        error_op_seqs);
  }

  release_tail_writer(version, writer_version);
}

__global__ void apply_gpu_draft_tail_verify_commit_kernel(
    const int64_t* __restrict__ gpu_seats,
    const int64_t* __restrict__ expected_request_epochs,
    const int64_t* __restrict__ pre_verify_seq_lens,
    const int32_t* __restrict__ accept_tokens,
    const int32_t* __restrict__ num_accept_tokens,
    const bool* __restrict__ commit_mask,
    int64_t accept_token_stride,
    int64_t batch_size,
    int64_t num_seats,
    int64_t tail_capacity,
    int64_t* __restrict__ versions,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ prompt_lens,
    int64_t* __restrict__ committed_lens,
    int64_t* __restrict__ can_accept_prefix_lens,
    int64_t* __restrict__ raw_tail_lens,
    int64_t* __restrict__ consumable_tail_lens,
    int64_t* __restrict__ pending_expected_lens,
    int64_t* __restrict__ pending_expected_tokens,
    int64_t* __restrict__ tail_tokens,
    const int64_t* __restrict__ last_op_seqs,
    int64_t* __restrict__ error_codes,
    int64_t* __restrict__ error_op_seqs) {
  const int64_t batch_index = static_cast<int64_t>(blockIdx.x);
  if (batch_index >= batch_size || threadIdx.x != 0) return;
  if (commit_mask != nullptr && !commit_mask[batch_index]) return;

  const int64_t seat = gpu_seats[batch_index];
  if (seat < 0 || seat >= num_seats) return;

  const int64_t expected_request_epoch =
      expected_request_epochs[batch_index];
  if (expected_request_epoch < 0) return;

  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(versions[seat]);
  const int64_t writer_version = acquire_tail_writer(version);
  // The resident epoch is authoritative while the writer lock is held. An old
  // batch may linearize before a reseat OPEN, but can never modify the new row.
  if (atomic_load_relaxed(request_epochs + seat) != expected_request_epoch) {
    release_tail_writer(version, writer_version);
    return;
  }

  const int64_t error_code = atomic_load_relaxed(error_codes + seat);
  int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
  int64_t can_accept_prefix_len =
      atomic_load_relaxed(can_accept_prefix_lens + seat);
  const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
  int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
  const int64_t consumable_tail_len =
      atomic_load_relaxed(consumable_tail_lens + seat);
  int64_t pending_len =
      atomic_load_relaxed(pending_expected_lens + seat);
  const int64_t accept_len =
      static_cast<int64_t>(num_accept_tokens[batch_index]);
  const int64_t expected_pre_verify_output_len =
      pre_verify_seq_lens == nullptr
          ? committed_len
          : pre_verify_seq_lens[batch_index] - prompt_len + 1;
  const int64_t last_op_seq = atomic_load_relaxed(last_op_seqs + seat);

  const bool metadata_valid = committed_len >= 0 &&
      can_accept_prefix_len >= 0 && can_accept_prefix_len <= committed_len &&
      raw_tail_len >= 0 &&
      raw_tail_len <= tail_capacity && consumable_tail_len >= 0 &&
      consumable_tail_len <= raw_tail_len && pending_len >= 0 &&
      pending_len <= committed_len &&
      prompt_len >= 0 && expected_pre_verify_output_len >= 0;
  if (error_code == 0 && !metadata_valid) {
    latch_update_error(
        seat,
        last_op_seq,
        GpuDraftTailUpdateError::kInvalidMetadata,
        error_codes,
        error_op_seqs);
  } else if (
      error_code == 0 &&
      committed_len != expected_pre_verify_output_len) {
    latch_update_error(
        seat,
        last_op_seq,
        GpuDraftTailUpdateError::kCriticalCommitLogicalCursorMismatch,
        error_codes,
        error_op_seqs);
  } else if (
      error_code == 0 &&
      (accept_len <= 0 || accept_len > accept_token_stride ||
       accept_len > tail_capacity)) {
    latch_update_error(
        seat,
        last_op_seq,
        GpuDraftTailUpdateError::kCriticalCommitInvalidLength,
        error_codes,
        error_op_seqs);
  } else if (
      error_code == 0 && pending_len > 0 &&
      (raw_tail_len != 0 || consumable_tail_len != 0)) {
    latch_update_error(
        seat,
        last_op_seq,
        GpuDraftTailUpdateError::kPendingTailInvariant,
        error_codes,
        error_op_seqs);
  } else if (error_code == 0) {
    int64_t* seat_pending =
        pending_expected_tokens + seat * tail_capacity;
    int64_t* seat_tail = tail_tokens + seat * tail_capacity;
    const int32_t* row_accept_tokens =
        accept_tokens + batch_index * accept_token_stride;

    if (pending_len > 0) {
      for (int64_t index = 0; index < accept_len; ++index) {
        atomic_store_relaxed(
            seat_pending + (committed_len + index) % tail_capacity,
            static_cast<int64_t>(row_accept_tokens[index]));
      }
      committed_len += accept_len;
      pending_len += accept_len;
      atomic_store_relaxed(committed_lens + seat, committed_len);
      atomic_store_relaxed(pending_expected_lens + seat, pending_len);
      atomic_store_relaxed(consumable_tail_lens + seat, 0);
    } else {
      int64_t match_len = 0;
      const int64_t max_match =
          accept_len < raw_tail_len ? accept_len : raw_tail_len;
      while (
          match_len < max_match &&
          atomic_load_relaxed(seat_tail + match_len) ==
              row_accept_tokens[match_len]) {
        ++match_len;
      }
      const bool resident_mismatch = match_len < raw_tail_len;

      if (match_len > 0) {
        for (int64_t i = match_len; i < raw_tail_len; ++i) {
          atomic_store_relaxed(
              seat_tail + i - match_len,
              atomic_load_relaxed(seat_tail + i));
        }
        raw_tail_len -= match_len;
        atomic_store_relaxed(raw_tail_lens + seat, raw_tail_len);
      }

      committed_len += accept_len;
      atomic_store_relaxed(committed_lens + seat, committed_len);

      if (match_len < accept_len) {
        if (resident_mismatch) {
          const int64_t mismatch_fence =
              committed_len - accept_len + match_len + 1;
          if (mismatch_fence > can_accept_prefix_len) {
            can_accept_prefix_len = mismatch_fence;
            atomic_store_relaxed(
                can_accept_prefix_lens + seat, can_accept_prefix_len);
          }
        }
        atomic_store_relaxed(raw_tail_lens + seat, 0);
        atomic_store_relaxed(consumable_tail_lens + seat, 0);
        pending_len = accept_len - match_len;
        for (int64_t index = 0; index < pending_len; ++index) {
          atomic_store_relaxed(
              seat_pending +
                  (committed_len - accept_len + match_len + index) %
                      tail_capacity,
              static_cast<int64_t>(row_accept_tokens[match_len + index]));
        }
        atomic_store_relaxed(pending_expected_lens + seat, pending_len);
      } else {
        atomic_store_relaxed(consumable_tail_lens + seat, raw_tail_len);
      }
    }
  }

  release_tail_writer(version, writer_version);
}

__device__ __forceinline__ int64_t load_sample_token(
    const int32_t* tokens_int32,
    const int64_t* tokens_int64,
    bool tokens_are_int32,
    int64_t index) {
  return tokens_are_int32 ? static_cast<int64_t>(tokens_int32[index])
                          : tokens_int64[index];
}

__global__ void prepare_gpu_drafter_decode_kernel(
    const int64_t* __restrict__ mirror_seats,
    const int64_t* __restrict__ expected_request_epochs,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ candidate_out_cache_locs,
    const int64_t* __restrict__ checkpoint_slot_table,
    int64_t checkpoint_capacity,
    int32_t* __restrict__ req_to_token,
    int64_t req_to_token_num_rows,
    int64_t req_to_token_row_stride,
    int64_t* __restrict__ resolved_input_ids,
    int64_t* __restrict__ resolved_seq_lens,
    int32_t* __restrict__ resolved_orig_seq_lens,
    int64_t* __restrict__ mamba_src_indices,
    int64_t* __restrict__ mamba_dst_indices,
    int64_t* __restrict__ captured_state_positions,
    int64_t* __restrict__ old_cache_locs,
    int64_t* __restrict__ kv_ownership,
    int64_t batch_size,
    int64_t num_seats,
    int64_t* __restrict__ versions,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ prompt_lens,
    const int64_t* __restrict__ committed_lens,
    const int64_t* __restrict__ can_accept_prefix_lens,
    const int64_t* __restrict__ raw_tail_lens,
    const int64_t* __restrict__ pending_expected_lens,
    const int64_t* __restrict__ last_op_seqs,
    int64_t* __restrict__ error_codes,
    int64_t* __restrict__ error_op_seqs,
    const int64_t* __restrict__ model_output_lens,
    const int64_t* __restrict__ model_state_positions,
    const int64_t* __restrict__ model_input_tokens,
    const int64_t* __restrict__ checkpoint_positions,
    int64_t tail_capacity,
    int64_t pending_token_capacity) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size || threadIdx.x != 0) return;

  // Invalid lifecycle/pacing rows can coexist with live rows in one captured
  // batch. Keep them executable as the established FA/GDN padding row while
  // captured_state_position=-1 tells finish that no candidate was bound.
  resolved_input_ids[row] = 0;
  resolved_seq_lens[row] = 1;
  resolved_orig_seq_lens[row] = 1;
  if (mamba_src_indices != nullptr) {
    mamba_src_indices[row] = -1;
    mamba_dst_indices[row] = -1;
  }
  captured_state_positions[row] = -1;
  old_cache_locs[row] = -1;

  const int64_t seat = mirror_seats[row];
  if (seat < 0 || seat >= num_seats) return;
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(versions[seat]);
  const int64_t writer_version = acquire_tail_writer(version);
  if (atomic_load_relaxed(request_epochs + seat) !=
      expected_request_epochs[row]) {
    release_tail_writer(version, writer_version);
    return;
  }

  const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
  const int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
  const int64_t ack_ready_len =
      atomic_load_relaxed(can_accept_prefix_lens + seat);
  const int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
  const int64_t pending_len =
      atomic_load_relaxed(pending_expected_lens + seat);
  const int64_t model_output_len =
      atomic_load_relaxed(model_output_lens + seat);
  const int64_t state_position =
      atomic_load_relaxed(model_state_positions + seat);
  const int64_t input_token = atomic_load_relaxed(model_input_tokens + seat);
  const int64_t req_pool_index = req_pool_indices[row];
  const int64_t candidate_loc = candidate_out_cache_locs[row];
  const bool state_valid =
      atomic_load_relaxed(error_codes + seat) == 0 && prompt_len >= 0 &&
      committed_len >= 0 && ack_ready_len >= 0 &&
      ack_ready_len <= committed_len && raw_tail_len >= 0 &&
      raw_tail_len <= tail_capacity && pending_len >= 0 &&
      pending_len <= pending_token_capacity &&
      (pending_len == 0 || raw_tail_len == 0) && model_output_len > 0 &&
      model_output_len + pending_len == committed_len + raw_tail_len &&
      state_position == prompt_len + model_output_len - 1 && input_token >= 0;
  if (!state_valid) {
    if (atomic_load_relaxed(error_codes + seat) == 0) {
      latch_update_error(
          seat,
          atomic_load_relaxed(last_op_seqs + seat),
          GpuDraftTailUpdateError::kDrafterModelStateInvalid,
          error_codes,
          error_op_seqs);
    }
    release_tail_writer(version, writer_version);
    return;
  }
  if (raw_tail_len == tail_capacity) {
    // Compact CPU pacing can lag already-enqueued GPU work by one forward.
    // A full transcript is ordinary backpressure, not a poisoned request.
    // Returning before binding the physical candidate also enforces the single
    // unretired-writer modulo proof used by direct mismatch rollback: no dst
    // can be a full ring ahead of a resident restore checkpoint.
    release_tail_writer(version, writer_version);
    return;
  }
  const bool metadata_valid =
      req_pool_index >= 0 &&
      req_pool_index < req_to_token_num_rows &&
      state_position < req_to_token_row_stride && candidate_loc > 0 &&
      candidate_loc <= int64_t{0x7fffffff};
  if (!metadata_valid) {
    if (atomic_load_relaxed(error_codes + seat) == 0) {
      latch_update_error(
          seat,
          atomic_load_relaxed(last_op_seqs + seat),
          GpuDraftTailUpdateError::kDrafterModelStateInvalid,
          error_codes,
          error_op_seqs);
    }
    release_tail_writer(version, writer_version);
    return;
  }

  const int64_t src_offset = state_position % checkpoint_capacity;
  const int64_t dst_offset = (state_position + 1) % checkpoint_capacity;
  if (atomic_load_relaxed(
          checkpoint_positions + seat * checkpoint_capacity + src_offset) !=
      state_position) {
    latch_update_error(
        seat,
        atomic_load_relaxed(last_op_seqs + seat),
        GpuDraftTailUpdateError::kDrafterCheckpointUnavailable,
        error_codes,
        error_op_seqs);
    release_tail_writer(version, writer_version);
    return;
  }
  // Dense attention restores its prefix through KV positions alone. Recurrent
  // drafts additionally require physical state slots from the checkpoint ring.
  const int64_t src_slot = checkpoint_slot_table == nullptr ? -1 :
      checkpoint_slot_table[seat * checkpoint_capacity + src_offset];
  const int64_t dst_slot = checkpoint_slot_table == nullptr ? -1 :
      checkpoint_slot_table[seat * checkpoint_capacity + dst_offset];
  if (checkpoint_slot_table != nullptr && (src_slot < 0 || dst_slot < 0)) {
    latch_update_error(
        seat,
        atomic_load_relaxed(last_op_seqs + seat),
        GpuDraftTailUpdateError::kDrafterCheckpointUnavailable,
        error_codes,
        error_op_seqs);
    release_tail_writer(version, writer_version);
    return;
  }

  int32_t* logical_cache_loc =
      req_to_token + req_pool_index * req_to_token_row_stride + state_position;
  int64_t* ownership = kv_ownership + seat * 2;
  const int64_t epoch = expected_request_epochs[row];
  if (ownership[0] != epoch) {
    // The first decode starts immediately after the owned prefill prefix.
    // Later rollback must retain the largest bound position of this lifetime.
    ownership[0] = epoch;
    ownership[1] = state_position;
  }
  const int64_t old_loc = state_position < ownership[1]
      ? static_cast<int64_t>(*logical_cache_loc) : -1;
  ownership[1] = max(ownership[1], state_position + 1);
  *logical_cache_loc = static_cast<int32_t>(candidate_loc);
  resolved_input_ids[row] = input_token;
  resolved_seq_lens[row] = state_position + 1;
  resolved_orig_seq_lens[row] = static_cast<int32_t>(state_position + 1);
  if (mamba_src_indices != nullptr) {
    mamba_src_indices[row] = src_slot;
    mamba_dst_indices[row] = dst_slot;
  }
  captured_state_positions[row] = state_position;
  old_cache_locs[row] = old_loc;
  release_tail_writer(version, writer_version);
}

__global__ void finish_gpu_drafter_decode_kernel(
    const int64_t* __restrict__ mirror_seats,
    const int64_t* __restrict__ expected_request_epochs,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ candidate_out_cache_locs,
    const int32_t* __restrict__ sampled_tokens_int32,
    const int64_t* __restrict__ sampled_tokens_int64,
    bool sampled_tokens_are_int32,
    int32_t* __restrict__ req_to_token,
    int64_t req_to_token_num_rows,
    int64_t req_to_token_row_stride,
    const int64_t* resolved_input_tokens,
    const int64_t* __restrict__ captured_state_positions,
    const int64_t* __restrict__ old_cache_locs,
    int64_t* __restrict__ kv_outcomes,
    int64_t* future_output_tokens,
    int64_t future_output_tokens_size,
    int64_t batch_size,
    int64_t num_seats,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    int64_t* __restrict__ versions,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ prompt_lens,
    const int64_t* __restrict__ committed_lens,
    int64_t* __restrict__ can_accept_prefix_lens,
    int64_t* __restrict__ raw_tail_lens,
    int64_t* __restrict__ consumable_tail_lens,
    int64_t* __restrict__ pending_expected_lens,
    const int64_t* __restrict__ pending_expected_tokens,
    int64_t* __restrict__ tail_tokens,
    const int64_t* __restrict__ last_op_seqs,
    int64_t* __restrict__ error_codes,
    int64_t* __restrict__ error_op_seqs,
    int64_t* __restrict__ model_output_lens,
    int64_t* __restrict__ model_state_positions,
    int64_t* __restrict__ model_input_tokens,
    int64_t* __restrict__ checkpoint_positions,
    int64_t* __restrict__ egress_seqs,
    int64_t* __restrict__ last_commit_tokens) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size || threadIdx.x != 0) return;
  int64_t* outcome = kv_outcomes + row * 3;
  outcome[0] = 0;
  outcome[1] = -1;
  outcome[2] = -1;

  const int64_t candidate_loc = candidate_out_cache_locs[row];
  const int64_t seat = mirror_seats[row];
  if (seat < 0 || seat >= num_seats) {
    // A dummy lifecycle row never acquired transcript/KV ownership, but its
    // batch allocator candidate still belongs to this finish epilogue.
    if (candidate_loc > 0) outcome[2] = candidate_loc;
    return;
  }
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(versions[seat]);
  const int64_t writer_version = acquire_tail_writer(version);
  const int64_t expected_epoch = expected_request_epochs[row];
  const int64_t resident_epoch = atomic_load_relaxed(request_epochs + seat);
  const int64_t captured_state = captured_state_positions[row];
  // BS1 may use a view into future_output_tokens as its resolved input.
  // These two pointers deliberately are not restrict-qualified. Load the
  // input before updating its relay slot below; no other stream writes it.
  const int64_t captured_input = resolved_input_tokens[row];
  const int64_t req_pool_index = req_pool_indices[row];
  const int64_t old_loc = old_cache_locs[row];
  const int64_t sampled_token = load_sample_token(
      sampled_tokens_int32,
      sampled_tokens_int64,
      sampled_tokens_are_int32,
      row);

  const bool prepare_bound =
      captured_state >= 0 && req_pool_index >= 0 &&
      req_pool_index < req_to_token_num_rows &&
      captured_state < req_to_token_row_stride && candidate_loc > 0;
  const bool candidate_owned =
      prepare_bound && resident_epoch == expected_epoch;
  if (candidate_owned) outcome[1] = captured_state;

  // One prepare/forward/finish is in flight per seat on the forward stream.
  // A rollback either lowers the position or replaces the current input.
  // Later commits only queue forced tokens, so they cannot restore this pair
  // before finish. All-match commits preserve it and keep the forward useful.
  bool can_commit = candidate_owned &&
      captured_input >= 0 && sampled_token >= 0 &&
      atomic_load_relaxed(error_codes + seat) == 0 &&
      atomic_load_relaxed(model_state_positions + seat) == captured_state &&
      atomic_load_relaxed(model_input_tokens + seat) == captured_input;
  int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
  int64_t num_forced_tokens =
      atomic_load_relaxed(pending_expected_lens + seat);
  const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
  const int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
  const int64_t ack_ready_len =
      atomic_load_relaxed(can_accept_prefix_lens + seat);
  const int64_t model_output_len =
      atomic_load_relaxed(model_output_lens + seat);
  const int64_t model_input_token =
      atomic_load_relaxed(model_input_tokens + seat);
  if (can_commit &&
      (raw_tail_len < 0 || raw_tail_len > tail_capacity ||
       num_forced_tokens < 0 ||
       num_forced_tokens > pending_token_capacity ||
       (num_forced_tokens > 0 && raw_tail_len != 0) || prompt_len < 0 ||
       committed_len < 0 || ack_ready_len < 0 ||
       ack_ready_len > committed_len || model_output_len <= 0 ||
       model_output_len + num_forced_tokens !=
           committed_len + raw_tail_len ||
       captured_state != prompt_len + model_output_len - 1 ||
       model_input_token < 0)) {
    latch_update_error(
        seat,
        atomic_load_relaxed(last_op_seqs + seat),
        raw_tail_len > tail_capacity
            ? GpuDraftTailUpdateError::kTailCapacityExceeded
            : GpuDraftTailUpdateError::kDrafterModelStateInvalid,
        error_codes,
        error_op_seqs);
    can_commit = false;
  }
  if (can_commit && raw_tail_len == tail_capacity) {
    // Compact pacing progress is intentionally asynchronous with respect to
    // already-enqueued forwards. A final stale forward can therefore reach
    // finish after this bounded transcript became full. Rejecting its sample
    // is ordinary backpressure: retain candidate KV ownership for the
    // allocator high-water invariant, but do not poison the request row.
    can_commit = false;
  }

  if (can_commit) {
    int64_t relay_token = sampled_token;
    if (num_forced_tokens > 0) {
      // This forward materialized the checkpoint after the current forced
      // input. Its speculative sample is obsolete; advance exactly one step
      // through the verifier-owned queue and relay that next input instead.
      relay_token = atomic_load_relaxed(
          pending_expected_tokens +
          seat * pending_token_capacity +
          model_output_len % pending_token_capacity);
      --num_forced_tokens;
      atomic_store_relaxed(
          pending_expected_lens + seat, num_forced_tokens);
      atomic_store_relaxed(consumable_tail_lens + seat, 0);
    } else {
      atomic_store_relaxed(
          tail_tokens + seat * tail_capacity + raw_tail_len, sampled_token);
      ++raw_tail_len;
      atomic_store_relaxed(raw_tail_lens + seat, raw_tail_len);
      atomic_store_relaxed(consumable_tail_lens + seat, raw_tail_len);
      if (ack_ready_len < committed_len) {
        // Processing the last verifier-committed input produced a private
        // lookahead, making the entire cumulative prefix safe to ACK.
        atomic_store_relaxed(
            can_accept_prefix_lens + seat, committed_len);
        atomic_store_relaxed(
            last_commit_tokens + seat, model_input_token);
      }
    }
    atomic_store_relaxed(model_output_lens + seat, model_output_len + 1);
    atomic_store_relaxed(model_state_positions + seat, captured_state + 1);
    atomic_store_relaxed(model_input_tokens + seat, relay_token);
    if (req_pool_index < future_output_tokens_size) {
      future_output_tokens[req_pool_index] = relay_token;
    }
    atomic_store_relaxed(
        checkpoint_positions +
            seat * tail_capacity + (captured_state + 1) % tail_capacity,
        captured_state + 1);
    outcome[0] = 1;
    atomic_store_relaxed(
        egress_seqs + seat,
        atomic_load_relaxed(egress_seqs + seat) + 1);
    if (old_loc > 0 && old_loc != candidate_loc) outcome[2] = old_loc;
  } else {
    if (candidate_owned) {
      // A branch update can invalidate this forward after prepare bound its KV
      // location. Keep that location as the owned (but stale) logical cell so
      // the CPU allocator's high-water release range never contains a hole.
      // The next prepare at this position replaces it and reclaims it.
      if (old_loc > 0 && old_loc != candidate_loc) {
        outcome[2] = old_loc;
      }
    } else {
      // A lifecycle change can make the request row independently releasable.
      // Restore the previous owner and return the unowned candidate instead of
      // leaving a post-CLOSE allocation in a row that may already be released.
      if (prepare_bound) {
        req_to_token[
            req_pool_index * req_to_token_row_stride + captured_state] =
            static_cast<int32_t>(old_loc);
      }
      if (candidate_loc > 0 && candidate_loc != old_loc) {
        outcome[2] = candidate_loc;
      }
    }
    if (resident_epoch == expected_epoch && req_pool_index >= 0 &&
        req_pool_index < future_output_tokens_size) {
      future_output_tokens[req_pool_index] =
          atomic_load_relaxed(model_input_tokens + seat);
    }
  }
  release_tail_writer(version, writer_version);
}

__global__ void append_gpu_drafter_prefill_sample_kernel(
    const int64_t* __restrict__ mirror_seats,
    const int64_t* __restrict__ expected_request_epochs,
    int32_t* __restrict__ sampled_tokens_int32,
    int64_t* __restrict__ sampled_tokens_int64,
    bool sampled_tokens_are_int32,
    bool* __restrict__ accept_out,
    int64_t batch_size,
    int64_t num_seats,
    int64_t tail_capacity,
    int64_t pending_token_capacity,
    int64_t* __restrict__ versions,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ prompt_lens,
    const int64_t* __restrict__ committed_lens,
    const int64_t* __restrict__ can_accept_prefix_lens,
    int64_t* __restrict__ raw_tail_lens,
    int64_t* __restrict__ consumable_tail_lens,
    int64_t* __restrict__ pending_expected_lens,
    const int64_t* __restrict__ pending_expected_tokens,
    int64_t* __restrict__ tail_tokens,
    const int64_t* __restrict__ last_op_seqs,
    int64_t* __restrict__ error_codes,
    int64_t* __restrict__ error_op_seqs,
    int64_t* __restrict__ model_output_lens,
    int64_t* __restrict__ model_state_positions,
    int64_t* __restrict__ model_input_tokens,
    int64_t* __restrict__ checkpoint_positions,
    int64_t* __restrict__ egress_seqs) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size || threadIdx.x != 0) return;
  accept_out[row] = false;
  const int64_t seat = mirror_seats[row];
  if (seat < 0 || seat >= num_seats) return;
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(versions[seat]);
  const int64_t writer_version = acquire_tail_writer(version);
  if (atomic_load_relaxed(request_epochs + seat) !=
      expected_request_epochs[row]) {
    release_tail_writer(version, writer_version);
    return;
  }

  const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
  const int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
  const int64_t ack_ready_len =
      atomic_load_relaxed(can_accept_prefix_lens + seat);
  const int64_t model_output_len =
      atomic_load_relaxed(model_output_lens + seat);
  const int64_t state_position =
      atomic_load_relaxed(model_state_positions + seat);
  const int64_t sampled_token = load_sample_token(
      sampled_tokens_int32,
      sampled_tokens_int64,
      sampled_tokens_are_int32,
      row);
  int64_t num_forced_tokens =
      atomic_load_relaxed(pending_expected_lens + seat);
  const bool metadata_valid =
      atomic_load_relaxed(error_codes + seat) == 0 && prompt_len >= 0 &&
      committed_len >= 0 && ack_ready_len >= 0 &&
      ack_ready_len <= committed_len && model_output_len >= 0 &&
      num_forced_tokens >= 0 &&
      num_forced_tokens <= pending_token_capacity &&
      model_output_len + num_forced_tokens == committed_len &&
      state_position == prompt_len + model_output_len &&
      atomic_load_relaxed(model_input_tokens + seat) == -1 &&
      atomic_load_relaxed(raw_tail_lens + seat) == 0 &&
      sampled_token >= 0 && tail_capacity > 0;
  if (!metadata_valid) {
    if (atomic_load_relaxed(error_codes + seat) == 0) {
      latch_update_error(
          seat,
          atomic_load_relaxed(last_op_seqs + seat),
          GpuDraftTailUpdateError::kDrafterModelStateInvalid,
          error_codes,
          error_op_seqs);
    }
    release_tail_writer(version, writer_version);
    return;
  }

  int64_t relay_token = sampled_token;
  if (num_forced_tokens > 0) {
    // A verifier commit can land after the prefill forward was launched but
    // before this epilogue. The sampled token is then obsolete: install the
    // first verifier-owned token as the next model input and rewrite the
    // device result tensor so final-prefill result admission observes the same
    // authoritative value without a CPU transcript.
    relay_token = atomic_load_relaxed(
        pending_expected_tokens +
        seat * pending_token_capacity +
        model_output_len % pending_token_capacity);
    --num_forced_tokens;
    atomic_store_relaxed(
        pending_expected_lens + seat, num_forced_tokens);
    if (sampled_tokens_are_int32) {
      sampled_tokens_int32[row] = static_cast<int32_t>(relay_token);
    } else {
      sampled_tokens_int64[row] = relay_token;
    }
    atomic_store_relaxed(raw_tail_lens + seat, 0);
    atomic_store_relaxed(consumable_tail_lens + seat, 0);
  } else {
    atomic_store_relaxed(
        tail_tokens + seat * tail_capacity, sampled_token);
    atomic_store_relaxed(raw_tail_lens + seat, 1);
    atomic_store_relaxed(consumable_tail_lens + seat, 1);
  }
  atomic_store_relaxed(model_output_lens + seat, model_output_len + 1);
  atomic_store_relaxed(model_input_tokens + seat, relay_token);
  atomic_store_relaxed(
      checkpoint_positions +
          seat * tail_capacity + state_position % tail_capacity,
      state_position);
  atomic_store_relaxed(
      egress_seqs + seat,
      atomic_load_relaxed(egress_seqs + seat) + 1);
  accept_out[row] = true;
  release_tail_writer(version, writer_version);
}

__global__ void snapshot_gpu_drafter_egress_kernel(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ seat_indices,
    int64_t num_rows,
    int64_t num_seats,
    int64_t tail_capacity,
    const int64_t* __restrict__ versions,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ committed_lens,
    const int64_t* __restrict__ can_accept_prefix_lens,
    const int64_t* __restrict__ raw_tail_lens,
    const int64_t* __restrict__ pending_expected_lens,
    const int64_t* __restrict__ tail_tokens,
    const int64_t* __restrict__ error_codes,
    const int64_t* __restrict__ model_output_lens,
    const int64_t* __restrict__ egress_seqs,
    const int64_t* __restrict__ last_commit_tokens) {
  const int64_t row_index = static_cast<int64_t>(blockIdx.x);
  if (row_index >= num_rows || threadIdx.x != 0) return;
  const int64_t seat = seat_indices[row_index];
  if (seat < 0 || seat >= num_seats) return;
  const int64_t row_width = kGpuDrafterEgressMetadataWidth + tail_capacity;
  int64_t* row = out + row_index * row_width;
  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(
      *const_cast<int64_t*>(versions + seat));

  while (true) {
    const int64_t version_before = version.load(cuda::memory_order_acquire);
    if ((version_before & int64_t{1}) != 0) {
      __nanosleep(64);
      continue;
    }

    const int64_t request_epoch =
        atomic_load_relaxed(request_epochs + seat);
    const int64_t egress_seq = atomic_load_relaxed(egress_seqs + seat);
    const int64_t committed_len =
        atomic_load_relaxed(committed_lens + seat);
    const int64_t materialized_output_len =
        atomic_load_relaxed(model_output_lens + seat);
    const int64_t num_forced_tokens =
        atomic_load_relaxed(pending_expected_lens + seat);
    const int64_t logical_output_len =
        materialized_output_len + num_forced_tokens;
    const int64_t ack_ready_len =
        atomic_load_relaxed(can_accept_prefix_lens + seat);
    const int64_t last_commit_token =
        atomic_load_relaxed(last_commit_tokens + seat);
    const int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
    const int64_t error_code = atomic_load_relaxed(error_codes + seat);
    const bool tail_len_valid =
        raw_tail_len >= 0 && raw_tail_len <= tail_capacity;

    row[0] = request_epoch;
    row[1] = egress_seq;
    row[2] = committed_len;
    row[3] = logical_output_len;
    row[4] = ack_ready_len;
    row[5] = last_commit_token;
    row[6] = raw_tail_len;
    row[7] = error_code;
    for (int64_t index = 0; index < tail_capacity; ++index) {
      row[kGpuDrafterEgressMetadataWidth + index] =
          tail_len_valid && index < raw_tail_len
          ? atomic_load_relaxed(
                tail_tokens + seat * tail_capacity + index)
          : 0;
    }

    cuda::atomic_thread_fence(
        cuda::memory_order_acquire, cuda::thread_scope_device);
    const int64_t version_after = version.load(cuda::memory_order_relaxed);
    if (version_before == version_after &&
        (version_after & int64_t{1}) == 0) {
      return;
    }
    if ((version_after & int64_t{1}) != 0) __nanosleep(64);
  }
}

__global__ void select_gpu_draft_tail_kernel(
    const int64_t* __restrict__ gpu_seats,
    const int64_t* __restrict__ expected_request_epochs,
    const int64_t* __restrict__ seq_lens,
    const int32_t* __restrict__ bonus_tokens_int32,
    const int64_t* __restrict__ bonus_tokens_int64,
    bool bonus_tokens_are_int32,
    int64_t* __restrict__ compact_out,
    int64_t* __restrict__ logical_committed_lens_out,
    int64_t* __restrict__ debug_out,
    int64_t debug_width,
    int64_t batch_size,
    int64_t num_seats,
    int64_t num_draft_tokens,
    int64_t tail_capacity,
    const int64_t* __restrict__ versions,
    const int64_t* __restrict__ publish_seqs,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ prompt_lens,
    const int64_t* __restrict__ committed_lens,
    const int64_t* __restrict__ raw_tail_lens,
    const int64_t* __restrict__ consumable_tail_lens,
    const int64_t* __restrict__ pending_expected_lens,
    const int64_t* __restrict__ tail_tokens,
    const int64_t* __restrict__ error_codes,
    const int64_t* __restrict__ error_op_seqs,
    const int64_t* __restrict__ pending_prefix_fast_forward_cts,
    bool allow_partial,
    int64_t required_tail_len) {
  for (int64_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;
       batch_index < batch_size;
       batch_index += gridDim.x * blockDim.x) {
    // VerifyCommit is applied from the exact accepted run before the next
    // direct-only selector; bonus tokens no longer participate in selection.
    (void)bonus_tokens_int32;
    (void)bonus_tokens_int64;
    (void)bonus_tokens_are_int32;

    const int64_t out_width = num_draft_tokens + 2;
    int64_t* out = compact_out + batch_index * out_width;
    for (int64_t i = 0; i < num_draft_tokens; ++i) out[i] = 0;
    out[num_draft_tokens] = 0;
    out[num_draft_tokens + 1] = 0;
    int64_t* debug = debug_out == nullptr
        ? nullptr
        : debug_out + batch_index * debug_width;
    if (debug != nullptr) {
      for (int64_t i = 0; i < debug_width; ++i) debug[i] = -1;
      debug[0] = static_cast<int64_t>(GpuDraftTailSelectReason::kUnset);
      if (debug_width > 7) debug[7] = 0;
      if (debug_width > 10) debug[10] = 0;
    }
    if (logical_committed_lens_out != nullptr) {
      logical_committed_lens_out[batch_index] = -1;
    }

    const int64_t seat = gpu_seats[batch_index];
    if (seat < 0 || seat >= num_seats) {
      if (debug != nullptr) {
        debug[0] =
            static_cast<int64_t>(GpuDraftTailSelectReason::kInvalidSeat);
      }
      continue;
    }

    cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(
        *const_cast<int64_t*>(versions + seat));
    // The reader never takes the writer lock. Spin only on this request's seat
    // until one complete old or new snapshot is available.
    int64_t seqlock_retries = 0;
    while (true) {
      const int64_t version_before =
          version.load(cuda::memory_order_acquire);
      if ((version_before & int64_t{1}) != 0) {
        ++seqlock_retries;
        __nanosleep(64);
        continue;
      }

      const int64_t publish_seq = atomic_load_relaxed(publish_seqs + seat);
      const int64_t request_epoch = atomic_load_relaxed(request_epochs + seat);
      const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
      const int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
      const int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
      const int64_t consumable_tail_len =
          atomic_load_relaxed(consumable_tail_lens + seat);
      const int64_t pending_expected_len =
          atomic_load_relaxed(pending_expected_lens + seat);
      const int64_t update_error_code = atomic_load_relaxed(error_codes + seat);
      const int64_t update_error_op_seq =
          atomic_load_relaxed(error_op_seqs + seat);
      const int64_t pending_prefix_fast_forward_ct =
          debug != nullptr && debug_width > 9
              ? atomic_load_relaxed(pending_prefix_fast_forward_cts + seat)
              : -1;
      const int64_t expected_epoch = expected_request_epochs[batch_index];
      const int64_t logical_output_len = seq_lens[batch_index] - prompt_len + 1;
      const bool identity_valid =
          request_epoch == expected_epoch && prompt_len >= 0;
      const int64_t delta = logical_output_len - committed_len;

      bool row_valid = false;
      GpuDraftTailSelectReason reason = GpuDraftTailSelectReason::kUnset;
      const bool metadata_valid = committed_len >= 0 && raw_tail_len >= 0 &&
          consumable_tail_len >= 0 && consumable_tail_len <= raw_tail_len &&
          raw_tail_len <= tail_capacity && pending_expected_len >= 0 &&
          pending_expected_len <= committed_len &&
          update_error_code == 0;
      if (!identity_valid) {
        reason = GpuDraftTailSelectReason::kIdentityMismatch;
      } else if (!metadata_valid) {
        reason = GpuDraftTailSelectReason::kMetadataInvalid;
      } else if (delta != 0) {
        reason = GpuDraftTailSelectReason::kLogicalCursorMismatch;
      } else if (pending_expected_len > 0) {
        reason = GpuDraftTailSelectReason::kPendingPrefix;
      } else {
        row_valid = true;
        reason = GpuDraftTailSelectReason::kDirect;
      }

      int64_t selected_len = 0;
      if (row_valid) {
        selected_len = consumable_tail_len;
        if (selected_len > num_draft_tokens) selected_len = num_draft_tokens;
        if (allow_partial || selected_len >= required_tail_len) {
          for (int64_t i = 0; i < selected_len; ++i) {
            out[i] = atomic_load_relaxed(
                tail_tokens + seat * tail_capacity + i);
          }
        }
      }

      // Close the read section before sampling the version again. The fence
      // orders every protected relaxed load before this validation load.
      cuda::atomic_thread_fence(
          cuda::memory_order_acquire, cuda::thread_scope_device);
      const int64_t version_after =
          version.load(cuda::memory_order_relaxed);
      if (version_before != version_after ||
          (version_after & int64_t{1}) != 0) {
        for (int64_t i = 0; i < num_draft_tokens; ++i) out[i] = 0;
        ++seqlock_retries;
        if ((version_after & int64_t{1}) != 0) __nanosleep(64);
        continue;
      }

      // Do not hold the writer lock while waiting: the landing stream must be
      // able to resolve pending commits and append the remaining draft tokens.
      // Lifecycle/cursor/update errors still return the existing invalid snapshot.
      if (!allow_partial && identity_valid && metadata_valid && delta == 0 &&
          (pending_expected_len > 0 || selected_len < required_tail_len)) {
        __nanosleep(64);
        continue;
      }

      if (logical_committed_lens_out != nullptr && identity_valid) {
        logical_committed_lens_out[batch_index] = logical_output_len;
      }
      out[num_draft_tokens] = selected_len;
      out[num_draft_tokens + 1] = row_valid ? 1 : 0;
      if (debug != nullptr) {
        debug[0] = static_cast<int64_t>(reason);
        debug[1] = publish_seq;
        debug[2] = delta;
        debug[3] = raw_tail_len;
        debug[4] = consumable_tail_len;
        debug[5] = pending_expected_len;
        debug[6] = committed_len;
        if (debug_width > 7) debug[7] = update_error_code;
        if (debug_width > 8) debug[8] = update_error_op_seq;
        if (debug_width > 9) debug[9] = pending_prefix_fast_forward_ct;
        if (debug_width > 10) debug[10] = seqlock_retries;
      }
      break;
    }
  }
}

__global__ void select_mock_gpu_draft_tail_kernel(
    const int64_t* __restrict__ gpu_seats,
    const int64_t* __restrict__ seq_lens,
    const int32_t* __restrict__ bonus_tokens_int32,
    const int64_t* __restrict__ bonus_tokens_int64,
    bool bonus_tokens_are_int32,
    int64_t* __restrict__ compact_out,
    int64_t* __restrict__ logical_committed_lens_out,
    int64_t* __restrict__ debug_out,
    int64_t debug_width,
    int64_t batch_size,
    int64_t num_seats,
    int64_t num_draft_tokens,
    int64_t tail_capacity,
    const int64_t* __restrict__ mock_tail_tokens) {
  const int64_t batch_index = static_cast<int64_t>(blockIdx.x);
  if (batch_index >= batch_size || threadIdx.x != 0) return;

  const int64_t out_width = num_draft_tokens + 2;
  int64_t* out = compact_out + batch_index * out_width;
  for (int64_t i = 0; i < num_draft_tokens; ++i) out[i] = 0;
  out[num_draft_tokens] = 0;
  out[num_draft_tokens + 1] = 0;
  int64_t* debug = debug_out == nullptr
      ? nullptr
      : debug_out + batch_index * debug_width;
  if (debug != nullptr) {
    for (int64_t i = 0; i < debug_width; ++i) debug[i] = -1;
    debug[0] = static_cast<int64_t>(GpuDraftTailSelectReason::kUnset);
    if (debug_width > 7) debug[7] = 0;
    if (debug_width > 8) debug[8] = -1;
    if (debug_width > 9) debug[9] = 0;
    if (debug_width > 10) debug[10] = 0;
  }
  if (logical_committed_lens_out != nullptr) {
    // Keep manager-side commit accounting on the real request state.
    logical_committed_lens_out[batch_index] = -1;
  }

  const int64_t seat = gpu_seats[batch_index];
  if (seat < 0 || seat >= num_seats) {
    if (debug != nullptr) {
      debug[0] =
          static_cast<int64_t>(GpuDraftTailSelectReason::kInvalidSeat);
    }
    return;
  }

  for (int64_t i = 0; i < num_draft_tokens; ++i) {
    out[i] = atomic_load_relaxed(
        mock_tail_tokens + seat * tail_capacity + i);
  }
  out[num_draft_tokens] = num_draft_tokens;
  out[num_draft_tokens + 1] = 1;
  if (debug != nullptr) {
    const int64_t bonus_token = bonus_tokens_are_int32
        ? static_cast<int64_t>(bonus_tokens_int32[batch_index])
        : bonus_tokens_int64[batch_index];
    debug[0] = static_cast<int64_t>(GpuDraftTailSelectReason::kDirect);
    debug[1] = -1;
    debug[2] = 0;
    debug[3] = num_draft_tokens;
    debug[4] = num_draft_tokens;
    debug[5] = 0;
    // Preserve real device reads of seq_lens and bonus_tokens without making
    // either value part of the persistent profile-tail state machine.
    debug[6] = seq_lens[batch_index] + (bonus_token & int64_t{1});
  }
}

}  // namespace

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
    void* stream) {
  if (num_groups <= 0) return;
  update_gpu_draft_tail_kernel<<<
      static_cast<unsigned int>(num_groups),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      update_rows,
      group_offsets,
      num_groups,
      tail_capacity,
      pending_token_capacity,
      versions,
      publish_seqs,
      request_epochs,
      prompt_lens,
      committed_lens,
      can_accept_prefix_lens,
      raw_tail_lens,
      consumable_tail_lens,
      pending_expected_lens,
      pending_expected_tokens,
      tail_tokens,
      last_op_seqs,
      error_codes,
      error_op_seqs,
      pending_prefix_fast_forward_cts,
      drafter_authoritative,
      model_output_lens,
      model_state_positions,
      model_input_tokens,
      checkpoint_positions,
      egress_seqs,
      last_commit_tokens);
  check_cuda(cudaGetLastError(), "update_gpu_draft_tail_kernel");
}

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
    void* stream) {
  if (batch_size <= 0) return;
  prepare_gpu_drafter_decode_kernel<<<
      static_cast<unsigned int>(batch_size),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      mirror_seats,
      expected_request_epochs,
      req_pool_indices,
      candidate_out_cache_locs,
      checkpoint_slot_table,
      checkpoint_capacity,
      req_to_token,
      req_to_token_num_rows,
      req_to_token_row_stride,
      resolved_input_ids,
      resolved_seq_lens,
      resolved_orig_seq_lens,
      mamba_src_indices,
      mamba_dst_indices,
      captured_state_positions,
      old_cache_locs,
      kv_ownership,
      batch_size,
      num_seats,
      versions,
      request_epochs,
      prompt_lens,
      committed_lens,
      can_accept_prefix_lens,
      raw_tail_lens,
      pending_expected_lens,
      last_op_seqs,
      error_codes,
      error_op_seqs,
      model_output_lens,
      model_state_positions,
      model_input_tokens,
      checkpoint_positions,
      tail_capacity,
      pending_token_capacity);
  check_cuda(cudaGetLastError(), "prepare_gpu_drafter_decode_kernel");
}

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
    void* stream) {
  if (batch_size <= 0) return;
  finish_gpu_drafter_decode_kernel<<<
      static_cast<unsigned int>(batch_size),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      mirror_seats,
      expected_request_epochs,
      req_pool_indices,
      candidate_out_cache_locs,
      sampled_tokens_are_int32
          ? reinterpret_cast<const int32_t*>(sampled_tokens)
          : nullptr,
      sampled_tokens_are_int32
          ? nullptr
          : reinterpret_cast<const int64_t*>(sampled_tokens),
      sampled_tokens_are_int32,
      req_to_token,
      req_to_token_num_rows,
      req_to_token_row_stride,
      resolved_input_tokens,
      captured_state_positions,
      old_cache_locs,
      kv_outcomes,
      future_output_tokens,
      future_output_tokens_size,
      batch_size,
      num_seats,
      tail_capacity,
      pending_token_capacity,
      versions,
      request_epochs,
      prompt_lens,
      committed_lens,
      can_accept_prefix_lens,
      raw_tail_lens,
      consumable_tail_lens,
      pending_expected_lens,
      pending_expected_tokens,
      tail_tokens,
      last_op_seqs,
      error_codes,
      error_op_seqs,
      model_output_lens,
      model_state_positions,
      model_input_tokens,
      checkpoint_positions,
      egress_seqs,
      last_commit_tokens);
  check_cuda(cudaGetLastError(), "finish_gpu_drafter_decode_kernel");
}

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
    void* stream) {
  if (batch_size <= 0) return;
  append_gpu_drafter_prefill_sample_kernel<<<
      static_cast<unsigned int>(batch_size),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      mirror_seats,
      expected_request_epochs,
      sampled_tokens_are_int32
          ? reinterpret_cast<int32_t*>(sampled_tokens)
          : nullptr,
      sampled_tokens_are_int32
          ? nullptr
          : reinterpret_cast<int64_t*>(sampled_tokens),
      sampled_tokens_are_int32,
      accept_out,
      batch_size,
      num_seats,
      tail_capacity,
      pending_token_capacity,
      versions,
      request_epochs,
      prompt_lens,
      committed_lens,
      can_accept_prefix_lens,
      raw_tail_lens,
      consumable_tail_lens,
      pending_expected_lens,
      pending_expected_tokens,
      tail_tokens,
      last_op_seqs,
      error_codes,
      error_op_seqs,
      model_output_lens,
      model_state_positions,
      model_input_tokens,
      checkpoint_positions,
      egress_seqs);
  check_cuda(
      cudaGetLastError(), "append_gpu_drafter_prefill_sample_kernel");
}

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
    void* stream) {
  if (num_rows <= 0 || num_seats <= 0) return;
  snapshot_gpu_drafter_egress_kernel<<<
      static_cast<unsigned int>(num_rows),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      out,
      seat_indices,
      num_rows,
      num_seats,
      tail_capacity,
      versions,
      request_epochs,
      committed_lens,
      can_accept_prefix_lens,
      raw_tail_lens,
      pending_expected_lens,
      tail_tokens,
      error_codes,
      model_output_lens,
      egress_seqs,
      last_commit_tokens);
  check_cuda(cudaGetLastError(), "snapshot_gpu_drafter_egress_kernel");
}

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
    void* stream) {
  if (batch_size <= 0) return;
  apply_gpu_draft_tail_verify_commit_kernel<<<
      static_cast<unsigned int>(batch_size),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      gpu_seats,
      expected_request_epochs,
      pre_verify_seq_lens,
      accept_tokens,
      num_accept_tokens,
      commit_mask,
      accept_token_stride,
      batch_size,
      num_seats,
      tail_capacity,
      versions,
      request_epochs,
      prompt_lens,
      committed_lens,
      can_accept_prefix_lens,
      raw_tail_lens,
      consumable_tail_lens,
      pending_expected_lens,
      pending_expected_tokens,
      tail_tokens,
      last_op_seqs,
      error_codes,
      error_op_seqs);
  check_cuda(
      cudaGetLastError(),
      "apply_gpu_draft_tail_verify_commit_kernel");
}

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
    int64_t required_tail_len) {
  if (batch_size <= 0) return;
  // A strict selector occupies only one CTA, leaving SM capacity for landing
  // kernels regardless of batch size. Partial selection keeps its usual grid.
  select_gpu_draft_tail_kernel<<<
      allow_partial ? static_cast<unsigned int>(batch_size) : 1,
      allow_partial ? 1 : 32,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      gpu_seats,
      expected_request_epochs,
      seq_lens,
      bonus_tokens_are_int32
          ? reinterpret_cast<const int32_t*>(bonus_tokens)
          : nullptr,
      bonus_tokens_are_int32
          ? nullptr
          : reinterpret_cast<const int64_t*>(bonus_tokens),
      bonus_tokens_are_int32,
      compact_out,
      logical_committed_lens_out,
      debug_out,
      debug_width,
      batch_size,
      num_seats,
      num_draft_tokens,
      tail_capacity,
      versions,
      publish_seqs,
      request_epochs,
      prompt_lens,
      committed_lens,
      raw_tail_lens,
      consumable_tail_lens,
      pending_expected_lens,
      tail_tokens,
      error_codes,
      error_op_seqs,
      pending_prefix_fast_forward_cts,
      allow_partial,
      required_tail_len);
  check_cuda(cudaGetLastError(), "select_gpu_draft_tail_kernel");
}

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
    void* stream) {
  if (batch_size <= 0) return;
  select_mock_gpu_draft_tail_kernel<<<
      static_cast<unsigned int>(batch_size),
      1,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      gpu_seats,
      seq_lens,
      bonus_tokens_are_int32
          ? reinterpret_cast<const int32_t*>(bonus_tokens)
          : nullptr,
      bonus_tokens_are_int32
          ? nullptr
          : reinterpret_cast<const int64_t*>(bonus_tokens),
      bonus_tokens_are_int32,
      compact_out,
      logical_committed_lens_out,
      debug_out,
      debug_width,
      batch_size,
      num_seats,
      num_draft_tokens,
      tail_capacity,
      mock_tail_tokens);
  check_cuda(cudaGetLastError(), "select_mock_gpu_draft_tail_kernel");
}

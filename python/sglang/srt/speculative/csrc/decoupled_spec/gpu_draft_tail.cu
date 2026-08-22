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

__global__ void publish_gpu_draft_tail_kernel(
    const int64_t* __restrict__ rows,
    int64_t num_rows,
    int64_t tail_capacity,
    int64_t* __restrict__ versions,
    int64_t* __restrict__ publish_seqs,
    int64_t* __restrict__ request_epochs,
    int64_t* __restrict__ prompt_lens,
    int64_t* __restrict__ committed_lens,
    int64_t* __restrict__ raw_tail_lens,
    int64_t* __restrict__ consumable_tail_lens,
    int64_t* __restrict__ tail_tokens) {
  const int64_t row_index = static_cast<int64_t>(blockIdx.x);
  if (row_index >= num_rows) return;
  const int64_t row_width = kGpuDraftTailPublishMetadataWidth + tail_capacity;
  const int64_t* row = rows + row_index * row_width;
  const int64_t seat = row[0];
  const int64_t publish_seq = row[1];
  __shared__ bool should_publish;

  // A single landing stream normally gives strict order. This guard also
  // makes a delayed staging record harmless if the transport changes later.
  if (threadIdx.x == 0) {
    should_publish = publish_seq > atomic_load_relaxed(publish_seqs + seat);
  }
  __syncthreads();
  if (!should_publish) return;

  if (threadIdx.x == 0) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(
        versions[seat]);
    version.fetch_add(int64_t{1}, cuda::memory_order_acq_rel);
  }
  __syncthreads();

  for (int64_t token_index = threadIdx.x; token_index < tail_capacity;
       token_index += blockDim.x) {
    atomic_store_relaxed(
        tail_tokens + seat * tail_capacity + token_index,
        row[kGpuDraftTailPublishMetadataWidth + token_index]);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomic_store_relaxed(request_epochs + seat, row[2]);
    atomic_store_relaxed(prompt_lens + seat, row[3]);
    atomic_store_relaxed(committed_lens + seat, row[4]);
    atomic_store_relaxed(raw_tail_lens + seat, row[5]);
    atomic_store_relaxed(consumable_tail_lens + seat, row[6]);
    atomic_store_relaxed(publish_seqs + seat, publish_seq);
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(
        versions[seat]);
    version.fetch_add(int64_t{1}, cuda::memory_order_release);
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
    int64_t batch_size,
    int64_t num_seats,
    int64_t num_draft_tokens,
    int64_t tail_capacity,
    const int64_t* __restrict__ versions,
    const int64_t* __restrict__ request_epochs,
    const int64_t* __restrict__ active_request_epochs,
    const int64_t* __restrict__ prompt_lens,
    const int64_t* __restrict__ committed_lens,
    const int64_t* __restrict__ raw_tail_lens,
    const int64_t* __restrict__ consumable_tail_lens,
    const int64_t* __restrict__ tail_tokens) {
  const int64_t batch_index = static_cast<int64_t>(blockIdx.x);
  if (batch_index >= batch_size || threadIdx.x != 0) return;

  const int64_t out_width = num_draft_tokens + 2;
  int64_t* out = compact_out + batch_index * out_width;
  for (int64_t i = 0; i < num_draft_tokens; ++i) out[i] = 0;
  out[num_draft_tokens] = 0;
  out[num_draft_tokens + 1] = 0;
  if (logical_committed_lens_out != nullptr) {
    logical_committed_lens_out[batch_index] = -1;
  }

  const int64_t seat = gpu_seats[batch_index];
  if (seat < 0 || seat >= num_seats) return;

  cuda::atomic_ref<int64_t, cuda::thread_scope_device> version(
      *const_cast<int64_t*>(versions + seat));
  const int64_t version_before =
      version.load(cuda::memory_order_acquire);
  if ((version_before & int64_t{1}) != 0) return;

  const int64_t request_epoch = atomic_load_relaxed(request_epochs + seat);
  const int64_t prompt_len = atomic_load_relaxed(prompt_lens + seat);
  const int64_t committed_len = atomic_load_relaxed(committed_lens + seat);
  const int64_t raw_tail_len = atomic_load_relaxed(raw_tail_lens + seat);
  const int64_t consumable_tail_len =
      atomic_load_relaxed(consumable_tail_lens + seat);
  const int64_t expected_epoch = expected_request_epochs == nullptr
      ? atomic_load_relaxed(active_request_epochs + seat)
      : expected_request_epochs[batch_index];
  const int64_t logical_output_len = seq_lens[batch_index] - prompt_len + 1;
  const bool identity_valid =
      request_epoch == expected_epoch && prompt_len >= 0;
  if (logical_committed_lens_out != nullptr && identity_valid) {
    logical_committed_lens_out[batch_index] = logical_output_len;
  }
  const int64_t delta = logical_output_len - committed_len;

  const int64_t bonus_token = bonus_tokens_are_int32
      ? static_cast<int64_t>(bonus_tokens_int32[batch_index])
      : bonus_tokens_int64[batch_index];
  bool row_valid = identity_valid && committed_len >= 0 && raw_tail_len >= 0 &&
      consumable_tail_len >= 0 && consumable_tail_len <= raw_tail_len &&
      raw_tail_len <= tail_capacity;
  int64_t offset = 0;
  if (row_valid && delta == 0) {
    offset = 0;
  } else if (
      row_valid && delta > 0 && delta <= num_draft_tokens + 1 &&
      delta <= consumable_tail_len &&
      atomic_load_relaxed(
          tail_tokens + seat * tail_capacity + delta - 1) == bonus_token) {
    offset = delta;
  } else {
    row_valid = false;
  }

  int64_t selected_len = 0;
  if (row_valid) {
    selected_len = consumable_tail_len - offset;
    if (selected_len < 0) selected_len = 0;
    if (selected_len > num_draft_tokens) selected_len = num_draft_tokens;
    for (int64_t i = 0; i < selected_len; ++i) {
      out[i] = atomic_load_relaxed(
          tail_tokens + seat * tail_capacity + offset + i);
    }
  }

  const int64_t version_after =
      version.load(cuda::memory_order_acquire);
  if (version_before != version_after ||
      (version_after & int64_t{1}) != 0) {
    for (int64_t i = 0; i < num_draft_tokens; ++i) out[i] = 0;
    selected_len = 0;
    row_valid = false;
    if (logical_committed_lens_out != nullptr) {
      logical_committed_lens_out[batch_index] = -1;
    }
  }
  out[num_draft_tokens] = selected_len;
  out[num_draft_tokens + 1] = row_valid ? 1 : 0;
}

}  // namespace

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
    void* stream) {
  if (num_rows <= 0) return;
  publish_gpu_draft_tail_kernel<<<
      static_cast<unsigned int>(num_rows),
      32,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      publish_rows,
      num_rows,
      tail_capacity,
      versions,
      publish_seqs,
      request_epochs,
      prompt_lens,
      committed_lens,
      raw_tail_lens,
      consumable_tail_lens,
      tail_tokens);
  check_cuda(cudaGetLastError(), "publish_gpu_draft_tail_kernel");
}

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
    void* stream) {
  if (batch_size <= 0) return;
  select_gpu_draft_tail_kernel<<<
      static_cast<unsigned int>(batch_size),
      1,
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
      batch_size,
      num_seats,
      num_draft_tokens,
      tail_capacity,
      versions,
      request_epochs,
      active_request_epochs,
      prompt_lens,
      committed_lens,
      raw_tail_lens,
      consumable_tail_lens,
      tail_tokens);
  check_cuda(cudaGetLastError(), "select_gpu_draft_tail_kernel");
}

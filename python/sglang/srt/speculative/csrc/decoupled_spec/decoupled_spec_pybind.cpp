#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ATen/record_function.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "gpu_draft_tail.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace py = pybind11;

class NvtxScopedRange {
 public:
  explicit NvtxScopedRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxScopedRange() { nvtxRangePop(); }

  NvtxScopedRange(const NvtxScopedRange&) = delete;
  NvtxScopedRange& operator=(const NvtxScopedRange&) = delete;
};

void set_current_thread_name(const char* short_name, const char* nvtx_name) {
  // Linux limits thread names to 15 visible bytes plus the terminator.
  (void)pthread_setname_np(pthread_self(), short_name);
  nvtxNameOsThreadA(
      static_cast<uint32_t>(syscall(SYS_gettid)), nvtx_name);
}

constexpr char kMagic[] = {'D', 'S', 'C', '1'};
constexpr uint8_t kVersion = 5;
constexpr uint8_t kKindControlBatch = 1;
constexpr uint8_t kKindTailStreamBatch = 2;
constexpr uint8_t kKindClockCalibrationProbe = 3;
constexpr uint8_t kKindClockCalibrationReply = 4;
constexpr size_t kWireFrameMetadataOffset = 6;
constexpr size_t kWireFrameMetadataSize = 4 * sizeof(int64_t);
constexpr size_t kWireFrameHeaderSize =
    kWireFrameMetadataOffset + kWireFrameMetadataSize;

constexpr int kZmqPull = 7;
constexpr int kZmqPush = 8;
constexpr int kZmqDONTWAIT = 1;
constexpr short kZmqPOLLIN = 1;
constexpr short kZmqPOLLOUT = 2;
constexpr int kZmqLINGER = 17;
constexpr int kZmqSNDHWM = 23;
constexpr int kZmqRCVHWM = 24;
constexpr int kZmqSNDBUF = 11;
constexpr int kZmqRCVBUF = 12;
constexpr int kZmqIMMEDIATE = 39;
constexpr int kZmqIPV6 = 42;
constexpr int kErrAgain = EAGAIN;
constexpr long kZmqSendPollSliceMs = 1;
constexpr int kZmqMessageHighWaterMark = 8192;
constexpr size_t kMaxPendingZmqFrames = 8192;
constexpr size_t kMaxPendingZmqBytes = 512ULL * 1024ULL * 1024ULL;
constexpr size_t kMaxZmqMessageBytes = 64ULL * 1024ULL * 1024ULL;
constexpr int64_t kVerifierSnapshotTensorMagic = 0x53474c534e415001LL;
constexpr size_t kVerifierSnapshotTensorMetadataWidth = 7;

uint64_t stable_request_id_hash(const std::string& value, uint64_t seed) {
  uint64_t hash = 1469598103934665603ULL ^ seed;
  for (unsigned char byte : value) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ULL;
  }
  hash ^= static_cast<uint64_t>(value.size());
  hash *= 1099511628211ULL;
  return hash;
}

int64_t uint64_as_int64(uint64_t value) {
  int64_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct WireFrameMetadataCpp {
  int64_t frame_seq = 0;
  int64_t result_ready_ns = 0;
  int64_t send_start_ns = 0;
  int64_t calibration_epoch = 0;
};

constexpr std::array<int64_t, 17> kLatencyBucketUpperBoundsUs = {
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    250000,
    500000,
    1000000};

struct LatencyHistogramSnapshotCpp {
  uint64_t count = 0;
  uint64_t sum_us = 0;
  std::array<uint64_t, kLatencyBucketUpperBoundsUs.size() + 1>
      bucket_counts{};

  void merge(const LatencyHistogramSnapshotCpp& other) {
    count += other.count;
    sum_us += other.sum_us;
    for (size_t index = 0; index < bucket_counts.size(); ++index) {
      bucket_counts[index] += other.bucket_counts[index];
    }
  }

  py::dict to_python() const {
    py::dict out;
    out["count"] = count;
    out["sum_us"] = sum_us;
    py::list bounds;
    for (int64_t bound : kLatencyBucketUpperBoundsUs) bounds.append(bound);
    py::list counts;
    for (uint64_t bucket_count : bucket_counts) counts.append(bucket_count);
    out["bucket_upper_bounds_us"] = std::move(bounds);
    out["bucket_counts"] = std::move(counts);
    return out;
  }
};

class FixedLatencyHistogram {
 public:
  void record_ns(int64_t latency_ns) {
    if (latency_ns < 0) return;
    const uint64_t latency_us = static_cast<uint64_t>(latency_ns / 1000);
    size_t bucket = 0;
    while (bucket < kLatencyBucketUpperBoundsUs.size() &&
           latency_us >
               static_cast<uint64_t>(kLatencyBucketUpperBoundsUs[bucket])) {
      ++bucket;
    }
    std::lock_guard<std::mutex> guard(mu_);
    ++count_;
    sum_us_ += latency_us;
    ++bucket_counts_[bucket];
  }

  py::dict take() {
    return take_snapshot().to_python();
  }

  LatencyHistogramSnapshotCpp take_snapshot() {
    LatencyHistogramSnapshotCpp snapshot;
    {
      std::lock_guard<std::mutex> guard(mu_);
      snapshot.count = std::exchange(count_, 0);
      snapshot.sum_us = std::exchange(sum_us_, 0);
      snapshot.bucket_counts = bucket_counts_;
      bucket_counts_.fill(0);
    }
    return snapshot;
  }

 private:
  std::mutex mu_;
  uint64_t count_ = 0;
  uint64_t sum_us_ = 0;
  std::array<uint64_t, kLatencyBucketUpperBoundsUs.size() + 1>
      bucket_counts_{};
};

struct PeerTransportLatencyCpp {
  FixedLatencyHistogram one_way_latency;
  FixedLatencyHistogram result_ready_to_receive_latency;
  // Both histograms accept samples only while the peer epoch is calibrated.
  // Retain the worst bound associated with those recorded samples so a later
  // recalibration cannot make their provenance look better than it was.
  int64_t max_recorded_error_bound_ns = 0;
};

void update_atomic_max(std::atomic<uint64_t>& maximum, uint64_t value) {
  uint64_t previous = maximum.load(std::memory_order_relaxed);
  while (previous < value &&
         !maximum.compare_exchange_weak(
             previous,
             value,
             std::memory_order_relaxed,
             std::memory_order_relaxed)) {
  }
}

constexpr int64_t kClockCalibrationSampleCount = 8;
constexpr int64_t kClockCalibrationPeriodNs = 30LL * 1000 * 1000 * 1000;
constexpr int64_t kClockCalibrationStaleNs = 60LL * 1000 * 1000 * 1000;
// One-way latency is withheld when the NTP-style minimum-RTT sample cannot
// bound clock error to one millisecond.
constexpr int64_t kClockMaxErrorBoundNs = 1000LL * 1000;

struct ClockCalibrationStateCpp {
  int64_t epoch = 0;
  int64_t num_samples = 0;
  int64_t best_rtt_ns = std::numeric_limits<int64_t>::max();
  int64_t offset_drafter_minus_verifier_ns = 0;
  int64_t error_bound_ns = 0;
  int64_t last_update_ns = 0;
  bool valid = false;
};

struct PendingClockProbeCpp {
  int32_t drafter_rank = -1;
  int64_t epoch = 0;
  int64_t verifier_send_ns = 0;
};

class BinaryReader {
 public:
  explicit BinaryReader(const std::string& bytes) : data_(bytes), pos_(0) {
    if (data_.size() < 6 || std::memcmp(data_.data(), kMagic, 4) != 0) {
      throw std::runtime_error("Invalid decoupled-spec frame magic");
    }
    pos_ = 4;
    uint8_t version = read_u8();
    if (version != kVersion) {
      throw std::runtime_error("Unsupported decoupled-spec frame version");
    }
    kind_ = read_u8();
    metadata_.frame_seq = read_i64();
    metadata_.result_ready_ns = read_i64();
    metadata_.send_start_ns = read_i64();
    metadata_.calibration_epoch = read_i64();
  }

  uint8_t kind() const { return kind_; }
  const WireFrameMetadataCpp& metadata() const { return metadata_; }

  void expect_kind(uint8_t expected) const {
    if (kind_ != expected) {
      throw std::runtime_error("Unexpected decoupled-spec frame kind");
    }
  }

  uint8_t read_u8() {
    ensure(1);
    return static_cast<uint8_t>(data_[pos_++]);
  }

  uint32_t read_u32() {
    ensure(4);
    uint32_t value = 0;
    std::memcpy(&value, data_.data() + pos_, 4);
    pos_ += 4;
    return value;
  }

  int32_t read_i32() { return static_cast<int32_t>(read_u32()); }

  int64_t read_i64() {
    ensure(8);
    int64_t value = 0;
    std::memcpy(&value, data_.data() + pos_, 8);
    pos_ += 8;
    return value;
  }

  std::string read_string() {
    uint32_t n = read_u32();
    ensure(n);
    std::string value(data_.data() + pos_, data_.data() + pos_ + n);
    pos_ += n;
    return value;
  }

  std::vector<int32_t> read_int_list() {
    uint32_t n = read_u32();
    std::vector<int32_t> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) out.push_back(read_i32());
    return out;
  }

  void finish() const {
    if (pos_ != data_.size()) {
      throw std::runtime_error("Trailing bytes in decoupled-spec frame");
    }
  }

 private:
  void ensure(size_t n) const {
    if (pos_ + n > data_.size()) {
      throw std::runtime_error("Truncated decoupled-spec frame");
    }
  }

  const std::string& data_;
  size_t pos_;
  uint8_t kind_;
  WireFrameMetadataCpp metadata_;
};

class BinaryWriter {
 public:
  explicit BinaryWriter(
      uint8_t kind,
      const WireFrameMetadataCpp& metadata = {}) {
    data_.append(kMagic, sizeof(kMagic));
    write_u8(kVersion);
    write_u8(kind);
    write_i64(metadata.frame_seq);
    write_i64(metadata.result_ready_ns);
    write_i64(metadata.send_start_ns);
    write_i64(metadata.calibration_epoch);
  }

  void write_u8(uint8_t value) { data_.push_back(static_cast<char>(value)); }

  void write_u32(uint32_t value) {
    const char* ptr = reinterpret_cast<const char*>(&value);
    data_.append(ptr, ptr + sizeof(value));
  }

  void write_i32(int32_t value) { write_u32(static_cast<uint32_t>(value)); }

  void write_i64(int64_t value) {
    const char* ptr = reinterpret_cast<const char*>(&value);
    data_.append(ptr, ptr + sizeof(value));
  }

  void write_string(const std::string& value) {
    write_u32(static_cast<uint32_t>(value.size()));
    data_.append(value);
  }

  const std::string& data() const { return data_; }

 private:
  std::string data_;
};

uint8_t wire_frame_kind(const std::string& frame) {
  BinaryReader reader(frame);
  return reader.kind();
}

void patch_wire_send_start_ns(std::string& frame, int64_t send_start_ns) {
  if (frame.size() < kWireFrameHeaderSize ||
      std::memcmp(frame.data(), kMagic, sizeof(kMagic)) != 0 ||
      static_cast<uint8_t>(frame[4]) != kVersion) {
    throw std::runtime_error(
        "Cannot stamp an invalid decoupled-spec network frame");
  }
  std::memcpy(
      frame.data() + kWireFrameMetadataOffset + 2 * sizeof(int64_t),
      &send_start_ns,
      sizeof(send_start_ns));
}

struct DraftReqKeyCpp {
  int32_t src_verifier_rank = 0;
  std::string request_id;

  std::string map_key() const {
    return std::to_string(src_verifier_rank) + "\n" + request_id;
  }
};

struct DraftSyncCpp {
  std::string request_id;
  int32_t src_verifier_rank = 0;
  int32_t dst_drafter_rank = 0;
  int64_t max_new_tokens = 0;
  std::vector<int32_t> prompt_token_ids;
  std::vector<int32_t> committed_outputs;

  DraftReqKeyCpp draft_key() const { return {src_verifier_rank, request_id}; }
};

struct DraftSyncOpenCpp {
  std::string request_id;
  int32_t dst_drafter_rank = 0;
  int64_t prompt_len = 0;
  int64_t committed_len = 0;
};

struct VerifyCommitCpp {
  std::string request_id;
  int32_t src_verifier_rank = 0;
  int32_t dst_drafter_rank = 0;
  int64_t pre_verify_committed_len = 0;
  std::vector<int32_t> committed_tokens;

  DraftReqKeyCpp draft_key() const { return {src_verifier_rank, request_id}; }

  void validate() const {
    if (committed_tokens.empty()) {
      throw std::runtime_error("VerifyCommit committed_tokens must be non-empty");
    }
    if (pre_verify_committed_len < 0) {
      throw std::runtime_error("VerifyCommit pre_verify_committed_len must be non-negative");
    }
  }
};

struct DraftCloseCpp {
  std::string request_id;
  int32_t src_verifier_rank = 0;
  int32_t dst_drafter_rank = 0;
  std::string reason;

  DraftReqKeyCpp draft_key() const { return {src_verifier_rank, request_id}; }
};

struct DraftTailStreamOutputCpp {
  int32_t src_drafter_rank = 0;
  int32_t dst_verifier_rank = 0;
  std::string request_id;
  int64_t base_committed_len = 0;
  int64_t start_token_pos = 0;
  std::vector<int32_t> tokens;
  bool is_commit_echo = false;

  void validate() const {
    if (base_committed_len < 0 || start_token_pos < 0) {
      throw std::runtime_error(
          "DraftTailStreamOutput positions must be non-negative");
    }
    if (tokens.empty()) {
      throw std::runtime_error(
          "DraftTailStreamOutput token span must be non-empty");
    }
    if (is_commit_echo) {
      if (tokens.size() != 1 ||
          start_token_pos == std::numeric_limits<int64_t>::max() ||
          base_committed_len != start_token_pos + 1) {
        throw std::runtime_error(
            "DraftTailStreamOutput commit echo must be a singleton "
            "cumulative ACK");
      }
    } else if (start_token_pos < base_committed_len) {
      throw std::runtime_error(
          "DraftTailStreamOutput append span starts before its committed base");
    }
    if (tokens.size() > static_cast<size_t>(
                            std::numeric_limits<int64_t>::max() -
                            start_token_pos)) {
      throw std::runtime_error(
          "DraftTailStreamOutput token span position overflows int64");
    }
  }
};

struct DraftControlBatchCpp {
  WireFrameMetadataCpp wire_metadata;
  int32_t dst_drafter_rank = 0;
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<VerifyCommitCpp> verify_commit_messages;
  std::vector<DraftCloseCpp> close_messages;
};

struct DraftTailStreamOutputBatchCpp {
  WireFrameMetadataCpp wire_metadata;
  std::vector<DraftTailStreamOutputCpp> outputs;
};

struct ClockCalibrationProbeCpp {
  int32_t src_verifier_rank = -1;
  int32_t dst_drafter_rank = -1;
  int64_t probe_seq = 0;
  int64_t verifier_send_ns = 0;
  int64_t calibration_epoch = 0;
};

struct ClockCalibrationReplyCpp {
  int32_t src_drafter_rank = -1;
  int32_t dst_verifier_rank = -1;
  int64_t probe_seq = 0;
  int64_t verifier_send_ns = 0;
  int64_t drafter_receive_ns = 0;
  int64_t drafter_send_ns = 0;
  int64_t calibration_epoch = 0;
};

DraftControlBatchCpp parse_control_batch(const std::string& frame) {
  BinaryReader reader(frame);
  reader.expect_kind(kKindControlBatch);
  DraftControlBatchCpp batch;
  batch.wire_metadata = reader.metadata();
  batch.dst_drafter_rank = reader.read_i32();
  uint32_t sync_count = reader.read_u32();
  batch.sync_messages.reserve(sync_count);
  for (uint32_t i = 0; i < sync_count; ++i) {
    DraftSyncCpp msg;
    msg.request_id = reader.read_string();
    msg.src_verifier_rank = reader.read_i32();
    msg.dst_drafter_rank = reader.read_i32();
    msg.max_new_tokens = reader.read_i64();
    msg.prompt_token_ids = reader.read_int_list();
    msg.committed_outputs = reader.read_int_list();
    batch.sync_messages.push_back(std::move(msg));
  }
  uint32_t commit_count = reader.read_u32();
  batch.verify_commit_messages.reserve(commit_count);
  for (uint32_t i = 0; i < commit_count; ++i) {
    VerifyCommitCpp msg;
    msg.request_id = reader.read_string();
    msg.src_verifier_rank = reader.read_i32();
    msg.dst_drafter_rank = reader.read_i32();
    msg.pre_verify_committed_len = reader.read_i64();
    msg.committed_tokens = reader.read_int_list();
    batch.verify_commit_messages.push_back(std::move(msg));
  }
  uint32_t close_count = reader.read_u32();
  batch.close_messages.reserve(close_count);
  for (uint32_t i = 0; i < close_count; ++i) {
    DraftCloseCpp msg;
    msg.request_id = reader.read_string();
    msg.src_verifier_rank = reader.read_i32();
    msg.dst_drafter_rank = reader.read_i32();
    msg.reason = reader.read_string();
    batch.close_messages.push_back(std::move(msg));
  }
  reader.finish();
  return batch;
}

DraftTailStreamOutputBatchCpp parse_tail_stream_batch(const std::string& frame) {
  BinaryReader reader(frame);
  reader.expect_kind(kKindTailStreamBatch);
  DraftTailStreamOutputBatchCpp batch;
  batch.wire_metadata = reader.metadata();
  uint32_t n = reader.read_u32();
  batch.outputs.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    DraftTailStreamOutputCpp output;
    output.src_drafter_rank = reader.read_i32();
    output.dst_verifier_rank = reader.read_i32();
    output.request_id = reader.read_string();
    output.base_committed_len = reader.read_i64();
    output.start_token_pos = reader.read_i64();
    output.tokens = reader.read_int_list();
    output.is_commit_echo = reader.read_u8() != 0;
    output.validate();
    batch.outputs.push_back(std::move(output));
  }
  reader.finish();
  return batch;
}

std::string encode_tail_stream_batch(const DraftTailStreamOutputBatchCpp& batch) {
  BinaryWriter writer(kKindTailStreamBatch, batch.wire_metadata);
  writer.write_u32(static_cast<uint32_t>(batch.outputs.size()));
  for (const auto& output : batch.outputs) {
    output.validate();
    if (output.tokens.size() > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(
          "DraftTailStreamOutput token span exceeds uint32 wire length");
    }
    writer.write_i32(output.src_drafter_rank);
    writer.write_i32(output.dst_verifier_rank);
    writer.write_string(output.request_id);
    writer.write_i64(output.base_committed_len);
    writer.write_i64(output.start_token_pos);
    writer.write_u32(static_cast<uint32_t>(output.tokens.size()));
    for (int32_t token : output.tokens) writer.write_i32(token);
    writer.write_u8(output.is_commit_echo ? 1 : 0);
  }
  return writer.data();
}

std::string encode_control_batch(const DraftControlBatchCpp& batch) {
  BinaryWriter writer(kKindControlBatch, batch.wire_metadata);
  writer.write_i32(batch.dst_drafter_rank);
  writer.write_u32(static_cast<uint32_t>(batch.sync_messages.size()));
  for (const auto& msg : batch.sync_messages) {
    writer.write_string(msg.request_id);
    writer.write_i32(msg.src_verifier_rank);
    writer.write_i32(msg.dst_drafter_rank);
    writer.write_i64(msg.max_new_tokens);
    writer.write_u32(static_cast<uint32_t>(msg.prompt_token_ids.size()));
    for (int32_t token : msg.prompt_token_ids) writer.write_i32(token);
    writer.write_u32(static_cast<uint32_t>(msg.committed_outputs.size()));
    for (int32_t token : msg.committed_outputs) writer.write_i32(token);
  }
  writer.write_u32(static_cast<uint32_t>(batch.verify_commit_messages.size()));
  for (const auto& msg : batch.verify_commit_messages) {
    writer.write_string(msg.request_id);
    writer.write_i32(msg.src_verifier_rank);
    writer.write_i32(msg.dst_drafter_rank);
    writer.write_i64(msg.pre_verify_committed_len);
    writer.write_u32(static_cast<uint32_t>(msg.committed_tokens.size()));
    for (int32_t token : msg.committed_tokens) writer.write_i32(token);
  }
  writer.write_u32(static_cast<uint32_t>(batch.close_messages.size()));
  for (const auto& msg : batch.close_messages) {
    writer.write_string(msg.request_id);
    writer.write_i32(msg.src_verifier_rank);
    writer.write_i32(msg.dst_drafter_rank);
    writer.write_string(msg.reason);
  }
  return writer.data();
}

std::string encode_clock_calibration_probe(
    const ClockCalibrationProbeCpp& probe) {
  WireFrameMetadataCpp metadata;
  metadata.frame_seq = probe.probe_seq;
  metadata.calibration_epoch = probe.calibration_epoch;
  BinaryWriter writer(kKindClockCalibrationProbe, metadata);
  writer.write_i32(probe.src_verifier_rank);
  writer.write_i32(probe.dst_drafter_rank);
  writer.write_i64(probe.probe_seq);
  writer.write_i64(probe.verifier_send_ns);
  writer.write_i64(probe.calibration_epoch);
  return writer.data();
}

ClockCalibrationProbeCpp parse_clock_calibration_probe(
    const std::string& frame) {
  BinaryReader reader(frame);
  reader.expect_kind(kKindClockCalibrationProbe);
  ClockCalibrationProbeCpp probe;
  probe.src_verifier_rank = reader.read_i32();
  probe.dst_drafter_rank = reader.read_i32();
  probe.probe_seq = reader.read_i64();
  probe.verifier_send_ns = reader.read_i64();
  probe.calibration_epoch = reader.read_i64();
  reader.finish();
  return probe;
}

std::string encode_clock_calibration_reply(
    const ClockCalibrationReplyCpp& reply) {
  WireFrameMetadataCpp metadata;
  metadata.frame_seq = reply.probe_seq;
  metadata.send_start_ns = reply.drafter_send_ns;
  metadata.calibration_epoch = reply.calibration_epoch;
  BinaryWriter writer(kKindClockCalibrationReply, metadata);
  writer.write_i32(reply.src_drafter_rank);
  writer.write_i32(reply.dst_verifier_rank);
  writer.write_i64(reply.probe_seq);
  writer.write_i64(reply.verifier_send_ns);
  writer.write_i64(reply.drafter_receive_ns);
  writer.write_i64(reply.drafter_send_ns);
  writer.write_i64(reply.calibration_epoch);
  return writer.data();
}

ClockCalibrationReplyCpp parse_clock_calibration_reply(
    const std::string& frame) {
  BinaryReader reader(frame);
  reader.expect_kind(kKindClockCalibrationReply);
  ClockCalibrationReplyCpp reply;
  reply.src_drafter_rank = reader.read_i32();
  reply.dst_verifier_rank = reader.read_i32();
  reply.probe_seq = reader.read_i64();
  reply.verifier_send_ns = reader.read_i64();
  reply.drafter_receive_ns = reader.read_i64();
  reply.drafter_send_ns = reader.read_i64();
  reply.calibration_epoch = reader.read_i64();
  reader.finish();
  return reply;
}

py::object sequence_fast(const py::handle& value, const char* message) {
  PyObject* seq = PySequence_Fast(value.ptr(), message);
  if (seq == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(seq);
}

int32_t py_int32_from_object(PyObject* item) {
  long long value = PyLong_AsLongLong(item);
  if (value == -1 && PyErr_Occurred()) throw py::error_already_set();
  return static_cast<int32_t>(value);
}

std::vector<int32_t> py_int32_vector(const py::handle& value) {
  py::object tokens = sequence_fast(value, "Expected an integer token sequence");
  Py_ssize_t n = PySequence_Fast_GET_SIZE(tokens.ptr());
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(n));
  PyObject** items = PySequence_Fast_ITEMS(tokens.ptr());
  for (Py_ssize_t i = 0; i < n; ++i) {
    out.push_back(py_int32_from_object(items[i]));
  }
  return out;
}

std::vector<std::string> py_string_vector(const py::handle& value) {
  return py::cast<std::vector<std::string>>(value);
}

std::vector<int64_t> py_int64_vector(const py::handle& value) {
  return py::cast<std::vector<int64_t>>(value);
}

struct RankedPeerEndpoint {
  int32_t rank;
  std::string endpoint;
};

int32_t checked_transport_rank(int64_t rank, const char* field_name) {
  if (rank < 0 || rank > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(
        std::string(field_name) + " must fit in a non-negative int32");
  }
  return static_cast<int32_t>(rank);
}

std::vector<RankedPeerEndpoint> build_ranked_peer_endpoints(
    const py::sequence& peer_rows) {
  if (py::len(peer_rows) == 0) {
    throw std::runtime_error("Decoupled-spec transport requires at least one peer");
  }
  std::vector<RankedPeerEndpoint> peers;
  peers.reserve(py::len(peer_rows));
  std::map<int32_t, bool> seen_ranks;
  for (const auto& item : peer_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 2) {
      throw std::runtime_error("Decoupled-spec peer row must have rank and endpoint");
    }
    int64_t rank = row[0].cast<int64_t>();
    auto rank32 = checked_transport_rank(rank, "Decoupled-spec peer rank");
    auto endpoint = row[1].cast<std::string>();
    if (endpoint.rfind("tcp://", 0) != 0 &&
        endpoint.rfind("ipc://", 0) != 0 &&
        endpoint.rfind("inproc://", 0) != 0) {
      throw std::runtime_error(
          "Decoupled-spec peer endpoint must be a TCP, IPC, or inproc endpoint");
    }
    if (!seen_ranks.emplace(rank32, true).second) {
      throw std::runtime_error("Decoupled-spec peer ranks must be unique");
    }
    peers.push_back(RankedPeerEndpoint{rank32, std::move(endpoint)});
  }
  return peers;
}

std::vector<DraftSyncOpenCpp> build_sync_open_rows_native(
    const py::sequence& sync_rows) {
  std::vector<DraftSyncOpenCpp> rows;
  rows.reserve(py::len(sync_rows));
  for (const auto& item : sync_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 6) {
      throw std::runtime_error("DraftSync native row must have 6 fields");
    }
    DraftSyncOpenCpp open;
    open.request_id = row[0].cast<std::string>();
    open.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    open.prompt_len = static_cast<int64_t>(py::len(row[4]));
    open.committed_len = static_cast<int64_t>(py::len(row[5]));
    rows.push_back(std::move(open));
  }
  return rows;
}

DraftControlBatchCpp build_control_batch_native(
    int64_t dst_drafter_rank,
    const py::sequence& sync_rows,
    const py::sequence& commit_rows,
    const py::sequence& close_rows) {
  DraftControlBatchCpp batch;
  batch.dst_drafter_rank = static_cast<int32_t>(dst_drafter_rank);

  batch.sync_messages.reserve(py::len(sync_rows));
  for (const auto& item : sync_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 6) {
      throw std::runtime_error("DraftSync native row must have 6 fields");
    }
    DraftSyncCpp msg;
    msg.request_id = row[0].cast<std::string>();
    msg.src_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    msg.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    msg.max_new_tokens = row[3].cast<int64_t>();
    msg.prompt_token_ids = py_int32_vector(row[4]);
    msg.committed_outputs = py_int32_vector(row[5]);
    batch.sync_messages.push_back(std::move(msg));
  }

  batch.verify_commit_messages.reserve(py::len(commit_rows));
  for (const auto& item : commit_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 5) {
      throw std::runtime_error("VerifyCommit native row must have 5 fields");
    }
    VerifyCommitCpp msg;
    msg.request_id = row[0].cast<std::string>();
    msg.src_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    msg.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    msg.pre_verify_committed_len = row[3].cast<int64_t>();
    msg.committed_tokens = py_int32_vector(row[4]);
    batch.verify_commit_messages.push_back(std::move(msg));
  }

  batch.close_messages.reserve(py::len(close_rows));
  for (const auto& item : close_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 4) {
      throw std::runtime_error("DraftClose native row must have 4 fields");
    }
    DraftCloseCpp msg;
    msg.request_id = row[0].cast<std::string>();
    msg.src_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    msg.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    msg.reason = row[3].cast<std::string>();
    batch.close_messages.push_back(std::move(msg));
  }
  return batch;
}

DraftTailStreamOutputBatchCpp build_tail_stream_batch_native(
    const py::sequence& rows) {
  DraftTailStreamOutputBatchCpp batch;
  batch.outputs.reserve(py::len(rows));
  for (const auto& item : rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 7) {
      throw std::runtime_error("DraftTailStreamOutput native row must have 7 fields");
    }
    DraftTailStreamOutputCpp output;
    output.src_drafter_rank = static_cast<int32_t>(row[0].cast<int64_t>());
    output.dst_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    output.request_id = row[2].cast<std::string>();
    output.base_committed_len = row[3].cast<int64_t>();
    output.start_token_pos = row[4].cast<int64_t>();
    output.tokens = row[5].cast<std::vector<int32_t>>();
    output.is_commit_echo = row[6].cast<bool>();
    output.validate();
    batch.outputs.push_back(std::move(output));
  }
  return batch;
}

struct ReadyDraftControlsCpp {
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<DraftReqKeyCpp> close_keys;
};

py::tuple draft_key_row(const DraftReqKeyCpp& key) {
  return py::make_tuple(key.request_id, key.src_verifier_rank);
}

py::tuple sync_row(const DraftSyncCpp& msg) {
  return py::make_tuple(
      msg.request_id,
      msg.src_verifier_rank,
      msg.dst_drafter_rank,
      msg.max_new_tokens,
      msg.prompt_token_ids,
      msg.committed_outputs);
}


py::tuple ready_controls_native_rows(const ReadyDraftControlsCpp& ready) {
  py::list sync_rows;
  py::list close_rows;
  for (const auto& msg : ready.sync_messages) sync_rows.append(sync_row(msg));
  for (const auto& key : ready.close_keys) close_rows.append(draft_key_row(key));
  return py::make_tuple(sync_rows, close_rows);
}

// Host lifecycle queue. Verifier commits are consumed by the GPU state machine;
// only OPEN/CLOSE and scheduling wakeups cross into the scheduler.
class DrafterDataPlaneCore {
 public:
  int64_t pending_control_count() {
    std::lock_guard<std::mutex> guard(mu_);
    return static_cast<int64_t>(sync_messages_.size() + close_keys_.size());
  }

  bool wait_for_pending_control(int64_t timeout_us) {
    if (timeout_us < 0) {
      throw std::runtime_error("Control wait timeout must be non-negative");
    }
    std::unique_lock<std::mutex> lock(mu_);
    auto ready = [&] {
      return !sync_messages_.empty() || !close_keys_.empty() || external_wake_pending_;
    };
    bool woke = ready() || control_cv_.wait_for(
        lock, std::chrono::microseconds(timeout_us), ready);
    if (woke && external_wake_pending_) external_wake_pending_ = false;
    return woke;
  }

  void notify_external_progress() {
    {
      std::lock_guard<std::mutex> guard(mu_);
      external_wake_pending_ = true;
    }
    control_cv_.notify_all();
  }

  void add_control_batch(const DraftControlBatchCpp& batch) {
    {
      std::lock_guard<std::mutex> guard(mu_);
      close_keys_.reserve(close_keys_.size() + batch.close_messages.size());
      sync_messages_.reserve(sync_messages_.size() + batch.sync_messages.size());
      for (const auto& msg : batch.close_messages) {
        auto key = msg.draft_key();
        close_keys_[key.map_key()] = key;
        sync_messages_.erase(
            std::remove_if(
                sync_messages_.begin(), sync_messages_.end(),
                [&](const DraftSyncCpp& sync) {
                  return sync.draft_key().map_key() == key.map_key();
                }),
            sync_messages_.end());
      }
      for (const auto& msg : batch.sync_messages) {
        if (close_keys_.count(msg.draft_key().map_key()) == 0) {
          sync_messages_.push_back(msg);
        }
      }
    }
    control_cv_.notify_all();
  }

  ReadyDraftControlsCpp extract_lifecycle_controls() {
    ReadyDraftControlsCpp ready;
    std::lock_guard<std::mutex> guard(mu_);
    for (const auto& kv : close_keys_) ready.close_keys.push_back(kv.second);
    close_keys_.clear();
    ready.sync_messages = std::move(sync_messages_);
    sync_messages_.clear();
    return ready;
  }

 private:
  std::mutex mu_;
  std::condition_variable control_cv_;
  std::vector<DraftSyncCpp> sync_messages_;
  std::unordered_map<std::string, DraftReqKeyCpp> close_keys_;
  bool external_wake_pending_ = false;
};

struct ZmqPollItem {
  void* socket;
  int fd;
  short events;
  short revents;
};

struct ZmqMsg {
  unsigned char storage[64];
};

class ZmqApi {
 public:
  using ctx_new_t = void* (*)();
  using ctx_term_t = int (*)(void*);
  using init_t = void* (*)(int);
  using term_t = int (*)(void*);
  using socket_t = void* (*)(void*, int);
  using close_t = int (*)(void*);
  using setsockopt_t = int (*)(void*, int, const void*, size_t);
  using bind_t = int (*)(void*, const char*);
  using connect_t = int (*)(void*, const char*);
  using send_t = int (*)(void*, const void*, size_t, int);
  using recv_t = int (*)(void*, void*, size_t, int);
  using msg_init_t = int (*)(ZmqMsg*);
  using msg_close_t = int (*)(ZmqMsg*);
  using msg_recv_t = int (*)(ZmqMsg*, void*, int);
  using msg_data_t = void* (*)(ZmqMsg*);
  using msg_size_t = size_t (*)(const ZmqMsg*);
  using poll_t = int (*)(ZmqPollItem*, int, long);
  using errno_t = int (*)();
  using strerror_t = const char* (*)(int);

  static ZmqApi& instance() {
    static ZmqApi api;
    return api;
  }

  void* ctx_new() {
    if (ctx_new_) return checked(ctx_new_(), "zmq_ctx_new");
    return checked(init_(1), "zmq_init");
  }
  void ctx_term(void* ctx) {
    if (!ctx) return;
    if (ctx_term_) {
      ctx_term_(ctx);
    } else {
      term_(ctx);
    }
  }

  void* socket(void* ctx, int type) { return checked(socket_(ctx, type), "zmq_socket"); }
  void close_socket(void* socket) {
    if (socket) close_(socket);
  }

  void set_int(void* socket, int option, int value) {
    if (setsockopt_(socket, option, &value, sizeof(value)) != 0) throw_last("zmq_setsockopt");
  }

  void try_set_int(void* socket, int option, int value) {
    setsockopt_(socket, option, &value, sizeof(value));
  }

  void configure_socket(void* socket, int type) {
    int linger = 0;
    try_set_int(socket, kZmqLINGER, linger);
    int hwm = kZmqMessageHighWaterMark;
    int buf_size = static_cast<int>(512 * 1024 * 1024);
    if (type == kZmqPush) {
      // Keep an unsent frame in the owner thread until a peer has completed
      // its connection handshake. This makes send success mean that libzmq
      // accepted the frame for a live pipe instead of a provisional one.
      set_int(socket, kZmqIMMEDIATE, 1);
      try_set_int(socket, kZmqSNDHWM, hwm);
      try_set_int(socket, kZmqSNDBUF, buf_size);
    } else if (type == kZmqPull) {
      try_set_int(socket, kZmqRCVHWM, hwm);
      try_set_int(socket, kZmqRCVBUF, buf_size);
    }
  }

  void bind(void* socket, const std::string& endpoint) {
    if (endpoint.find('[') != std::string::npos) {
      int one = 1;
      try_set_int(socket, kZmqIPV6, one);
    }
    if (bind_(socket, endpoint.c_str()) != 0) throw_last("zmq_bind");
  }

  void connect(void* socket, const std::string& endpoint) {
    if (endpoint.find('[') != std::string::npos) {
      int one = 1;
      try_set_int(socket, kZmqIPV6, one);
    }
    if (connect_(socket, endpoint.c_str()) != 0) throw_last("zmq_connect");
  }

  bool send_nonblock(void* socket, const std::string& frame) {
    int rc = send_(socket, frame.data(), frame.size(), kZmqDONTWAIT);
    if (rc >= 0) {
      if (static_cast<size_t>(rc) != frame.size()) {
        throw std::runtime_error("zmq_send returned a partial frame");
      }
      return true;
    }
    int err = errno_();
    if (err == kErrAgain) return false;
    throw std::runtime_error(std::string("zmq_send failed: ") + strerror_(err));
  }

  bool recv_nonblock(void* socket, std::string& out) {
    if (msg_init_ && msg_recv_ && msg_data_ && msg_size_ && msg_close_) {
      ZmqMsg msg;
      if (msg_init_(&msg) != 0) throw_last("zmq_msg_init");
      int rc = msg_recv_(&msg, socket, kZmqDONTWAIT);
      if (rc < 0) {
        int err = errno_();
        msg_close_(&msg);
        if (err == kErrAgain) return false;
        throw std::runtime_error(std::string("zmq_msg_recv failed: ") + strerror_(err));
      }
      void* data = msg_data_(&msg);
      size_t size = msg_size_(&msg);
      out.assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
      msg_close_(&msg);
      return true;
    }

    std::vector<char> buffer(kMaxZmqMessageBytes);
    int rc = recv_(socket, buffer.data(), buffer.size(), kZmqDONTWAIT);
    if (rc < 0) {
      int err = errno_();
      if (err == kErrAgain) return false;
      throw std::runtime_error(std::string("zmq_recv failed: ") + strerror_(err));
    }
    out.assign(buffer.data(), buffer.data() + rc);
    return true;
  }

  bool poll_in(void* socket, long timeout_ms) {
    ZmqPollItem item{socket, 0, kZmqPOLLIN, 0};
    int rc = poll_(&item, 1, timeout_ms);
    if (rc < 0) throw_last("zmq_poll");
    return rc > 0 && (item.revents & kZmqPOLLIN);
  }

  bool poll_out(void* socket, long timeout_ms) {
    ZmqPollItem item{socket, 0, kZmqPOLLOUT, 0};
    int rc = poll_(&item, 1, timeout_ms);
    if (rc < 0) throw_last("zmq_poll");
    return rc > 0 && (item.revents & kZmqPOLLOUT);
  }

 private:
  ZmqApi() {
    const char* env_path = std::getenv("SGLANG_DECOUPLED_SPEC_ZMQ_LIB");
    if (env_path && env_path[0]) handle_ = dlopen(env_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle_) handle_ = dlopen("libzmq.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle_) handle_ = dlopen("/path/to/ss_lib/so/libzmq.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      throw std::runtime_error(std::string("Failed to load libzmq: ") + dlerror());
    }
    load_optional(ctx_new_, "zmq_ctx_new");
    load_optional(ctx_term_, "zmq_ctx_term");
    load_optional(init_, "zmq_init");
    load_optional(term_, "zmq_term");
    if ((!ctx_new_ || !ctx_term_) && (!init_ || !term_)) {
      throw std::runtime_error("libzmq exposes neither ctx_new/ctx_term nor init/term");
    }
    load(socket_, "zmq_socket");
    load(close_, "zmq_close");
    load(setsockopt_, "zmq_setsockopt");
    load(bind_, "zmq_bind");
    load(connect_, "zmq_connect");
    load(send_, "zmq_send");
    load(recv_, "zmq_recv");
    load_optional(msg_init_, "zmq_msg_init");
    load_optional(msg_close_, "zmq_msg_close");
    load_optional(msg_recv_, "zmq_msg_recv");
    load_optional(msg_data_, "zmq_msg_data");
    load_optional(msg_size_, "zmq_msg_size");
    load(poll_, "zmq_poll");
    load(errno_, "zmq_errno");
    load(strerror_, "zmq_strerror");
  }

  template <typename Fn>
  void load(Fn& fn, const char* name) {
    void* ptr = dlsym(handle_, name);
    if (!ptr) throw std::runtime_error(std::string("Missing libzmq symbol: ") + name);
    fn = reinterpret_cast<Fn>(ptr);
  }

  template <typename Fn>
  void load_optional(Fn& fn, const char* name) {
    void* ptr = dlsym(handle_, name);
    fn = ptr ? reinterpret_cast<Fn>(ptr) : nullptr;
  }

  void* checked(void* value, const char* op) {
    if (!value) throw_last(op);
    return value;
  }

  [[noreturn]] void throw_last(const char* op) {
    int err = errno_();
    throw std::runtime_error(std::string(op) + " failed: " + strerror_(err));
  }

  void* handle_ = nullptr;
  ctx_new_t ctx_new_ = nullptr;
  ctx_term_t ctx_term_ = nullptr;
  init_t init_ = nullptr;
  term_t term_ = nullptr;
  socket_t socket_ = nullptr;
  close_t close_ = nullptr;
  setsockopt_t setsockopt_ = nullptr;
  bind_t bind_ = nullptr;
  connect_t connect_ = nullptr;
  send_t send_ = nullptr;
  recv_t recv_ = nullptr;
  msg_init_t msg_init_ = nullptr;
  msg_close_t msg_close_ = nullptr;
  msg_recv_t msg_recv_ = nullptr;
  msg_data_t msg_data_ = nullptr;
  msg_size_t msg_size_ = nullptr;
  poll_t poll_ = nullptr;
  errno_t errno_ = nullptr;
  strerror_t strerror_ = nullptr;
};

bool send_frame_until_ready_or_closed(
    ZmqApi& api,
    void* socket,
    const std::string& frame,
    const std::atomic<bool>& closed) {
  while (!closed.load(std::memory_order_acquire)) {
    if (api.send_nonblock(socket, frame)) return true;
    // This slice is only shutdown-cancellation granularity. There is no
    // delivery deadline: while the owner remains open, the same queue-front
    // frame is retried indefinitely and later frames cannot overtake it.
    api.poll_out(socket, kZmqSendPollSliceMs);
  }
  return false;
}

bool try_send_draft_frame(
    ZmqApi& api, void* socket, std::string& frame) {
  // A failed nonblocking attempt is local socket backpressure, not network
  // transit. Refresh the wire timestamp before every attempt so the accepted
  // frame carries the final attempt time and one-way latency does not double
  // count the send-queue wait.
  patch_wire_send_start_ns(frame, now_ns());
  return api.send_nonblock(socket, frame);
}

class ZmqContextOwner {
 public:
  explicit ZmqContextOwner(uintptr_t external_context = 0)
      : api_(ZmqApi::instance()),
        ctx_(external_context == 0
                 ? api_.ctx_new()
                 : reinterpret_cast<void*>(external_context)),
        owns_context_(external_context == 0) {}
  ~ZmqContextOwner() { close_context(); }
  ZmqContextOwner(const ZmqContextOwner&) = delete;
  ZmqContextOwner& operator=(const ZmqContextOwner&) = delete;

  ZmqApi& api() { return api_; }
  void* ctx() { return ctx_; }

  void close_context() {
    if (ctx_) {
      if (owns_context_) api_.ctx_term(ctx_);
      ctx_ = nullptr;
    }
  }

 private:
  ZmqApi& api_;
  void* ctx_;
  bool owns_context_;
};

struct GpuDraftTailBindingCpp {
  int64_t seat = -1;
  int64_t request_epoch = -1;
};

struct GpuDrafterEgressSnapshotCpp {
  int64_t seat = -1;
  int64_t request_epoch = -1;
  int64_t egress_seq = 0;
  int64_t committed_len = -1;
  int64_t model_output_len = -1;
  int64_t ack_ready_len = -1;
  int64_t last_commit_token = -1;
  int64_t raw_tail_len = 0;
  int64_t error_code = 0;
  std::vector<int64_t> raw_tail_tokens;
};

struct GpuDrafterProgressCpp {
  std::string request_id;
  int32_t src_verifier_rank = -1;
  int64_t request_epoch = -1;
  int64_t model_output_len = -1;
  int64_t committed_len = -1;
  int64_t raw_tail_len = -1;
};

struct GpuDrafterEgressWakeStateCpp {
  std::atomic<uint64_t> notify_seq{0};
  std::condition_variable queue_cv;
};

struct GpuDraftTailUpdateRowCpp {
  int64_t seat = -1;
  int64_t op_seq = 0;
  int64_t request_epoch = -1;
  GpuDraftTailUpdateOp op = GpuDraftTailUpdateOp::kOpen;
  int64_t arg0 = 0;
  int64_t arg1 = 0;
  int64_t reserved = 0;
  std::vector<int64_t> tokens;
};

class GpuDraftTailBufferCore {
 public:
  GpuDraftTailBufferCore(
      int64_t device_index,
      int64_t num_seats,
      int64_t num_draft_tokens,
      int64_t pending_token_capacity,
      uintptr_t landing_stream,
      uintptr_t versions,
      uintptr_t publish_seqs,
      uintptr_t request_epochs,
      uintptr_t prompt_lens,
      uintptr_t committed_lens,
      uintptr_t can_accept_prefix_lens,
      uintptr_t raw_tail_lens,
      uintptr_t consumable_tail_lens,
      uintptr_t pending_expected_lens,
      uintptr_t pending_expected_tokens,
      uintptr_t tail_tokens,
      uintptr_t last_op_seqs,
      uintptr_t error_codes,
      uintptr_t error_op_seqs,
      uintptr_t pending_prefix_fast_forward_cts,
      uintptr_t model_output_lens,
      uintptr_t model_state_positions,
      uintptr_t model_input_tokens,
      uintptr_t checkpoint_positions,
      uintptr_t egress_seqs,
      uintptr_t last_commit_tokens,
      bool drafter_authoritative,
      bool allow_partial)
      : device_index_(device_index),
        num_seats_(num_seats),
        num_draft_tokens_(num_draft_tokens),
        tail_capacity_(2 * num_draft_tokens + 1),
        pending_token_capacity_(pending_token_capacity),
        landing_stream_(reinterpret_cast<void*>(landing_stream)),
        versions_(reinterpret_cast<int64_t*>(versions)),
        publish_seqs_(reinterpret_cast<int64_t*>(publish_seqs)),
        request_epochs_(reinterpret_cast<int64_t*>(request_epochs)),
        prompt_lens_(reinterpret_cast<int64_t*>(prompt_lens)),
        committed_lens_(reinterpret_cast<int64_t*>(committed_lens)),
        can_accept_prefix_lens_(
            reinterpret_cast<int64_t*>(can_accept_prefix_lens)),
        raw_tail_lens_(reinterpret_cast<int64_t*>(raw_tail_lens)),
        consumable_tail_lens_(
            reinterpret_cast<int64_t*>(consumable_tail_lens)),
        pending_expected_lens_(
            reinterpret_cast<int64_t*>(pending_expected_lens)),
        pending_expected_tokens_(
            reinterpret_cast<int64_t*>(pending_expected_tokens)),
        tail_tokens_(reinterpret_cast<int64_t*>(tail_tokens)),
        last_op_seqs_(reinterpret_cast<int64_t*>(last_op_seqs)),
        error_codes_(reinterpret_cast<int64_t*>(error_codes)),
        error_op_seqs_(reinterpret_cast<int64_t*>(error_op_seqs)),
        pending_prefix_fast_forward_cts_(
            reinterpret_cast<int64_t*>(pending_prefix_fast_forward_cts)),
        model_output_lens_(reinterpret_cast<int64_t*>(model_output_lens)),
        model_state_positions_(
            reinterpret_cast<int64_t*>(model_state_positions)),
        model_input_tokens_(reinterpret_cast<int64_t*>(model_input_tokens)),
        checkpoint_positions_(
            reinterpret_cast<int64_t*>(checkpoint_positions)),
        egress_seqs_(reinterpret_cast<int64_t*>(egress_seqs)),
        last_commit_tokens_(
            reinterpret_cast<int64_t*>(last_commit_tokens)),
        drafter_authoritative_(drafter_authoritative),
        strict_selection_(!allow_partial && !drafter_authoritative) {
    if (device_index_ < 0 || num_seats_ <= 0 || num_draft_tokens_ <= 0 ||
        pending_token_capacity_ < tail_capacity_) {
      throw std::runtime_error(
          "GPU draft-tail dimensions are invalid or pending capacity is "
          "smaller than the tail capacity");
    }
    if (landing_stream_ == nullptr || versions_ == nullptr ||
        publish_seqs_ == nullptr || request_epochs_ == nullptr ||
        prompt_lens_ == nullptr || committed_lens_ == nullptr ||
        raw_tail_lens_ == nullptr ||
        can_accept_prefix_lens_ == nullptr ||
        consumable_tail_lens_ == nullptr ||
        pending_expected_lens_ == nullptr ||
        pending_expected_tokens_ == nullptr || tail_tokens_ == nullptr ||
        last_op_seqs_ == nullptr ||
        error_codes_ == nullptr || error_op_seqs_ == nullptr ||
        pending_prefix_fast_forward_cts_ == nullptr ||
        model_output_lens_ == nullptr || model_state_positions_ == nullptr ||
        model_input_tokens_ == nullptr ||
        checkpoint_positions_ == nullptr || egress_seqs_ == nullptr ||
        last_commit_tokens_ == nullptr) {
      throw std::runtime_error(
          "GPU draft-tail buffers and landing stream must be non-null");
    }
    set_device();
    for (int64_t seat = 0; seat < num_seats_; ++seat) {
      free_seats_.push_back(seat);
    }
    staging_slots_.reserve(kMaxStagingSlots);
    // Include seat-group offsets. Each completed egress snapshot can contribute
    // an echo and a tail per seat. No pinned allocation/free is safe while a
    // strict selector waits for this landing thread to publish remaining rows.
    const size_t rows_per_seat = strict_selection_ ? 2 * kEgressSnapshotSlots : 1;
    const size_t staging_words = static_cast<size_t>(num_seats_) *
        (rows_per_seat * static_cast<size_t>(
            kGpuDraftTailUpdateMetadataWidth + tail_capacity_) + 1) + 1;
    const size_t initial_slots = strict_selection_
        ? kMaxStagingSlots : kInitialStagingSlots;
    for (size_t i = 0; i < initial_slots; ++i) {
      add_staging_slot(staging_words);
    }
    for (auto& slot : commit_handoff_slots_) {
      check_cuda(
          cudaEventCreateWithFlags(&slot.applied, cudaEventDisableTiming),
          "cudaEventCreateWithFlags commit applied");
    }
    check_cuda(
        cudaEventCreateWithFlags(&lifecycle_event_, cudaEventDisableTiming),
        "cudaEventCreateWithFlags lifecycle fence");
    if (drafter_authoritative_) {
      check_cuda(
          cudaStreamCreateWithFlags(&egress_stream_, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags drafter egress");
      const size_t egress_words = static_cast<size_t>(num_seats_) *
          static_cast<size_t>(
              kGpuDrafterEgressMetadataWidth + tail_capacity_);
      const size_t egress_bytes = egress_words * sizeof(int64_t);
      const size_t seat_index_bytes =
          static_cast<size_t>(num_seats_) * sizeof(int64_t);
      for (auto& slot : egress_trigger_slots_) {
        check_cuda(
            cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming),
            "cudaEventCreateWithFlags drafter egress trigger");
      }
      for (auto& slot : egress_snapshot_slots_) {
        check_cuda(
            cudaHostAlloc(
                reinterpret_cast<void**>(&slot.host),
                egress_bytes,
                cudaHostAllocPortable),
            "cudaHostAlloc drafter egress snapshot");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&slot.device), egress_bytes),
            "cudaMalloc drafter egress snapshot");
        check_cuda(
            cudaHostAlloc(
                reinterpret_cast<void**>(&slot.seat_indices_host),
                seat_index_bytes,
                cudaHostAllocPortable),
            "cudaHostAlloc drafter egress seat indices");
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&slot.seat_indices_device),
                seat_index_bytes),
            "cudaMalloc drafter egress seat indices");
        check_cuda(
            cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming),
            "cudaEventCreateWithFlags drafter egress snapshot");
      }
    }
  }

  ~GpuDraftTailBufferCore() {
    try {
      close();
    } catch (...) {
      // Destructors must not terminate the process. Explicit close() retains
      // the actionable CUDA error for normal lifecycle management.
    }
  }

  GpuDraftTailBufferCore(const GpuDraftTailBufferCore&) = delete;
  GpuDraftTailBufferCore& operator=(const GpuDraftTailBufferCore&) = delete;

  int64_t num_seats() const { return num_seats_; }
  int64_t num_draft_tokens() const { return num_draft_tokens_; }
  int64_t tail_capacity() const { return tail_capacity_; }
  int64_t pending_token_capacity() const { return pending_token_capacity_; }
  size_t staging_slot_count() const {
    return staging_slot_count_.load(std::memory_order_acquire);
  }
  size_t max_staging_slots() const { return kMaxStagingSlots; }

  void bind_request(
      const std::string& request_id,
      int64_t seat,
      int64_t request_epoch) {
    if (seat < 0 || seat >= num_seats_) {
      throw std::runtime_error("GPU draft-tail seat is out of range");
    }
    if (request_epoch < 0 ||
        request_epoch == std::numeric_limits<int64_t>::max()) {
      throw std::runtime_error(
          "GPU draft-tail request epoch must be non-negative and incrementable");
    }
    std::lock_guard<std::mutex> guard(binding_mu_);
    const std::string binding_key = request_id;
    auto seat_it = seat_bindings_.find(seat);
    if (seat_it != seat_bindings_.end()) {
      const auto& previous = seat_it->second;
      if (request_epoch <= previous.request_epoch &&
          previous.request_id != request_id) {
        throw std::runtime_error(
            "GPU draft-tail seat reuse requires a newer request epoch");
      }
      request_bindings_.erase(previous.binding_key);
    }
    auto request_it = request_bindings_.find(binding_key);
    if (request_it != request_bindings_.end() &&
        (request_it->second.seat != seat ||
         request_it->second.request_epoch != request_epoch)) {
      throw std::runtime_error(
          "GPU draft-tail request identity was rebound inconsistently");
    }
    request_bindings_[binding_key] = {seat, request_epoch};
    seat_bindings_[seat] = {
        binding_key, request_id, -1, 0, request_epoch};
    auto free_it = std::find(free_seats_.begin(), free_seats_.end(), seat);
    if (free_it != free_seats_.end()) free_seats_.erase(free_it);
    next_request_epoch_ = std::max(next_request_epoch_, request_epoch + 1);
  }

  py::tuple lookup_binding(
      const std::string& request_id,
      int64_t src_verifier_rank) {
    if (src_verifier_rank < std::numeric_limits<int32_t>::min() ||
        src_verifier_rank > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("Verifier rank does not fit int32");
    }
    std::lock_guard<std::mutex> guard(binding_mu_);
    auto binding = find_control_binding_locked(
        request_id, static_cast<int32_t>(src_verifier_rank));
    return py::make_tuple(binding.seat, binding.request_epoch);
  }

  void wait_for_landing(uintptr_t caller_stream) {
    set_device();
    std::lock_guard<std::mutex> guard(binding_mu_);
    check_cuda(
        cudaEventRecord(
            lifecycle_event_, reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaEventRecord GPU draft-tail lifecycle fence");
    check_cuda(
        cudaStreamWaitEvent(
            reinterpret_cast<cudaStream_t>(caller_stream),
            lifecycle_event_,
            0),
        "cudaStreamWaitEvent GPU draft-tail lifecycle fence");
  }

  void set_egress_notifier(std::function<void()> notifier) {
    std::lock_guard<std::mutex> guard(egress_notifier_mu_);
    egress_notifier_ = std::move(notifier);
  }

  void clear_egress_notifier() {
    std::lock_guard<std::mutex> guard(egress_notifier_mu_);
    egress_notifier_ = nullptr;
  }

  bool has_pending_egress_work() {
    if (!drafter_authoritative_) return false;
    std::lock_guard<std::mutex> guard(egress_mu_);
    return std::any_of(
               egress_trigger_slots_.begin(),
               egress_trigger_slots_.end(),
               [](const auto& slot) { return slot.in_flight; }) ||
        std::any_of(
               egress_snapshot_slots_.begin(),
               egress_snapshot_slots_.end(),
               [](const auto& slot) { return slot.in_flight; });
  }

  bool lookup_egress_binding(
      int64_t seat,
      int64_t request_epoch,
      std::string* request_id,
      int32_t* src_verifier_rank,
      int64_t* initial_committed_len) {
    std::lock_guard<std::mutex> guard(binding_mu_);
    auto it = seat_bindings_.find(seat);
    if (it == seat_bindings_.end() ||
        it->second.request_epoch != request_epoch ||
        it->second.src_verifier_rank < 0) {
      return false;
    }
    *request_id = it->second.request_id;
    *src_verifier_rank = it->second.src_verifier_rank;
    *initial_committed_len = it->second.initial_committed_len;
    return true;
  }

  std::vector<GpuDrafterEgressSnapshotCpp>
  poll_ready_egress_snapshots() {
    std::vector<GpuDrafterEgressSnapshotCpp> snapshots;
    if (!drafter_authoritative_) return snapshots;
    std::vector<int64_t> active_seats;
    {
      std::lock_guard<std::mutex> binding_guard(binding_mu_);
      active_seats.reserve(seat_bindings_.size());
      for (const auto& item : seat_bindings_) {
        active_seats.push_back(item.first);
      }
    }
    std::sort(active_seats.begin(), active_seats.end());
    set_device();
    std::lock_guard<std::mutex> guard(egress_mu_);

    const int64_t row_width =
        kGpuDrafterEgressMetadataWidth + tail_capacity_;
    for (auto& slot : egress_snapshot_slots_) {
      if (!slot.in_flight) continue;
      const cudaError_t status = cudaEventQuery(slot.ready);
      if (status == cudaErrorNotReady) continue;
      check_cuda(status, "cudaEventQuery drafter egress snapshot");
      for (size_t row_index = 0; row_index < slot.active_seats.size();
           ++row_index) {
        const int64_t seat = slot.active_seats[row_index];
        const int64_t* row = slot.host + row_index * row_width;
        GpuDrafterEgressSnapshotCpp snapshot;
        snapshot.seat = seat;
        snapshot.request_epoch = row[0];
        snapshot.egress_seq = row[1];
        snapshot.committed_len = row[2];
        snapshot.model_output_len = row[3];
        snapshot.ack_ready_len = row[4];
        snapshot.last_commit_token = row[5];
        snapshot.raw_tail_len = row[6];
        snapshot.error_code = row[7];
        if (snapshot.raw_tail_len >= 0 &&
            snapshot.raw_tail_len <= tail_capacity_) {
          snapshot.raw_tail_tokens.reserve(
              static_cast<size_t>(snapshot.raw_tail_len));
          for (int64_t index = 0; index < snapshot.raw_tail_len; ++index) {
            snapshot.raw_tail_tokens.push_back(
                row[kGpuDrafterEgressMetadataWidth + index]);
          }
        }
        snapshots.push_back(std::move(snapshot));
      }
      slot.active_seats.clear();
      slot.in_flight = false;
    }

    auto snapshot_slot = std::find_if(
        egress_snapshot_slots_.begin(),
        egress_snapshot_slots_.end(),
        [](const auto& slot) { return !slot.in_flight; });
    if (snapshot_slot == egress_snapshot_slots_.end()) return snapshots;

    bool trigger_ready = false;
    for (auto& trigger : egress_trigger_slots_) {
      if (!trigger.in_flight) continue;
      const cudaError_t status = cudaEventQuery(trigger.ready);
      if (status == cudaErrorNotReady) continue;
      check_cuda(status, "cudaEventQuery drafter egress trigger");
      trigger.in_flight = false;
      trigger_ready = true;
    }
    if (!trigger_ready) return snapshots;
    if (active_seats.empty()) return snapshots;

    std::copy(
        active_seats.begin(),
        active_seats.end(),
        snapshot_slot->seat_indices_host);
    const size_t seat_index_bytes = active_seats.size() * sizeof(int64_t);
    check_cuda(
        cudaMemcpyAsync(
            snapshot_slot->seat_indices_device,
            snapshot_slot->seat_indices_host,
            seat_index_bytes,
            cudaMemcpyHostToDevice,
            egress_stream_),
        "cudaMemcpyAsync drafter egress seat indices");
    snapshot_slot->active_seats = active_seats;

    launch_snapshot_gpu_drafter_egress(
        snapshot_slot->device,
        snapshot_slot->seat_indices_device,
        static_cast<int64_t>(active_seats.size()),
        num_seats_,
        tail_capacity_,
        versions_,
        request_epochs_,
        committed_lens_,
        can_accept_prefix_lens_,
        raw_tail_lens_,
        pending_expected_lens_,
        tail_tokens_,
        error_codes_,
        model_output_lens_,
        egress_seqs_,
        last_commit_tokens_,
        reinterpret_cast<void*>(egress_stream_));
    check_cuda(
        cudaMemcpyAsync(
            snapshot_slot->host,
            snapshot_slot->device,
            active_seats.size() * static_cast<size_t>(row_width) *
                sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            egress_stream_),
        "cudaMemcpyAsync drafter egress snapshot");
    check_cuda(
        cudaEventRecord(snapshot_slot->ready, egress_stream_),
        "cudaEventRecord drafter egress snapshot");
    snapshot_slot->in_flight = true;
    return snapshots;
  }

  void apply_control_batch(
      const DraftControlBatchCpp& batch,
      bool auto_bind_sync = false) {
    std::vector<GpuDraftTailUpdateRowCpp> rows;
    rows.reserve(
        batch.sync_messages.size() + batch.verify_commit_messages.size() +
        batch.close_messages.size());
    std::lock_guard<std::mutex> guard(binding_mu_);
    for (const auto& message : batch.sync_messages) {
      const auto binding = auto_bind_sync
          ? auto_bind_control_request_locked(
                message.draft_key(),
                static_cast<int64_t>(message.committed_outputs.size()))
          : require_binding_locked(message.request_id, true);
      GpuDraftTailUpdateRowCpp row;
      row.seat = binding.seat;
      row.op_seq = next_op_seq_++;
      row.request_epoch = binding.request_epoch;
      row.op = GpuDraftTailUpdateOp::kOpen;
      row.arg0 = static_cast<int64_t>(message.prompt_token_ids.size());
      row.arg1 = static_cast<int64_t>(message.committed_outputs.size());
      rows.push_back(std::move(row));
    }
    for (const auto& message : batch.verify_commit_messages) {
      message.validate();
      auto binding = find_control_binding_locked(
          message.request_id, message.src_verifier_rank);
      if (binding.seat < 0) continue;
      if (message.committed_tokens.size() >
          static_cast<size_t>(tail_capacity_)) {
        throw std::runtime_error(
            "GPU draft-tail commit token payload exceeds fixed row capacity");
      }
      GpuDraftTailUpdateRowCpp row;
      row.seat = binding.seat;
      row.op_seq = next_op_seq_++;
      row.request_epoch = binding.request_epoch;
      row.op = GpuDraftTailUpdateOp::kVerifyCommit;
      row.arg0 = message.pre_verify_committed_len;
      row.tokens.reserve(message.committed_tokens.size());
      for (int32_t token : message.committed_tokens) row.tokens.push_back(token);
      rows.push_back(std::move(row));
    }
    for (const auto& message : batch.close_messages) {
      const std::string control_key = message.draft_key().map_key();
      auto binding = find_control_binding_locked(
          message.request_id, message.src_verifier_rank);
      if (binding.seat < 0) continue;
      GpuDraftTailUpdateRowCpp row;
      row.seat = binding.seat;
      row.op_seq = next_op_seq_++;
      row.request_epoch = binding.request_epoch;
      row.op = GpuDraftTailUpdateOp::kClose;
      rows.push_back(std::move(row));
      auto binding_it = request_bindings_.find(control_key);
      const std::string binding_key = binding_it != request_bindings_.end()
          ? control_key
          : message.request_id;
      request_bindings_.erase(binding_key);
      auto seat_it = seat_bindings_.find(binding.seat);
      if (seat_it != seat_bindings_.end() &&
          seat_it->second.request_id == message.request_id &&
          seat_it->second.request_epoch == binding.request_epoch) {
        seat_bindings_.erase(seat_it);
        free_seats_.push_back(binding.seat);
      }
    }
    publish_update_rows(std::move(rows));
  }

  void append_draft_stream_batch(
      const DraftTailStreamOutputBatchCpp& batch) {
    if (batch.outputs.empty()) return;
    std::vector<GpuDraftTailUpdateRowCpp> rows;
    rows.reserve(batch.outputs.size());
    std::lock_guard<std::mutex> guard(binding_mu_);
    for (const auto& output : batch.outputs) {
      output.validate();
      auto binding = find_binding_locked(output.request_id);
      if (binding.seat < 0) continue;
      if (!output.is_commit_echo &&
          output.tokens.size() > static_cast<size_t>(tail_capacity_)) {
        throw std::runtime_error(
            "GPU draft-tail append span exceeds fixed row capacity");
      }
      GpuDraftTailUpdateRowCpp row;
      row.seat = binding.seat;
      row.op_seq = next_op_seq_++;
      row.request_epoch = binding.request_epoch;
      row.op = output.is_commit_echo
          ? GpuDraftTailUpdateOp::kCommitAck
          : GpuDraftTailUpdateOp::kAppendDraft;
      row.arg0 = output.base_committed_len;
      row.arg1 = output.start_token_pos;
      if (!output.is_commit_echo) {
        row.tokens.reserve(output.tokens.size());
        for (int32_t token : output.tokens) row.tokens.push_back(token);
      }
      rows.push_back(std::move(row));
    }
    publish_update_rows(std::move(rows));
  }

  void apply_verify_commit_from_device(
      uintptr_t gpu_seats,
      uintptr_t expected_request_epochs,
      uintptr_t pre_verify_seq_lens,
      uintptr_t accept_tokens,
      uintptr_t num_accept_tokens,
      uintptr_t commit_mask,
      int64_t accept_token_stride,
      int64_t batch_size,
      uintptr_t forward_stream) {
    if (batch_size < 0) {
      throw std::runtime_error(
          "GPU draft-tail critical commit batch size must be non-negative");
    }
    if (batch_size == 0) return;
    if (gpu_seats == 0 || expected_request_epochs == 0 || accept_tokens == 0 ||
        num_accept_tokens == 0) {
      throw std::runtime_error(
          "GPU draft-tail critical commit received a null required pointer");
    }
    if (accept_token_stride <= 0 || accept_token_stride > tail_capacity_) {
      throw std::runtime_error(
          "GPU draft-tail critical commit stride is outside tail capacity");
    }

    set_device();
    auto* forward_stream_ptr = reinterpret_cast<cudaStream_t>(forward_stream);
    std::lock_guard<std::mutex> guard(binding_mu_);
    auto& slot = acquire_commit_handoff_slot();
    launch_apply_gpu_draft_tail_verify_commit(
        reinterpret_cast<const int64_t*>(gpu_seats),
        reinterpret_cast<const int64_t*>(expected_request_epochs),
        pre_verify_seq_lens == 0
            ? nullptr
            : reinterpret_cast<const int64_t*>(pre_verify_seq_lens),
        reinterpret_cast<const int32_t*>(accept_tokens),
        reinterpret_cast<const int32_t*>(num_accept_tokens),
        commit_mask == 0 ? nullptr : reinterpret_cast<const bool*>(commit_mask),
        accept_token_stride,
        batch_size,
        num_seats_,
        tail_capacity_,
        versions_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        can_accept_prefix_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        pending_expected_lens_,
        pending_expected_tokens_,
        tail_tokens_,
        last_op_seqs_,
        error_codes_,
        error_op_seqs_,
        forward_stream_ptr);
    check_cuda(
        cudaEventRecord(slot.applied, forward_stream_ptr),
        "cudaEventRecord GPU draft-tail commit applied");
    slot.in_flight = true;
  }

  void select_snapshot(
      uintptr_t gpu_seats,
      uintptr_t expected_request_epochs,
      uintptr_t seq_lens,
      uintptr_t bonus_tokens,
      bool bonus_tokens_are_int32,
      uintptr_t compact_out,
      uintptr_t logical_committed_lens_out,
      uintptr_t debug_out,
      int64_t debug_width,
      int64_t batch_size,
      uintptr_t verify_stream,
      bool allow_partial,
      int64_t required_tail_len) {
    if (batch_size < 0) {
      throw std::runtime_error("GPU draft-tail batch size must be non-negative");
    }
    if (batch_size == 0) return;
    if (required_tail_len < 0 || required_tail_len > num_draft_tokens_) {
      throw std::runtime_error("GPU draft-tail required length is outside draft width");
    }
    if (gpu_seats == 0 || expected_request_epochs == 0 || seq_lens == 0 ||
        bonus_tokens == 0 || compact_out == 0) {
      throw std::runtime_error(
          "GPU draft-tail select received a null required pointer");
    }
    if (debug_out != 0 && debug_width < kGpuDraftTailDebugWidth) {
      throw std::runtime_error(
          "GPU draft-tail debug output must have at least seven columns");
    }
    set_device();
    launch_select_gpu_draft_tail(
        reinterpret_cast<const int64_t*>(gpu_seats),
        reinterpret_cast<const int64_t*>(expected_request_epochs),
        reinterpret_cast<const int64_t*>(seq_lens),
        reinterpret_cast<const void*>(bonus_tokens),
        bonus_tokens_are_int32,
        reinterpret_cast<int64_t*>(compact_out),
        logical_committed_lens_out == 0
            ? nullptr
            : reinterpret_cast<int64_t*>(logical_committed_lens_out),
        debug_out == 0 ? nullptr : reinterpret_cast<int64_t*>(debug_out),
        debug_width,
        batch_size,
        num_seats_,
        num_draft_tokens_,
        tail_capacity_,
        versions_,
        publish_seqs_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        pending_expected_lens_,
        tail_tokens_,
        error_codes_,
        error_op_seqs_,
        pending_prefix_fast_forward_cts_,
        reinterpret_cast<void*>(verify_stream),
        allow_partial,
        required_tail_len);
  }

  void select_mock_snapshot(
      uintptr_t gpu_seats,
      uintptr_t seq_lens,
      uintptr_t bonus_tokens,
      bool bonus_tokens_are_int32,
      uintptr_t compact_out,
      uintptr_t logical_committed_lens_out,
      uintptr_t debug_out,
      int64_t debug_width,
      int64_t batch_size,
      uintptr_t verify_stream) {
    if (batch_size < 0) {
      throw std::runtime_error(
          "Mock GPU draft-tail batch size must be non-negative");
    }
    if (batch_size == 0) return;
    if (gpu_seats == 0 || seq_lens == 0 || bonus_tokens == 0 ||
        compact_out == 0) {
      throw std::runtime_error(
          "Mock GPU draft-tail select received a null required pointer");
    }
    if (debug_out != 0 && debug_width < kGpuDraftTailDebugWidth) {
      throw std::runtime_error(
          "Mock GPU draft-tail debug output must have at least seven columns");
    }
    set_device();
    launch_select_mock_gpu_draft_tail(
        reinterpret_cast<const int64_t*>(gpu_seats),
        reinterpret_cast<const int64_t*>(seq_lens),
        reinterpret_cast<const void*>(bonus_tokens),
        bonus_tokens_are_int32,
        reinterpret_cast<int64_t*>(compact_out),
        logical_committed_lens_out == 0
            ? nullptr
            : reinterpret_cast<int64_t*>(logical_committed_lens_out),
        debug_out == 0 ? nullptr : reinterpret_cast<int64_t*>(debug_out),
        debug_width,
        batch_size,
        num_seats_,
        num_draft_tokens_,
        tail_capacity_,
        tail_tokens_,
        reinterpret_cast<void*>(verify_stream));
  }

  void prepare_decode(
      uintptr_t mirror_seats,
      uintptr_t expected_request_epochs,
      uintptr_t req_pool_indices,
      uintptr_t candidate_out_cache_locs,
      uintptr_t checkpoint_slot_table,
      int64_t checkpoint_capacity,
      uintptr_t req_to_token,
      int64_t req_to_token_num_rows,
      int64_t req_to_token_row_stride,
      uintptr_t resolved_input_ids,
      uintptr_t resolved_seq_lens,
      uintptr_t resolved_orig_seq_lens,
      uintptr_t mamba_src_indices,
      uintptr_t mamba_dst_indices,
      uintptr_t captured_state_positions,
      uintptr_t old_cache_locs,
      uintptr_t kv_ownership,
      int64_t batch_size,
      uintptr_t caller_stream) {
    require_drafter_authoritative("prepare_decode");
    if (batch_size < 0 || checkpoint_capacity != tail_capacity_ ||
        req_to_token_num_rows <= 0 || req_to_token_row_stride <= 0) {
      throw std::runtime_error(
          "GPU drafter decode dimensions are invalid or checkpoint capacity "
          "does not match the control mirror");
    }
    if (batch_size == 0) return;
    if ((checkpoint_slot_table == 0) != (mamba_src_indices == 0) ||
        (checkpoint_slot_table == 0) != (mamba_dst_indices == 0)) {
      throw std::runtime_error("Recurrent checkpoint table and routes must be supplied together");
    }
    const std::array<uintptr_t, 11> required = {
        mirror_seats,
        expected_request_epochs,
        req_pool_indices,
        candidate_out_cache_locs,
        req_to_token,
        resolved_input_ids,
        resolved_seq_lens,
        resolved_orig_seq_lens,
        captured_state_positions,
        old_cache_locs,
        kv_ownership};
    if (std::any_of(required.begin(), required.end(), [](uintptr_t ptr) {
          return ptr == 0;
        })) {
      throw std::runtime_error(
          "GPU drafter prepare_decode received a null required pointer");
    }
    set_device();
    launch_prepare_gpu_drafter_decode(
        reinterpret_cast<const int64_t*>(mirror_seats),
        reinterpret_cast<const int64_t*>(expected_request_epochs),
        reinterpret_cast<const int64_t*>(req_pool_indices),
        reinterpret_cast<const int64_t*>(candidate_out_cache_locs),
        reinterpret_cast<const int64_t*>(checkpoint_slot_table),
        checkpoint_capacity,
        reinterpret_cast<int32_t*>(req_to_token),
        req_to_token_num_rows,
        req_to_token_row_stride,
        reinterpret_cast<int64_t*>(resolved_input_ids),
        reinterpret_cast<int64_t*>(resolved_seq_lens),
        reinterpret_cast<int32_t*>(resolved_orig_seq_lens),
        reinterpret_cast<int64_t*>(mamba_src_indices),
        reinterpret_cast<int64_t*>(mamba_dst_indices),
        reinterpret_cast<int64_t*>(captured_state_positions),
        reinterpret_cast<int64_t*>(old_cache_locs),
        reinterpret_cast<int64_t*>(kv_ownership),
        batch_size,
        num_seats_,
        versions_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        can_accept_prefix_lens_,
        raw_tail_lens_,
        pending_expected_lens_,
        last_op_seqs_,
        error_codes_,
        error_op_seqs_,
        model_output_lens_,
        model_state_positions_,
        model_input_tokens_,
        checkpoint_positions_,
        tail_capacity_,
        pending_token_capacity_,
        reinterpret_cast<void*>(caller_stream));
  }

  void finish_decode(
      uintptr_t mirror_seats,
      uintptr_t expected_request_epochs,
      uintptr_t req_pool_indices,
      uintptr_t candidate_out_cache_locs,
      uintptr_t sampled_tokens,
      bool sampled_tokens_are_int32,
      uintptr_t req_to_token,
      int64_t req_to_token_num_rows,
      int64_t req_to_token_row_stride,
      uintptr_t resolved_input_tokens,
      uintptr_t captured_state_positions,
      uintptr_t old_cache_locs,
      uintptr_t kv_outcomes,
      uintptr_t future_output_tokens,
      int64_t future_output_tokens_size,
      int64_t batch_size,
      uintptr_t caller_stream) {
    require_drafter_authoritative("finish_decode");
    if (batch_size < 0 || req_to_token_num_rows <= 0 ||
        req_to_token_row_stride <= 0 || future_output_tokens_size <= 0) {
      throw std::runtime_error("GPU drafter finish_decode dimensions are invalid");
    }
    if (batch_size == 0) return;
    const std::array<uintptr_t, 11> required = {
        mirror_seats,
        expected_request_epochs,
        req_pool_indices,
        candidate_out_cache_locs,
        sampled_tokens,
        req_to_token,
        resolved_input_tokens,
        captured_state_positions,
        old_cache_locs,
        kv_outcomes,
        future_output_tokens};
    if (std::any_of(required.begin(), required.end(), [](uintptr_t ptr) {
          return ptr == 0;
        })) {
      throw std::runtime_error(
          "GPU drafter finish_decode received a null required pointer");
    }
    set_device();
    launch_finish_gpu_drafter_decode(
        reinterpret_cast<const int64_t*>(mirror_seats),
        reinterpret_cast<const int64_t*>(expected_request_epochs),
        reinterpret_cast<const int64_t*>(req_pool_indices),
        reinterpret_cast<const int64_t*>(candidate_out_cache_locs),
        reinterpret_cast<const void*>(sampled_tokens),
        sampled_tokens_are_int32,
        reinterpret_cast<int32_t*>(req_to_token),
        req_to_token_num_rows,
        req_to_token_row_stride,
        reinterpret_cast<const int64_t*>(resolved_input_tokens),
        reinterpret_cast<const int64_t*>(captured_state_positions),
        reinterpret_cast<const int64_t*>(old_cache_locs),
        reinterpret_cast<int64_t*>(kv_outcomes),
        reinterpret_cast<int64_t*>(future_output_tokens),
        future_output_tokens_size,
        batch_size,
        num_seats_,
        tail_capacity_,
        pending_token_capacity_,
        versions_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        can_accept_prefix_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        pending_expected_lens_,
        pending_expected_tokens_,
        tail_tokens_,
        last_op_seqs_,
        error_codes_,
        error_op_seqs_,
        model_output_lens_,
        model_state_positions_,
        model_input_tokens_,
        checkpoint_positions_,
        egress_seqs_,
        last_commit_tokens_,
        reinterpret_cast<void*>(caller_stream));
    enqueue_egress_trigger(reinterpret_cast<void*>(caller_stream));
  }

  void append_prefill_sample(
      uintptr_t mirror_seats,
      uintptr_t expected_request_epochs,
      uintptr_t sampled_tokens,
      bool sampled_tokens_are_int32,
      uintptr_t accept_out,
      int64_t batch_size,
      uintptr_t caller_stream) {
    require_drafter_authoritative("append_prefill_sample");
    if (batch_size < 0) {
      throw std::runtime_error(
          "GPU drafter prefill-sample batch size must be non-negative");
    }
    if (batch_size == 0) return;
    if (mirror_seats == 0 || expected_request_epochs == 0 ||
        sampled_tokens == 0 || accept_out == 0) {
      throw std::runtime_error(
          "GPU drafter append_prefill_sample received a null required pointer");
    }
    set_device();
    launch_append_gpu_drafter_prefill_sample(
        reinterpret_cast<const int64_t*>(mirror_seats),
        reinterpret_cast<const int64_t*>(expected_request_epochs),
        reinterpret_cast<void*>(sampled_tokens),
        sampled_tokens_are_int32,
        reinterpret_cast<bool*>(accept_out),
        batch_size,
        num_seats_,
        tail_capacity_,
        pending_token_capacity_,
        versions_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        can_accept_prefix_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        pending_expected_lens_,
        pending_expected_tokens_,
        tail_tokens_,
        last_op_seqs_,
        error_codes_,
        error_op_seqs_,
        model_output_lens_,
        model_state_positions_,
        model_input_tokens_,
        checkpoint_positions_,
        egress_seqs_,
        reinterpret_cast<void*>(caller_stream));
    enqueue_egress_trigger(reinterpret_cast<void*>(caller_stream));
  }

  void close() {
    std::lock_guard<std::mutex> guard(close_mu_);
    if (closed_.load(std::memory_order_acquire)) return;
    closed_.store(true, std::memory_order_release);
    clear_egress_notifier();
    set_device();
    // Shutdown is the one place where blocking the landing stream is
    // required: every event and allocation below may still be referenced by
    // queued H2D copies or publish kernels.
    check_cuda(
        cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaStreamSynchronize GPU draft-tail landing stream");
    if (drafter_authoritative_) {
      std::lock_guard<std::mutex> egress_guard(egress_mu_);
      for (auto& slot : egress_trigger_slots_) {
        if (slot.in_flight) {
          check_cuda(
              cudaEventSynchronize(slot.ready),
              "cudaEventSynchronize drafter egress trigger");
        }
        if (slot.ready != nullptr) cudaEventDestroy(slot.ready);
        slot = {};
      }
      if (egress_stream_ != nullptr) {
        check_cuda(
            cudaStreamSynchronize(egress_stream_),
            "cudaStreamSynchronize drafter egress");
      }
      for (auto& slot : egress_snapshot_slots_) {
        if (slot.ready != nullptr) cudaEventDestroy(slot.ready);
        if (slot.device != nullptr) cudaFree(slot.device);
        if (slot.host != nullptr) cudaFreeHost(slot.host);
        if (slot.seat_indices_device != nullptr) {
          cudaFree(slot.seat_indices_device);
        }
        if (slot.seat_indices_host != nullptr) {
          cudaFreeHost(slot.seat_indices_host);
        }
        slot = {};
      }
      if (egress_stream_ != nullptr) {
        cudaStreamDestroy(egress_stream_);
        egress_stream_ = nullptr;
      }
    }
    for (auto& slot : staging_slots_) {
      if (slot.event != nullptr) cudaEventDestroy(slot.event);
      if (slot.device != nullptr) cudaFree(slot.device);
      if (slot.host != nullptr) cudaFreeHost(slot.host);
      slot = {};
    }
    for (auto& slot : commit_handoff_slots_) {
      if (slot.in_flight) {
        check_cuda(
            cudaEventSynchronize(slot.applied),
            "cudaEventSynchronize GPU draft-tail commit");
      }
      if (slot.applied != nullptr) cudaEventDestroy(slot.applied);
      slot = {};
    }
    if (lifecycle_event_ != nullptr) {
      cudaEventDestroy(lifecycle_event_);
      lifecycle_event_ = nullptr;
    }
    staging_slots_.clear();
    staging_slot_count_.store(0, std::memory_order_release);
  }

 private:
  struct SeatBindingCpp {
    std::string binding_key;
    std::string request_id;
    int32_t src_verifier_rank = -1;
    int64_t initial_committed_len = 0;
    int64_t request_epoch = -1;
  };

  struct StagingSlotCpp {
    int64_t* host = nullptr;
    int64_t* device = nullptr;
    size_t capacity_words = 0;
    cudaEvent_t event = nullptr;
    bool in_flight = false;
  };

  struct CommitHandoffSlotCpp {
    cudaEvent_t applied = nullptr;
    bool in_flight = false;
  };

  struct EgressTriggerSlotCpp {
    cudaEvent_t ready = nullptr;
    bool in_flight = false;
  };

  struct EgressSnapshotSlotCpp {
    int64_t* host = nullptr;
    int64_t* device = nullptr;
    int64_t* seat_indices_host = nullptr;
    int64_t* seat_indices_device = nullptr;
    std::vector<int64_t> active_seats;
    cudaEvent_t ready = nullptr;
    bool in_flight = false;
  };

  static constexpr size_t kInitialStagingSlots = 8;
  static constexpr size_t kMaxStagingSlots = 64;
  static constexpr size_t kCommitHandoffSlots = 8;
  static constexpr size_t kEgressTriggerSlots = 256;
  static constexpr size_t kEgressSnapshotSlots = 2;

  static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
  }

  void set_device() const {
    check_cuda(cudaSetDevice(static_cast<int>(device_index_)), "cudaSetDevice");
  }

  void enqueue_egress_trigger(void* producer_stream) {
    if (!drafter_authoritative_ ||
        closed_.load(std::memory_order_acquire)) {
      return;
    }
    set_device();
    {
      std::lock_guard<std::mutex> guard(egress_mu_);
      auto slot = std::find_if(
          egress_trigger_slots_.begin(),
          egress_trigger_slots_.end(),
          [](const auto& candidate) { return !candidate.in_flight; });
      if (slot == egress_trigger_slots_.end()) {
        throw std::runtime_error(
            "GPU drafter egress trigger ring reached its hard capacity");
      }
      check_cuda(
          cudaEventRecord(
              slot->ready, reinterpret_cast<cudaStream_t>(producer_stream)),
          "cudaEventRecord drafter egress trigger");
      slot->in_flight = true;
    }
    std::function<void()> notifier;
    {
      std::lock_guard<std::mutex> guard(egress_notifier_mu_);
      notifier = egress_notifier_;
    }
    if (notifier) notifier();
  }

  void require_drafter_authoritative(const char* operation) const {
    if (!drafter_authoritative_) {
      throw std::runtime_error(
          std::string("GPU draft-tail ") + operation +
          " requires a drafter-authoritative buffer");
    }
  }

  StagingSlotCpp make_staging_slot() {
    StagingSlotCpp slot;
    check_cuda(
        cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming),
        "cudaEventCreateWithFlags");
    return slot;
  }

  CommitHandoffSlotCpp& acquire_commit_handoff_slot() {
    for (auto& slot : commit_handoff_slots_) {
      if (slot.in_flight) {
        cudaError_t status = cudaEventQuery(slot.applied);
        if (status == cudaErrorNotReady) continue;
        check_cuda(status, "cudaEventQuery GPU draft-tail commit handoff");
        slot.in_flight = false;
      }
      return slot;
    }
    throw std::runtime_error(
        "GPU draft-tail commit event ring reached its hard limit; the forward "
        "stream is not retiring critical-path commits");
  }

  StagingSlotCpp& acquire_staging_slot(size_t required_words) {
    for (auto& slot : staging_slots_) {
      if (slot.in_flight) {
        cudaError_t status = cudaEventQuery(slot.event);
        if (status == cudaErrorNotReady) continue;
        check_cuda(status, "cudaEventQuery");
        slot.in_flight = false;
      }
      ensure_staging_capacity(slot, required_words);
      return slot;
    }
    if (staging_slots_.size() < kMaxStagingSlots) {
      return add_staging_slot(required_words);
    }
    throw std::runtime_error(
        "GPU draft-tail landing staging ring reached its hard limit of " +
        std::to_string(kMaxStagingSlots) +
        " in-flight publishes; the landing stream is not retiring work "
        "fast enough");
  }

  StagingSlotCpp& add_staging_slot(size_t required_words) {
    if (staging_slots_.size() >= kMaxStagingSlots) {
      throw std::runtime_error(
          "GPU draft-tail staging slot capacity invariant was violated");
    }
    staging_slots_.push_back(make_staging_slot());
    try {
      ensure_staging_capacity(staging_slots_.back(), required_words);
    } catch (...) {
      auto& slot = staging_slots_.back();
      if (slot.event != nullptr) cudaEventDestroy(slot.event);
      if (slot.device != nullptr) cudaFree(slot.device);
      if (slot.host != nullptr) cudaFreeHost(slot.host);
      staging_slots_.pop_back();
      throw;
    }
    staging_slot_count_.store(
        staging_slots_.size(), std::memory_order_release);
    return staging_slots_.back();
  }

  void ensure_staging_capacity(
      StagingSlotCpp& slot,
      size_t required_words) {
    if (slot.capacity_words >= required_words) return;
    if (strict_selection_ && slot.capacity_words != 0) {
      throw std::runtime_error(
          "Strict GPU draft-tail publication exceeds its preallocated "
          "egress-snapshot staging capacity");
    }
    if (slot.in_flight) {
      throw std::runtime_error(
          "Cannot resize an in-flight GPU draft-tail staging slot");
    }
    if (slot.device != nullptr) check_cuda(cudaFree(slot.device), "cudaFree");
    if (slot.host != nullptr) check_cuda(cudaFreeHost(slot.host), "cudaFreeHost");
    slot.device = nullptr;
    slot.host = nullptr;
    slot.capacity_words = 0;
    const size_t bytes = required_words * sizeof(int64_t);
    check_cuda(
        cudaHostAlloc(
            reinterpret_cast<void**>(&slot.host),
            bytes,
            cudaHostAllocPortable),
        "cudaHostAlloc");
    check_cuda(
        cudaMallocAsync(
            reinterpret_cast<void**>(&slot.device),
            bytes,
            reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaMallocAsync");
    slot.capacity_words = required_words;
  }

  GpuDraftTailBindingCpp find_binding_locked(
      const std::string& request_id) const {
    auto binding_it = request_bindings_.find(request_id);
    if (binding_it == request_bindings_.end()) return {};
    const auto binding = binding_it->second;
    auto seat_it = seat_bindings_.find(binding.seat);
    if (seat_it == seat_bindings_.end() ||
        seat_it->second.request_id != request_id ||
        seat_it->second.request_epoch != binding.request_epoch) {
      return {};
    }
    return binding;
  }

  GpuDraftTailBindingCpp find_control_binding_locked(
      const std::string& request_id,
      int32_t src_verifier_rank) const {
    const std::string control_key =
        DraftReqKeyCpp{src_verifier_rank, request_id}.map_key();
    auto binding = find_binding_key_locked(control_key);
    return binding.seat >= 0 ? binding : find_binding_locked(request_id);
  }

  GpuDraftTailBindingCpp find_binding_key_locked(
      const std::string& binding_key) const {
    auto binding_it = request_bindings_.find(binding_key);
    if (binding_it == request_bindings_.end()) return {};
    const auto binding = binding_it->second;
    auto seat_it = seat_bindings_.find(binding.seat);
    if (seat_it == seat_bindings_.end() ||
        seat_it->second.binding_key != binding_key ||
        seat_it->second.request_epoch != binding.request_epoch) {
      return {};
    }
    return binding;
  }

  GpuDraftTailBindingCpp auto_bind_control_request_locked(
      const DraftReqKeyCpp& key,
      int64_t initial_committed_len) {
    const std::string binding_key = key.map_key();
    auto binding = find_binding_key_locked(binding_key);
    if (binding.seat >= 0) {
      auto seat_it = seat_bindings_.find(binding.seat);
      if (seat_it == seat_bindings_.end() ||
          seat_it->second.initial_committed_len != initial_committed_len) {
        throw std::runtime_error(
            "GPU drafter duplicate OPEN changed its committed prefix length");
      }
      return binding;
    }
    if (free_seats_.empty()) {
      throw std::runtime_error(
          "GPU drafter control mirror has no free seat for request_id=" +
          key.request_id);
    }
    if (next_request_epoch_ == std::numeric_limits<int64_t>::max()) {
      throw std::runtime_error("GPU drafter request epoch exhausted");
    }
    const int64_t seat = free_seats_.front();
    free_seats_.pop_front();
    const int64_t request_epoch = next_request_epoch_++;
    request_bindings_[binding_key] = {seat, request_epoch};
    seat_bindings_[seat] = {
        binding_key,
        key.request_id,
        key.src_verifier_rank,
        initial_committed_len,
        request_epoch};
    return {seat, request_epoch};
  }

  GpuDraftTailBindingCpp require_binding_locked(
      const std::string& request_id,
      bool is_open) const {
    auto binding = find_binding_locked(request_id);
    if (binding.seat < 0) {
      throw std::runtime_error(
          std::string("GPU draft-tail ") + (is_open ? "OPEN" : "update") +
          " has no request-to-seat binding: request_id=" + request_id);
    }
    return binding;
  }

  void publish_update_rows(std::vector<GpuDraftTailUpdateRowCpp> rows) {
    if (rows.empty()) return;
    set_device();
    std::stable_sort(
        rows.begin(),
        rows.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.seat < rhs.seat; });
    std::vector<int64_t> group_offsets;
    group_offsets.reserve(rows.size() + 1);
    group_offsets.push_back(0);
    for (size_t index = 1; index < rows.size(); ++index) {
      if (rows[index - 1].seat != rows[index].seat) {
        group_offsets.push_back(static_cast<int64_t>(index));
      }
    }
    group_offsets.push_back(static_cast<int64_t>(rows.size()));

    const size_t row_width = static_cast<size_t>(
        kGpuDraftTailUpdateMetadataWidth + tail_capacity_);
    const size_t row_words = rows.size() * row_width;
    const size_t required_words = row_words + group_offsets.size();
    auto& slot = acquire_staging_slot(required_words);
    std::fill(slot.host, slot.host + required_words, 0);
    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
      const auto& row = rows[row_index];
      int64_t* dst = slot.host + row_index * row_width;
      dst[0] = row.seat;
      dst[1] = row.op_seq;
      dst[2] = row.request_epoch;
      dst[3] = static_cast<int64_t>(row.op);
      dst[4] = row.arg0;
      dst[5] = row.arg1;
      dst[6] = row.reserved;
      dst[7] = static_cast<int64_t>(row.tokens.size());
      if (row.tokens.size() > static_cast<size_t>(tail_capacity_)) {
        throw std::runtime_error(
            "GPU draft-tail update row exceeds its fixed token capacity");
      }
      std::copy(
          row.tokens.begin(),
          row.tokens.end(),
          dst + kGpuDraftTailUpdateMetadataWidth);
    }
    std::copy(
        group_offsets.begin(), group_offsets.end(), slot.host + row_words);
    const size_t bytes = required_words * sizeof(int64_t);
    check_cuda(
        cudaMemcpyAsync(
            slot.device,
            slot.host,
            bytes,
            cudaMemcpyHostToDevice,
            reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaMemcpyAsync");
    launch_update_gpu_draft_tail(
        slot.device,
        slot.device + row_words,
        static_cast<int64_t>(group_offsets.size() - 1),
        tail_capacity_,
        pending_token_capacity_,
        versions_,
        publish_seqs_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        can_accept_prefix_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        pending_expected_lens_,
        pending_expected_tokens_,
        tail_tokens_,
        last_op_seqs_,
        error_codes_,
        error_op_seqs_,
        pending_prefix_fast_forward_cts_,
        drafter_authoritative_,
        model_output_lens_,
        model_state_positions_,
        model_input_tokens_,
        checkpoint_positions_,
        egress_seqs_,
        last_commit_tokens_,
        landing_stream_);
    check_cuda(
        cudaEventRecord(
            slot.event, reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaEventRecord");
    slot.in_flight = true;
    enqueue_egress_trigger(landing_stream_);
  }

  int64_t device_index_;
  int64_t num_seats_;
  int64_t num_draft_tokens_;
  int64_t tail_capacity_;
  int64_t pending_token_capacity_;
  void* landing_stream_;
  int64_t* versions_;
  int64_t* publish_seqs_;
  int64_t* request_epochs_;
  int64_t* prompt_lens_;
  int64_t* committed_lens_;
  int64_t* can_accept_prefix_lens_;
  int64_t* raw_tail_lens_;
  int64_t* consumable_tail_lens_;
  int64_t* pending_expected_lens_;
  int64_t* pending_expected_tokens_;
  int64_t* tail_tokens_;
  int64_t* last_op_seqs_;
  int64_t* error_codes_;
  int64_t* error_op_seqs_;
  int64_t* pending_prefix_fast_forward_cts_;
  int64_t* model_output_lens_;
  int64_t* model_state_positions_;
  int64_t* model_input_tokens_;
  int64_t* checkpoint_positions_;
  int64_t* egress_seqs_;
  int64_t* last_commit_tokens_;
  bool drafter_authoritative_;
  const bool strict_selection_;
  std::mutex binding_mu_;
  std::unordered_map<std::string, GpuDraftTailBindingCpp> request_bindings_;
  std::unordered_map<int64_t, SeatBindingCpp> seat_bindings_;
  std::deque<int64_t> free_seats_;
  int64_t next_request_epoch_ = 1;
  int64_t next_op_seq_ = 1;
  std::vector<StagingSlotCpp> staging_slots_;
  std::array<CommitHandoffSlotCpp, kCommitHandoffSlots>
      commit_handoff_slots_{};
  cudaEvent_t lifecycle_event_ = nullptr;
  cudaStream_t egress_stream_ = nullptr;
  std::array<EgressTriggerSlotCpp, kEgressTriggerSlots>
      egress_trigger_slots_{};
  std::array<EgressSnapshotSlotCpp, kEgressSnapshotSlots>
      egress_snapshot_slots_{};
  std::mutex egress_mu_;
  std::mutex egress_notifier_mu_;
  std::function<void()> egress_notifier_;
  std::atomic<size_t> staging_slot_count_{0};
  std::mutex close_mu_;
  std::atomic<bool> closed_{false};
};

struct QueuedFrame {
  int32_t dst_rank = -1;
  std::string frame;
  std::vector<std::string> changed_request_ids;
  std::shared_ptr<DraftControlBatchCpp> control_batch;
  int64_t enqueue_ns = 0;
  int64_t num_tokens = 0;
};

void ensure_outbound_queue_capacity(
    size_t current_frames,
    size_t current_bytes,
    size_t additional_frames,
    size_t additional_bytes,
    const char* owner_name) {
  bool frame_capacity_exceeded = current_frames > kMaxPendingZmqFrames ||
      additional_frames > kMaxPendingZmqFrames - current_frames;
  bool byte_capacity_exceeded = current_bytes > kMaxPendingZmqBytes ||
      additional_bytes > kMaxPendingZmqBytes - current_bytes;
  if (frame_capacity_exceeded || byte_capacity_exceeded) {
    throw std::runtime_error(
        std::string(owner_name) +
        " outbound queue reached its hard capacity while the peer was not "
        "draining: current_frames=" + std::to_string(current_frames) +
        " additional_frames=" + std::to_string(additional_frames) +
        " current_bytes=" + std::to_string(current_bytes) +
        " additional_bytes=" + std::to_string(additional_bytes));
  }
}

}  // namespace

class DecoupledSpecDraftProxyThread {
 public:
  DecoupledSpecDraftProxyThread(
      int64_t verifier_rank,
      const std::string& bind_endpoint,
      const py::sequence& drafter_peer_rows,
      std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer,
      uintptr_t external_context = 0,
      bool mock_profile = false,
      bool python_transport = false)
      : python_transport_(python_transport),
        verifier_rank_(checked_transport_rank(verifier_rank, "Verifier rank")),
        zmq_(python_transport ? nullptr : std::make_unique<ZmqContextOwner>(external_context)),
        mock_profile_(mock_profile) {
    if (gpu_tail_buffer == nullptr) {
      throw std::runtime_error("Draft proxy requires a GPU draft-tail owner");
    }
    if (bind_endpoint.rfind("tcp://", 0) != 0 &&
        bind_endpoint.rfind("ipc://", 0) != 0 &&
        (external_context == 0 || bind_endpoint.rfind("inproc://", 0) != 0)) {
      throw std::runtime_error(
          "CppDraftProxyThread requires TCP/IPC, or inproc with a shared context");
    }
    auto drafter_peers = build_ranked_peer_endpoints(drafter_peer_rows);
    if (external_context == 0) {
      for (const auto& peer : drafter_peers) {
        if (peer.endpoint.rfind("inproc://", 0) == 0) {
          throw std::runtime_error(
              "CppDraftProxyThread requires a shared context for inproc peers");
        }
      }
    }
    gpu_tail_buffer_ = std::move(gpu_tail_buffer);
    if (!python_transport_) {
      result_recv_socket_ = zmq_->api().socket(zmq_->ctx(), kZmqPull);
      zmq_->api().configure_socket(result_recv_socket_, kZmqPull);
      zmq_->api().bind(result_recv_socket_, bind_endpoint);
    }
    for (const auto& peer : drafter_peers) {
      void* socket = nullptr;
      if (!python_transport_) {
        socket = zmq_->api().socket(zmq_->ctx(), kZmqPush);
        zmq_->api().configure_socket(socket, kZmqPush);
      }
      control_send_sockets_[peer.rank] = socket;
      control_peer_endpoints_[peer.rank] = peer.endpoint;
      clock_calibration_states_.emplace(peer.rank, ClockCalibrationStateCpp{});
      peer_transport_latencies_.try_emplace(peer.rank);
      clock_calibration_peer_order_.push_back(peer.rank);
      clock_probe_attempts_remaining_[peer.rank] = 0;
    }
  }

  ~DecoupledSpecDraftProxyThread() { close(); }

  py::dict take_transport_metrics() {
    std::lock_guard<std::mutex> metrics_guard(transport_metrics_mu_);
    py::dict out;
    out["num_draft_result_frames"] = std::exchange(num_result_frames_, 0);
    out["num_draft_result_tokens"] = std::exchange(num_result_tokens_, 0);
    out["draft_receive_to_gpu_publish_enqueue_latency_us"] =
        receive_to_publish_enqueue_latency_.take();
    out["draft_gpu_publish_completion_latency_us"] = py::none();
    out["gpu_publish_staging_slots_max"] =
        std::exchange(gpu_publish_staging_slots_max_, 0);

    uint64_t num_valid_peers = 0;
    uint64_t num_invalid_peers = 0;
    int64_t max_current_error_bound_ns = 0;
    int64_t max_recorded_error_bound_ns = 0;
    const int64_t current_ns = now_ns();
    LatencyHistogramSnapshotCpp healthy_one_way;
    LatencyHistogramSnapshotCpp healthy_result_ready_to_receive;
    {
      std::lock_guard<std::mutex> guard(clock_calibration_mu_);
      for (auto& item : peer_transport_latencies_) {
        const int32_t peer_rank = item.first;
        auto& latency = item.second;
        const auto state_it = clock_calibration_states_.find(peer_rank);
        const bool peer_valid = state_it != clock_calibration_states_.end() &&
            state_it->second.valid &&
            current_ns - state_it->second.last_update_ns <=
                kClockCalibrationStaleNs;
        auto one_way = latency.one_way_latency.take_snapshot();
        auto result_ready_to_receive =
            latency.result_ready_to_receive_latency.take_snapshot();
        // Samples were gated by a valid, matching calibration epoch when they
        // were recorded. Do not discard them merely because a new calibration
        // is in progress (or the peer became stale) at window-drain time.
        healthy_one_way.merge(one_way);
        healthy_result_ready_to_receive.merge(result_ready_to_receive);
        if (one_way.count > 0 || result_ready_to_receive.count > 0) {
          max_recorded_error_bound_ns = std::max(
              max_recorded_error_bound_ns,
              latency.max_recorded_error_bound_ns);
        }
        latency.max_recorded_error_bound_ns = 0;
        if (peer_valid) {
          ++num_valid_peers;
          max_current_error_bound_ns = std::max(
              max_current_error_bound_ns, state_it->second.error_bound_ns);
        } else {
          ++num_invalid_peers;
        }
      }
    }
    const bool clock_sync_valid =
        num_valid_peers > 0 && num_invalid_peers == 0;
    out["clock_sync_valid"] = clock_sync_valid;
    out["num_clock_sync_valid_peers"] = num_valid_peers;
    out["num_clock_sync_invalid_peers"] = num_invalid_peers;
    const bool has_recorded_calibrated_samples =
        healthy_one_way.count > 0 || healthy_result_ready_to_receive.count > 0;
    if (has_recorded_calibrated_samples || num_valid_peers > 0) {
      const int64_t reported_error_bound_ns = has_recorded_calibrated_samples
          ? max_recorded_error_bound_ns
          : max_current_error_bound_ns;
      out["clock_error_bound_us"] =
          static_cast<double>(reported_error_bound_ns) / 1000.0;
      out["draft_transport_one_way_latency_us"] =
          healthy_one_way.to_python();
      out["draft_result_ready_to_receive_latency_us"] =
          healthy_result_ready_to_receive.to_python();
    } else {
      out["clock_error_bound_us"] = py::none();
      out["draft_transport_one_way_latency_us"] = py::none();
      out["draft_result_ready_to_receive_latency_us"] = py::none();
    }
    return out;
  }


  void start() {
    check_thread_error();
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    if (python_transport_) {
      closed_.store(false);
      next_clock_calibration_ns_ = now_ns() + 1000LL * 1000 * 1000;
      return;
    }
    try {
      for (const auto& kv : control_send_sockets_) {
        zmq_->api().connect(kv.second, control_peer_endpoints_.at(kv.first));
      }
    } catch (...) {
      started_.store(false);
      throw;
    }
    closed_.store(false);
    next_clock_calibration_ns_ = now_ns() + 1000LL * 1000 * 1000;
    thread_ = std::thread([this] {
      set_current_thread_name(
          "dspec-vrfy-io", "decoupled-spec-verifier-daemon");
      run_guarded();
    });
  }

  void close() {
    closed_.store(true);
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (zmq_) {
      for (auto& kv : control_send_sockets_) zmq_->api().close_socket(kv.second);
    }
    control_send_sockets_.clear();
    if (result_recv_socket_) {
      zmq_->api().close_socket(result_recv_socket_);
      result_recv_socket_ = nullptr;
    }
    if (zmq_) zmq_->close_context();
  }

  void submit_control_batch_native(
      int64_t dst_drafter_rank,
      const py::sequence& sync_rows,
      const py::sequence& commit_rows,
      const py::sequence& close_rows,
      bool apply_local_verify_commits) {
    check_thread_error();
    auto batch = build_control_batch_native(
        dst_drafter_rank,
        sync_rows,
        commit_rows,
        close_rows);
    {
      py::gil_scoped_release release;
      submit_control_batch(
          std::move(batch), apply_local_verify_commits);
    }
  }

  // Python owns the socket loop; these operations share the native wire codec,
  // bounded FIFO and GPU landing semantics with the native daemon.
  bool send_pending(const py::function& send) {
    check_python_transport();
    py::gil_scoped_release release;
    // The sole consumer retains the front through retries; producers only append.
    QueuedFrame* queued;
    {
      std::lock_guard<std::mutex> guard(queue_mu_);
      if (send_queue_.empty()) return false;
      queued = &send_queue_.front();
    }
    if (!mock_profile_) {
      py::gil_scoped_acquire acquire;
      if (!send(queued->dst_rank, py::bytes(queued->frame)).cast<bool>()) return false;
    }
    std::lock_guard<std::mutex> guard(queue_mu_);
    pending_send_bytes_ -= send_queue_.front().frame.size();
    send_queue_.pop_front();
    return true;
  }

  void receive_frame(const py::bytes& bytes) {
    check_python_transport();
    const int64_t receive_ns = now_ns();
    std::string frame = bytes.cast<std::string>();
    py::gil_scoped_release release;
    const uint8_t kind = wire_frame_kind(frame);
    if (kind == kKindTailStreamBatch) {
      recv_tail_stream_batch(frame, receive_ns);
    } else if (kind == kKindClockCalibrationReply) {
      recv_clock_calibration_reply(frame, receive_ns);
    } else {
      throw std::runtime_error("Verifier received an unexpected frame kind");
    }
  }

  bool send_clock_probe(const py::function& send) {
    check_python_transport();
    py::gil_scoped_release release;
    if (mock_profile_) return false;
    return maybe_send_one_clock_calibration_probe([&](int32_t rank, const std::string& frame) {
      py::gil_scoped_acquire acquire;
      return send(rank, py::bytes(frame)).cast<bool>();
    });
  }

  void check_python_transport() {
    check_thread_error();
    if (!python_transport_ || !started_.load() || closed_.load()) {
      throw std::runtime_error("Python transport is not running");
    }
  }

 private:

  void submit_control_batch(
      DraftControlBatchCpp batch,
      bool apply_local_verify_commits = true) {
    if (control_send_sockets_.count(batch.dst_drafter_rank) == 0) {
      throw std::runtime_error("Missing control socket for dst_drafter_rank");
    }
    auto frame = encode_control_batch(batch);
    auto batch_owner = std::make_shared<DraftControlBatchCpp>(std::move(batch));
    {
      std::lock_guard<std::mutex> guard(queue_mu_);
      ensure_outbound_queue_capacity(
          send_queue_.size(),
          pending_send_bytes_,
          1,
          frame.size(),
          "Verifier control");
      if (!mock_profile_) {
        std::lock_guard<std::mutex> update_guard(gpu_update_mu_);
        if (apply_local_verify_commits) {
          gpu_tail_buffer_->apply_control_batch(*batch_owner);
        } else {
          DraftControlBatchCpp local_batch;
          local_batch.wire_metadata = batch_owner->wire_metadata;
          local_batch.dst_drafter_rank = batch_owner->dst_drafter_rank;
          local_batch.sync_messages = batch_owner->sync_messages;
          local_batch.close_messages = batch_owner->close_messages;
          if (!local_batch.sync_messages.empty() ||
              !local_batch.close_messages.empty()) {
            gpu_tail_buffer_->apply_control_batch(local_batch);
          }
        }
      }
      pending_send_bytes_ += frame.size();
      send_queue_.push_back(QueuedFrame{
          batch_owner->dst_drafter_rank,
          std::move(frame),
          {},
          std::move(batch_owner)});
    }
    queue_cv_.notify_one();
  }

  void run_guarded() {
    try {
      run();
    } catch (...) {
      record_thread_error(std::current_exception());
    }
  }

  void record_thread_error(std::exception_ptr error) {
    std::string message = "unknown C++ draft proxy thread error";
    try {
      if (error) std::rethrow_exception(error);
    } catch (const std::exception& exc) {
      message = exc.what();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> guard(error_mu_);
      thread_error_ = std::move(message);
    }
    closed_.store(true);
    queue_cv_.notify_all();
  }

  void check_thread_error() {
    std::lock_guard<std::mutex> guard(error_mu_);
    if (!thread_error_.empty()) {
      throw std::runtime_error("CppDraftProxyThread failed: " + thread_error_);
    }
  }

  void run() {
    while (!closed_.load()) {
      bool did_work = false;
      while (true) {
        QueuedFrame queued;
        {
          std::lock_guard<std::mutex> guard(queue_mu_);
          if (send_queue_.empty()) break;
          size_t frame_bytes = send_queue_.front().frame.size();
          queued = std::move(send_queue_.front());
          send_queue_.pop_front();
          pending_send_bytes_ -= frame_bytes;
        }
        if (!send_control_batch(queued)) return;
        did_work = true;
      }
      try {
        if (zmq_->api().poll_in(result_recv_socket_, 1)) {
          std::string frame;
          bool received = false;
          {
            RECORD_USER_SCOPE(
                "sglang.decoupled_spec.verifier_daemon.zmq_recv");
            NvtxScopedRange nvtx_range(
                "sglang.decoupled_spec.verifier_daemon.zmq_recv");
            received = zmq_->api().recv_nonblock(result_recv_socket_, frame);
          }
          if (received) {
            const int64_t receive_ns = now_ns();
            const uint8_t kind = wire_frame_kind(frame);
            if (kind == kKindTailStreamBatch) {
              recv_tail_stream_batch(frame, receive_ns);
            } else if (kind == kKindClockCalibrationReply) {
              recv_clock_calibration_reply(frame, receive_ns);
            } else {
              throw std::runtime_error(
                  "Verifier result socket received an unexpected frame kind");
            }
            did_work = true;
          }
        }
      } catch (...) {
        if (!closed_.load()) throw;
      }
      // Calibration is strictly best-effort observability. Attempt at most one
      // probe after reliable business traffic in each I/O-loop iteration so a
      // disconnected peer cannot stall control or result progress elsewhere.
      if (!mock_profile_) {
        did_work = maybe_send_one_clock_calibration_probe() || did_work;
      }
      if (!did_work) {
        std::unique_lock<std::mutex> lock(queue_mu_);
        queue_cv_.wait_for(lock, std::chrono::microseconds(500), [&] {
          return closed_.load() || !send_queue_.empty();
        });
      }
    }
  }

  bool send_control_batch(const QueuedFrame& queued) {
    RECORD_USER_SCOPE("sglang.decoupled_spec.verifier_daemon.control_tx");
    NvtxScopedRange nvtx_range(
        "sglang.decoupled_spec.verifier_daemon.control_tx");
    if (mock_profile_) return true;
    auto it = control_send_sockets_.find(queued.dst_rank);
    if (it == control_send_sockets_.end()) {
      throw std::runtime_error("Missing control socket for dst_drafter_rank");
    }
    return send_frame_until_ready_or_closed(
        zmq_->api(), it->second, queued.frame, closed_);
  }

  void recv_tail_stream_batch(
      const std::string& frame,
      int64_t receive_ns) {
    RECORD_USER_SCOPE("sglang.decoupled_spec.verifier_daemon.tail_rx");
    NvtxScopedRange nvtx_range(
        "sglang.decoupled_spec.verifier_daemon.tail_rx");
    auto batch = parse_tail_stream_batch(frame);
    for (const auto& output : batch.outputs) {
      if (output.dst_verifier_rank != verifier_rank_) {
        throw std::runtime_error("Draft proxy received a tail stream batch for the wrong verifier");
      }
    }
    {
      std::lock_guard<std::mutex> update_guard(gpu_update_mu_);
      RECORD_USER_SCOPE(
          "sglang.decoupled_spec.verifier_daemon.gpu_publish_enqueue");
      NvtxScopedRange gpu_enqueue_nvtx_range(
          "sglang.decoupled_spec.verifier_daemon.gpu_publish_enqueue");
      gpu_tail_buffer_->append_draft_stream_batch(batch);
    }
    {
      std::lock_guard<std::mutex> metrics_guard(transport_metrics_mu_);
      receive_to_publish_enqueue_latency_.record_ns(now_ns() - receive_ns);
      gpu_publish_staging_slots_max_ = std::max<uint64_t>(
          gpu_publish_staging_slots_max_,
          gpu_tail_buffer_->staging_slot_count());

      ++num_result_frames_;
      for (const auto& output : batch.outputs) {
        if (!output.is_commit_echo) {
          num_result_tokens_ += output.tokens.size();
        }
      }
      record_calibrated_draft_latencies(batch, receive_ns);
    }
  }

  bool maybe_send_one_clock_calibration_probe(
      const std::function<bool(int32_t, const std::string&)>& send = {}) {
    const int64_t current_ns = now_ns();
    if (current_ns >= next_clock_calibration_ns_) {
      next_clock_calibration_ns_ =
          current_ns + kClockCalibrationPeriodNs;
      const int64_t epoch = next_clock_calibration_epoch_++;
      pending_clock_probes_.clear();
      {
        std::lock_guard<std::mutex> guard(clock_calibration_mu_);
        for (auto& item : clock_calibration_states_) {
          auto& state = item.second;
          state.epoch = epoch;
          state.num_samples = 0;
          state.best_rtt_ns = std::numeric_limits<int64_t>::max();
          state.valid = false;
        }
      }
      for (auto& item : clock_probe_attempts_remaining_) {
        item.second = kClockCalibrationSampleCount;
      }
      next_clock_calibration_peer_index_ = 0;
    }

    const size_t num_peers = clock_calibration_peer_order_.size();
    for (size_t scanned = 0; scanned < num_peers; ++scanned) {
      const int32_t drafter_rank = clock_calibration_peer_order_[
          next_clock_calibration_peer_index_];
      next_clock_calibration_peer_index_ =
          (next_clock_calibration_peer_index_ + 1) % num_peers;
      auto attempts_it = clock_probe_attempts_remaining_.find(drafter_rank);
      if (attempts_it == clock_probe_attempts_remaining_.end() ||
          attempts_it->second <= 0) {
        continue;
      }
      --attempts_it->second;

      const auto state_it = clock_calibration_states_.find(drafter_rank);
      const auto socket_it = control_send_sockets_.find(drafter_rank);
      if (state_it == clock_calibration_states_.end() ||
          socket_it == control_send_sockets_.end()) {
        continue;
      }
      ClockCalibrationProbeCpp probe;
      probe.src_verifier_rank = verifier_rank_;
      probe.dst_drafter_rank = drafter_rank;
      probe.probe_seq = next_clock_probe_seq_++;
      probe.calibration_epoch = state_it->second.epoch;
      probe.verifier_send_ns = now_ns();
      auto frame = encode_clock_calibration_probe(probe);
      if (!(send ? send(drafter_rank, frame)
                 : zmq_->api().send_nonblock(socket_it->second, frame))) {
        return false;
      }
      pending_clock_probes_[probe.probe_seq] = {
          drafter_rank, probe.calibration_epoch, probe.verifier_send_ns};
      return true;
    }
    return false;
  }

  void recv_clock_calibration_reply(
      const std::string& frame,
      int64_t verifier_receive_ns) {
    auto reply = parse_clock_calibration_reply(frame);
    if (reply.dst_verifier_rank != verifier_rank_) {
      throw std::runtime_error(
          "Clock calibration reply targets a different verifier");
    }
    auto pending_it = pending_clock_probes_.find(reply.probe_seq);
    if (pending_it == pending_clock_probes_.end()) return;
    const auto pending = pending_it->second;
    pending_clock_probes_.erase(pending_it);
    if (reply.src_drafter_rank != pending.drafter_rank ||
        reply.calibration_epoch != pending.epoch ||
        reply.verifier_send_ns != pending.verifier_send_ns) {
      return;
    }
    const int64_t drafter_service_ns =
        reply.drafter_send_ns - reply.drafter_receive_ns;
    const int64_t rtt_ns =
        (verifier_receive_ns - reply.verifier_send_ns) - drafter_service_ns;
    if (rtt_ns < 0) return;
    const int64_t offset_ns =
        ((reply.drafter_receive_ns - reply.verifier_send_ns) +
         (reply.drafter_send_ns - verifier_receive_ns)) /
        2;
    std::lock_guard<std::mutex> guard(clock_calibration_mu_);
    auto& state = clock_calibration_states_[reply.src_drafter_rank];
    if (state.epoch != reply.calibration_epoch) return;
    ++state.num_samples;
    if (rtt_ns < state.best_rtt_ns) {
      state.best_rtt_ns = rtt_ns;
      state.offset_drafter_minus_verifier_ns = offset_ns;
      state.error_bound_ns = rtt_ns / 2;
    }
    if (state.num_samples >= kClockCalibrationSampleCount) {
      state.valid = state.error_bound_ns <= kClockMaxErrorBoundNs;
      state.last_update_ns = verifier_receive_ns;
    }
  }

  void record_calibrated_draft_latencies(
      const DraftTailStreamOutputBatchCpp& batch,
      int64_t verifier_receive_ns) {
    if (batch.outputs.empty() || batch.wire_metadata.send_start_ns <= 0 ||
        batch.wire_metadata.calibration_epoch <= 0) {
      return;
    }
    const int32_t drafter_rank = batch.outputs.front().src_drafter_rank;
    std::lock_guard<std::mutex> guard(clock_calibration_mu_);
    auto state_it = clock_calibration_states_.find(drafter_rank);
    auto latency_it = peer_transport_latencies_.find(drafter_rank);
    if (state_it == clock_calibration_states_.end() ||
        latency_it == peer_transport_latencies_.end()) {
      return;
    }
    const auto& state = state_it->second;
    if (!state.valid ||
        state.epoch != batch.wire_metadata.calibration_epoch ||
        verifier_receive_ns - state.last_update_ns >
            kClockCalibrationStaleNs) {
      return;
    }
    const int64_t calibrated_send_ns =
        batch.wire_metadata.send_start_ns -
        state.offset_drafter_minus_verifier_ns;
    bool recorded_sample = false;
    const int64_t one_way_latency_ns =
        verifier_receive_ns - calibrated_send_ns;
    if (one_way_latency_ns >= 0) {
      latency_it->second.one_way_latency.record_ns(one_way_latency_ns);
      recorded_sample = true;
    }
    if (batch.wire_metadata.result_ready_ns > 0) {
      const int64_t calibrated_result_ready_ns =
          batch.wire_metadata.result_ready_ns -
          state.offset_drafter_minus_verifier_ns;
      const int64_t ready_to_receive_latency_ns =
          verifier_receive_ns - calibrated_result_ready_ns;
      if (ready_to_receive_latency_ns >= 0) {
        latency_it->second.result_ready_to_receive_latency.record_ns(
            ready_to_receive_latency_ns);
        recorded_sample = true;
      }
    }
    if (recorded_sample) {
      latency_it->second.max_recorded_error_bound_ns = std::max(
          latency_it->second.max_recorded_error_bound_ns,
          state.error_bound_ns);
    }
  }

  static std::vector<std::string> control_request_ids(
      const DraftControlBatchCpp& batch) {
    std::vector<std::string> request_ids;
    request_ids.reserve(
        batch.sync_messages.size() + batch.verify_commit_messages.size() +
        batch.close_messages.size());
    std::unordered_set<std::string> seen;
    for (const auto& message : batch.sync_messages) {
      if (seen.insert(message.request_id).second) {
        request_ids.push_back(message.request_id);
      }
    }
    for (const auto& message : batch.verify_commit_messages) {
      if (seen.insert(message.request_id).second) {
        request_ids.push_back(message.request_id);
      }
    }
    for (const auto& message : batch.close_messages) {
      if (seen.insert(message.request_id).second) {
        request_ids.push_back(message.request_id);
      }
    }
    return request_ids;
  }

  const bool python_transport_;
  int32_t verifier_rank_;
  std::unique_ptr<ZmqContextOwner> zmq_;
  std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer_;
  void* result_recv_socket_ = nullptr;
  std::map<int32_t, void*> control_send_sockets_;
  std::map<int32_t, std::string> control_peer_endpoints_;
  std::deque<QueuedFrame> send_queue_;
  size_t pending_send_bytes_ = 0;
  std::mutex queue_mu_;
  std::mutex gpu_update_mu_;
  std::condition_variable queue_cv_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> started_{false};
  std::mutex transport_metrics_mu_;
  uint64_t num_result_frames_ = 0;
  uint64_t num_result_tokens_ = 0;
  FixedLatencyHistogram receive_to_publish_enqueue_latency_;
  uint64_t gpu_publish_staging_slots_max_ = 0;
  std::mutex clock_calibration_mu_;
  std::map<int32_t, ClockCalibrationStateCpp> clock_calibration_states_;
  std::map<int32_t, PeerTransportLatencyCpp> peer_transport_latencies_;
  std::unordered_map<int64_t, PendingClockProbeCpp> pending_clock_probes_;
  std::vector<int32_t> clock_calibration_peer_order_;
  std::map<int32_t, int64_t> clock_probe_attempts_remaining_;
  size_t next_clock_calibration_peer_index_ = 0;
  int64_t next_clock_probe_seq_ = 1;
  int64_t next_clock_calibration_epoch_ = 1;
  int64_t next_clock_calibration_ns_ = 0;
  std::thread thread_;
  std::mutex error_mu_;
  std::string thread_error_;
  bool mock_profile_ = false;
};

class DecoupledSpecTokenSyncThread {
 public:
  DecoupledSpecTokenSyncThread(
      int64_t drafter_rank,
      const std::string& bind_endpoint,
      const py::sequence& verifier_peer_rows,
      uintptr_t external_context = 0,
      std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer = nullptr,
      bool python_transport = false,
      int64_t schedule_ahead_limit = -1)
      : python_transport_(python_transport),
        drafter_rank_(checked_transport_rank(drafter_rank, "Drafter rank")),
        zmq_(python_transport ? nullptr : std::make_unique<ZmqContextOwner>(external_context)),
        gpu_tail_buffer_(std::move(gpu_tail_buffer)),
        schedule_ahead_limit_(schedule_ahead_limit) {
    if (gpu_tail_buffer_ == nullptr) {
      throw std::runtime_error("Drafter requires a GPU tail buffer");
    }
    const int64_t capacity = gpu_tail_buffer_->tail_capacity();
    if (schedule_ahead_limit_ == -1) schedule_ahead_limit_ = capacity - 1;
    if (schedule_ahead_limit_ <= 0 || schedule_ahead_limit_ > capacity) {
      throw std::runtime_error(
          "Drafter scheduling limit must be within its tail capacity");
    }

    if (bind_endpoint.rfind("tcp://", 0) != 0 &&
        bind_endpoint.rfind("ipc://", 0) != 0 &&
        (external_context == 0 || bind_endpoint.rfind("inproc://", 0) != 0)) {
      throw std::runtime_error(
          "CppTokenSyncThread requires TCP/IPC, or inproc with a shared context");
    }
    auto verifier_peers = build_ranked_peer_endpoints(verifier_peer_rows);
    if (external_context == 0) {
      for (const auto& peer : verifier_peers) {
        if (peer.endpoint.rfind("inproc://", 0) == 0) {
          throw std::runtime_error(
              "CppTokenSyncThread requires a shared context for inproc peers");
        }
      }
    }
    if (!python_transport_) {
      control_recv_socket_ = zmq_->api().socket(zmq_->ctx(), kZmqPull);
      zmq_->api().configure_socket(control_recv_socket_, kZmqPull);
      zmq_->api().bind(control_recv_socket_, bind_endpoint);
    }
    for (const auto& peer : verifier_peers) {
      void* socket = nullptr;
      if (!python_transport_) {
        socket = zmq_->api().socket(zmq_->ctx(), kZmqPush);
        zmq_->api().configure_socket(socket, kZmqPush);
      }
      result_send_sockets_[peer.rank] = socket;
      result_peer_endpoints_[peer.rank] = peer.endpoint;
    }
    // The buffer can be called by a forward thread while transport teardown
    // clears this callback. Capture shared wake state rather than a raw
    // TokenSyncThread pointer so an already-copied callback cannot race the
    // owner's destructor.
    auto wake_state = egress_wake_state_;
    gpu_tail_buffer_->set_egress_notifier([wake_state = std::move(wake_state)] {
      wake_state->notify_seq.fetch_add(1, std::memory_order_release);
      wake_state->queue_cv.notify_one();
    });

  }

  ~DecoupledSpecTokenSyncThread() { close(); }

  py::tuple lookup_gpu_binding(
      const std::string& request_id,
      int64_t src_verifier_rank) {
    return gpu_tail_buffer_->lookup_binding(request_id, src_verifier_rank);
  }

  py::dict take_transport_metrics() {
    std::lock_guard<std::mutex> metrics_guard(transport_metrics_mu_);
    py::dict out;
    out["num_draft_result_frames"] = std::exchange(num_result_frames_, 0);
    out["num_draft_result_tokens"] = std::exchange(num_result_tokens_, 0);
    out["draft_send_queue_latency_us"] = send_queue_latency_.take();
    out["draft_send_queue_depth_max"] =
        std::exchange(send_queue_depth_max_, 0);
    return out;
  }

  void start() {
    check_thread_error();
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    if (python_transport_) {
      closed_.store(false);
      return;
    }
    try {
      for (const auto& kv : result_send_sockets_) {
        zmq_->api().connect(kv.second, result_peer_endpoints_.at(kv.first));
      }
    } catch (...) {
      started_.store(false);
      throw;
    }
    closed_.store(false);
    thread_ = std::thread([this] {
      set_current_thread_name(
          "dspec-draft-io", "decoupled-spec-drafter-daemon");
      run_guarded();
    });
  }

  void close() {
    gpu_tail_buffer_->clear_egress_notifier();

    closed_.store(true);
    egress_wake_state_->queue_cv.notify_all();
    if (thread_.joinable()) thread_.join();
    if (zmq_) {
      for (auto& kv : result_send_sockets_) zmq_->api().close_socket(kv.second);
    }
    result_send_sockets_.clear();
    if (control_recv_socket_) {
      zmq_->api().close_socket(control_recv_socket_);
      control_recv_socket_ = nullptr;
    }
    if (zmq_) zmq_->close_context();
  }

  void submit_draft_results_native(const py::sequence& rows) {
    check_thread_error();
    auto batch = build_tail_stream_batch_native(rows);
    if (batch.outputs.empty()) return;
    {
      py::gil_scoped_release release;
      submit_draft_results_batch(std::move(batch));
    }
  }

  int64_t pending_control_count() {
    check_thread_error();
    return data_plane_.pending_control_count();
  }

  bool wait_for_pending_control(int64_t timeout_us) {
    check_thread_error();
    return data_plane_.wait_for_pending_control(timeout_us);
  }

  py::tuple extract_lifecycle_controls_native() {
    check_thread_error();
    ReadyDraftControlsCpp ready;
    {
      py::gil_scoped_release release;
      ready = data_plane_.extract_lifecycle_controls();
    }
    return ready_controls_native_rows(ready);
  }

  py::list drain_gpu_progress_native() {
    check_thread_error();
    std::vector<GpuDrafterProgressCpp> progress_rows;
    {
      std::lock_guard<std::mutex> guard(gpu_progress_mu_);
      progress_rows.reserve(pending_gpu_progress_.size());
      for (auto& item : pending_gpu_progress_) {
        progress_rows.push_back(std::move(item.second));
      }
      pending_gpu_progress_.clear();
    }
    py::list out;
    for (const auto& progress : progress_rows) {
      out.append(py::make_tuple(
          progress.request_id,
          progress.src_verifier_rank,
          progress.request_epoch,
          progress.model_output_len,
          progress.committed_len,
          progress.raw_tail_len));
    }
    return out;
  }

  bool poll_gpu_egress() {
    check_python_transport();
    py::gil_scoped_release release;
    return drain_gpu_egress();
  }

  bool send_pending(const py::function& send) {
    check_python_transport();
    py::gil_scoped_release release;
    // The sole consumer retains the front through retries; producers only append.
    QueuedFrame* queued;
    {
      std::lock_guard<std::mutex> guard(queue_mu_);
      if (outgoing_results_.empty()) return false;
      queued = &outgoing_results_.front();
    }
    patch_wire_send_start_ns(queued->frame, now_ns());
    {
      py::gil_scoped_acquire acquire;
      if (!send(queued->dst_rank, py::bytes(queued->frame)).cast<bool>()) return false;
    }
    const int64_t enqueue_ns = queued->enqueue_ns;
    const int64_t num_tokens = queued->num_tokens;
    {
      std::lock_guard<std::mutex> guard(queue_mu_);
      pending_result_bytes_ -= outgoing_results_.front().frame.size();
      outgoing_results_.pop_front();
    }
    std::lock_guard<std::mutex> guard(transport_metrics_mu_);
    send_queue_latency_.record_ns(now_ns() - enqueue_ns);
    ++num_result_frames_;
    num_result_tokens_ += num_tokens;
    return true;
  }

  void receive_frame(const py::bytes& bytes, const py::function& send) {
    check_python_transport();
    const int64_t receive_ns = now_ns();
    std::string frame = bytes.cast<std::string>();
    py::gil_scoped_release release;
    receive_control_frame(frame, receive_ns, [&](int32_t rank, const std::string& reply) {
      py::gil_scoped_acquire acquire;
      return send(rank, py::bytes(reply)).cast<bool>();
    });
  }

  void check_python_transport() {
    check_thread_error();
    if (!python_transport_ || !started_.load() || closed_.load()) {
      throw std::runtime_error("Python transport is not running");
    }
  }

 private:
  void submit_draft_results_batch(DraftTailStreamOutputBatchCpp batch) {
    if (batch.outputs.empty()) return;
    const int64_t result_ready_ns = now_ns();
    std::map<int32_t, DraftTailStreamOutputBatchCpp> by_verifier;
    for (const auto& output : batch.outputs) {
      by_verifier[output.dst_verifier_rank].outputs.push_back(output);
    }
    std::vector<QueuedFrame> frames;
    frames.reserve(by_verifier.size());
    size_t frame_bytes = 0;
    for (auto& kv : by_verifier) {
      kv.second.wire_metadata.frame_seq =
          next_result_frame_seq_.fetch_add(1, std::memory_order_relaxed);
      kv.second.wire_metadata.result_ready_ns = result_ready_ns;
      {
        std::lock_guard<std::mutex> guard(calibration_epoch_mu_);
        auto epoch_it = latest_calibration_epoch_by_verifier_.find(kv.first);
        if (epoch_it != latest_calibration_epoch_by_verifier_.end()) {
          kv.second.wire_metadata.calibration_epoch = epoch_it->second;
        }
      }
      QueuedFrame queued;
      queued.dst_rank = kv.first;
      queued.frame = encode_tail_stream_batch(kv.second);
      queued.num_tokens = 0;
      for (const auto& output : kv.second.outputs) {
        if (!output.is_commit_echo) {
          queued.num_tokens += static_cast<int64_t>(output.tokens.size());
        }
      }
      frames.push_back(std::move(queued));
      frame_bytes += frames.back().frame.size();
    }

    {
      std::lock_guard<std::mutex> guard(queue_mu_);
      ensure_outbound_queue_capacity(
          outgoing_results_.size(),
          pending_result_bytes_,
          frames.size(),
          frame_bytes,
          "Drafter tail");
      for (auto& frame : frames) {
        frame.enqueue_ns = now_ns();
        pending_result_bytes_ += frame.frame.size();
        outgoing_results_.push_back(std::move(frame));
      }
      {
        std::lock_guard<std::mutex> metrics_guard(transport_metrics_mu_);
        send_queue_depth_max_ = std::max<uint64_t>(
            send_queue_depth_max_, outgoing_results_.size());
      }
    }
    egress_wake_state_->queue_cv.notify_one();
  }

  void run_guarded() {
    try {
      run();
    } catch (...) {
      record_thread_error(std::current_exception());
    }
  }

  void record_thread_error(std::exception_ptr error) {
    std::string message = "unknown C++ token sync thread error";
    try {
      if (error) std::rethrow_exception(error);
    } catch (const std::exception& exc) {
      message = exc.what();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> guard(error_mu_);
      thread_error_ = std::move(message);
    }
    closed_.store(true);
    egress_wake_state_->queue_cv.notify_all();
  }

  void check_thread_error() {
    std::lock_guard<std::mutex> guard(error_mu_);
    if (!thread_error_.empty()) {
      throw std::runtime_error("CppTokenSyncThread failed: " + thread_error_);
    }
  }

  void run() {
    while (!closed_.load()) {
      const uint64_t observed_gpu_notify =
          egress_wake_state_->notify_seq.load(std::memory_order_acquire);
      bool did_work = false;
      did_work = drain_gpu_egress() || did_work;
      did_work = drain_outgoing_results() || did_work;
      did_work = drain_control_socket() || did_work;
      if (!did_work) {
        std::unique_lock<std::mutex> lock(queue_mu_);
        const bool gpu_pending = gpu_tail_buffer_->has_pending_egress_work();
        const bool result_backpressured = !outgoing_results_.empty();
        const auto poll_interval =
            std::chrono::microseconds(
                gpu_pending || result_backpressured ? 50 : 500);
        egress_wake_state_->queue_cv.wait_for(lock, poll_interval, [&] {
          return closed_.load() ||
              (!result_backpressured && !outgoing_results_.empty()) ||
              egress_wake_state_->notify_seq.load(std::memory_order_acquire) !=
              observed_gpu_notify;
        });
      }
    }
  }

  bool drain_gpu_egress() {
    if (!gpu_tail_buffer_->has_pending_egress_work()) {
      return false;
    }
    RECORD_USER_SCOPE(
        "sglang.decoupled_spec.drafter_daemon.gpu_snapshot_d2h");
    NvtxScopedRange nvtx_range(
        "sglang.decoupled_spec.drafter_daemon.gpu_snapshot_d2h");
    auto snapshots = gpu_tail_buffer_->poll_ready_egress_snapshots();
    if (snapshots.empty()) return false;

    DraftTailStreamOutputBatchCpp output_batch;
    std::unordered_map<int64_t, GpuEgressPublicationStateCpp>
        staged_publications;
    std::map<std::string, GpuDrafterProgressCpp> staged_progress;
    for (const auto& snapshot : snapshots) {
      std::string request_id;
      int32_t src_verifier_rank = -1;
      int64_t initial_committed_len = -1;
      if (!gpu_tail_buffer_->lookup_egress_binding(
              snapshot.seat,
              snapshot.request_epoch,
              &request_id,
              &src_verifier_rank,
              &initial_committed_len)) {
        continue;
      }
      if (snapshot.error_code != 0) {
        throw std::runtime_error(
            "GPU drafter authoritative row latched error_code=" +
            std::to_string(snapshot.error_code) + " request_id=" +
            request_id + " request_epoch=" +
            std::to_string(snapshot.request_epoch));
      }
      if (snapshot.egress_seq < 0 || snapshot.committed_len < 0 ||
          snapshot.ack_ready_len < initial_committed_len ||
          snapshot.ack_ready_len > snapshot.committed_len ||
          snapshot.raw_tail_len < 0 ||
          snapshot.raw_tail_len > gpu_tail_buffer_->tail_capacity() ||
          snapshot.raw_tail_tokens.size() !=
              static_cast<size_t>(snapshot.raw_tail_len) ||
          snapshot.model_output_len !=
              snapshot.committed_len + snapshot.raw_tail_len ||
          (snapshot.raw_tail_len > 0 &&
           snapshot.ack_ready_len != snapshot.committed_len)) {
        throw std::runtime_error(
            "GPU drafter egress snapshot violated transcript metadata: "
            "request_id=" + request_id + " committed_len=" +
            std::to_string(snapshot.committed_len) + " model_output_len=" +
            std::to_string(snapshot.model_output_len) + " raw_tail_len=" +
            std::to_string(snapshot.raw_tail_len) + " ack_ready_len=" +
            std::to_string(snapshot.ack_ready_len));
      }

      auto staged_it = staged_publications.find(snapshot.seat);
      if (staged_it == staged_publications.end()) {
        auto current_it = gpu_egress_publications_.find(snapshot.seat);
        staged_it = staged_publications.emplace(
            snapshot.seat,
            current_it == gpu_egress_publications_.end()
                ? GpuEgressPublicationStateCpp{}
                : current_it->second).first;
      }
      auto& publication = staged_it->second;
      const bool epoch_changed =
          publication.request_epoch != snapshot.request_epoch;
      if (epoch_changed) {
        publication.request_epoch = snapshot.request_epoch;
        publication.last_egress_seq = 0;
        publication.last_committed_len = initial_committed_len;
        publication.pacing_initialized = false;
      }
      if (snapshot.egress_seq < publication.last_egress_seq) continue;
      if (!epoch_changed &&
          snapshot.egress_seq == publication.last_egress_seq) {
        continue;
      }
      if (snapshot.ack_ready_len < publication.last_committed_len) {
        throw std::runtime_error(
            "GPU drafter ACK-ready length regressed within one request epoch");
      }

      if (snapshot.ack_ready_len > publication.last_committed_len) {
        if (snapshot.ack_ready_len <= 0 || snapshot.last_commit_token < 0 ||
            snapshot.last_commit_token >
                std::numeric_limits<int32_t>::max()) {
          throw std::runtime_error(
              "GPU drafter cumulative ACK token is outside int32 range");
        }
        DraftTailStreamOutputCpp ack;
        ack.src_drafter_rank = drafter_rank_;
        ack.dst_verifier_rank = src_verifier_rank;
        ack.request_id = request_id;
        ack.base_committed_len = snapshot.ack_ready_len;
        ack.start_token_pos = snapshot.ack_ready_len - 1;
        ack.tokens = {static_cast<int32_t>(snapshot.last_commit_token)};
        ack.is_commit_echo = true;
        output_batch.outputs.push_back(std::move(ack));
      }

      // Forced-token replay makes an all-raw-match plus verifier bonus safe:
      // the bonus is queued until the drafter materializes each predecessor
      // checkpoint. Publish the complete bounded raw suffix so withholding a
      // token does not artificially reduce valid draft length.
      const int64_t publish_tail_len = snapshot.raw_tail_len;
      if (publish_tail_len > 0) {
        DraftTailStreamOutputCpp tail;
        tail.src_drafter_rank = drafter_rank_;
        tail.dst_verifier_rank = src_verifier_rank;
        tail.request_id = request_id;
        tail.base_committed_len = snapshot.committed_len;
        tail.start_token_pos = snapshot.committed_len;
        tail.tokens.reserve(static_cast<size_t>(publish_tail_len));
        for (int64_t index = 0; index < publish_tail_len; ++index) {
          const int64_t token = snapshot.raw_tail_tokens[index];
          if (token < 0 ||
              token > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(
                "GPU drafter retained-tail token is outside int32 range");
          }
          tail.tokens.push_back(static_cast<int32_t>(token));
        }
        output_batch.outputs.push_back(std::move(tail));
      }

      publication.last_egress_seq = snapshot.egress_seq;
      publication.last_committed_len = snapshot.ack_ready_len;
      // Match the scheduler's in-flight reservation in either scheduling mode.
      const bool can_schedule = snapshot.raw_tail_len < schedule_ahead_limit_;
      if (!publication.pacing_initialized ||
          publication.last_can_schedule != can_schedule) {
        publication.pacing_initialized = true;
        publication.last_can_schedule = can_schedule;
        GpuDrafterProgressCpp progress;
        progress.request_id = request_id;
        progress.src_verifier_rank = src_verifier_rank;
        progress.request_epoch = snapshot.request_epoch;
        progress.model_output_len = snapshot.model_output_len;
        progress.committed_len = snapshot.committed_len;
        progress.raw_tail_len = snapshot.raw_tail_len;
        staged_progress[
            DraftReqKeyCpp{src_verifier_rank, request_id}.map_key()] =
            std::move(progress);
      }
    }
    if (!output_batch.outputs.empty()) {
      RECORD_USER_SCOPE(
          "sglang.decoupled_spec.drafter_daemon.gpu_egress_encode");
      NvtxScopedRange encode_nvtx_range(
          "sglang.decoupled_spec.drafter_daemon.gpu_egress_encode");
      submit_draft_results_batch(std::move(output_batch));
    }
    // Publication/progress cursors become visible only after every required
    // wire frame was admitted to the bounded FIFO.
    for (auto& item : staged_publications) {
      gpu_egress_publications_[item.first] = std::move(item.second);
    }
    if (!staged_progress.empty()) {
      {
        std::lock_guard<std::mutex> guard(gpu_progress_mu_);
        for (auto& item : staged_progress) {
          pending_gpu_progress_[item.first] = std::move(item.second);
        }
      }
      data_plane_.notify_external_progress();
    }
    return true;
  }

  bool drain_outgoing_results() {
    // Attempt at most one queue-front frame per daemon cycle. Leaving a
    // backpressured frame in place preserves FIFO, byte accounting and its
    // original enqueue timestamp while allowing control_rx/GPU egress to make
    // progress before the next short retry.
    std::lock_guard<std::mutex> guard(queue_mu_);
    if (outgoing_results_.empty()) return false;
    auto& queued = outgoing_results_.front();
    if (!send_draft_results(queued)) return false;
    pending_result_bytes_ -= queued.frame.size();
    outgoing_results_.pop_front();
    return true;
  }

  bool drain_control_socket() {
    bool did_work = false;
    while (!closed_.load()) {
      std::string frame;
      bool received = zmq_->api().recv_nonblock(control_recv_socket_, frame);
      if (!received) break;
      const int64_t receive_ns = now_ns();
      receive_control_frame(frame, receive_ns);
      did_work = true;
    }
    return did_work;
  }

  void receive_control_frame(
      const std::string& frame, int64_t receive_ns,
      const std::function<bool(int32_t, const std::string&)>& send = {}) {
    const uint8_t kind = wire_frame_kind(frame);
    if (kind == kKindClockCalibrationProbe) {
      auto probe = parse_clock_calibration_probe(frame);
      if (probe.dst_drafter_rank != drafter_rank_) {
        throw std::runtime_error(
            "Clock calibration probe targets a different drafter");
      }
      auto socket_it = result_send_sockets_.find(probe.src_verifier_rank);
      if (socket_it == result_send_sockets_.end()) {
        throw std::runtime_error(
            "Clock calibration probe has no verifier result socket");
      }
      {
        std::lock_guard<std::mutex> guard(calibration_epoch_mu_);
        latest_calibration_epoch_by_verifier_[probe.src_verifier_rank] =
            probe.calibration_epoch;
      }
      ClockCalibrationReplyCpp reply;
      reply.src_drafter_rank = drafter_rank_;
      reply.dst_verifier_rank = probe.src_verifier_rank;
      reply.probe_seq = probe.probe_seq;
      reply.verifier_send_ns = probe.verifier_send_ns;
      reply.drafter_receive_ns = receive_ns;
      reply.drafter_send_ns = now_ns();
      reply.calibration_epoch = probe.calibration_epoch;
      auto reply_frame = encode_clock_calibration_reply(reply);
      // This reply is optional telemetry. Backpressure drops the sample
      // instead of delaying draft-result sends or correctness controls.
      if (send) {
        send(probe.src_verifier_rank, reply_frame);
      } else {
        zmq_->api().send_nonblock(socket_it->second, reply_frame);
      }
      return;
    }
    if (kind != kKindControlBatch) {
      throw std::runtime_error(
          "Drafter control socket received an unexpected frame kind");
    }
    RECORD_USER_SCOPE("sglang.decoupled_spec.drafter_daemon.control_rx");
    NvtxScopedRange nvtx_range(
        "sglang.decoupled_spec.drafter_daemon.control_rx");
    auto batch = parse_control_batch(frame);
    if (batch.dst_drafter_rank != drafter_rank_) {
      throw std::runtime_error(
          "Draft control batch targets a different drafter");
    }
    // Network ingress publishes controls directly to the authoritative
    // GPU transcript. The CPU inbox below owns lifecycle only; verifier
    // commit tokens never enter a production token shadow.
    gpu_tail_buffer_->apply_control_batch(batch, true);
    if (!batch.verify_commit_messages.empty()) {
      data_plane_.notify_external_progress();
    }

    {
      std::lock_guard<std::mutex> progress_guard(gpu_progress_mu_);
      for (const auto& close : batch.close_messages) {
        pending_gpu_progress_.erase(close.draft_key().map_key());
      }
    }
    if (!batch.sync_messages.empty() || !batch.close_messages.empty()) {
      data_plane_.add_control_batch(batch);
    }
  }

  bool send_draft_results(QueuedFrame& queued) {
    RECORD_USER_SCOPE("sglang.decoupled_spec.drafter_daemon.draft_tx");
    NvtxScopedRange nvtx_range(
        "sglang.decoupled_spec.drafter_daemon.draft_tx");
    auto it = result_send_sockets_.find(queued.dst_rank);
    if (it == result_send_sockets_.end()) {
      throw std::runtime_error("Missing result socket for dst_verifier_rank");
    }
    const bool sent =
        try_send_draft_frame(zmq_->api(), it->second, queued.frame);
    if (sent) {
      std::lock_guard<std::mutex> metrics_guard(transport_metrics_mu_);
      send_queue_latency_.record_ns(now_ns() - queued.enqueue_ns);
      ++num_result_frames_;
      num_result_tokens_ += queued.num_tokens;
    }
    return sent;
  }

  struct GpuEgressPublicationStateCpp {
    int64_t request_epoch = -1;
    int64_t last_egress_seq = -1;
    int64_t last_committed_len = -1;
    bool pacing_initialized = false;
    bool last_can_schedule = true;
  };

  const bool python_transport_;
  int32_t drafter_rank_;
  std::unique_ptr<ZmqContextOwner> zmq_;
  void* control_recv_socket_ = nullptr;
  std::map<int32_t, void*> result_send_sockets_;
  std::map<int32_t, std::string> result_peer_endpoints_;
  std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer_;
  int64_t schedule_ahead_limit_;
  DrafterDataPlaneCore data_plane_;
  std::deque<QueuedFrame> outgoing_results_;
  size_t pending_result_bytes_ = 0;
  std::mutex queue_mu_;
  std::shared_ptr<GpuDrafterEgressWakeStateCpp> egress_wake_state_ =
      std::make_shared<GpuDrafterEgressWakeStateCpp>();
  std::unordered_map<int64_t, GpuEgressPublicationStateCpp>
      gpu_egress_publications_;
  std::map<std::string, GpuDrafterProgressCpp> pending_gpu_progress_;
  std::mutex gpu_progress_mu_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> started_{false};
  std::atomic<int64_t> next_result_frame_seq_{1};
  std::mutex transport_metrics_mu_;
  uint64_t num_result_frames_ = 0;
  uint64_t num_result_tokens_ = 0;
  FixedLatencyHistogram send_queue_latency_;
  uint64_t send_queue_depth_max_ = 0;
  std::mutex calibration_epoch_mu_;
  std::map<int32_t, int64_t> latest_calibration_epoch_by_verifier_;
  std::thread thread_;
  std::mutex error_mu_;
  std::string thread_error_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;

  py::class_<GpuDraftTailBufferCore, std::shared_ptr<GpuDraftTailBufferCore>>(
      m, "GpuDraftTailBuffer")
      .def(
          py::init<
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              bool,
              bool>(),
          py::arg("device_index"),
          py::arg("num_seats"),
          py::arg("num_draft_tokens"),
          py::arg("pending_token_capacity"),
          py::arg("landing_stream"),
          py::arg("versions"),
          py::arg("publish_seqs"),
          py::arg("request_epochs"),
          py::arg("prompt_lens"),
          py::arg("committed_lens"),
          py::arg("can_accept_prefix_lens"),
          py::arg("raw_tail_lens"),
          py::arg("consumable_tail_lens"),
          py::arg("pending_expected_lens"),
          py::arg("pending_expected_tokens"),
          py::arg("tail_tokens"),
          py::arg("last_op_seqs"),
          py::arg("error_codes"),
          py::arg("error_op_seqs"),
          py::arg("pending_prefix_fast_forward_cts"),
          py::arg("model_output_lens"),
          py::arg("model_state_positions"),
          py::arg("model_input_tokens"),
          py::arg("checkpoint_positions"),
          py::arg("egress_seqs"),
          py::arg("last_commit_tokens"),
          py::arg("drafter_authoritative"),
          py::arg("allow_partial"))
      .def(
          "bind_request",
          &GpuDraftTailBufferCore::bind_request,
          py::arg("request_id"),
          py::arg("gpu_seat"),
          py::arg("request_epoch"))
      .def(
          "lookup_binding_native",
          &GpuDraftTailBufferCore::lookup_binding,
          py::arg("request_id"),
          py::arg("src_verifier_rank"))
      .def(
          "wait_for_landing_native",
          &GpuDraftTailBufferCore::wait_for_landing,
          py::arg("caller_stream"))
      .def(
          "apply_verify_commit_from_device_native",
          &GpuDraftTailBufferCore::apply_verify_commit_from_device,
          py::arg("gpu_seats"),
          py::arg("expected_request_epochs"),
          py::arg("pre_verify_seq_lens"),
          py::arg("accept_tokens"),
          py::arg("num_accept_tokens"),
          py::arg("commit_mask"),
          py::arg("accept_token_stride"),
          py::arg("batch_size"),
          py::arg("forward_stream"))
      .def(
          "select_snapshot_native",
          &GpuDraftTailBufferCore::select_snapshot,
          py::arg("gpu_seats"),
          py::arg("expected_request_epochs"),
          py::arg("seq_lens"),
          py::arg("bonus_tokens"),
          py::arg("bonus_tokens_are_int32"),
          py::arg("compact_out"),
          py::arg("logical_committed_lens_out"),
          py::arg("debug_out"),
          py::arg("debug_width"),
          py::arg("batch_size"),
          py::arg("verify_stream"),
          py::arg("allow_partial"),
          py::arg("required_tail_len"))
      .def(
          "prepare_decode_native",
          &GpuDraftTailBufferCore::prepare_decode,
          py::arg("mirror_seats"),
          py::arg("expected_request_epochs"),
          py::arg("req_pool_indices"),
          py::arg("candidate_out_cache_locs"),
          py::arg("checkpoint_slot_table"),
          py::arg("checkpoint_capacity"),
          py::arg("req_to_token"),
          py::arg("req_to_token_num_rows"),
          py::arg("req_to_token_row_stride"),
          py::arg("resolved_input_ids"),
          py::arg("resolved_seq_lens"),
          py::arg("resolved_orig_seq_lens"),
          py::arg("mamba_src_indices"),
          py::arg("mamba_dst_indices"),
          py::arg("captured_state_positions"),
          py::arg("old_cache_locs"),
          py::arg("kv_ownership"),
          py::arg("batch_size"),
          py::arg("caller_stream"))
      .def(
          "finish_decode_native",
          &GpuDraftTailBufferCore::finish_decode,
          py::arg("mirror_seats"),
          py::arg("expected_request_epochs"),
          py::arg("req_pool_indices"),
          py::arg("candidate_out_cache_locs"),
          py::arg("sampled_tokens"),
          py::arg("sampled_tokens_are_int32"),
          py::arg("req_to_token"),
          py::arg("req_to_token_num_rows"),
          py::arg("req_to_token_row_stride"),
          py::arg("resolved_input_tokens"),
          py::arg("captured_state_positions"),
          py::arg("old_cache_locs"),
          py::arg("kv_outcomes"),
          py::arg("future_output_tokens"),
          py::arg("future_output_tokens_size"),
          py::arg("batch_size"),
          py::arg("caller_stream"))
      .def(
          "append_prefill_sample_native",
          &GpuDraftTailBufferCore::append_prefill_sample,
          py::arg("mirror_seats"),
          py::arg("expected_request_epochs"),
          py::arg("sampled_tokens"),
          py::arg("sampled_tokens_are_int32"),
          py::arg("accept_out"),
          py::arg("batch_size"),
          py::arg("caller_stream"))
      .def(
          "select_mock_snapshot_native",
          &GpuDraftTailBufferCore::select_mock_snapshot,
          py::arg("gpu_seats"),
          py::arg("seq_lens"),
          py::arg("bonus_tokens"),
          py::arg("bonus_tokens_are_int32"),
          py::arg("compact_out"),
          py::arg("logical_committed_lens_out"),
          py::arg("debug_out"),
          py::arg("debug_width"),
          py::arg("batch_size"),
          py::arg("verify_stream"))
      .def("close", &GpuDraftTailBufferCore::close)
      .def_property_readonly(
          "num_seats", &GpuDraftTailBufferCore::num_seats)
      .def_property_readonly(
          "num_draft_tokens", &GpuDraftTailBufferCore::num_draft_tokens)
      .def_property_readonly(
          "tail_capacity", &GpuDraftTailBufferCore::tail_capacity)
      .def_property_readonly(
          "pending_token_capacity",
          &GpuDraftTailBufferCore::pending_token_capacity)
      .def_property_readonly(
          "staging_slot_count",
          &GpuDraftTailBufferCore::staging_slot_count)
      .def_property_readonly(
          "max_staging_slots",
          &GpuDraftTailBufferCore::max_staging_slots);

  py::class_<DecoupledSpecDraftProxyThread>(m, "DraftProxyThread")
      .def(
          py::init<
              int64_t,
              const std::string&,
              const py::sequence&,
              std::shared_ptr<GpuDraftTailBufferCore>,
              uintptr_t,
              bool,
              bool>(),
          py::arg("verifier_rank"),
          py::arg("bind_endpoint"),
          py::arg("drafter_peers"),
          py::arg("gpu_tail_buffer"),
          py::arg("external_context") = 0,
          py::arg("mock_profile") = false,
          py::arg("python_transport") = false)
      .def("send_pending", &DecoupledSpecDraftProxyThread::send_pending)
      .def("receive_frame", &DecoupledSpecDraftProxyThread::receive_frame)
      .def("send_clock_probe", &DecoupledSpecDraftProxyThread::send_clock_probe)
      .def(
          "take_transport_metrics",
          &DecoupledSpecDraftProxyThread::take_transport_metrics)
      .def("start", &DecoupledSpecDraftProxyThread::start, py::call_guard<py::gil_scoped_release>())
      .def("close", &DecoupledSpecDraftProxyThread::close, py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_control_batch_native",
          &DecoupledSpecDraftProxyThread::submit_control_batch_native,
          py::arg("dst_drafter_rank"),
          py::arg("sync_rows"),
          py::arg("commit_rows"),
          py::arg("close_rows"),
          py::arg("apply_local_verify_commits") = true)
;

  py::class_<DecoupledSpecTokenSyncThread>(m, "TokenSyncThread")
      .def(
          py::init<
              int64_t,
              const std::string&,
              const py::sequence&,
              uintptr_t,
              std::shared_ptr<GpuDraftTailBufferCore>,
              bool,
              int64_t>(),
          py::arg("drafter_rank"),
          py::arg("bind_endpoint"),
          py::arg("verifier_peers"),
          py::arg("external_context") = 0,
          py::arg("gpu_tail_buffer") = nullptr,
          py::arg("python_transport") = false,
          py::arg("schedule_ahead_limit") = -1)
      .def("send_pending", &DecoupledSpecTokenSyncThread::send_pending)
      .def("receive_frame", &DecoupledSpecTokenSyncThread::receive_frame)
      .def("poll_gpu_egress", &DecoupledSpecTokenSyncThread::poll_gpu_egress)
      .def(
          "lookup_gpu_binding_native",
          &DecoupledSpecTokenSyncThread::lookup_gpu_binding,
          py::arg("request_id"),
          py::arg("src_verifier_rank"))
      .def(
          "take_transport_metrics",
          &DecoupledSpecTokenSyncThread::take_transport_metrics)
      .def("start", &DecoupledSpecTokenSyncThread::start, py::call_guard<py::gil_scoped_release>())
      .def("close", &DecoupledSpecTokenSyncThread::close, py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_draft_results_native",
          &DecoupledSpecTokenSyncThread::submit_draft_results_native,
          py::arg("rows"))
      .def("pending_control_count", &DecoupledSpecTokenSyncThread::pending_control_count)
      .def(
          "wait_for_pending_control",
          &DecoupledSpecTokenSyncThread::wait_for_pending_control,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "extract_lifecycle_controls_native",
          &DecoupledSpecTokenSyncThread::extract_lifecycle_controls_native)
      .def(
          "drain_gpu_progress_native",
          &DecoupledSpecTokenSyncThread::drain_gpu_progress_native);

}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime_api.h>

#include "gpu_draft_tail.h"

#include <algorithm>
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

constexpr char kMagic[] = {'D', 'S', 'C', '1'};
constexpr uint8_t kVersion = 1;
constexpr uint8_t kKindControlBatch = 1;
constexpr uint8_t kKindTailStreamBatch = 2;

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
  }

  uint8_t kind() const { return kind_; }

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
};

class BinaryWriter {
 public:
  explicit BinaryWriter(uint8_t kind) {
    data_.append(kMagic, sizeof(kMagic));
    write_u8(kVersion);
    write_u8(kind);
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

// Local Python/C++ boundary frame. This is deliberately independent from the
// DSC1 verifier/drafter network wire above: network compatibility must not be
// coupled to the in-process batch ABI.
constexpr char kLocalFrameMagic[] = {'S', 'G', 'L', 'D', 'P', 'F', '\0', '\0'};
constexpr uint16_t kLocalFrameVersion = 1;
constexpr uint32_t kLocalFrameHeaderSize = 64;
constexpr uint32_t kLocalFrameRowSize = 128;

constexpr uint16_t kFrameKindRequest = 1;
constexpr uint16_t kFrameKindVerifierControl = 2;
constexpr uint16_t kFrameKindSnapshot = 3;
constexpr uint16_t kFrameKindDraftResult = 4;
constexpr uint16_t kFrameKindControlProbe = 5;
constexpr uint16_t kFrameKindDraftAction = 6;

constexpr uint16_t kRowOpRequestKey = 1;
constexpr uint16_t kRowOpRequestState = 2;
constexpr uint16_t kRowOpControlOpen = 10;
constexpr uint16_t kRowOpControlCommit = 11;
constexpr uint16_t kRowOpControlClose = 12;
constexpr uint16_t kRowOpSnapshot = 20;
constexpr uint16_t kRowOpDraftResult = 30;
constexpr uint16_t kRowOpDraftEcho = 31;
constexpr uint16_t kRowOpControlProbe = 40;
constexpr uint16_t kRowOpActionOpen = 50;
constexpr uint16_t kRowOpActionClose = 51;
constexpr uint16_t kRowOpActionAdvance = 52;
constexpr uint16_t kRowOpActionRewrite = 53;

uint16_t load_le_u16(const char* ptr) {
  return static_cast<uint16_t>(static_cast<uint8_t>(ptr[0])) |
         static_cast<uint16_t>(static_cast<uint8_t>(ptr[1])) << 8;
}

uint32_t load_le_u32(const char* ptr) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return value;
}

uint64_t load_le_u64(const char* ptr) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(static_cast<uint8_t>(ptr[i])) << (8 * i);
  }
  return value;
}

void store_le_u16(char* ptr, uint16_t value) {
  for (int i = 0; i < 2; ++i) ptr[i] = static_cast<char>(value >> (8 * i));
}

void store_le_u32(char* ptr, uint32_t value) {
  for (int i = 0; i < 4; ++i) ptr[i] = static_cast<char>(value >> (8 * i));
}

void store_le_u64(char* ptr, uint64_t value) {
  for (int i = 0; i < 8; ++i) ptr[i] = static_cast<char>(value >> (8 * i));
}

struct LocalFrameRowCpp {
  uint16_t op = 0;
  uint16_t flags = 0;
  uint32_t reserved = 0;
  int64_t src_rank = 0;
  int64_t dst_rank = 0;
  int64_t values[9] = {};
  uint32_t str0_offset = 0;
  uint32_t str0_length = 0;
  uint32_t str1_offset = 0;
  uint32_t str1_length = 0;
  uint32_t tok0_offset = 0;
  uint32_t tok0_length = 0;
  uint32_t tok1_offset = 0;
  uint32_t tok1_length = 0;
};

struct LocalFrameCpp {
  uint16_t kind = 0;
  uint32_t flags = 0;
  uint32_t aux[3] = {};
  std::vector<LocalFrameRowCpp> rows;
  std::vector<int32_t> tokens;
  std::string strings;

  std::pair<uint32_t, uint32_t> append_string(const std::string& value) {
    if (value.empty()) return {0, 0};
    if (value.size() > std::numeric_limits<uint32_t>::max() ||
        strings.size() > std::numeric_limits<uint32_t>::max() - value.size()) {
      throw std::runtime_error("Local data-plane string slab exceeds uint32");
    }
    uint32_t offset = static_cast<uint32_t>(strings.size());
    strings.append(value);
    return {offset, static_cast<uint32_t>(value.size())};
  }

  std::pair<uint32_t, uint32_t> append_tokens(
      const std::vector<int32_t>& value) {
    if (value.empty()) return {0, 0};
    if (value.size() > std::numeric_limits<uint32_t>::max() ||
        tokens.size() > std::numeric_limits<uint32_t>::max() - value.size()) {
      throw std::runtime_error("Local data-plane token slab exceeds uint32");
    }
    uint32_t offset = static_cast<uint32_t>(tokens.size());
    tokens.insert(tokens.end(), value.begin(), value.end());
    return {offset, static_cast<uint32_t>(value.size())};
  }

  std::string string_at(const LocalFrameRowCpp& row, bool second = false) const {
    uint32_t offset = second ? row.str1_offset : row.str0_offset;
    uint32_t length = second ? row.str1_length : row.str0_length;
    return strings.substr(offset, length);
  }

  std::vector<int32_t> tokens_at(
      const LocalFrameRowCpp& row,
      bool second = false) const {
    uint32_t offset = second ? row.tok1_offset : row.tok0_offset;
    uint32_t length = second ? row.tok1_length : row.tok0_length;
    return std::vector<int32_t>(
        tokens.begin() + offset, tokens.begin() + offset + length);
  }
};

void validate_local_row_ranges(
    const LocalFrameRowCpp& row,
    uint32_t token_count,
    uint32_t string_size) {
  auto validate_range = [](uint32_t offset, uint32_t length, uint32_t size,
                           const char* name) {
    if (offset > size || length > size - offset) {
      throw std::runtime_error(std::string("Invalid local frame ") + name + " range");
    }
  };
  validate_range(row.str0_offset, row.str0_length, string_size, "str0");
  validate_range(row.str1_offset, row.str1_length, string_size, "str1");
  validate_range(row.tok0_offset, row.tok0_length, token_count, "tok0");
  validate_range(row.tok1_offset, row.tok1_length, token_count, "tok1");
}

LocalFrameCpp decode_local_frame(
    const std::string& bytes,
    uint16_t expected_kind = 0) {
  if (bytes.size() < kLocalFrameHeaderSize ||
      std::memcmp(bytes.data(), kLocalFrameMagic, sizeof(kLocalFrameMagic)) != 0) {
    throw std::runtime_error("Invalid local data-plane frame magic");
  }
  const char* data = bytes.data();
  uint16_t version = load_le_u16(data + 8);
  uint16_t kind = load_le_u16(data + 10);
  if (version != kLocalFrameVersion) {
    throw std::runtime_error("Unsupported local data-plane frame version");
  }
  if (expected_kind != 0 && kind != expected_kind) {
    throw std::runtime_error("Unexpected local data-plane frame kind");
  }

  uint32_t flags = load_le_u32(data + 12);
  uint32_t header_size = load_le_u32(data + 16);
  uint32_t row_size = load_le_u32(data + 20);
  uint32_t row_count = load_le_u32(data + 24);
  uint32_t rows_offset = load_le_u32(data + 28);
  uint32_t tokens_offset = load_le_u32(data + 32);
  uint32_t token_count = load_le_u32(data + 36);
  uint32_t strings_offset = load_le_u32(data + 40);
  uint32_t string_size = load_le_u32(data + 44);
  uint32_t total_size = load_le_u32(data + 48);

  uint64_t expected_tokens_offset =
      static_cast<uint64_t>(kLocalFrameHeaderSize) +
      static_cast<uint64_t>(row_count) * kLocalFrameRowSize;
  uint64_t expected_strings_offset =
      expected_tokens_offset + static_cast<uint64_t>(token_count) * sizeof(int32_t);
  uint64_t expected_total_size = expected_strings_offset + string_size;
  if (header_size != kLocalFrameHeaderSize || row_size != kLocalFrameRowSize ||
      rows_offset != kLocalFrameHeaderSize ||
      tokens_offset != expected_tokens_offset ||
      strings_offset != expected_strings_offset ||
      total_size != expected_total_size || total_size != bytes.size()) {
    throw std::runtime_error("Non-canonical local data-plane frame layout");
  }

  LocalFrameCpp frame;
  frame.kind = kind;
  frame.flags = flags;
  frame.aux[0] = load_le_u32(data + 52);
  frame.aux[1] = load_le_u32(data + 56);
  frame.aux[2] = load_le_u32(data + 60);
  frame.rows.reserve(row_count);
  for (uint32_t i = 0; i < row_count; ++i) {
    const char* ptr = data + rows_offset + static_cast<size_t>(i) * row_size;
    LocalFrameRowCpp row;
    row.op = load_le_u16(ptr);
    row.flags = load_le_u16(ptr + 2);
    row.reserved = load_le_u32(ptr + 4);
    row.src_rank = static_cast<int64_t>(load_le_u64(ptr + 8));
    row.dst_rank = static_cast<int64_t>(load_le_u64(ptr + 16));
    for (int j = 0; j < 9; ++j) {
      row.values[j] = static_cast<int64_t>(load_le_u64(ptr + 24 + j * 8));
    }
    row.str0_offset = load_le_u32(ptr + 96);
    row.str0_length = load_le_u32(ptr + 100);
    row.str1_offset = load_le_u32(ptr + 104);
    row.str1_length = load_le_u32(ptr + 108);
    row.tok0_offset = load_le_u32(ptr + 112);
    row.tok0_length = load_le_u32(ptr + 116);
    row.tok1_offset = load_le_u32(ptr + 120);
    row.tok1_length = load_le_u32(ptr + 124);
    validate_local_row_ranges(row, token_count, string_size);
    frame.rows.push_back(row);
  }

  frame.tokens.reserve(token_count);
  for (uint32_t i = 0; i < token_count; ++i) {
    frame.tokens.push_back(static_cast<int32_t>(
        load_le_u32(data + tokens_offset + static_cast<size_t>(i) * 4)));
  }
  frame.strings.assign(data + strings_offset, string_size);
  return frame;
}

std::string encode_local_frame(const LocalFrameCpp& frame) {
  if (frame.rows.size() > std::numeric_limits<uint32_t>::max() ||
      frame.tokens.size() > std::numeric_limits<uint32_t>::max() ||
      frame.strings.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Local data-plane frame exceeds uint32 limits");
  }
  uint32_t row_count = static_cast<uint32_t>(frame.rows.size());
  uint32_t token_count = static_cast<uint32_t>(frame.tokens.size());
  uint32_t string_size = static_cast<uint32_t>(frame.strings.size());
  uint64_t tokens_offset64 =
      static_cast<uint64_t>(kLocalFrameHeaderSize) +
      static_cast<uint64_t>(row_count) * kLocalFrameRowSize;
  uint64_t strings_offset64 =
      tokens_offset64 + static_cast<uint64_t>(token_count) * sizeof(int32_t);
  uint64_t total_size64 = strings_offset64 + string_size;
  if (total_size64 > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Local data-plane frame exceeds uint32 size");
  }
  uint32_t tokens_offset = static_cast<uint32_t>(tokens_offset64);
  uint32_t strings_offset = static_cast<uint32_t>(strings_offset64);
  uint32_t total_size = static_cast<uint32_t>(total_size64);

  for (const auto& row : frame.rows) {
    validate_local_row_ranges(row, token_count, string_size);
  }

  std::string bytes(total_size, '\0');
  char* data = bytes.data();
  std::memcpy(data, kLocalFrameMagic, sizeof(kLocalFrameMagic));
  store_le_u16(data + 8, kLocalFrameVersion);
  store_le_u16(data + 10, frame.kind);
  store_le_u32(data + 12, frame.flags);
  store_le_u32(data + 16, kLocalFrameHeaderSize);
  store_le_u32(data + 20, kLocalFrameRowSize);
  store_le_u32(data + 24, row_count);
  store_le_u32(data + 28, kLocalFrameHeaderSize);
  store_le_u32(data + 32, tokens_offset);
  store_le_u32(data + 36, token_count);
  store_le_u32(data + 40, strings_offset);
  store_le_u32(data + 44, string_size);
  store_le_u32(data + 48, total_size);
  store_le_u32(data + 52, frame.aux[0]);
  store_le_u32(data + 56, frame.aux[1]);
  store_le_u32(data + 60, frame.aux[2]);

  for (uint32_t i = 0; i < row_count; ++i) {
    const auto& row = frame.rows[i];
    char* ptr = data + kLocalFrameHeaderSize + static_cast<size_t>(i) * kLocalFrameRowSize;
    store_le_u16(ptr, row.op);
    store_le_u16(ptr + 2, row.flags);
    store_le_u32(ptr + 4, row.reserved);
    store_le_u64(ptr + 8, static_cast<uint64_t>(row.src_rank));
    store_le_u64(ptr + 16, static_cast<uint64_t>(row.dst_rank));
    for (int j = 0; j < 9; ++j) {
      store_le_u64(ptr + 24 + j * 8, static_cast<uint64_t>(row.values[j]));
    }
    store_le_u32(ptr + 96, row.str0_offset);
    store_le_u32(ptr + 100, row.str0_length);
    store_le_u32(ptr + 104, row.str1_offset);
    store_le_u32(ptr + 108, row.str1_length);
    store_le_u32(ptr + 112, row.tok0_offset);
    store_le_u32(ptr + 116, row.tok0_length);
    store_le_u32(ptr + 120, row.tok1_offset);
    store_le_u32(ptr + 124, row.tok1_length);
  }
  for (uint32_t i = 0; i < token_count; ++i) {
    store_le_u32(
        data + tokens_offset + static_cast<size_t>(i) * 4,
        static_cast<uint32_t>(frame.tokens[i]));
  }
  if (string_size > 0) {
    std::memcpy(data + strings_offset, frame.strings.data(), string_size);
  }
  return bytes;
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
  int64_t new_token_pos = 0;
  int32_t new_token = 0;
  bool is_commit_echo = false;
};

struct DraftControlBatchCpp {
  int32_t dst_drafter_rank = 0;
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<VerifyCommitCpp> verify_commit_messages;
  std::vector<DraftCloseCpp> close_messages;
};

struct DraftTailStreamOutputBatchCpp {
  std::vector<DraftTailStreamOutputCpp> outputs;
};

struct ExtractDecisionCpp {
  DraftReqKeyCpp key;
  int32_t dst_drafter_rank = 0;
  int64_t pre_verify_committed_len = 0;
  int64_t consumable_len = 0;
};

DraftControlBatchCpp parse_control_batch(const std::string& frame) {
  BinaryReader reader(frame);
  reader.expect_kind(kKindControlBatch);
  DraftControlBatchCpp batch;
  batch.dst_drafter_rank = reader.read_i32();
  uint32_t sync_count = reader.read_u32();
  batch.sync_messages.reserve(sync_count);
  for (uint32_t i = 0; i < sync_count; ++i) {
    DraftSyncCpp msg;
    msg.request_id = reader.read_string();
    msg.src_verifier_rank = reader.read_i32();
    msg.dst_drafter_rank = reader.read_i32();
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
  uint32_t n = reader.read_u32();
  batch.outputs.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    DraftTailStreamOutputCpp output;
    output.src_drafter_rank = reader.read_i32();
    output.dst_verifier_rank = reader.read_i32();
    output.request_id = reader.read_string();
    output.base_committed_len = reader.read_i64();
    output.new_token_pos = reader.read_i64();
    output.new_token = reader.read_i32();
    batch.outputs.push_back(std::move(output));
  }
  reader.finish();
  return batch;
}

std::string encode_tail_stream_batch(const DraftTailStreamOutputBatchCpp& batch) {
  BinaryWriter writer(kKindTailStreamBatch);
  writer.write_u32(static_cast<uint32_t>(batch.outputs.size()));
  for (const auto& output : batch.outputs) {
    writer.write_i32(output.src_drafter_rank);
    writer.write_i32(output.dst_verifier_rank);
    writer.write_string(output.request_id);
    writer.write_i64(output.base_committed_len);
    writer.write_i64(output.new_token_pos);
    writer.write_i32(output.new_token);
  }
  return writer.data();
}

std::string encode_control_batch(const DraftControlBatchCpp& batch) {
  BinaryWriter writer(kKindControlBatch);
  writer.write_i32(batch.dst_drafter_rank);
  writer.write_u32(static_cast<uint32_t>(batch.sync_messages.size()));
  for (const auto& msg : batch.sync_messages) {
    writer.write_string(msg.request_id);
    writer.write_i32(msg.src_verifier_rank);
    writer.write_i32(msg.dst_drafter_rank);
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

void write_int_list_from_py(BinaryWriter& writer, const py::handle& value) {
  py::object tokens = sequence_fast(value, "Expected an integer token sequence");
  Py_ssize_t n = PySequence_Fast_GET_SIZE(tokens.ptr());
  writer.write_u32(static_cast<uint32_t>(n));
  PyObject** items = PySequence_Fast_ITEMS(tokens.ptr());
  for (Py_ssize_t i = 0; i < n; ++i) {
    writer.write_i32(py_int32_from_object(items[i]));
  }
}

std::string encode_control_batch_native_rows(
    int64_t dst_drafter_rank,
    const py::sequence& sync_rows,
    const py::sequence& commit_rows,
    const py::sequence& close_rows) {
  BinaryWriter writer(kKindControlBatch);
  writer.write_i32(static_cast<int32_t>(dst_drafter_rank));

  writer.write_u32(static_cast<uint32_t>(py::len(sync_rows)));
  for (const auto& item : sync_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 5) {
      throw std::runtime_error("DraftSync native row must have 5 fields");
    }
    writer.write_string(row[0].cast<std::string>());
    writer.write_i32(static_cast<int32_t>(row[1].cast<int64_t>()));
    writer.write_i32(static_cast<int32_t>(row[2].cast<int64_t>()));
    write_int_list_from_py(writer, row[3]);
    write_int_list_from_py(writer, row[4]);
  }

  writer.write_u32(static_cast<uint32_t>(py::len(commit_rows)));
  for (const auto& item : commit_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 5) {
      throw std::runtime_error("VerifyCommit native row must have 5 fields");
    }
    writer.write_string(row[0].cast<std::string>());
    writer.write_i32(static_cast<int32_t>(row[1].cast<int64_t>()));
    writer.write_i32(static_cast<int32_t>(row[2].cast<int64_t>()));
    writer.write_i64(row[3].cast<int64_t>());
    write_int_list_from_py(writer, row[4]);
  }

  writer.write_u32(static_cast<uint32_t>(py::len(close_rows)));
  for (const auto& item : close_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 4) {
      throw std::runtime_error("DraftClose native row must have 4 fields");
    }
    writer.write_string(row[0].cast<std::string>());
    writer.write_i32(static_cast<int32_t>(row[1].cast<int64_t>()));
    writer.write_i32(static_cast<int32_t>(row[2].cast<int64_t>()));
    writer.write_string(row[3].cast<std::string>());
  }

  return writer.data();
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
    if (py::len(row) != 5) {
      throw std::runtime_error("DraftSync native row must have 5 fields");
    }
    DraftSyncOpenCpp open;
    open.request_id = row[0].cast<std::string>();
    open.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    open.prompt_len = static_cast<int64_t>(py::len(row[3]));
    open.committed_len = static_cast<int64_t>(py::len(row[4]));
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
    if (py::len(row) != 5) {
      throw std::runtime_error("DraftSync native row must have 5 fields");
    }
    DraftSyncCpp msg;
    msg.request_id = row[0].cast<std::string>();
    msg.src_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    msg.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    msg.prompt_token_ids = py_int32_vector(row[3]);
    msg.committed_outputs = py_int32_vector(row[4]);
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

std::vector<DraftControlBatchCpp> build_verify_update_batches_native(
    const py::sequence& commit_rows,
    const py::sequence& close_rows) {
  std::map<int32_t, DraftControlBatchCpp> batches;
  auto get_batch = [&](int64_t dst_drafter_rank) -> DraftControlBatchCpp& {
    int32_t dst_rank =
        checked_transport_rank(dst_drafter_rank, "Destination drafter rank");
    auto& batch = batches[dst_rank];
    batch.dst_drafter_rank = dst_rank;
    return batch;
  };

  for (const auto& item : commit_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 5) {
      throw std::runtime_error("VerifyCommit native row must have 5 fields");
    }
    VerifyCommitCpp message;
    message.request_id = row[0].cast<std::string>();
    message.src_verifier_rank = checked_transport_rank(
        row[1].cast<int64_t>(), "Source verifier rank");
    message.dst_drafter_rank = checked_transport_rank(
        row[2].cast<int64_t>(), "Destination drafter rank");
    message.pre_verify_committed_len = row[3].cast<int64_t>();
    message.committed_tokens = py_int32_vector(row[4]);
    message.validate();
    get_batch(message.dst_drafter_rank)
        .verify_commit_messages.push_back(std::move(message));
  }

  for (const auto& item : close_rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 4) {
      throw std::runtime_error("DraftClose native row must have 4 fields");
    }
    DraftCloseCpp message;
    message.request_id = row[0].cast<std::string>();
    message.src_verifier_rank = checked_transport_rank(
        row[1].cast<int64_t>(), "Source verifier rank");
    message.dst_drafter_rank = checked_transport_rank(
        row[2].cast<int64_t>(), "Destination drafter rank");
    message.reason = row[3].cast<std::string>();
    get_batch(message.dst_drafter_rank)
        .close_messages.push_back(std::move(message));
  }

  std::vector<DraftControlBatchCpp> result;
  result.reserve(batches.size());
  for (auto& [_, batch] : batches) {
    result.push_back(std::move(batch));
  }
  return result;
}

DraftTailStreamOutputBatchCpp build_tail_stream_batch_native(
    const py::sequence& rows) {
  DraftTailStreamOutputBatchCpp batch;
  batch.outputs.reserve(py::len(rows));
  for (const auto& item : rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 6) {
      throw std::runtime_error("DraftTailStreamOutput native row must have 6 fields");
    }
    DraftTailStreamOutputCpp output;
    output.src_drafter_rank = static_cast<int32_t>(row[0].cast<int64_t>());
    output.dst_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    output.request_id = row[2].cast<std::string>();
    output.base_committed_len = row[3].cast<int64_t>();
    output.new_token_pos = row[4].cast<int64_t>();
    output.new_token = static_cast<int32_t>(row[5].cast<int64_t>());
    batch.outputs.push_back(std::move(output));
  }
  return batch;
}

std::vector<ExtractDecisionCpp> build_extract_decisions_native(
    const py::sequence& rows) {
  std::vector<ExtractDecisionCpp> out;
  out.reserve(py::len(rows));
  for (const auto& item : rows) {
    py::sequence row = py::reinterpret_borrow<py::sequence>(item);
    if (py::len(row) != 5) {
      throw std::runtime_error("ExtractDecision native row must have 5 fields");
    }
    ExtractDecisionCpp decision;
    decision.key.request_id = row[0].cast<std::string>();
    decision.key.src_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    decision.dst_drafter_rank = static_cast<int32_t>(row[2].cast<int64_t>());
    decision.pre_verify_committed_len = row[3].cast<int64_t>();
    decision.consumable_len = row[4].cast<int64_t>();
    out.push_back(std::move(decision));
  }
  return out;
}

struct VerifierCommitSegmentCpp {
  DraftReqKeyCpp draft_key;
  int32_t dst_drafter_rank = 0;
  int64_t pre_verify_committed_len = 0;
  std::vector<int32_t> committed_tokens;

  int64_t end_committed_len() const {
    return pre_verify_committed_len + static_cast<int64_t>(committed_tokens.size());
  }

  void append_message(const VerifyCommitCpp& message) {
    if (message.draft_key().map_key() != draft_key.map_key()) {
      throw std::runtime_error("Verifier commit segment received a commit for a different request");
    }
    if (message.dst_drafter_rank != dst_drafter_rank) {
      throw std::runtime_error("Verifier commit segment received a commit for a different drafter rank");
    }
    message.validate();
    if (message.pre_verify_committed_len != end_committed_len()) {
      throw std::runtime_error("Verifier commit segment requires contiguous VerifyCommit messages");
    }
    committed_tokens.insert(
        committed_tokens.end(), message.committed_tokens.begin(), message.committed_tokens.end());
  }

  VerifierCommitSegmentCpp extract_prefix(int64_t num_tokens) {
    if (num_tokens <= 0 || num_tokens > static_cast<int64_t>(committed_tokens.size())) {
      throw std::runtime_error("Invalid verifier commit segment prefix length");
    }
    VerifierCommitSegmentCpp prefix;
    prefix.draft_key = draft_key;
    prefix.dst_drafter_rank = dst_drafter_rank;
    prefix.pre_verify_committed_len = pre_verify_committed_len;
    prefix.committed_tokens.assign(committed_tokens.begin(), committed_tokens.begin() + num_tokens);
    committed_tokens.erase(committed_tokens.begin(), committed_tokens.begin() + num_tokens);
    pre_verify_committed_len += num_tokens;
    return prefix;
  }

  void discard_prefix(int64_t num_tokens) {
    if (num_tokens <= 0 || num_tokens > static_cast<int64_t>(committed_tokens.size())) {
      throw std::runtime_error("Invalid verifier commit segment prefix length");
    }
    committed_tokens.erase(
        committed_tokens.begin(), committed_tokens.begin() + num_tokens);
    pre_verify_committed_len += num_tokens;
  }
};

// Compact model-side action produced after comparing one verifier commit
// segment with the locally mirrored drafter transcript. Matching tokens are
// already present in the Python Req and therefore do not need to cross the
// boundary again; only their count and an optional divergent verifier token
// are required.
struct DraftCommitActionCpp {
  DraftReqKeyCpp draft_key;
  int32_t dst_drafter_rank = 0;
  int64_t pre_verify_committed_len = 0;
  int64_t expected_output_len = 0;
  int64_t matched_prefix_len = 0;
  int64_t new_committed_len = 0;
  int64_t rewrite_pos = -1;
  int32_t rewrite_token = -1;
  int64_t echo_pos = -1;
  int32_t echo_token = 0;

  int64_t committed_token_count() const {
    return new_committed_len - pre_verify_committed_len;
  }

  bool rewrites_suffix() const { return rewrite_pos >= 0; }
};

struct DraftCommitProbeCpp {
  bool ready = false;
  int64_t matched_prefix_len = 0;
  bool has_replacement_token = false;
  int32_t replacement_token_id = 0;

  int64_t consumable_len() const {
    if (!ready) return 0;
    return matched_prefix_len + (has_replacement_token ? 1 : 0);
  }
};

// Protocol-side shadow of drafter Req.output_ids. The verifier-committed
// prefix is represented only by its length; the deque stores the materialized
// speculative suffix at positions [committed_len, materialized_len). This
// keeps the data-plane state bounded by the drafter-ahead window rather than
// retaining the full prompt/output transcript.
class DraftTranscriptMirror {
 public:
  bool contains(const DraftReqKeyCpp& key) const {
    return transcripts_.count(key.map_key()) != 0;
  }

  void open(const DraftSyncCpp& sync) {
    Transcript state;
    state.dst_drafter_rank = sync.dst_drafter_rank;
    state.committed_len = static_cast<int64_t>(sync.committed_outputs.size());
    transcripts_.insert_or_assign(sync.draft_key().map_key(), std::move(state));
  }

  void close(const DraftReqKeyCpp& key) { transcripts_.erase(key.map_key()); }

  void append_draft_outputs(
      const DraftTailStreamOutputBatchCpp& batch,
      bool strict_local_contract = false) {
    for (const auto& output : batch.outputs) {
      append_draft_output(output, strict_local_contract);
    }
  }

  DraftCommitProbeCpp probe(const VerifierCommitSegmentCpp& segment) const {
    auto it = transcripts_.find(segment.draft_key.map_key());
    if (it == transcripts_.end()) return {};

    const auto& state = it->second;
    if (segment.dst_drafter_rank != state.dst_drafter_rank) {
      throw std::runtime_error(
          "Verifier commit segment targets a different drafter transcript");
    }
    if (segment.pre_verify_committed_len != state.committed_len) {
      throw std::runtime_error(
          "Verifier commit segment does not match mirrored committed prefix");
    }
    if (segment.committed_tokens.empty() || state.draft_suffix.empty()) {
      return {};
    }

    DraftCommitProbeCpp probe;
    int64_t max_match = std::min<int64_t>(
        static_cast<int64_t>(segment.committed_tokens.size()),
        static_cast<int64_t>(state.draft_suffix.size()));
    while (probe.matched_prefix_len < max_match &&
           state.draft_suffix[static_cast<size_t>(probe.matched_prefix_len)] ==
               segment.committed_tokens[static_cast<size_t>(probe.matched_prefix_len)]) {
      ++probe.matched_prefix_len;
    }

    if (probe.matched_prefix_len ==
        static_cast<int64_t>(segment.committed_tokens.size())) {
      probe.ready = true;
      return probe;
    }
    if (probe.matched_prefix_len < max_match) {
      probe.ready = true;
      probe.has_replacement_token = true;
      probe.replacement_token_id =
          segment.committed_tokens[static_cast<size_t>(probe.matched_prefix_len)];
      return probe;
    }

    // The available suffix is a strict matching prefix of the verifier
    // segment. Commit what is materialized now and leave the remainder queued.
    probe.ready = probe.matched_prefix_len > 0;
    return probe;
  }

  DraftCommitActionCpp consume(
      const VerifierCommitSegmentCpp& segment,
      const DraftCommitProbeCpp& probe) {
    if (!probe.ready || probe.consumable_len() <= 0) {
      throw std::runtime_error("Cannot consume a non-ready verifier commit probe");
    }
    auto it = transcripts_.find(segment.draft_key.map_key());
    if (it == transcripts_.end()) {
      throw std::runtime_error("Missing mirrored drafter transcript");
    }
    auto& state = it->second;
    if (segment.pre_verify_committed_len != state.committed_len) {
      throw std::runtime_error(
          "Verifier commit consume does not match mirrored committed prefix");
    }
    if (probe.matched_prefix_len > static_cast<int64_t>(state.draft_suffix.size())) {
      throw std::runtime_error("Verifier commit probe exceeds mirrored draft suffix");
    }

    DraftCommitActionCpp action;
    action.draft_key = segment.draft_key;
    action.dst_drafter_rank = segment.dst_drafter_rank;
    action.pre_verify_committed_len = segment.pre_verify_committed_len;
    action.expected_output_len =
        state.committed_len + static_cast<int64_t>(state.draft_suffix.size());
    action.matched_prefix_len = probe.matched_prefix_len;
    action.new_committed_len =
        action.pre_verify_committed_len + probe.consumable_len();
    if (probe.has_replacement_token) {
      action.rewrite_pos =
          action.pre_verify_committed_len + action.matched_prefix_len;
      action.rewrite_token = probe.replacement_token_id;
    }
    action.echo_pos = action.new_committed_len - 1;
    action.echo_token = segment.committed_tokens[
        static_cast<size_t>(probe.consumable_len() - 1)];

    for (int64_t i = 0; i < probe.matched_prefix_len; ++i) {
      state.draft_suffix.pop_front();
    }
    if (probe.has_replacement_token) {
      if (state.draft_suffix.empty()) {
        throw std::runtime_error(
            "Verifier replacement token has no materialized draft token");
      }
      if (state.draft_suffix.front() == probe.replacement_token_id) {
        throw std::runtime_error(
            "Verifier replacement token unexpectedly matches mirrored draft token");
      }
      // Python will truncate the complete mismatching suffix and install the
      // verifier token at the first divergent position. That token immediately
      // belongs to the committed prefix, so no speculative suffix remains.
      state.draft_suffix.clear();
    }
    state.committed_len = action.new_committed_len;
    return action;
  }

  // Keep the pre-existing callback-driven API coherent while the scheduler is
  // migrated to compact actions. Production flat-frame extraction never uses
  // this compatibility path. A callback is allowed to make a decision from
  // Python state that is not yet present in the mirror (some legacy tests do
  // this deliberately), in which case reset to the consumed verifier prefix.
  void consume_legacy(
      const VerifierCommitSegmentCpp& segment,
      int64_t consumable_len) {
    if (consumable_len <= 0 ||
        consumable_len > static_cast<int64_t>(segment.committed_tokens.size())) {
      throw std::runtime_error("Invalid legacy verifier commit consume length");
    }
    auto it = transcripts_.find(segment.draft_key.map_key());
    if (it == transcripts_.end()) return;
    auto& state = it->second;

    if (segment.pre_verify_committed_len == state.committed_len) {
      auto current_probe = probe(segment);
      if (current_probe.ready &&
          current_probe.consumable_len() == consumable_len) {
        consume(segment, current_probe);
        return;
      }
    }

    state.committed_len =
        segment.pre_verify_committed_len + consumable_len;
    state.draft_suffix.clear();
  }

 private:
  struct Transcript {
    int32_t dst_drafter_rank = 0;
    int64_t committed_len = 0;
    std::deque<int32_t> draft_suffix;
  };

  void append_draft_output(
      const DraftTailStreamOutputCpp& output,
      bool strict_local_contract) {
    DraftReqKeyCpp key{output.dst_verifier_rank, output.request_id};
    auto it = transcripts_.find(key.map_key());
    if (it == transcripts_.end()) {
      if (strict_local_contract) {
        throw std::runtime_error(
            "Draft result has no mirrored drafter transcript");
      }
      return;
    }

    auto& state = it->second;
    const int64_t token_pos = output.new_token_pos;
    if (output.is_commit_echo) {
      if (token_pos < 0 || token_pos >= state.committed_len) {
        throw std::runtime_error(
            "Draft commit echo must refer to an already committed token");
      }
      return;
    }
    if (strict_local_contract) {
      if (output.base_committed_len != state.committed_len) {
        throw std::runtime_error(
            "Draft result base does not match mirrored committed prefix");
      }
      int64_t expected_pos =
          state.committed_len + static_cast<int64_t>(state.draft_suffix.size());
      if (token_pos != expected_pos) {
        throw std::runtime_error(
            "Draft result publication must be contiguous with mirrored suffix");
      }
      state.draft_suffix.push_back(output.new_token);
      return;
    }
    if (token_pos < state.committed_len) {
      // Commit echoes and results produced from an older scheduler view are
      // protocol-idempotent once their position is committed.
      return;
    }
    if (output.base_committed_len < state.committed_len) return;
    if (output.base_committed_len > state.committed_len) {
      throw std::runtime_error(
          "Draft output base is ahead of mirrored committed prefix");
    }

    int64_t materialized_len =
        state.committed_len + static_cast<int64_t>(state.draft_suffix.size());
    if (token_pos < materialized_len) {
      int32_t existing = state.draft_suffix[static_cast<size_t>(
          token_pos - state.committed_len)];
      if (existing != output.new_token) {
        throw std::runtime_error(
            "Draft output conflicts with mirrored transcript suffix");
      }
      return;
    }
    if (token_pos > materialized_len) {
      throw std::runtime_error("Draft output skips mirrored transcript suffix");
    }
    state.draft_suffix.push_back(output.new_token);
  }

  std::unordered_map<std::string, Transcript> transcripts_;
};

struct ReadyDraftControlsCpp {
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<DraftReqKeyCpp> close_keys;
  std::vector<VerifierCommitSegmentCpp> ready_commit_segments;
};

struct ReadyDrafterActionsCpp {
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<DraftReqKeyCpp> close_keys;
  std::vector<DraftCommitActionCpp> commit_actions;
};

struct DraftControlProbeCpp {
  uint64_t probe_id = 0;
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<DraftReqKeyCpp> close_keys;
  std::vector<VerifierCommitSegmentCpp> verifier_commit_segments;
};

DraftTailStreamOutputBatchCpp draft_results_from_local_frame(
    const LocalFrameCpp& frame) {
  if (frame.kind != kFrameKindDraftResult) {
    throw std::runtime_error("Expected a DRAFT_RESULT local frame");
  }
  DraftTailStreamOutputBatchCpp batch;
  batch.outputs.reserve(frame.rows.size());
  for (const auto& row : frame.rows) {
    if (row.op != kRowOpDraftResult && row.op != kRowOpDraftEcho) {
      throw std::runtime_error("Unexpected row op in DRAFT_RESULT frame");
    }
    DraftTailStreamOutputCpp output;
    output.src_drafter_rank = checked_transport_rank(
        row.src_rank, "Local draft-result source rank");
    output.dst_verifier_rank = checked_transport_rank(
        row.dst_rank, "Local draft-result destination rank");
    output.request_id = frame.string_at(row);
    output.base_committed_len = row.values[0];
    output.new_token_pos = row.values[1];
    if (row.values[2] < std::numeric_limits<int32_t>::min() ||
        row.values[2] > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("Local draft-result token id must fit int32");
    }
    output.new_token = static_cast<int32_t>(row.values[2]);
    output.is_commit_echo = row.op == kRowOpDraftEcho;
    batch.outputs.push_back(std::move(output));
  }
  return batch;
}

std::vector<DraftControlBatchCpp> verifier_controls_from_local_frame(
    const LocalFrameCpp& frame) {
  if (frame.kind != kFrameKindVerifierControl) {
    throw std::runtime_error("Expected a VERIFIER_CONTROL local frame");
  }
  std::map<int32_t, DraftControlBatchCpp> by_drafter;
  for (const auto& row : frame.rows) {
    int32_t src_rank = checked_transport_rank(
        row.src_rank, "Local verifier-control source rank");
    int32_t dst_rank = checked_transport_rank(
        row.dst_rank, "Local verifier-control destination rank");
    auto& batch = by_drafter[dst_rank];
    batch.dst_drafter_rank = dst_rank;
    if (row.op == kRowOpControlOpen) {
      DraftSyncCpp sync;
      sync.request_id = frame.string_at(row);
      sync.src_verifier_rank = src_rank;
      sync.dst_drafter_rank = dst_rank;
      sync.prompt_token_ids = frame.tokens_at(row);
      sync.committed_outputs = frame.tokens_at(row, true);
      batch.sync_messages.push_back(std::move(sync));
    } else if (row.op == kRowOpControlCommit) {
      VerifyCommitCpp commit;
      commit.request_id = frame.string_at(row);
      commit.src_verifier_rank = src_rank;
      commit.dst_drafter_rank = dst_rank;
      commit.pre_verify_committed_len = row.values[0];
      commit.committed_tokens = frame.tokens_at(row);
      commit.validate();
      batch.verify_commit_messages.push_back(std::move(commit));
    } else if (row.op == kRowOpControlClose) {
      DraftCloseCpp close;
      close.request_id = frame.string_at(row);
      close.src_verifier_rank = src_rank;
      close.dst_drafter_rank = dst_rank;
      close.reason = frame.string_at(row, true);
      batch.close_messages.push_back(std::move(close));
    } else {
      throw std::runtime_error("Unexpected row op in VERIFIER_CONTROL frame");
    }
  }

  std::vector<DraftControlBatchCpp> batches;
  batches.reserve(by_drafter.size());
  for (auto& kv : by_drafter) batches.push_back(std::move(kv.second));
  return batches;
}

std::string encode_draft_actions_local_frame(
    const ReadyDrafterActionsCpp& ready,
    int32_t drafter_rank) {
  LocalFrameCpp frame;
  frame.kind = kFrameKindDraftAction;
  frame.rows.reserve(
      ready.close_keys.size() + ready.sync_messages.size() +
      ready.commit_actions.size());

  for (const auto& key : ready.close_keys) {
    LocalFrameRowCpp row;
    row.op = kRowOpActionClose;
    row.src_rank = key.src_verifier_rank;
    row.dst_rank = drafter_rank;
    auto rid = frame.append_string(key.request_id);
    row.str0_offset = rid.first;
    row.str0_length = rid.second;
    frame.rows.push_back(row);
  }
  for (const auto& sync : ready.sync_messages) {
    LocalFrameRowCpp row;
    row.op = kRowOpActionOpen;
    row.src_rank = sync.src_verifier_rank;
    row.dst_rank = sync.dst_drafter_rank;
    auto rid = frame.append_string(sync.request_id);
    row.str0_offset = rid.first;
    row.str0_length = rid.second;
    auto prompt = frame.append_tokens(sync.prompt_token_ids);
    row.tok0_offset = prompt.first;
    row.tok0_length = prompt.second;
    auto committed = frame.append_tokens(sync.committed_outputs);
    row.tok1_offset = committed.first;
    row.tok1_length = committed.second;
    frame.rows.push_back(row);
  }
  for (const auto& action : ready.commit_actions) {
    LocalFrameRowCpp row;
    row.op = action.rewrites_suffix() ?
        kRowOpActionRewrite : kRowOpActionAdvance;
    row.src_rank = action.draft_key.src_verifier_rank;
    row.dst_rank = action.dst_drafter_rank;
    auto rid = frame.append_string(action.draft_key.request_id);
    row.str0_offset = rid.first;
    row.str0_length = rid.second;
    row.values[0] = action.expected_output_len;
    row.values[1] = action.pre_verify_committed_len;
    row.values[2] = action.new_committed_len;
    row.values[3] = action.rewrite_pos;
    row.values[4] = action.rewrite_token;
    row.values[5] = action.echo_pos;
    row.values[6] = action.echo_token;
    frame.rows.push_back(row);
  }
  return encode_local_frame(frame);
}

constexpr uint32_t kControlProbeHasUnconditionalControls = 1U << 0;

std::string encode_control_probe_local_frame(
    const DraftControlProbeCpp& probe) {
  LocalFrameCpp frame;
  frame.kind = kFrameKindControlProbe;
  if (!probe.sync_messages.empty() || !probe.close_keys.empty()) {
    frame.flags |= kControlProbeHasUnconditionalControls;
  }
  frame.aux[0] = static_cast<uint32_t>(probe.probe_id);
  frame.aux[1] = static_cast<uint32_t>(probe.probe_id >> 32);
  frame.rows.reserve(probe.verifier_commit_segments.size());
  for (const auto& segment : probe.verifier_commit_segments) {
    LocalFrameRowCpp row;
    row.op = kRowOpControlProbe;
    row.src_rank = segment.draft_key.src_verifier_rank;
    row.dst_rank = segment.dst_drafter_rank;
    auto rid = frame.append_string(segment.draft_key.request_id);
    row.str0_offset = rid.first;
    row.str0_length = rid.second;
    frame.rows.push_back(row);
  }
  return encode_local_frame(frame);
}

std::vector<std::string> request_ids_from_local_frame(
    const LocalFrameCpp& frame) {
  if (frame.kind != kFrameKindRequest) {
    throw std::runtime_error("Expected a REQUEST local frame");
  }
  std::vector<std::string> request_ids;
  request_ids.reserve(frame.rows.size());
  for (const auto& row : frame.rows) {
    if (row.op != kRowOpRequestKey) {
      throw std::runtime_error(
          "REQUEST query frame rows must use REQUEST_KEY");
    }
    request_ids.push_back(frame.string_at(row));
  }
  return request_ids;
}

std::string encode_request_states_local_frame(
    const LocalFrameCpp& request_frame,
    const std::vector<int64_t>& committed_lens) {
  if (request_frame.rows.size() != committed_lens.size()) {
    throw std::runtime_error("Request-state result count mismatch");
  }
  LocalFrameCpp frame;
  frame.kind = kFrameKindRequest;
  frame.rows.reserve(request_frame.rows.size());
  for (size_t i = 0; i < request_frame.rows.size(); ++i) {
    const auto& request_row = request_frame.rows[i];
    LocalFrameRowCpp row;
    row.op = kRowOpRequestState;
    row.src_rank = request_row.src_rank;
    row.values[0] = committed_lens[i];
    auto rid = frame.append_string(request_frame.string_at(request_row));
    row.str0_offset = rid.first;
    row.str0_length = rid.second;
    frame.rows.push_back(row);
  }
  return encode_local_frame(frame);
}

struct RequestDraftTailStateCpp {
  int32_t drafter_rank = 0;
  int64_t prompt_len = 0;
  int64_t committed_len = 0;
  int64_t can_accept_prefix_len = 0;
  std::vector<int32_t> tail_tokens;
  std::deque<int32_t> pending_expected_tokens;

  std::vector<int32_t> consumable_tail_tokens() const {
    if (!pending_expected_tokens.empty()) return {};
    return tail_tokens;
  }

  int64_t consumable_tail_len() const {
    if (!pending_expected_tokens.empty()) return 0;
    return static_cast<int64_t>(tail_tokens.size());
  }
};

struct GpuDraftTailStateCpp {
  std::string request_id;
  bool exists = false;
  int64_t prompt_len = 0;
  int64_t committed_len = 0;
  int64_t raw_tail_len = 0;
  int64_t consumable_tail_len = 0;
  std::vector<int64_t> tail_tokens;
};

struct DraftTailSnapshotCpp {
  std::string request_id;
  int64_t committed_len = 0;
  std::vector<int32_t> tail_tokens;
  int64_t num_consumable_drafts = 0;
  int64_t raw_tail_len = 0;
};

struct DraftTailSnapshotBatchCpp {
  std::vector<DraftTailSnapshotCpp> snapshots;
  int64_t wait_ns = 0;
};

std::string encode_snapshots_local_frame(
    const DraftTailSnapshotBatchCpp& snapshot_batch) {
  LocalFrameCpp frame;
  frame.kind = kFrameKindSnapshot;
  uint64_t wait_ns = static_cast<uint64_t>(snapshot_batch.wait_ns);
  frame.aux[0] = static_cast<uint32_t>(wait_ns);
  frame.aux[1] = static_cast<uint32_t>(wait_ns >> 32);
  frame.rows.reserve(snapshot_batch.snapshots.size());
  for (const auto& snapshot : snapshot_batch.snapshots) {
    LocalFrameRowCpp row;
    row.op = kRowOpSnapshot;
    row.values[0] = snapshot.committed_len;
    row.values[1] = snapshot.raw_tail_len;
    row.values[2] = snapshot.num_consumable_drafts;
    auto rid = frame.append_string(snapshot.request_id);
    row.str0_offset = rid.first;
    row.str0_length = rid.second;
    auto tail = frame.append_tokens(snapshot.tail_tokens);
    row.tok0_offset = tail.first;
    row.tok0_length = tail.second;
    frame.rows.push_back(row);
  }
  return encode_local_frame(frame);
}

std::string encode_snapshots_local_frame(
    const LocalFrameCpp& request_frame,
    const DraftTailSnapshotBatchCpp& snapshot_batch) {
  if (request_frame.rows.size() != snapshot_batch.snapshots.size()) {
    throw std::runtime_error("Draft snapshot result count mismatch");
  }
  return encode_snapshots_local_frame(snapshot_batch);
}

class VerifierSnapshotBatchCpp
    : public std::enable_shared_from_this<VerifierSnapshotBatchCpp> {
 public:
  explicit VerifierSnapshotBatchCpp(DraftTailSnapshotBatchCpp batch)
      : batch_(std::move(batch)) {}

  static int64_t transport_width(int64_t max_tail_len) {
    if (max_tail_len < 0 ||
        static_cast<uint64_t>(max_tail_len) >
            static_cast<uint64_t>(std::numeric_limits<py::ssize_t>::max()) -
                kVerifierSnapshotTensorMetadataWidth) {
      throw std::runtime_error(
          "Verifier snapshot transport tail length is invalid");
    }
    return static_cast<int64_t>(kVerifierSnapshotTensorMetadataWidth) +
        max_tail_len;
  }

  static std::shared_ptr<VerifierSnapshotBatchCpp> from_transport_tensor(
      const py::array_t<int64_t, py::array::c_style>& payload,
      const py::sequence& request_ids) {
    auto ids = py_string_vector(request_ids);
    py::buffer_info info = payload.request();
    if (info.ndim != 2 || info.shape[0] != static_cast<py::ssize_t>(ids.size()) ||
        info.shape[1] <
            static_cast<py::ssize_t>(kVerifierSnapshotTensorMetadataWidth)) {
      throw std::runtime_error(
          "Verifier snapshot transport tensor shape is not aligned");
    }

    const auto* values = static_cast<const int64_t*>(info.ptr);
    const size_t row_width = static_cast<size_t>(info.shape[1]);
    const size_t max_tail_len =
        row_width - kVerifierSnapshotTensorMetadataWidth;
    DraftTailSnapshotBatchCpp batch;
    batch.snapshots.reserve(ids.size());
    for (size_t row_index = 0; row_index < ids.size(); ++row_index) {
      const int64_t* row = values + row_index * row_width;
      if (row[0] != kVerifierSnapshotTensorMagic) {
        throw std::runtime_error(
            "Verifier snapshot transport tensor has invalid magic");
      }
      if (row[1] != uint64_as_int64(stable_request_id_hash(
                        ids[row_index], 0x243f6a8885a308d3ULL)) ||
          row[2] != uint64_as_int64(stable_request_id_hash(
                        ids[row_index], 0x13198a2e03707344ULL))) {
        throw std::runtime_error(
            "Verifier snapshot transport changed request identity");
      }
      int64_t committed_len = row[3];
      int64_t raw_tail_len = row[4];
      int64_t num_consumable_drafts = row[5];
      int64_t tail_len = row[6];
      if (committed_len < 0 || raw_tail_len < 0 ||
          num_consumable_drafts < 0 || tail_len < 0 ||
          static_cast<uint64_t>(tail_len) > max_tail_len ||
          tail_len > raw_tail_len || tail_len > num_consumable_drafts) {
        throw std::runtime_error(
            "Verifier snapshot transport lengths are invalid");
      }

      DraftTailSnapshotCpp snapshot;
      snapshot.request_id = ids[row_index];
      snapshot.committed_len = committed_len;
      snapshot.raw_tail_len = raw_tail_len;
      snapshot.num_consumable_drafts = num_consumable_drafts;
      snapshot.tail_tokens.reserve(static_cast<size_t>(tail_len));
      for (int64_t token_index = 0; token_index < tail_len; ++token_index) {
        int64_t token = row[kVerifierSnapshotTensorMetadataWidth +
                            static_cast<size_t>(token_index)];
        if (token < std::numeric_limits<int32_t>::min() ||
            token > std::numeric_limits<int32_t>::max()) {
          throw std::runtime_error(
              "Verifier snapshot transport token does not fit int32");
        }
        snapshot.tail_tokens.push_back(static_cast<int32_t>(token));
      }
      batch.snapshots.push_back(std::move(snapshot));
    }
    return std::make_shared<VerifierSnapshotBatchCpp>(std::move(batch));
  }

  py::array_t<int64_t> to_transport_tensor(int64_t max_tail_len) {
    const int64_t width = transport_width(max_tail_len);
    if (transport_width_ >= 0 && transport_width_ != width) {
      throw std::runtime_error(
          "Verifier snapshot transport tensor cannot change shape");
    }
    if (transport_width_ < 0) {
      const size_t row_width = static_cast<size_t>(width);
      if (batch_.snapshots.size() >
          std::numeric_limits<size_t>::max() /
              std::max<size_t>(row_width, 1)) {
        throw std::runtime_error(
            "Verifier snapshot transport tensor is too large");
      }
      transport_values_.assign(batch_.snapshots.size() * row_width, 0);
      for (size_t row_index = 0; row_index < batch_.snapshots.size();
           ++row_index) {
        const auto& snapshot = batch_.snapshots[row_index];
        int64_t* row = transport_values_.data() + row_index * row_width;
        row[0] = kVerifierSnapshotTensorMagic;
        row[1] = uint64_as_int64(stable_request_id_hash(
            snapshot.request_id, 0x243f6a8885a308d3ULL));
        row[2] = uint64_as_int64(stable_request_id_hash(
            snapshot.request_id, 0x13198a2e03707344ULL));
        row[3] = snapshot.committed_len;
        row[4] = snapshot.raw_tail_len;
        row[5] = snapshot.num_consumable_drafts;
        size_t tail_len = std::min(
            snapshot.tail_tokens.size(), static_cast<size_t>(max_tail_len));
        row[6] = static_cast<int64_t>(tail_len);
        for (size_t token_index = 0; token_index < tail_len; ++token_index) {
          row[kVerifierSnapshotTensorMetadataWidth + token_index] =
              snapshot.tail_tokens[token_index];
        }
      }
      transport_width_ = width;
    }
    return array_view(
        transport_values_,
        {static_cast<py::ssize_t>(batch_.snapshots.size()),
         static_cast<py::ssize_t>(width)});
  }

  int64_t wait_ns() const { return batch_.wait_ns; }

  void align(
      const py::sequence& request_ids,
      const py::sequence& pre_committed_lens,
      const py::sequence& row_indices,
      int64_t batch_size) {
    auto ids = py_string_vector(request_ids);
    auto committed = py_int64_vector(pre_committed_lens);
    auto indices = py_int64_vector(row_indices);
    if (batch_size < 0 ||
        static_cast<uint64_t>(batch_size) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      throw std::runtime_error("Verifier snapshot batch size is invalid");
    }
    if (committed.size() != static_cast<size_t>(batch_size)) {
      throw std::runtime_error(
          "Verifier snapshot pre-committed lengths do not match batch size");
    }
    if (ids.size() != batch_.snapshots.size() ||
        indices.size() != batch_.snapshots.size()) {
      throw std::runtime_error("Verifier snapshot rows are not aligned");
    }

    std::vector<bool> seen(static_cast<size_t>(batch_size), false);
    pre_committed_lens_ = committed;
    request_id_to_batch_index_.clear();
    request_id_to_batch_index_.reserve(ids.size());
    row_indices_.clear();
    row_indices_.reserve(indices.size());
    active_rows_.assign(indices.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
      int64_t batch_index = indices[i];
      if (batch_index < 0 || batch_index >= batch_size) {
        throw std::runtime_error("Verifier snapshot row index is out of range");
      }
      size_t index = static_cast<size_t>(batch_index);
      if (seen[index]) {
        throw std::runtime_error("Verifier snapshot row index is duplicated");
      }
      seen[index] = true;

      const auto& snapshot = batch_.snapshots[i];
      if (snapshot.request_id != ids[i]) {
        throw std::runtime_error(
            "Verifier snapshot response changed request identity");
      }
      int64_t expected_committed_len = committed[index];
      if (expected_committed_len < 0) {
        throw std::runtime_error(
            "Verifier snapshot pre-committed length must be non-negative");
      }
      if (snapshot.committed_len > expected_committed_len) {
        throw std::runtime_error(
            "Verifier snapshot committed length is ahead of the request");
      }
      active_rows_[i] = snapshot.committed_len == expected_committed_len;
      if (!request_id_to_batch_index_.emplace(ids[i], index).second) {
        throw std::runtime_error(
            "Verifier snapshot request identity is duplicated");
      }
      row_indices_.push_back(index);
    }

    batch_size_ = static_cast<size_t>(batch_size);
    aligned_ = true;
    materialized_ = false;
  }

  py::tuple materialize(int64_t num_draft_tokens, int64_t pad_token) {
    ensure_aligned();
    if (num_draft_tokens < 0) {
      throw std::runtime_error(
          "Verifier snapshot num_draft_tokens must be non-negative");
    }
    if (materialized_ &&
        (materialized_num_draft_tokens_ != num_draft_tokens ||
         materialized_pad_token_ != pad_token)) {
      throw std::runtime_error(
          "Verifier snapshot batch cannot be rematerialized with a new shape");
    }
    if (!materialized_) {
      if (static_cast<uint64_t>(num_draft_tokens) >
          std::numeric_limits<size_t>::max() /
              std::max<size_t>(batch_size_, 1)) {
        throw std::runtime_error("Verifier snapshot token matrix is too large");
      }
      size_t row_width = static_cast<size_t>(num_draft_tokens);
      dense_draft_tokens_.assign(batch_size_ * row_width, pad_token);
      draft_lens_.assign(batch_size_, 0);
      for (size_t row_index = 0; row_index < batch_.snapshots.size(); ++row_index) {
        if (!active_rows_[row_index]) continue;
        size_t batch_index = row_indices_[row_index];
        const auto& tokens = batch_.snapshots[row_index].tail_tokens;
        size_t copy_size = std::min(tokens.size(), row_width);
        draft_lens_[batch_index] = static_cast<int64_t>(copy_size);
        for (size_t token_index = 0; token_index < copy_size; ++token_index) {
          dense_draft_tokens_[batch_index * row_width + token_index] =
              static_cast<int64_t>(tokens[token_index]);
        }
      }
      materialized_num_draft_tokens_ = num_draft_tokens;
      materialized_pad_token_ = pad_token;
      materialized_ = true;
    }

    return py::make_tuple(
        array_view(
            dense_draft_tokens_,
            {static_cast<py::ssize_t>(batch_size_),
             static_cast<py::ssize_t>(num_draft_tokens)}),
        array_view(
            draft_lens_, {static_cast<py::ssize_t>(batch_size_)}));
  }

  std::vector<int64_t> draft_lens(int64_t num_draft_tokens) const {
    ensure_aligned();
    if (num_draft_tokens < 0) {
      throw std::runtime_error(
          "Verifier snapshot num_draft_tokens must be non-negative");
    }
    std::vector<int64_t> result(batch_size_, 0);
    size_t row_width = static_cast<size_t>(num_draft_tokens);
    for (size_t row_index = 0; row_index < batch_.snapshots.size(); ++row_index) {
      if (!active_rows_[row_index]) continue;
      result[row_indices_[row_index]] = static_cast<int64_t>(
          std::min(batch_.snapshots[row_index].tail_tokens.size(), row_width));
    }
    return result;
  }

  std::vector<int64_t> num_consumable_drafts() const {
    ensure_aligned();
    std::vector<int64_t> result(batch_size_, 0);
    for (size_t row_index = 0; row_index < batch_.snapshots.size(); ++row_index) {
      if (!active_rows_[row_index]) continue;
      result[row_indices_[row_index]] =
          batch_.snapshots[row_index].num_consumable_drafts;
    }
    return result;
  }

  void validate_verify_updates(
      const std::vector<DraftControlBatchCpp>& batches) const {
    ensure_aligned();
    std::unordered_map<std::string, bool> seen;
    for (const auto& batch : batches) {
      for (const auto& message : batch.verify_commit_messages) {
        auto it = request_id_to_batch_index_.find(message.request_id);
        if (it == request_id_to_batch_index_.end()) {
          throw std::runtime_error(
              "Verifier update request is absent from the snapshot batch");
        }
        if (!seen.emplace(message.request_id, true).second) {
          throw std::runtime_error(
              "Verifier update request identity is duplicated");
        }
        if (message.pre_verify_committed_len !=
            pre_committed_lens_[it->second]) {
          throw std::runtime_error(
              "Verifier update pre-commit length does not match snapshot batch");
        }
      }
      for (const auto& message : batch.close_messages) {
        if (request_id_to_batch_index_.count(message.request_id) == 0) {
          throw std::runtime_error(
              "Verifier close request is absent from the snapshot batch");
        }
        if (!seen.emplace(message.request_id, true).second) {
          throw std::runtime_error(
              "Verifier update request identity is duplicated");
        }
      }
    }
  }

 private:
  void ensure_aligned() const {
    if (!aligned_) {
      throw std::runtime_error(
          "Verifier snapshot batch must be aligned before consumption");
    }
  }

  py::array_t<int64_t> array_view(
      std::vector<int64_t>& values,
      const std::vector<py::ssize_t>& shape) {
    std::vector<py::ssize_t> strides(shape.size(), 0);
    py::ssize_t stride = static_cast<py::ssize_t>(sizeof(int64_t));
    for (size_t i = shape.size(); i > 0; --i) {
      strides[i - 1] = stride;
      stride *= shape[i - 1];
    }
    return py::array_t<int64_t>(
        shape,
        strides,
        values.data(),
        py::cast(shared_from_this()));
  }

  DraftTailSnapshotBatchCpp batch_;
  size_t batch_size_ = 0;
  std::vector<size_t> row_indices_;
  std::vector<bool> active_rows_;
  std::vector<int64_t> dense_draft_tokens_;
  std::vector<int64_t> draft_lens_;
  std::vector<int64_t> transport_values_;
  std::vector<int64_t> pre_committed_lens_;
  std::unordered_map<std::string, size_t> request_id_to_batch_index_;
  int64_t materialized_num_draft_tokens_ = -1;
  int64_t materialized_pad_token_ = 0;
  int64_t transport_width_ = -1;
  bool aligned_ = false;
  bool materialized_ = false;
};

py::tuple draft_key_row(const DraftReqKeyCpp& key) {
  return py::make_tuple(key.request_id, key.src_verifier_rank);
}

py::tuple sync_row(const DraftSyncCpp& msg) {
  return py::make_tuple(
      msg.request_id,
      msg.src_verifier_rank,
      msg.dst_drafter_rank,
      msg.prompt_token_ids,
      msg.committed_outputs);
}

py::tuple commit_row(const VerifyCommitCpp& msg) {
  return py::make_tuple(
      msg.request_id,
      msg.src_verifier_rank,
      msg.dst_drafter_rank,
      msg.pre_verify_committed_len,
      msg.committed_tokens);
}

py::tuple close_row(const DraftCloseCpp& msg) {
  return py::make_tuple(
      msg.request_id,
      msg.src_verifier_rank,
      msg.dst_drafter_rank,
      msg.reason);
}

py::tuple control_batch_native_rows(const DraftControlBatchCpp& batch) {
  py::list sync_rows;
  py::list commit_rows;
  py::list close_rows;
  for (const auto& msg : batch.sync_messages) sync_rows.append(sync_row(msg));
  for (const auto& msg : batch.verify_commit_messages) {
    commit_rows.append(commit_row(msg));
  }
  for (const auto& msg : batch.close_messages) close_rows.append(close_row(msg));
  return py::make_tuple(
      batch.dst_drafter_rank, sync_rows, commit_rows, close_rows);
}

py::tuple segment_row(const VerifierCommitSegmentCpp& segment) {
  return py::make_tuple(
      segment.draft_key.request_id,
      segment.draft_key.src_verifier_rank,
      segment.dst_drafter_rank,
      segment.pre_verify_committed_len,
      segment.committed_tokens);
}

py::tuple ready_controls_native_rows(const ReadyDraftControlsCpp& ready) {
  py::list sync_rows;
  py::list close_rows;
  py::list segment_rows;
  for (const auto& msg : ready.sync_messages) sync_rows.append(sync_row(msg));
  for (const auto& key : ready.close_keys) close_rows.append(draft_key_row(key));
  for (const auto& segment : ready.ready_commit_segments) {
    segment_rows.append(segment_row(segment));
  }
  return py::make_tuple(sync_rows, close_rows, segment_rows);
}

class DraftTailBufferCore {
 public:
  DraftTailBufferCore(int64_t verifier_rank, int64_t required_tail_len)
      : verifier_rank_(checked_transport_rank(verifier_rank, "Verifier rank")),
        required_tail_len_(std::max<int64_t>(0, required_tail_len)) {}

  void close() {
    std::lock_guard<std::mutex> guard(mu_);
    closed_ = true;
    states_.clear();
    cv_.notify_all();
  }

  bool has_request(const std::string& request_id) {
    std::lock_guard<std::mutex> guard(mu_);
    return states_.count(request_id) != 0;
  }

  int64_t get_committed_len(const std::string& request_id) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = states_.find(request_id);
    return it == states_.end() ? -1 : it->second.committed_len;
  }

  std::vector<int64_t> query_committed_lens(
      const std::vector<std::string>& request_ids) {
    std::lock_guard<std::mutex> guard(mu_);
    std::vector<int64_t> out;
    out.reserve(request_ids.size());
    for (const auto& request_id : request_ids) {
      auto it = states_.find(request_id);
      out.push_back(it == states_.end() ? -1 : it->second.committed_len);
    }
    return out;
  }

  std::vector<GpuDraftTailStateCpp> get_gpu_publish_states(
      const std::vector<std::string>& request_ids,
      int64_t tail_capacity) {
    std::lock_guard<std::mutex> guard(mu_);
    ensure_open_locked();
    std::vector<GpuDraftTailStateCpp> out;
    out.reserve(request_ids.size());
    for (const auto& request_id : request_ids) {
      GpuDraftTailStateCpp row;
      row.request_id = request_id;
      auto it = states_.find(request_id);
      if (it != states_.end()) {
        const auto& state = it->second;
        row.exists = true;
        row.prompt_len = state.prompt_len;
        row.committed_len = state.committed_len;
        if (static_cast<int64_t>(state.tail_tokens.size()) > tail_capacity) {
          throw std::runtime_error(
              "GPU draft-tail capacity exceeded: request_id=" + request_id +
              " tail_len=" + std::to_string(state.tail_tokens.size()) +
              " capacity=" + std::to_string(tail_capacity));
        }
        row.raw_tail_len = static_cast<int64_t>(state.tail_tokens.size());
        row.consumable_tail_len = state.pending_expected_tokens.empty()
            ? row.raw_tail_len
            : 0;
        row.tail_tokens.reserve(static_cast<size_t>(row.raw_tail_len));
        for (int64_t i = 0; i < row.raw_tail_len; ++i) {
          row.tail_tokens.push_back(state.tail_tokens[static_cast<size_t>(i)]);
        }
      }
      out.push_back(std::move(row));
    }
    return out;
  }

  void apply_control_batch_native(const DraftControlBatchCpp& batch) {
    {
      std::lock_guard<std::mutex> guard(mu_);
      ensure_open_locked();
      for (const auto& msg : batch.sync_messages) open_request_locked(msg);
      for (const auto& msg : batch.verify_commit_messages) apply_commit_locked(msg);
      for (const auto& msg : batch.close_messages) close_request_locked(msg);
      cv_.notify_all();
    }
  }

  void open_request_rows_native(std::vector<DraftSyncOpenCpp> rows) {
    std::lock_guard<std::mutex> guard(mu_);
    ensure_open_locked();
    states_.reserve(states_.size() + rows.size());
    for (auto& row : rows) {
      RequestDraftTailStateCpp state;
      state.drafter_rank = row.dst_drafter_rank;
      state.prompt_len = row.prompt_len;
      state.committed_len = row.committed_len;
      state.can_accept_prefix_len = state.committed_len;
      states_.insert_or_assign(std::move(row.request_id), std::move(state));
    }
    cv_.notify_all();
  }

  void append_draft_stream_batch_native(const DraftTailStreamOutputBatchCpp& batch) {
    if (batch.outputs.empty()) return;
    {
      std::lock_guard<std::mutex> guard(mu_);
      ensure_open_locked();
      for (const auto& output : batch.outputs) push_one_locked(output);
      cv_.notify_all();
    }
  }

  void wait_for_draft_tokens(const std::vector<std::string>& rids, int64_t min_draft_tokens) {
    min_draft_tokens = std::max<int64_t>(0, min_draft_tokens);
    if (min_draft_tokens <= 0) return;
    std::unique_lock<std::mutex> lock(mu_);
    ensure_open_locked();
    cv_.wait(lock, [&] { return closed_ || has_min_draft_tokens_locked(rids, min_draft_tokens); });
    if (closed_) {
      throw std::runtime_error("DraftTailBuffer closed while waiting for draft tail tokens.");
    }
  }

  DraftTailSnapshotBatchCpp get_draft_snapshots_native(
      const std::vector<std::string>& rids,
      bool allow_partial,
      int64_t max_tail_len) {
    int64_t tail_cap = std::max<int64_t>(-1, max_tail_len);
    std::unique_lock<std::mutex> lock(mu_);
    ensure_open_locked();
    int64_t wait_ns = 0;
    if (!allow_partial) {
      int64_t required_tail_len = required_tail_len_;
      if (tail_cap >= 0) {
        required_tail_len = std::min<int64_t>(required_tail_len, tail_cap);
      }
      int64_t min_raw_tail_len =
          std::max<int64_t>(tail_cap == 0 ? 0 : 1, required_tail_len);
      while (!closed_ && !has_min_draft_tokens_locked(rids, min_raw_tail_len)) {
        int64_t wait_start_ns = now_ns();
        cv_.wait(lock);
        wait_ns += now_ns() - wait_start_ns;
      }
      if (closed_) {
        throw std::runtime_error("DraftTailBuffer closed while waiting for draft tail tokens.");
      }
    }

    std::vector<DraftTailSnapshotCpp> snapshots;
    snapshots.reserve(rids.size());
    for (const auto& rid : rids) {
      auto it = states_.find(rid);
      if (it == states_.end()) {
        throw std::runtime_error("unexpected request_id=" + rid);
      }
      const auto& state = it->second;
      DraftTailSnapshotCpp snapshot;
      snapshot.request_id = rid;
      snapshot.committed_len = state.committed_len;
      snapshot.tail_tokens = state.consumable_tail_tokens();
      snapshot.num_consumable_drafts =
          static_cast<int64_t>(snapshot.tail_tokens.size());
      if (tail_cap >= 0 && static_cast<int64_t>(snapshot.tail_tokens.size()) > tail_cap) {
        snapshot.tail_tokens.resize(static_cast<size_t>(tail_cap));
      }
      snapshot.raw_tail_len = static_cast<int64_t>(state.tail_tokens.size());
      snapshots.push_back(std::move(snapshot));
    }
    return {std::move(snapshots), wait_ns};
  }

 private:
  void open_request_locked(const DraftSyncCpp& message) {
    if (message.src_verifier_rank != verifier_rank_) {
      throw std::runtime_error(
          "DraftSync belongs to a different verifier");
    }
    RequestDraftTailStateCpp state;
    state.drafter_rank = message.dst_drafter_rank;
    state.prompt_len = static_cast<int64_t>(message.prompt_token_ids.size());
    state.committed_len = static_cast<int64_t>(message.committed_outputs.size());
    state.can_accept_prefix_len = state.committed_len;
    states_[message.request_id] = std::move(state);
  }

  void apply_commit_locked(const VerifyCommitCpp& message) {
    message.validate();

    if (message.src_verifier_rank != verifier_rank_) {
      throw std::runtime_error(
          "VerifyCommit belongs to a different verifier");
    }

    auto it = states_.find(message.request_id);
    if (it == states_.end()) return;
    auto& state = it->second;
    if (message.dst_drafter_rank != state.drafter_rank) {
      throw std::runtime_error(
          "VerifyCommit targets a different drafter");
    }
    int64_t pending_expected_len_before = static_cast<int64_t>(state.pending_expected_tokens.size());
    int64_t target_committed_len = state.committed_len + pending_expected_len_before;
    if (message.pre_verify_committed_len != target_committed_len) {
      throw std::runtime_error(
          "VerifyCommit pre-verify prefix does not match draft-tail confirmed plus pending prefix");
    }

    int64_t raw_tail_len_before = static_cast<int64_t>(state.tail_tokens.size());

    if (pending_expected_len_before) {
      if (!state.tail_tokens.empty()) {
        throw std::runtime_error("Draft tail tokens must be empty while expected prefix tokens are pending");
      }
      for (int32_t token : message.committed_tokens) state.pending_expected_tokens.push_back(token);
      return;
    }

    int64_t matched_tail_len = 0;
    int64_t max_possible_match_len =
        std::min<int64_t>(static_cast<int64_t>(message.committed_tokens.size()), raw_tail_len_before);
    while (matched_tail_len < max_possible_match_len &&
           state.tail_tokens[matched_tail_len] == message.committed_tokens[matched_tail_len]) {
      ++matched_tail_len;
    }
    if (matched_tail_len) {
      state.tail_tokens.erase(state.tail_tokens.begin(), state.tail_tokens.begin() + matched_tail_len);
      state.committed_len += matched_tail_len;
    }

    if (matched_tail_len < static_cast<int64_t>(message.committed_tokens.size())) {
      if (matched_tail_len < raw_tail_len_before) {
        state.can_accept_prefix_len = state.committed_len;
      }
      state.tail_tokens.clear();
      for (size_t i = static_cast<size_t>(matched_tail_len); i < message.committed_tokens.size(); ++i) {
        state.pending_expected_tokens.push_back(message.committed_tokens[i]);
      }
    }
  }

  void close_request_locked(const DraftCloseCpp& message) {
    if (message.src_verifier_rank != verifier_rank_) {
      throw std::runtime_error("DraftClose belongs to a different verifier");
    }
    auto it = states_.find(message.request_id);
    if (it != states_.end() &&
        message.dst_drafter_rank != it->second.drafter_rank) {
      throw std::runtime_error("DraftClose targets a different drafter");
    }
    states_.erase(message.request_id);
  }

  void push_one_locked(const DraftTailStreamOutputCpp& output) {
    const auto& request_id = output.request_id;
    int64_t base_committed_len = output.base_committed_len;
    int64_t token_pos = output.new_token_pos;
    int32_t token_id = output.new_token;
    int32_t src_drafter_rank = output.src_drafter_rank;
    int32_t dst_verifier_rank = output.dst_verifier_rank;

    if (dst_verifier_rank != verifier_rank_) {
      throw std::runtime_error("Draft stream output targets the wrong verifier");
    }
    auto it = states_.find(request_id);
    if (it == states_.end()) return;

    auto& state = it->second;
    int64_t state_committed_len = state.committed_len;
    int64_t can_accept_prefix_len = state.can_accept_prefix_len;
    int64_t tail_len_before = static_cast<int64_t>(state.tail_tokens.size());
    int64_t buffer_end_len = state_committed_len + tail_len_before;

    if (src_drafter_rank != state.drafter_rank) {
      throw std::runtime_error("Unexpected draft stream drafter rank");
    }

    if (!state.pending_expected_tokens.empty()) {
      if (!state.tail_tokens.empty()) {
        throw std::runtime_error("Draft tail tokens must be empty while expected prefix tokens are pending");
      }
      if (base_committed_len < can_accept_prefix_len) return;
      if (token_pos < state_committed_len) return;
      if (base_committed_len > state_committed_len) return;
      if (token_pos > state_committed_len) return;
      int32_t expected_token_id = state.pending_expected_tokens.front();
      if (token_id == expected_token_id) {
        state.pending_expected_tokens.pop_front();
        state.committed_len += 1;
        return;
      }
      state.can_accept_prefix_len = state.committed_len;
      return;
    }

    if (base_committed_len > state_committed_len) {
      throw std::runtime_error("Draft stream base is ahead of verifier state");
    }
    if (base_committed_len < can_accept_prefix_len) return;
    if (token_pos < state_committed_len) return;

    if (token_pos < buffer_end_len) {
      int32_t existing_token_id = state.tail_tokens[token_pos - state_committed_len];
      if (existing_token_id != token_id) {
        throw std::runtime_error("Draft stream token conflicts with buffered tail");
      }
      return;
    }

    if (token_pos > buffer_end_len) {
      if (base_committed_len == state_committed_len) {
        throw std::runtime_error("Draft stream token skips buffered tail");
      }
      return;
    }

    state.tail_tokens.push_back(token_id);
  }

  bool has_min_draft_tokens_locked(const std::vector<std::string>& rids, int64_t min_draft_tokens) const {
    for (const auto& rid : rids) {
      auto it = states_.find(rid);
      if (it == states_.end()) {
        throw std::runtime_error("unexpected request_id=" + rid);
      }
      if (!it->second.pending_expected_tokens.empty()) return false;
      if (static_cast<int64_t>(it->second.tail_tokens.size()) < min_draft_tokens) return false;
    }
    return true;
  }

  void ensure_open_locked() const {
    if (closed_) {
      throw std::runtime_error("DraftTailBuffer is closed");
    }
  }

  int32_t verifier_rank_;
  int64_t required_tail_len_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool closed_ = false;
  std::unordered_map<std::string, RequestDraftTailStateCpp> states_;
};

// Drafter-side protocol owner. Network ingress, the mirrored local transcript,
// pending verifier commits, and ready-action extraction all share one lock so
// probe/consume is atomic with respect to newly arriving controls and locally
// published draft tokens.
class DrafterDataPlaneCore {
 public:
  bool is_empty() {
    std::lock_guard<std::mutex> guard(mu_);
    return sync_messages_.empty() && verifier_commit_segments_.empty() && close_keys_.empty();
  }

  int64_t pending_control_count() {
    std::lock_guard<std::mutex> guard(mu_);
    return static_cast<int64_t>(sync_messages_.size() + verifier_commit_segments_.size() + close_keys_.size());
  }

  bool wait_for_pending_control(int64_t timeout_us) {
    if (timeout_us < 0) {
      throw std::runtime_error("Control wait timeout must be non-negative");
    }
    std::unique_lock<std::mutex> lock(mu_);
    auto ready = [&] {
      return !sync_messages_.empty() || !verifier_commit_segments_.empty() ||
          !close_keys_.empty();
    };
    if (ready()) return true;
    return control_cv_.wait_for(
        lock, std::chrono::microseconds(timeout_us), ready);
  }

  void add_control_batch(const DraftControlBatchCpp& batch) {
    {
      std::lock_guard<std::mutex> guard(mu_);
      close_keys_.reserve(close_keys_.size() + batch.close_messages.size());
      sync_messages_.reserve(sync_messages_.size() + batch.sync_messages.size());
      verifier_commit_segments_.reserve(
          verifier_commit_segments_.size() + batch.verify_commit_messages.size());
      for (const auto& msg : batch.close_messages) {
        add_close_key_locked(msg.draft_key());
      }
      for (const auto& msg : batch.sync_messages) {
        if (close_keys_.count(msg.draft_key().map_key()) == 0) {
          sync_messages_.push_back(msg);
        }
      }
      for (const auto& msg : batch.verify_commit_messages) {
        add_verify_commit_locked(msg);
      }
    }
    control_cv_.notify_all();
  }

  void append_draft_outputs(
      const DraftTailStreamOutputBatchCpp& batch,
      bool strict_local_contract = false) {
    if (batch.outputs.empty()) return;
    std::lock_guard<std::mutex> guard(mu_);
    transcript_.append_draft_outputs(batch, strict_local_contract);
  }

  DraftControlProbeCpp probe_pending_controls() {
    std::lock_guard<std::mutex> guard(mu_);
    if (outstanding_probe_) {
      throw std::runtime_error(
          "Drafter data plane already has an outstanding control probe");
    }
    if (next_probe_id_ == 0) {
      throw std::runtime_error("Drafter control probe id exhausted");
    }

    DraftControlProbeCpp probe;
    probe.probe_id = next_probe_id_++;
    probe.sync_messages = sync_messages_;
    probe.close_keys.reserve(close_keys_.size());
    for (const auto& kv : close_keys_) probe.close_keys.push_back(kv.second);
    probe.verifier_commit_segments.reserve(verifier_commit_segments_.size());
    for (const auto& map_key : commit_key_order_) {
      auto it = verifier_commit_segments_.find(map_key);
      if (it != verifier_commit_segments_.end()) {
        probe.verifier_commit_segments.push_back(it->second);
      }
    }
    outstanding_probe_ = std::make_unique<DraftControlProbeCpp>(probe);
    return probe;
  }

  ReadyDrafterActionsCpp consume_ready_controls(
      uint64_t probe_id,
      const std::string& eligibility_mask) {
    std::lock_guard<std::mutex> guard(mu_);
    if (!outstanding_probe_ || outstanding_probe_->probe_id != probe_id) {
      throw std::runtime_error("Drafter control probe id mismatch");
    }
    const auto& control_probe = *outstanding_probe_;
    ReadyDrafterActionsCpp ready;
    if (eligibility_mask.size() !=
        control_probe.verifier_commit_segments.size()) {
      throw std::runtime_error(
          "Drafter control eligibility mask length does not match probe rows");
    }
    for (char value : eligibility_mask) {
      if (value != 0 && value != 1) {
        throw std::runtime_error(
            "Drafter control eligibility mask values must be 0 or 1");
      }
    }

    std::unordered_map<std::string, bool> probed_close_keys;
    probed_close_keys.reserve(control_probe.close_keys.size());
    for (const auto& key : control_probe.close_keys) {
      probed_close_keys.emplace(key.map_key(), true);
      ready.close_keys.push_back(key);
    }

    // A close received after probe cancels same-request work visible in that
    // probe, but the new close itself remains pending for the next round.
    std::unordered_map<std::string, bool> cancelled_keys;
    for (const auto& kv : close_keys_) {
      if (probed_close_keys.count(kv.first) == 0) {
        cancelled_keys.emplace(kv.first, true);
      }
    }

    for (const auto& sync : control_probe.sync_messages) {
      auto map_key = sync.draft_key().map_key();
      if (cancelled_keys.count(map_key) != 0) continue;
      if (transcript_.contains(sync.draft_key())) {
        throw std::runtime_error(
            "DraftSync received for an existing mirrored transcript");
      }
      ready.sync_messages.push_back(sync);
    }

    struct PlannedCommit {
      std::string map_key;
      VerifierCommitSegmentCpp segment;
      DraftCommitProbeCpp probe;
    };
    std::vector<PlannedCommit> planned;
    planned.reserve(control_probe.verifier_commit_segments.size());
    for (size_t probe_index = 0;
         probe_index < control_probe.verifier_commit_segments.size();
         ++probe_index) {
      const auto& segment =
          control_probe.verifier_commit_segments[probe_index];
      auto map_key = segment.draft_key.map_key();
      if (close_keys_.count(map_key) != 0 ||
          eligibility_mask[probe_index] == 0 ||
          !transcript_.contains(segment.draft_key)) {
        continue;
      }
      auto commit_probe = transcript_.probe(segment);
      if (!commit_probe.ready) continue;

      auto live_it = verifier_commit_segments_.find(map_key);
      if (live_it == verifier_commit_segments_.end()) {
        throw std::runtime_error(
            "Probed verifier commit segment disappeared before consume");
      }
      const auto& live = live_it->second;
      if (live.pre_verify_committed_len != segment.pre_verify_committed_len ||
          live.dst_drafter_rank != segment.dst_drafter_rank ||
          live.committed_tokens.size() < segment.committed_tokens.size() ||
          !std::equal(
              segment.committed_tokens.begin(),
              segment.committed_tokens.end(),
              live.committed_tokens.begin())) {
        throw std::runtime_error(
            "Probed verifier commit segment changed before consume");
      }
      planned.push_back(PlannedCommit{map_key, segment, commit_probe});
    }

    // Remove only sync messages visible to this probe. Later arrivals remain
    // queued for the next probe.
    for (const auto& probed_sync : control_probe.sync_messages) {
      auto target_key = probed_sync.draft_key().map_key();
      auto it = std::find_if(
          sync_messages_.begin(), sync_messages_.end(),
          [&](const DraftSyncCpp& live) {
            return live.draft_key().map_key() == target_key;
          });
      if (it != sync_messages_.end()) sync_messages_.erase(it);
    }
    for (const auto& key : control_probe.close_keys) {
      close_keys_.erase(key.map_key());
      transcript_.close(key);
    }
    for (const auto& sync : ready.sync_messages) transcript_.open(sync);

    for (const auto& item : planned) {
      auto live_it = verifier_commit_segments_.find(item.map_key);
      if (live_it == verifier_commit_segments_.end()) {
        throw std::runtime_error(
            "Verifier commit segment disappeared during consume");
      }
      auto action = transcript_.consume(item.segment, item.probe);
      int64_t consumed = action.committed_token_count();
      live_it->second.discard_prefix(consumed);
      ready.commit_actions.push_back(std::move(action));
      if (live_it->second.committed_tokens.empty()) {
        erase_commit_segment_locked(item.map_key);
      }
    }

    outstanding_probe_.reset();
    return ready;
  }

  std::vector<VerifierCommitSegmentCpp> snapshot_pending_commit_segments_native() {
    std::lock_guard<std::mutex> guard(mu_);
    if (outstanding_probe_) {
      throw std::runtime_error(
          "Cannot use legacy control snapshot with an outstanding probe");
    }
    std::vector<VerifierCommitSegmentCpp> out;
    out.reserve(verifier_commit_segments_.size());
    for (const auto& map_key : commit_key_order_) {
      auto it = verifier_commit_segments_.find(map_key);
      if (it != verifier_commit_segments_.end()) out.push_back(it->second);
    }
    return out;
  }

  ReadyDraftControlsCpp extract_ready_controls_native(const std::vector<ExtractDecisionCpp>& decisions) {
    ReadyDraftControlsCpp ready;
    std::lock_guard<std::mutex> guard(mu_);
    if (outstanding_probe_) {
      throw std::runtime_error(
          "Cannot use legacy control extraction with an outstanding probe");
    }
    for (const auto& kv : close_keys_) {
      ready.close_keys.push_back(kv.second);
      transcript_.close(kv.second);
    }
    close_keys_.clear();
    ready.sync_messages = std::move(sync_messages_);
    sync_messages_.clear();
    for (const auto& sync : ready.sync_messages) transcript_.open(sync);

    for (const auto& decision : decisions) {
      if (decision.consumable_len <= 0) continue;
      auto it = verifier_commit_segments_.find(decision.key.map_key());
      if (it == verifier_commit_segments_.end()) continue;
      auto& segment = it->second;
      if (segment.pre_verify_committed_len != decision.pre_verify_committed_len ||
          segment.dst_drafter_rank != decision.dst_drafter_rank) {
        continue;
      }
      transcript_.consume_legacy(segment, decision.consumable_len);
      ready.ready_commit_segments.push_back(segment.extract_prefix(decision.consumable_len));
      if (segment.committed_tokens.empty()) {
        erase_commit_segment_locked(decision.key.map_key());
      }
    }
    return ready;
  }

 private:
  void add_close_key_locked(const DraftReqKeyCpp& key) {
    close_keys_[key.map_key()] = key;
    erase_commit_segment_locked(key.map_key());
    sync_messages_.erase(
        std::remove_if(
            sync_messages_.begin(), sync_messages_.end(),
            [&](const DraftSyncCpp& msg) { return msg.draft_key().map_key() == key.map_key(); }),
        sync_messages_.end());
  }

  void add_verify_commit_locked(const VerifyCommitCpp& message) {
    auto key = message.draft_key();
    if (close_keys_.count(key.map_key())) return;
    auto it = verifier_commit_segments_.find(key.map_key());
    if (it == verifier_commit_segments_.end()) {
      VerifierCommitSegmentCpp segment;
      segment.draft_key = key;
      segment.dst_drafter_rank = message.dst_drafter_rank;
      segment.pre_verify_committed_len = message.pre_verify_committed_len;
      segment.append_message(message);
      verifier_commit_segments_[key.map_key()] = std::move(segment);
      commit_key_order_.push_back(key.map_key());
      return;
    }
    it->second.append_message(message);
  }

  void erase_commit_segment_locked(const std::string& map_key) {
    verifier_commit_segments_.erase(map_key);
    commit_key_order_.erase(
        std::remove(
            commit_key_order_.begin(), commit_key_order_.end(), map_key),
        commit_key_order_.end());
  }

  std::mutex mu_;
  std::condition_variable control_cv_;
  DraftTranscriptMirror transcript_;
  std::vector<DraftSyncCpp> sync_messages_;
  std::unordered_map<std::string, VerifierCommitSegmentCpp> verifier_commit_segments_;
  std::vector<std::string> commit_key_order_;
  std::unordered_map<std::string, DraftReqKeyCpp> close_keys_;
  uint64_t next_probe_id_ = 1;
  std::unique_ptr<DraftControlProbeCpp> outstanding_probe_;
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

struct GpuDraftTailPublishRowCpp {
  int64_t seat = -1;
  int64_t publish_seq = 0;
  int64_t request_epoch = -1;
  int64_t prompt_len = -1;
  int64_t committed_len = -1;
  int64_t raw_tail_len = 0;
  int64_t consumable_tail_len = 0;
  std::vector<int64_t> tail_tokens;
};

class GpuDraftTailBufferCore {
 public:
  GpuDraftTailBufferCore(
      int64_t device_index,
      int64_t num_seats,
      int64_t num_draft_tokens,
      uintptr_t landing_stream,
      uintptr_t versions,
      uintptr_t publish_seqs,
      uintptr_t request_epochs,
      uintptr_t active_request_epochs,
      uintptr_t prompt_lens,
      uintptr_t committed_lens,
      uintptr_t raw_tail_lens,
      uintptr_t consumable_tail_lens,
      uintptr_t tail_tokens)
      : device_index_(device_index),
        num_seats_(num_seats),
        num_draft_tokens_(num_draft_tokens),
        tail_capacity_(2 * num_draft_tokens + 1),
        landing_stream_(reinterpret_cast<void*>(landing_stream)),
        versions_(reinterpret_cast<int64_t*>(versions)),
        publish_seqs_(reinterpret_cast<int64_t*>(publish_seqs)),
        request_epochs_(reinterpret_cast<int64_t*>(request_epochs)),
        active_request_epochs_(
            reinterpret_cast<int64_t*>(active_request_epochs)),
        prompt_lens_(reinterpret_cast<int64_t*>(prompt_lens)),
        committed_lens_(reinterpret_cast<int64_t*>(committed_lens)),
        raw_tail_lens_(reinterpret_cast<int64_t*>(raw_tail_lens)),
        consumable_tail_lens_(
            reinterpret_cast<int64_t*>(consumable_tail_lens)),
        tail_tokens_(reinterpret_cast<int64_t*>(tail_tokens)) {
    if (device_index_ < 0 || num_seats_ <= 0 || num_draft_tokens_ <= 0) {
      throw std::runtime_error(
          "GPU draft-tail dimensions and device must be positive");
    }
    if (landing_stream_ == nullptr || versions_ == nullptr ||
        publish_seqs_ == nullptr || request_epochs_ == nullptr ||
        active_request_epochs_ == nullptr || prompt_lens_ == nullptr ||
        committed_lens_ == nullptr || raw_tail_lens_ == nullptr ||
        consumable_tail_lens_ == nullptr || tail_tokens_ == nullptr) {
      throw std::runtime_error(
          "GPU draft-tail buffers and landing stream must be non-null");
    }
    set_device();
    staging_slots_.reserve(kMaxStagingSlots);
    const size_t staging_words = static_cast<size_t>(num_seats_) *
        static_cast<size_t>(
            kGpuDraftTailPublishMetadataWidth + tail_capacity_);
    for (size_t i = 0; i < kInitialStagingSlots; ++i) {
      add_staging_slot(staging_words);
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
    if (request_epoch < 0) {
      throw std::runtime_error("GPU draft-tail request epoch must be non-negative");
    }
    std::lock_guard<std::mutex> guard(binding_mu_);
    auto seat_it = seat_bindings_.find(seat);
    if (seat_it != seat_bindings_.end()) {
      const auto& previous = seat_it->second;
      if (request_epoch <= previous.request_epoch &&
          previous.request_id != request_id) {
        throw std::runtime_error(
            "GPU draft-tail seat reuse requires a newer request epoch");
      }
      request_bindings_.erase(previous.request_id);
    }
    auto request_it = request_bindings_.find(request_id);
    if (request_it != request_bindings_.end() &&
        (request_it->second.seat != seat ||
         request_it->second.request_epoch != request_epoch)) {
      throw std::runtime_error(
          "GPU draft-tail request identity was rebound inconsistently");
    }
    request_bindings_[request_id] = {seat, request_epoch};
    seat_bindings_[seat] = {request_id, request_epoch};
  }

  void publish_requests(
      DraftTailBufferCore& cpu_core,
      const std::vector<std::string>& request_ids) {
    if (request_ids.empty()) return;
    auto states = cpu_core.get_gpu_publish_states(
        request_ids, tail_capacity_);
    std::unordered_map<int64_t, GpuDraftTailPublishRowCpp> rows_by_seat;
    {
      std::lock_guard<std::mutex> guard(binding_mu_);
      for (const auto& state : states) {
        auto binding_it = request_bindings_.find(state.request_id);
        if (binding_it == request_bindings_.end()) continue;
        const auto binding = binding_it->second;
        auto seat_it = seat_bindings_.find(binding.seat);
        if (seat_it == seat_bindings_.end() ||
            seat_it->second.request_id != state.request_id ||
            seat_it->second.request_epoch != binding.request_epoch) {
          continue;
        }

        GpuDraftTailPublishRowCpp row;
        row.seat = binding.seat;
        row.publish_seq = next_publish_seq_++;
        row.request_epoch = state.exists ? binding.request_epoch : -1;
        row.prompt_len = state.exists ? state.prompt_len : -1;
        row.committed_len = state.exists ? state.committed_len : -1;
        row.raw_tail_len = state.raw_tail_len;
        row.consumable_tail_len = state.consumable_tail_len;
        row.tail_tokens = state.tail_tokens;
        rows_by_seat.insert_or_assign(row.seat, std::move(row));
        if (!state.exists) {
          request_bindings_.erase(binding_it);
          seat_bindings_.erase(seat_it);
        }
      }
    }
    if (rows_by_seat.empty()) return;
    std::vector<GpuDraftTailPublishRowCpp> rows;
    rows.reserve(rows_by_seat.size());
    for (auto& item : rows_by_seat) rows.push_back(std::move(item.second));
    publish_rows(rows);
  }

  void select_snapshot(
      uintptr_t gpu_seats,
      uintptr_t expected_request_epochs,
      uintptr_t seq_lens,
      uintptr_t bonus_tokens,
      bool bonus_tokens_are_int32,
      uintptr_t compact_out,
      uintptr_t logical_committed_lens_out,
      int64_t batch_size,
      uintptr_t verify_stream) {
    if (batch_size < 0) {
      throw std::runtime_error("GPU draft-tail batch size must be non-negative");
    }
    if (batch_size == 0) return;
    if (gpu_seats == 0 || seq_lens == 0 || bonus_tokens == 0 ||
        compact_out == 0) {
      throw std::runtime_error(
          "GPU draft-tail select received a null required pointer");
    }
    set_device();
    launch_select_gpu_draft_tail(
        reinterpret_cast<const int64_t*>(gpu_seats),
        expected_request_epochs == 0
            ? nullptr
            : reinterpret_cast<const int64_t*>(expected_request_epochs),
        reinterpret_cast<const int64_t*>(seq_lens),
        reinterpret_cast<const void*>(bonus_tokens),
        bonus_tokens_are_int32,
        reinterpret_cast<int64_t*>(compact_out),
        logical_committed_lens_out == 0
            ? nullptr
            : reinterpret_cast<int64_t*>(logical_committed_lens_out),
        batch_size,
        num_seats_,
        num_draft_tokens_,
        tail_capacity_,
        versions_,
        request_epochs_,
        active_request_epochs_,
        prompt_lens_,
        committed_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        tail_tokens_,
        reinterpret_cast<void*>(verify_stream));
  }

  void close() {
    std::lock_guard<std::mutex> guard(close_mu_);
    if (closed_) return;
    set_device();
    // Shutdown is the one place where blocking the landing stream is
    // required: every event and allocation below may still be referenced by
    // queued H2D copies or publish kernels.
    check_cuda(
        cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaStreamSynchronize GPU draft-tail landing stream");
    closed_ = true;
    for (auto& slot : staging_slots_) {
      if (slot.event != nullptr) cudaEventDestroy(slot.event);
      if (slot.device != nullptr) cudaFree(slot.device);
      if (slot.host != nullptr) cudaFreeHost(slot.host);
      slot = {};
    }
    staging_slots_.clear();
    staging_slot_count_.store(0, std::memory_order_release);
  }

 private:
  struct SeatBindingCpp {
    std::string request_id;
    int64_t request_epoch = -1;
  };

  struct StagingSlotCpp {
    int64_t* host = nullptr;
    int64_t* device = nullptr;
    size_t capacity_words = 0;
    cudaEvent_t event = nullptr;
    bool in_flight = false;
  };

  static constexpr size_t kInitialStagingSlots = 8;
  static constexpr size_t kMaxStagingSlots = 64;

  static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
  }

  void set_device() const {
    check_cuda(cudaSetDevice(static_cast<int>(device_index_)), "cudaSetDevice");
  }

  StagingSlotCpp make_staging_slot() {
    StagingSlotCpp slot;
    check_cuda(
        cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming),
        "cudaEventCreateWithFlags");
    return slot;
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

  void publish_rows(const std::vector<GpuDraftTailPublishRowCpp>& rows) {
    set_device();
    const size_t row_width = static_cast<size_t>(
        kGpuDraftTailPublishMetadataWidth + tail_capacity_);
    const size_t required_words = rows.size() * row_width;
    auto& slot = acquire_staging_slot(required_words);
    std::fill(slot.host, slot.host + required_words, 0);
    for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
      const auto& row = rows[row_index];
      int64_t* dst = slot.host + row_index * row_width;
      dst[0] = row.seat;
      dst[1] = row.publish_seq;
      dst[2] = row.request_epoch;
      dst[3] = row.prompt_len;
      dst[4] = row.committed_len;
      dst[5] = row.raw_tail_len;
      dst[6] = row.consumable_tail_len;
      if (row.tail_tokens.size() > static_cast<size_t>(tail_capacity_)) {
        throw std::runtime_error(
            "GPU draft-tail publish row exceeds its fixed capacity");
      }
      const size_t token_count = row.tail_tokens.size();
      std::copy(
          row.tail_tokens.begin(),
          row.tail_tokens.begin() + token_count,
          dst + kGpuDraftTailPublishMetadataWidth);
    }
    const size_t bytes = required_words * sizeof(int64_t);
    check_cuda(
        cudaMemcpyAsync(
            slot.device,
            slot.host,
            bytes,
            cudaMemcpyHostToDevice,
            reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaMemcpyAsync");
    launch_publish_gpu_draft_tail(
        slot.device,
        static_cast<int64_t>(rows.size()),
        tail_capacity_,
        versions_,
        publish_seqs_,
        request_epochs_,
        prompt_lens_,
        committed_lens_,
        raw_tail_lens_,
        consumable_tail_lens_,
        tail_tokens_,
        landing_stream_);
    check_cuda(
        cudaEventRecord(
            slot.event, reinterpret_cast<cudaStream_t>(landing_stream_)),
        "cudaEventRecord");
    slot.in_flight = true;
  }

  int64_t device_index_;
  int64_t num_seats_;
  int64_t num_draft_tokens_;
  int64_t tail_capacity_;
  void* landing_stream_;
  int64_t* versions_;
  int64_t* publish_seqs_;
  int64_t* request_epochs_;
  int64_t* active_request_epochs_;
  int64_t* prompt_lens_;
  int64_t* committed_lens_;
  int64_t* raw_tail_lens_;
  int64_t* consumable_tail_lens_;
  int64_t* tail_tokens_;
  std::mutex binding_mu_;
  std::unordered_map<std::string, GpuDraftTailBindingCpp> request_bindings_;
  std::unordered_map<int64_t, SeatBindingCpp> seat_bindings_;
  int64_t next_publish_seq_ = 1;
  std::vector<StagingSlotCpp> staging_slots_;
  std::atomic<size_t> staging_slot_count_{0};
  std::mutex close_mu_;
  bool closed_ = false;
};

struct QueuedFrame {
  int32_t dst_rank = -1;
  std::string frame;
  std::vector<std::string> changed_request_ids;
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

class DecoupledSpecDraftTailBuffer {
 public:
  DecoupledSpecDraftTailBuffer(int64_t verifier_rank, int64_t required_tail_len)
      : core_(std::make_shared<DraftTailBufferCore>(verifier_rank, required_tail_len)) {}

  void close() { core_->close(); }
  bool has_request(const std::string& request_id) { return core_->has_request(request_id); }
  int64_t get_committed_len(const std::string& request_id) { return core_->get_committed_len(request_id); }

  void attach_gpu_tail_buffer(
      std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer) {
    if (gpu_tail_buffer == nullptr) {
      throw std::runtime_error("Cannot attach a null GPU draft-tail buffer");
    }
    if (gpu_tail_buffer_ != nullptr) {
      throw std::runtime_error("GPU draft-tail buffer is already attached");
    }
    gpu_tail_buffer_ = std::move(gpu_tail_buffer);
  }

  void apply_control_batch_native(
      int64_t dst_drafter_rank,
      const py::sequence& sync_rows,
      const py::sequence& commit_rows,
      const py::sequence& close_rows) {
    if (py::len(sync_rows) > 0 && py::len(commit_rows) == 0 && py::len(close_rows) == 0) {
      auto rows = build_sync_open_rows_native(sync_rows);
      {
        py::gil_scoped_release release;
        core_->open_request_rows_native(std::move(rows));
      }
      return;
    }
    auto batch = build_control_batch_native(dst_drafter_rank, sync_rows, commit_rows, close_rows);
    {
      py::gil_scoped_release release;
      core_->apply_control_batch_native(batch);
    }
  }

  void append_draft_stream_batch_native(const py::sequence& rows) {
    auto batch = build_tail_stream_batch_native(rows);
    {
      py::gil_scoped_release release;
      core_->append_draft_stream_batch_native(batch);
    }
  }

  void wait_for_draft_tokens_native(const py::sequence& rids, int64_t min_draft_tokens) {
    auto rid_vec = py_string_vector(rids);
    py::gil_scoped_release release;
    core_->wait_for_draft_tokens(rid_vec, min_draft_tokens);
  }

  std::shared_ptr<VerifierSnapshotBatchCpp> get_draft_snapshot_batch(
      const py::sequence& rids,
      bool allow_partial,
      int64_t max_tail_len) {
    auto rid_vec = py_string_vector(rids);
    DraftTailSnapshotBatchCpp snapshot_batch;
    {
      py::gil_scoped_release release;
      snapshot_batch = core_->get_draft_snapshots_native(
          rid_vec,
          allow_partial,
          max_tail_len);
    }
    return std::make_shared<VerifierSnapshotBatchCpp>(
        std::move(snapshot_batch));
  }

  py::tuple get_draft_snapshots_native(
      const py::sequence& rids,
      bool allow_partial,
      int64_t max_tail_len) {
    auto rid_vec = py_string_vector(rids);
    DraftTailSnapshotBatchCpp snapshot_batch;
    {
      py::gil_scoped_release release;
      snapshot_batch = core_->get_draft_snapshots_native(
          rid_vec,
          allow_partial,
          max_tail_len);
    }
    py::list out;
    for (const auto& snapshot : snapshot_batch.snapshots) {
      out.append(py::make_tuple(
          snapshot.request_id,
          snapshot.committed_len,
          snapshot.tail_tokens,
          snapshot.raw_tail_len,
          snapshot.num_consumable_drafts));
    }
    return py::make_tuple(out, snapshot_batch.wait_ns);
  }

  py::bytes query_request_states(const py::bytes& request_frame_bytes) {
    std::string bytes = request_frame_bytes.cast<std::string>();
    std::string result;
    {
      py::gil_scoped_release release;
      auto request_frame = decode_local_frame(bytes, kFrameKindRequest);
      auto request_ids = request_ids_from_local_frame(request_frame);
      auto committed_lens = core_->query_committed_lens(request_ids);
      result = encode_request_states_local_frame(
          request_frame, committed_lens);
    }
    return py::bytes(result);
  }

  std::vector<int64_t> query_committed_lens_native(
      const py::sequence& request_ids) {
    auto ids = py_string_vector(request_ids);
    py::gil_scoped_release release;
    return core_->query_committed_lens(ids);
  }

  py::bytes get_draft_snapshots_frame(
      const py::bytes& request_frame_bytes,
      bool allow_partial,
      int64_t max_tail_len) {
    std::string bytes = request_frame_bytes.cast<std::string>();
    std::string result;
    {
      py::gil_scoped_release release;
      auto request_frame = decode_local_frame(bytes, kFrameKindRequest);
      auto request_ids = request_ids_from_local_frame(request_frame);
      auto snapshot_batch = core_->get_draft_snapshots_native(
          request_ids, allow_partial, max_tail_len);
      result = encode_snapshots_local_frame(
          request_frame, snapshot_batch);
    }
    return py::bytes(result);
  }

  std::shared_ptr<DraftTailBufferCore> core() { return core_; }
  std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer() {
    return gpu_tail_buffer_;
  }

 private:
  std::shared_ptr<DraftTailBufferCore> core_;
  std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer_;
};

class DecoupledSpecDraftProxyThread {
 public:
  DecoupledSpecDraftProxyThread(
      int64_t verifier_rank,
      const std::string& bind_endpoint,
      const py::sequence& drafter_peer_rows,
      std::shared_ptr<DecoupledSpecDraftTailBuffer> draft_tail_buffer_ref,
      uintptr_t external_context = 0)
      : verifier_rank_(checked_transport_rank(verifier_rank, "Verifier rank")),
        zmq_(std::make_unique<ZmqContextOwner>(external_context)) {
    if (draft_tail_buffer_ref == nullptr) {
      throw std::runtime_error("CppDraftProxyThread requires a CppDraftTailBuffer");
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
    draft_tail_buffer_ref_ = std::move(draft_tail_buffer_ref);
    draft_tail_buffer_ = draft_tail_buffer_ref_->core();
    gpu_tail_buffer_ = draft_tail_buffer_ref_->gpu_tail_buffer();
    result_recv_socket_ = zmq_->api().socket(zmq_->ctx(), kZmqPull);
    zmq_->api().configure_socket(result_recv_socket_, kZmqPull);
    zmq_->api().bind(result_recv_socket_, bind_endpoint);
    result_bind_endpoint_ = bind_endpoint;
    for (const auto& peer : drafter_peers) {
      void* socket = zmq_->api().socket(zmq_->ctx(), kZmqPush);
      zmq_->api().configure_socket(socket, kZmqPush);
      control_send_sockets_[peer.rank] = socket;
      control_peer_endpoints_[peer.rank] = peer.endpoint;
    }
  }

  ~DecoupledSpecDraftProxyThread() { close(); }

  std::string result_bind_endpoint() const { return result_bind_endpoint_; }

  void bind_gpu_request_native(
      const std::string& request_id,
      int64_t gpu_seat,
      int64_t request_epoch) {
    if (gpu_tail_buffer_ == nullptr) {
      throw std::runtime_error(
          "Verifier data plane has no attached GPU draft-tail buffer");
    }
    gpu_tail_buffer_->bind_request(request_id, gpu_seat, request_epoch);
  }

  void start() {
    check_thread_error();
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    try {
      for (const auto& kv : control_send_sockets_) {
        zmq_->api().connect(kv.second, control_peer_endpoints_.at(kv.first));
      }
    } catch (...) {
      started_.store(false);
      throw;
    }
    closed_.store(false);
    thread_ = std::thread([this] { run_guarded(); });
  }

  void close() {
    closed_.store(true);
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    for (auto& kv : control_send_sockets_) zmq_->api().close_socket(kv.second);
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
      const py::sequence& close_rows) {
    int64_t sync_count = static_cast<int64_t>(py::len(sync_rows));
    int64_t commit_count = static_cast<int64_t>(py::len(commit_rows));
    int64_t close_count = static_cast<int64_t>(py::len(close_rows));
    int32_t dst_rank = checked_transport_rank(
        dst_drafter_rank, "Destination drafter rank");
    check_thread_error();
    if (sync_count > 0 && commit_count == 0 && close_count == 0) {
      auto rows = build_sync_open_rows_native(sync_rows);
      std::vector<std::string> changed_request_ids;
      changed_request_ids.reserve(rows.size());
      for (const auto& row : rows) {
        changed_request_ids.push_back(row.request_id);
      }
      auto frame = encode_control_batch_native_rows(
          dst_drafter_rank,
          sync_rows,
          commit_rows,
          close_rows);
      {
        py::gil_scoped_release release;
        {
          std::lock_guard<std::mutex> guard(queue_mu_);
          ensure_outbound_queue_capacity(
              send_queue_.size(),
              pending_send_bytes_,
              1,
              frame.size(),
              "Verifier control");
          draft_tail_buffer_->open_request_rows_native(std::move(rows));
          pending_send_bytes_ += frame.size();
          send_queue_.push_back(QueuedFrame{
              dst_rank, std::move(frame), std::move(changed_request_ids)});
        }
      }
      queue_cv_.notify_one();
      return;
    }
    auto batch = build_control_batch_native(
        dst_drafter_rank,
        sync_rows,
        commit_rows,
        close_rows);
    auto changed_request_ids = control_request_ids(batch);
    auto frame = encode_control_batch(batch);
    {
      py::gil_scoped_release release;
      {
        std::lock_guard<std::mutex> guard(queue_mu_);
        ensure_outbound_queue_capacity(
            send_queue_.size(),
            pending_send_bytes_,
            1,
            frame.size(),
            "Verifier control");
        draft_tail_buffer_->apply_control_batch_native(batch);
        pending_send_bytes_ += frame.size();
        send_queue_.push_back(QueuedFrame{
            batch.dst_drafter_rank,
            std::move(frame),
            std::move(changed_request_ids)});
      }
    }
    queue_cv_.notify_one();
  }

  void submit_control_frame(const py::bytes& frame_bytes) {
    check_thread_error();
    std::string bytes = frame_bytes.cast<std::string>();
    py::gil_scoped_release release;
    auto frame = decode_local_frame(bytes, kFrameKindVerifierControl);
    auto batches = verifier_controls_from_local_frame(frame);
    for (auto& batch : batches) {
      submit_control_batch(std::move(batch));
    }
  }

  void submit_verify_updates_native(
      const std::shared_ptr<VerifierSnapshotBatchCpp>& snapshot_batch,
      const py::sequence& commit_rows,
      const py::sequence& close_rows) {
    if (snapshot_batch == nullptr) {
      throw std::runtime_error(
          "Native verifier updates require a snapshot batch owner");
    }
    check_thread_error();
    auto batches =
        build_verify_update_batches_native(commit_rows, close_rows);
    {
      py::gil_scoped_release release;
      snapshot_batch->validate_verify_updates(batches);
      submit_control_batches(std::move(batches));
    }
  }

 private:
  void submit_control_batches(std::vector<DraftControlBatchCpp> batches) {
    for (const auto& batch : batches) {
      if (control_send_sockets_.count(batch.dst_drafter_rank) == 0) {
        throw std::runtime_error(
            "Missing control socket for dst_drafter_rank");
      }
    }
    for (auto& batch : batches) {
      submit_control_batch(std::move(batch));
    }
  }

  void submit_control_batch(DraftControlBatchCpp batch) {
    if (control_send_sockets_.count(batch.dst_drafter_rank) == 0) {
      throw std::runtime_error("Missing control socket for dst_drafter_rank");
    }
    auto changed_request_ids = control_request_ids(batch);
    auto frame = encode_control_batch(batch);
    {
      std::lock_guard<std::mutex> guard(queue_mu_);
      ensure_outbound_queue_capacity(
          send_queue_.size(),
          pending_send_bytes_,
          1,
          frame.size(),
          "Verifier control");
      draft_tail_buffer_->apply_control_batch_native(batch);
      pending_send_bytes_ += frame.size();
      send_queue_.push_back(QueuedFrame{
          batch.dst_drafter_rank,
          std::move(frame),
          std::move(changed_request_ids)});
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
          if (zmq_->api().recv_nonblock(result_recv_socket_, frame)) {
            recv_tail_stream_batch(frame);
            did_work = true;
          }
        }
      } catch (...) {
        if (!closed_.load()) throw;
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
    auto it = control_send_sockets_.find(queued.dst_rank);
    if (it == control_send_sockets_.end()) {
      throw std::runtime_error("Missing control socket for dst_drafter_rank");
    }
    if (gpu_tail_buffer_ != nullptr) {
      gpu_tail_buffer_->publish_requests(
          *draft_tail_buffer_, queued.changed_request_ids);
    }
    return send_frame_until_ready_or_closed(
        zmq_->api(), it->second, queued.frame, closed_);
  }

  void recv_tail_stream_batch(const std::string& frame) {
    auto batch = parse_tail_stream_batch(frame);
    std::vector<std::string> changed_request_ids;
    changed_request_ids.reserve(batch.outputs.size());
    for (const auto& output : batch.outputs) {
      if (output.dst_verifier_rank != verifier_rank_) {
        throw std::runtime_error("Draft proxy received a tail stream batch for the wrong verifier");
      }
      changed_request_ids.push_back(output.request_id);
    }
    draft_tail_buffer_->append_draft_stream_batch_native(batch);
    if (gpu_tail_buffer_ != nullptr) {
      gpu_tail_buffer_->publish_requests(
          *draft_tail_buffer_, changed_request_ids);
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

  int32_t verifier_rank_;
  std::unique_ptr<ZmqContextOwner> zmq_;
  std::shared_ptr<DecoupledSpecDraftTailBuffer> draft_tail_buffer_ref_;
  std::shared_ptr<DraftTailBufferCore> draft_tail_buffer_;
  std::shared_ptr<GpuDraftTailBufferCore> gpu_tail_buffer_;
  void* result_recv_socket_ = nullptr;
  std::string result_bind_endpoint_;
  std::map<int32_t, void*> control_send_sockets_;
  std::map<int32_t, std::string> control_peer_endpoints_;
  std::deque<QueuedFrame> send_queue_;
  size_t pending_send_bytes_ = 0;
  std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> started_{false};
  std::thread thread_;
  std::mutex error_mu_;
  std::string thread_error_;
};

class DecoupledSpecTokenSyncThread {
 public:
  DecoupledSpecTokenSyncThread(
      int64_t drafter_rank,
      const std::string& bind_endpoint,
      const py::sequence& verifier_peer_rows,
      uintptr_t external_context = 0)
      : drafter_rank_(checked_transport_rank(drafter_rank, "Drafter rank")),
        zmq_(std::make_unique<ZmqContextOwner>(external_context)) {
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
    control_recv_socket_ = zmq_->api().socket(zmq_->ctx(), kZmqPull);
    zmq_->api().configure_socket(control_recv_socket_, kZmqPull);
    zmq_->api().bind(control_recv_socket_, bind_endpoint);
    control_bind_endpoint_ = bind_endpoint;
    for (const auto& peer : verifier_peers) {
      void* socket = zmq_->api().socket(zmq_->ctx(), kZmqPush);
      zmq_->api().configure_socket(socket, kZmqPush);
      result_send_sockets_[peer.rank] = socket;
      result_peer_endpoints_[peer.rank] = peer.endpoint;
    }
  }

  ~DecoupledSpecTokenSyncThread() { close(); }

  std::string control_bind_endpoint() const { return control_bind_endpoint_; }

  void start() {
    check_thread_error();
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    try {
      for (const auto& kv : result_send_sockets_) {
        zmq_->api().connect(kv.second, result_peer_endpoints_.at(kv.first));
      }
    } catch (...) {
      started_.store(false);
      throw;
    }
    closed_.store(false);
    thread_ = std::thread([this] { run_guarded(); });
  }

  void close() {
    closed_.store(true);
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    for (auto& kv : result_send_sockets_) zmq_->api().close_socket(kv.second);
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
      submit_draft_results_batch(std::move(batch), false);
    }
  }

  void submit_draft_result_frame(const py::bytes& frame_bytes) {
    check_thread_error();
    std::string bytes = frame_bytes.cast<std::string>();
    py::gil_scoped_release release;
    auto frame = decode_local_frame(bytes, kFrameKindDraftResult);
    auto batch = draft_results_from_local_frame(frame);
    submit_draft_results_batch(std::move(batch), true);
  }

  py::bytes probe_pending_controls() {
    check_thread_error();
    std::string result;
    {
      py::gil_scoped_release release;
      auto probe = data_plane_.probe_pending_controls();
      result = encode_control_probe_local_frame(probe);
    }
    return py::bytes(result);
  }

  py::bytes consume_ready_controls(
      uint64_t probe_id,
      const py::bytes& eligibility_mask_bytes) {
    check_thread_error();
    std::string eligibility_mask =
        eligibility_mask_bytes.cast<std::string>();
    std::string result;
    {
      py::gil_scoped_release release;
      auto ready = data_plane_.consume_ready_controls(
          probe_id, eligibility_mask);
      result = encode_draft_actions_local_frame(
          ready, drafter_rank_);
    }
    return py::bytes(result);
  }

  int64_t pending_control_count() {
    check_thread_error();
    return data_plane_.pending_control_count();
  }

  bool wait_for_pending_control(int64_t timeout_us) {
    check_thread_error();
    return data_plane_.wait_for_pending_control(timeout_us);
  }

  int64_t pending_control_batch_count() {
    check_thread_error();
    std::lock_guard<std::mutex> guard(pending_control_mu_);
    return static_cast<int64_t>(pending_control_batches_.size());
  }

  py::list drain_control_batches_native(int64_t max_batches) {
    check_thread_error();
    if (max_batches < -1) {
      throw std::runtime_error(
          "max_batches must be non-negative or -1 for no limit");
    }
    std::vector<DraftControlBatchCpp> batches;
    {
      std::lock_guard<std::mutex> guard(pending_control_mu_);
      size_t limit = max_batches < 0
          ? pending_control_batches_.size()
          : std::min<size_t>(
                pending_control_batches_.size(),
                static_cast<size_t>(max_batches));
      batches.reserve(limit);
      for (size_t i = 0; i < limit; ++i) {
        batches.push_back(std::move(pending_control_batches_.front()));
        pending_control_batches_.pop_front();
      }
    }
    py::list rows;
    for (const auto& batch : batches) {
      rows.append(control_batch_native_rows(batch));
    }
    return rows;
  }

  py::list snapshot_pending_commit_segments_native() {
    check_thread_error();
    std::vector<VerifierCommitSegmentCpp> segments;
    {
      py::gil_scoped_release release;
      segments = data_plane_.snapshot_pending_commit_segments_native();
    }
    py::list out;
    for (const auto& segment : segments) {
      out.append(segment_row(segment));
    }
    return out;
  }

  py::tuple extract_ready_controls_native(const py::sequence& rows) {
    check_thread_error();
    auto decisions = build_extract_decisions_native(rows);
    ReadyDraftControlsCpp ready;
    {
      py::gil_scoped_release release;
      ready = data_plane_.extract_ready_controls_native(decisions);
    }
    return ready_controls_native_rows(ready);
  }

 private:
  void submit_draft_results_batch(
      DraftTailStreamOutputBatchCpp batch,
      bool strict_local_contract) {
    if (batch.outputs.empty()) return;
    std::map<int32_t, DraftTailStreamOutputBatchCpp> by_verifier;
    for (const auto& output : batch.outputs) {
      by_verifier[output.dst_verifier_rank].outputs.push_back(output);
    }
    std::vector<QueuedFrame> frames;
    frames.reserve(by_verifier.size());
    size_t frame_bytes = 0;
    for (const auto& kv : by_verifier) {
      frames.push_back(QueuedFrame{
          kv.first, encode_tail_stream_batch(kv.second)});
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
      data_plane_.append_draft_outputs(batch, strict_local_contract);
      for (auto& frame : frames) {
        pending_result_bytes_ += frame.frame.size();
        outgoing_results_.push_back(std::move(frame));
      }
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
    queue_cv_.notify_all();
  }

  void check_thread_error() {
    std::lock_guard<std::mutex> guard(error_mu_);
    if (!thread_error_.empty()) {
      throw std::runtime_error("CppTokenSyncThread failed: " + thread_error_);
    }
  }

  void run() {
    while (!closed_.load()) {
      bool did_work = false;
      did_work = drain_outgoing_results() || did_work;
      did_work = drain_control_socket() || did_work;
      if (!did_work) {
        std::unique_lock<std::mutex> lock(queue_mu_);
        queue_cv_.wait_for(lock, std::chrono::microseconds(500), [&] {
          return closed_.load() || !outgoing_results_.empty();
        });
      }
    }
  }

  bool drain_outgoing_results() {
    bool did_work = false;
    while (true) {
      QueuedFrame queued;
      {
        std::lock_guard<std::mutex> guard(queue_mu_);
        if (outgoing_results_.empty()) break;
        size_t frame_bytes = outgoing_results_.front().frame.size();
        queued = std::move(outgoing_results_.front());
        outgoing_results_.pop_front();
        pending_result_bytes_ -= frame_bytes;
      }
      if (!send_draft_results(queued)) return did_work;
      did_work = true;
    }
    return did_work;
  }

  bool drain_control_socket() {
    bool did_work = false;
    while (!closed_.load()) {
      std::string frame;
      bool received = zmq_->api().recv_nonblock(control_recv_socket_, frame);
      if (!received) break;
      auto batch = parse_control_batch(frame);
      if (batch.dst_drafter_rank != drafter_rank_) {
        throw std::runtime_error(
            "Draft control batch targets a different drafter");
      }
      {
        std::lock_guard<std::mutex> guard(pending_control_mu_);
        pending_control_batches_.push_back(batch);
      }
      // The segmented inbox is the production owner. The raw queue above is
      // retained only for compatibility diagnostics and is drained by the
      // Python facade without participating in scheduler decisions.
      data_plane_.add_control_batch(batch);
      did_work = true;
    }
    return did_work;
  }

  bool send_draft_results(const QueuedFrame& queued) {
    auto it = result_send_sockets_.find(queued.dst_rank);
    if (it == result_send_sockets_.end()) {
      throw std::runtime_error("Missing result socket for dst_verifier_rank");
    }
    return send_frame_until_ready_or_closed(
        zmq_->api(), it->second, queued.frame, closed_);
  }

  int32_t drafter_rank_;
  std::unique_ptr<ZmqContextOwner> zmq_;
  void* control_recv_socket_ = nullptr;
  std::string control_bind_endpoint_;
  std::map<int32_t, void*> result_send_sockets_;
  std::map<int32_t, std::string> result_peer_endpoints_;
  DrafterDataPlaneCore data_plane_;
  std::deque<DraftControlBatchCpp> pending_control_batches_;
  std::mutex pending_control_mu_;
  std::deque<QueuedFrame> outgoing_results_;
  size_t pending_result_bytes_ = 0;
  std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> started_{false};
  std::thread thread_;
  std::mutex error_mu_;
  std::string thread_error_;
};

// Role-level owners used by the flat bytes-only API. They keep the verifier
// tail state and its proxy thread, or the drafter inbox/transcript and token
// thread, under one Python-visible lifetime. Legacy component classes remain
// exported during scheduler migration.
class DecoupledSpecVerifierDataPlane {
 public:
  DecoupledSpecVerifierDataPlane(
      int64_t verifier_rank,
      int64_t required_tail_len,
      const std::string& bind_endpoint,
      const py::sequence& drafter_peer_rows) {
    draft_tail_buffer_ = std::make_shared<DecoupledSpecDraftTailBuffer>(
        verifier_rank, required_tail_len);
    proxy_ = std::make_unique<DecoupledSpecDraftProxyThread>(
        verifier_rank,
        bind_endpoint,
        drafter_peer_rows,
        draft_tail_buffer_);
  }

  ~DecoupledSpecVerifierDataPlane() { close(); }

  std::string result_bind_endpoint() const {
    return proxy_->result_bind_endpoint();
  }

  void start() { proxy_->start(); }

  void close() {
    if (proxy_) proxy_->close();
    if (draft_tail_buffer_) draft_tail_buffer_->close();
  }

  void submit_control_frame(const py::bytes& frame) {
    proxy_->submit_control_frame(frame);
  }

  void submit_verify_updates_native(
      const std::shared_ptr<VerifierSnapshotBatchCpp>& snapshot_batch,
      const py::sequence& commit_rows,
      const py::sequence& close_rows) {
    proxy_->submit_verify_updates_native(
        snapshot_batch, commit_rows, close_rows);
  }

  py::bytes query_request_states(const py::bytes& request_frame) {
    return draft_tail_buffer_->query_request_states(request_frame);
  }

  std::vector<int64_t> query_committed_lens_native(
      const py::sequence& request_ids) {
    return draft_tail_buffer_->query_committed_lens_native(request_ids);
  }

  std::shared_ptr<VerifierSnapshotBatchCpp> get_draft_snapshot_batch(
      const py::sequence& request_ids,
      bool allow_partial,
      int64_t max_tail_len) {
    return draft_tail_buffer_->get_draft_snapshot_batch(
        request_ids, allow_partial, max_tail_len);
  }

  py::bytes get_draft_snapshots(
      const py::bytes& request_frame,
      bool allow_partial,
      int64_t max_tail_len) {
    return draft_tail_buffer_->get_draft_snapshots_frame(
        request_frame, allow_partial, max_tail_len);
  }

 private:
  std::shared_ptr<DecoupledSpecDraftTailBuffer> draft_tail_buffer_;
  std::unique_ptr<DecoupledSpecDraftProxyThread> proxy_;
};

class DecoupledSpecDrafterDataPlane {
 public:
  DecoupledSpecDrafterDataPlane(
      int64_t drafter_rank,
      const std::string& bind_endpoint,
      const py::sequence& verifier_peer_rows) {
    token_sync_ = std::make_unique<DecoupledSpecTokenSyncThread>(
        drafter_rank, bind_endpoint, verifier_peer_rows);
  }

  ~DecoupledSpecDrafterDataPlane() { close(); }

  std::string control_bind_endpoint() const {
    return token_sync_->control_bind_endpoint();
  }

  void start() { token_sync_->start(); }
  void close() {
    if (token_sync_) token_sync_->close();
  }

  void submit_draft_result_frame(const py::bytes& frame) {
    token_sync_->submit_draft_result_frame(frame);
  }

  py::bytes probe_pending_controls() {
    return token_sync_->probe_pending_controls();
  }

  py::bytes consume_ready_controls(
      uint64_t probe_id,
      const py::bytes& eligibility_mask) {
    return token_sync_->consume_ready_controls(
        probe_id, eligibility_mask);
  }

 private:
  std::unique_ptr<DecoupledSpecTokenSyncThread> token_sync_;
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
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t,
              uintptr_t>(),
          py::arg("device_index"),
          py::arg("num_seats"),
          py::arg("num_draft_tokens"),
          py::arg("landing_stream"),
          py::arg("versions"),
          py::arg("publish_seqs"),
          py::arg("request_epochs"),
          py::arg("active_request_epochs"),
          py::arg("prompt_lens"),
          py::arg("committed_lens"),
          py::arg("raw_tail_lens"),
          py::arg("consumable_tail_lens"),
          py::arg("tail_tokens"))
      .def(
          "bind_request",
          &GpuDraftTailBufferCore::bind_request,
          py::arg("request_id"),
          py::arg("gpu_seat"),
          py::arg("request_epoch"))
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
          "staging_slot_count",
          &GpuDraftTailBufferCore::staging_slot_count)
      .def_property_readonly(
          "max_staging_slots",
          &GpuDraftTailBufferCore::max_staging_slots);

  py::class_<VerifierSnapshotBatchCpp, std::shared_ptr<VerifierSnapshotBatchCpp>>(
      m, "VerifierSnapshotBatch")
      .def_static(
          "from_transport_tensor",
          &VerifierSnapshotBatchCpp::from_transport_tensor,
          py::arg("payload"),
          py::arg("request_ids"))
      .def_static(
          "transport_width",
          &VerifierSnapshotBatchCpp::transport_width,
          py::arg("max_tail_len"))
      .def(
          "to_transport_tensor",
          &VerifierSnapshotBatchCpp::to_transport_tensor,
          py::arg("max_tail_len"))
      .def_property_readonly("wait_ns", &VerifierSnapshotBatchCpp::wait_ns)
      .def(
          "align",
          &VerifierSnapshotBatchCpp::align,
          py::arg("request_ids"),
          py::arg("pre_committed_lens"),
          py::arg("row_indices"),
          py::arg("batch_size"))
      .def(
          "materialize",
          &VerifierSnapshotBatchCpp::materialize,
          py::arg("num_draft_tokens"),
          py::arg("pad_token"))
      .def(
          "draft_lens",
          &VerifierSnapshotBatchCpp::draft_lens,
          py::arg("num_draft_tokens"))
      .def(
          "num_consumable_drafts",
          &VerifierSnapshotBatchCpp::num_consumable_drafts);

  py::class_<DecoupledSpecDraftTailBuffer, std::shared_ptr<DecoupledSpecDraftTailBuffer>>(m, "DraftTailBuffer")
      .def(py::init<int64_t, int64_t>(), py::arg("verifier_rank"), py::arg("required_tail_len"))
      .def("close", &DecoupledSpecDraftTailBuffer::close, py::call_guard<py::gil_scoped_release>())
      .def("has_request", &DecoupledSpecDraftTailBuffer::has_request, py::arg("request_id"))
      .def("get_committed_len", &DecoupledSpecDraftTailBuffer::get_committed_len, py::arg("request_id"))
      .def(
          "attach_gpu_tail_buffer",
          &DecoupledSpecDraftTailBuffer::attach_gpu_tail_buffer,
          py::arg("gpu_tail_buffer"))
      .def(
          "apply_control_batch_native",
          &DecoupledSpecDraftTailBuffer::apply_control_batch_native,
          py::arg("dst_drafter_rank"),
          py::arg("sync_rows"),
          py::arg("commit_rows"),
          py::arg("close_rows"))
      .def(
          "append_draft_stream_batch_native",
          &DecoupledSpecDraftTailBuffer::append_draft_stream_batch_native,
          py::arg("rows"))
      .def(
          "wait_for_draft_tokens_native",
          &DecoupledSpecDraftTailBuffer::wait_for_draft_tokens_native,
          py::arg("rids"),
          py::arg("min_draft_tokens"))
      .def(
          "get_draft_snapshots_native",
          &DecoupledSpecDraftTailBuffer::get_draft_snapshots_native,
          py::arg("rids"),
          py::arg("allow_partial"),
          py::arg("max_tail_len"))
      .def(
          "query_request_states",
          &DecoupledSpecDraftTailBuffer::query_request_states,
          py::arg("request_frame"))
      .def(
          "get_draft_snapshots_frame",
          &DecoupledSpecDraftTailBuffer::get_draft_snapshots_frame,
          py::arg("request_frame"),
          py::arg("allow_partial"),
          py::arg("max_tail_len"));

  py::class_<DecoupledSpecDraftProxyThread>(m, "DraftProxyThread")
      .def(
          py::init<
              int64_t,
              const std::string&,
              const py::sequence&,
              std::shared_ptr<DecoupledSpecDraftTailBuffer>,
              uintptr_t>(),
          py::arg("verifier_rank"),
          py::arg("bind_endpoint"),
          py::arg("drafter_peers"),
          py::arg("draft_tail_buffer"),
          py::arg("external_context") = 0)
      .def("result_bind_endpoint", &DecoupledSpecDraftProxyThread::result_bind_endpoint)
      .def(
          "bind_gpu_request_native",
          &DecoupledSpecDraftProxyThread::bind_gpu_request_native,
          py::arg("request_id"),
          py::arg("gpu_seat"),
          py::arg("request_epoch"))
      .def("start", &DecoupledSpecDraftProxyThread::start, py::call_guard<py::gil_scoped_release>())
      .def("close", &DecoupledSpecDraftProxyThread::close, py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_control_batch_native",
          &DecoupledSpecDraftProxyThread::submit_control_batch_native,
          py::arg("dst_drafter_rank"),
          py::arg("sync_rows"),
          py::arg("commit_rows"),
          py::arg("close_rows"))
      .def(
          "submit_control_frame",
          &DecoupledSpecDraftProxyThread::submit_control_frame,
          py::arg("frame"));

  py::class_<DecoupledSpecTokenSyncThread>(m, "TokenSyncThread")
      .def(
          py::init<int64_t, const std::string&, const py::sequence&, uintptr_t>(),
          py::arg("drafter_rank"),
          py::arg("bind_endpoint"),
          py::arg("verifier_peers"),
          py::arg("external_context") = 0)
      .def("control_bind_endpoint", &DecoupledSpecTokenSyncThread::control_bind_endpoint)
      .def("start", &DecoupledSpecTokenSyncThread::start, py::call_guard<py::gil_scoped_release>())
      .def("close", &DecoupledSpecTokenSyncThread::close, py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_draft_results_native",
          &DecoupledSpecTokenSyncThread::submit_draft_results_native,
          py::arg("rows"))
      .def(
          "submit_draft_result_frame",
          &DecoupledSpecTokenSyncThread::submit_draft_result_frame,
          py::arg("frame"))
      .def(
          "probe_pending_controls",
          &DecoupledSpecTokenSyncThread::probe_pending_controls)
      .def(
          "consume_ready_controls",
          &DecoupledSpecTokenSyncThread::consume_ready_controls,
          py::arg("probe_id"),
          py::arg("eligibility_mask"))
      .def("pending_control_count", &DecoupledSpecTokenSyncThread::pending_control_count)
      .def(
          "wait_for_pending_control",
          &DecoupledSpecTokenSyncThread::wait_for_pending_control,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "pending_control_batch_count",
          &DecoupledSpecTokenSyncThread::pending_control_batch_count)
      .def(
          "drain_control_batches_native",
          &DecoupledSpecTokenSyncThread::drain_control_batches_native,
          py::arg("max_batches"))
      .def(
          "snapshot_pending_commit_segments_native",
          &DecoupledSpecTokenSyncThread::snapshot_pending_commit_segments_native)
      .def(
          "extract_ready_controls_native",
          &DecoupledSpecTokenSyncThread::extract_ready_controls_native,
          py::arg("rows"));

  py::class_<DecoupledSpecVerifierDataPlane>(m, "VerifierDataPlane")
      .def(
          py::init<
              int64_t,
              int64_t,
              const std::string&,
              const py::sequence&>(),
          py::arg("verifier_rank"),
          py::arg("required_tail_len"),
          py::arg("bind_endpoint"),
          py::arg("drafter_peers"))
      .def(
          "result_bind_endpoint",
          &DecoupledSpecVerifierDataPlane::result_bind_endpoint)
      .def(
          "start",
          &DecoupledSpecVerifierDataPlane::start,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "close",
          &DecoupledSpecVerifierDataPlane::close,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_control_frame",
          &DecoupledSpecVerifierDataPlane::submit_control_frame,
          py::arg("frame"))
      .def(
          "submit_verify_updates_native",
          &DecoupledSpecVerifierDataPlane::submit_verify_updates_native,
          py::arg("snapshot_batch"),
          py::arg("commit_rows"),
          py::arg("close_rows"))
      .def(
          "query_request_states",
          &DecoupledSpecVerifierDataPlane::query_request_states,
          py::arg("request_frame"))
      .def(
          "query_committed_lens_native",
          &DecoupledSpecVerifierDataPlane::query_committed_lens_native,
          py::arg("request_ids"))
      .def(
          "get_draft_snapshot_batch",
          &DecoupledSpecVerifierDataPlane::get_draft_snapshot_batch,
          py::arg("request_ids"),
          py::arg("allow_partial"),
          py::arg("max_tail_len"))
      .def(
          "get_draft_snapshots",
          &DecoupledSpecVerifierDataPlane::get_draft_snapshots,
          py::arg("request_frame"),
          py::arg("allow_partial"),
          py::arg("max_tail_len"));

  py::class_<DecoupledSpecDrafterDataPlane>(m, "DrafterDataPlane")
      .def(
          py::init<int64_t, const std::string&, const py::sequence&>(),
          py::arg("drafter_rank"),
          py::arg("bind_endpoint"),
          py::arg("verifier_peers"))
      .def(
          "control_bind_endpoint",
          &DecoupledSpecDrafterDataPlane::control_bind_endpoint)
      .def(
          "start",
          &DecoupledSpecDrafterDataPlane::start,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "close",
          &DecoupledSpecDrafterDataPlane::close,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_draft_result_frame",
          &DecoupledSpecDrafterDataPlane::submit_draft_result_frame,
          py::arg("frame"))
      .def(
          "probe_pending_controls",
          &DecoupledSpecDrafterDataPlane::probe_pending_controls)
      .def(
          "consume_ready_controls",
          &DecoupledSpecDrafterDataPlane::consume_ready_controls,
          py::arg("probe_id"),
          py::arg("eligibility_mask"));
}

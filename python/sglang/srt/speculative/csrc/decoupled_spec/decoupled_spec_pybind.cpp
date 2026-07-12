#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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
constexpr int kZmqLINGER = 17;
constexpr int kZmqSNDHWM = 23;
constexpr int kZmqRCVHWM = 24;
constexpr int kZmqSNDBUF = 11;
constexpr int kZmqRCVBUF = 12;
constexpr int kZmqIPV6 = 42;
constexpr int kZmqLAST_ENDPOINT = 32;
constexpr int kErrAgain = 11;
constexpr size_t kMaxZmqMessageBytes = 64ULL * 1024ULL * 1024ULL;

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
  std::vector<int32_t> committed_output_ids;

  DraftReqKeyCpp draft_key() const { return {src_verifier_rank, request_id}; }
};

struct DraftSyncOpenCpp {
  std::string request_id;
  int32_t dst_drafter_rank = 0;
  int64_t committed_len = 0;
};

struct VerifyCommitCpp {
  std::string request_id;
  int32_t src_verifier_rank = 0;
  int32_t dst_drafter_rank = 0;
  int64_t pre_verify_committed_len = 0;
  std::vector<int32_t> committed_token_ids;

  DraftReqKeyCpp draft_key() const { return {src_verifier_rank, request_id}; }

  void validate() const {
    if (committed_token_ids.empty()) {
      throw std::runtime_error("VerifyCommit committed_token_ids must be non-empty");
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
  int32_t new_token_id = 0;
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
    msg.committed_output_ids = reader.read_int_list();
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
    msg.committed_token_ids = reader.read_int_list();
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
    output.new_token_id = reader.read_i32();
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
    writer.write_i32(output.new_token_id);
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
    writer.write_u32(static_cast<uint32_t>(msg.committed_output_ids.size()));
    for (int32_t token : msg.committed_output_ids) writer.write_i32(token);
  }
  writer.write_u32(static_cast<uint32_t>(batch.verify_commit_messages.size()));
  for (const auto& msg : batch.verify_commit_messages) {
    writer.write_string(msg.request_id);
    writer.write_i32(msg.src_verifier_rank);
    writer.write_i32(msg.dst_drafter_rank);
    writer.write_i64(msg.pre_verify_committed_len);
    writer.write_u32(static_cast<uint32_t>(msg.committed_token_ids.size()));
    for (int32_t token : msg.committed_token_ids) writer.write_i32(token);
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
    msg.committed_output_ids = py_int32_vector(row[4]);
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
    msg.committed_token_ids = py_int32_vector(row[4]);
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
    if (py::len(row) != 6) {
      throw std::runtime_error("DraftTailStreamOutput native row must have 6 fields");
    }
    DraftTailStreamOutputCpp output;
    output.src_drafter_rank = static_cast<int32_t>(row[0].cast<int64_t>());
    output.dst_verifier_rank = static_cast<int32_t>(row[1].cast<int64_t>());
    output.request_id = row[2].cast<std::string>();
    output.base_committed_len = row[3].cast<int64_t>();
    output.new_token_pos = row[4].cast<int64_t>();
    output.new_token_id = static_cast<int32_t>(row[5].cast<int64_t>());
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
  std::vector<int32_t> committed_token_ids;

  int64_t end_committed_len() const {
    return pre_verify_committed_len + static_cast<int64_t>(committed_token_ids.size());
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
    committed_token_ids.insert(
        committed_token_ids.end(), message.committed_token_ids.begin(), message.committed_token_ids.end());
  }

  VerifierCommitSegmentCpp extract_prefix(int64_t num_tokens) {
    if (num_tokens <= 0 || num_tokens > static_cast<int64_t>(committed_token_ids.size())) {
      throw std::runtime_error("Invalid verifier commit segment prefix length");
    }
    VerifierCommitSegmentCpp prefix;
    prefix.draft_key = draft_key;
    prefix.dst_drafter_rank = dst_drafter_rank;
    prefix.pre_verify_committed_len = pre_verify_committed_len;
    prefix.committed_token_ids.assign(committed_token_ids.begin(), committed_token_ids.begin() + num_tokens);
    committed_token_ids.erase(committed_token_ids.begin(), committed_token_ids.begin() + num_tokens);
    pre_verify_committed_len += num_tokens;
    return prefix;
  }
};

struct ReadyDraftControlsCpp {
  std::vector<DraftSyncCpp> sync_messages;
  std::vector<DraftReqKeyCpp> close_keys;
  std::vector<VerifierCommitSegmentCpp> ready_commit_segments;
};

struct RequestDraftTailStateCpp {
  int32_t drafter_rank = 0;
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

struct DraftTailSnapshotCpp {
  std::string request_id;
  int64_t committed_len = 0;
  std::vector<int32_t> tail_tokens;
  int64_t raw_tail_len = 0;
};

struct DraftTailSnapshotBatchCpp {
  std::vector<DraftTailSnapshotCpp> snapshots;
  int64_t wait_ns = 0;
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
      msg.committed_output_ids);
}

py::tuple segment_row(const VerifierCommitSegmentCpp& segment) {
  return py::make_tuple(
      segment.draft_key.request_id,
      segment.draft_key.src_verifier_rank,
      segment.dst_drafter_rank,
      segment.pre_verify_committed_len,
      segment.committed_token_ids);
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
      : verifier_rank_(static_cast<int32_t>(verifier_rank)),
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

  void apply_control_batch_native(const DraftControlBatchCpp& batch) {
    {
      std::lock_guard<std::mutex> guard(mu_);
      for (const auto& msg : batch.sync_messages) open_request_locked(msg);
      for (const auto& msg : batch.verify_commit_messages) apply_commit_locked(msg);
      for (const auto& msg : batch.close_messages) close_request_locked(msg);
      cv_.notify_all();
    }
  }

  void open_request_rows_native(std::vector<DraftSyncOpenCpp> rows) {
    std::lock_guard<std::mutex> guard(mu_);
    states_.reserve(states_.size() + rows.size());
    for (auto& row : rows) {
      RequestDraftTailStateCpp state;
      state.drafter_rank = row.dst_drafter_rank;
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
      for (const auto& output : batch.outputs) push_one_locked(output);
      cv_.notify_all();
    }
  }

  void wait_for_draft_tokens(const std::vector<std::string>& rids, int64_t min_draft_tokens) {
    min_draft_tokens = std::max<int64_t>(0, min_draft_tokens);
    if (min_draft_tokens <= 0) return;
    std::unique_lock<std::mutex> lock(mu_);
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
    RequestDraftTailStateCpp state;
    state.drafter_rank = message.dst_drafter_rank;
    state.committed_len = static_cast<int64_t>(message.committed_output_ids.size());
    state.can_accept_prefix_len = state.committed_len;
    states_[message.request_id] = std::move(state);
  }

  void apply_commit_locked(const VerifyCommitCpp& message) {
    message.validate();

    auto it = states_.find(message.request_id);
    if (it == states_.end()) return;
    auto& state = it->second;
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
      for (int32_t token : message.committed_token_ids) state.pending_expected_tokens.push_back(token);
      return;
    }

    int64_t matched_tail_len = 0;
    int64_t max_possible_match_len =
        std::min<int64_t>(static_cast<int64_t>(message.committed_token_ids.size()), raw_tail_len_before);
    while (matched_tail_len < max_possible_match_len &&
           state.tail_tokens[matched_tail_len] == message.committed_token_ids[matched_tail_len]) {
      ++matched_tail_len;
    }
    if (matched_tail_len) {
      state.tail_tokens.erase(state.tail_tokens.begin(), state.tail_tokens.begin() + matched_tail_len);
      state.committed_len += matched_tail_len;
    }

    if (matched_tail_len < static_cast<int64_t>(message.committed_token_ids.size())) {
      if (matched_tail_len < raw_tail_len_before) {
        state.can_accept_prefix_len = state.committed_len;
      }
      state.tail_tokens.clear();
      for (size_t i = static_cast<size_t>(matched_tail_len); i < message.committed_token_ids.size(); ++i) {
        state.pending_expected_tokens.push_back(message.committed_token_ids[i]);
      }
    }
  }

  void close_request_locked(const DraftCloseCpp& message) { states_.erase(message.request_id); }

  void push_one_locked(const DraftTailStreamOutputCpp& output) {
    const auto& request_id = output.request_id;
    int64_t base_committed_len = output.base_committed_len;
    int64_t token_pos = output.new_token_pos;
    int32_t token_id = output.new_token_id;
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

  int32_t verifier_rank_;
  int64_t required_tail_len_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool closed_ = false;
  std::unordered_map<std::string, RequestDraftTailStateCpp> states_;
};

class DraftControlInboxCore {
 public:
  bool is_empty() {
    std::lock_guard<std::mutex> guard(mu_);
    return sync_messages_.empty() && verifier_commit_segments_.empty() && close_keys_.empty();
  }

  int64_t pending_control_count() {
    std::lock_guard<std::mutex> guard(mu_);
    return static_cast<int64_t>(sync_messages_.size() + verifier_commit_segments_.size() + close_keys_.size());
  }

  void add_control_batch(const DraftControlBatchCpp& batch) {
    std::lock_guard<std::mutex> guard(mu_);
    close_keys_.reserve(close_keys_.size() + batch.close_messages.size());
    sync_messages_.reserve(sync_messages_.size() + batch.sync_messages.size());
    verifier_commit_segments_.reserve(
        verifier_commit_segments_.size() + batch.verify_commit_messages.size());
    for (const auto& msg : batch.close_messages) add_close_key_locked(msg.draft_key());
    for (const auto& msg : batch.sync_messages) {
      if (close_keys_.count(msg.draft_key().map_key()) == 0) sync_messages_.push_back(msg);
    }
    for (const auto& msg : batch.verify_commit_messages) add_verify_commit_locked(msg);
  }

  std::vector<VerifierCommitSegmentCpp> snapshot_pending_commit_segments_native() {
    std::lock_guard<std::mutex> guard(mu_);
    std::vector<VerifierCommitSegmentCpp> out;
    out.reserve(verifier_commit_segments_.size());
    for (const auto& kv : verifier_commit_segments_) {
      out.push_back(kv.second);
    }
    return out;
  }

  ReadyDraftControlsCpp extract_ready_controls_native(const std::vector<ExtractDecisionCpp>& decisions) {
    ReadyDraftControlsCpp ready;
    std::lock_guard<std::mutex> guard(mu_);
    for (const auto& kv : close_keys_) ready.close_keys.push_back(kv.second);
    close_keys_.clear();
    ready.sync_messages = std::move(sync_messages_);
    sync_messages_.clear();

    for (const auto& decision : decisions) {
      if (decision.consumable_len <= 0) continue;
      auto it = verifier_commit_segments_.find(decision.key.map_key());
      if (it == verifier_commit_segments_.end()) continue;
      auto& segment = it->second;
      if (segment.pre_verify_committed_len != decision.pre_verify_committed_len ||
          segment.dst_drafter_rank != decision.dst_drafter_rank) {
        continue;
      }
      ready.ready_commit_segments.push_back(segment.extract_prefix(decision.consumable_len));
      if (segment.committed_token_ids.empty()) verifier_commit_segments_.erase(it);
    }
    return ready;
  }

 private:
  void add_close_key_locked(const DraftReqKeyCpp& key) {
    close_keys_[key.map_key()] = key;
    verifier_commit_segments_.erase(key.map_key());
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
      return;
    }
    it->second.append_message(message);
  }

  std::mutex mu_;
  std::vector<DraftSyncCpp> sync_messages_;
  std::unordered_map<std::string, VerifierCommitSegmentCpp> verifier_commit_segments_;
  std::unordered_map<std::string, DraftReqKeyCpp> close_keys_;
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
  using getsockopt_t = int (*)(void*, int, void*, size_t*);
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
    int hwm = 0;
    int buf_size = static_cast<int>(512 * 1024 * 1024);
    if (type == kZmqPush) {
      try_set_int(socket, kZmqSNDHWM, hwm);
      try_set_int(socket, kZmqSNDBUF, buf_size);
    } else if (type == kZmqPull) {
      try_set_int(socket, kZmqRCVHWM, hwm);
      try_set_int(socket, kZmqRCVBUF, buf_size);
    }
  }

  void bind(void* socket, const std::string& endpoint) {
    if (bind_(socket, endpoint.c_str()) != 0) throw_last("zmq_bind");
  }

  void connect(void* socket, const std::string& endpoint) {
    if (endpoint.find('[') != std::string::npos) {
      int one = 1;
      try_set_int(socket, kZmqIPV6, one);
    }
    if (connect_(socket, endpoint.c_str()) != 0) throw_last("zmq_connect");
  }

  std::string bind_random_tcp(void* socket, const std::string& host) {
    std::string endpoint = host.find(':') != std::string::npos
                               ? "tcp://[" + host + "]:*"
                               : "tcp://" + host + ":*";
    if (host.find(':') != std::string::npos) {
      int one = 1;
      try_set_int(socket, kZmqIPV6, one);
    }
    bind(socket, endpoint);
    char last_endpoint[512] = {0};
    size_t size = sizeof(last_endpoint);
    if (getsockopt_(socket, kZmqLAST_ENDPOINT, last_endpoint, &size) != 0) {
      throw_last("zmq_getsockopt(LAST_ENDPOINT)");
    }
    return std::string(last_endpoint);
  }

  void send(void* socket, const std::string& frame) {
    int rc = send_(socket, frame.data(), frame.size(), 0);
    if (rc < 0) throw_last("zmq_send");
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

 private:
  ZmqApi() {
    const char* env_path = std::getenv("SGLANG_DECOUPLED_SPEC_ZMQ_LIB");
    if (env_path && env_path[0]) handle_ = dlopen(env_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle_) handle_ = dlopen("libzmq.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle_) handle_ = dlopen("/opt/tiger/ss_lib/so/libzmq.so", RTLD_NOW | RTLD_LOCAL);
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
    load(getsockopt_, "zmq_getsockopt");
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
  getsockopt_t getsockopt_ = nullptr;
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

class ZmqContextOwner {
 public:
  ZmqContextOwner() : api_(ZmqApi::instance()), ctx_(api_.ctx_new()) {}
  ~ZmqContextOwner() { close_context(); }
  ZmqContextOwner(const ZmqContextOwner&) = delete;
  ZmqContextOwner& operator=(const ZmqContextOwner&) = delete;

  ZmqApi& api() { return api_; }
  void* ctx() { return ctx_; }

  void close_context() {
    if (ctx_) {
      api_.ctx_term(ctx_);
      ctx_ = nullptr;
    }
  }

 private:
  ZmqApi& api_;
  void* ctx_;
};

struct QueuedFrame {
  std::string frame;
};

}  // namespace

class DecoupledSpecDraftTailBuffer {
 public:
  DecoupledSpecDraftTailBuffer(int64_t verifier_rank, int64_t required_tail_len)
      : core_(std::make_shared<DraftTailBufferCore>(verifier_rank, required_tail_len)) {}

  void close() { core_->close(); }
  bool has_request(const std::string& request_id) { return core_->has_request(request_id); }
  int64_t get_committed_len(const std::string& request_id) { return core_->get_committed_len(request_id); }

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
          snapshot.raw_tail_len));
    }
    return py::make_tuple(out, snapshot_batch.wait_ns);
  }

  std::shared_ptr<DraftTailBufferCore> core() { return core_; }

 private:
  std::shared_ptr<DraftTailBufferCore> core_;
};

class DecoupledSpecDraftProxyThread {
 public:
  DecoupledSpecDraftProxyThread(
      int64_t verifier_rank,
      const std::string& bind_endpoint_or_host,
      std::shared_ptr<DecoupledSpecDraftTailBuffer> draft_tail_buffer_ref)
      : verifier_rank_(static_cast<int32_t>(verifier_rank)), zmq_(std::make_unique<ZmqContextOwner>()) {
    if (draft_tail_buffer_ref == nullptr) {
      throw std::runtime_error("CppDraftProxyThread requires a CppDraftTailBuffer");
    }
    draft_tail_buffer_ref_ = std::move(draft_tail_buffer_ref);
    draft_tail_buffer_ = draft_tail_buffer_ref_->core();
    result_recv_socket_ = zmq_->api().socket(zmq_->ctx(), kZmqPull);
    zmq_->api().configure_socket(result_recv_socket_, kZmqPull);
    if (bind_endpoint_or_host.rfind("tcp://", 0) == 0) {
      zmq_->api().bind(result_recv_socket_, bind_endpoint_or_host);
      result_bind_endpoint_ = bind_endpoint_or_host;
    } else {
      result_bind_endpoint_ = zmq_->api().bind_random_tcp(result_recv_socket_, bind_endpoint_or_host);
    }
  }

  ~DecoupledSpecDraftProxyThread() { close(); }

  std::string result_bind_endpoint() const { return result_bind_endpoint_; }

  void configure_peer_endpoints_native(const py::sequence& endpoint_rows) {
    check_thread_error();
    auto endpoints = py_string_vector(endpoint_rows);
    py::gil_scoped_release release;
    configure_peer_endpoints_vector(std::move(endpoints));
  }

  void configure_peer_endpoints_vector(std::vector<std::string> endpoints) {
    if (endpoints.empty()) {
      throw std::runtime_error("Decoupled verify requires at least one drafter control endpoint");
    }
    if (!control_send_sockets_.empty()) {
      if (endpoints == drafter_control_endpoints_) return;
      throw std::runtime_error("Decoupled verify peer endpoints are already configured");
    }
    for (size_t rank = 0; rank < endpoints.size(); ++rank) {
      void* socket = zmq_->api().socket(zmq_->ctx(), kZmqPush);
      zmq_->api().configure_socket(socket, kZmqPush);
      zmq_->api().connect(socket, endpoints[rank]);
      control_send_sockets_[static_cast<int32_t>(rank)] = socket;
    }
    drafter_control_endpoints_ = std::move(endpoints);
  }

  void start() {
    check_thread_error();
    if (control_send_sockets_.empty()) return;
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
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
    check_thread_error();
    if (sync_count > 0 && commit_count == 0 && close_count == 0) {
      auto rows = build_sync_open_rows_native(sync_rows);
      auto frame = encode_control_batch_native_rows(
          dst_drafter_rank,
          sync_rows,
          commit_rows,
          close_rows);
      {
        py::gil_scoped_release release;
        draft_tail_buffer_->open_request_rows_native(std::move(rows));
        {
          std::lock_guard<std::mutex> guard(queue_mu_);
          send_queue_.push_back(QueuedFrame{std::move(frame)});
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
    {
      py::gil_scoped_release release;
      draft_tail_buffer_->apply_control_batch_native(batch);
      auto frame = encode_control_batch(batch);
      {
        std::lock_guard<std::mutex> guard(queue_mu_);
        send_queue_.push_back(QueuedFrame{std::move(frame)});
      }
    }
    queue_cv_.notify_one();
  }

 private:
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
          queued = std::move(send_queue_.front());
          send_queue_.pop_front();
        }
        send_control_batch(queued);
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

  void send_control_batch(const QueuedFrame& queued) {
    auto batch = parse_control_batch(queued.frame);
    auto it = control_send_sockets_.find(batch.dst_drafter_rank);
    if (it == control_send_sockets_.end()) {
      throw std::runtime_error("Missing control socket for dst_drafter_rank");
    }
    zmq_->api().send(it->second, queued.frame);
  }

  void recv_tail_stream_batch(const std::string& frame) {
    auto batch = parse_tail_stream_batch(frame);
    for (const auto& output : batch.outputs) {
      if (output.dst_verifier_rank != verifier_rank_) {
        throw std::runtime_error("Draft proxy received a tail stream batch for the wrong verifier");
      }
    }
    draft_tail_buffer_->append_draft_stream_batch_native(batch);
  }

  int32_t verifier_rank_;
  std::unique_ptr<ZmqContextOwner> zmq_;
  std::shared_ptr<DecoupledSpecDraftTailBuffer> draft_tail_buffer_ref_;
  std::shared_ptr<DraftTailBufferCore> draft_tail_buffer_;
  void* result_recv_socket_ = nullptr;
  std::string result_bind_endpoint_;
  std::map<int32_t, void*> control_send_sockets_;
  std::vector<std::string> drafter_control_endpoints_;
  std::deque<QueuedFrame> send_queue_;
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
  DecoupledSpecTokenSyncThread(int64_t drafter_rank, const std::string& bind_endpoint_or_host)
      : drafter_rank_(static_cast<int32_t>(drafter_rank)), zmq_(std::make_unique<ZmqContextOwner>()) {
    control_recv_socket_ = zmq_->api().socket(zmq_->ctx(), kZmqPull);
    zmq_->api().configure_socket(control_recv_socket_, kZmqPull);
    if (bind_endpoint_or_host.rfind("tcp://", 0) == 0) {
      zmq_->api().bind(control_recv_socket_, bind_endpoint_or_host);
      control_bind_endpoint_ = bind_endpoint_or_host;
    } else {
      control_bind_endpoint_ = zmq_->api().bind_random_tcp(control_recv_socket_, bind_endpoint_or_host);
    }
  }

  ~DecoupledSpecTokenSyncThread() { close(); }

  std::string control_bind_endpoint() const { return control_bind_endpoint_; }

  void configure_peer_endpoints_native(const py::sequence& endpoint_rows) {
    check_thread_error();
    auto endpoints = py_string_vector(endpoint_rows);
    py::gil_scoped_release release;
    configure_peer_endpoints_vector(std::move(endpoints));
  }

  void configure_peer_endpoints_vector(std::vector<std::string> endpoints) {
    if (endpoints.empty()) {
      throw std::runtime_error("Decoupled drafter requires at least one verifier result endpoint");
    }
    if (!result_send_sockets_.empty()) {
      if (endpoints == verifier_result_endpoints_) return;
      throw std::runtime_error("Decoupled drafter peer endpoints are already configured");
    }
    for (size_t rank = 0; rank < endpoints.size(); ++rank) {
      void* socket = zmq_->api().socket(zmq_->ctx(), kZmqPush);
      zmq_->api().configure_socket(socket, kZmqPush);
      zmq_->api().connect(socket, endpoints[rank]);
      result_send_sockets_[static_cast<int32_t>(rank)] = socket;
    }
    verifier_result_endpoints_ = std::move(endpoints);
  }

  void start() {
    check_thread_error();
    if (result_send_sockets_.empty()) return;
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
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
      auto frame = encode_tail_stream_batch(batch);
      {
        std::lock_guard<std::mutex> guard(queue_mu_);
        outgoing_results_.push_back(QueuedFrame{std::move(frame)});
      }
    }
    queue_cv_.notify_one();
  }

  int64_t pending_control_count() {
    check_thread_error();
    return inbox_.pending_control_count();
  }

  py::list snapshot_pending_commit_segments_native() {
    check_thread_error();
    std::vector<VerifierCommitSegmentCpp> segments;
    {
      py::gil_scoped_release release;
      segments = inbox_.snapshot_pending_commit_segments_native();
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
      ready = inbox_.extract_ready_controls_native(decisions);
    }
    return ready_controls_native_rows(ready);
  }

 private:
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
        queued = std::move(outgoing_results_.front());
        outgoing_results_.pop_front();
      }
      send_draft_results(queued);
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
      if (batch.dst_drafter_rank == drafter_rank_) inbox_.add_control_batch(batch);
      did_work = true;
    }
    return did_work;
  }

  void send_draft_results(const QueuedFrame& queued) {
    auto batch = parse_tail_stream_batch(queued.frame);
    std::map<int32_t, DraftTailStreamOutputBatchCpp> by_verifier;
    for (const auto& output : batch.outputs) {
      by_verifier[output.dst_verifier_rank].outputs.push_back(output);
    }
    for (const auto& kv : by_verifier) {
      auto it = result_send_sockets_.find(kv.first);
      if (it == result_send_sockets_.end()) {
        throw std::runtime_error("Missing result socket for dst_verifier_rank");
      }
      zmq_->api().send(it->second, encode_tail_stream_batch(kv.second));
    }
  }

  int32_t drafter_rank_;
  std::unique_ptr<ZmqContextOwner> zmq_;
  void* control_recv_socket_ = nullptr;
  std::string control_bind_endpoint_;
  std::map<int32_t, void*> result_send_sockets_;
  std::vector<std::string> verifier_result_endpoints_;
  DraftControlInboxCore inbox_;
  std::deque<QueuedFrame> outgoing_results_;
  std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::atomic<bool> closed_{false};
  std::atomic<bool> started_{false};
  std::thread thread_;
  std::mutex error_mu_;
  std::string thread_error_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;

  py::class_<DecoupledSpecDraftTailBuffer, std::shared_ptr<DecoupledSpecDraftTailBuffer>>(m, "DraftTailBuffer")
      .def(py::init<int64_t, int64_t>(), py::arg("verifier_rank"), py::arg("required_tail_len"))
      .def("close", &DecoupledSpecDraftTailBuffer::close, py::call_guard<py::gil_scoped_release>())
      .def("has_request", &DecoupledSpecDraftTailBuffer::has_request, py::arg("request_id"))
      .def("get_committed_len", &DecoupledSpecDraftTailBuffer::get_committed_len, py::arg("request_id"))
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
          py::arg("max_tail_len"));

  py::class_<DecoupledSpecDraftProxyThread>(m, "DraftProxyThread")
      .def(
          py::init<int64_t, const std::string&, std::shared_ptr<DecoupledSpecDraftTailBuffer>>(),
          py::arg("verifier_rank"),
          py::arg("bind_endpoint_or_host"),
          py::arg("draft_tail_buffer"))
      .def("result_bind_endpoint", &DecoupledSpecDraftProxyThread::result_bind_endpoint)
      .def(
          "configure_peer_endpoints_native",
          &DecoupledSpecDraftProxyThread::configure_peer_endpoints_native,
          py::arg("endpoints"))
      .def("start", &DecoupledSpecDraftProxyThread::start, py::call_guard<py::gil_scoped_release>())
      .def("close", &DecoupledSpecDraftProxyThread::close, py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_control_batch_native",
          &DecoupledSpecDraftProxyThread::submit_control_batch_native,
          py::arg("dst_drafter_rank"),
          py::arg("sync_rows"),
          py::arg("commit_rows"),
          py::arg("close_rows"));

  py::class_<DecoupledSpecTokenSyncThread>(m, "TokenSyncThread")
      .def(py::init<int64_t, const std::string&>(), py::arg("drafter_rank"), py::arg("bind_endpoint_or_host"))
      .def("control_bind_endpoint", &DecoupledSpecTokenSyncThread::control_bind_endpoint)
      .def(
          "configure_peer_endpoints_native",
          &DecoupledSpecTokenSyncThread::configure_peer_endpoints_native,
          py::arg("endpoints"))
      .def("start", &DecoupledSpecTokenSyncThread::start, py::call_guard<py::gil_scoped_release>())
      .def("close", &DecoupledSpecTokenSyncThread::close, py::call_guard<py::gil_scoped_release>())
      .def(
          "submit_draft_results_native",
          &DecoupledSpecTokenSyncThread::submit_draft_results_native,
          py::arg("rows"))
      .def("pending_control_count", &DecoupledSpecTokenSyncThread::pending_control_count)
      .def(
          "snapshot_pending_commit_segments_native",
          &DecoupledSpecTokenSyncThread::snapshot_pending_commit_segments_native)
      .def(
          "extract_ready_controls_native",
          &DecoupledSpecTokenSyncThread::extract_ready_controls_native,
          py::arg("rows"));
}

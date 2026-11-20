#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <executorch/runtime/core/error.h>

namespace etrb {

enum class ErrorKind : uint8_t {
  Ok = 0,
  Load,
  MethodNotFound,
  OperatorMissing,
  Delegate,
  Shape,
  DType,
  Execution,
  Memory,
  Closed,
  Argument,
  Type,
  Range,
};

struct Status {
  ErrorKind kind = ErrorKind::Ok;
  uint32_t et_code = 0;
  std::string message;
  std::string pal_log;

  bool ok() const noexcept { return kind == ErrorKind::Ok; }

  static Status Ok() noexcept { return Status{}; }

  static Status Make(
      ErrorKind k,
      std::string msg,
      uint32_t code = 0,
      std::string log = {}) noexcept {
    Status s;
    s.kind = k;
    s.et_code = code;
    s.message = std::move(msg);
    s.pal_log = std::move(log);
    return s;
  }
};

template <typename T>
struct StatusOr {
  Status status;
  T value{};

  bool ok() const noexcept { return status.ok(); }

  static StatusOr<T> Ok(T v) noexcept {
    StatusOr<T> out;
    out.status = Status::Ok();
    out.value = std::move(v);
    return out;
  }

  static StatusOr<T> Err(Status s) noexcept {
    StatusOr<T> out;
    out.status = std::move(s);
    return out;
  }
};

// Initialize ExecuTorch runtime and register thread-local PAL log capture once.
void init_runtime_once() noexcept;

// Scoped thread-local PAL log collector. Captures Error/Fatal PAL messages
// emitted on the current thread during an ExecuTorch call so they can be
// attached to Ruby exceptions.
class PalLogScope {
 public:
  PalLogScope() noexcept;
  ~PalLogScope() noexcept;

  PalLogScope(const PalLogScope&) = delete;
  PalLogScope& operator=(const PalLogScope&) = delete;

  std::string take() noexcept;

 private:
  std::string* prev_buf_ = nullptr;
  std::string buf_;
};

// Map an ExecuTorch runtime::Error + context + captured PAL log to Status.
Status status_from_et_error(
    executorch::runtime::Error err,
    const char* context,
    std::string pal_log = {},
    ErrorKind default_kind = ErrorKind::Execution) noexcept;

}  // namespace etrb

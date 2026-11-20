#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>

#include "status.hpp"

namespace etrb {

enum class ValueKind : uint8_t {
  None = 0,
  Int,
  Double,
  Bool,
  String,
  Tensor,
  IntList,
  DoubleList,
  BoolList,
  TensorList,
};

// Owned, Ruby-independent representation of an ExecuTorch EValue.
// Safe to pass across GVL boundaries without touching any Ruby VALUEs.
struct CoreValue {
  ValueKind kind = ValueKind::None;
  int64_t int_val = 0;
  double double_val = 0.0;
  bool bool_val = false;
  std::string string_val;
  executorch::extension::TensorPtr tensor_val;
  std::vector<int64_t> int_list;
  std::vector<double> double_list;
  std::vector<bool> bool_list;
  std::vector<executorch::extension::TensorPtr> tensor_list;

  static CoreValue None() noexcept { return CoreValue{}; }
  static CoreValue Int(int64_t v) noexcept {
    CoreValue c;
    c.kind = ValueKind::Int;
    c.int_val = v;
    return c;
  }
  static CoreValue Double(double v) noexcept {
    CoreValue c;
    c.kind = ValueKind::Double;
    c.double_val = v;
    return c;
  }
  static CoreValue Bool(bool v) noexcept {
    CoreValue c;
    c.kind = ValueKind::Bool;
    c.bool_val = v;
    return c;
  }
  static CoreValue String(std::string v) noexcept {
    CoreValue c;
    c.kind = ValueKind::String;
    c.string_val = std::move(v);
    return c;
  }
  static CoreValue Tensor(executorch::extension::TensorPtr ptr) noexcept {
    CoreValue c;
    c.kind = ValueKind::Tensor;
    c.tensor_val = std::move(ptr);
    return c;
  }
};

struct TensorMetaSnapshot {
  std::string name;
  executorch::aten::ScalarType dtype = executorch::aten::ScalarType::Undefined;
  std::vector<int32_t> sizes;
  std::vector<uint8_t> dim_order;
  size_t nbytes = 0;
  bool memory_planned = false;
};

struct ValueMetaSnapshot {
  executorch::runtime::Tag tag = executorch::runtime::Tag::None;
  TensorMetaSnapshot tensor;  // Valid when tag == Tag::Tensor
};

struct MethodMetaSnapshot {
  std::string name;
  std::vector<ValueMetaSnapshot> inputs;
  std::vector<ValueMetaSnapshot> outputs;
  std::vector<std::string> backends;
  std::vector<int64_t> planned_buffer_sizes;
  int64_t planned_memory_bytes = 0;
};

class ModuleCore {
 public:
  using LoadMode = executorch::extension::Module::LoadMode;

  ~ModuleCore() noexcept = default;

  ModuleCore(const ModuleCore&) = delete;
  ModuleCore& operator=(const ModuleCore&) = delete;

  static StatusOr<std::unique_ptr<ModuleCore>> open_file(
      const std::string& path,
      const std::vector<std::string>& data_files,
      LoadMode load_mode) noexcept;

  static StatusOr<std::unique_ptr<ModuleCore>> open_bytes(
      std::vector<uint8_t> bytes) noexcept;

  // Deterministic resource release. Waits for in-flight calls via unique lock.
  void close() noexcept;
  bool is_closed() const noexcept;

  std::string source_label() const noexcept;
  size_t estimated_memory_bytes() const noexcept;

  StatusOr<std::vector<std::string>> method_names() noexcept;
  StatusOr<MethodMetaSnapshot> method_meta(const std::string& name) noexcept;
  Status load_method(const std::string& name) noexcept;
  StatusOr<bool> unload_method(const std::string& name) noexcept;
  bool is_method_loaded(const std::string& name) noexcept;

  // Thread-safe execution (per-method mutex under shared lifecycle lock).
  // If `out_targets` is non-empty, writes tensor outputs directly into the
  // caller-provided pre-allocated output tensors (validating count, shape,
  // dtype, and contiguity) instead of allocating new output tensors.
  StatusOr<std::vector<CoreValue>> execute(
      const std::string& method_name,
      const std::vector<CoreValue>& inputs,
      const std::vector<executorch::extension::TensorPtr>& out_targets) noexcept;

 private:
  ModuleCore() = default;

  std::mutex& mutex_for_method(const std::string& name);

  mutable std::shared_mutex lifecycle_mu_;
  bool closed_ = false;
  std::string source_label_;
  size_t estimated_bytes_ = 0;
  // Owned backing bytes when loaded via open_bytes (must outlive module_).
  std::vector<uint8_t> owned_bytes_;
  std::unique_ptr<executorch::extension::Module> module_;

  std::mutex method_mutexes_mu_;
  std::unordered_map<std::string, std::unique_ptr<std::mutex>> method_mutexes_;
};

// Runtime introspection & threadpool controls
std::vector<std::string> registered_backends() noexcept;
size_t registered_kernel_count() noexcept;
int64_t get_num_threads() noexcept;
Status set_num_threads(int64_t n) noexcept;

}  // namespace etrb

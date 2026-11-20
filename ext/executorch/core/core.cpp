#include "module_core.hpp"
#include "tensor_core.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <new>
#include <sys/stat.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#ifdef ET_USE_THREADPOOL
namespace executorch::extension::threadpool {
class ThreadPool {
 public:
  void _unsafe_reset_threadpool(uint32_t num_threads);
  size_t get_thread_count() const;
};
ThreadPool* get_threadpool();
}  // namespace executorch::extension::threadpool
#endif

namespace etrb {

namespace {

thread_local std::string* g_pal_log_target = nullptr;
std::once_flag g_runtime_init_flag;
std::mutex g_threadpool_mu;
pal_emit_log_message_method g_default_pal_emit = nullptr;

void etrb_pal_emit_log(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  if (g_pal_log_target != nullptr &&
      (level == kError || level == kFatal)) {
    try {
      if (!g_pal_log_target->empty()) {
        g_pal_log_target->push_back('\n');
      }
      if (message != nullptr && length > 0) {
        g_pal_log_target->append(message, length);
      }
    } catch (...) {
      // Never throw across C PAL boundary.
    }
    return;
  }
  if (g_default_pal_emit != nullptr && (level == kError || level == kFatal)) {
    // Only forward uncaptured errors if user opted in via EXECUTORCH_LOG_STDERR.
    static const bool forward_stderr =
        std::getenv("EXECUTORCH_LOG_STDERR") != nullptr;
    if (forward_stderr) {
      g_default_pal_emit(
          timestamp, level, filename, function, line, message, length);
    }
  }
}

bool mul_overflow_size(size_t a, size_t b, size_t* out) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow(a, b, out);
#else
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    return true;
  }
  *out = a * b;
  return false;
#endif
}

}  // namespace

void init_runtime_once() noexcept {
  std::call_once(g_runtime_init_flag, []() {
    executorch::runtime::runtime_init();
    const auto* existing = executorch::runtime::get_pal_impl();
    if (existing != nullptr) {
      g_default_pal_emit = existing->emit_log_message;
    }
    auto impl = executorch::runtime::PalImpl::create(
        &etrb_pal_emit_log, __FILE__);
    executorch::runtime::register_pal(impl);
  });
}

PalLogScope::PalLogScope() noexcept {
  prev_buf_ = g_pal_log_target;
  g_pal_log_target = &buf_;
}

PalLogScope::~PalLogScope() noexcept {
  g_pal_log_target = prev_buf_;
}

std::string PalLogScope::take() noexcept {
  return std::move(buf_);
}

Status status_from_et_error(
    executorch::runtime::Error err,
    const char* context,
    std::string pal_log,
    ErrorKind default_kind) noexcept {
  if (err == executorch::runtime::Error::Ok) {
    return Status::Ok();
  }
  ErrorKind kind = default_kind;
  switch (err) {
    case executorch::runtime::Error::OperatorMissing:
      kind = ErrorKind::OperatorMissing;
      break;
    case executorch::runtime::Error::DelegateInvalidCompatibility:
    case executorch::runtime::Error::DelegateMemoryAllocationFailed:
    case executorch::runtime::Error::DelegateInvalidHandle:
      kind = ErrorKind::Delegate;
      break;
    case executorch::runtime::Error::MemoryAllocationFailed:
    case executorch::runtime::Error::OutOfResources:
      kind = ErrorKind::Memory;
      break;
    case executorch::runtime::Error::AccessFailed:
    case executorch::runtime::Error::InvalidProgram:
    case executorch::runtime::Error::InvalidExternalData:
      kind = ErrorKind::Load;
      break;
    case executorch::runtime::Error::NotFound:
      kind = (default_kind == ErrorKind::MethodNotFound)
                 ? ErrorKind::MethodNotFound
                 : ErrorKind::Load;
      break;
    case executorch::runtime::Error::InvalidArgument:
      if (default_kind == ErrorKind::Load) {
        kind = ErrorKind::Load;
      } else if (default_kind == ErrorKind::MethodNotFound ||
                 pal_log.find("No method named") != std::string::npos) {
        kind = ErrorKind::MethodNotFound;
      } else {
        kind = ErrorKind::Execution;
      }
      break;
    case executorch::runtime::Error::InvalidType:
      kind = ErrorKind::Type;
      break;
    default:
      kind = default_kind;
      break;
  }

  try {
    const char* ename = executorch::runtime::to_string(err);
    std::string msg = std::string(context) + ": " + ename + " (0x" +
                      [&]() {
                        char hex[16];
                        std::snprintf(
                            hex, sizeof(hex), "%x", static_cast<unsigned>(err));
                        return std::string(hex);
                      }() +
                      ")";
    if (!pal_log.empty()) {
      msg += " — " + pal_log;
    }
    return Status::Make(
        kind, std::move(msg), static_cast<uint32_t>(err), std::move(pal_log));
  } catch (...) {
    return Status::Make(
        ErrorKind::Memory, "out of memory formatting ExecuTorch error");
  }
}

// ---------------------------------------------------------------------------
// TensorCore helpers
// ---------------------------------------------------------------------------

AlignedBuffer::~AlignedBuffer() noexcept {
  if (ptr != nullptr) {
    std::free(ptr);
    ptr = nullptr;
    nbytes = 0;
  }
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : ptr(other.ptr), nbytes(other.nbytes) {
  other.ptr = nullptr;
  other.nbytes = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
  if (this != &other) {
    if (ptr != nullptr) {
      std::free(ptr);
    }
    ptr = other.ptr;
    nbytes = other.nbytes;
    other.ptr = nullptr;
    other.nbytes = 0;
  }
  return *this;
}

StatusOr<AlignedBuffer> AlignedBuffer::allocate(size_t nbytes) noexcept {
  AlignedBuffer buf;
  buf.nbytes = nbytes;
  if (nbytes == 0) {
    return StatusOr<AlignedBuffer>::Ok(std::move(buf));
  }
  void* mem = nullptr;
  if (::posix_memalign(&mem, 64, nbytes) != 0 || mem == nullptr) {
    return StatusOr<AlignedBuffer>::Err(
        Status::Make(ErrorKind::Memory, "failed to allocate tensor buffer"));
  }
  buf.ptr = mem;
  return StatusOr<AlignedBuffer>::Ok(std::move(buf));
}

StatusOr<AlignedBuffer> AlignedBuffer::allocate_zeroed(size_t nbytes) noexcept {
  auto res = allocate(nbytes);
  if (!res.ok()) {
    return res;
  }
  if (nbytes > 0 && res.value.ptr != nullptr) {
    std::memset(res.value.ptr, 0, nbytes);
  }
  return res;
}

StatusOr<ShapeInfo> validate_shape(
    const std::vector<int64_t>& raw_dims,
    executorch::aten::ScalarType dtype) noexcept {
  try {
    if (!is_supported_dtype(dtype)) {
      return StatusOr<ShapeInfo>::Err(Status::Make(
          ErrorKind::DType,
          std::string("unsupported dtype: ") + scalar_type_name(dtype)));
    }
    const size_t item_size = scalar_type_size(dtype);
    ShapeInfo info;
    info.sizes.reserve(raw_dims.size());

    constexpr int64_t kMaxDim =
        static_cast<int64_t>(std::numeric_limits<executorch::aten::SizesType>::max());
    size_t numel = 1;
    bool has_zero = false;

    for (size_t i = 0; i < raw_dims.size(); ++i) {
      const int64_t d = raw_dims[i];
      if (d < 0) {
        return StatusOr<ShapeInfo>::Err(Status::Make(
            ErrorKind::Shape,
            "shape dimension at axis " + std::to_string(i) +
                " must be non-negative, got " + std::to_string(d)));
      }
      if (d > kMaxDim) {
        return StatusOr<ShapeInfo>::Err(Status::Make(
            ErrorKind::Shape,
            "shape dimension at axis " + std::to_string(i) +
                " exceeds INT32_MAX: " + std::to_string(d)));
      }
      info.sizes.push_back(static_cast<executorch::aten::SizesType>(d));
      if (d == 0) {
        has_zero = true;
      } else if (!has_zero) {
        size_t next = 0;
        if (mul_overflow_size(numel, static_cast<size_t>(d), &next)) {
          return StatusOr<ShapeInfo>::Err(Status::Make(
              ErrorKind::Shape, "tensor element count overflows size_t"));
        }
        numel = next;
      }
    }

    info.numel = has_zero ? 0 : numel;
    if (mul_overflow_size(info.numel, item_size, &info.nbytes)) {
      return StatusOr<ShapeInfo>::Err(Status::Make(
          ErrorKind::Shape, "tensor byte size overflows size_t"));
    }
    if (info.nbytes > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
      return StatusOr<ShapeInfo>::Err(Status::Make(
          ErrorKind::Shape, "tensor byte size exceeds SSIZE_MAX"));
    }
    return StatusOr<ShapeInfo>::Ok(std::move(info));
  } catch (const std::bad_alloc&) {
    return StatusOr<ShapeInfo>::Err(
        Status::Make(ErrorKind::Memory, "out of memory validating shape"));
  } catch (...) {
    return StatusOr<ShapeInfo>::Err(
        Status::Make(ErrorKind::Shape, "unexpected error validating shape"));
  }
}

bool is_standard_contiguous(const executorch::aten::Tensor& tensor) noexcept {
  const ssize_t ndim = tensor.dim();
  if (ndim <= 0 || tensor.numel() <= 1) {
    return true;
  }
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  if (static_cast<ssize_t>(strides.size()) != ndim) {
    return false;
  }
  int64_t expected_stride = 1;
  for (ssize_t i = ndim - 1; i >= 0; --i) {
    const int64_t sz = static_cast<int64_t>(sizes[i]);
    if (sz == 0) {
      return true;
    }
    if (sz > 1 && static_cast<int64_t>(strides[i]) != expected_stride) {
      return false;
    }
    expected_stride *= sz;
  }
  return true;
}

Status copy_to_contiguous_bytes(
    const executorch::aten::Tensor& tensor,
    void* dst,
    size_t dst_nbytes) noexcept {
  const size_t elem_size = static_cast<size_t>(tensor.element_size());
  const size_t numel = static_cast<size_t>(tensor.numel());
  size_t needed = 0;
  if (mul_overflow_size(numel, elem_size, &needed) || needed != dst_nbytes) {
    return Status::Make(
        ErrorKind::Shape, "destination buffer size does not match tensor nbytes");
  }
  if (needed == 0) {
    return Status::Ok();
  }
  const auto* src = static_cast<const uint8_t*>(tensor.const_data_ptr());
  if (src == nullptr || dst == nullptr) {
    return Status::Make(ErrorKind::Memory, "null tensor data pointer");
  }
  if (is_standard_contiguous(tensor)) {
    std::memcpy(dst, src, needed);
    return Status::Ok();
  }

  // General strided copy handling NHWC / channels-last / transposed layouts.
  try {
    const ssize_t ndim = tensor.dim();
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    auto* out = static_cast<uint8_t*>(dst);

    std::vector<int64_t> coords(static_cast<size_t>(ndim), 0);
    for (size_t linear = 0; linear < numel; ++linear) {
      int64_t elem_offset = 0;
      for (ssize_t d = 0; d < ndim; ++d) {
        elem_offset += coords[static_cast<size_t>(d)] *
                       static_cast<int64_t>(strides[d]);
      }
      std::memcpy(
          out + linear * elem_size,
          src + static_cast<size_t>(elem_offset) * elem_size,
          elem_size);

      for (ssize_t d = ndim - 1; d >= 0; --d) {
        coords[static_cast<size_t>(d)]++;
        if (coords[static_cast<size_t>(d)] < static_cast<int64_t>(sizes[d])) {
          break;
        }
        coords[static_cast<size_t>(d)] = 0;
      }
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::Make(
        ErrorKind::Memory, "out of memory copying strided tensor");
  } catch (...) {
    return Status::Make(
        ErrorKind::Execution, "unexpected error copying strided tensor");
  }
}

StatusOr<executorch::extension::TensorPtr> make_owned_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType dtype,
    AlignedBuffer buf) noexcept {
  try {
    void* raw = buf.ptr;
    auto holder = std::make_shared<AlignedBuffer>(std::move(buf));
    auto ptr = executorch::extension::make_tensor_ptr(
        std::move(sizes),
        raw,
        dtype,
        executorch::aten::Device(executorch::aten::DeviceType::CPU),
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
        [holder = std::move(holder)](void*) {});
    return StatusOr<executorch::extension::TensorPtr>::Ok(std::move(ptr));
  } catch (const std::bad_alloc&) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Memory, "out of memory creating tensor"));
  } catch (const std::exception& ex) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Execution, ex.what()));
  } catch (...) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Execution, "failed creating tensor"));
  }
}

StatusOr<executorch::extension::TensorPtr> make_borrowed_tensor_ptr(
    std::vector<executorch::aten::SizesType> sizes,
    executorch::aten::ScalarType dtype,
    void* data) noexcept {
  try {
    auto ptr = executorch::extension::make_tensor_ptr(
        std::move(sizes),
        data,
        dtype,
        executorch::aten::Device(executorch::aten::DeviceType::CPU),
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
        nullptr);
    return StatusOr<executorch::extension::TensorPtr>::Ok(std::move(ptr));
  } catch (const std::bad_alloc&) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Memory, "out of memory creating borrowed tensor"));
  } catch (const std::exception& ex) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Execution, ex.what()));
  } catch (...) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Execution, "failed creating borrowed tensor"));
  }
}

StatusOr<executorch::extension::TensorPtr> clone_contiguous_tensor_ptr(
    const executorch::aten::Tensor& tensor) noexcept {
  try {
    const auto dtype = tensor.scalar_type();
    const size_t nbytes =
        static_cast<size_t>(tensor.numel()) * static_cast<size_t>(tensor.element_size());
    auto buf_res = AlignedBuffer::allocate(nbytes);
    if (!buf_res.ok()) {
      return StatusOr<executorch::extension::TensorPtr>::Err(
          std::move(buf_res.status));
    }
    auto cp = copy_to_contiguous_bytes(tensor, buf_res.value.ptr, nbytes);
    if (!cp.ok()) {
      return StatusOr<executorch::extension::TensorPtr>::Err(std::move(cp));
    }
    std::vector<executorch::aten::SizesType> sizes(
        tensor.sizes().begin(), tensor.sizes().end());
    return make_owned_tensor_ptr(
        std::move(sizes), dtype, std::move(buf_res.value));
  } catch (const std::bad_alloc&) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Memory, "out of memory cloning tensor"));
  } catch (...) {
    return StatusOr<executorch::extension::TensorPtr>::Err(
        Status::Make(ErrorKind::Execution, "failed cloning tensor"));
  }
}

// ---------------------------------------------------------------------------
// ModuleCore
// ---------------------------------------------------------------------------

StatusOr<std::unique_ptr<ModuleCore>> ModuleCore::open_file(
    const std::string& path,
    const std::vector<std::string>& data_files,
    LoadMode load_mode) noexcept {
  init_runtime_once();
  PalLogScope pal;
  try {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
      return StatusOr<std::unique_ptr<ModuleCore>>::Err(Status::Make(
          ErrorKind::Load,
          "Cannot open ExecuTorch program '" + path + "': " +
              std::strerror(errno)));
    }
    if (S_ISDIR(st.st_mode)) {
      return StatusOr<std::unique_ptr<ModuleCore>>::Err(Status::Make(
          ErrorKind::Load,
          "ExecuTorch program path is a directory: '" + path + "'"));
    }

    size_t est = static_cast<size_t>(st.st_size);
    for (const auto& df : data_files) {
      struct stat dst {};
      if (::stat(df.c_str(), &dst) != 0 || S_ISDIR(dst.st_mode)) {
        return StatusOr<std::unique_ptr<ModuleCore>>::Err(Status::Make(
            ErrorKind::Load,
            "Cannot open external data file '" + df + "'"));
      }
      est += static_cast<size_t>(dst.st_size);
    }

    std::unique_ptr<ModuleCore> core(new ModuleCore());
    core->source_label_ = path;
    core->estimated_bytes_ = est;

    if (data_files.empty()) {
      core->module_ = std::make_unique<executorch::extension::Module>(
          path, load_mode);
    } else {
      core->module_ = std::make_unique<executorch::extension::Module>(
          path, data_files, load_mode);
    }

    auto err = core->module_->load(
        executorch::runtime::Program::Verification::Minimal);
    if (err != executorch::runtime::Error::Ok) {
      return StatusOr<std::unique_ptr<ModuleCore>>::Err(status_from_et_error(
          err,
          ("Failed to load ExecuTorch program '" + path + "'").c_str(),
          pal.take(),
          ErrorKind::Load));
    }

    return StatusOr<std::unique_ptr<ModuleCore>>::Ok(std::move(core));
  } catch (const std::bad_alloc&) {
    return StatusOr<std::unique_ptr<ModuleCore>>::Err(
        Status::Make(ErrorKind::Memory, "out of memory loading module"));
  } catch (const std::exception& ex) {
    return StatusOr<std::unique_ptr<ModuleCore>>::Err(
        Status::Make(ErrorKind::Load, ex.what(), 0, pal.take()));
  } catch (...) {
    return StatusOr<std::unique_ptr<ModuleCore>>::Err(
        Status::Make(ErrorKind::Load, "unknown error loading module", 0, pal.take()));
  }
}

StatusOr<std::unique_ptr<ModuleCore>> ModuleCore::open_bytes(
    std::vector<uint8_t> bytes) noexcept {
  init_runtime_once();
  PalLogScope pal;
  try {
    if (bytes.empty()) {
      return StatusOr<std::unique_ptr<ModuleCore>>::Err(Status::Make(
          ErrorKind::Load, "Cannot load ExecuTorch program from empty bytes"));
    }
    std::unique_ptr<ModuleCore> core(new ModuleCore());
    core->source_label_ = "<bytes:" + std::to_string(bytes.size()) + ">";
    core->estimated_bytes_ = bytes.size();
    core->owned_bytes_ = std::move(bytes);

    auto loader = std::make_unique<executorch::extension::BufferDataLoader>(
        core->owned_bytes_.data(), core->owned_bytes_.size());
    core->module_ = std::make_unique<executorch::extension::Module>(
        std::move(loader));

    auto err = core->module_->load(
        executorch::runtime::Program::Verification::Minimal);
    if (err != executorch::runtime::Error::Ok) {
      return StatusOr<std::unique_ptr<ModuleCore>>::Err(status_from_et_error(
          err,
          "Failed to load ExecuTorch program from bytes",
          pal.take(),
          ErrorKind::Load));
    }
    return StatusOr<std::unique_ptr<ModuleCore>>::Ok(std::move(core));
  } catch (const std::bad_alloc&) {
    return StatusOr<std::unique_ptr<ModuleCore>>::Err(
        Status::Make(ErrorKind::Memory, "out of memory loading module from bytes"));
  } catch (const std::exception& ex) {
    return StatusOr<std::unique_ptr<ModuleCore>>::Err(
        Status::Make(ErrorKind::Load, ex.what(), 0, pal.take()));
  } catch (...) {
    return StatusOr<std::unique_ptr<ModuleCore>>::Err(
        Status::Make(ErrorKind::Load, "unknown error loading bytes", 0, pal.take()));
  }
}

void ModuleCore::close() noexcept {
  std::unique_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_) {
    return;
  }
  closed_ = true;
  // Destroy module BEFORE clearing owned_bytes_ (BufferDataLoader borrows it).
  module_.reset();
  owned_bytes_.clear();
  owned_bytes_.shrink_to_fit();
  estimated_bytes_ = 0;
}

bool ModuleCore::is_closed() const noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  return closed_;
}

std::string ModuleCore::source_label() const noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  return source_label_;
}

size_t ModuleCore::estimated_memory_bytes() const noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  return estimated_bytes_;
}

std::mutex& ModuleCore::mutex_for_method(const std::string& name) {
  std::lock_guard<std::mutex> g(method_mutexes_mu_);
  auto& ptr = method_mutexes_[name];
  if (!ptr) {
    ptr = std::make_unique<std::mutex>();
  }
  return *ptr;
}

StatusOr<std::vector<std::string>> ModuleCore::method_names() noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_ || !module_) {
    return StatusOr<std::vector<std::string>>::Err(
        Status::Make(ErrorKind::Closed, "ExecuTorch module is closed"));
  }
  PalLogScope pal;
  try {
    auto res = module_->method_names();
    if (!res.ok()) {
      return StatusOr<std::vector<std::string>>::Err(status_from_et_error(
          res.error(), "Failed to list method names", pal.take()));
    }
    std::vector<std::string> names(res->begin(), res->end());
    std::sort(names.begin(), names.end());
    return StatusOr<std::vector<std::string>>::Ok(std::move(names));
  } catch (const std::bad_alloc&) {
    return StatusOr<std::vector<std::string>>::Err(
        Status::Make(ErrorKind::Memory, "out of memory listing methods"));
  } catch (...) {
    return StatusOr<std::vector<std::string>>::Err(
        Status::Make(ErrorKind::Execution, "unexpected error listing methods"));
  }
}

StatusOr<MethodMetaSnapshot> ModuleCore::method_meta(
    const std::string& name) noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_ || !module_) {
    return StatusOr<MethodMetaSnapshot>::Err(
        Status::Make(ErrorKind::Closed, "ExecuTorch module is closed"));
  }
  PalLogScope pal;
  try {
    auto res = module_->method_meta(name);
    if (!res.ok()) {
      return StatusOr<MethodMetaSnapshot>::Err(status_from_et_error(
          res.error(),
          ("Failed to read metadata for method '" + name + "'").c_str(),
          pal.take(),
          ErrorKind::MethodNotFound));
    }
    const auto& meta = res.get();
    MethodMetaSnapshot snap;
    snap.name = meta.name() ? meta.name() : name;

    const size_t n_in = meta.num_inputs();
    snap.inputs.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
      ValueMetaSnapshot vm;
      auto tag_res = meta.input_tag(i);
      if (tag_res.ok()) {
        vm.tag = tag_res.get();
        if (vm.tag == executorch::runtime::Tag::Tensor) {
          auto tm_res = meta.input_tensor_meta(i);
          if (tm_res.ok()) {
            const auto& tm = tm_res.get();
            vm.tensor.name = std::string(tm.name());
            vm.tensor.dtype = tm.scalar_type();
            vm.tensor.sizes.assign(tm.sizes().begin(), tm.sizes().end());
            vm.tensor.dim_order.assign(
                tm.dim_order().begin(), tm.dim_order().end());
            vm.tensor.nbytes = tm.nbytes();
            vm.tensor.memory_planned = tm.is_memory_planned();
          }
        }
      }
      snap.inputs.push_back(std::move(vm));
    }

    const size_t n_out = meta.num_outputs();
    snap.outputs.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
      ValueMetaSnapshot vm;
      auto tag_res = meta.output_tag(i);
      if (tag_res.ok()) {
        vm.tag = tag_res.get();
        if (vm.tag == executorch::runtime::Tag::Tensor) {
          auto tm_res = meta.output_tensor_meta(i);
          if (tm_res.ok()) {
            const auto& tm = tm_res.get();
            vm.tensor.name = std::string(tm.name());
            vm.tensor.dtype = tm.scalar_type();
            vm.tensor.sizes.assign(tm.sizes().begin(), tm.sizes().end());
            vm.tensor.dim_order.assign(
                tm.dim_order().begin(), tm.dim_order().end());
            vm.tensor.nbytes = tm.nbytes();
            vm.tensor.memory_planned = tm.is_memory_planned();
          }
        }
      }
      snap.outputs.push_back(std::move(vm));
    }

    const size_t n_backends = meta.num_backends();
    snap.backends.reserve(n_backends);
    for (size_t i = 0; i < n_backends; ++i) {
      auto b_res = meta.get_backend_name(i);
      if (b_res.ok() && b_res.get() != nullptr) {
        snap.backends.emplace_back(b_res.get());
      }
    }

    const size_t n_bufs = meta.num_memory_planned_buffers();
    snap.planned_buffer_sizes.reserve(n_bufs);
    int64_t total = 0;
    for (size_t i = 0; i < n_bufs; ++i) {
      auto sz_res = meta.memory_planned_buffer_size(i);
      if (sz_res.ok()) {
        snap.planned_buffer_sizes.push_back(sz_res.get());
        if (sz_res.get() > 0) {
          total += sz_res.get();
        }
      }
    }
    snap.planned_memory_bytes = total;

    return StatusOr<MethodMetaSnapshot>::Ok(std::move(snap));
  } catch (const std::bad_alloc&) {
    return StatusOr<MethodMetaSnapshot>::Err(
        Status::Make(ErrorKind::Memory, "out of memory reading method_meta"));
  } catch (...) {
    return StatusOr<MethodMetaSnapshot>::Err(
        Status::Make(ErrorKind::Execution, "unexpected error reading method_meta"));
  }
}

Status ModuleCore::load_method(const std::string& name) noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_ || !module_) {
    return Status::Make(ErrorKind::Closed, "ExecuTorch module is closed");
  }
  PalLogScope pal;
  try {
    // Module::methods_ map is mutated on load_method, so serialize on method_mutexes_mu_
    // while checking/loading the method.
    std::mutex& m_mu = mutex_for_method(name);
    std::lock_guard<std::mutex> mg(m_mu);
    auto err = module_->load_method(name);
    if (err != executorch::runtime::Error::Ok) {
      return status_from_et_error(
          err,
          ("Failed to load method '" + name + "'").c_str(),
          pal.take(),
          ErrorKind::MethodNotFound);
    }
    return Status::Ok();
  } catch (const std::bad_alloc&) {
    return Status::Make(ErrorKind::Memory, "out of memory loading method");
  } catch (const std::exception& ex) {
    return Status::Make(ErrorKind::Execution, ex.what(), 0, pal.take());
  } catch (...) {
    return Status::Make(
        ErrorKind::Execution, "unknown error loading method", 0, pal.take());
  }
}

StatusOr<bool> ModuleCore::unload_method(const std::string& name) noexcept {
  // Unique lock ensures no concurrent execute() is running on any method
  // while Module::methods_ hash map is modified.
  std::unique_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_ || !module_) {
    return StatusOr<bool>::Err(
        Status::Make(ErrorKind::Closed, "ExecuTorch module is closed"));
  }
  try {
    return StatusOr<bool>::Ok(module_->unload_method(name));
  } catch (...) {
    return StatusOr<bool>::Err(
        Status::Make(ErrorKind::Execution, "failed unloading method"));
  }
}

bool ModuleCore::is_method_loaded(const std::string& name) noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_ || !module_) {
    return false;
  }
  try {
    std::mutex& m_mu = mutex_for_method(name);
    std::lock_guard<std::mutex> mg(m_mu);
    return module_->is_method_loaded(name);
  } catch (...) {
    return false;
  }
}

StatusOr<std::vector<CoreValue>> ModuleCore::execute(
    const std::string& method_name,
    const std::vector<CoreValue>& inputs,
    const std::vector<executorch::extension::TensorPtr>& out_targets) noexcept {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mu_);
  if (closed_ || !module_) {
    return StatusOr<std::vector<CoreValue>>::Err(
        Status::Make(ErrorKind::Closed, "ExecuTorch module is closed"));
  }
  PalLogScope pal;
  try {
    // Ensure method is loaded under an exclusive check if not already loaded,
    // because Module::methods_ (std::unordered_map) is mutated on first load.
    {
      std::mutex& m_mu = mutex_for_method(method_name);
      std::lock_guard<std::mutex> mg(m_mu);
      if (!module_->is_method_loaded(method_name)) {
        lock.unlock();
        {
          std::unique_lock<std::shared_mutex> ulock(lifecycle_mu_);
          if (closed_ || !module_) {
            return StatusOr<std::vector<CoreValue>>::Err(
                Status::Make(ErrorKind::Closed, "ExecuTorch module is closed"));
          }
          if (!module_->is_method_loaded(method_name)) {
            auto load_err = module_->load_method(method_name);
            if (load_err != executorch::runtime::Error::Ok) {
              return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
                  load_err,
                  ("Failed to load method '" + method_name + "'").c_str(),
                  pal.take(),
                  ErrorKind::MethodNotFound));
            }
          }
        }
        lock.lock();
        if (closed_ || !module_) {
          return StatusOr<std::vector<CoreValue>>::Err(
              Status::Make(ErrorKind::Closed, "ExecuTorch module is closed"));
        }
      }
    }

    // Per-method lock: ExecuTorch Method owns a single mutable planned arena.
    std::mutex& m_mu = mutex_for_method(method_name);
    std::lock_guard<std::mutex> mg(m_mu);

    // Backing containers (std::deque guarantees pointer stability on push_back).
    std::deque<executorch::aten::ArrayRef<char>> str_refs;
    std::deque<executorch::aten::ArrayRef<double>> dbl_list_refs;
    std::deque<std::vector<bool>> bool_vecs;
    std::deque<std::unique_ptr<bool[]>> bool_bufs;
    std::deque<executorch::aten::ArrayRef<bool>> bool_list_refs;

    std::vector<executorch::runtime::EValue> evalues;
    evalues.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& in = inputs[i];
      switch (in.kind) {
        case ValueKind::None:
          evalues.emplace_back();
          break;
        case ValueKind::Int:
          evalues.emplace_back(in.int_val);
          break;
        case ValueKind::Double:
          evalues.emplace_back(in.double_val);
          break;
        case ValueKind::Bool:
          evalues.emplace_back(in.bool_val);
          break;
        case ValueKind::String: {
          str_refs.emplace_back(in.string_val.data(), in.string_val.size());
          evalues.emplace_back(&str_refs.back());
          break;
        }
        case ValueKind::Tensor: {
          if (!in.tensor_val) {
            return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
                ErrorKind::Closed,
                "Input tensor " + std::to_string(i) + " is closed"));
          }
          evalues.emplace_back(*in.tensor_val);
          break;
        }
        case ValueKind::DoubleList: {
          dbl_list_refs.emplace_back(in.double_list.data(), in.double_list.size());
          evalues.emplace_back(&dbl_list_refs.back());
          break;
        }
        case ValueKind::BoolList: {
          auto arr = std::make_unique<bool[]>(in.bool_list.size());
          for (size_t j = 0; j < in.bool_list.size(); ++j) {
            arr[j] = in.bool_list[j];
          }
          bool_list_refs.emplace_back(arr.get(), in.bool_list.size());
          bool_bufs.push_back(std::move(arr));
          evalues.emplace_back(&bool_list_refs.back());
          break;
        }
        default:
          return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
              ErrorKind::Type,
              "Unsupported input EValue kind at index " + std::to_string(i)));
      }
    }

    auto res = module_->execute(method_name, evalues);
    if (!res.ok()) {
      return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
          res.error(),
          ("Execution of method '" + method_name + "' failed").c_str(),
          pal.take(),
          ErrorKind::Execution));
    }

    const auto& raw_outputs = res.get();
    if (!out_targets.empty() && out_targets.size() != raw_outputs.size()) {
      return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
          ErrorKind::Shape,
          "out: expected " + std::to_string(raw_outputs.size()) +
              " output tensor(s), got " + std::to_string(out_targets.size())));
    }

    std::vector<CoreValue> outputs;
    outputs.reserve(raw_outputs.size());

    for (size_t i = 0; i < raw_outputs.size(); ++i) {
      const auto& ev = raw_outputs[i];
      if (ev.isTensor()) {
        const auto& src = ev.toTensor();
        if (!out_targets.empty()) {
          const auto& dst = out_targets[i];
          if (!dst) {
            return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
                ErrorKind::Closed,
                "out: tensor " + std::to_string(i) + " is closed"));
          }
          if (dst->scalar_type() != src.scalar_type()) {
            return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
                ErrorKind::DType,
                std::string("out: dtype mismatch at output ") +
                    std::to_string(i) + ": expected " +
                    scalar_type_name(src.scalar_type()) + ", got " +
                    scalar_type_name(dst->scalar_type())));
          }
          bool same_shape = (dst->dim() == src.dim());
          if (same_shape) {
            for (ssize_t d = 0; d < src.dim(); ++d) {
              if (dst->size(d) != src.size(d)) {
                same_shape = false;
                break;
              }
            }
          }
          if (!same_shape) {
            return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
                ErrorKind::Shape,
                "out: shape mismatch at output " + std::to_string(i)));
          }
          auto cp = copy_to_contiguous_bytes(
              src, dst->mutable_data_ptr(), dst->nbytes());
          if (!cp.ok()) {
            return StatusOr<std::vector<CoreValue>>::Err(std::move(cp));
          }
          outputs.push_back(CoreValue::Tensor(dst));
        } else {
          auto cloned = clone_contiguous_tensor_ptr(src);
          if (!cloned.ok()) {
            return StatusOr<std::vector<CoreValue>>::Err(
                std::move(cloned.status));
          }
          outputs.push_back(CoreValue::Tensor(std::move(cloned.value)));
        }
      } else if (!out_targets.empty()) {
        return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
            ErrorKind::Type,
            "out: target provided for non-tensor output " + std::to_string(i)));
      } else if (ev.isInt()) {
        outputs.push_back(CoreValue::Int(ev.toInt()));
      } else if (ev.isDouble()) {
        outputs.push_back(CoreValue::Double(ev.toDouble()));
      } else if (ev.isBool()) {
        outputs.push_back(CoreValue::Bool(ev.toBool()));
      } else if (ev.isNone()) {
        outputs.push_back(CoreValue::None());
      } else if (ev.isString()) {
        auto sv = ev.tryToString();
        if (!sv.ok()) {
          return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
              sv.error(), "Failed to extract String output", pal.take()));
        }
        outputs.push_back(CoreValue::String(std::string(sv.get())));
      } else if (ev.isIntList()) {
        auto lr = ev.tryToIntList();
        if (!lr.ok()) {
          return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
              lr.error(), "Failed to extract IntList output", pal.take()));
        }
        CoreValue cv;
        cv.kind = ValueKind::IntList;
        cv.int_list.assign(lr.get().begin(), lr.get().end());
        outputs.push_back(std::move(cv));
      } else if (ev.isDoubleList()) {
        auto lr = ev.tryToDoubleList();
        if (!lr.ok()) {
          return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
              lr.error(), "Failed to extract DoubleList output", pal.take()));
        }
        CoreValue cv;
        cv.kind = ValueKind::DoubleList;
        cv.double_list.assign(lr.get().begin(), lr.get().end());
        outputs.push_back(std::move(cv));
      } else if (ev.isBoolList()) {
        auto lr = ev.tryToBoolList();
        if (!lr.ok()) {
          return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
              lr.error(), "Failed to extract BoolList output", pal.take()));
        }
        CoreValue cv;
        cv.kind = ValueKind::BoolList;
        cv.bool_list.assign(lr.get().begin(), lr.get().end());
        outputs.push_back(std::move(cv));
      } else if (ev.isTensorList()) {
        auto lr = ev.tryToTensorList();
        if (!lr.ok()) {
          return StatusOr<std::vector<CoreValue>>::Err(status_from_et_error(
              lr.error(), "Failed to extract TensorList output", pal.take()));
        }
        CoreValue cv;
        cv.kind = ValueKind::TensorList;
        for (const auto& t : lr.get()) {
          auto cl = clone_contiguous_tensor_ptr(t);
          if (!cl.ok()) {
            return StatusOr<std::vector<CoreValue>>::Err(std::move(cl.status));
          }
          cv.tensor_list.push_back(std::move(cl.value));
        }
        outputs.push_back(std::move(cv));
      } else {
        return StatusOr<std::vector<CoreValue>>::Err(Status::Make(
            ErrorKind::Type,
            "Unsupported output EValue tag at output " + std::to_string(i)));
      }
    }

    return StatusOr<std::vector<CoreValue>>::Ok(std::move(outputs));
  } catch (const std::bad_alloc&) {
    return StatusOr<std::vector<CoreValue>>::Err(
        Status::Make(ErrorKind::Memory, "out of memory executing method"));
  } catch (const std::exception& ex) {
    return StatusOr<std::vector<CoreValue>>::Err(
        Status::Make(ErrorKind::Execution, ex.what(), 0, pal.take()));
  } catch (...) {
    return StatusOr<std::vector<CoreValue>>::Err(
        Status::Make(ErrorKind::Execution, "unknown error executing method", 0, pal.take()));
  }
}

// ---------------------------------------------------------------------------
// Runtime introspection & threadpool controls
// ---------------------------------------------------------------------------

std::vector<std::string> registered_backends() noexcept {
  init_runtime_once();
  std::vector<std::string> out;
  try {
    const size_t n = executorch::runtime::get_num_registered_backends();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      auto res = executorch::runtime::get_backend_name(i);
      if (res.ok() && res.get() != nullptr) {
        out.emplace_back(res.get());
      }
    }
    std::sort(out.begin(), out.end());
  } catch (...) {
  }
  return out;
}

size_t registered_kernel_count() noexcept {
  init_runtime_once();
  return executorch::runtime::get_registered_kernels().size();
}

int64_t get_num_threads() noexcept {
  init_runtime_once();
#ifdef ET_USE_THREADPOOL
  std::lock_guard<std::mutex> g(g_threadpool_mu);
  auto* tp = executorch::extension::threadpool::get_threadpool();
  if (tp != nullptr) {
    return static_cast<int64_t>(tp->get_thread_count());
  }
#endif
  return executorch::extension::get_thread_count();
}

Status set_num_threads(int64_t n) noexcept {
  init_runtime_once();
  if (n < 1 || n > 1024) {
    return Status::Make(
        ErrorKind::Argument,
        "num_threads must be between 1 and 1024, got " + std::to_string(n));
  }
#ifdef ET_USE_THREADPOOL
  try {
    std::lock_guard<std::mutex> g(g_threadpool_mu);
    auto* tp = executorch::extension::threadpool::get_threadpool();
    if (tp == nullptr) {
      return Status::Make(
          ErrorKind::Execution, "ExecuTorch threadpool is unavailable");
    }
    tp->_unsafe_reset_threadpool(static_cast<uint32_t>(n));
    return Status::Ok();
  } catch (...) {
    return Status::Make(
        ErrorKind::Execution, "failed resetting ExecuTorch threadpool");
  }
#else
  return Status::Make(
      ErrorKind::Execution,
      "ExecuTorch was built without threadpool support (ET_USE_THREADPOOL)");
#endif
}

}  // namespace etrb

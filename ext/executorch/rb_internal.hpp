#pragma once

#include <ruby.h>
#include <ruby/encoding.h>
#include <ruby/io/buffer.h>
#include <ruby/thread.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/dtype.hpp"
#include "core/module_core.hpp"
#include "core/status.hpp"
#include "core/tensor_core.hpp"

namespace etrb {

// Cached Ruby modules, classes, and symbols initialized once in Init_executorch.
struct RubyRefs {
  VALUE mExecutorch = Qnil;
  VALUE cModule = Qnil;
  VALUE cMethodMeta = Qnil;
  VALUE cTensorInfo = Qnil;
  VALUE cTensor = Qnil;

  // Error hierarchy (all < Executorch::Error < StandardError)
  VALUE eError = Qnil;
  VALUE eLoadError = Qnil;
  VALUE eMethodNotFoundError = Qnil;
  VALUE eOperatorMissingError = Qnil;
  VALUE eDelegateError = Qnil;
  VALUE eShapeError = Qnil;
  VALUE eDTypeError = Qnil;
  VALUE eExecutionError = Qnil;
  VALUE eMemoryError = Qnil;
  VALUE eClosedError = Qnil;

  // Cached IDs
  ID id_code = 0;
  ID id_pal_log = 0;
  ID id_float32 = 0;
  ID id_float64 = 0;
  ID id_float16 = 0;
  ID id_bfloat16 = 0;
  ID id_uint8 = 0;
  ID id_int8 = 0;
  ID id_int16 = 0;
  ID id_uint16 = 0;
  ID id_int32 = 0;
  ID id_uint32 = 0;
  ID id_int64 = 0;
  ID id_uint64 = 0;
  ID id_bool = 0;
  ID id_unknown = 0;

  // LoadMode symbols
  ID id_mmap = 0;
  ID id_file = 0;
  ID id_mmap_mlock = 0;
  ID id_mmap_mlock_ignore_errors = 0;
  ID id_mmap_madvise = 0;

  // Keyword IDs
  ID id_kw_shape = 0;
  ID id_kw_dtype = 0;
  ID id_kw_load_mode = 0;
  ID id_kw_data_files = 0;
  ID id_kw_out = 0;
};

extern RubyRefs g_refs;

// TypedData structs
struct RbTensorWrap {
  // Owned TensorPtr (null if borrowed_buf != Qnil or closed).
  executorch::extension::TensorPtr tensor;
  // When created via Tensor.from_buffer, holds the Ruby IO::Buffer VALUE
  // (marked in dmark so GC keeps it alive; locked around off-GVL execute).
  VALUE borrowed_buf = Qnil;
  std::vector<executorch::aten::SizesType> borrowed_sizes;
  executorch::aten::ScalarType borrowed_dtype =
      executorch::aten::ScalarType::Undefined;
  size_t borrowed_nbytes = 0;
  size_t tracked_bytes = 0;
  bool closed = false;
};

struct RbModuleWrap {
  std::unique_ptr<ModuleCore> core;
  size_t tracked_bytes = 0;
};

extern const rb_data_type_t kTensorDataType;
extern const rb_data_type_t kModuleDataType;

// Subsystem initializers
void init_errors(VALUE mExecutorch);
void init_tensor(VALUE mExecutorch);
void init_module(VALUE mExecutorch);

// Convert Status into a Ruby exception VALUE WITHOUT raising (so callers can
// finish destroying all C++ objects on the stack before rb_exc_raise).
VALUE status_to_ruby_exception(const Status& st);

// Helper: build Ruby exception from Status and raise it.
// SAFETY: Caller MUST ensure NO C++ objects with non-trivial destructors are
// live on the stack in the caller's frame (or any intervening C++ frames)!
[[noreturn]] void raise_status_no_live_cpp(VALUE exc);

// Tensor interop helpers used by rb_module.cpp
VALUE wrap_owned_tensor(executorch::extension::TensorPtr ptr);
bool is_rb_tensor(VALUE obj);
RbTensorWrap* get_rb_tensor(VALUE obj);

// Dtype Symbol <-> ScalarType helpers (using cached IDs, zero rb_intern on hot path)
VALUE scalar_type_to_sym(executorch::aten::ScalarType st);
bool sym_to_scalar_type(VALUE sym, executorch::aten::ScalarType* out);

}  // namespace etrb

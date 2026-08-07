#ifndef EXECUTORCH_RUBY_UTILS_H
#define EXECUTORCH_RUBY_UTILS_H

#include <rice/rice.hpp>
#include <rice/stl.hpp>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <cstdint>
#include <vector>

// Error handling macros - translate C++ exceptions to Ruby exceptions
#define HANDLE_ET_ERRORS try {

#define END_HANDLE_ET_ERRORS                                                    \
  }                                                                             \
  catch (const Rice::Exception &ex) {                                           \
    throw;                                                                      \
  }                                                                             \
  catch (const std::exception &ex) {                                            \
    rb_raise(rb_eRuntimeError, "ExecuTorch error: %s", ex.what());              \
  }

namespace executorch_ruby {

// Human-readable name for a runtime error code.
inline const char* error_name(executorch::runtime::Error error) {
  switch (error) {
    case executorch::runtime::Error::Ok:
      return "Ok";
    case executorch::runtime::Error::Internal:
      return "Internal error";
    case executorch::runtime::Error::InvalidState:
      return "Invalid state";
    case executorch::runtime::Error::InvalidArgument:
      return "Invalid argument";
    case executorch::runtime::Error::InvalidType:
      return "Invalid type";
    case executorch::runtime::Error::NotFound:
      return "Not found";
    case executorch::runtime::Error::MemoryAllocationFailed:
      return "Memory allocation failed";
    case executorch::runtime::Error::AccessFailed:
      return "Access failed";
    case executorch::runtime::Error::NotSupported:
      return "Not supported";
    case executorch::runtime::Error::DelegateInvalidCompatibility:
      return "Delegate invalid compatibility";
    case executorch::runtime::Error::DelegateMemoryAllocationFailed:
      return "Delegate memory allocation failed";
    case executorch::runtime::Error::DelegateInvalidHandle:
      return "Delegate invalid handle";
    default:
      return "Unknown error";
  }
}

inline void check_error(executorch::runtime::Error error) {
  if (error != executorch::runtime::Error::Ok) {
    rb_raise(rb_eRuntimeError, "ExecuTorch error: %s", error_name(error));
  }
}

// Helper to unwrap Result<T> and raise Ruby exception on error
template <typename T>
T unwrap_result(executorch::runtime::Result<T>&& result) {
  if (!result.ok()) {
    check_error(result.error());
    // Should never reach here, but just in case
    rb_raise(rb_eRuntimeError, "ExecuTorch: unexpected error");
  }
  return std::move(result.get());
}

// Convert ExecuTorch ScalarType to Ruby symbol
// Uses short names (:int, :long, :float) to match input symbols
inline VALUE scalar_type_to_symbol(executorch::aten::ScalarType dtype) {
  switch (dtype) {
    case executorch::aten::ScalarType::Byte:
      return ID2SYM(rb_intern("byte"));
    case executorch::aten::ScalarType::Char:
      return ID2SYM(rb_intern("char"));
    case executorch::aten::ScalarType::Short:
      return ID2SYM(rb_intern("short"));
    case executorch::aten::ScalarType::Int:
      return ID2SYM(rb_intern("int"));
    case executorch::aten::ScalarType::Long:
      return ID2SYM(rb_intern("long"));
    case executorch::aten::ScalarType::Half:
      return ID2SYM(rb_intern("half"));
    case executorch::aten::ScalarType::Float:
      return ID2SYM(rb_intern("float"));
    case executorch::aten::ScalarType::Double:
      return ID2SYM(rb_intern("double"));
    case executorch::aten::ScalarType::Bool:
      return ID2SYM(rb_intern("bool"));
    default:
      return ID2SYM(rb_intern("unknown"));
  }
}

inline executorch::aten::ScalarType symbol_to_scalar_type(VALUE sym) {
  if (!RB_TYPE_P(sym, T_SYMBOL)) {
    rb_raise(rb_eTypeError, "Expected Symbol for dtype");
  }

  ID id = SYM2ID(sym);

  if (id == rb_intern("uint8") || id == rb_intern("byte")) {
    return executorch::aten::ScalarType::Byte;
  } else if (id == rb_intern("int8") || id == rb_intern("char")) {
    return executorch::aten::ScalarType::Char;
  } else if (id == rb_intern("int16") || id == rb_intern("short")) {
    return executorch::aten::ScalarType::Short;
  } else if (id == rb_intern("int32") || id == rb_intern("int")) {
    return executorch::aten::ScalarType::Int;
  } else if (id == rb_intern("int64") || id == rb_intern("long")) {
    return executorch::aten::ScalarType::Long;
  } else if (id == rb_intern("float16") || id == rb_intern("half")) {
    return executorch::aten::ScalarType::Half;
  } else if (id == rb_intern("float32") || id == rb_intern("float")) {
    return executorch::aten::ScalarType::Float;
  } else if (id == rb_intern("float64") || id == rb_intern("double")) {
    return executorch::aten::ScalarType::Double;
  } else if (id == rb_intern("bool")) {
    return executorch::aten::ScalarType::Bool;
  } else {
    rb_raise(rb_eArgError, "Unknown dtype: %s", rb_id2name(id));
  }
}

} // namespace executorch_ruby

// ---------------------------------------------------------------------------
// Fast scalar conversion
// ---------------------------------------------------------------------------
//
// Rice's From_Ruby/To_Ruby route every single conversion through
// detail::protect(), which wraps the call in rb_protect() -- a VM tag push plus
// a setjmp. That is the right default for arbitrary Ruby calls, but for reading
// a Float out of an Array it costs far more than the conversion itself, and we
// pay it once per tensor element.
//
// The helpers below take the inline path for the two types that make up
// essentially all real tensor data (Float and Fixnum) and fall back to Rice's
// protected conversion only for the rare cases -- Bignum, Rational, or an
// object with a coercion method -- where the call really can raise or run Ruby
// code.

namespace executorch_ruby {

inline double to_double_fast(VALUE v) {
  if (RB_FLOAT_TYPE_P(v)) {
    return RFLOAT_VALUE(v);
  }
  if (FIXNUM_P(v)) {
    return static_cast<double>(FIX2LONG(v));
  }
  return Rice::detail::protect(rb_num2dbl, v);
}

inline int64_t to_int64_fast(VALUE v) {
  if (FIXNUM_P(v)) {
    return FIX2LONG(v);
  }
  if (RB_FLOAT_TYPE_P(v)) {
    return static_cast<int64_t>(RFLOAT_VALUE(v));
  }
  return Rice::detail::protect(rb_num2ll, v);
}

// Read a Ruby Array of numbers into a freshly built std::vector<T>.
//
// RARRAY_AREF is re-read every iteration rather than caching RARRAY_CONST_PTR:
// the slow-path conversion above can run Ruby code, which could reallocate the
// array's backing store and leave a cached pointer dangling.
template <typename T, typename Convert_T>
std::vector<T> read_array(VALUE ary, Convert_T convert) {
  const long n = RARRAY_LEN(ary);
  std::vector<T> out;
  out.reserve(static_cast<size_t>(n));
  for (long i = 0; i < n; i++) {
    out.push_back(static_cast<T>(convert(RARRAY_AREF(ary, i))));
  }
  return out;
}

// Build a Ruby Array from a contiguous C buffer.
//
// rb_ary_new_capa sizes the backing store once, and rb_ary_push on an array we
// just created cannot raise (it is neither frozen nor shared), so there is no
// need for the protected call Rice::Array::push would make. Doubles that fit
// become flonums, so the common case allocates nothing per element.
template <typename T, typename Box_T>
VALUE build_array(const T* data, int64_t n, Box_T box) {
  VALUE ary = rb_ary_new_capa(static_cast<long>(n));
  for (int64_t i = 0; i < n; i++) {
    rb_ary_push(ary, box(data[i]));
  }
  return ary;
}

// Bytes per element, for the raw-binary tensor path.
inline size_t element_size(executorch::aten::ScalarType dtype) {
  switch (dtype) {
    case executorch::aten::ScalarType::Float:  return sizeof(float);
    case executorch::aten::ScalarType::Double: return sizeof(double);
    case executorch::aten::ScalarType::Int:    return sizeof(int32_t);
    case executorch::aten::ScalarType::Long:   return sizeof(int64_t);
    default:
      rb_raise(rb_eArgError, "Binary tensor I/O supports :float, :double, :int, :long");
  }
}

} // namespace executorch_ruby

#endif // EXECUTORCH_RUBY_UTILS_H

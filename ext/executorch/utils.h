#ifndef EXECUTORCH_RUBY_UTILS_H
#define EXECUTORCH_RUBY_UTILS_H

#include <rice/rice.hpp>
#include <rice/stl.hpp>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

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

inline void check_error(executorch::runtime::Error error) {
  if (error != executorch::runtime::Error::Ok) {
    const char* error_name = "Unknown error";
    switch (error) {
      case executorch::runtime::Error::Ok:
        return;
      case executorch::runtime::Error::Internal:
        error_name = "Internal error";
        break;
      case executorch::runtime::Error::InvalidState:
        error_name = "Invalid state";
        break;
      case executorch::runtime::Error::InvalidArgument:
        error_name = "Invalid argument";
        break;
      case executorch::runtime::Error::InvalidType:
        error_name = "Invalid type";
        break;
      case executorch::runtime::Error::NotFound:
        error_name = "Not found";
        break;
      case executorch::runtime::Error::MemoryAllocationFailed:
        error_name = "Memory allocation failed";
        break;
      case executorch::runtime::Error::AccessFailed:
        error_name = "Access failed";
        break;
      case executorch::runtime::Error::NotSupported:
        error_name = "Not supported";
        break;
      case executorch::runtime::Error::DelegateInvalidCompatibility:
        error_name = "Delegate invalid compatibility";
        break;
      case executorch::runtime::Error::DelegateMemoryAllocationFailed:
        error_name = "Delegate memory allocation failed";
        break;
      case executorch::runtime::Error::DelegateInvalidHandle:
        error_name = "Delegate invalid handle";
        break;
      default:
        error_name = "Unknown error";
        break;
    }
    rb_raise(rb_eRuntimeError, "ExecuTorch error: %s", error_name);
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

#endif // EXECUTORCH_RUBY_UTILS_H

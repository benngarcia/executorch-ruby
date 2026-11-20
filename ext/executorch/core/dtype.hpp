#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/core/portable_type/half.h>

namespace etrb {

// Single source of truth for all Ruby-constructible / readable ExecuTorch dtypes.
// Tuple: (ScalarType enumerator, canonical Ruby name, C++ storage type, byte size)
#define ETRB_FORALL_DTYPES(_)                                       \
  _(Float, "float32", float, 4)                                     \
  _(Double, "float64", double, 8)                                   \
  _(Half, "float16", executorch::runtime::etensor::Half, 2)         \
  _(BFloat16, "bfloat16", executorch::runtime::etensor::BFloat16, 2) \
  _(Byte, "uint8", uint8_t, 1)                                      \
  _(Char, "int8", int8_t, 1)                                        \
  _(Short, "int16", int16_t, 2)                                     \
  _(Int, "int32", int32_t, 4)                                       \
  _(Long, "int64", int64_t, 8)                                      \
  _(UInt16, "uint16", uint16_t, 2)                                  \
  _(UInt32, "uint32", uint32_t, 4)                                  \
  _(UInt64, "uint64", uint64_t, 8)                                  \
  _(Bool, "bool", bool, 1)

inline constexpr const char* scalar_type_name(
    executorch::aten::ScalarType st) noexcept {
  switch (st) {
#define CASE_NAME(et_enum, name, ctype, sz) \
  case executorch::aten::ScalarType::et_enum: \
    return name;
    ETRB_FORALL_DTYPES(CASE_NAME)
#undef CASE_NAME
    case executorch::aten::ScalarType::QInt8:
      return "qint8";
    case executorch::aten::ScalarType::QUInt8:
      return "quint8";
    case executorch::aten::ScalarType::QInt32:
      return "qint32";
    case executorch::aten::ScalarType::ComplexHalf:
      return "complex32";
    case executorch::aten::ScalarType::ComplexFloat:
      return "complex64";
    case executorch::aten::ScalarType::ComplexDouble:
      return "complex128";
    case executorch::aten::ScalarType::Float8_e5m2:
      return "float8_e5m2";
    case executorch::aten::ScalarType::Float8_e4m3fn:
      return "float8_e4m3fn";
    default:
      return "unknown";
  }
}

inline constexpr size_t scalar_type_size(
    executorch::aten::ScalarType st) noexcept {
  switch (st) {
#define CASE_SIZE(et_enum, name, ctype, sz) \
  case executorch::aten::ScalarType::et_enum: \
    return sz;
    ETRB_FORALL_DTYPES(CASE_SIZE)
#undef CASE_SIZE
    case executorch::aten::ScalarType::QInt8:
    case executorch::aten::ScalarType::QUInt8:
    case executorch::aten::ScalarType::Float8_e5m2:
    case executorch::aten::ScalarType::Float8_e4m3fn:
      return 1;
    case executorch::aten::ScalarType::QInt32:
    case executorch::aten::ScalarType::ComplexHalf:
      return 4;
    case executorch::aten::ScalarType::ComplexFloat:
      return 8;
    case executorch::aten::ScalarType::ComplexDouble:
      return 16;
    default:
      return 0;
  }
}

inline constexpr bool is_supported_dtype(
    executorch::aten::ScalarType st) noexcept {
  switch (st) {
#define CASE_SUP(et_enum, name, ctype, sz) \
  case executorch::aten::ScalarType::et_enum: \
    return true;
    ETRB_FORALL_DTYPES(CASE_SUP)
#undef CASE_SUP
    default:
      return false;
  }
}

// Parse canonical dtype names AND backward-compatible short aliases.
inline bool parse_dtype_name(
    std::string_view name,
    executorch::aten::ScalarType* out) noexcept {
  if (name == "float32" || name == "float") {
    *out = executorch::aten::ScalarType::Float;
    return true;
  }
  if (name == "float64" || name == "double") {
    *out = executorch::aten::ScalarType::Double;
    return true;
  }
  if (name == "float16" || name == "half") {
    *out = executorch::aten::ScalarType::Half;
    return true;
  }
  if (name == "bfloat16" || name == "bf16") {
    *out = executorch::aten::ScalarType::BFloat16;
    return true;
  }
  if (name == "uint8" || name == "byte") {
    *out = executorch::aten::ScalarType::Byte;
    return true;
  }
  if (name == "int8" || name == "char") {
    *out = executorch::aten::ScalarType::Char;
    return true;
  }
  if (name == "int16" || name == "short") {
    *out = executorch::aten::ScalarType::Short;
    return true;
  }
  if (name == "int32" || name == "int") {
    *out = executorch::aten::ScalarType::Int;
    return true;
  }
  if (name == "int64" || name == "long") {
    *out = executorch::aten::ScalarType::Long;
    return true;
  }
  if (name == "uint16") {
    *out = executorch::aten::ScalarType::UInt16;
    return true;
  }
  if (name == "uint32") {
    *out = executorch::aten::ScalarType::UInt32;
    return true;
  }
  if (name == "uint64") {
    *out = executorch::aten::ScalarType::UInt64;
    return true;
  }
  if (name == "bool") {
    *out = executorch::aten::ScalarType::Bool;
    return true;
  }
  return false;
}

}  // namespace etrb

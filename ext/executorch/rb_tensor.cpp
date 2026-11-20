#include "rb_internal.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace etrb {

namespace {

void tensor_mark(void* p) {
  auto* wrap = static_cast<RbTensorWrap*>(p);
  if (wrap->borrowed_buf != Qnil) {
    rb_gc_mark(wrap->borrowed_buf);
  }
}

void tensor_free(void* p) {
  auto* wrap = static_cast<RbTensorWrap*>(p);
  if (wrap->tracked_bytes > 0) {
    rb_gc_adjust_memory_usage(-static_cast<ssize_t>(wrap->tracked_bytes));
    wrap->tracked_bytes = 0;
  }
  delete wrap;
}

size_t tensor_memsize(const void* p) {
  const auto* wrap = static_cast<const RbTensorWrap*>(p);
  return sizeof(RbTensorWrap) + wrap->tracked_bytes;
}

VALUE tensor_alloc(VALUE klass) {
  auto* wrap = new (std::nothrow) RbTensorWrap();
  if (wrap == nullptr) {
    rb_memerror();
  }
  return TypedData_Wrap_Struct(klass, &kTensorDataType, wrap);
}

RbTensorWrap* check_open_tensor(VALUE self) {
  RbTensorWrap* wrap = nullptr;
  TypedData_Get_Struct(self, RbTensorWrap, &kTensorDataType, wrap);
  if (wrap == nullptr || wrap->closed ||
      (!wrap->tensor && wrap->borrowed_buf == Qnil)) {
    rb_raise(g_refs.eClosedError, "Executorch::Tensor is closed");
  }
  return wrap;
}

// Validate & extract a borrowed IO::Buffer's bytes right now under the GVL.
void check_borrowed_read(
    const RbTensorWrap* wrap,
    const void** out_ptr,
    size_t* out_len) {
  const void* ptr = nullptr;
  size_t len = 0;
  rb_io_buffer_get_bytes_for_reading(wrap->borrowed_buf, &ptr, &len);
  if (len != wrap->borrowed_nbytes) {
    rb_raise(
        g_refs.eShapeError,
        "IO::Buffer size changed: expected %zu bytes, got %zu",
        wrap->borrowed_nbytes,
        len);
  }
  *out_ptr = ptr;
  *out_len = len;
}

// Parse Ruby Array of dims into int64_t vector with reentrancy-safe length reads.
void parse_ruby_shape(VALUE shape_ary, std::vector<int64_t>* out) {
  Check_Type(shape_ary, T_ARRAY);
  long len = RARRAY_LEN(shape_ary);
  out->reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    if (i >= RARRAY_LEN(shape_ary)) {
      rb_raise(rb_eArgError, "shape array was mutated during conversion");
    }
    VALUE v = rb_ary_entry(shape_ary, i);
    if (!RB_INTEGER_TYPE_P(v)) {
      rb_raise(rb_eTypeError, "shape dimensions must be Integer");
    }
    out->push_back(NUM2LL(v));
  }
}

// Safe scalar extraction from Ruby VALUE to target C++ dtype.
template <typename T>
T ruby_to_elem(VALUE v);

template <>
float ruby_to_elem<float>(VALUE v) {
  if (RB_FLOAT_TYPE_P(v)) return static_cast<float>(RFLOAT_VALUE(v));
  if (FIXNUM_P(v)) return static_cast<float>(FIX2LONG(v));
  return static_cast<float>(NUM2DBL(v));
}

template <>
double ruby_to_elem<double>(VALUE v) {
  if (RB_FLOAT_TYPE_P(v)) return RFLOAT_VALUE(v);
  if (FIXNUM_P(v)) return static_cast<double>(FIX2LONG(v));
  return NUM2DBL(v);
}

template <>
executorch::runtime::etensor::Half ruby_to_elem<executorch::runtime::etensor::Half>(VALUE v) {
  return executorch::runtime::etensor::Half(ruby_to_elem<float>(v));
}

template <>
executorch::runtime::etensor::BFloat16 ruby_to_elem<executorch::runtime::etensor::BFloat16>(VALUE v) {
  return executorch::runtime::etensor::BFloat16(ruby_to_elem<float>(v));
}

template <>
bool ruby_to_elem<bool>(VALUE v) {
  if (v == Qtrue) return true;
  if (v == Qfalse) return false;
  if (FIXNUM_P(v)) return FIX2LONG(v) != 0;
  rb_raise(rb_eTypeError, "expected true, false, or Integer for :bool tensor");
}

template <typename IntT>
IntT ruby_to_signed_int(VALUE v) {
  long long val = 0;
  if (FIXNUM_P(v)) {
    val = static_cast<long long>(FIX2LONG(v));
  } else {
    if (RB_FLOAT_TYPE_P(v)) {
      double d = RFLOAT_VALUE(v);
      if (std::isnan(d) || std::isinf(d)) {
        rb_raise(rb_eFloatDomainError, "non-finite float cannot convert to integer");
      }
    }
    val = NUM2LL(v);
  }
  constexpr long long lo = static_cast<long long>(std::numeric_limits<IntT>::min());
  constexpr long long hi = static_cast<long long>(std::numeric_limits<IntT>::max());
  if (val < lo || val > hi) {
    rb_raise(
        rb_eRangeError,
        "integer %lld out of range for %zu-bit signed dtype",
        val,
        sizeof(IntT) * 8);
  }
  return static_cast<IntT>(val);
}

template <typename UIntT>
UIntT ruby_to_unsigned_int(VALUE v) {
  unsigned long long val = 0;
  if (FIXNUM_P(v)) {
    long lv = FIX2LONG(v);
    if (lv < 0) {
      rb_raise(rb_eRangeError, "negative value %ld out of range for unsigned dtype", lv);
    }
    val = static_cast<unsigned long long>(lv);
  } else {
    if (RB_FLOAT_TYPE_P(v)) {
      double d = RFLOAT_VALUE(v);
      if (std::isnan(d) || std::isinf(d)) {
        rb_raise(rb_eFloatDomainError, "non-finite float cannot convert to unsigned integer");
      }
      if (d < 0.0) {
        rb_raise(rb_eRangeError, "negative float out of range for unsigned dtype");
      }
    }
    val = NUM2ULL(v);
  }
  constexpr unsigned long long hi =
      static_cast<unsigned long long>(std::numeric_limits<UIntT>::max());
  if (val > hi) {
    rb_raise(
        rb_eRangeError,
        "integer %llu out of range for %zu-bit unsigned dtype",
        val,
        sizeof(UIntT) * 8);
  }
  return static_cast<UIntT>(val);
}

template <> int8_t ruby_to_elem<int8_t>(VALUE v) { return ruby_to_signed_int<int8_t>(v); }
template <> int16_t ruby_to_elem<int16_t>(VALUE v) { return ruby_to_signed_int<int16_t>(v); }
template <> int32_t ruby_to_elem<int32_t>(VALUE v) { return ruby_to_signed_int<int32_t>(v); }
template <> int64_t ruby_to_elem<int64_t>(VALUE v) { return ruby_to_signed_int<int64_t>(v); }
template <> uint8_t ruby_to_elem<uint8_t>(VALUE v) { return ruby_to_unsigned_int<uint8_t>(v); }
template <> uint16_t ruby_to_elem<uint16_t>(VALUE v) { return ruby_to_unsigned_int<uint16_t>(v); }
template <> uint32_t ruby_to_elem<uint32_t>(VALUE v) { return ruby_to_unsigned_int<uint32_t>(v); }
template <> uint64_t ruby_to_elem<uint64_t>(VALUE v) { return ruby_to_unsigned_int<uint64_t>(v); }

// Convert C++ element to Ruby VALUE.
template <typename T>
VALUE elem_to_ruby(T v);

template <> VALUE elem_to_ruby<float>(float v) { return DBL2NUM(static_cast<double>(v)); }
template <> VALUE elem_to_ruby<double>(double v) { return DBL2NUM(v); }
template <> VALUE elem_to_ruby<executorch::runtime::etensor::Half>(executorch::runtime::etensor::Half v) {
  return DBL2NUM(static_cast<double>(static_cast<float>(v)));
}
template <> VALUE elem_to_ruby<executorch::runtime::etensor::BFloat16>(executorch::runtime::etensor::BFloat16 v) {
  return DBL2NUM(static_cast<double>(static_cast<float>(v)));
}
template <> VALUE elem_to_ruby<bool>(bool v) { return v ? Qtrue : Qfalse; }
template <> VALUE elem_to_ruby<int8_t>(int8_t v) { return INT2FIX(v); }
template <> VALUE elem_to_ruby<int16_t>(int16_t v) { return INT2FIX(v); }
template <> VALUE elem_to_ruby<int32_t>(int32_t v) { return LONG2NUM(v); }
template <> VALUE elem_to_ruby<int64_t>(int64_t v) { return LL2NUM(v); }
template <> VALUE elem_to_ruby<uint8_t>(uint8_t v) { return INT2FIX(v); }
template <> VALUE elem_to_ruby<uint16_t>(uint16_t v) { return INT2FIX(v); }
template <> VALUE elem_to_ruby<uint32_t>(uint32_t v) { return ULONG2NUM(v); }
template <> VALUE elem_to_ruby<uint64_t>(uint64_t v) { return ULL2NUM(v); }

// Exact nested-array shape inference AND full structural validation (no silent
// data loss on [[], [1, 2]] or [[], "x"], no bypass on [[1], [[2]]]).
// Does NOT allocate any C++ objects that could leak if rb_raise runs because
// we only call rb_raise AFTER returning an error code from pure C++ traversal!
int infer_and_validate_nested_c(
    VALUE node,
    size_t depth,
    std::vector<int64_t>& shape,
    bool& shape_locked,
    std::string& err_msg) {
  if (!RB_TYPE_P(node, T_ARRAY)) {
    if (!shape_locked) {
      shape_locked = true;
    } else if (depth != shape.size()) {
      err_msg = "inconsistent nesting depth at level " + std::to_string(depth) +
                ": expected Array";
      return -1;
    }
    return 0;
  }

  const long n = RARRAY_LEN(node);
  if (!shape_locked && depth == shape.size()) {
    shape.push_back(static_cast<int64_t>(n));
  } else if (depth >= shape.size()) {
    err_msg = "unexpected nested Array at leaf depth " + std::to_string(depth);
    return -1;
  } else if (shape[depth] != static_cast<int64_t>(n)) {
    err_msg = "jagged array at depth " + std::to_string(depth) +
              ": expected size " + std::to_string(shape[depth]) + ", got " +
              std::to_string(n);
    return -1;
  }

  if (n == 0) {
    // An empty sub-array at depth d means every sibling at depth d MUST also be
    // an empty Array (size 0), and no deeper dimension exists below it.
    if (!shape_locked) {
      shape_locked = true;
    } else if (depth + 1 != shape.size()) {
      err_msg = "jagged empty array at depth " + std::to_string(depth);
      return -1;
    }
    return 0;
  }

  for (long i = 0; i < n; ++i) {
    VALUE child = rb_ary_entry(node, i);
    if (infer_and_validate_nested_c(child, depth + 1, shape, shape_locked, err_msg) != 0) {
      return -1;
    }
  }
  return 0;
}

// Reentrancy-safe single-pass write from nested or flat Ruby Array into raw
// typed buffer. Re-checks RARRAY_LEN on EVERY iteration in case a custom #to_f
// or #to_int mutates the Ruby Array!
// Protected wrapper ensures C++ locals are never leaked across rb_raise.
template <typename T>
struct FillCtx {
  VALUE root;
  T* dst;
  size_t cap;
  size_t pos;
  size_t target_depth;
};

template <typename T>
VALUE fill_nested_body(VALUE arg) {
  auto* ctx = reinterpret_cast<FillCtx<T>*>(arg);

  auto walk = [&](auto& self, VALUE node, size_t depth) -> void {
    if (depth == ctx->target_depth) {
      if (RB_TYPE_P(node, T_ARRAY)) {
        rb_raise(rb_eArgError, "unexpected Array at leaf level during fill");
      }
      if (ctx->pos >= ctx->cap) {
        rb_raise(rb_eArgError, "array grew during conversion");
      }
      T val = ruby_to_elem<T>(node);
      ctx->dst[ctx->pos++] = val;
      return;
    }
    if (!RB_TYPE_P(node, T_ARRAY)) {
      rb_raise(rb_eArgError, "array mutated to non-Array during conversion");
    }
    long len = RARRAY_LEN(node);
    for (long i = 0; i < len; ++i) {
      // Re-check length on every step: prevents OOB read if a custom coercion
      // shrank `node` via `node.clear` or `node.pop`.
      if (i >= RARRAY_LEN(node)) {
        rb_raise(rb_eArgError, "array was mutated during conversion");
      }
      VALUE elem = rb_ary_entry(node, i);
      self(self, elem, depth + 1);
    }
  };

  walk(walk, ctx->root, 0);
  if (ctx->pos != ctx->cap) {
    rb_raise(
        rb_eArgError,
        "data length (%zu) does not match shape element count (%zu)",
        ctx->pos,
        ctx->cap);
  }
  return Qnil;
}

template <typename T>
VALUE fill_flat_body(VALUE arg) {
  auto* ctx = reinterpret_cast<FillCtx<T>*>(arg);
  if (!RB_TYPE_P(ctx->root, T_ARRAY)) {
    if (ctx->cap != 1) {
      rb_raise(rb_eArgError, "scalar data requires shape [] or [1]");
    }
    ctx->dst[0] = ruby_to_elem<T>(ctx->root);
    ctx->pos = 1;
    return Qnil;
  }
  long initial_len = RARRAY_LEN(ctx->root);
  if (static_cast<size_t>(initial_len) != ctx->cap) {
    rb_raise(
        g_refs.eShapeError,
        "data length (%ld) does not match shape element count (%zu)",
        initial_len,
        ctx->cap);
  }
  for (long i = 0; i < initial_len; ++i) {
    if (i >= RARRAY_LEN(ctx->root)) {
      rb_raise(rb_eArgError, "array was mutated during conversion");
    }
    VALUE elem = rb_ary_entry(ctx->root, i);
    ctx->dst[i] = ruby_to_elem<T>(elem);
  }
  ctx->pos = static_cast<size_t>(initial_len);
  return Qnil;
}

// Dispatch typed fill inside rb_protect so AlignedBuffer is ALWAYS freed if
// Ruby raises TypeError/RangeError/FloatDomainError during element conversion!
int fill_buffer_protected(
    VALUE data,
    executorch::aten::ScalarType dtype,
    void* dst,
    size_t numel,
    bool nested,
    size_t target_depth) {
  int state = 0;
  switch (dtype) {
#define CASE_FILL(et_enum, name, ctype, sz)                                   \
  case executorch::aten::ScalarType::et_enum: {                               \
    FillCtx<ctype> ctx{data, static_cast<ctype*>(dst), numel, 0, target_depth}; \
    rb_protect(                                                               \
        nested ? fill_nested_body<ctype> : fill_flat_body<ctype>,             \
        reinterpret_cast<VALUE>(&ctx),                                        \
        &state);                                                              \
    break;                                                                    \
  }
    ETRB_FORALL_DTYPES(CASE_FILL)
#undef CASE_FILL
    default:
      state = -1;
      break;
  }
  return state;
}

// Recursively build nested Ruby Array directly from contiguous raw memory
// (zero intermediate flat array allocation, zero [result, offset] pairs!).
template <typename T>
VALUE build_nested_ruby(
    const T* base,
    const executorch::aten::SizesType* sizes,
    size_t ndim,
    size_t dim,
    size_t& offset) {
  const long count = static_cast<long>(sizes[dim]);
  VALUE ary = rb_ary_new_capa(count);
  if (dim + 1 == ndim) {
    for (long i = 0; i < count; ++i) {
      rb_ary_push(ary, elem_to_ruby<T>(base[offset++]));
    }
  } else {
    for (long i = 0; i < count; ++i) {
      rb_ary_push(ary, build_nested_ruby<T>(base, sizes, ndim, dim + 1, offset));
    }
  }
  return ary;
}

VALUE build_ruby_from_contiguous(
    const void* raw,
    executorch::aten::ScalarType dtype,
    const executorch::aten::SizesType* sizes,
    size_t ndim,
    size_t numel,
    bool flat) {
  if (ndim == 0) {
    if (numel != 1) {
      return Qnil;
    }
    switch (dtype) {
#define CASE_SCALAR(et_enum, name, ctype, sz) \
  case executorch::aten::ScalarType::et_enum: \
    return elem_to_ruby<ctype>(*static_cast<const ctype*>(raw));
      ETRB_FORALL_DTYPES(CASE_SCALAR)
#undef CASE_SCALAR
      default:
        rb_raise(g_refs.eDTypeError, "unsupported dtype for scalar extraction");
    }
  }

  if (flat || ndim == 1) {
    VALUE ary = rb_ary_new_capa(static_cast<long>(numel));
    switch (dtype) {
#define CASE_FLAT(et_enum, name, ctype, sz)                                \
  case executorch::aten::ScalarType::et_enum: {                            \
    const auto* p = static_cast<const ctype*>(raw);                        \
    for (size_t i = 0; i < numel; ++i) {                                   \
      rb_ary_push(ary, elem_to_ruby<ctype>(p[i]));                         \
    }                                                                      \
    return ary;                                                            \
  }
      ETRB_FORALL_DTYPES(CASE_FLAT)
#undef CASE_FLAT
      default:
        rb_raise(g_refs.eDTypeError, "unsupported dtype for to_a");
    }
  }

  size_t offset = 0;
  switch (dtype) {
#define CASE_NESTED(et_enum, name, ctype, sz)                           \
  case executorch::aten::ScalarType::et_enum:                           \
    return build_nested_ruby<ctype>(                                    \
        static_cast<const ctype*>(raw), sizes, ndim, 0, offset);
    ETRB_FORALL_DTYPES(CASE_NESTED)
#undef CASE_NESTED
    default:
      rb_raise(g_refs.eDTypeError, "unsupported dtype for to_a");
  }
}

// Helper: parse dtype option (default :float32).
executorch::aten::ScalarType parse_dtype_opt(VALUE v) {
  if (v == Qundef || v == Qnil) {
    return executorch::aten::ScalarType::Float;
  }
  executorch::aten::ScalarType st = executorch::aten::ScalarType::Undefined;
  if (!sym_to_scalar_type(v, &st)) {
    if (!RB_SYMBOL_P(v)) {
      rb_raise(rb_eTypeError, "dtype must be a Symbol");
    }
    rb_raise(g_refs.eDTypeError, "unknown or unsupported dtype: :%s", rb_id2name(SYM2ID(v)));
  }
  return st;
}

// ---------------------------------------------------------------------------
// Ruby methods for Executorch::Tensor
// ---------------------------------------------------------------------------

// Tensor.new(data, shape: nil, dtype: :float32)
VALUE rb_tensor_s_new(int argc, VALUE* argv, VALUE klass) {
  VALUE data = Qnil;
  VALUE opts = Qnil;
  rb_scan_args(argc, argv, "1:", &data, &opts);

  VALUE kw_vals[2] = {Qundef, Qundef};
  ID kw_ids[2] = {g_refs.id_kw_shape, g_refs.id_kw_dtype};
  if (!NIL_P(opts)) {
    rb_get_kwargs(opts, kw_ids, 0, 2, kw_vals);
  }
  VALUE shape_val = kw_vals[0];
  VALUE dtype_val = kw_vals[1];

  const auto dtype = parse_dtype_opt(dtype_val);
  const bool explicit_shape = (shape_val != Qundef && shape_val != Qnil);

  VALUE err_exc = Qnil;
  VALUE result = Qnil;
  int protect_state = 0;

  {
    std::vector<int64_t> raw_dims;
    bool nested = false;
    size_t target_depth = 0;

    if (explicit_shape) {
      if (!RB_TYPE_P(shape_val, T_ARRAY)) {
        err_exc = rb_exc_new_cstr(rb_eTypeError, "shape must be an Array of Integers");
      } else {
        // Reentrancy-safe shape extraction inside rb_protect.
        struct ShapeArg {
          VALUE ary;
          std::vector<int64_t>* dims;
        } sa{shape_val, &raw_dims};
        rb_protect(
            [](VALUE a) -> VALUE {
              auto* s = reinterpret_cast<ShapeArg*>(a);
              parse_ruby_shape(s->ary, s->dims);
              return Qnil;
            },
            reinterpret_cast<VALUE>(&sa),
            &protect_state);
      }
    } else {
      if (RB_TYPE_P(data, T_ARRAY)) {
        bool locked = false;
        std::string err_msg;
        if (infer_and_validate_nested_c(data, 0, raw_dims, locked, err_msg) != 0) {
          err_exc = rb_exc_new_str(
              rb_eArgError,
              rb_utf8_str_new(err_msg.data(), static_cast<long>(err_msg.size())));
        } else {
          nested = true;
          target_depth = raw_dims.size();
        }
      } else if (RB_INTEGER_TYPE_P(data) || RB_FLOAT_TYPE_P(data) ||
                 data == Qtrue || data == Qfalse ||
                 rb_respond_to(data, rb_intern("to_f"))) {
        // 0-D scalar tensor!
        raw_dims.clear();
        nested = false;
      } else {
        err_exc = rb_exc_new_cstr(
            rb_eTypeError, "Tensor data must be an Array or scalar Numeric/Boolean");
      }
    }

    if (protect_state == 0 && err_exc == Qnil) {
      auto shape_res = validate_shape(raw_dims, dtype);
      if (!shape_res.ok()) {
        err_exc = status_to_ruby_exception(shape_res.status);
      } else {
        auto buf_res = AlignedBuffer::allocate(shape_res.value.nbytes);
        if (!buf_res.ok()) {
          err_exc = status_to_ruby_exception(buf_res.status);
        } else {
          // If explicit shape was passed with a nested array, validate & flatten cleanly!
          if (explicit_shape && RB_TYPE_P(data, T_ARRAY) &&
              RARRAY_LEN(data) > 0 && RB_TYPE_P(rb_ary_entry(data, 0), T_ARRAY)) {
            std::vector<int64_t> inferred;
            bool locked = false;
            std::string err_msg;
            if (infer_and_validate_nested_c(data, 0, inferred, locked, err_msg) != 0) {
              err_exc = rb_exc_new_str(
                  rb_eArgError,
                  rb_utf8_str_new(err_msg.data(), static_cast<long>(err_msg.size())));
            } else {
              nested = true;
              target_depth = inferred.size();
            }
          }

          if (err_exc == Qnil) {
            protect_state = fill_buffer_protected(
                data,
                dtype,
                buf_res.value.ptr,
                shape_res.value.numel,
                nested,
                target_depth);
          }

          if (protect_state == 0 && err_exc == Qnil) {
            auto tp_res = make_owned_tensor_ptr(
                std::move(shape_res.value.sizes), dtype, std::move(buf_res.value));
            if (!tp_res.ok()) {
              err_exc = status_to_ruby_exception(tp_res.status);
            } else {
              result = wrap_owned_tensor(std::move(tp_res.value));
              (void)klass;
            }
          }
        }
      }
    }
  }  // All C++ locals destroyed here!

  if (protect_state != 0) {
    rb_jump_tag(protect_state);
  }
  if (err_exc != Qnil) {
    raise_status_no_live_cpp(err_exc);
  }
  return result;
}

// Tensor.from_bytes(str, shape:, dtype: :float32)
VALUE rb_tensor_s_from_bytes(int argc, VALUE* argv, VALUE klass) {
  (void)klass;
  VALUE bytes_val = Qnil;
  VALUE opts = Qnil;
  rb_scan_args(argc, argv, "1:", &bytes_val, &opts);
  Check_Type(bytes_val, T_STRING);

  VALUE kw_vals[2] = {Qundef, Qundef};
  ID kw_ids[2] = {g_refs.id_kw_shape, g_refs.id_kw_dtype};
  if (NIL_P(opts)) {
    rb_raise(rb_eArgError, "missing keyword: :shape");
  }
  rb_get_kwargs(opts, kw_ids, 1, 1, kw_vals);

  VALUE shape_val = kw_vals[0];
  const auto dtype = parse_dtype_opt(kw_vals[1]);
  Check_Type(shape_val, T_ARRAY);

  VALUE err_exc = Qnil;
  VALUE result = Qnil;
  int protect_state = 0;

  {
    std::vector<int64_t> raw_dims;
    struct ShapeArg {
      VALUE ary;
      std::vector<int64_t>* dims;
    } sa{shape_val, &raw_dims};
    rb_protect(
        [](VALUE a) -> VALUE {
          auto* s = reinterpret_cast<ShapeArg*>(a);
          parse_ruby_shape(s->ary, s->dims);
          return Qnil;
        },
        reinterpret_cast<VALUE>(&sa),
        &protect_state);

    if (protect_state == 0) {
      auto shape_res = validate_shape(raw_dims, dtype);
      if (!shape_res.ok()) {
        err_exc = status_to_ruby_exception(shape_res.status);
      } else {
        const size_t actual = static_cast<size_t>(RSTRING_LEN(bytes_val));
        const size_t expected = shape_res.value.nbytes;
        if (actual != expected) {
          std::string msg = "byte length mismatch: expected " +
                            std::to_string(expected) + " bytes, got " +
                            std::to_string(actual);
          err_exc = rb_exc_new_str(
              g_refs.eShapeError,
              rb_utf8_str_new(msg.data(), static_cast<long>(msg.size())));
        } else {
          auto buf_res = AlignedBuffer::allocate(expected);
          if (!buf_res.ok()) {
            err_exc = status_to_ruby_exception(buf_res.status);
          } else {
            if (expected > 0) {
              std::memcpy(buf_res.value.ptr, RSTRING_PTR(bytes_val), expected);
            }
            auto tp_res = make_owned_tensor_ptr(
                std::move(shape_res.value.sizes), dtype, std::move(buf_res.value));
            if (!tp_res.ok()) {
              err_exc = status_to_ruby_exception(tp_res.status);
            } else {
              result = wrap_owned_tensor(std::move(tp_res.value));
            }
          }
        }
      }
    }
  }  // All C++ locals destroyed here!

  RB_GC_GUARD(bytes_val);
  if (protect_state != 0) {
    rb_jump_tag(protect_state);
  }
  if (err_exc != Qnil) {
    raise_status_no_live_cpp(err_exc);
  }
  return result;
}

// Tensor.from_buffer(io_buffer, shape:, dtype: :float32) — zero-copy view!
VALUE rb_tensor_s_from_buffer(int argc, VALUE* argv, VALUE klass) {
  VALUE buf_val = Qnil;
  VALUE opts = Qnil;
  rb_scan_args(argc, argv, "1:", &buf_val, &opts);

  if (!rb_obj_is_kind_of(buf_val, rb_cIOBuffer)) {
    rb_raise(rb_eTypeError, "expected an IO::Buffer");
  }
  if (NIL_P(opts)) {
    rb_raise(rb_eArgError, "missing keyword: :shape");
  }
  VALUE kw_vals[2] = {Qundef, Qundef};
  ID kw_ids[2] = {g_refs.id_kw_shape, g_refs.id_kw_dtype};
  rb_get_kwargs(opts, kw_ids, 1, 1, kw_vals);

  VALUE shape_val = kw_vals[0];
  const auto dtype = parse_dtype_opt(kw_vals[1]);
  Check_Type(shape_val, T_ARRAY);

  const void* ptr = nullptr;
  size_t actual = 0;
  rb_io_buffer_get_bytes_for_reading(buf_val, &ptr, &actual);

  VALUE err_exc = Qnil;
  VALUE result = Qnil;
  int protect_state = 0;

  {
    std::vector<int64_t> raw_dims;
    struct ShapeArg {
      VALUE ary;
      std::vector<int64_t>* dims;
    } sa{shape_val, &raw_dims};
    rb_protect(
        [](VALUE a) -> VALUE {
          auto* s = reinterpret_cast<ShapeArg*>(a);
          parse_ruby_shape(s->ary, s->dims);
          return Qnil;
        },
        reinterpret_cast<VALUE>(&sa),
        &protect_state);

    if (protect_state == 0) {
      auto shape_res = validate_shape(raw_dims, dtype);
      if (!shape_res.ok()) {
        err_exc = status_to_ruby_exception(shape_res.status);
      } else if (actual != shape_res.value.nbytes) {
        std::string msg = "IO::Buffer byte length mismatch: expected " +
                          std::to_string(shape_res.value.nbytes) +
                          " bytes, got " + std::to_string(actual);
        err_exc = rb_exc_new_str(
            g_refs.eShapeError,
            rb_utf8_str_new(msg.data(), static_cast<long>(msg.size())));
      } else {
        result = tensor_alloc(klass);
        RbTensorWrap* wrap = nullptr;
        TypedData_Get_Struct(result, RbTensorWrap, &kTensorDataType, wrap);
        wrap->borrowed_buf = buf_val;
        wrap->borrowed_sizes = std::move(shape_res.value.sizes);
        wrap->borrowed_dtype = dtype;
        wrap->borrowed_nbytes = shape_res.value.nbytes;
      }
    }
  }

  if (protect_state != 0) {
    rb_jump_tag(protect_state);
  }
  if (err_exc != Qnil) {
    raise_status_no_live_cpp(err_exc);
  }
  return result;
}

// Tensor.zeros(shape, dtype: :float32)
VALUE rb_tensor_s_zeros(int argc, VALUE* argv, VALUE klass) {
  (void)klass;
  VALUE shape_val = Qnil;
  VALUE opts = Qnil;
  rb_scan_args(argc, argv, "1:", &shape_val, &opts);
  Check_Type(shape_val, T_ARRAY);

  VALUE kw_vals[1] = {Qundef};
  ID kw_ids[1] = {g_refs.id_kw_dtype};
  if (!NIL_P(opts)) {
    rb_get_kwargs(opts, kw_ids, 0, 1, kw_vals);
  }
  const auto dtype = parse_dtype_opt(kw_vals[0]);

  VALUE err_exc = Qnil;
  VALUE result = Qnil;
  int protect_state = 0;

  {
    std::vector<int64_t> raw_dims;
    struct ShapeArg {
      VALUE ary;
      std::vector<int64_t>* dims;
    } sa{shape_val, &raw_dims};
    rb_protect(
        [](VALUE a) -> VALUE {
          auto* s = reinterpret_cast<ShapeArg*>(a);
          parse_ruby_shape(s->ary, s->dims);
          return Qnil;
        },
        reinterpret_cast<VALUE>(&sa),
        &protect_state);

    if (protect_state == 0) {
      auto shape_res = validate_shape(raw_dims, dtype);
      if (!shape_res.ok()) {
        err_exc = status_to_ruby_exception(shape_res.status);
      } else {
        auto buf_res = AlignedBuffer::allocate_zeroed(shape_res.value.nbytes);
        if (!buf_res.ok()) {
          err_exc = status_to_ruby_exception(buf_res.status);
        } else {
          auto tp_res = make_owned_tensor_ptr(
              std::move(shape_res.value.sizes), dtype, std::move(buf_res.value));
          if (!tp_res.ok()) {
            err_exc = status_to_ruby_exception(tp_res.status);
          } else {
            result = wrap_owned_tensor(std::move(tp_res.value));
          }
        }
      }
    }
  }

  if (protect_state != 0) {
    rb_jump_tag(protect_state);
  }
  if (err_exc != Qnil) {
    raise_status_no_live_cpp(err_exc);
  }
  return result;
}

VALUE rb_tensor_shape(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    VALUE ary = rb_ary_new_capa(static_cast<long>(wrap->borrowed_sizes.size()));
    for (auto d : wrap->borrowed_sizes) {
      rb_ary_push(ary, LONG2NUM(d));
    }
    return ary;
  }
  const auto sizes = wrap->tensor->sizes();
  VALUE ary = rb_ary_new_capa(static_cast<long>(sizes.size()));
  for (auto d : sizes) {
    rb_ary_push(ary, LONG2NUM(d));
  }
  return ary;
}

VALUE rb_tensor_strides(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    const size_t ndim = wrap->borrowed_sizes.size();
    VALUE ary = rb_ary_new_capa(static_cast<long>(ndim));
    if (ndim == 0) return ary;
    std::vector<int64_t> st(ndim, 1);
    for (size_t i = ndim - 1; i > 0; --i) {
      st[i - 1] = st[i] * static_cast<int64_t>(wrap->borrowed_sizes[i]);
    }
    for (auto s : st) {
      rb_ary_push(ary, LL2NUM(s));
    }
    return ary;
  }
  const auto strides = wrap->tensor->strides();
  VALUE ary = rb_ary_new_capa(static_cast<long>(strides.size()));
  for (auto s : strides) {
    rb_ary_push(ary, LONG2NUM(s));
  }
  return ary;
}

VALUE rb_tensor_dim_order(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    const size_t ndim = wrap->borrowed_sizes.size();
    VALUE ary = rb_ary_new_capa(static_cast<long>(ndim));
    for (size_t i = 0; i < ndim; ++i) {
      rb_ary_push(ary, ULONG2NUM(i));
    }
    return ary;
  }
  const auto order = wrap->tensor->dim_order();
  VALUE ary = rb_ary_new_capa(static_cast<long>(order.size()));
  for (auto o : order) {
    rb_ary_push(ary, INT2FIX(o));
  }
  return ary;
}

VALUE rb_tensor_contiguous_p(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    return Qtrue;
  }
  return is_standard_contiguous(*wrap->tensor) ? Qtrue : Qfalse;
}

VALUE rb_tensor_dim(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    return LONG2NUM(static_cast<long>(wrap->borrowed_sizes.size()));
  }
  return LONG2NUM(wrap->tensor->dim());
}

VALUE rb_tensor_numel(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    size_t item_sz = scalar_type_size(wrap->borrowed_dtype);
    return ULL2NUM(item_sz == 0 ? 0 : wrap->borrowed_nbytes / item_sz);
  }
  return LL2NUM(wrap->tensor->numel());
}

VALUE rb_tensor_nbytes(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    return ULL2NUM(wrap->borrowed_nbytes);
  }
  return ULL2NUM(wrap->tensor->nbytes());
}

VALUE rb_tensor_dtype(VALUE self) {
  auto* wrap = check_open_tensor(self);
  auto st = (wrap->borrowed_buf != Qnil) ? wrap->borrowed_dtype
                                         : wrap->tensor->scalar_type();
  return scalar_type_to_sym(st);
}

VALUE rb_tensor_to_bytes(VALUE self) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    const void* ptr = nullptr;
    size_t len = 0;
    check_borrowed_read(wrap, &ptr, &len);
    VALUE str = rb_str_buf_new(static_cast<long>(len));
    if (len > 0) {
      rb_str_cat(str, static_cast<const char*>(ptr), static_cast<long>(len));
    }
    RB_GC_GUARD(self);
    return str;
  }

  const auto& t = *wrap->tensor;
  const size_t nbytes = t.nbytes();
  if (nbytes > static_cast<size_t>(std::numeric_limits<long>::max())) {
    rb_raise(g_refs.eMemoryError, "tensor too large for Ruby String");
  }
  VALUE str = rb_str_new(nullptr, static_cast<long>(nbytes));
  VALUE err_exc = Qnil;
  {
    auto st = copy_to_contiguous_bytes(t, RSTRING_PTR(str), nbytes);
    if (!st.ok()) {
      err_exc = status_to_ruby_exception(st);
    }
  }
  RB_GC_GUARD(self);
  if (err_exc != Qnil) {
    raise_status_no_live_cpp(err_exc);
  }
  return str;
}

VALUE rb_tensor_to_buffer(VALUE self) {
  auto* wrap = check_open_tensor(self);
  size_t nbytes = 0;
  if (wrap->borrowed_buf != Qnil) {
    const void* src = nullptr;
    check_borrowed_read(wrap, &src, &nbytes);
    VALUE io_buf = rb_io_buffer_new(nullptr, nbytes, RB_IO_BUFFER_INTERNAL);
    if (nbytes > 0) {
      void* dst = nullptr;
      size_t dst_len = 0;
      rb_io_buffer_get_bytes_for_writing(io_buf, &dst, &dst_len);
      std::memcpy(dst, src, nbytes);
    }
    RB_GC_GUARD(self);
    return io_buf;
  }

  const auto& t = *wrap->tensor;
  nbytes = t.nbytes();
  VALUE io_buf = rb_io_buffer_new(nullptr, nbytes, RB_IO_BUFFER_INTERNAL);
  VALUE err_exc = Qnil;
  if (nbytes > 0) {
    void* dst = nullptr;
    size_t dst_len = 0;
    rb_io_buffer_get_bytes_for_writing(io_buf, &dst, &dst_len);
    auto st = copy_to_contiguous_bytes(t, dst, nbytes);
    if (!st.ok()) {
      err_exc = status_to_ruby_exception(st);
    }
  }
  RB_GC_GUARD(self);
  if (err_exc != Qnil) {
    raise_status_no_live_cpp(err_exc);
  }
  return io_buf;
}

static VALUE tensor_to_a_impl(VALUE self, bool flat) {
  auto* wrap = check_open_tensor(self);
  if (wrap->borrowed_buf != Qnil) {
    const void* ptr = nullptr;
    size_t len = 0;
    check_borrowed_read(wrap, &ptr, &len);
    const size_t item_sz = scalar_type_size(wrap->borrowed_dtype);
    const size_t numel = item_sz == 0 ? 0 : len / item_sz;
    VALUE out = build_ruby_from_contiguous(
        ptr,
        wrap->borrowed_dtype,
        wrap->borrowed_sizes.data(),
        wrap->borrowed_sizes.size(),
        numel,
        flat);
    RB_GC_GUARD(self);
    return out;
  }

  const auto& t = *wrap->tensor;
  const size_t numel = static_cast<size_t>(t.numel());
  const size_t ndim = static_cast<size_t>(t.dim());

  if (is_standard_contiguous(t)) {
    VALUE out = build_ruby_from_contiguous(
        t.const_data_ptr(),
        t.scalar_type(),
        t.sizes().data(),
        ndim,
        numel,
        flat);
    RB_GC_GUARD(self);
    return out;
  }

  // Non-contiguous: copy into temporary contiguous buffer first, then convert
  // inside rb_protect so buffer is always released if Ruby raises OOM.
  VALUE err_exc = Qnil;
  VALUE out = Qnil;
  int protect_state = 0;
  {
    auto buf_res = AlignedBuffer::allocate(t.nbytes());
    if (!buf_res.ok()) {
      err_exc = status_to_ruby_exception(buf_res.status);
    } else {
      auto cp = copy_to_contiguous_bytes(t, buf_res.value.ptr, t.nbytes());
      if (!cp.ok()) {
        err_exc = status_to_ruby_exception(cp);
      } else {
        struct BuildArg {
          const void* raw;
          executorch::aten::ScalarType dtype;
          const executorch::aten::SizesType* sizes;
          size_t ndim;
          size_t numel;
          bool flat;
        } ba{buf_res.value.ptr, t.scalar_type(), t.sizes().data(), ndim, numel, flat};
        out = rb_protect(
            [](VALUE a) -> VALUE {
              auto* b = reinterpret_cast<BuildArg*>(a);
              return build_ruby_from_contiguous(
                  b->raw, b->dtype, b->sizes, b->ndim, b->numel, b->flat);
            },
            reinterpret_cast<VALUE>(&ba),
            &protect_state);
      }
    }
  }
  RB_GC_GUARD(self);
  if (protect_state != 0) rb_jump_tag(protect_state);
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return out;
}

VALUE rb_tensor_to_a(VALUE self) { return tensor_to_a_impl(self, false); }
VALUE rb_tensor_flat_to_a(VALUE self) { return tensor_to_a_impl(self, true); }

VALUE rb_tensor_item(VALUE self) {
  auto* wrap = check_open_tensor(self);
  const size_t numel = (wrap->borrowed_buf != Qnil)
                           ? (wrap->borrowed_nbytes /
                              std::max<size_t>(1, scalar_type_size(wrap->borrowed_dtype)))
                           : static_cast<size_t>(wrap->tensor->numel());
  if (numel != 1) {
    rb_raise(
        g_refs.eShapeError,
        "item requires a tensor with exactly 1 element, got %zu",
        numel);
  }
  const void* ptr = nullptr;
  executorch::aten::ScalarType dtype = executorch::aten::ScalarType::Undefined;
  if (wrap->borrowed_buf != Qnil) {
    size_t len = 0;
    check_borrowed_read(wrap, &ptr, &len);
    dtype = wrap->borrowed_dtype;
  } else {
    ptr = wrap->tensor->const_data_ptr();
    dtype = wrap->tensor->scalar_type();
  }
  VALUE out = build_ruby_from_contiguous(ptr, dtype, nullptr, 0, 1, true);
  RB_GC_GUARD(self);
  return out;
}

VALUE rb_tensor_close(VALUE self) {
  RbTensorWrap* wrap = nullptr;
  TypedData_Get_Struct(self, RbTensorWrap, &kTensorDataType, wrap);
  if (wrap != nullptr && !wrap->closed) {
    wrap->closed = true;
    wrap->tensor.reset();
    wrap->borrowed_buf = Qnil;
    wrap->borrowed_sizes.clear();
    if (wrap->tracked_bytes > 0) {
      rb_gc_adjust_memory_usage(-static_cast<ssize_t>(wrap->tracked_bytes));
      wrap->tracked_bytes = 0;
    }
  }
  return Qnil;
}

VALUE rb_tensor_closed_p(VALUE self) {
  RbTensorWrap* wrap = nullptr;
  TypedData_Get_Struct(self, RbTensorWrap, &kTensorDataType, wrap);
  return (wrap == nullptr || wrap->closed) ? Qtrue : Qfalse;
}

VALUE rb_tensor_eq(VALUE self, VALUE other) {
  if (!is_rb_tensor(other)) {
    return Qfalse;
  }
  auto* a = check_open_tensor(self);
  auto* b = check_open_tensor(other);
  if (a == b) return Qtrue;

  const auto dtype_a = (a->borrowed_buf != Qnil) ? a->borrowed_dtype : a->tensor->scalar_type();
  const auto dtype_b = (b->borrowed_buf != Qnil) ? b->borrowed_dtype : b->tensor->scalar_type();
  if (dtype_a != dtype_b) return Qfalse;

  const executorch::aten::SizesType* sa =
      (a->borrowed_buf != Qnil) ? a->borrowed_sizes.data() : a->tensor->sizes().data();
  const size_t da =
      (a->borrowed_buf != Qnil) ? a->borrowed_sizes.size() : static_cast<size_t>(a->tensor->dim());
  const executorch::aten::SizesType* sb =
      (b->borrowed_buf != Qnil) ? b->borrowed_sizes.data() : b->tensor->sizes().data();
  const size_t db =
      (b->borrowed_buf != Qnil) ? b->borrowed_sizes.size() : static_cast<size_t>(b->tensor->dim());
  if (da != db) return Qfalse;
  for (size_t i = 0; i < da; ++i) {
    if (sa[i] != sb[i]) return Qfalse;
  }

  VALUE ba = rb_tensor_to_bytes(self);
  VALUE bb = rb_tensor_to_bytes(other);
  if (RSTRING_LEN(ba) != RSTRING_LEN(bb)) return Qfalse;
  const size_t n = static_cast<size_t>(RSTRING_LEN(ba));
  if (n == 0) return Qtrue;
  bool eq = (std::memcmp(RSTRING_PTR(ba), RSTRING_PTR(bb), n) == 0);
  RB_GC_GUARD(ba);
  RB_GC_GUARD(bb);
  return eq ? Qtrue : Qfalse;
}

}  // namespace

const rb_data_type_t kTensorDataType = {
    "Executorch::Tensor",
    {
        tensor_mark,
        tensor_free,
        tensor_memsize,
        nullptr,
        {nullptr},
    },
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY | RUBY_TYPED_FROZEN_SHAREABLE,
};

VALUE wrap_owned_tensor(executorch::extension::TensorPtr ptr) {
  VALUE obj = tensor_alloc(g_refs.cTensor);
  RbTensorWrap* wrap = nullptr;
  TypedData_Get_Struct(obj, RbTensorWrap, &kTensorDataType, wrap);
  const size_t nb = ptr ? ptr->nbytes() : 0;
  wrap->tensor = std::move(ptr);
  wrap->tracked_bytes = nb;
  if (nb > 0) {
    rb_gc_adjust_memory_usage(static_cast<ssize_t>(nb));
  }
  return obj;
}

bool is_rb_tensor(VALUE obj) {
  return rb_typeddata_is_kind_of(obj, &kTensorDataType) != 0;
}

RbTensorWrap* get_rb_tensor(VALUE obj) {
  return check_open_tensor(obj);
}

VALUE scalar_type_to_sym(executorch::aten::ScalarType st) {
  switch (st) {
    case executorch::aten::ScalarType::Float: return ID2SYM(g_refs.id_float32);
    case executorch::aten::ScalarType::Double: return ID2SYM(g_refs.id_float64);
    case executorch::aten::ScalarType::Half: return ID2SYM(g_refs.id_float16);
    case executorch::aten::ScalarType::BFloat16: return ID2SYM(g_refs.id_bfloat16);
    case executorch::aten::ScalarType::Byte: return ID2SYM(g_refs.id_uint8);
    case executorch::aten::ScalarType::Char: return ID2SYM(g_refs.id_int8);
    case executorch::aten::ScalarType::Short: return ID2SYM(g_refs.id_int16);
    case executorch::aten::ScalarType::UInt16: return ID2SYM(g_refs.id_uint16);
    case executorch::aten::ScalarType::Int: return ID2SYM(g_refs.id_int32);
    case executorch::aten::ScalarType::UInt32: return ID2SYM(g_refs.id_uint32);
    case executorch::aten::ScalarType::Long: return ID2SYM(g_refs.id_int64);
    case executorch::aten::ScalarType::UInt64: return ID2SYM(g_refs.id_uint64);
    case executorch::aten::ScalarType::Bool: return ID2SYM(g_refs.id_bool);
    default: return ID2SYM(rb_intern(scalar_type_name(st)));
  }
}

bool sym_to_scalar_type(VALUE sym, executorch::aten::ScalarType* out) {
  if (!RB_SYMBOL_P(sym)) {
    return false;
  }
  ID id = SYM2ID(sym);
  if (id == g_refs.id_float32) { *out = executorch::aten::ScalarType::Float; return true; }
  if (id == g_refs.id_float64) { *out = executorch::aten::ScalarType::Double; return true; }
  if (id == g_refs.id_int64) { *out = executorch::aten::ScalarType::Long; return true; }
  if (id == g_refs.id_int32) { *out = executorch::aten::ScalarType::Int; return true; }
  if (id == g_refs.id_uint8) { *out = executorch::aten::ScalarType::Byte; return true; }
  if (id == g_refs.id_bool) { *out = executorch::aten::ScalarType::Bool; return true; }
  if (id == g_refs.id_float16) { *out = executorch::aten::ScalarType::Half; return true; }
  if (id == g_refs.id_bfloat16) { *out = executorch::aten::ScalarType::BFloat16; return true; }
  if (id == g_refs.id_int8) { *out = executorch::aten::ScalarType::Char; return true; }
  if (id == g_refs.id_int16) { *out = executorch::aten::ScalarType::Short; return true; }
  if (id == g_refs.id_uint16) { *out = executorch::aten::ScalarType::UInt16; return true; }
  if (id == g_refs.id_uint32) { *out = executorch::aten::ScalarType::UInt32; return true; }
  if (id == g_refs.id_uint64) { *out = executorch::aten::ScalarType::UInt64; return true; }
  const char* name = rb_id2name(id);
  return name != nullptr && parse_dtype_name(name, out);
}

void init_tensor(VALUE mExecutorch) {
  g_refs.id_float32 = rb_intern("float32");
  g_refs.id_float64 = rb_intern("float64");
  g_refs.id_float16 = rb_intern("float16");
  g_refs.id_bfloat16 = rb_intern("bfloat16");
  g_refs.id_uint8 = rb_intern("uint8");
  g_refs.id_int8 = rb_intern("int8");
  g_refs.id_int16 = rb_intern("int16");
  g_refs.id_uint16 = rb_intern("uint16");
  g_refs.id_int32 = rb_intern("int32");
  g_refs.id_uint32 = rb_intern("uint32");
  g_refs.id_int64 = rb_intern("int64");
  g_refs.id_uint64 = rb_intern("uint64");
  g_refs.id_bool = rb_intern("bool");
  g_refs.id_unknown = rb_intern("unknown");
  g_refs.id_kw_shape = rb_intern("shape");
  g_refs.id_kw_dtype = rb_intern("dtype");

  g_refs.cTensor = rb_define_class_under(mExecutorch, "Tensor", rb_cObject);
  rb_undef_alloc_func(g_refs.cTensor);

  rb_define_singleton_method(g_refs.cTensor, "new", rb_tensor_s_new, -1);
  rb_define_singleton_method(g_refs.cTensor, "from_bytes", rb_tensor_s_from_bytes, -1);
  rb_define_singleton_method(g_refs.cTensor, "from_buffer", rb_tensor_s_from_buffer, -1);
  rb_define_singleton_method(g_refs.cTensor, "zeros", rb_tensor_s_zeros, -1);

  rb_define_method(g_refs.cTensor, "shape", rb_tensor_shape, 0);
  rb_define_method(g_refs.cTensor, "strides", rb_tensor_strides, 0);
  rb_define_method(g_refs.cTensor, "dim_order", rb_tensor_dim_order, 0);
  rb_define_method(g_refs.cTensor, "contiguous?", rb_tensor_contiguous_p, 0);
  rb_define_method(g_refs.cTensor, "dim", rb_tensor_dim, 0);
  rb_define_method(g_refs.cTensor, "ndim", rb_tensor_dim, 0);
  rb_define_method(g_refs.cTensor, "numel", rb_tensor_numel, 0);
  rb_define_method(g_refs.cTensor, "nbytes", rb_tensor_nbytes, 0);
  rb_define_method(g_refs.cTensor, "dtype", rb_tensor_dtype, 0);
  rb_define_method(g_refs.cTensor, "to_a", rb_tensor_to_a, 0);
  rb_define_method(g_refs.cTensor, "flat_to_a", rb_tensor_flat_to_a, 0);
  rb_define_method(g_refs.cTensor, "to_bytes", rb_tensor_to_bytes, 0);
  rb_define_method(g_refs.cTensor, "to_buffer", rb_tensor_to_buffer, 0);
  rb_define_method(g_refs.cTensor, "item", rb_tensor_item, 0);
  rb_define_method(g_refs.cTensor, "close", rb_tensor_close, 0);
  rb_define_method(g_refs.cTensor, "closed?", rb_tensor_closed_p, 0);
  rb_define_method(g_refs.cTensor, "==", rb_tensor_eq, 1);
}

}  // namespace etrb

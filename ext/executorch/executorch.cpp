/**
 * ExecuTorch Ruby Bindings
 *
 * This file provides Ruby bindings for Meta's ExecuTorch library using the Rice gem.
 * It wraps the high-level Module API for loading and executing PyTorch models.
 */

#include <rice/rice.hpp>
#include <rice/stl.hpp>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <unordered_set>

#include "utils.h"

using namespace Rice;
using namespace executorch::runtime;
using namespace executorch::extension;
namespace et = executorch;

// Forward declarations
class RubyTensor;
class RubyEValue;

/**
 * Ruby wrapper for executorch::aten::Tensor via TensorPtr
 *
 * This class manages tensor data and provides methods for creating tensors
 * from Ruby arrays and extracting data back to Ruby.
 */
class RubyTensor {
public:
  // Create a tensor from Ruby data array and shape
  static RubyTensor create(Array data, Array shape, Symbol dtype) {
    HANDLE_ET_ERRORS

    std::vector<et::aten::SizesType> sizes = read_sizes(shape.value());
    et::aten::ScalarType scalar_type = executorch_ruby::symbol_to_scalar_type(dtype.value());

    // make_tensor_ptr takes ownership of the data vector, so each branch fills
    // one and moves it in.
    VALUE ary = data.value();
    Check_Type(ary, T_ARRAY);

    switch (scalar_type) {
      case et::aten::ScalarType::Float:
        return RubyTensor(make_tensor_ptr<float>(
          std::move(sizes),
          executorch_ruby::read_array<float>(ary, executorch_ruby::to_double_fast)));
      case et::aten::ScalarType::Double:
        return RubyTensor(make_tensor_ptr<double>(
          std::move(sizes),
          executorch_ruby::read_array<double>(ary, executorch_ruby::to_double_fast)));
      case et::aten::ScalarType::Long:
        return RubyTensor(make_tensor_ptr<int64_t>(
          std::move(sizes),
          executorch_ruby::read_array<int64_t>(ary, executorch_ruby::to_int64_fast)));
      case et::aten::ScalarType::Int:
        return RubyTensor(make_tensor_ptr<int32_t>(
          std::move(sizes),
          executorch_ruby::read_array<int32_t>(ary, executorch_ruby::to_int64_fast)));
      default:
        rb_raise(rb_eArgError, "Unsupported dtype. Use :float, :double, :long, or :int");
    }

    // Should never reach here but compiler needs it
    rb_raise(rb_eRuntimeError, "Unexpected code path in Tensor.create");
    END_HANDLE_ET_ERRORS
  }

  // Create a tensor by copying raw bytes straight into the backing buffer.
  //
  // This is the escape hatch from per-element conversion: Ruby packs the data
  // once (Array#pack, or bytes read from a file/socket) and the whole tensor
  // arrives as one memcpy instead of numel boxed-number conversions. Bytes are
  // native-endian and must match the element width of `dtype` exactly.
  static RubyTensor from_binary(String data, Array shape, Symbol dtype) {
    HANDLE_ET_ERRORS

    std::vector<et::aten::SizesType> sizes = read_sizes(shape.value());
    et::aten::ScalarType scalar_type = executorch_ruby::symbol_to_scalar_type(dtype.value());
    const size_t item_size = executorch_ruby::element_size(scalar_type);

    int64_t numel = 1;
    for (auto size : sizes) {
      numel *= size;
    }

    VALUE str = data.value();
    Check_Type(str, T_STRING);
    const size_t expected = static_cast<size_t>(numel) * item_size;
    const size_t actual = static_cast<size_t>(RSTRING_LEN(str));
    if (actual != expected) {
      rb_raise(rb_eArgError,
               "Binary data is %zu bytes but shape %lld x %zu bytes requires %zu",
               actual, static_cast<long long>(numel), item_size, expected);
    }

    const char* bytes = RSTRING_PTR(str);
    switch (scalar_type) {
      case et::aten::ScalarType::Float:
        return RubyTensor(make_tensor_ptr<float>(std::move(sizes), copy_bytes<float>(bytes, numel)));
      case et::aten::ScalarType::Double:
        return RubyTensor(make_tensor_ptr<double>(std::move(sizes), copy_bytes<double>(bytes, numel)));
      case et::aten::ScalarType::Long:
        return RubyTensor(make_tensor_ptr<int64_t>(std::move(sizes), copy_bytes<int64_t>(bytes, numel)));
      case et::aten::ScalarType::Int:
        return RubyTensor(make_tensor_ptr<int32_t>(std::move(sizes), copy_bytes<int32_t>(bytes, numel)));
      default:
        rb_raise(rb_eArgError, "Unsupported dtype. Use :float, :double, :long, or :int");
    }

    rb_raise(rb_eRuntimeError, "Unexpected code path in Tensor.from_binary");
    END_HANDLE_ET_ERRORS
  }

  // Create a float tensor (convenience method)
  static RubyTensor from_array(Array data, Array shape) {
    return create(data, shape, Symbol("float"));
  }

  // Get shape as Ruby array
  Array shape() const {
    Array result;
    for (int i = 0; i < tensor_ptr_->dim(); i++) {
      result.push(tensor_ptr_->size(i));
    }
    return result;
  }

  // Get number of dimensions
  int64_t dim() const {
    return tensor_ptr_->dim();
  }

  // Get total number of elements
  int64_t numel() const {
    return tensor_ptr_->numel();
  }

  // Get dtype as symbol
  Object dtype() const {
    return Object(executorch_ruby::scalar_type_to_symbol(tensor_ptr_->scalar_type()));
  }

  // Convert tensor data to Ruby array (flattened)
  Array to_a() const {
    const int64_t n = tensor_ptr_->numel();
    auto scalar_type = tensor_ptr_->scalar_type();

    switch (scalar_type) {
      case et::aten::ScalarType::Float:
        return Array(executorch_ruby::build_array(
          tensor_ptr_->const_data_ptr<float>(), n,
          [](float v) { return DBL2NUM(static_cast<double>(v)); }));
      case et::aten::ScalarType::Double:
        return Array(executorch_ruby::build_array(
          tensor_ptr_->const_data_ptr<double>(), n,
          [](double v) { return DBL2NUM(v); }));
      case et::aten::ScalarType::Long:
        return Array(executorch_ruby::build_array(
          tensor_ptr_->const_data_ptr<int64_t>(), n,
          [](int64_t v) { return LL2NUM(v); }));
      case et::aten::ScalarType::Int:
        return Array(executorch_ruby::build_array(
          tensor_ptr_->const_data_ptr<int32_t>(), n,
          [](int32_t v) { return LONG2NUM(static_cast<long>(v)); }));
      default:
        rb_raise(rb_eRuntimeError, "Unsupported tensor dtype for to_a");
    }
  }

  // Copy the tensor's buffer out as a binary String, native-endian.
  // The mirror of from_binary: unpack it in Ruby, or hand it straight to
  // whatever wants bytes.
  String to_binary() const {
    const size_t nbytes = static_cast<size_t>(tensor_ptr_->numel()) *
                          executorch_ruby::element_size(tensor_ptr_->scalar_type());
    return String(rb_str_new(
      static_cast<const char*>(tensor_ptr_->const_data_ptr()), static_cast<long>(nbytes)));
  }

  // Get string representation
  std::string to_s() const {
    std::string result = "Tensor(shape=[";
    for (int i = 0; i < tensor_ptr_->dim(); i++) {
      if (i > 0) result += ", ";
      result += std::to_string(tensor_ptr_->size(i));
    }
    result += "], dtype=";

    auto scalar_type = tensor_ptr_->scalar_type();
    if (scalar_type == et::aten::ScalarType::Float) result += "float";
    else if (scalar_type == et::aten::ScalarType::Double) result += "double";
    else if (scalar_type == et::aten::ScalarType::Long) result += "long";
    else if (scalar_type == et::aten::ScalarType::Int) result += "int";
    else result += "unknown";

    result += ")";
    return result;
  }

  // Access the underlying tensor
  et::aten::Tensor& get() { return *tensor_ptr_; }
  const et::aten::Tensor& get() const { return *tensor_ptr_; }

  // Get the TensorPtr for ownership transfer
  TensorPtr& get_ptr() { return tensor_ptr_; }

  // Create from existing TensorPtr (used internally)
  explicit RubyTensor(TensorPtr ptr) : tensor_ptr_(std::move(ptr)) {}

  // Create from Tensor reference (clones the tensor)
  static RubyTensor from_tensor(const et::aten::Tensor& tensor) {
    return RubyTensor(clone_tensor_ptr(tensor));
  }

private:
  static std::vector<et::aten::SizesType> read_sizes(VALUE shape) {
    Check_Type(shape, T_ARRAY);
    const long dims = RARRAY_LEN(shape);
    std::vector<et::aten::SizesType> sizes;
    sizes.reserve(static_cast<size_t>(dims));
    for (long i = 0; i < dims; i++) {
      sizes.push_back(static_cast<et::aten::SizesType>(
        executorch_ruby::to_int64_fast(RARRAY_AREF(shape, i))));
    }
    return sizes;
  }

  template <typename T>
  static std::vector<T> copy_bytes(const char* bytes, int64_t numel) {
    std::vector<T> out(static_cast<size_t>(numel));
    std::memcpy(out.data(), bytes, static_cast<size_t>(numel) * sizeof(T));
    return out;
  }

  TensorPtr tensor_ptr_;
};

/**
 * Ruby wrapper for executorch::runtime::EValue
 *
 * EValue is a tagged union that can hold different value types:
 * - None
 * - Int (int64_t)
 * - Double
 * - Bool
 * - String
 * - Tensor
 * - Lists (IntList, DoubleList, BoolList, TensorList)
 */
class RubyEValue {
public:
  // Create from different types
  static RubyEValue from_none() {
    return RubyEValue(EValue());
  }

  static RubyEValue from_int(int64_t value) {
    return RubyEValue(EValue(value));
  }

  static RubyEValue from_double(double value) {
    return RubyEValue(EValue(value));
  }

  static RubyEValue from_bool(bool value) {
    return RubyEValue(EValue(value));
  }

  static RubyEValue from_tensor(RubyTensor& tensor) {
    return RubyEValue(EValue(tensor.get()), tensor.get_ptr());
  }

  // Type checking
  bool is_none() const { return evalue_.isNone(); }
  bool is_int() const { return evalue_.isInt(); }
  bool is_double() const { return evalue_.isDouble(); }
  bool is_bool() const { return evalue_.isBool(); }
  bool is_string() const { return evalue_.isString(); }
  bool is_tensor() const { return evalue_.isTensor(); }

  // Value extraction
  int64_t to_int() const {
    if (!is_int()) {
      rb_raise(rb_eTypeError, "EValue is not an Int");
    }
    return evalue_.toInt();
  }

  double to_double() const {
    if (!is_double()) {
      rb_raise(rb_eTypeError, "EValue is not a Double");
    }
    return evalue_.toDouble();
  }

  bool to_bool() const {
    if (!is_bool()) {
      rb_raise(rb_eTypeError, "EValue is not a Bool");
    }
    return evalue_.toBool();
  }

  RubyTensor to_tensor() const {
    if (!is_tensor()) {
      rb_raise(rb_eTypeError, "EValue is not a Tensor");
    }
    return RubyTensor::from_tensor(evalue_.toTensor());
  }

  // Convert to Ruby object based on type
  Object to_ruby() const {
    if (is_none()) {
      return Object(Qnil);
    } else if (is_int()) {
      return Object(LONG2NUM(evalue_.toInt()));
    } else if (is_double()) {
      return Object(DBL2NUM(evalue_.toDouble()));
    } else if (is_bool()) {
      return Object(evalue_.toBool() ? Qtrue : Qfalse);
    } else if (is_tensor()) {
      // Return tensor info as hash
      auto tensor = evalue_.toTensor();
      VALUE hash = rb_hash_new();
      rb_hash_aset(hash, ID2SYM(rb_intern("type")), rb_str_new_cstr("tensor"));
      rb_hash_aset(hash, ID2SYM(rb_intern("dtype")),
                   executorch_ruby::scalar_type_to_symbol(tensor.scalar_type()));

      // Build shape array
      VALUE shape = rb_ary_new();
      for (int i = 0; i < tensor.dim(); i++) {
        rb_ary_push(shape, LONG2NUM(tensor.size(i)));
      }
      rb_hash_aset(hash, ID2SYM(rb_intern("shape")), shape);

      return Object(hash);
    } else {
      return Object(Qnil);
    }
  }

  // Get type name
  std::string type_name() const {
    if (is_none()) return "None";
    if (is_int()) return "Int";
    if (is_double()) return "Double";
    if (is_bool()) return "Bool";
    if (is_string()) return "String";
    if (is_tensor()) return "Tensor";
    return "Unknown";
  }

  // Access underlying EValue
  const EValue& get() const { return evalue_; }
  EValue& get() { return evalue_; }

  // For internal use - create from EValue with tensor ownership
  explicit RubyEValue(EValue evalue, TensorPtr tensor_ptr = nullptr)
    : evalue_(std::move(evalue)), tensor_ptr_(std::move(tensor_ptr)) {}

private:
  EValue evalue_;
  TensorPtr tensor_ptr_;  // Keep tensor alive if EValue references it
};

/**
 * Ruby wrapper for executorch::extension::Module
 *
 * This class manages the lifecycle of an ExecuTorch model and provides
 * methods for loading and executing inference.
 */
class RubyModel {
public:
  RubyModel(const std::string& path)
    : path_(path), module_(nullptr) {
    HANDLE_ET_ERRORS
    module_ = std::make_unique<et::extension::Module>(path);
    // Auto-load on construction
    auto err = module_->load();
    executorch_ruby::check_error(err);
    END_HANDLE_ET_ERRORS
  }

  bool is_loaded() const {
    return module_ && module_->is_loaded();
  }

  Array method_names() {
    HANDLE_ET_ERRORS
    if (!is_loaded()) {
      rb_raise(rb_eRuntimeError, "Module not loaded");
    }
    auto result = module_->method_names();
    auto names_set = executorch_ruby::unwrap_result(std::move(result));
    Array arr;
    for (const auto& name : names_set) {
      arr.push(String(name));
    }
    return arr;
    END_HANDLE_ET_ERRORS
  }

  // Execute the forward method with tensor inputs
  Array forward(Array inputs) {
    HANDLE_ET_ERRORS
    if (!is_loaded()) {
      rb_raise(rb_eRuntimeError, "Module not loaded");
    }

    // Load forward method if not already loaded
    if (!module_->is_method_loaded("forward")) {
      auto load_err = module_->load_method("forward");
      if (load_err != executorch::runtime::Error::Ok) {
        rb_raise(rb_eRuntimeError, "Failed to load forward method: %s",
                 executorch_ruby::error_name(load_err));
      }
    }

    build_inputs(inputs);

    auto result = module_->forward(input_evalues_);
    if (!result.ok()) {
      rb_raise(rb_eRuntimeError, "Forward execution failed: %s",
               executorch_ruby::error_name(result.error()));
    }

    return wrap_outputs(result.get());
    END_HANDLE_ET_ERRORS
  }

  // Execute a named method
  Array execute(const std::string& method_name, Array inputs) {
    HANDLE_ET_ERRORS
    if (!is_loaded()) {
      rb_raise(rb_eRuntimeError, "Module not loaded");
    }

    build_inputs(inputs);

    auto result = module_->execute(method_name, input_evalues_);
    if (!result.ok()) {
      rb_raise(rb_eRuntimeError, "Execution of '%s' failed: %s", method_name.c_str(),
               executorch_ruby::error_name(result.error()));
    }

    return wrap_outputs(result.get());
    END_HANDLE_ET_ERRORS
  }

  // Get the file path
  std::string path() const {
    return path_;
  }

  // Access the underlying module for advanced use
  et::extension::Module* get_module() {
    return module_.get();
  }

private:
  // Turn the Ruby argument array into the EValue vector the runtime wants.
  //
  // Two things worth noting:
  //
  // * The input tensors are used in place rather than cloned. The old code
  //   deep-copied every input on every call to "own the data during forward",
  //   but the caller's Array holds a live reference to each Tensor for the
  //   whole call, so the buffer cannot be collected underneath us -- the copy
  //   bought nothing and cost a full pass over the input.
  // * Dispatch is by type check, not by catching the exception Rice throws on a
  //   failed conversion. Throwing and unwinding to identify a type costs more
  //   than a small model's entire inference.
  void build_inputs(Array& inputs) {
    VALUE ary = inputs.value();
    Check_Type(ary, T_ARRAY);
    const long n = RARRAY_LEN(ary);

    input_evalues_.clear();
    input_evalues_.reserve(static_cast<size_t>(n));

    for (long i = 0; i < n; i++) {
      VALUE item = RARRAY_AREF(ary, i);

      if (Data_Type<RubyTensor>::is_descendant(item)) {
        RubyTensor* tensor = detail::unwrap<RubyTensor>(
          item, Data_Type<RubyTensor>::ruby_data_type(), false);
        input_evalues_.push_back(EValue(tensor->get()));
      } else if (Data_Type<RubyEValue>::is_descendant(item)) {
        RubyEValue* evalue = detail::unwrap<RubyEValue>(
          item, Data_Type<RubyEValue>::ruby_data_type(), false);
        input_evalues_.push_back(evalue->get());
      } else {
        rb_raise(rb_eTypeError, "Input %ld must be a Tensor or EValue", i);
      }
    }
  }

  // Outputs *are* cloned: they point into the method's planned memory arena,
  // which the next call overwrites.
  template <typename Outputs_T>
  Array wrap_outputs(Outputs_T& outputs) {
    VALUE ruby_outputs = rb_ary_new_capa(static_cast<long>(outputs.size()));
    for (auto& output : outputs) {
      if (output.isTensor()) {
        rb_ary_push(ruby_outputs,
                    detail::To_Ruby<RubyTensor>().convert(RubyTensor::from_tensor(output.toTensor())));
      } else if (output.isInt()) {
        rb_ary_push(ruby_outputs, LL2NUM(output.toInt()));
      } else if (output.isDouble()) {
        rb_ary_push(ruby_outputs, DBL2NUM(output.toDouble()));
      } else if (output.isBool()) {
        rb_ary_push(ruby_outputs, output.toBool() ? Qtrue : Qfalse);
      } else {
        rb_ary_push(ruby_outputs, Qnil);
      }
    }
    return Array(ruby_outputs);
  }

  std::string path_;
  std::unique_ptr<et::extension::Module> module_;
  std::vector<EValue> input_evalues_;  // reused across calls to avoid churn
};

/**
 * Initialize the Executorch Ruby module
 */
extern "C"
void Init_executorch() {
  Rice::Module m = define_module("Executorch");

  // Define version constant
  m.const_set("NATIVE_VERSION", String("0.1.0"));

  // Define error class
  define_class_under<std::runtime_error>(m, "NativeError")
    .define_constructor(Constructor<std::runtime_error, const std::string&>());

  // Define Tensor class
  define_class_under<RubyTensor>(m, "Tensor")
    .define_singleton_function("create", &RubyTensor::create,
      Arg("data"), Arg("shape"), Arg("dtype"))
    .define_singleton_function("from_array", &RubyTensor::from_array,
      Arg("data"), Arg("shape"))
    .define_singleton_function("from_binary", &RubyTensor::from_binary,
      Arg("data"), Arg("shape"), Arg("dtype"))
    .define_method("to_binary", &RubyTensor::to_binary)
    .define_method("shape", &RubyTensor::shape)
    .define_method("dim", &RubyTensor::dim)
    .define_method("numel", &RubyTensor::numel)
    .define_method("dtype", &RubyTensor::dtype)
    .define_method("_original_to_a", &RubyTensor::to_a)
    .define_method("to_s", &RubyTensor::to_s)
    .define_method("inspect", &RubyTensor::to_s);

  // Define Model class
  define_class_under<RubyModel>(m, "Model")
    .define_constructor(Constructor<RubyModel, const std::string&>(),
      Arg("path"))
    .define_method("loaded?", &RubyModel::is_loaded)
    .define_method("method_names", &RubyModel::method_names)
    .define_method("path", &RubyModel::path)
    .define_method("forward", &RubyModel::forward,
      Arg("inputs"))
    .define_method("execute", &RubyModel::execute,
      Arg("method_name"), Arg("inputs"));

  // Note: EValue is kept internal - users interact with Tensor and native Ruby types
}

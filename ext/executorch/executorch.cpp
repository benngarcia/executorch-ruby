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

    // Convert shape to vector using the proper SizesType
    std::vector<et::aten::SizesType> sizes;
    for (size_t i = 0; i < shape.size(); i++) {
      sizes.push_back(static_cast<et::aten::SizesType>(
        detail::From_Ruby<int64_t>().convert(shape[i].value())));
    }

    // Get scalar type from symbol
    et::aten::ScalarType scalar_type = executorch_ruby::symbol_to_scalar_type(dtype.value());

    // Convert data based on dtype and create tensor using templated make_tensor_ptr
    // The templated version takes ownership of the data vector and handles memory management
    if (scalar_type == et::aten::ScalarType::Float) {
      std::vector<float> float_data;
      float_data.reserve(data.size());
      for (size_t i = 0; i < data.size(); i++) {
        float_data.push_back(static_cast<float>(detail::From_Ruby<double>().convert(data[i].value())));
      }
      // Use templated make_tensor_ptr which manages data ownership
      TensorPtr tensor_ptr = make_tensor_ptr<float>(
        std::move(sizes),
        std::move(float_data)
      );
      return RubyTensor(std::move(tensor_ptr));
    } else if (scalar_type == et::aten::ScalarType::Double) {
      std::vector<double> double_data;
      double_data.reserve(data.size());
      for (size_t i = 0; i < data.size(); i++) {
        double_data.push_back(detail::From_Ruby<double>().convert(data[i].value()));
      }
      TensorPtr tensor_ptr = make_tensor_ptr<double>(
        std::move(sizes),
        std::move(double_data)
      );
      return RubyTensor(std::move(tensor_ptr));
    } else if (scalar_type == et::aten::ScalarType::Long) {
      std::vector<int64_t> int_data;
      int_data.reserve(data.size());
      for (size_t i = 0; i < data.size(); i++) {
        int_data.push_back(detail::From_Ruby<int64_t>().convert(data[i].value()));
      }
      TensorPtr tensor_ptr = make_tensor_ptr<int64_t>(
        std::move(sizes),
        std::move(int_data)
      );
      return RubyTensor(std::move(tensor_ptr));
    } else if (scalar_type == et::aten::ScalarType::Int) {
      std::vector<int32_t> int_data;
      int_data.reserve(data.size());
      for (size_t i = 0; i < data.size(); i++) {
        int_data.push_back(static_cast<int32_t>(detail::From_Ruby<int64_t>().convert(data[i].value())));
      }
      TensorPtr tensor_ptr = make_tensor_ptr<int32_t>(
        std::move(sizes),
        std::move(int_data)
      );
      return RubyTensor(std::move(tensor_ptr));
    } else {
      rb_raise(rb_eArgError, "Unsupported dtype. Use :float, :double, :long, or :int");
    }

    // Should never reach here but compiler needs it
    rb_raise(rb_eRuntimeError, "Unexpected code path in Tensor.create");
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
    Array result;
    auto scalar_type = tensor_ptr_->scalar_type();

    if (scalar_type == et::aten::ScalarType::Float) {
      const float* data = tensor_ptr_->const_data_ptr<float>();
      for (int64_t i = 0; i < tensor_ptr_->numel(); i++) {
        result.push(data[i]);
      }
    } else if (scalar_type == et::aten::ScalarType::Double) {
      const double* data = tensor_ptr_->const_data_ptr<double>();
      for (int64_t i = 0; i < tensor_ptr_->numel(); i++) {
        result.push(data[i]);
      }
    } else if (scalar_type == et::aten::ScalarType::Long) {
      const int64_t* data = tensor_ptr_->const_data_ptr<int64_t>();
      for (int64_t i = 0; i < tensor_ptr_->numel(); i++) {
        result.push(data[i]);
      }
    } else if (scalar_type == et::aten::ScalarType::Int) {
      const int32_t* data = tensor_ptr_->const_data_ptr<int32_t>();
      for (int64_t i = 0; i < tensor_ptr_->numel(); i++) {
        result.push(static_cast<int64_t>(data[i]));
      }
    } else {
      rb_raise(rb_eRuntimeError, "Unsupported tensor dtype for to_a");
    }

    return result;
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
        const char* load_error_name = "Unknown";
        switch (load_err) {
          case executorch::runtime::Error::Ok: load_error_name = "Ok"; break;
          case executorch::runtime::Error::Internal: load_error_name = "Internal"; break;
          case executorch::runtime::Error::InvalidState: load_error_name = "InvalidState"; break;
          case executorch::runtime::Error::InvalidArgument: load_error_name = "InvalidArgument"; break;
          case executorch::runtime::Error::InvalidType: load_error_name = "InvalidType"; break;
          case executorch::runtime::Error::NotFound: load_error_name = "NotFound"; break;
          case executorch::runtime::Error::MemoryAllocationFailed: load_error_name = "MemoryAllocationFailed"; break;
          case executorch::runtime::Error::AccessFailed: load_error_name = "AccessFailed"; break;
          case executorch::runtime::Error::NotSupported: load_error_name = "NotSupported"; break;
          default: load_error_name = "Unknown"; break;
        }
        rb_raise(rb_eRuntimeError, "Failed to load forward method: %s (%d)", load_error_name, static_cast<int>(load_err));
      }
    }

    // Convert Ruby inputs to EValues
    // Keep tensors alive during forward execution
    std::vector<TensorPtr> input_tensors;
    std::vector<EValue> input_evalues;

    for (size_t i = 0; i < inputs.size(); i++) {
      Object input = inputs[i];

      // Check if it's a RubyTensor
      if (input.is_a(rb_cObject)) {
        try {
          RubyTensor& tensor = detail::From_Ruby<RubyTensor&>().convert(input.value());
          // Clone the tensor to ensure we own the data during forward
          TensorPtr cloned = clone_tensor_ptr(tensor.get());
          input_tensors.push_back(cloned);
          input_evalues.push_back(EValue(*cloned));
        } catch (...) {
          // Try as RubyEValue
          try {
            RubyEValue& evalue = detail::From_Ruby<RubyEValue&>().convert(input.value());
            input_evalues.push_back(evalue.get());
          } catch (...) {
            rb_raise(rb_eTypeError, "Input %zu must be a Tensor or EValue", i);
          }
        }
      }
    }

    // Execute forward
    auto result = module_->forward(input_evalues);
    if (!result.ok()) {
      auto error = result.error();
      const char* error_name = "Unknown";
      switch (error) {
        case executorch::runtime::Error::Ok: error_name = "Ok"; break;
        case executorch::runtime::Error::Internal: error_name = "Internal"; break;
        case executorch::runtime::Error::InvalidState: error_name = "InvalidState"; break;
        case executorch::runtime::Error::InvalidArgument: error_name = "InvalidArgument"; break;
        case executorch::runtime::Error::InvalidType: error_name = "InvalidType"; break;
        case executorch::runtime::Error::NotFound: error_name = "NotFound"; break;
        case executorch::runtime::Error::MemoryAllocationFailed: error_name = "MemoryAllocationFailed"; break;
        case executorch::runtime::Error::AccessFailed: error_name = "AccessFailed"; break;
        case executorch::runtime::Error::NotSupported: error_name = "NotSupported"; break;
        default: error_name = "Unknown"; break;
      }
      rb_raise(rb_eRuntimeError, "Forward execution failed: %s (%d)", error_name, static_cast<int>(error));
    }
    auto outputs = std::move(result.get());

    // Convert outputs to Ruby array of RubyTensors
    Array ruby_outputs;
    for (auto& output : outputs) {
      if (output.isTensor()) {
        // Clone the tensor to own the data
        ruby_outputs.push(RubyTensor::from_tensor(output.toTensor()));
      } else if (output.isInt()) {
        ruby_outputs.push(output.toInt());
      } else if (output.isDouble()) {
        ruby_outputs.push(output.toDouble());
      } else if (output.isBool()) {
        ruby_outputs.push(output.toBool() ? Qtrue : Qfalse);
      } else {
        ruby_outputs.push(Qnil);
      }
    }

    return ruby_outputs;
    END_HANDLE_ET_ERRORS
  }

  // Execute a named method
  Array execute(const std::string& method_name, Array inputs) {
    HANDLE_ET_ERRORS
    if (!is_loaded()) {
      rb_raise(rb_eRuntimeError, "Module not loaded");
    }

    // Convert Ruby inputs to EValues
    // Keep tensors alive during execution
    std::vector<TensorPtr> input_tensors;
    std::vector<EValue> input_evalues;
    for (size_t i = 0; i < inputs.size(); i++) {
      Object input = inputs[i];

      try {
        RubyTensor& tensor = detail::From_Ruby<RubyTensor&>().convert(input.value());
        // Clone the tensor to ensure we own the data during execution
        TensorPtr cloned = clone_tensor_ptr(tensor.get());
        input_tensors.push_back(cloned);
        input_evalues.push_back(EValue(*cloned));
      } catch (...) {
        try {
          RubyEValue& evalue = detail::From_Ruby<RubyEValue&>().convert(input.value());
          input_evalues.push_back(evalue.get());
        } catch (...) {
          rb_raise(rb_eTypeError, "Input %zu must be a Tensor or EValue", i);
        }
      }
    }

    // Execute method
    auto result = module_->execute(method_name, input_evalues);
    auto outputs = executorch_ruby::unwrap_result(std::move(result));

    // Convert outputs to Ruby
    Array ruby_outputs;
    for (auto& output : outputs) {
      if (output.isTensor()) {
        ruby_outputs.push(RubyTensor::from_tensor(output.toTensor()));
      } else if (output.isInt()) {
        ruby_outputs.push(output.toInt());
      } else if (output.isDouble()) {
        ruby_outputs.push(output.toDouble());
      } else if (output.isBool()) {
        ruby_outputs.push(output.toBool() ? Qtrue : Qfalse);
      } else {
        ruby_outputs.push(Qnil);
      }
    }

    return ruby_outputs;
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
  std::string path_;
  std::unique_ptr<et::extension::Module> module_;
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

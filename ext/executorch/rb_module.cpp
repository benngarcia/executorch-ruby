#include "rb_internal.hpp"

#include <atomic>
#include <new>

namespace etrb {

namespace {

void module_free(void* p) {
  auto* wrap = static_cast<RbModuleWrap*>(p);
  if (wrap->tracked_bytes > 0) {
    rb_gc_adjust_memory_usage(-static_cast<ssize_t>(wrap->tracked_bytes));
    wrap->tracked_bytes = 0;
  }
  delete wrap;
}

size_t module_memsize(const void* p) {
  const auto* wrap = static_cast<const RbModuleWrap*>(p);
  return sizeof(RbModuleWrap) + wrap->tracked_bytes;
}

VALUE module_alloc(VALUE klass) {
  auto* wrap = new (std::nothrow) RbModuleWrap();
  if (wrap == nullptr) {
    rb_memerror();
  }
  return TypedData_Wrap_Struct(klass, &kModuleDataType, wrap);
}

RbModuleWrap* get_module_wrap(VALUE self) {
  RbModuleWrap* wrap = nullptr;
  TypedData_Get_Struct(self, RbModuleWrap, &kModuleDataType, wrap);
  if (wrap == nullptr || !wrap->core || wrap->core->is_closed()) {
    rb_raise(g_refs.eClosedError, "Executorch::Module is closed");
  }
  return wrap;
}

ModuleCore::LoadMode parse_load_mode(VALUE val) {
  if (val == Qundef || val == Qnil) {
    return ModuleCore::LoadMode::Mmap;
  }
  if (!RB_SYMBOL_P(val)) {
    rb_raise(rb_eTypeError, "load_mode must be a Symbol");
  }
  ID id = SYM2ID(val);
  if (id == g_refs.id_mmap) return ModuleCore::LoadMode::Mmap;
  if (id == g_refs.id_file) return ModuleCore::LoadMode::File;
  if (id == g_refs.id_mmap_mlock) return ModuleCore::LoadMode::MmapUseMlock;
  if (id == g_refs.id_mmap_mlock_ignore_errors) {
    return ModuleCore::LoadMode::MmapUseMlockIgnoreErrors;
  }
  if (id == g_refs.id_mmap_madvise) return ModuleCore::LoadMode::MmapUseMadvise;
  rb_raise(
      rb_eArgError,
      "unknown load_mode :%s (expected :mmap, :file, :mmap_mlock, "
      ":mmap_mlock_ignore_errors, or :mmap_madvise)",
      rb_id2name(id));
}

std::string value_to_method_name(VALUE v) {
  if (RB_SYMBOL_P(v)) {
    const char* s = rb_id2name(SYM2ID(v));
    return s ? std::string(s) : std::string();
  }
  VALUE str = StringValue(v);
  return std::string(RSTRING_PTR(str), static_cast<size_t>(RSTRING_LEN(str)));
}

const char* tag_to_name(executorch::runtime::Tag tag) noexcept {
  switch (tag) {
    case executorch::runtime::Tag::None: return "none";
    case executorch::runtime::Tag::Tensor: return "tensor";
    case executorch::runtime::Tag::String: return "string";
    case executorch::runtime::Tag::Double: return "double";
    case executorch::runtime::Tag::Int: return "int";
    case executorch::runtime::Tag::Bool: return "bool";
    case executorch::runtime::Tag::ListBool: return "list_bool";
    case executorch::runtime::Tag::ListDouble: return "list_double";
    case executorch::runtime::Tag::ListTensor: return "list_tensor";
    case executorch::runtime::Tag::ListScalar: return "list_scalar";
    case executorch::runtime::Tag::ListOptionalTensor: return "list_optional_tensor";
    case executorch::runtime::Tag::ListInt: return "list_int";
    default: return "unknown";
  }
}

VALUE build_value_meta(const ValueMetaSnapshot& vm) {
  VALUE tag_sym = ID2SYM(rb_intern(tag_to_name(vm.tag)));
  if (vm.tag != executorch::runtime::Tag::Tensor) {
    return rb_funcall(
        g_refs.cTensorInfo,
        rb_intern("new"),
        6,
        tag_sym,
        Qnil,
        Qnil,
        Qnil,
        Qnil,
        Qfalse);
  }
  VALUE shape = rb_ary_new_capa(static_cast<long>(vm.tensor.sizes.size()));
  for (auto s : vm.tensor.sizes) rb_ary_push(shape, LONG2NUM(s));
  VALUE order = rb_ary_new_capa(static_cast<long>(vm.tensor.dim_order.size()));
  for (auto o : vm.tensor.dim_order) rb_ary_push(order, INT2FIX(o));
  VALUE dtype = scalar_type_to_sym(vm.tensor.dtype);
  VALUE nbytes = ULL2NUM(vm.tensor.nbytes);
  VALUE planned = vm.tensor.memory_planned ? Qtrue : Qfalse;
  return rb_funcall(
      g_refs.cTensorInfo,
      rb_intern("new"),
      6,
      tag_sym,
      dtype,
      shape,
      order,
      nbytes,
      planned);
}

VALUE build_method_meta(const MethodMetaSnapshot& snap) {
  VALUE name = rb_utf8_str_new(snap.name.data(), static_cast<long>(snap.name.size()));
  VALUE inputs = rb_ary_new_capa(static_cast<long>(snap.inputs.size()));
  for (const auto& vm : snap.inputs) rb_ary_push(inputs, build_value_meta(vm));
  VALUE outputs = rb_ary_new_capa(static_cast<long>(snap.outputs.size()));
  for (const auto& vm : snap.outputs) rb_ary_push(outputs, build_value_meta(vm));
  VALUE backends = rb_ary_new_capa(static_cast<long>(snap.backends.size()));
  for (const auto& b : snap.backends) {
    rb_ary_push(backends, rb_utf8_str_new(b.data(), static_cast<long>(b.size())));
  }
  VALUE planned = LL2NUM(snap.planned_memory_bytes);
  return rb_funcall(
      g_refs.cMethodMeta,
      rb_intern("new"),
      5,
      name,
      inputs,
      outputs,
      backends,
      planned);
}

// Convert Ruby VALUE to CoreValue under the GVL. For borrowed IO::Buffer
// tensors, records the IO::Buffer VALUE in `locked_bufs` so we lock it around
// the off-GVL execute call and guarantee unlock via rb_ensure.
Status ruby_to_core_value(
    VALUE v,
    CoreValue* out,
    std::vector<VALUE>* locked_bufs,
    bool for_output_target) {
  if (is_rb_tensor(v)) {
    RbTensorWrap* wrap = nullptr;
    TypedData_Get_Struct(v, RbTensorWrap, &kTensorDataType, wrap);
    if (wrap == nullptr || wrap->closed ||
        (!wrap->tensor && wrap->borrowed_buf == Qnil)) {
      return Status::Make(ErrorKind::Closed, "Executorch::Tensor is closed");
    }
    if (wrap->borrowed_buf != Qnil) {
      void* ptr = nullptr;
      size_t len = 0;
      if (for_output_target) {
        rb_io_buffer_get_bytes_for_writing(wrap->borrowed_buf, &ptr, &len);
      } else {
        const void* cptr = nullptr;
        rb_io_buffer_get_bytes_for_reading(wrap->borrowed_buf, &cptr, &len);
        ptr = const_cast<void*>(cptr);
      }
      if (len != wrap->borrowed_nbytes) {
        return Status::Make(
            ErrorKind::Shape,
            "IO::Buffer size changed: expected " +
                std::to_string(wrap->borrowed_nbytes) + ", got " +
                std::to_string(len));
      }
      auto tp_res = make_borrowed_tensor_ptr(
          wrap->borrowed_sizes, wrap->borrowed_dtype, ptr);
      if (!tp_res.ok()) {
        return tp_res.status;
      }
      locked_bufs->push_back(wrap->borrowed_buf);
      *out = CoreValue::Tensor(std::move(tp_res.value));
      return Status::Ok();
    }
    *out = CoreValue::Tensor(wrap->tensor);
    return Status::Ok();
  }

  if (for_output_target) {
    return Status::Make(ErrorKind::Type, "out: target must be an Executorch::Tensor");
  }

  if (v == Qnil) {
    *out = CoreValue::None();
    return Status::Ok();
  }
  if (v == Qtrue) {
    *out = CoreValue::Bool(true);
    return Status::Ok();
  }
  if (v == Qfalse) {
    *out = CoreValue::Bool(false);
    return Status::Ok();
  }
  if (RB_INTEGER_TYPE_P(v)) {
    *out = CoreValue::Int(NUM2LL(v));
    return Status::Ok();
  }
  if (RB_FLOAT_TYPE_P(v)) {
    *out = CoreValue::Double(RFLOAT_VALUE(v));
    return Status::Ok();
  }
  if (RB_TYPE_P(v, T_STRING)) {
    *out = CoreValue::String(
        std::string(RSTRING_PTR(v), static_cast<size_t>(RSTRING_LEN(v))));
    return Status::Ok();
  }
  return Status::Make(
      ErrorKind::Type,
      std::string("unsupported ExecuTorch input type: ") + rb_obj_classname(v));
}

VALUE core_value_to_ruby(CoreValue& cv) {
  switch (cv.kind) {
    case ValueKind::None:
      return Qnil;
    case ValueKind::Int:
      return LL2NUM(cv.int_val);
    case ValueKind::Double:
      return DBL2NUM(cv.double_val);
    case ValueKind::Bool:
      return cv.bool_val ? Qtrue : Qfalse;
    case ValueKind::String:
      return rb_utf8_str_new(
          cv.string_val.data(), static_cast<long>(cv.string_val.size()));
    case ValueKind::Tensor:
      return wrap_owned_tensor(std::move(cv.tensor_val));
    case ValueKind::IntList: {
      VALUE ary = rb_ary_new_capa(static_cast<long>(cv.int_list.size()));
      for (auto x : cv.int_list) rb_ary_push(ary, LL2NUM(x));
      return ary;
    }
    case ValueKind::DoubleList: {
      VALUE ary = rb_ary_new_capa(static_cast<long>(cv.double_list.size()));
      for (auto x : cv.double_list) rb_ary_push(ary, DBL2NUM(x));
      return ary;
    }
    case ValueKind::BoolList: {
      VALUE ary = rb_ary_new_capa(static_cast<long>(cv.bool_list.size()));
      for (bool x : cv.bool_list) rb_ary_push(ary, x ? Qtrue : Qfalse);
      return ary;
    }
    case ValueKind::TensorList: {
      VALUE ary = rb_ary_new_capa(static_cast<long>(cv.tensor_list.size()));
      for (auto& tp : cv.tensor_list) {
        rb_ary_push(ary, wrap_owned_tensor(std::move(tp)));
      }
      return ary;
    }
  }
  return Qnil;
}

// ---------------------------------------------------------------------------
// Ruby methods for Executorch::Module
// ---------------------------------------------------------------------------

// Module._open_file(path, load_mode:, data_files:)
VALUE rb_module_s_open_file(int argc, VALUE* argv, VALUE klass) {
  VALUE path_val = Qnil;
  VALUE opts = Qnil;
  rb_scan_args(argc, argv, "1:", &path_val, &opts);

  // Supports String and Pathname via rb_get_path!
  VALUE path_str = rb_get_path(path_val);
  VALUE kw_vals[2] = {Qundef, Qundef};
  ID kw_ids[2] = {g_refs.id_kw_load_mode, g_refs.id_kw_data_files};
  if (!NIL_P(opts)) {
    rb_get_kwargs(opts, kw_ids, 0, 2, kw_vals);
  }
  const auto mode = parse_load_mode(kw_vals[0]);

  VALUE err_exc = Qnil;
  VALUE obj = Qnil;
  int protect_state = 0;

  {
    std::string path(RSTRING_PTR(path_str), static_cast<size_t>(RSTRING_LEN(path_str)));
    std::vector<std::string> data_files;

    if (kw_vals[1] != Qundef && kw_vals[1] != Qnil) {
      VALUE df_ary = kw_vals[1];
      if (!RB_TYPE_P(df_ary, T_ARRAY)) {
        err_exc = rb_exc_new_cstr(rb_eTypeError, "data_files must be an Array of paths");
      } else {
        struct DfArg {
          VALUE ary;
          std::vector<std::string>* out;
        } dfa{df_ary, &data_files};
        rb_protect(
            [](VALUE a) -> VALUE {
              auto* d = reinterpret_cast<DfArg*>(a);
              long n = RARRAY_LEN(d->ary);
              d->out->reserve(static_cast<size_t>(n));
              for (long i = 0; i < n; ++i) {
                VALUE ps = rb_get_path(rb_ary_entry(d->ary, i));
                d->out->emplace_back(
                    RSTRING_PTR(ps), static_cast<size_t>(RSTRING_LEN(ps)));
              }
              return Qnil;
            },
            reinterpret_cast<VALUE>(&dfa),
            &protect_state);
      }
    }

    if (protect_state == 0 && err_exc == Qnil) {
      struct OpenCtx {
        const std::string* path;
        const std::vector<std::string>* data_files;
        ModuleCore::LoadMode mode;
        StatusOr<std::unique_ptr<ModuleCore>> result;
      } ctx{&path, &data_files, mode, {}};

      rb_thread_call_without_gvl(
          [](void* p) -> void* {
            auto* c = static_cast<OpenCtx*>(p);
            c->result = ModuleCore::open_file(*c->path, *c->data_files, c->mode);
            return nullptr;
          },
          &ctx,
          RUBY_UBF_IO,
          nullptr);

      if (!ctx.result.ok()) {
        err_exc = status_to_ruby_exception(ctx.result.status);
      } else {
        obj = module_alloc(klass);
        RbModuleWrap* wrap = nullptr;
        TypedData_Get_Struct(obj, RbModuleWrap, &kModuleDataType, wrap);
        wrap->core = std::move(ctx.result.value);
        wrap->tracked_bytes = wrap->core->estimated_memory_bytes();
        if (wrap->tracked_bytes > 0) {
          rb_gc_adjust_memory_usage(static_cast<ssize_t>(wrap->tracked_bytes));
        }
      }
    }
  }

  RB_GC_GUARD(path_str);
  if (protect_state != 0) rb_jump_tag(protect_state);
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return obj;
}

// Module.from_bytes(str)
VALUE rb_module_s_from_bytes(VALUE klass, VALUE bytes_val) {
  Check_Type(bytes_val, T_STRING);
  VALUE err_exc = Qnil;
  VALUE obj = Qnil;

  {
    const auto* raw = reinterpret_cast<const uint8_t*>(RSTRING_PTR(bytes_val));
    const size_t len = static_cast<size_t>(RSTRING_LEN(bytes_val));
    std::vector<uint8_t> copy;
    try {
      copy.assign(raw, raw + len);
    } catch (const std::bad_alloc&) {
      err_exc = rb_exc_new_cstr(g_refs.eMemoryError, "out of memory copying .pte bytes");
    }

    if (err_exc == Qnil) {
      struct BytesCtx {
        std::vector<uint8_t>* bytes;
        StatusOr<std::unique_ptr<ModuleCore>> result;
      } ctx{&copy, {}};

      rb_thread_call_without_gvl(
          [](void* p) -> void* {
            auto* c = static_cast<BytesCtx*>(p);
            c->result = ModuleCore::open_bytes(std::move(*c->bytes));
            return nullptr;
          },
          &ctx,
          RUBY_UBF_IO,
          nullptr);

      if (!ctx.result.ok()) {
        err_exc = status_to_ruby_exception(ctx.result.status);
      } else {
        obj = module_alloc(klass);
        RbModuleWrap* wrap = nullptr;
        TypedData_Get_Struct(obj, RbModuleWrap, &kModuleDataType, wrap);
        wrap->core = std::move(ctx.result.value);
        wrap->tracked_bytes = wrap->core->estimated_memory_bytes();
        if (wrap->tracked_bytes > 0) {
          rb_gc_adjust_memory_usage(static_cast<ssize_t>(wrap->tracked_bytes));
        }
      }
    }
  }

  RB_GC_GUARD(bytes_val);
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return obj;
}

VALUE rb_module_source(VALUE self) {
  auto* wrap = get_module_wrap(self);
  std::string s = wrap->core->source_label();
  return rb_utf8_str_new(s.data(), static_cast<long>(s.size()));
}

VALUE rb_module_close(VALUE self) {
  RbModuleWrap* wrap = nullptr;
  TypedData_Get_Struct(self, RbModuleWrap, &kModuleDataType, wrap);
  if (wrap != nullptr && wrap->core && !wrap->core->is_closed()) {
    rb_thread_call_without_gvl(
        [](void* p) -> void* {
          static_cast<ModuleCore*>(p)->close();
          return nullptr;
        },
        wrap->core.get(),
        RUBY_UBF_IO,
        nullptr);
    if (wrap->tracked_bytes > 0) {
      rb_gc_adjust_memory_usage(-static_cast<ssize_t>(wrap->tracked_bytes));
      wrap->tracked_bytes = 0;
    }
  }
  return Qnil;
}

VALUE rb_module_closed_p(VALUE self) {
  RbModuleWrap* wrap = nullptr;
  TypedData_Get_Struct(self, RbModuleWrap, &kModuleDataType, wrap);
  return (wrap == nullptr || !wrap->core || wrap->core->is_closed()) ? Qtrue : Qfalse;
}

VALUE rb_module_method_names(VALUE self) {
  auto* wrap = get_module_wrap(self);
  VALUE err_exc = Qnil;
  VALUE out = Qnil;
  {
    auto res = wrap->core->method_names();
    if (!res.ok()) {
      err_exc = status_to_ruby_exception(res.status);
    } else {
      out = rb_ary_new_capa(static_cast<long>(res.value.size()));
      for (const auto& name : res.value) {
        rb_ary_push(out, rb_utf8_str_new(name.data(), static_cast<long>(name.size())));
      }
    }
  }
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return out;
}

VALUE rb_module_meta(int argc, VALUE* argv, VALUE self) {
  auto* wrap = get_module_wrap(self);
  VALUE name_val = Qnil;
  rb_scan_args(argc, argv, "01", &name_val);
  VALUE err_exc = Qnil;
  VALUE out = Qnil;
  {
    std::string mname = NIL_P(name_val) ? std::string("forward")
                                        : value_to_method_name(name_val);
    auto res = wrap->core->method_meta(mname);
    if (!res.ok()) {
      err_exc = status_to_ruby_exception(res.status);
    } else {
      out = build_method_meta(res.value);
    }
  }
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return out;
}

VALUE rb_module_load_method(int argc, VALUE* argv, VALUE self) {
  auto* wrap = get_module_wrap(self);
  VALUE name_val = Qnil;
  rb_scan_args(argc, argv, "01", &name_val);
  VALUE err_exc = Qnil;
  {
    std::string mname = NIL_P(name_val) ? std::string("forward")
                                        : value_to_method_name(name_val);
    struct LoadCtx {
      ModuleCore* core;
      const std::string* name;
      Status status;
    } ctx{wrap->core.get(), &mname, Status::Ok()};

    rb_thread_call_without_gvl(
        [](void* p) -> void* {
          auto* c = static_cast<LoadCtx*>(p);
          c->status = c->core->load_method(*c->name);
          return nullptr;
        },
        &ctx,
        RUBY_UBF_IO,
        nullptr);

    if (!ctx.status.ok()) {
      err_exc = status_to_ruby_exception(ctx.status);
    }
  }
  RB_GC_GUARD(self);
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return self;
}

VALUE rb_module_unload_method(int argc, VALUE* argv, VALUE self) {
  auto* wrap = get_module_wrap(self);
  VALUE name_val = Qnil;
  rb_scan_args(argc, argv, "01", &name_val);
  VALUE err_exc = Qnil;
  bool unloaded = false;
  {
    std::string mname = NIL_P(name_val) ? std::string("forward")
                                        : value_to_method_name(name_val);
    auto res = wrap->core->unload_method(mname);
    if (!res.ok()) {
      err_exc = status_to_ruby_exception(res.status);
    } else {
      unloaded = res.value;
    }
  }
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return unloaded ? Qtrue : Qfalse;
}

VALUE rb_module_method_loaded_p(int argc, VALUE* argv, VALUE self) {
  auto* wrap = get_module_wrap(self);
  VALUE name_val = Qnil;
  rb_scan_args(argc, argv, "01", &name_val);
  std::string mname = NIL_P(name_val) ? std::string("forward")
                                      : value_to_method_name(name_val);
  return wrap->core->is_method_loaded(mname) ? Qtrue : Qfalse;
}

// _execute(method_name, inputs_array, out_array_or_nil) -> Array of outputs
VALUE rb_module_execute_raw(VALUE self, VALUE name_val, VALUE inputs_ary, VALUE out_val) {
  auto* wrap = get_module_wrap(self);
  Check_Type(inputs_ary, T_ARRAY);

  VALUE err_exc = Qnil;
  VALUE result_ary = Qnil;
  int protect_state = 0;

  {
    std::string mname = value_to_method_name(name_val);
    std::vector<CoreValue> core_inputs;
    std::vector<executorch::extension::TensorPtr> out_targets;
    std::vector<VALUE> out_ruby_objs;
    std::vector<VALUE> locked_bufs;

    // 1. Convert inputs & out targets under the GVL.
    struct PrepArg {
      VALUE inputs_ary;
      VALUE out_val;
      std::vector<CoreValue>* core_inputs;
      std::vector<executorch::extension::TensorPtr>* out_targets;
      std::vector<VALUE>* out_ruby_objs;
      std::vector<VALUE>* locked_bufs;
      Status status;
    } prep{inputs_ary, out_val, &core_inputs, &out_targets, &out_ruby_objs, &locked_bufs, Status::Ok()};

    rb_protect(
        [](VALUE a) -> VALUE {
          auto* p = reinterpret_cast<PrepArg*>(a);
          long n_in = RARRAY_LEN(p->inputs_ary);
          p->core_inputs->reserve(static_cast<size_t>(n_in));
          for (long i = 0; i < n_in; ++i) {
            CoreValue cv;
            auto st = ruby_to_core_value(
                rb_ary_entry(p->inputs_ary, i), &cv, p->locked_bufs, false);
            if (!st.ok()) {
              p->status = std::move(st);
              return Qnil;
            }
            p->core_inputs->push_back(std::move(cv));
          }
          if (!NIL_P(p->out_val)) {
            if (RB_TYPE_P(p->out_val, T_ARRAY)) {
              long n_out = RARRAY_LEN(p->out_val);
              p->out_targets->reserve(static_cast<size_t>(n_out));
              p->out_ruby_objs->reserve(static_cast<size_t>(n_out));
              for (long i = 0; i < n_out; ++i) {
                VALUE item = rb_ary_entry(p->out_val, i);
                CoreValue cv;
                auto st = ruby_to_core_value(item, &cv, p->locked_bufs, true);
                if (!st.ok()) {
                  p->status = std::move(st);
                  return Qnil;
                }
                p->out_targets->push_back(std::move(cv.tensor_val));
                p->out_ruby_objs->push_back(item);
              }
            } else {
              CoreValue cv;
              auto st = ruby_to_core_value(p->out_val, &cv, p->locked_bufs, true);
              if (!st.ok()) {
                p->status = std::move(st);
                return Qnil;
              }
              p->out_targets->push_back(std::move(cv.tensor_val));
              p->out_ruby_objs->push_back(p->out_val);
            }
          }
          return Qnil;
        },
        reinterpret_cast<VALUE>(&prep),
        &protect_state);

    if (protect_state == 0 && !prep.status.ok()) {
      err_exc = status_to_ruby_exception(prep.status);
    }

    // 2. Lock any borrowed IO::Buffers before releasing GVL, and unlock all of
    //    them unconditionally immediately after (even if an exception occurs).
    if (protect_state == 0 && err_exc == Qnil) {
      std::vector<VALUE> actually_locked;
      actually_locked.reserve(locked_bufs.size());

      struct ExecCtx {
        ModuleCore* core;
        const std::string* method_name;
        const std::vector<CoreValue>* inputs;
        const std::vector<executorch::extension::TensorPtr>* out_targets;
        StatusOr<std::vector<CoreValue>> result;
      } ectx{wrap->core.get(), &mname, &core_inputs, &out_targets, {}};

      struct LockRunArg {
        std::vector<VALUE>* to_lock;
        std::vector<VALUE>* locked;
        ExecCtx* ectx;
      } lra{&locked_bufs, &actually_locked, &ectx};

      rb_protect(
          [](VALUE a) -> VALUE {
            auto* l = reinterpret_cast<LockRunArg*>(a);
            for (VALUE b : *l->to_lock) {
              // Avoid double-locking the exact same IO::Buffer object twice.
              bool seen = false;
              for (VALUE already : *l->locked) {
                if (already == b) { seen = true; break; }
              }
              if (!seen) {
                rb_io_buffer_lock(b);
                l->locked->push_back(b);
              }
            }
            rb_thread_call_without_gvl(
                [](void* p) -> void* {
                  auto* c = static_cast<ExecCtx*>(p);
                  c->result = c->core->execute(
                      *c->method_name, *c->inputs, *c->out_targets);
                  return nullptr;
                },
                l->ectx,
                RUBY_UBF_IO,
                nullptr);
            return Qnil;
          },
          reinterpret_cast<VALUE>(&lra),
          &protect_state);

      // Always unlock every locked IO::Buffer under the GVL.
      for (auto it = actually_locked.rbegin(); it != actually_locked.rend(); ++it) {
        rb_io_buffer_try_unlock(*it);
      }

      if (protect_state == 0) {
        if (!ectx.result.ok()) {
          err_exc = status_to_ruby_exception(ectx.result.status);
        } else if (!out_ruby_objs.empty()) {
          // Return the exact caller-provided out: Ruby Tensor objects!
          result_ary = rb_ary_new_capa(static_cast<long>(out_ruby_objs.size()));
          for (VALUE ro : out_ruby_objs) {
            rb_ary_push(result_ary, ro);
          }
        } else {
          struct WrapOutArg {
            std::vector<CoreValue>* vals;
          } woa{&ectx.result.value};
          result_ary = rb_protect(
              [](VALUE a) -> VALUE {
                auto* w = reinterpret_cast<WrapOutArg*>(a);
                VALUE ary = rb_ary_new_capa(static_cast<long>(w->vals->size()));
                for (auto& cv : *w->vals) {
                  rb_ary_push(ary, core_value_to_ruby(cv));
                }
                return ary;
              },
              reinterpret_cast<VALUE>(&woa),
              &protect_state);
        }
      }
    }
  }  // All C++ locals destroyed here before any rb_jump_tag / rb_exc_raise!

  RB_GC_GUARD(self);
  RB_GC_GUARD(inputs_ary);
  RB_GC_GUARD(out_val);
  if (protect_state != 0) rb_jump_tag(protect_state);
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return result_ary;
}

}  // namespace

const rb_data_type_t kModuleDataType = {
    "Executorch::Module",
    {
        nullptr,
        module_free,
        module_memsize,
        nullptr,
        {nullptr},
    },
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

void init_module(VALUE mExecutorch) {
  g_refs.id_mmap = rb_intern("mmap");
  g_refs.id_file = rb_intern("file");
  g_refs.id_mmap_mlock = rb_intern("mmap_mlock");
  g_refs.id_mmap_mlock_ignore_errors = rb_intern("mmap_mlock_ignore_errors");
  g_refs.id_mmap_madvise = rb_intern("mmap_madvise");
  g_refs.id_kw_load_mode = rb_intern("load_mode");
  g_refs.id_kw_data_files = rb_intern("data_files");
  g_refs.id_kw_out = rb_intern("out");

  g_refs.cMethodMeta = rb_const_get(mExecutorch, rb_intern("MethodMeta"));
  g_refs.cTensorInfo = rb_const_get(mExecutorch, rb_intern("TensorInfo"));
  rb_global_variable(&g_refs.cMethodMeta);
  rb_global_variable(&g_refs.cTensorInfo);

  g_refs.cModule = rb_define_class_under(mExecutorch, "Module", rb_cObject);
  rb_global_variable(&g_refs.cModule);
  rb_undef_alloc_func(g_refs.cModule);

  rb_define_singleton_method(g_refs.cModule, "_open_file", rb_module_s_open_file, -1);
  rb_define_singleton_method(g_refs.cModule, "from_bytes", rb_module_s_from_bytes, 1);

  rb_define_method(g_refs.cModule, "source", rb_module_source, 0);
  rb_define_method(g_refs.cModule, "close", rb_module_close, 0);
  rb_define_method(g_refs.cModule, "closed?", rb_module_closed_p, 0);
  rb_define_method(g_refs.cModule, "method_names", rb_module_method_names, 0);
  rb_define_method(g_refs.cModule, "meta", rb_module_meta, -1);
  rb_define_method(g_refs.cModule, "load_method", rb_module_load_method, -1);
  rb_define_method(g_refs.cModule, "unload_method", rb_module_unload_method, -1);
  rb_define_method(g_refs.cModule, "method_loaded?", rb_module_method_loaded_p, -1);
  rb_define_private_method(g_refs.cModule, "_execute", rb_module_execute_raw, 3);
}

}  // namespace etrb

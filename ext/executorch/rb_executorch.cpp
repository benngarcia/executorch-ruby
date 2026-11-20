#include "rb_internal.hpp"

namespace etrb {

RubyRefs g_refs;

namespace {

VALUE rb_et_backends(VALUE) {
  auto list = registered_backends();
  VALUE ary = rb_ary_new_capa(static_cast<long>(list.size()));
  for (const auto& name : list) {
    rb_ary_push(ary, rb_utf8_str_new(name.data(), static_cast<long>(name.size())));
  }
  return ary;
}

VALUE rb_et_kernel_count(VALUE) {
  return ULL2NUM(registered_kernel_count());
}

VALUE rb_et_num_threads(VALUE) {
  return LL2NUM(get_num_threads());
}

VALUE rb_et_set_num_threads(VALUE, VALUE val) {
  if (!RB_INTEGER_TYPE_P(val)) {
    rb_raise(rb_eTypeError, "num_threads must be an Integer");
  }
  int64_t n = NUM2LL(val);
  VALUE err_exc = Qnil;
  {
    auto st = set_num_threads(n);
    if (!st.ok()) {
      err_exc = status_to_ruby_exception(st);
    }
  }
  if (err_exc != Qnil) raise_status_no_live_cpp(err_exc);
  return val;
}

VALUE rb_et_executorch_version(VALUE) {
#ifdef EXECUTORCH_RUBY_SDK_VERSION
  return rb_utf8_str_new_cstr(EXECUTORCH_RUBY_SDK_VERSION);
#else
  return rb_utf8_str_new_cstr("unknown");
#endif
}

}  // namespace
}  // namespace etrb

extern "C" __attribute__((visibility("default"))) void Init_executorch(void) {
  etrb::init_runtime_once();

  VALUE mExecutorch = rb_define_module("Executorch");
  etrb::g_refs.mExecutorch = mExecutorch;
  rb_global_variable(&etrb::g_refs.mExecutorch);

  etrb::init_errors(mExecutorch);
  etrb::init_tensor(mExecutorch);
  etrb::init_module(mExecutorch);

  rb_define_singleton_method(mExecutorch, "backends", etrb::rb_et_backends, 0);
  rb_define_singleton_method(mExecutorch, "kernel_count", etrb::rb_et_kernel_count, 0);
  rb_define_singleton_method(mExecutorch, "num_threads", etrb::rb_et_num_threads, 0);
  rb_define_singleton_method(mExecutorch, "num_threads=", etrb::rb_et_set_num_threads, 1);
  rb_define_singleton_method(mExecutorch, "executorch_version", etrb::rb_et_executorch_version, 0);
}

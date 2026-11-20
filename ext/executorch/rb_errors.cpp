#include "rb_internal.hpp"

namespace etrb {

static VALUE rb_error_code(VALUE self) {
  return rb_ivar_get(self, g_refs.id_code);
}

static VALUE rb_error_pal_log(VALUE self) {
  return rb_ivar_get(self, g_refs.id_pal_log);
}

void init_errors(VALUE mExecutorch) {
  g_refs.id_code = rb_intern("@code");
  g_refs.id_pal_log = rb_intern("@pal_log");

  g_refs.eError =
      rb_define_class_under(mExecutorch, "Error", rb_eStandardError);
  rb_define_method(g_refs.eError, "code", rb_error_code, 0);
  rb_define_method(g_refs.eError, "pal_log", rb_error_pal_log, 0);

  g_refs.eLoadError =
      rb_define_class_under(mExecutorch, "LoadError", g_refs.eError);
  g_refs.eMethodNotFoundError =
      rb_define_class_under(mExecutorch, "MethodNotFoundError", g_refs.eError);
  g_refs.eOperatorMissingError =
      rb_define_class_under(mExecutorch, "OperatorMissingError", g_refs.eError);
  g_refs.eDelegateError =
      rb_define_class_under(mExecutorch, "DelegateError", g_refs.eError);
  g_refs.eShapeError =
      rb_define_class_under(mExecutorch, "ShapeError", g_refs.eError);
  g_refs.eDTypeError =
      rb_define_class_under(mExecutorch, "DTypeError", g_refs.eError);
  g_refs.eExecutionError =
      rb_define_class_under(mExecutorch, "ExecutionError", g_refs.eError);
  g_refs.eMemoryError =
      rb_define_class_under(mExecutorch, "MemoryError", g_refs.eError);
  g_refs.eClosedError =
      rb_define_class_under(mExecutorch, "ClosedError", g_refs.eError);
}

VALUE status_to_ruby_exception(const Status& st) {
  VALUE klass = g_refs.eExecutionError;
  switch (st.kind) {
    case ErrorKind::Load:
      klass = g_refs.eLoadError;
      break;
    case ErrorKind::MethodNotFound:
      klass = g_refs.eMethodNotFoundError;
      break;
    case ErrorKind::OperatorMissing:
      klass = g_refs.eOperatorMissingError;
      break;
    case ErrorKind::Delegate:
      klass = g_refs.eDelegateError;
      break;
    case ErrorKind::Shape:
      klass = g_refs.eShapeError;
      break;
    case ErrorKind::DType:
      klass = g_refs.eDTypeError;
      break;
    case ErrorKind::Memory:
      klass = g_refs.eMemoryError;
      break;
    case ErrorKind::Closed:
      klass = g_refs.eClosedError;
      break;
    case ErrorKind::Argument:
      klass = rb_eArgError;
      break;
    case ErrorKind::Type:
      klass = rb_eTypeError;
      break;
    case ErrorKind::Range:
      klass = rb_eRangeError;
      break;
    case ErrorKind::Execution:
    default:
      klass = g_refs.eExecutionError;
      break;
  }

  VALUE msg = rb_utf8_str_new(st.message.data(), static_cast<long>(st.message.size()));
  VALUE exc = rb_exc_new_str(klass, msg);
  if (rb_obj_is_kind_of(exc, g_refs.eError)) {
    rb_ivar_set(
        exc,
        g_refs.id_code,
        st.et_code == 0 ? Qnil : ULONG2NUM(static_cast<unsigned long>(st.et_code)));
    rb_ivar_set(
        exc,
        g_refs.id_pal_log,
        st.pal_log.empty()
            ? Qnil
            : rb_utf8_str_new(
                  st.pal_log.data(), static_cast<long>(st.pal_log.size())));
  }
  return exc;
}

[[noreturn]] void raise_status_no_live_cpp(VALUE exc) {
  rb_exc_raise(exc);
}

}  // namespace etrb

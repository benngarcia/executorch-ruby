# frozen_string_literal: true

require_relative "test_helper"
require "rbconfig"

class RuntimeTest < Minitest::Test
  def test_version_constants
    assert_equal "1.0.0", Executorch::VERSION
    assert_match(/\A\d+\.\d+\.\d+/, Executorch.executorch_version)
  end

  def test_xnnpack_backend_and_kernels_registered
    assert_includes Executorch.backends, "XnnpackBackend"
    assert_operator Executorch.kernel_count, :>, 50
  end

  def test_num_threads_get_and_set
    orig = Executorch.num_threads
    assert_kind_of Integer, orig
    assert_operator orig, :>=, 1

    Executorch.num_threads = 2
    assert_equal 2, Executorch.num_threads
  ensure
    Executorch.num_threads = orig if orig && orig >= 1
  end

  def test_num_threads_validation
    assert_raises(ArgumentError) { Executorch.num_threads = 0 }
    assert_raises(ArgumentError) { Executorch.num_threads = -4 }
    assert_raises(ArgumentError) { Executorch.num_threads = 10_000 }
    assert_raises(TypeError) { Executorch.num_threads = "2" }
  end

  def test_error_hierarchy
    assert_operator Executorch::Error, :<, StandardError
    %i[
      LoadError
      MethodNotFoundError
      OperatorMissingError
      DelegateError
      ShapeError
      DTypeError
      ExecutionError
      MemoryError
      ClosedError
    ].each do |name|
      klass = Executorch.const_get(name)
      assert_operator klass, :<, Executorch::Error, "#{name} must inherit from Executorch::Error"
    end
  end

  def test_native_extension_exports_only_init_executorch
    dlext = RbConfig::CONFIG["DLEXT"]
    ext_path = File.expand_path("../lib/executorch/executorch.#{dlext}", __dir__)
    assert File.exist?(ext_path), "expected compiled extension at #{ext_path}"

    case RbConfig::CONFIG["target_os"]
    when /linux/i
      out = `nm -D --defined-only #{ext_path} 2>/dev/null`
      assert $?.success?, "nm -D failed on #{ext_path}"
      syms = out.lines.map { |l| l.split.last }.compact
      assert_equal ["Init_executorch"], syms, "Unexpected exported symbols leak: #{syms.inspect}"
    when /darwin/i
      out = `nm -gU #{ext_path} 2>/dev/null`
      assert $?.success?, "nm -gU failed on #{ext_path}"
      syms = out.lines.map { |l| l.split.last }.compact
      assert_equal ["_Init_executorch"], syms, "Unexpected exported symbols leak: #{syms.inspect}"
    else
      flunk "Unsupported target_os for symbol check: #{RbConfig::CONFIG['target_os']}"
    end
  end
end

# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "executorch"
require "json"
require "pathname"
require "minitest/autorun"

module TestHelper
  FIXTURES_DIR = File.expand_path("fixtures/models", __dir__)
  MANIFEST_PATH = File.join(FIXTURES_DIR, "manifest.json")

  def fixtures_manifest
    @fixtures_manifest ||= begin
      unless File.exist?(MANIFEST_PATH)
        raise "Missing test fixtures at #{MANIFEST_PATH}. Run `bundle exec rake fixtures:export` first."
      end

      JSON.parse(File.read(MANIFEST_PATH))
    end
  end

  def model_spec(name)
    fixtures_manifest.fetch("models").fetch(name.to_s)
  end

  def fixture_path(rel)
    File.join(FIXTURES_DIR, rel)
  end

  def tensor_from_golden(desc)
    dtype = desc.fetch("dtype").to_sym
    Executorch::Tensor.new(desc.fetch("values"), shape: desc.fetch("shape"), dtype: dtype)
  end

  def assert_tensor_matches_golden(actual, golden, rtol: 1e-3, atol: 1e-4)
    assert_instance_of Executorch::Tensor, actual
    assert_equal golden.fetch("shape"), actual.shape, "shape mismatch"
    assert_equal golden.fetch("dtype").to_sym, actual.dtype, "dtype mismatch"

    got = actual.flat_to_a
    exp = golden.fetch("values")
    assert_equal exp.size, got.size, "element count mismatch"

    exp.each_with_index do |e, i|
      g = got[i]
      if e == true || e == false
        assert_equal e, g, "bool mismatch at index #{i}"
      elsif e.is_a?(Integer) && actual.dtype != :float32 && actual.dtype != :float64 &&
            actual.dtype != :float16 && actual.dtype != :bfloat16
        assert_equal e, g, "int mismatch at index #{i}"
      else
        tol = atol + rtol * e.abs
        assert_in_delta e, g, tol, "value mismatch at index #{i}: expected #{e}, got #{g}"
      end
    end
  end
end

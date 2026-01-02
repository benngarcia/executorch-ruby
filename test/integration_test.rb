# frozen_string_literal: true

require "test_helper"

class IntegrationTest < Minitest::Test
  include TestHelper

  # Full workflow test: load model -> check methods -> run inference
  def test_full_workflow_with_simple_model
    skip_without_models

    # Load model
    model = Executorch::Model.new(model_path("simple.pte"))
    assert model.loaded?

    # Check methods
    assert_includes model.method_names, "forward"

    # Run inference - simple.pte does x * 2 + 1
    input = Executorch::Tensor.new([1.0, 2.0, 3.0], shape: [1, 3], dtype: :float)
    outputs = model.forward([input])

    assert_kind_of Array, outputs
    assert_equal 1, outputs.size

    output = outputs.first
    assert_kind_of Executorch::Tensor, output
    assert_equal [1, 3], output.shape

    # Verify: [1.0, 2.0, 3.0] * 2 + 1 = [3.0, 5.0, 7.0]
    # Output shape is [1, 3], so to_a returns [[3.0, 5.0, 7.0]]
    expected = [[3.0, 5.0, 7.0]]
    assert_equal expected.size, output.to_a.size
    expected.first.zip(output.to_a.first).each do |e, a|
      assert_in_delta e, a, 0.001
    end
  end

  def test_predict_workflow
    skip_without_models

    model = Executorch::Model.new(model_path("simple.pte"))
    input = Executorch::Tensor.new([1.0, 2.0, 3.0], shape: [1, 3], dtype: :float)

    # Use predict instead of forward
    outputs = model.predict([input])

    assert_kind_of Array, outputs
    output = outputs.first
    assert_kind_of Executorch::Tensor, output

    # Output shape is [1, 3], so to_a returns [[3.0, 5.0, 7.0]]
    expected = [[3.0, 5.0, 7.0]]
    assert_equal expected.size, output.to_a.size
    expected.first.zip(output.to_a.first).each do |e, a|
      assert_in_delta e, a, 0.001
    end
  end

  def test_error_handling_invalid_model
    # Verify proper error messages for common failure cases
    error = assert_raises(RuntimeError) do
      Executorch::Model.new("/definitely/not/a/real/path.pte")
    end

    assert_match(/error/i, error.message)
  end

  def test_version_constant
    assert_kind_of String, Executorch::VERSION
    assert_match(/\d+\.\d+\.\d+/, Executorch::VERSION)
  end

  def test_native_version_constant
    assert_kind_of String, Executorch::NATIVE_VERSION
  end
end

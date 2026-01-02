# frozen_string_literal: true

require "test_helper"

class ModelTest < Minitest::Test
  include TestHelper

  # Tests for Model initialization
  def test_model_requires_path
    assert_raises(ArgumentError) do
      Executorch::Model.new
    end
  end

  def test_model_raises_on_invalid_path
    assert_raises(RuntimeError) do
      Executorch::Model.new("/nonexistent/path/model.pte")
    end
  end

  def test_model_stores_path
    skip_without_models
    model = Executorch::Model.new(model_path("simple.pte"))
    assert_equal model_path("simple.pte"), model.path
  end

  def test_model_is_loaded_after_construction
    skip_without_models
    model = Executorch::Model.new(model_path("simple.pte"))
    assert model.loaded?
  end

  def test_model_method_names_returns_array
    skip_without_models
    model = Executorch::Model.new(model_path("simple.pte"))
    names = model.method_names
    assert_kind_of Array, names
    assert names.all? { |n| n.is_a?(String) }
  end

  def test_model_has_forward_method
    skip_without_models
    model = Executorch::Model.new(model_path("simple.pte"))
    assert_includes model.method_names, "forward"
  end

  def test_model_predict_is_alias_for_forward
    skip_without_models
    model = Executorch::Model.new(model_path("simple.pte"))
    input = Executorch::Tensor.new([1.0, 2.0, 3.0], shape: [1, 3], dtype: :float)

    forward_result = model.forward([input])
    predict_result = model.predict([input])

    assert_equal forward_result.first.to_a, predict_result.first.to_a
  end

  def test_model_is_callable
    skip_without_models
    model = Executorch::Model.new(model_path("simple.pte"))
    input = Executorch::Tensor.new([1.0, 2.0, 3.0], shape: [1, 3], dtype: :float)

    forward_result = model.forward([input])
    call_result = model.call([input])

    assert_equal forward_result.first.to_a, call_result.first.to_a
  end
end

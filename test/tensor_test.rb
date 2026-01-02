# frozen_string_literal: true

require "test_helper"

class TensorTest < Minitest::Test
  # ===========================================================================
  # Tests for nested array INPUT (Tensor.new with shape inference)
  # ===========================================================================

  # --- Basic nested array creation ---

  def test_create_1d_tensor_from_flat_array
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0])
    assert_equal [3], tensor.shape
    assert_equal 3, tensor.numel
  end

  def test_create_2d_tensor_from_nested_array
    tensor = Executorch::Tensor.new([[1.0, 2.0], [3.0, 4.0]])
    assert_equal [2, 2], tensor.shape
    assert_equal 4, tensor.numel
  end

  def test_create_3d_tensor_from_nested_array
    data = [
      [[1.0, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]]
    ]
    tensor = Executorch::Tensor.new(data)
    assert_equal [2, 2, 2], tensor.shape
    assert_equal 8, tensor.numel
  end

  def test_create_4d_tensor_from_nested_array
    # Shape [2, 1, 2, 3]
    data = [
      [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
      [[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    ]
    tensor = Executorch::Tensor.new(data)
    assert_equal [2, 1, 2, 3], tensor.shape
    assert_equal 12, tensor.numel
  end

  # --- Non-square arrays ---

  def test_create_2d_non_square_tensor
    tensor = Executorch::Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_equal [2, 3], tensor.shape
  end

  def test_create_2d_tall_tensor
    tensor = Executorch::Tensor.new([[1.0], [2.0], [3.0], [4.0]])
    assert_equal [4, 1], tensor.shape
  end

  # --- Single element tensors ---

  def test_create_single_element_1d
    tensor = Executorch::Tensor.new([42.0])
    assert_equal [1], tensor.shape
    assert_equal 1, tensor.numel
  end

  def test_create_single_element_2d
    tensor = Executorch::Tensor.new([[42.0]])
    assert_equal [1, 1], tensor.shape
    assert_equal 1, tensor.numel
  end

  def test_create_single_element_3d
    tensor = Executorch::Tensor.new([[[42.0]]])
    assert_equal [1, 1, 1], tensor.shape
    assert_equal 1, tensor.numel
  end

  def test_create_single_element_4d
    tensor = Executorch::Tensor.new([[[[42.0]]]])
    assert_equal [1, 1, 1, 1], tensor.shape
    assert_equal 1, tensor.numel
  end

  # --- Empty arrays ---

  def test_create_empty_1d_tensor
    tensor = Executorch::Tensor.new([])
    assert_equal [0], tensor.shape
    assert_equal 0, tensor.numel
  end

  def test_create_empty_2d_tensor
    tensor = Executorch::Tensor.new([[], []])
    assert_equal [2, 0], tensor.shape
    assert_equal 0, tensor.numel
  end

  # --- Data types ---

  def test_infer_float_dtype_from_floats
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0])
    assert_equal :float, tensor.dtype
  end

  def test_infer_float_dtype_from_integers
    # Integers should be coerced to float by default
    tensor = Executorch::Tensor.new([1, 2, 3])
    assert_equal :float, tensor.dtype
  end

  def test_explicit_dtype_int
    tensor = Executorch::Tensor.new([1, 2, 3], dtype: :int)
    assert_equal :int, tensor.dtype
  end

  def test_explicit_dtype_long
    tensor = Executorch::Tensor.new([1, 2, 3], dtype: :long)
    assert_equal :long, tensor.dtype
  end

  def test_explicit_dtype_double
    tensor = Executorch::Tensor.new([1.0, 2.0], dtype: :double)
    assert_equal :double, tensor.dtype
  end

  def test_nested_array_with_explicit_dtype
    tensor = Executorch::Tensor.new([[1, 2], [3, 4]], dtype: :long)
    assert_equal [2, 2], tensor.shape
    assert_equal :long, tensor.dtype
  end

  # --- Error cases: Jagged arrays ---

  def test_jagged_array_raises_error
    assert_raises(ArgumentError) do
      Executorch::Tensor.new([[1, 2], [3]])
    end
  end

  def test_jagged_array_3d_raises_error
    assert_raises(ArgumentError) do
      Executorch::Tensor.new([[[1, 2], [3, 4]], [[5], [6]]])
    end
  end

  def test_deeply_jagged_array_raises_error
    assert_raises(ArgumentError) do
      Executorch::Tensor.new([[[1, 2]], [[3]]])
    end
  end

  def test_mixed_nesting_raises_error
    # First element is scalar, second is array
    assert_raises(ArgumentError) do
      Executorch::Tensor.new([1, [2, 3]])
    end
  end

  def test_inconsistent_nesting_depth_raises_error
    assert_raises(ArgumentError) do
      Executorch::Tensor.new([[1, 2], [[3, 4]]])
    end
  end

  # --- Backward compatibility: explicit shape still works ---

  def test_explicit_shape_with_flat_array
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [2, 2], dtype: :float)
    assert_equal [2, 2], tensor.shape
    assert_equal :float, tensor.dtype
  end

  def test_explicit_shape_overrides_nesting
    # Even if data looks nested, explicit shape takes precedence
    # This tests backward compatibility
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [4], dtype: :float)
    assert_equal [4], tensor.shape
  end

  # ===========================================================================
  # Tests for nested array OUTPUT (to_a returns nested structure)
  # ===========================================================================

  # --- Basic reshaping ---

  def test_to_a_1d_returns_flat_array
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0], shape: [3], dtype: :float)
    assert_equal [1.0, 2.0, 3.0], tensor.to_a
  end

  def test_to_a_2d_returns_nested_array
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [2, 2], dtype: :float)
    expected = [[1.0, 2.0], [3.0, 4.0]]
    assert_equal expected, tensor.to_a
  end

  def test_to_a_2d_non_square
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3], dtype: :float)
    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert_equal expected, tensor.to_a
  end

  def test_to_a_2d_tall
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [4, 1], dtype: :float)
    expected = [[1.0], [2.0], [3.0], [4.0]]
    assert_equal expected, tensor.to_a
  end

  def test_to_a_3d_returns_nested_array
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    tensor = Executorch::Tensor.new(data, shape: [2, 2, 2], dtype: :float)
    expected = [
      [[1.0, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]]
    ]
    assert_equal expected, tensor.to_a
  end

  def test_to_a_4d_returns_nested_array
    data = (1..24).map(&:to_f)
    tensor = Executorch::Tensor.new(data, shape: [2, 3, 2, 2], dtype: :float)
    result = tensor.to_a

    assert_equal 2, result.size
    assert_equal 3, result[0].size
    assert_equal 2, result[0][0].size
    assert_equal 2, result[0][0][0].size
    assert_equal 1.0, result[0][0][0][0]
    assert_equal 24.0, result[1][2][1][1]
  end

  # --- Single element tensors ---

  def test_to_a_single_element_1d
    tensor = Executorch::Tensor.new([42.0], shape: [1], dtype: :float)
    assert_equal [42.0], tensor.to_a
  end

  def test_to_a_single_element_2d
    tensor = Executorch::Tensor.new([42.0], shape: [1, 1], dtype: :float)
    assert_equal [[42.0]], tensor.to_a
  end

  def test_to_a_single_element_3d
    tensor = Executorch::Tensor.new([42.0], shape: [1, 1, 1], dtype: :float)
    assert_equal [[[42.0]]], tensor.to_a
  end

  # --- Empty tensors ---

  def test_to_a_empty_1d
    tensor = Executorch::Tensor.new([], shape: [0], dtype: :float)
    assert_equal [], tensor.to_a
  end

  def test_to_a_empty_2d
    tensor = Executorch::Tensor.new([], shape: [2, 0], dtype: :float)
    assert_equal [[], []], tensor.to_a
  end

  def test_to_a_empty_3d
    tensor = Executorch::Tensor.new([], shape: [2, 3, 0], dtype: :float)
    expected = [[[], [], []], [[], [], []]]
    assert_equal expected, tensor.to_a
  end

  # --- Round-trip tests ---

  def test_roundtrip_2d_nested_array
    original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor = Executorch::Tensor.new(original)
    assert_equal original, tensor.to_a
  end

  def test_roundtrip_3d_nested_array
    original = [
      [[1.0, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]]
    ]
    tensor = Executorch::Tensor.new(original)
    assert_equal original, tensor.to_a
  end

  def test_roundtrip_non_square
    original = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    tensor = Executorch::Tensor.new(original)
    assert_equal [3, 4], tensor.shape
    assert_equal original, tensor.to_a
  end

  def test_roundtrip_with_integers_and_long_dtype
    original = [[1, 2], [3, 4]]
    tensor = Executorch::Tensor.new(original, dtype: :long)
    result = tensor.to_a
    assert_equal [[1, 2], [3, 4]], result
  end

  # --- Data type preservation in output ---

  def test_to_a_preserves_float_values
    tensor = Executorch::Tensor.new([[1.5, 2.5], [3.5, 4.5]])
    result = tensor.to_a
    assert_in_delta 1.5, result[0][0], 0.001
    assert_in_delta 4.5, result[1][1], 0.001
  end

  def test_to_a_preserves_integer_values_with_long_dtype
    tensor = Executorch::Tensor.new([[100, 200], [300, 400]], dtype: :long)
    result = tensor.to_a
    assert_equal 100, result[0][0]
    assert_equal 400, result[1][1]
  end

  # --- Edge case: very deep nesting ---

  def test_5d_tensor_roundtrip
    # Shape [2, 1, 2, 1, 3]
    original = [
      [[[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]],
      [[[[7.0, 8.0, 9.0]], [[10.0, 11.0, 12.0]]]]
    ]
    tensor = Executorch::Tensor.new(original)
    assert_equal [2, 1, 2, 1, 3], tensor.shape
    assert_equal original, tensor.to_a
  end

  # ===========================================================================
  # Tests for flat_to_a (backward compatibility method)
  # ===========================================================================

  def test_flat_to_a_returns_flat_array
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [2, 2], dtype: :float)
    assert_equal [1.0, 2.0, 3.0, 4.0], tensor.flat_to_a
  end

  def test_flat_to_a_for_3d_tensor
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    tensor = Executorch::Tensor.new(data, shape: [2, 2, 2], dtype: :float)
    assert_equal data, tensor.flat_to_a
  end
end

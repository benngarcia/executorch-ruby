# frozen_string_literal: true

require "test_helper"

# Tests for the raw-binary tensor path (Tensor.from_bytes / Tensor#to_binary),
# which skips per-element conversion and memcpys the buffer instead.
class BinaryTensorTest < Minitest::Test
  def test_from_bytes_builds_expected_tensor
    tensor = Executorch::Tensor.from_bytes([1.0, 2.0, 3.0, 4.0].pack("f*"), shape: [2, 2])

    assert_equal [2, 2], tensor.shape
    assert_equal :float, tensor.dtype
    assert_equal [[1.0, 2.0], [3.0, 4.0]], tensor.to_a
  end

  def test_from_bytes_matches_array_constructor
    values = Array.new(64) { |i| i * 0.5 }

    from_array = Executorch::Tensor.new(values, shape: [8, 8])
    from_bytes = Executorch::Tensor.from_bytes(values.pack("f*"), shape: [8, 8])

    assert_equal from_array.flat_to_a, from_bytes.flat_to_a
    assert_equal from_array.shape, from_bytes.shape
  end

  def test_to_binary_round_trips
    values = Array.new(32) { |i| (i - 16) * 1.25 }
    tensor = Executorch::Tensor.new(values, shape: [4, 8])

    round_tripped = Executorch::Tensor.from_bytes(tensor.to_binary, shape: [4, 8])

    assert_equal values, round_tripped.flat_to_a
  end

  def test_to_binary_length_matches_element_width
    tensor = Executorch::Tensor.new([1.0, 2.0, 3.0], shape: [3])
    assert_equal 3 * 4, tensor.to_binary.bytesize
  end

  def test_double_dtype_round_trips
    values = [1.5, -2.25, 3.125]
    tensor = Executorch::Tensor.from_bytes(values.pack("d*"), shape: [3], dtype: :double)

    assert_equal :double, tensor.dtype
    assert_equal values, tensor.flat_to_a
    assert_equal 3 * 8, tensor.to_binary.bytesize
  end

  def test_long_dtype_round_trips
    values = [1, -2, 3_000_000_000]
    tensor = Executorch::Tensor.from_bytes(values.pack("q*"), shape: [3], dtype: :long)

    assert_equal :long, tensor.dtype
    assert_equal values, tensor.flat_to_a
  end

  def test_int_dtype_round_trips
    values = [1, -2, 3]
    tensor = Executorch::Tensor.from_bytes(values.pack("l*"), shape: [3], dtype: :int)

    assert_equal :int, tensor.dtype
    assert_equal values, tensor.flat_to_a
  end

  def test_too_few_bytes_raises
    assert_raises(ArgumentError) do
      Executorch::Tensor.from_bytes([1.0, 2.0].pack("f*"), shape: [2, 2])
    end
  end

  def test_too_many_bytes_raises
    assert_raises(ArgumentError) do
      Executorch::Tensor.from_bytes([1.0, 2.0, 3.0, 4.0, 5.0].pack("f*"), shape: [2, 2])
    end
  end

  def test_wrong_element_width_raises
    # Doubles packed, but read as floats: twice the bytes the shape calls for.
    assert_raises(ArgumentError) do
      Executorch::Tensor.from_bytes([1.0, 2.0].pack("d*"), shape: [2], dtype: :float)
    end
  end

  def test_binary_tensor_runs_through_a_model
    skip "Test models not available. See test/support/README.md" unless models_available?

    model = Executorch::Model.new(model_path("simple.pte"))
    tensor = Executorch::Tensor.from_bytes([1.0, 2.0, 3.0].pack("f*"), shape: [1, 3])

    output = model.predict([tensor]).first

    # simple.pte computes x * 2 + 1
    assert_equal [3.0, 5.0, 7.0], output.flat_to_a
    assert_equal [3.0, 5.0, 7.0], output.to_binary.unpack("f*")
  end

  include TestHelper
end

# frozen_string_literal: true

require_relative "test_helper"

class TensorTest < Minitest::Test
  ALL_DTYPES = %i[
    float32 float64 float16 bfloat16
    uint8 int8 int16 uint16 int32 uint32 int64 uint64
    bool
  ].freeze

  def test_all_supported_dtypes_round_trip
    ALL_DTYPES.each do |dtype|
      input = dtype == :bool ? [true, false, true, false] : [0, 1, 2, 3]
      t = Executorch::Tensor.new(input, shape: [2, 2], dtype: dtype)

      assert_equal dtype, t.dtype
      assert_equal [2, 2], t.shape
      assert_equal 2, t.dim
      assert_equal 2, t.ndim
      assert_equal 4, t.numel
      assert_predicate t, :contiguous?
      assert_equal [2, 1], t.strides
      assert_equal [0, 1], t.dim_order

      # Nested to_a
      if dtype == :bool
        assert_equal [[true, false], [true, false]], t.to_a
      elsif %i[float32 float64 float16 bfloat16].include?(dtype)
        assert_equal [[0.0, 1.0], [2.0, 3.0]], t.to_a
      else
        assert_equal [[0, 1], [2, 3]], t.to_a
      end

      # Binary round-trip
      bytes = t.to_bytes
      assert_equal Encoding::ASCII_8BIT, bytes.encoding
      assert_equal t.nbytes, bytes.bytesize

      t2 = Executorch::Tensor.from_bytes(bytes, shape: [2, 2], dtype: dtype)
      assert_equal t, t2
      assert_equal t.to_a, t2.to_a

      # IO::Buffer round-trip
      buf = t.to_buffer
      assert_instance_of IO::Buffer, buf
      assert_equal t.nbytes, buf.size
      t3 = Executorch::Tensor.from_buffer(buf, shape: [2, 2], dtype: dtype)
      assert_equal t, t3
      assert_equal t.to_a, t3.to_a
    end
  end

  def test_dtype_aliases_map_to_canonical_symbols
    aliases = {
      float: :float32,
      double: :float64,
      half: :float16,
      bf16: :bfloat16,
      byte: :uint8,
      char: :int8,
      short: :int16,
      int: :int32,
      long: :int64,
    }
    aliases.each do |alias_sym, canonical|
      t = Executorch::Tensor.new([1, 2], dtype: alias_sym)
      assert_equal canonical, t.dtype
    end
  end

  def test_nested_array_shape_inference_and_explicit_nested_reshape
    t = Executorch::Tensor.new([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    assert_equal [2, 2, 2], t.shape
    assert_equal :float32, t.dtype
    assert_equal [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], t.to_a
    assert_equal [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], t.flat_to_a

    # Explicit shape with nested array now works cleanly
    t_flat = Executorch::Tensor.new([[1.0, 2.0], [3.0, 4.0]], shape: [4])
    assert_equal [4], t_flat.shape
    assert_equal [1.0, 2.0, 3.0, 4.0], t_flat.to_a
  end

  def test_zero_dim_scalar_tensor
    t = Executorch::Tensor.new(42.5, dtype: :float32)
    assert_equal [], t.shape
    assert_equal 0, t.dim
    assert_equal 1, t.numel
    assert_equal 4, t.nbytes
    assert_in_delta 42.5, t.item, 1e-5
    assert_in_delta 42.5, t.to_a, 1e-5

    t_b = Executorch::Tensor.from_bytes([42.5].pack("f"), shape: [], dtype: :float32)
    assert_equal t, t_b
    assert_in_delta 42.5, t_b.item, 1e-5

    t_bool = Executorch::Tensor.new(true, dtype: :bool)
    assert_equal [], t_bool.shape
    assert_equal true, t_bool.item
    assert_equal true, t_bool.to_a
  end

  def test_empty_tensors
    t0 = Executorch::Tensor.new([], dtype: :float32)
    assert_equal [0], t0.shape
    assert_equal 0, t0.numel
    assert_equal 0, t0.nbytes
    assert_predicate t0, :empty?
    assert_equal [], t0.to_a
    assert_equal "".b, t0.to_bytes

    t20 = Executorch::Tensor.new([[], []], dtype: :int32)
    assert_equal [2, 0], t20.shape
    assert_equal 0, t20.numel
    assert_equal [[], []], t20.to_a

    tb = Executorch::Tensor.from_bytes("", shape: [0], dtype: :float32)
    assert_equal t0, tb
  end

  def test_factory_constructors_zeros_ones_full
    z = Executorch::Tensor.zeros([2, 3], dtype: :float32)
    assert_equal [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], z.to_a

    o = Executorch::Tensor.ones([2, 2], dtype: :int64)
    assert_equal [[1, 1], [1, 1]], o.to_a

    f = Executorch::Tensor.full([2], 7, dtype: :uint8)
    assert_equal [7, 7], f.to_a
  end

  def test_zero_copy_io_buffer_reflects_mutations_and_detects_resize
    buf = IO::Buffer.new(16)
    buf.set_string([1.0, 2.0, 3.0, 4.0].pack("f*"))
    t = Executorch::Tensor.from_buffer(buf, shape: [2, 2], dtype: :float32)
    assert_equal [[1.0, 2.0], [3.0, 4.0]], t.to_a

    # Mutate underlying IO::Buffer in place — zero-copy view reflects it immediately
    buf.set_value(:f32, 0, 99.0)
    assert_equal [[99.0, 2.0], [3.0, 4.0]], t.to_a

    # Resizing underlying buffer is caught safely instead of UAF
    buf.resize(8)
    assert_raises(Executorch::ShapeError) { t.to_a }
  end

  def test_equality_inspect_and_close
    a = Executorch::Tensor.new([1, 2], dtype: :int32)
    b = Executorch::Tensor.new([1, 2], dtype: :int32)
    c = Executorch::Tensor.new([1, 2], dtype: :int64)
    d = Executorch::Tensor.new([1, 3], dtype: :int32)

    assert_equal a, b
    refute_equal a, c
    refute_equal a, d
    refute_equal a, [1, 2]

    assert_includes a.inspect, "Executorch::Tensor"
    assert_includes a.inspect, "dtype=:int32"
    refute_predicate a, :closed?

    a.close
    assert_predicate a, :closed?
    assert_includes a.inspect, "closed"
    assert_raises(Executorch::ClosedError) { a.to_a }
    assert_raises(Executorch::ClosedError) { a.shape }
    # Idempotent close
    a.close
  end
end

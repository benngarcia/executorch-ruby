# frozen_string_literal: true

module Executorch
  class Tensor
    class << self
      # Create a tensor filled with ones.
      def ones(shape, dtype: :float32)
        full(shape, 1, dtype: dtype)
      end

      # Create a tensor filled with a constant scalar value.
      def full(shape, value, dtype: :float32)
        raise TypeError, "shape must be an Array" unless shape.is_a?(Array)

        numel = shape.empty? ? 1 : shape.reduce(1, :*)
        raise ShapeError, "shape dimensions must be non-negative" if shape.any? { |d| !d.is_a?(Integer) || d < 0 }

        new(Array.new(numel, value), shape: shape, dtype: dtype)
      end

      # Build a Tensor from a Numo::NArray (zero-copy byte path).
      def from_numo(narray)
        dtype = NUMO_TO_DTYPE[narray.class.name]
        raise DTypeError, "Unsupported Numo::NArray type: #{narray.class.name}" unless dtype

        from_bytes(narray.to_binary, shape: narray.shape, dtype: dtype)
      end

      # Build a Tensor from a Torch::Tensor (torch.rb interop).
      def from_torch(torch_tensor)
        dtype = TORCH_TO_DTYPE[torch_tensor.dtype]
        raise DTypeError, "Unsupported Torch::Tensor dtype: #{torch_tensor.dtype.inspect}" unless dtype

        new(torch_tensor.contiguous.to_a, shape: torch_tensor.shape, dtype: dtype)
      end
    end

    NUMO_TO_DTYPE = {
      "Numo::SFloat" => :float32,
      "Numo::DFloat" => :float64,
      "Numo::Int8" => :int8,
      "Numo::Int16" => :int16,
      "Numo::Int32" => :int32,
      "Numo::Int64" => :int64,
      "Numo::UInt8" => :uint8,
      "Numo::UInt16" => :uint16,
      "Numo::UInt32" => :uint32,
      "Numo::UInt64" => :uint64,
    }.freeze

    TORCH_TO_DTYPE = {
      float32: :float32,
      float64: :float64,
      float16: :float16,
      bfloat16: :bfloat16,
      int8: :int8,
      int16: :int16,
      int32: :int32,
      int64: :int64,
      uint8: :uint8,
      bool: :bool,
    }.freeze

    def empty?
      numel == 0
    end

    def to_numo
      raise LoadError, "Numo::NArray is not loaded (require 'numo/narray')" unless defined?(::Numo::NArray)

      klass_name = NUMO_TO_DTYPE.key(dtype)
      raise DTypeError, "No Numo::NArray equivalent for dtype #{dtype.inspect}" unless klass_name

      klass = Object.const_get(klass_name)
      klass.from_binary(to_bytes, shape)
    end

    def inspect
      return "#<Executorch::Tensor (closed)>" if closed?

      "#<Executorch::Tensor shape=#{shape.inspect} dtype=#{dtype.inspect} nbytes=#{nbytes}>"
    end
    alias to_s inspect
  end
end

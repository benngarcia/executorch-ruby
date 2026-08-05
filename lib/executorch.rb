# frozen_string_literal: true

require_relative "executorch/version"

# Load the native extension
begin
  require_relative "executorch/executorch"
rescue LoadError => e
  warn "Failed to load ExecuTorch native extension: #{e.message}"
  warn "Make sure to run: bundle exec rake compile"
  raise
end

module Executorch
  # Extend Tensor with Ruby-friendly methods
  class Tensor
    class << self
      # Create a new tensor from data
      #
      # @param data [Array] Array of numeric values (can be nested or flat)
      # @param shape [Array<Integer>, nil] Shape of the tensor. If nil, inferred from data structure.
      # @param dtype [Symbol] Data type (:float, :double, :int, :long)
      # @return [Tensor]
      #
      # @example Create from nested array (shape inferred)
      #   Tensor.new([[1.0, 2.0], [3.0, 4.0]])  # shape: [2, 2]
      #
      # @example Create from flat array with explicit shape
      #   Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [2, 2])
      #
      def new(data, shape: nil, dtype: :float)
        if shape.nil?
          # Infer shape from nested array structure
          shape = infer_shape(data)
          flat_data = flatten_nested(data, shape)
        else
          # Use flat data directly (backward compatibility)
          flat_data = data
        end

        create(flat_data, shape, dtype)
      end

      # Build a tensor from packed binary data, skipping per-element conversion.
      #
      # Every other constructor walks the Array and converts each element
      # individually. This one hands the runtime a buffer to memcpy, which is
      # dramatically faster once tensors get large -- worth it whenever the data
      # is already bytes (an image decoded to a string, a file, a socket) or
      # when you can pack it once and reuse it.
      #
      # Data must be native-endian and match the element width of dtype:
      # :float => "f*", :double => "d*", :int => "l*", :long => "q*".
      #
      # @example
      #   bytes = pixels.pack("f*")
      #   Executorch::Tensor.from_bytes(bytes, shape: [1, 3, 224, 224])
      #
      def from_bytes(data, shape:, dtype: :float)
        from_binary(data, shape, dtype)
      end

      private

      # Infer the shape from a nested array structure
      # @param data [Array] Potentially nested array
      # @return [Array<Integer>] Inferred shape
      def infer_shape(data)
        return [0] if data.empty?

        shape = []
        current = data

        while current.is_a?(Array)
          shape << current.size
          break if current.empty?
          current = current.first
        end

        shape
      end

      # Flatten a nested array into a 1D array, validating its structure on the
      # way down.
      #
      # @param data [Array] Potentially nested array
      # @param shape [Array<Integer>] The shape inferred from the first branch
      # @return [Array] Flat array
      # @raise [ArgumentError] If the array is jagged or inconsistently nested
      def flatten_nested(data, shape)
        return [] if shape.include?(0)

        # Walk one whole level at a time rather than recursing per element: at
        # depth k every node must be an Array of exactly shape[k] entries, and
        # Array#concat gathers the next level in C. The recursive flat_map this
        # replaces allocated an intermediate Array for *every leaf*, which on a
        # 150k-element input meant 150k throwaway objects before a single
        # number crossed into C++.
        level = [data]
        shape.each_with_index do |dim, depth|
          nxt = []
          level.each do |node|
            unless node.is_a?(Array)
              raise ArgumentError,
                    "Inconsistent nesting depth at level #{depth}: expected Array, got #{node.class}"
            end
            unless node.size == dim
              raise ArgumentError,
                    "Jagged array at depth #{depth}: expected size #{dim}, got #{node.size}"
            end
            nxt.concat(node)
          end
          level = nxt
        end

        # `level` now holds what should be the leaves. If any branch nested
        # deeper than the shape says, flattening changes the element count.
        unless level.flatten.size == level.size
          raise ArgumentError,
                "Inconsistent nesting: some elements nest deeper than shape #{shape.inspect}"
        end

        level
      end
    end

    # Convert tensor to a nested Ruby array matching the tensor's shape
    # @return [Array] Nested array with structure matching shape
    #
    # @example
    #   tensor = Tensor.new([1, 2, 3, 4], shape: [2, 2])
    #   tensor.to_a  # => [[1.0, 2.0], [3.0, 4.0]]
    #
    def to_a
      flat = flat_to_a
      reshape_flat_to_nested(flat, shape)
    end

    # Convert tensor to a flat Ruby array (original behavior)
    # @return [Array] Flat array of all values
    alias_method :flat_to_a, :_original_to_a

    private

    # Reshape a flat array into nested arrays according to shape
    # @param flat [Array] Flat array of values
    # @param shape [Array<Integer>] Target shape
    # @return [Array] Nested array
    def reshape_flat_to_nested(flat, shape)
      return flat if shape.size <= 1

      # Handle empty dimensions
      if shape.include?(0)
        return build_empty_nested(shape)
      end

      # Calculate strides for each dimension
      build_nested(flat, shape, 0, 0).first
    end

    # Build nested array structure
    # @param flat [Array] Flat data
    # @param shape [Array<Integer>] Shape
    # @param dim [Integer] Current dimension
    # @param offset [Integer] Current offset in flat array
    # @return [Array] [nested_result, new_offset]
    def build_nested(flat, shape, dim, offset)
      if dim == shape.size - 1
        # Last dimension: slice the flat array
        result = flat[offset, shape[dim]]
        [result, offset + shape[dim]]
      else
        # Build sub-arrays
        result = []
        current_offset = offset
        shape[dim].times do
          sub_result, current_offset = build_nested(flat, shape, dim + 1, current_offset)
          result << sub_result
        end
        [result, current_offset]
      end
    end

    # Build empty nested structure for shapes with zero dimensions
    # @param shape [Array<Integer>] Shape with at least one zero
    # @return [Array] Empty nested structure
    def build_empty_nested(shape)
      return [] if shape.size == 1

      first_zero = shape.index(0)

      if first_zero == 0
        []
      else
        # Build arrays up to the zero dimension
        build_empty_recursive(shape, 0)
      end
    end

    # Recursively build empty nested arrays
    # @param shape [Array<Integer>] Shape
    # @param dim [Integer] Current dimension
    # @return [Array] Empty nested structure
    def build_empty_recursive(shape, dim)
      return [] if dim >= shape.size || shape[dim] == 0

      Array.new(shape[dim]) { build_empty_recursive(shape, dim + 1) }
    end
  end

  # Extend Model with Ruby-friendly methods
  class Model
    # Alias predict to forward for more intuitive API
    alias_method :predict, :forward

    # Make model callable
    def call(inputs)
      forward(inputs)
    end
  end
end

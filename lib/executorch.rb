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

        # Validate that all elements at each level have consistent sizes
        validate_shape(data, shape, 0)

        shape
      end

      # Validate that the array has consistent shape at all levels
      # @param data [Array] The data to validate
      # @param expected_shape [Array<Integer>] The expected shape
      # @param depth [Integer] Current depth in the array
      # @raise [ArgumentError] If the array is jagged or inconsistent
      def validate_shape(data, expected_shape, depth)
        return if depth >= expected_shape.size
        return if expected_shape[depth] == 0

        unless data.is_a?(Array)
          raise ArgumentError, "Inconsistent nesting depth at level #{depth}: expected Array, got #{data.class}"
        end

        unless data.size == expected_shape[depth]
          raise ArgumentError, "Jagged array at depth #{depth}: expected size #{expected_shape[depth]}, got #{data.size}"
        end

        data.each_with_index do |element, i|
          if depth + 1 < expected_shape.size
            # Expect more nesting
            unless element.is_a?(Array)
              raise ArgumentError, "Inconsistent nesting at depth #{depth}, index #{i}: expected Array, got #{element.class}"
            end
            validate_shape(element, expected_shape, depth + 1)
          else
            # At leaf level, should be numeric
            if element.is_a?(Array)
              raise ArgumentError, "Inconsistent nesting at depth #{depth}, index #{i}: unexpected Array at leaf level"
            end
          end
        end
      end

      # Flatten a nested array into a 1D array
      # @param data [Array] Potentially nested array
      # @param shape [Array<Integer>] The shape (used to handle empty arrays)
      # @return [Array] Flat array
      def flatten_nested(data, shape)
        return [] if shape.include?(0)
        deep_flatten(data)
      end

      # Recursively flatten an array
      # @param data [Array, Numeric] The data to flatten
      # @return [Array] Flat array
      def deep_flatten(data)
        return [data] unless data.is_a?(Array)
        data.flat_map { |element| deep_flatten(element) }
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

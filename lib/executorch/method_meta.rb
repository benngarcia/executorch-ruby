# frozen_string_literal: true

module Executorch
  # Immutable metadata describing a single input or output slot of a method.
  class TensorInfo
    attr_reader :tag, :dtype, :shape, :dim_order, :nbytes

    def initialize(tag, dtype, shape, dim_order, nbytes, memory_planned)
      @tag = tag
      @dtype = dtype
      @shape = shape&.freeze
      @dim_order = dim_order&.freeze
      @nbytes = nbytes
      @memory_planned = !!memory_planned
      freeze
    end

    def tensor?
      @tag == :tensor
    end

    def memory_planned?
      @memory_planned
    end

    def ==(other)
      other.is_a?(TensorInfo) &&
        @tag == other.tag &&
        @dtype == other.dtype &&
        @shape == other.shape &&
        @dim_order == other.dim_order &&
        @nbytes == other.nbytes &&
        @memory_planned == other.memory_planned?
    end
    alias eql? ==

    def inspect
      if tensor?
        "#<Executorch::TensorInfo dtype=#{@dtype.inspect} shape=#{@shape.inspect} nbytes=#{@nbytes}>"
      else
        "#<Executorch::TensorInfo tag=#{@tag.inspect}>"
      end
    end
    alias to_s inspect
  end

  # Immutable metadata snapshot for an ExecuTorch method, readable without
  # paying the memory-planning cost of loading the full method.
  class MethodMeta
    attr_reader :name, :inputs, :outputs, :backends, :planned_memory_bytes

    def initialize(name, inputs, outputs, backends, planned_memory_bytes)
      @name = name.dup.freeze
      @inputs = inputs.freeze
      @outputs = outputs.freeze
      @backends = backends.freeze
      @planned_memory_bytes = planned_memory_bytes
      freeze
    end

    def uses_backend?(backend_name)
      @backends.include?(backend_name.to_s)
    end

    def inspect
      "#<Executorch::MethodMeta name=#{@name.inspect} inputs=#{@inputs.size} outputs=#{@outputs.size} backends=#{@backends.inspect} planned_memory_bytes=#{@planned_memory_bytes}>"
    end
    alias to_s inspect
  end
end

# frozen_string_literal: true

module Executorch
  class Module
    class << self
      # Open an ExecuTorch .pte program from disk.
      # Defaults to :mmap for copy-on-write page sharing across forked workers.
      # Supports an optional block for deterministic resource cleanup.
      #
      # @param path [String, Pathname] Path to .pte file
      # @param load_mode [Symbol] :mmap (default), :file, :mmap_mlock,
      #   :mmap_mlock_ignore_errors, or :mmap_madvise
      # @param data_files [Array<String, Pathname>, nil] Optional .ptd external weight files
      def open(path, load_mode: :mmap, data_files: nil)
        mod = _open_file(path, load_mode: load_mode, data_files: data_files)
        return mod unless block_given?

        begin
          yield mod
        ensure
          mod.close
        end
      end
      alias load open
    end

    # Execute a named method and always return an Array of outputs.
    #
    # @param method_name [String, Symbol] Method name (e.g. :forward, :encode)
    # @param inputs [Array<Tensor, Integer, Float, Boolean, String, nil>]
    # @param out [Tensor, Array<Tensor>, nil] Optional pre-allocated output tensor(s)
    # @return [Array]
    def execute(method_name, *inputs, out: nil)
      _execute(method_name, inputs, out)
    end

    # Execute the "forward" method and always return an Array of outputs.
    def forward(*inputs, out: nil)
      _execute("forward", inputs, out)
    end

    # Ergonomic invocation of "forward": returns the single output directly when
    # the method has 1 output, or an Array when it has 0 or >1 outputs.
    def call(*inputs, out: nil)
      outputs = _execute("forward", inputs, out)
      outputs.size == 1 ? outputs.first : outputs
    end

    def inspect
      return "#<Executorch::Module (closed)>" if closed?

      "#<Executorch::Module source=#{source.inspect} methods=#{method_names.inspect}>"
    end
    alias to_s inspect
  end
end

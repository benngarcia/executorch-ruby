#!/usr/bin/env ruby
# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "executorch"
require "json"

FIXTURES_DIR = File.expand_path("../test/fixtures/models", __dir__)
MANIFEST = JSON.parse(File.read(File.join(FIXTURES_DIR, "manifest.json")))

def measure(label, warmup: 10, iters: 100)
  warmup.times { yield }
  GC.start(full_mark: true, immediate_sweep: true)

  samples = Array.new(iters) do
    t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    yield
    (Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0) * 1000.0
  end
  samples.sort!
  p50 = samples[(samples.size * 0.50).floor]
  p95 = samples[[(samples.size * 0.95).floor, samples.size - 1].min]
  printf "%-36s p50=%8.4f ms  p95=%8.4f ms  (n=%d)\n", label, p50, p95, iters
end

puts "executorch-ruby #{Executorch::VERSION} | ExecuTorch SDK #{Executorch.executorch_version}"
puts "Backends: #{Executorch.backends.inspect} | Kernels: #{Executorch.kernel_count} | Threads: #{Executorch.num_threads}"
puts "-" * 78

%w[add_mul cnn_xnnpack heavy_mlp].each do |name|
  spec = MANIFEST.fetch("models").fetch(name)
  path = File.join(FIXTURES_DIR, spec["file"])
  in_desc = spec["inputs"].first
  out_desc = spec["outputs"].first

  nested = in_desc["values"]
  shape = in_desc["shape"]
  raw = nested.pack("f*")
  in_buf = IO::Buffer.for(raw)
  out_buf = IO::Buffer.new(out_desc["values"].size * 4)

  Executorch::Module.open(path) do |mod|
    mod.load_method(:forward)
    t_in = Executorch::Tensor.from_buffer(in_buf, shape: shape, dtype: :float32)
    t_out = Executorch::Tensor.from_buffer(out_buf, shape: out_desc["shape"], dtype: :float32)

    iters = name == "heavy_mlp" ? 30 : 200
    measure("#{name} / forward (owned)", iters: iters) { mod.call(t_in) }
    measure("#{name} / forward (zero-copy out:)", iters: iters) { mod.call(t_in, out: t_out) }
    measure("#{name} / e2e from_bytes + to_bytes", iters: iters) do
      t = Executorch::Tensor.from_bytes(raw, shape: shape, dtype: :float32)
      mod.call(t).to_bytes
    end
  end
end

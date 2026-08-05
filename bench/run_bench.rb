# frozen_string_literal: true

# Eval + benchmark harness for executorch-ruby.
#
#   bundle exec ruby bench/run_bench.rb --label baseline
#
# For every model under bench/models/*.pte it does two things:
#
#   1. Eval    -- runs the golden input through the gem and compares against the
#                 output PyTorch produced for the same checkpoint.
#   2. Profile -- times each leg of a Ruby inference call separately, so you can
#                 see where the time actually goes:
#
#                   tensor_new_flat    Array(flat) + shape -> Tensor
#                   tensor_new_nested  nested Array         -> Tensor
#                   forward            model.predict([t])   (tensor pre-built)
#                   to_a_flat          Tensor -> flat Array
#                   to_a_nested        Tensor -> nested Array
#                   e2e                all of the above, as an app would call it
#
# Timings use batched sampling: each sample runs the block enough times to span
# ~500us, so a per-call cost of a few microseconds isn't swamped by the cost of
# reading the clock. Allocations per call come from GC.stat, because on this
# boundary object churn is usually the story.

require "fileutils"
require "json"
require "optparse"

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))
require "executorch"

BENCH_DIR = __dir__
MODELS_DIR = File.join(BENCH_DIR, "models")
RESULTS_DIR = File.join(BENCH_DIR, "results")

options = {
  label: "current",
  models: nil,
  variant: nil,
  samples: 25,
  warmup: 20,
  sample_us: 500,
  budget_s: 3.0,
  tolerance: 1e-4,
  out: nil
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bench/run_bench.rb [options]"
  o.on("--label LABEL", "name for this run (used in the results filename)") { |v| options[:label] = v }
  o.on("--models a,b,c", Array, "only these models") { |v| options[:models] = v }
  o.on("--variant NAME", "use <model>.<NAME>.pte instead of <model>.pte (e.g. xnnpack)") do |v|
    options[:variant] = v
  end
  o.on("--samples N", Integer, "timing samples per phase") { |v| options[:samples] = v }
  o.on("--warmup N", Integer, "warmup iterations per phase") { |v| options[:warmup] = v }
  o.on("--sample-us N", Integer, "target duration of one timing sample") { |v| options[:sample_us] = v }
  o.on("--budget-s F", Float, "max seconds to spend timing one phase") { |v| options[:budget_s] = v }
  o.on("--tolerance F", Float, "max abs error allowed vs PyTorch") { |v| options[:tolerance] = v }
  o.on("--out PATH", "results JSON path") { |v| options[:out] = v }
  o.on("-h", "--help") { puts o; exit }
end.parse!

def now
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# Time a block, reporting per-call cost. Batches calls per sample so that short
# operations are measured against a clock that can actually see them, and stops
# early once `budget_s` is spent so a 2-second resnet18 call doesn't stall the
# suite waiting for its 25th sample.
def measure(samples:, warmup:, sample_us:, budget_s:)
  # One cold call tells us what scale we're working at.
  t0 = now
  yield
  single = [now - t0, 1e-9].max

  [warmup, (budget_s / 4 / single).floor].min.clamp(1, warmup).times { yield }

  batch = ((sample_us / 1e6) / single).ceil.clamp(1, 200_000)

  timings = []
  deadline = now + budget_s
  while timings.size < samples
    t = now
    batch.times { yield }
    timings << (now - t) / batch
    break if timings.size >= 5 && now > deadline
  end

  GC.start
  before = GC.stat(:total_allocated_objects)
  batch.times { yield }
  allocs = (GC.stat(:total_allocated_objects) - before).to_f / batch

  timings.sort!
  {
    "p50_ms" => timings[timings.size / 2] * 1000,
    "p90_ms" => timings[(timings.size * 0.9).floor] * 1000,
    "min_ms" => timings.first * 1000,
    "mean_ms" => (timings.sum / timings.size) * 1000,
    "allocs_per_call" => allocs.round(1),
    "batch" => batch,
    "samples" => timings.size
  }
end

# Read the golden input/output pair written by make_pt_models.py.
def read_golden(meta)
  path = File.join(MODELS_DIR, "#{meta['name']}.io.bin")
  floats = File.binread(path).unpack("e*")
  input = floats[0, meta["input_numel"]]
  output = floats[meta["input_numel"], meta["output_numel"]]
  [input, output]
end

def nest(flat, shape)
  return flat.dup if shape.size <= 1

  stride = shape[1..].inject(:*)
  Array.new(shape[0]) { |i| nest(flat[i * stride, stride], shape[1..]) }
end

def max_abs_error(actual, expected)
  actual.each_with_index.map { |v, i| (v - expected[i]).abs }.max || 0.0
end

def fmt(ms)
  ms >= 1 ? format("%8.3f", ms) : format("%8.4f", ms)
end

def pte_path(name, variant)
  suffix = variant ? ".#{variant}.pte" : ".pte"
  File.join(MODELS_DIR, "#{name}#{suffix}")
end

metas = Dir[File.join(MODELS_DIR, "*.meta.json")].sort.map { |p| JSON.parse(File.read(p)) }
metas.select! { |m| options[:models].include?(m["name"]) } if options[:models]
metas.select! { |m| File.exist?(pte_path(m["name"], options[:variant])) }

if metas.empty?
  abort "No#{options[:variant] ? " #{options[:variant]}" : ''} .pte models found in #{MODELS_DIR}.\n" \
        "Run: python3 bench/make_pt_models.py && python3 bench/pt_to_pte.py bench/models/*.pt" \
        "#{options[:variant] ? " --#{options[:variant]}" : ''}"
end

puts "executorch-ruby bench  (label=#{options[:label]}, " \
     "variant=#{options[:variant] || 'portable'}, ruby=#{RUBY_VERSION})"
puts

results = {
  "label" => options[:label],
  "variant" => options[:variant] || "portable",
  "ruby" => RUBY_VERSION,
  "models" => {}
}
failures = []

metas.each do |meta|
  name = meta["name"]
  shape = meta["input_shape"]
  out_shape = meta["output_shape"]
  flat_input, expected = read_golden(meta)
  nested_input = nest(flat_input, shape)

  model = Executorch::Model.new(pte_path(name, options[:variant]))

  # --- eval: do we agree with PyTorch? ---
  tensor = Executorch::Tensor.new(flat_input, shape: shape)
  output = model.predict([tensor]).first
  actual = output.flat_to_a
  error = max_abs_error(actual, expected)
  ok = actual.size == expected.size && error <= options[:tolerance]
  failures << "#{name}: max_abs_error=#{error}" unless ok

  # --- profile: where does an inference call spend its time? ---
  cfg = {
    samples: options[:samples],
    warmup: options[:warmup],
    sample_us: options[:sample_us],
    budget_s: options[:budget_s]
  }
  phases = {
    "tensor_new_flat" => measure(**cfg) { Executorch::Tensor.new(flat_input, shape: shape) },
    "tensor_new_nested" => measure(**cfg) { Executorch::Tensor.new(nested_input) },
    "forward" => measure(**cfg) { model.predict([tensor]) },
    "to_a_flat" => measure(**cfg) { output.flat_to_a },
    "to_a_nested" => measure(**cfg) { output.to_a },
    "e2e" => measure(**cfg) do
      t = Executorch::Tensor.new(flat_input, shape: shape)
      model.predict([t]).first.to_a
    end
  }

  # The binary path is opt-in and newer than the rest of the API, so only
  # measure it when the extension actually has it.
  if Executorch::Tensor.respond_to?(:from_bytes)
    packed = flat_input.pack("f*")
    phases["tensor_from_bytes"] = measure(**cfg) do
      Executorch::Tensor.from_bytes(packed, shape: shape)
    end
    phases["to_binary"] = measure(**cfg) { output.to_binary }
    phases["e2e_binary"] = measure(**cfg) do
      t = Executorch::Tensor.from_bytes(packed, shape: shape)
      model.predict([t]).first.to_binary.unpack("f*")
    end

    packed_ok = Executorch::Tensor.from_bytes(packed, shape: shape).flat_to_a
    binary_err = max_abs_error(packed_ok, flat_input)
    failures << "#{name}: from_bytes round-trip error=#{binary_err}" if binary_err > 0
  end

  e2e = phases["e2e"]["p50_ms"]
  fwd = phases["forward"]["p50_ms"]
  overhead_pct = e2e.positive? ? ((e2e - fwd) / e2e * 100) : 0.0

  results["models"][name] = {
    "regime" => meta["regime"],
    "input_shape" => shape,
    "output_shape" => out_shape,
    "params" => meta["params"],
    "eager_p50_ms" => meta["eager_p50_ms"],
    "eval" => { "passed" => ok, "max_abs_error" => error, "tolerance" => options[:tolerance] },
    "phases" => phases,
    "binding_overhead_pct" => overhead_pct.round(1)
  }

  status = ok ? "PASS" : "FAIL"
  puts "#{name}  [#{meta['regime']}]  in=#{shape.inspect} out=#{out_shape.inspect} " \
       "params=#{meta['params']}"
  puts "  eval: #{status}  max_abs_error=#{format('%.2e', error)}  " \
       "(pytorch eager p50 #{format('%.3f', meta['eager_p50_ms'])} ms)"
  puts format("  %-18s %10s %10s %14s", "phase", "p50 ms", "p90 ms", "allocs/call")
  phases.each do |phase, stats|
    puts format("  %-18s %10s %10s %14.1f",
                phase, fmt(stats["p50_ms"]), fmt(stats["p90_ms"]), stats["allocs_per_call"])
  end
  puts format("  binding overhead: %.1f%% of e2e is not forward()", overhead_pct)
  puts
end

FileUtils.mkdir_p(RESULTS_DIR)
out_path = options[:out] || File.join(RESULTS_DIR, "#{options[:label]}.json")
File.write(out_path, JSON.pretty_generate(results))
puts "results -> #{out_path}"

if failures.empty?
  puts "all evals passed"
else
  puts "EVAL FAILURES:"
  failures.each { |f| puts "  #{f}" }
  exit 1
end

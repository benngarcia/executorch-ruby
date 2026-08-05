# frozen_string_literal: true

# Diff two bench result files.
#
#   bundle exec ruby bench/compare.rb bench/results/baseline.json bench/results/optimized.json
#
# Speedup is baseline/current, so 2.00x means the current run is twice as fast.

require "json"

base_path, cur_path = ARGV
abort "Usage: ruby bench/compare.rb BASELINE.json CURRENT.json" unless base_path && cur_path

base = JSON.parse(File.read(base_path))
cur = JSON.parse(File.read(cur_path))

PHASES = %w[
  tensor_new_flat tensor_new_nested tensor_from_bytes
  forward
  to_a_flat to_a_nested to_binary
  e2e e2e_binary
].freeze

puts "#{base['label']} -> #{cur['label']}"
puts

(base["models"].keys & cur["models"].keys).each do |name|
  b = base["models"][name]
  c = cur["models"][name]

  puts "#{name}  [#{c['regime']}]"
  puts format("  %-18s %12s %12s %9s %10s %10s",
              "phase", "base ms", "cur ms", "speedup", "base allocs", "cur allocs")

  PHASES.each do |phase|
    bp = b["phases"][phase]
    cp = c["phases"][phase]
    next unless bp && cp

    speedup = cp["p50_ms"].positive? ? bp["p50_ms"] / cp["p50_ms"] : 0.0
    puts format("  %-18s %12.4f %12.4f %8.2fx %10.1f %10.1f",
                phase, bp["p50_ms"], cp["p50_ms"], speedup,
                bp["allocs_per_call"], cp["allocs_per_call"])
  end

  puts format("  binding overhead: %.1f%% -> %.1f%% of e2e",
              b["binding_overhead_pct"], c["binding_overhead_pct"])
  puts
end

only_cur = cur["models"].keys - base["models"].keys
puts "only in #{cur['label']}: #{only_cur.join(', ')}" unless only_cur.empty?

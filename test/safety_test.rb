# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

class SafetyTest < Minitest::Test
  include TestHelper

  # 1. Heap OOB, negative dims, and integer overflows must raise, NEVER crash or leak memory
  def test_shape_mismatch_negative_dims_and_integer_overflows
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.new([1.0], shape: [100_000]) }
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.new([1.0], shape: [-1]) }
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.new([1.0], shape: [1 << 32]) }

    raw4 = [1.0, 2.0, 3.0, 4.0].pack("f*")
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.from_bytes(raw4, shape: [-2, -2]) }
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.from_bytes("", shape: [1 << 30, 1 << 30, 4]) }
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.from_bytes(raw4, shape: [2, 3]) }
    assert_raises(Executorch::ShapeError) { Executorch::Tensor.from_bytes(raw4, shape: [1, 1]) }
  end

  # 2. Reentrant array-shrinking #to_f must not cause heap OOB read
  def test_reentrant_mutating_array_does_not_oob
    ary = []
    shrinker = Object.new
    shrinker.define_singleton_method(:to_f) do
      ary.clear
      1.0
    end
    ary.push(shrinker, 2.0, 3.0, 4.0)

    assert_raises(ArgumentError) { Executorch::Tensor.new(ary, shape: [4], dtype: :float32) }
  end

  # 3. Jagged & malformed nested arrays (including PR #1 regressions)
  def test_jagged_and_malformed_nested_arrays_rejected
    [
      [[], [1.0, 2.0]],
      [[], "garbage", nil],
      [[1.0], [[2.0]]],
      [[1.0, 2.0], [[3.0], [4.0]]],
      [[1.0, 2.0], [3.0]],
      [1.0, [2.0]],
    ].each do |bad|
      assert_raises(ArgumentError, "Expected ArgumentError for #{bad.inspect}") do
        Executorch::Tensor.new(bad)
      end
    end
  end

  # 4. IEEE-754 special values & integer range checks (zero C++ UB, zero silent truncation)
  def test_nan_inf_and_integer_range_checks
    t = Executorch::Tensor.new([Float::NAN, Float::INFINITY, -Float::INFINITY, -0.0], dtype: :float32)
    vals = t.to_a
    assert vals[0].nan?
    assert_equal Float::INFINITY, vals[1]
    assert_equal(-Float::INFINITY, vals[2])

    assert_raises(FloatDomainError) { Executorch::Tensor.new([Float::NAN], dtype: :int64) }
    assert_raises(FloatDomainError) { Executorch::Tensor.new([Float::INFINITY], dtype: :int64) }
    assert_raises(FloatDomainError) { Executorch::Tensor.new([Float::NAN], dtype: :uint64) }
    assert_raises(RangeError) { Executorch::Tensor.new([1 << 40], dtype: :int32) }
    assert_raises(RangeError) { Executorch::Tensor.new([-1], dtype: :uint8) }
    assert_raises(RangeError) { Executorch::Tensor.new([256], dtype: :uint8) }
    assert_raises(Executorch::DTypeError) { Executorch::Tensor.new([1], dtype: :not_a_dtype) }
  end

  # 5. Corrupt, empty, truncated, nonexistent, and directory .pte files
  def test_corrupt_and_invalid_pte_files
    err = assert_raises(Executorch::LoadError) { Executorch::Module.open("/nonexistent/model.pte") }
    assert_includes err.message, "/nonexistent/model.pte"

    assert_raises(Executorch::LoadError) { Executorch::Module.open(Dir.tmpdir) }
    assert_raises(Executorch::LoadError) { Executorch::Module.from_bytes("") }
    assert_raises(Executorch::LoadError) { Executorch::Module.from_bytes("not a flatbuffer") }

    valid_bytes = File.binread(fixture_path(model_spec(:add_mul)["file"]))
    assert_raises(Executorch::LoadError) { Executorch::Module.from_bytes(valid_bytes.byteslice(0, 32)) }

    Dir.mktmpdir do |dir|
      empty = File.join(dir, "empty.pte")
      File.binwrite(empty, "")
      assert_raises(Executorch::LoadError) { Executorch::Module.open(empty) }
    end
  end

  # 6. Invalid inputs to Module#execute (no unbound Rice type crash!)
  def test_invalid_inputs_and_missing_method
    spec = model_spec(:add_mul)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      assert_raises(Executorch::MethodNotFoundError) { mod.meta(:does_not_exist) }
      assert_raises(Executorch::MethodNotFoundError) { mod.execute(:does_not_exist) }

      # Non-Tensor unsupported Ruby object raises TypeError cleanly
      assert_raises(TypeError) { mod.call(Object.new) }

      # Wrong shape / arity raises Executorch::Error with diagnostic info
      wrong_shape = Executorch::Tensor.new([[1.0, 2.0, 3.0, 4.0]])
      err = assert_raises(Executorch::Error) { mod.call(wrong_shape) }
      assert_kind_of Integer, err.code
      refute_nil err.pal_log
      refute_empty err.pal_log

      assert_raises(Executorch::Error) { mod.forward }
    end
  end

  # 7. Sequential execute calls with DISTINCT inputs must NOT alias or overwrite earlier outputs
  def test_sequential_execute_outputs_are_independent
    spec = model_spec(:add_mul)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      in1 = Executorch::Tensor.new([[1.0, 2.0, 3.0]])
      in2 = Executorch::Tensor.new([[10.0, 20.0, 30.0]])

      out1 = mod.call(in1)
      out2 = mod.call(in2)

      assert_equal [[3.0, 5.0, 7.0]], out1.to_a
      assert_equal [[21.0, 41.0, 61.0]], out2.to_a
      assert_equal [[1.0, 2.0, 3.0]], in1.to_a
      assert_equal [[10.0, 20.0, 30.0]], in2.to_a
    end
  end

  # 8. GC.stress and GC.compact stability
  def test_gc_stress_and_compaction
    spec = model_spec(:add_mul)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      prev = GC.stress
      begin
        GC.stress = true
        3.times do |i|
          t = Executorch::Tensor.new([[i.to_f, i + 1.0, i + 2.0]])
          out = mod.call(t)
          assert_equal [[i * 2.0 + 1.0, (i + 1) * 2.0 + 1.0, (i + 2) * 2.0 + 1.0]], out.to_a
        end
      ensure
        GC.stress = prev
      end

      GC.compact if GC.respond_to?(:compact)
      # Exercise every cached Ruby class VALUE (MethodMeta, TensorInfo, Tensor, Errors) after GC.compact
      meta = mod.meta(:forward)
      assert_equal "forward", meta.name
      assert_equal 1, meta.inputs.size
      assert_raises(Executorch::MethodNotFoundError) { mod.meta(:missing_after_compact) }
      out = mod.call(Executorch::Tensor.new([[2.0, 3.0, 4.0]]))
      assert_equal [[5.0, 7.0, 9.0]], out.to_a
    end
  end

  # 9. Multi-threaded GVL release proof & thread-safety under concurrent calls
  def test_gvl_released_and_concurrent_thread_safety
    spec = model_spec(:heavy_mlp)
    pte = fixture_path(spec["file"])
    input = tensor_from_golden(spec["inputs"].first)

    # A. Prove background Ruby thread ticks continuously during native inference
    Executorch::Module.open(pte) do |mod|
      mod.load_method(:forward)
      ready = Queue.new
      ticks = 0
      counting = false
      stop = false
      ticker = Thread.new do
        ready << true
        until stop
          ticks += 1 if counting
          Thread.pass
        end
      end
      ready.pop

      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      min_end = start + 0.25
      max_end = start + 2.0
      runs = 0
      counting = true
      while (runs < 5 || Process.clock_gettime(Process::CLOCK_MONOTONIC) < min_end) &&
            Process.clock_gettime(Process::CLOCK_MONOTONIC) < max_end
        mod.call(input)
        runs += 1
      end
      counting = false
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
      stop = true
      ticker.join

      assert_operator ticks, :>, 20,
        "Background Ruby thread starved (ticks=#{ticks}, runs=#{runs}, elapsed=#{format('%.3fs', elapsed)}) — GVL was held!"
    end

    # B. 4 threads sharing ONE Module instance serialize safely with exact results
    Executorch::Module.open(pte) do |shared|
      threads = 4.times.map do
        Thread.new do
          5.times do
            out = shared.call(input)
            assert_tensor_matches_golden(out, spec["outputs"].first, rtol: 1e-2, atol: 1e-3)
          end
        end
      end
      threads.each(&:join)
    end

    # C. 2 threads with independent Module instances execute cleanly in parallel
    threads = 2.times.map do
      Thread.new do
        Executorch::Module.open(pte) do |mod|
          5.times do
            out = mod.call(input)
            assert_tensor_matches_golden(out, spec["outputs"].first, rtol: 1e-2, atol: 1e-3)
          end
        end
      end
    end
    threads.each(&:join)
  end

  # 10. Memory stability check (no C++ leak across 2,000 iterations)
  def test_no_unbounded_rss_growth
    spec = model_spec(:add_mul)
    raw_in = [1.0, 2.0, 3.0].pack("f*")

    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      # Warmup
      50.times { mod.call(Executorch::Tensor.from_bytes(raw_in, shape: [1, 3])) }
      GC.start(full_mark: true, immediate_sweep: true)
      rss_before = current_rss_kb

      2_000.times do
        t = Executorch::Tensor.from_bytes(raw_in, shape: [1, 3])
        out = mod.call(t)
        out.to_bytes
      end
      GC.start(full_mark: true, immediate_sweep: true)
      rss_after = current_rss_kb

      if rss_before && rss_after
        delta_mb = (rss_after - rss_before) / 1024.0
        assert_operator delta_mb, :<, 32.0, "RSS grew by #{delta_mb.round(2)} MB over 2,000 iterations"
      end
    end
  end

  private

  def current_rss_kb
    return nil unless File.readable?("/proc/self/status")

    line = File.readlines("/proc/self/status").find { |l| l.start_with?("VmRSS:") }
    line ? line.split[1].to_i : nil
  end
end

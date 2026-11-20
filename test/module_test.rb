# frozen_string_literal: true

require_relative "test_helper"

class ModuleTest < Minitest::Test
  include TestHelper

  def test_open_string_and_pathname_with_block_and_load_modes
    spec = model_spec(:add_mul)
    path = fixture_path(spec["file"])
    input = tensor_from_golden(spec["inputs"].first)

    %i[mmap file mmap_madvise].each do |mode|
      closed_mod = nil
      Executorch::Module.open(Pathname.new(path), load_mode: mode) do |mod|
        closed_mod = mod
        refute_predicate mod, :closed?
        assert_equal path, mod.source
        assert_includes mod.method_names, "forward"
        out = mod.call(input)
        assert_tensor_matches_golden(out, spec["outputs"].first)
      end
      assert_predicate closed_mod, :closed?
      assert_raises(Executorch::ClosedError) { closed_mod.call(input) }
    end
  end

  def test_from_bytes_in_memory_load
    spec = model_spec(:add_mul)
    raw = File.binread(fixture_path(spec["file"]))
    mod = Executorch::Module.from_bytes(raw)
    assert_includes mod.source, "<bytes:"

    out = mod.call(tensor_from_golden(spec["inputs"].first))
    assert_tensor_matches_golden(out, spec["outputs"].first)
  ensure
    mod&.close
  end

  def test_method_meta_and_lifecycle
    spec = model_spec(:cnn_xnnpack)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      refute mod.method_loaded?(:forward)

      meta = mod.meta(:forward)
      assert_instance_of Executorch::MethodMeta, meta
      assert_equal "forward", meta.name
      assert_equal 1, meta.inputs.size
      assert_equal :float32, meta.inputs.first.dtype
      assert_equal [1, 3, 16, 16], meta.inputs.first.shape
      assert_equal 1, meta.outputs.size
      assert_equal [1, 4, 16, 16], meta.outputs.first.shape
      assert meta.uses_backend?("XnnpackBackend"), "expected XnnpackBackend in #{meta.backends.inspect}"
      assert_operator meta.planned_memory_bytes, :>=, 0

      # Explicit pre-warm & unload
      mod.load_method(:forward)
      assert mod.method_loaded?(:forward)

      out = mod.call(tensor_from_golden(spec["inputs"].first))
      assert_tensor_matches_golden(out, spec["outputs"].first, rtol: 1e-2, atol: 1e-3)

      assert_equal true, mod.unload_method(:forward)
      refute mod.method_loaded?(:forward)
    end
  end

  def test_multi_input_and_multi_output
    spec = model_spec(:multi_io)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      a = tensor_from_golden(spec["inputs"][0])
      b = tensor_from_golden(spec["inputs"][1])

      outs = mod.call(a, b)
      assert_instance_of Array, outs
      assert_equal 2, outs.size
      assert_tensor_matches_golden(outs[0], spec["outputs"][0])
      assert_tensor_matches_golden(outs[1], spec["outputs"][1])

      # #forward and #execute always return Array
      outs2 = mod.execute(:forward, a, b)
      assert_equal outs, outs2
    end
  end

  def test_multi_method_encode_and_decode
    spec = model_spec(:multi_method)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      assert_equal %w[decode encode], mod.method_names

      enc_spec = spec["methods"]["encode"]
      dec_spec = spec["methods"]["decode"]

      z = mod.execute(:encode, tensor_from_golden(enc_spec["inputs"].first)).first
      assert_tensor_matches_golden(z, enc_spec["outputs"].first)

      recon = mod.execute("decode", z).first
      assert_tensor_matches_golden(recon, dec_spec["outputs"].first)
    end
  end

  def test_multiple_dtypes_end_to_end
    spec = model_spec(:dtypes)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      inputs = spec["inputs"].map { |d| tensor_from_golden(d) }
      outputs = mod.forward(*inputs)
      assert_equal 4, outputs.size
      outputs.each_with_index do |out, i|
        assert_tensor_matches_golden(out, spec["outputs"][i])
      end
    end
  end

  def test_external_ptd_weights
    spec = model_spec(:external_weights)
    pte = fixture_path(spec["file"])
    ptds = spec["data_files"].map { |rel| Pathname.new(fixture_path(rel)) }
    refute_empty ptds, "expected at least one .ptd external data file"

    Executorch::Module.open(pte, data_files: ptds) do |mod|
      out = mod.call(tensor_from_golden(spec["inputs"].first))
      assert_tensor_matches_golden(out, spec["outputs"].first)
    end
  end

  def test_preallocated_out_tensor_and_zero_copy_io_buffer
    spec = model_spec(:add_mul)
    Executorch::Module.open(fixture_path(spec["file"])) do |mod|
      in_buf = IO::Buffer.new(12)
      in_buf.set_string([1.0, 2.0, 3.0].pack("f*"))
      inp = Executorch::Tensor.from_buffer(in_buf, shape: [1, 3], dtype: :float32)

      out_buf = IO::Buffer.new(12)
      out = Executorch::Tensor.from_buffer(out_buf, shape: [1, 3], dtype: :float32)

      ret = mod.call(inp, out: out)
      assert_same out, ret
      assert_tensor_matches_golden(out, spec["outputs"].first)
      # Verify written straight into out_buf
      assert_equal [3.0, 5.0, 7.0], out_buf.get_string.unpack("f*")

      # Mismatched out shape/dtype raises cleanly
      bad_shape = Executorch::Tensor.zeros([1, 4], dtype: :float32)
      assert_raises(Executorch::ShapeError) { mod.call(inp, out: bad_shape) }

      bad_dtype = Executorch::Tensor.zeros([1, 3], dtype: :float64)
      assert_raises(Executorch::DTypeError) { mod.call(inp, out: bad_dtype) }
    end
  end
end

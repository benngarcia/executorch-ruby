# executorch-ruby

[![CI](https://github.com/benngarcia/executorch-ruby/actions/workflows/ci.yml/badge.svg)](https://github.com/benngarcia/executorch-ruby/actions/workflows/ci.yml)
[![Gem Version](https://badge.fury.io/rb/executorch.svg)](https://rubygems.org/gems/executorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)

Native Ruby bindings for Meta's [ExecuTorch](https://pytorch.org/executorch/) runtime (`1.6.0`), built directly on the Ruby C API.

Run exported PyTorch models (`.pte` / `.ptd`) in Ruby with precompiled platform gems, GVL-free execution, zero-copy `IO::Buffer` tensors, and hardware delegates (XNNPACK, CoreML, MLX).

## Installation

Add to your `Gemfile`:

```ruby
gem "executorch", "~> 1.0"
```

Or install directly:

```bash
gem install executorch
```

Requires **Ruby `>= 3.2`**.

### Supported platforms

| Platform | Precompiled Gem | Bundled Delegates & Kernels |
|---|---|---|
| `x86_64-linux` | Yes | XNNPACK, Optimized CPU, Quantized |
| `aarch64-linux` | Yes | XNNPACK (KleidiAI), Optimized CPU, Quantized, TorchAO |
| `arm64-darwin` | Yes | XNNPACK, CoreML, MLX, Optimized CPU, Quantized, TorchAO |
| Source (`ruby`) | Yes | Fetches the official ExecuTorch 1.6.0 C++ SDK (or uses `--with-executorch-dir`) |

## Quick Start

Export a PyTorch model to `.pte` using ExecuTorch:

```python
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

class AddMul(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0

example_input = (torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),)
exported = torch.export.export(AddMul().eval(), example_input)
et_program = to_edge_transform_and_lower(
    exported, partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as f:
    et_program.write_to_file(f)
```

Load and run it in Ruby:

```ruby
require "executorch"

Executorch::Module.open("model.pte") do |mod|
  meta = mod.meta(:forward)
  p meta.backends # => ["XnnpackBackend"]
  p meta.inputs   # => [#<Executorch::TensorInfo dtype=:float32 shape=[1, 3] nbytes=12>]

  input  = Executorch::Tensor.new([[1.0, 2.0, 3.0]])
  output = mod.call(input)
  p output.to_a   # => [[3.0, 5.0, 7.0]]
end
```

## Usage

### Modules

`Executorch::Module.open` accepts a `String` or `Pathname`, defaults to `:mmap` loading (sharing read-only weight pages across forked Puma/Unicorn workers), and closes deterministically when given a block.

```ruby
# Load modes: :mmap (default), :file, :mmap_mlock, :mmap_mlock_ignore_errors, :mmap_madvise
mod = Executorch::Module.open(Pathname.new("model.pte"), load_mode: :mmap)
mod.load_method(:forward) # optional pre-warm before serving requests
out = mod.call(input)
mod.close

# Multi-method programs
Executorch::Module.open("autoencoder.pte") do |m|
  z = m.execute(:encode, x).first
  reconstructed = m.execute(:decode, z).first
end

# External .ptd weight files
Executorch::Module.open("model.pte", data_files: ["weights.ptd"]) do |m|
  logits = m.call(tokens)
end

# Load directly from an in-memory String (e.g. S3 or ActiveStorage)
mod = Executorch::Module.from_bytes(pte_bytes)
mod.close
```

- `#call(*inputs, out: nil)` runs `"forward"` and returns the single output directly when there is one output, or an `Array` otherwise.
- `#forward(*inputs, out: nil)` and `#execute(name, *inputs, out: nil)` always return an `Array`.
- `#meta(name = :forward)` returns an `Executorch::MethodMeta` describing input/output shapes, dtypes, planned memory size, and lowered backends without loading the full method.

### Tensors

Supported dtypes: `:float32`, `:float64`, `:float16`, `:bfloat16`, `:uint8`, `:int8`, `:int16`, `:uint16`, `:int32`, `:uint32`, `:int64`, `:uint64`, and `:bool`. Short aliases (`:float`, `:double`, `:half`, `:byte`, `:char`, `:short`, `:int`, `:long`) are accepted at construction.

```ruby
t = Executorch::Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t.shape   # => [2, 3]
t.dtype   # => :float32
t.nbytes  # => 24
t.to_a    # => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

# 0-D scalar tensors
scalar = Executorch::Tensor.new(42.5, dtype: :float32)
scalar.shape # => []
scalar.item  # => 42.5

# Packed binary I/O & factories
raw = t.to_bytes
t2  = Executorch::Tensor.from_bytes(raw, shape: [2, 3], dtype: :float32)
t == t2 # => true

zeros = Executorch::Tensor.zeros([2, 3], dtype: :float32)
ones  = Executorch::Tensor.ones([2, 3], dtype: :int64)
full  = Executorch::Tensor.full([2, 3], 7, dtype: :uint8)
```

Optional adapters are provided for `Numo::NArray` (`Tensor.from_numo` / `Tensor#to_numo`) and `Torch::Tensor` (`Tensor.from_torch`).

### Zero-copy `IO::Buffer` and output reuse

Pass Ruby `IO::Buffer` instances directly into ExecuTorch without copying or allocating intermediate arrays. Pairing `Tensor.from_buffer` with `out:` writes inference outputs straight into a caller-owned buffer:

```ruby
in_buf  = IO::Buffer.for([1.0, 2.0, 3.0].pack("f*"))
out_buf = IO::Buffer.new(12)

input  = Executorch::Tensor.from_buffer(in_buf, shape: [1, 3], dtype: :float32)
output = Executorch::Tensor.from_buffer(out_buf, shape: [1, 3], dtype: :float32)

Executorch::Module.open("model.pte") do |mod|
  mod.call(input, out: output)
  p out_buf.get_string.unpack("f*") # => [3.0, 5.0, 7.0]
end
```

### Concurrency, runtime introspection & errors

```ruby
Executorch.backends           # => ["XnnpackBackend"]
Executorch.kernel_count       # => 263
Executorch.num_threads        # => 24
Executorch.num_threads = 2    # size the ExecuTorch threadpool for multi-worker servers
Executorch.executorch_version # => "1.6.0"
```

- **GVL release & thread safety:** Model loading and execution release the Ruby GVL (`rb_thread_call_without_gvl`). Calls on separate `Module` instances run in parallel across cores; concurrent calls to the same method on a shared `Module` serialize safely via a per-method mutex.
- **Typed errors:** Runtime failures inherit from `Executorch::Error < StandardError` (`LoadError`, `MethodNotFoundError`, `OperatorMissingError`, `DelegateError`, `ShapeError`, `DTypeError`, `ExecutionError`, `MemoryError`, `ClosedError`) and expose `#code` and `#pal_log`.

## Architecture

The native extension is written directly against the Ruby C API (no binding generator or runtime gem dependencies) in two layers:

- **`ext/executorch/core/`** — A `noexcept` C++17 core with zero Ruby headers. Handles ExecuTorch lifecycle, per-method locking, thread-local PAL log capture, overflow-checked shape math, and strided/`NHWC` layout normalization.
- **`ext/executorch/rb_*.cpp`** — A thin Ruby C-API layer managing `TypedData`, `dsize` + `rb_gc_adjust_memory_usage`, `GC.compact`-safe global roots, `rb_io_buffer_lock` around GVL-free execution, and single-pass C++ nested-array flattening. All C++ stack objects are destroyed before any Ruby exception is raised.
- **Symbol isolation:** Compiled with `-fvisibility=hidden` and a linker export map so only `Init_executorch` is exported, avoiding symbol collisions with `torch.rb`, `onnxruntime`, or `ruby-vips`.

## Custom ExecuTorch builds

To compile against a local ExecuTorch installation instead of the default SDK:

```bash
bundle config set --local build.executorch --with-executorch-dir=/absolute/path/to/executorch
```

Or pass `EXECUTORCH_DIR=/absolute/path/to/executorch`. Set `EXECUTORCH_NO_DOWNLOAD=1` to disable automatic SDK fetching.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes and migration details.

## Development

```bash
bundle install
bundle exec rake sdk:fetch        # Fetches verified ExecuTorch 1.6.0 C++ SDK into ~/.cache/executorch-ruby/
bundle exec rake compile          # Compiles native C++17 extension
bundle exec rake fixtures:export  # Exports deterministic .pte/.ptd test models via uv
bundle exec rake test             # Runs full test suite
ruby bench/run_bench.rb           # Runs local latency benchmarks
```

To run the suite under AddressSanitizer + UndefinedBehaviorSanitizer on Linux:

```bash
bundle exec rake clean
EXECUTORCH_SANITIZE=1 bundle exec rake compile
LD_PRELOAD="$(g++ -print-file-name=libasan.so):$(g++ -print-file-name=libubsan.so)" \
  ASAN_OPTIONS=detect_leaks=0:use_sigaltstack=0:halt_on_error=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  bundle exec rake test
```

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).

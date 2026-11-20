# Executorch for Ruby

[![CI](https://github.com/benngarcia/executorch-ruby/actions/workflows/ci.yml/badge.svg)](https://github.com/benngarcia/executorch-ruby/actions/workflows/ci.yml)
[![Gem Version](https://badge.fury.io/rb/executorch.svg)](https://rubygems.org/gems/executorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)

Run PyTorch models natively in Ruby via Meta's [ExecuTorch](https://pytorch.org/executorch/) (`1.6.0`) runtime — with **zero Python at inference time**, **zero runtime gem dependencies**, and **precompiled native gems** that install in seconds.

## Why ExecuTorch in Ruby?

- **`gem install executorch` just works:** Precompiled platform gems bundle the official ExecuTorch 1.6.0 C++ runtime, XNNPACK, optimized CPU kernels, and quantized kernels (+ CoreML & MLX on Apple Silicon) in a ~5 MB package. No CMake, Python, Buck2, or 40-minute source builds.
- **Built for production Ruby servers (Puma, Sidekiq, Falcon):**
  - **Releases the GVL** during model load and inference so background Ruby threads and web workers never stall, with per-method thread-safety locks.
  - **Defaults to `mmap`** for copy-on-write weight sharing across forked workers.
  - **GC-aware & deterministic cleanup:** Reports native memory to Ruby's GC (`dsize` + `rb_gc_adjust_memory_usage`) and supports `#close` / block-form `Module.open`.
- **C++17 speed without Rice overhead:** Direct Ruby C API over a `noexcept` C++17 core (`~6.3 µs` zero-copy forward floor; `~30.5 µs` XNNPACK Conv2d CNN; strict symbol hiding exporting only `Init_executorch` so it co-loads safely alongside `torch.rb`, `onnxruntime`, or `ruby-vips`).
- **Complete ExecuTorch surface:** `MethodMeta` introspection, multi-method `.pte` programs, external `.ptd` weights, in-memory `Module.from_bytes`, all 13 numeric/bool dtypes (including `:float16`, `:bfloat16`, `:uint8`, `:bool`), zero-copy `IO::Buffer`, pre-allocated `out:` output reuse, and typed `Executorch::Error` exceptions with captured PAL diagnostics.

## Platform & Delegate Support

| Platform | Precompiled Gem | Included Delegates & Kernels |
|---|---|---|
| `x86_64-linux` | Yes | XNNPACK, Optimized CPU, Quantized |
| `aarch64-linux` | Yes | XNNPACK (KleidiAI), Optimized CPU, Quantized, TorchAO |
| `arm64-darwin` (Apple Silicon) | Yes | XNNPACK, CoreML, MLX, Optimized CPU, Quantized, TorchAO |
| Source fallback (`ruby`) | Yes | Auto-fetches SHA-256-verified ExecuTorch 1.6.0 SDK (or uses `--with-executorch-dir`) |

Requires **Ruby `>= 3.2`**.

## Installation

Add to your `Gemfile`:

```ruby
gem "executorch", "~> 1.0"
```

Or install directly:

```bash
gem install executorch
```

## Quick Start: PyTorch to Ruby in 60 Seconds

### 1. Export a PyTorch model to `.pte` (once, in Python)

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

### 2. Run it natively in Ruby (no Python required)

```ruby
require "executorch"

# Defaults to :mmap (CoW-friendly across forked Puma workers).
# Accepts String or Pathname; block form closes deterministically.
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

### Zero-copy `IO::Buffer` and pre-allocated output reuse

For hot inference loops (images, audio, embeddings), avoid array boxing and per-call allocations completely:

```ruby
Executorch.num_threads = 2 # Prevent oversubscription in multi-worker Puma servers

in_buf  = IO::Buffer.for([1.0, 2.0, 3.0].pack("f*"))
out_buf = IO::Buffer.new(12)

input  = Executorch::Tensor.from_buffer(in_buf, shape: [1, 3], dtype: :float32)
output = Executorch::Tensor.from_buffer(out_buf, shape: [1, 3], dtype: :float32)

Executorch::Module.open("model.pte") do |mod|
  mod.call(input, out: output) # Releases GVL; writes directly into out_buf
  p out_buf.get_string.unpack("f*") # => [3.0, 5.0, 7.0]
end
```

### Multi-method models, external `.ptd` weights, and in-memory loading

```ruby
# Multi-method .pte (e.g. encode / decode, prefill / decode)
Executorch::Module.open("autoencoder.pte") do |mod|
  z = mod.execute(:encode, x).first
  reconstructed = mod.execute(:decode, z).first
end

# External .ptd weights (LLMs and larger models)
Executorch::Module.open("model.pte", data_files: ["weights.ptd"], load_mode: :mmap_mlock) do |mod|
  logits = mod.call(tokens)
end

# Load straight from memory (e.g. S3 / ActiveStorage) without a tempfile
mod = Executorch::Module.from_bytes(pte_bytes)
mod.close
```

### Tensors, dtypes & ecosystem adapters

Canonical dtypes: `:float32`, `:float64`, `:float16`, `:bfloat16`, `:uint8`, `:int8`, `:int16`, `:uint16`, `:int32`, `:uint32`, `:int64`, `:uint64`, `:bool`. Short aliases (`:float`, `:double`, `:half`, `:byte`, `:char`, `:short`, `:int`, `:long`) are accepted on input.

```ruby
t = Executorch::Tensor.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t.shape  # => [2, 3]
t.dtype  # => :float32
t.nbytes # => 24
t.to_a   # => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

# 0-D scalar tensors
scalar = Executorch::Tensor.new(42.5, dtype: :float32)
scalar.shape # => []
scalar.item  # => 42.5

# Binary I/O & factories
raw = t.to_bytes
t2  = Executorch::Tensor.from_bytes(raw, shape: [2, 3], dtype: :float32)
t == t2 # => true

zeros = Executorch::Tensor.zeros([2, 3], dtype: :float32)
ones  = Executorch::Tensor.ones([2, 3], dtype: :int64)
```

Optional adapters are built in for `Numo::NArray` (`Tensor.from_numo` / `Tensor#to_numo`) and `Torch::Tensor` (`Tensor.from_torch`).

### Runtime introspection & typed errors

```ruby
Executorch.backends           # => ["XnnpackBackend"]
Executorch.kernel_count       # => 263
Executorch.num_threads        # => 24
Executorch.executorch_version # => "1.6.0"
```

All runtime failures inherit from `Executorch::Error < StandardError` (`LoadError`, `MethodNotFoundError`, `OperatorMissingError`, `DelegateError`, `ShapeError`, `DTypeError`, `ExecutionError`, `MemoryError`, `ClosedError`) and expose `#code` (ExecuTorch error code) and `#pal_log` (captured PAL diagnostics).

### Using a custom ExecuTorch installation

If you built ExecuTorch yourself (shared or static), point `extconf.rb` at its absolute prefix:

```bash
bundle config set --local build.executorch --with-executorch-dir=/absolute/path/to/executorch
```

Or pass `EXECUTORCH_DIR=/absolute/path/to/executorch`. Set `EXECUTORCH_NO_DOWNLOAD=1` to disable automatic SDK fetching.

## Upgrading from 0.1.x

- `Executorch::Model` is replaced by `Executorch::Module` (`Module.open`, `#call`, `#forward`, `#execute`, `#meta`, `#close`).
- Canonical dtype symbols are now explicit-width (`:float32`, `:int64`, etc.); short aliases remain accepted on input.
- Minimum Ruby version is now `>= 3.2`.
- `rice` is no longer a dependency.

## Development

```bash
bundle install
bundle exec rake sdk:fetch        # Fetches verified ExecuTorch 1.6.0 C++ SDK into ~/.cache/executorch-ruby/
bundle exec rake compile          # Compiles native C++17 extension
bundle exec rake fixtures:export  # Exports deterministic .pte/.ptd test models via uv
bundle exec rake test             # Runs full test suite (0 skips)
ruby bench/run_bench.rb           # Runs local latency benchmarks
```

To build and test under AddressSanitizer + UndefinedBehaviorSanitizer on Linux:

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

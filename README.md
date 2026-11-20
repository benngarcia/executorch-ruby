# Executorch for Ruby

Native Ruby bindings for Meta's [ExecuTorch](https://pytorch.org/executorch/) runtime (`1.6.0`).

- **Zero-config install**: Precompiled platform gems (`x86_64-linux`, `aarch64-linux`, `arm64-darwin`) bundle ExecuTorch + XNNPACK + optimized & quantized kernels (+ CoreML/MLX on Apple Silicon). Source fallback installs auto-fetch the SHA-256-verified official ExecuTorch C++ SDK using Ruby stdlib — no CMake, Python, or Buck2 required.
- **Zero runtime gem dependencies**: Direct Ruby C API over a `noexcept` C++17 core (no Rice, strict symbol hiding — only `Init_executorch` is exported).
- **Production Ruby VM citizenship**: Releases the GVL during model load and inference with per-method thread-safety locks, reports native memory to Ruby's GC (`dsize` + `rb_gc_adjust_memory_usage`), and supports deterministic `#close` / block-form `Module.open`.
- **Full ExecuTorch surface**: `mmap` default loading, `Module.from_bytes`, external `.ptd` weights, `MethodMeta` introspection, multi-method execution, all 13 numeric/bool dtypes (including `:float16`, `:bfloat16`, `:uint8`, `:bool`), zero-copy `IO::Buffer`, pre-allocated `out:` output reuse, and typed `Executorch::Error` exceptions carrying PAL diagnostics.

## Requirements

- Ruby `>= 3.2`
- Linux (`x86_64`, `aarch64`) or macOS (`arm64`)

## Installation

Add to your Gemfile:

```ruby
gem "executorch", "~> 1.0"
```

Or install directly:

```bash
gem install executorch
```

### Using a custom ExecuTorch installation

If you built ExecuTorch yourself (shared or static), point `extconf.rb` at its absolute prefix:

```bash
bundle config set --local build.executorch --with-executorch-dir=/absolute/path/to/executorch
```

Or pass `EXECUTORCH_DIR=/absolute/path/to/executorch`. Set `EXECUTORCH_NO_DOWNLOAD=1` to disable automatic SDK fetching.

## Usage

### Basic inference

```ruby
require "executorch"

# Defaults to :mmap for copy-on-write page sharing across forked Puma workers.
# Accepts String or Pathname; block form closes deterministically.
Executorch::Module.open("model.pte") do |mod|
  meta = mod.meta(:forward)
  p meta.backends # => ["XnnpackBackend"]
  p meta.inputs   # => [#<Executorch::TensorInfo dtype=:float32 shape=[1, 3] nbytes=12>]

  input = Executorch::Tensor.new([[1.0, 2.0, 3.0]])
  output = mod.call(input)
  p output.to_a   # => [[3.0, 5.0, 7.0]]
end
```

### Zero-copy `IO::Buffer` and pre-allocated output reuse

```ruby
Executorch.num_threads = 2 # Prevent oversubscription in multi-worker servers

in_buf  = IO::Buffer.for([1.0, 2.0, 3.0].pack("f*"))
out_buf = IO::Buffer.new(12)

input  = Executorch::Tensor.from_buffer(in_buf, shape: [1, 3], dtype: :float32)
output = Executorch::Tensor.from_buffer(out_buf, shape: [1, 3], dtype: :float32)

Executorch::Module.open("model.pte") do |mod|
  mod.call(input, out: output) # Releases GVL; writes directly into out_buf
  p output.to_a
end
```

### Multi-method models, external `.ptd` weights, and in-memory loading

```ruby
# Multi-method .pte
Executorch::Module.open("autoencoder.pte") do |mod|
  z = mod.execute(:encode, x).first
  reconstructed = mod.execute(:decode, z).first
end

# External .ptd weights
Executorch::Module.open("model.pte", data_files: ["weights.ptd"], load_mode: :mmap_mlock) do |mod|
  logits = mod.call(tokens)
end

# Load straight from memory (e.g., S3 / ActiveStorage)
mod = Executorch::Module.from_bytes(pte_bytes)
mod.close
```

### Tensors & dtypes

Canonical dtypes: `:float32`, `:float64`, `:float16`, `:bfloat16`, `:uint8`, `:int8`, `:int16`, `:uint16`, `:int32`, `:uint32`, `:int64`, `:uint64`, `:bool`. Short aliases (`:float`, `:double`, `:half`, `:byte`, `:char`, `:short`, `:int`, `:long`) are accepted at construction.

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

Optional adapters are provided for `Numo::NArray` (`Tensor.from_numo` / `Tensor#to_numo`) and `Torch::Tensor` (`Tensor.from_torch`).

### Runtime introspection & errors

```ruby
Executorch.backends           # => ["XnnpackBackend"]
Executorch.kernel_count       # => 263
Executorch.num_threads        # => 24
Executorch.executorch_version # => "1.6.0"
```

All runtime failures inherit from `Executorch::Error < StandardError` (`LoadError`, `MethodNotFoundError`, `OperatorMissingError`, `DelegateError`, `ShapeError`, `DTypeError`, `ExecutionError`, `MemoryError`, `ClosedError`) and expose `#code` (ExecuTorch error code) and `#pal_log` (captured PAL diagnostics).

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

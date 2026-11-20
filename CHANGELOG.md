# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-19

Complete from-scratch rewrite targeting ExecuTorch 1.6.0 and Ruby >= 3.2.

### Breaking Changes

- Replaced `Executorch::Model` with `Executorch::Module` (`Module.open`, `Module.from_bytes`, `#call`, `#forward`, `#execute`, `#meta`, `#load_method`, `#unload_method`, `#close`).
- Canonical tensor dtype symbols are now explicit-width (`:float32`, `:float64`, `:float16`, `:bfloat16`, `:uint8`, `:int8`, `:int16`, `:uint16`, `:int32`, `:uint32`, `:int64`, `:uint64`, `:bool`). Short aliases (`:float`, `:double`, `:int`, `:long`, etc.) remain accepted on input.
- Raised minimum Ruby version to `>= 3.2.0`.
- Removed `rice` dependency and leaked internal methods (`_original_to_a`, `Tensor.create`, `Tensor.from_array`, `NATIVE_VERSION`, `NativeError`).

### Added

- Two-layer `noexcept` C++17 core beneath a direct Ruby C-API seam — eliminates `rb_raise` `longjmp`-past-destructor UB by construction.
- Strict symbol visibility (`-fvisibility=hidden` + ELF version script / macOS `-exported_symbol,_Init_executorch`) preventing ODR collisions with `torch.rb`, `onnxruntime-ruby`, or `ruby-vips`.
- GVL release (`rb_thread_call_without_gvl`) during module load, method load, and execution, paired with per-method thread-safety mutexes and `RB_GC_GUARD`.
- Native memory pressure reporting (`dsize` + `rb_gc_adjust_memory_usage`) and deterministic `#close` / block-form `Module.open`.
- Support for all 13 numeric/boolean dtypes, 0-D scalar tensors (`Tensor#item`), non-contiguous / `NHWC` `dim_order` normalization, value equality (`Tensor#==`), `Tensor.zeros`/`ones`/`full`, `Tensor.from_bytes`/`#to_bytes`, zero-copy `IO::Buffer` (`Tensor.from_buffer`/`#to_buffer`), pre-allocated `out:` output reuse, and optional `Numo::NArray`/`Torch::Tensor` adapters.
- `Module.open` defaults to `:mmap` (supporting `Pathname`, all `LoadMode`s, and external `.ptd` `data_files`), plus `Module.from_bytes` (`BufferDataLoader`) and `MethodMeta` / `TensorInfo` introspection.
- Runtime introspection (`Executorch.backends`, `.kernel_count`, `.num_threads`, `.num_threads=`, `.executorch_version`), PAL initialization (`runtime_init`), and thread-local PAL error log capture attached to typed `Executorch::Error` subclasses (`LoadError`, `MethodNotFoundError`, `OperatorMissingError`, `DelegateError`, `ShapeError`, `DTypeError`, `ExecutionError`, `MemoryError`, `ClosedError`).
- Precompiled platform gems (`x86_64-linux`, `aarch64-linux`, `arm64-darwin`) bundling the official ExecuTorch 1.6.0 shared SDK (XNNPACK, optimized & quantized kernels, CoreML/MLX on Apple Silicon) plus a pure-stdlib SHA-256-verified SDK fetcher for source installs — no Python, CMake, or Buck2 required at install or inference time.
- Deterministic `.pte`/`.ptd` test fixture export pipeline (`test/fixtures/export_fixtures.py`) and comprehensive correctness, ASan/UBSan, GC stress, concurrency, and adversarial safety test suites (0 skipped tests).

## [0.1.0] - 2025-11-16

Initial prototype release.

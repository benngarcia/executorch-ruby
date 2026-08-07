# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Tensor.from_bytes(packed, shape:, dtype:)` and `Tensor#to_binary` - raw
  binary tensor I/O that skips per-element conversion. Roughly 174x faster than
  the Array constructor on a 150k-element input.
- `bench/` - eval and profiling harness. Converts `.pt` checkpoints to `.pte`,
  checks output against PyTorch, and times each leg of an inference call. See
  `bench/FINDINGS.md`.
- `script/build-executorch.sh` - reproducible ExecuTorch build used by both
  humans and CI.
- CI (GitHub Actions) for lint and tests across Ruby 3.1-3.4 on Linux and
  macOS, plus a tag-triggered RubyGems release workflow.

### Changed

- Large reduction in Ruby/C++ boundary cost. Tensor creation is ~13x faster,
  `#to_a` up to ~65x faster, and building a tensor from a nested Array now
  allocates 11 objects instead of 153,238 for a 150k-element input.
- `extconf.rb` now links optimized CPU kernels and the XNNPACK delegate when
  they are present in the ExecuTorch install, whole-archiving them so their
  self-registration is not dropped by the linker.

### Fixed

- The documented install path did not work: `cmake --install` does not install
  `extension/module/module.h` or the `runtime/executor` headers, so
  `rake compile` failed with "module.h header not found" on a correct build.
  `script/build-executorch.sh` copies them.
- `extconf.rb` whole-archived two operator libraries when both were present,
  which aborts the runtime at init on duplicate operator registration.
- Build failure against Rice 4.12, where `Rice::Array::Proxy` no longer
  converts implicitly to `Rice::Object`.

## [0.1.0] - 2024-12-27

### Added

- `Executorch::Model` - Load and execute `.pte` models
  - `Model.new(path)` - Load model from file
  - `#predict(inputs)` - Run inference (recommended)
  - `#forward(inputs)` - Run inference (PyTorch-style alias)
  - `#call(inputs)` - Run inference (Ruby-style, makes model callable)
  - `#execute(method_name, inputs)` - Execute named method
  - `#method_names` - List available methods
  - `#loaded?` - Check if model is loaded
- `Executorch::Tensor` - Multi-dimensional array support
  - `Tensor.new(nested_data)` - Create from nested array (shape inferred)
  - `Tensor.new(flat_data, shape:, dtype:)` - Create from flat array with explicit shape
  - Supported dtypes: `:float`, `:double`, `:int`, `:long`
  - `#to_a` - Convert to nested Ruby array matching shape
  - `#flat_to_a` - Convert to flat Ruby array
  - `#shape`, `#dim`, `#numel`, `#dtype` - Inspection methods
- Build configuration via `--with-executorch-dir` flag or `EXECUTORCH_DIR` environment variable
  - `bundle config set --local build.executorch --with-executorch-dir=/path` (recommended)
  - `EXECUTORCH_DIR=/path bundle install` (for CI)
  - `EXECUTORCH_EXTRA_LIBS` for additional backend libraries
- Test models and export script

### Known Limitations

- Bool tensors (`:bool` dtype) not yet supported for creation
- No prebuilt binaries - requires ExecuTorch built from source
- macOS and Linux only (no Windows)

### Design Decisions

- Tensors are cloned when passed to inference methods for memory safety (slight performance overhead)
- dtype symbols use short names (`:int`, `:long`) rather than explicit sizes (`:int32`, `:int64`)
- EValue (ExecuTorch's internal tagged union) is not exposed; scalar outputs are auto-converted to Ruby types

## [Unreleased]

_No unreleased changes._

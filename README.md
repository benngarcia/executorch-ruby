# ExecuTorch Ruby

[![CI](https://github.com/benngarcia/executorch-ruby/actions/workflows/ci.yml/badge.svg)](https://github.com/benngarcia/executorch-ruby/actions/workflows/ci.yml)

Run PyTorch models in Ruby.

[ExecuTorch](https://pytorch.org/executorch/) is Meta's lightweight runtime for deploying PyTorch models on edge devices. This gem provides Ruby bindings so you can run exported models (`.pte` files) directly in your Ruby applications.

## Quick Start

```ruby
require "executorch"

# Load a model
model = Executorch::Model.new("model.pte")

# Create input tensor
input = Executorch::Tensor.new([[1.0, 2.0, 3.0]])

# Run inference
output = model.predict([input]).first
puts output.to_a  # => [[3.0, 5.0, 7.0]]
```

## Installation

**Requirements:** Ruby 3.0+, macOS or Linux, C++17 compiler

### Step 1: Build ExecuTorch

ExecuTorch must be built from source. This repo ships a script that does it:

```bash
script/build-executorch.sh
```

It clones ExecuTorch at a pinned version, builds it, installs into
`vendor/executorch`, and copies the headers that `cmake --install` leaves
behind (`extension/module/module.h` and the `runtime/executor` tree are not
installed by ExecuTorch's own install step, and this gem includes them
directly).

Add the XNNPACK delegate — strongly recommended, see [Performance](#performance)
— with:

```bash
EXECUTORCH_BACKENDS=xnnpack script/build-executorch.sh
```

Needs CMake ≥ 3.29, Ninja, and a C++17 compiler. The first build takes a while;
the ExecuTorch source and submodules are several GB.

### Step 2: Install the Gem

Tell Bundler where ExecuTorch is installed (only needed once per project):

```bash
bundle config set --local build.executorch --with-executorch-dir=vendor/executorch
```

Add to your Gemfile:

```ruby
gem "executorch"
```

Then:

```bash
bundle install
```

> **CI/CD:** Use the environment variable instead: `EXECUTORCH_DIR=/path/to/executorch bundle install`

## Usage

### Tensors

Create tensors from nested arrays (shape is inferred):

```ruby
# 2D tensor, shape [2, 3]
tensor = Executorch::Tensor.new([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])

# With explicit dtype
tensor = Executorch::Tensor.new([[1, 2], [3, 4]], dtype: :long)

# Inspect
tensor.shape   # => [2, 3]
tensor.dtype   # => :float
tensor.to_a    # => [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

Or from flat arrays with explicit shape:

```ruby
tensor = Executorch::Tensor.new([1.0, 2.0, 3.0, 4.0], shape: [2, 2])
```

**Supported dtypes:** `:float` (default), `:double`, `:int`, `:long`

For large tensors, skip per-element conversion entirely and hand over packed
bytes — the runtime memcpys them straight into the tensor buffer:

```ruby
# ~13x faster than the Array constructor on a 150k-element input
tensor = Executorch::Tensor.from_bytes(pixels.pack("f*"), shape: [1, 3, 224, 224])

# and back out
bytes = tensor.to_binary
values = bytes.unpack("f*")
```

This is the right path whenever your data is already bytes (an image decoded to
a string, a file, a socket) or when you can pack once and reuse. Data must be
native-endian and match the dtype's element width: `:float` → `"f*"`,
`:double` → `"d*"`, `:int` → `"l*"`, `:long` → `"q*"`.

### Models

```ruby
model = Executorch::Model.new("model.pte")

# Run inference (all equivalent)
outputs = model.predict([input])
outputs = model.forward([input])
outputs = model.call([input])

# Introspection
model.loaded?       # => true
model.method_names  # => ["forward"]
```

## Exporting Models

Export PyTorch models to `.pte` format:

```python
import torch
from executorch.exir import to_edge

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 1

model = MyModel()
example_input = torch.randn(1, 3)

exported = torch.export.export(model, (example_input,))
et_program = to_edge(exported).to_executorch()

with open("model.pte", "wb") as f:
    et_program.write_to_file(f)
```

## Performance

The single biggest factor in inference speed is **which kernels your ExecuTorch
build links** — not the Ruby layer.

A default build uses the portable kernels: reference implementations written for
correctness and portability, with no vectorization or threading. They work
everywhere and they are slow — resnet18 takes **8.3 s** per call.

Build with the XNNPACK delegate instead:

```bash
cmake -B cmake-out \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  ... # other flags as above
```

and export your model through the XNNPACK partitioner (see
`bench/pt_to_pte.py --xnnpack`). Both halves are required — the backend has to
be linked at build time *and* targeted at export time.

That takes resnet18 from 8.3 s to **12.6 ms** — 658×, and about 2× faster than
PyTorch eager on the same machine. `extconf.rb` links the backend automatically
when it finds it in your ExecuTorch install.

On the Ruby side, once the runtime is fast the boundary becomes the bottleneck:

- Prefer `Tensor.from_bytes` over the Array constructor for large inputs. On
  mobilenet_v2 that's 0.85 ms instead of 7.6 ms, against a 3.4 ms inference.
- Prefer `Tensor#flat_to_a` over `#to_a` when you don't need the nested shape.
- Prefer `Tensor.new(flat, shape: ...)` over a nested Array when you have the
  choice — shape inference has to walk the nesting.

See [`bench/`](bench/) for the eval + profiling harness, and
[`bench/FINDINGS.md`](bench/FINDINGS.md) for the full walkthrough — including
why fixing the kernels is what makes the binding work matter.

## Troubleshooting

<details>
<summary><strong>"ExecuTorch installation not found"</strong></summary>

Verify your installation and configure the path:

```bash
ls vendor/executorch/include/executorch  # Should exist
bundle config set --local build.executorch --with-executorch-dir=vendor/executorch
```
</details>

<details>
<summary><strong>"module.h header not found"</strong></summary>

Usually not a missing build flag — `cmake --install` doesn't install
`extension/module/module.h` at all, even on a correct build. If you built
ExecuTorch by hand, copy the remaining headers across:

```bash
cd /path/to/executorch
find runtime kernels extension -name '*.h' -not -path '*/test/*' \
  | while read -r f; do
      mkdir -p "$PREFIX/include/executorch/$(dirname "$f")"
      cp "$f" "$PREFIX/include/executorch/$f"
    done
```

Or just use `script/build-executorch.sh`, which handles it.
</details>

<details>
<summary><strong>"undefined symbol" at runtime</strong></summary>

Try linking additional libraries:

```bash
EXECUTORCH_EXTRA_LIBS=portable_ops_lib,portable_kernels bundle exec rake compile
```
</details>

## Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [Changelog](CHANGELOG.md)
- [Report an Issue](https://github.com/benngarcia/executorch-ruby/issues)

## Development

```bash
git clone https://github.com/benngarcia/executorch-ruby.git
cd executorch-ruby
bundle install
bundle config set --local build.executorch --with-executorch-dir=vendor/executorch
bundle exec rake compile
bundle exec rake test
```

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/benngarcia/executorch-ruby).

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).

## Acknowledgments

Built with [Rice](https://github.com/jasonroelofs/rice). Inspired by [onnxruntime-ruby](https://github.com/ankane/onnxruntime-ruby) and [torch.rb](https://github.com/ankane/torch.rb).

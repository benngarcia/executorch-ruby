# ExecuTorch Ruby

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

ExecuTorch must be built from source. Follow the [official guide](https://pytorch.org/executorch/stable/getting-started-setup.html), or use these commands:

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
./install_requirements.sh

cmake -B cmake-out \
  -DCMAKE_INSTALL_PREFIX=vendor/executorch \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build cmake-out -j4
cmake --install cmake-out
```

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

ExecuTorch was built without required extensions. Rebuild with:

```bash
cmake -B cmake-out \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  ...
```
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

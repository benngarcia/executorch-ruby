# Test Models

This directory should contain `.pte` (PyTorch ExecuTorch) model files for testing.

## Creating Test Models

You need Python with PyTorch and ExecuTorch installed:

```bash
pip install torch executorch
```

### Simple Model (simple.pte)

Create a simple model that doubles and adds 1:

```python
# scripts/export_simple_model.py
import torch
from executorch.exir import to_edge

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 1

model = SimpleModel()
example = torch.randn(1, 3)

# Export to ExecuTorch format
exported = torch.export.export(model, (example,))
edge = to_edge(exported)
et_program = edge.to_executorch()

# Save to file
with open("test/support/models/simple.pte", "wb") as f:
    et_program.write_to_file(f)

print("Created simple.pte")
```

### Linear Model (linear.pte)

A model with a linear layer:

```python
# scripts/export_linear_model.py
import torch
from executorch.exir import to_edge

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
example = torch.randn(1, 10)

exported = torch.export.export(model, (example,))
edge = to_edge(exported)
et_program = edge.to_executorch()

with open("test/support/models/linear.pte", "wb") as f:
    et_program.write_to_file(f)

print("Created linear.pte")
```

## Running Export Scripts

```bash
mkdir -p test/support/models
python scripts/export_simple_model.py
python scripts/export_linear_model.py
```

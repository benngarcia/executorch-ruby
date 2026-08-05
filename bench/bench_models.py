"""Model definitions used by the executorch-ruby benchmark suite.

This module is imported by both ``make_pt_models.py`` (which pickles the
modules into ``.pt`` files) and ``pt_to_pte.py`` (which unpickles them), so the
classes must live here rather than in ``__main__`` -- ``torch.save`` on a whole
module stores a reference to the defining module path.

The suite deliberately spans three regimes:

* ``overhead`` -- so little math that wall time is dominated by the Ruby <-> C++
  boundary. These are the models that tell you whether the bindings are good.
* ``mixed``    -- real work, but small enough that per-call overhead still shows.
* ``compute``  -- the kernels dominate; binding cost should vanish into noise.
"""

import torch
import torch.nn as nn


class AddMul(nn.Module):
    """The model from the README: ``x * 2 + 1``. Pure overhead probe."""

    def forward(self, x):
        return x * 2 + 1


class TinyMLP(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.fc = nn.Linear(size, size)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, size=512, depth=2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(size, size), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MnistCNN(nn.Module):
    """Small conv net in the shape of the classic MNIST example."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def _torchvision(name):
    """Build an untrained torchvision model, or return None if unavailable.

    Weights are random on purpose: the accuracy check compares Ruby against
    this exact checkpoint, so trained weights would only add a download.
    """
    try:
        import torchvision.models as tvm
    except ImportError:
        return None
    return getattr(tvm, name)(weights=None)


# name -> (builder, input shape, regime)
REGISTRY = {
    "add_mul":      (lambda: AddMul(),        (1, 3),            "overhead"),
    "tiny_mlp":     (lambda: TinyMLP(8),      (1, 8),            "overhead"),
    "mlp_512x2":    (lambda: MLP(512, 2),     (1, 512),          "mixed"),
    "mlp_1024x4":   (lambda: MLP(1024, 4),    (1, 1024),         "compute"),
    "mnist_cnn":    (lambda: MnistCNN(),      (1, 1, 28, 28),    "mixed"),
    "resnet18":     (lambda: _torchvision("resnet18"),     (1, 3, 224, 224), "compute"),
    "mobilenet_v2": (lambda: _torchvision("mobilenet_v2"), (1, 3, 224, 224), "compute"),
}

DEFAULT_MODELS = ["add_mul", "tiny_mlp", "mlp_512x2", "mnist_cnn", "mlp_1024x4"]

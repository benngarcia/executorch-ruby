#!/usr/bin/env python3
"""Export deterministic ExecuTorch 1.6.0 test fixtures (.pte / .ptd + golden JSON).

Every model is seeded deterministically and paired with golden PyTorch outputs so
Ruby tests verify exact numerical accuracy, shapes, dtypes, delegates, multi-method
dispatch, external .ptd weights, scalar EValues, and GVL concurrency.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower


def tensor_to_golden(t: torch.Tensor) -> dict[str, Any]:
    cpu = t.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16 or cpu.dtype == torch.float16:
        values = cpu.to(torch.float32).flatten().tolist()
    elif cpu.dtype == torch.bool:
        values = [bool(x) for x in cpu.flatten().tolist()]
    else:
        values = cpu.flatten().tolist()
    dtype_map = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    return {
        "shape": list(cpu.shape),
        "dtype": dtype_map[cpu.dtype],
        "values": values,
    }


def export_program(
    out_dir: Path,
    name: str,
    programs: dict[str, torch.export.ExportedProgram],
    partitioners: list[Any] | None = None,
    external_constants: bool = False,
) -> Path:
    edge_kwargs: dict[str, Any] = {
        "compile_config": EdgeCompileConfig(_check_ir_validity=False),
    }
    if partitioners:
        edge_kwargs["partitioner"] = partitioners

    edge = to_edge_transform_and_lower(programs, **edge_kwargs)
    et_cfg = ExecutorchBackendConfig(
        extract_delegate_segments=True,
        external_constants=external_constants,
    )
    et_prog = edge.to_executorch(config=et_cfg)

    pte_path = out_dir / f"{name}.pte"
    with open(pte_path, "wb") as f:
        et_prog.write_to_file(f)

    if external_constants:
        et_prog.write_tensor_data_to_file(str(out_dir))
    return pte_path


class AddMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 + 1.0


class MultiIO(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return a + b, a * b


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class DtypeEcho(nn.Module):
    def forward(
        self,
        f64: torch.Tensor,
        i64: torch.Tensor,
        u8: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return f64 + 0.5, i64 + 1, u8 + 2, torch.logical_not(mask)


class SmallCnnXnnpack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExternalWeightsMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class HeavyMlp(nn.Module):
    """Big enough (~15-40ms on CPU) to cleanly prove GVL release & parallel scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for fixtures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    manifest: dict[str, Any] = {
        "torch_version": torch.__version__,
        "models": {},
    }

    # 1. add_mul (simple 1-in/1-out)
    m = AddMul().eval()
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y = m(x)
    export_program(out_dir, "add_mul", {"forward": torch.export.export(m, (x,))})
    manifest["models"]["add_mul"] = {
        "file": "add_mul.pte",
        "inputs": [tensor_to_golden(x)],
        "outputs": [tensor_to_golden(y)],
    }

    # 2. multi_io (2 inputs, 2 outputs)
    m = MultiIO().eval()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    y1, y2 = m(a, b)
    export_program(out_dir, "multi_io", {"forward": torch.export.export(m, (a, b))})
    manifest["models"]["multi_io"] = {
        "file": "multi_io.pte",
        "inputs": [tensor_to_golden(a), tensor_to_golden(b)],
        "outputs": [tensor_to_golden(y1), tensor_to_golden(y2)],
    }

    # 3. multi_method (encode & decode methods in one .pte)
    enc = Encoder().eval()
    dec = Decoder().eval()
    x_enc = torch.tensor([[0.5, -1.0, 2.0, 0.25]], dtype=torch.float32)
    z = enc(x_enc)
    x_dec = dec(z)
    export_program(
        out_dir,
        "multi_method",
        {
            "encode": torch.export.export(enc, (x_enc,)),
            "decode": torch.export.export(dec, (z,)),
        },
    )
    manifest["models"]["multi_method"] = {
        "file": "multi_method.pte",
        "methods": {
            "encode": {"inputs": [tensor_to_golden(x_enc)], "outputs": [tensor_to_golden(z)]},
            "decode": {"inputs": [tensor_to_golden(z)], "outputs": [tensor_to_golden(x_dec)]},
        },
    }

    # 4. dtypes (float64, int64, uint8, bool)
    m = DtypeEcho().eval()
    f64 = torch.tensor([1.25, -2.5], dtype=torch.float64)
    i64 = torch.tensor([10, -20, 30], dtype=torch.int64)
    u8 = torch.tensor([[0, 10, 250]], dtype=torch.uint8)
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    outs = m(f64, i64, u8, mask)
    export_program(out_dir, "dtypes", {"forward": torch.export.export(m, (f64, i64, u8, mask))})
    manifest["models"]["dtypes"] = {
        "file": "dtypes.pte",
        "inputs": [tensor_to_golden(t) for t in (f64, i64, u8, mask)],
        "outputs": [tensor_to_golden(t) for t in outs],
    }

    # 5. cnn_xnnpack (real Conv2d lowered to XnnpackBackend)
    m = SmallCnnXnnpack().eval()
    img = torch.randn(1, 3, 16, 16, dtype=torch.float32)
    out_cnn = m(img)
    export_program(
        out_dir,
        "cnn_xnnpack",
        {"forward": torch.export.export(m, (img,))},
        partitioners=[XnnpackPartitioner()],
    )
    manifest["models"]["cnn_xnnpack"] = {
        "file": "cnn_xnnpack.pte",
        "inputs": [tensor_to_golden(img)],
        "outputs": [tensor_to_golden(out_cnn)],
        "expected_backends": ["XnnpackBackend"],
    }

    # 6. external_weights (.pte + .ptd)
    m = ExternalWeightsMlp().eval()
    x_ext = torch.randn(2, 16, dtype=torch.float32)
    y_ext = m(x_ext)
    ptd_dir = out_dir / "external_weights_data"
    ptd_dir.mkdir(exist_ok=True)
    export_program(
        ptd_dir,
        "external_weights",
        {"forward": torch.export.export(m, (x_ext,))},
        external_constants=True,
    )
    ptd_files = sorted(str(p.relative_to(out_dir)) for p in ptd_dir.glob("*.ptd"))
    manifest["models"]["external_weights"] = {
        "file": "external_weights_data/external_weights.pte",
        "data_files": ptd_files,
        "inputs": [tensor_to_golden(x_ext)],
        "outputs": [tensor_to_golden(y_ext)],
    }

    # 7. heavy_mlp (XNNPACK-delegated batch=32 MLP for GVL & multi-thread proof)
    m = HeavyMlp().eval()
    x_h = torch.randn(32, 512, dtype=torch.float32)
    y_h = m(x_h)
    export_program(
        out_dir,
        "heavy_mlp",
        {"forward": torch.export.export(m, (x_h,))},
        partitioners=[XnnpackPartitioner()],
    )
    manifest["models"]["heavy_mlp"] = {
        "file": "heavy_mlp.pte",
        "inputs": [tensor_to_golden(x_h)],
        "outputs": [tensor_to_golden(y_h)],
        "expected_backends": ["XnnpackBackend"],
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {len(manifest['models'])} fixture models to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Produce the ``.pt`` benchmark checkpoints plus golden inputs/outputs.

Each model in ``bench_models.REGISTRY`` becomes three files under
``bench/models/``:

    <name>.pt        eager nn.Module, pickled with torch.save
    <name>.meta.json shapes, dtypes, regime, and the eager reference latency
    <name>.io.bin    golden input followed by golden output, raw float32 LE

The ``.bin`` sidecar keeps the Ruby harness honest: it compares its own output
against numbers PyTorch produced, and reading raw floats costs nothing next to
parsing a 150k-element JSON array.

Usage:
    python3 bench/make_pt_models.py                    # default set
    python3 bench/make_pt_models.py --models resnet18  # specific models
    python3 bench/make_pt_models.py --all
"""

import argparse
import json
import os
import struct
import time

import torch

import bench_models

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")


def write_floats(fh, tensor):
    flat = tensor.detach().contiguous().flatten().tolist()
    fh.write(struct.pack(f"<{len(flat)}f", *flat))


def time_eager(model, example, iters=30):
    """Reference latency: the same model, same input, run by PyTorch itself."""
    with torch.no_grad():
        for _ in range(5):
            model(example)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(example)
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def build(name, out_dir):
    builder, shape, regime = bench_models.REGISTRY[name]

    # Seed before constructing so a model's weights (and therefore its golden
    # output) are the same whether you build one model or all of them.
    torch.manual_seed(0)
    model = builder()
    if model is None:
        print(f"  skip {name}: torchvision not installed")
        return None

    model.eval()
    example = torch.randn(*shape)

    with torch.no_grad():
        expected = model(example)
    if not isinstance(expected, torch.Tensor):
        raise TypeError(f"{name}: expected a single tensor output, got {type(expected)}")

    eager_ms = time_eager(model, example)

    pt_path = os.path.join(out_dir, f"{name}.pt")
    torch.save(model, pt_path)

    with open(os.path.join(out_dir, f"{name}.io.bin"), "wb") as fh:
        write_floats(fh, example)
        write_floats(fh, expected)

    meta = {
        "name": name,
        "regime": regime,
        "input_shape": list(example.shape),
        "output_shape": list(expected.shape),
        "input_numel": example.numel(),
        "output_numel": expected.numel(),
        "dtype": "float",
        "params": sum(p.numel() for p in model.parameters()),
        "eager_p50_ms": round(eager_ms, 4),
        "torch_version": torch.__version__,
    }
    with open(os.path.join(out_dir, f"{name}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"  {name:14s} {str(list(example.shape)):18s} "
        f"params={meta['params']:>9,d}  eager p50={eager_ms:7.3f} ms  -> {pt_path}"
    )
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="*",
        help="model names, space- or comma-separated (default: the standard set)",
    )
    ap.add_argument("--all", action="store_true", help="build every registered model")
    ap.add_argument("--out", default=MODELS_DIR)
    args = ap.parse_args()

    if args.all:
        names = list(bench_models.REGISTRY)
    elif args.models:
        names = [n for arg in args.models for n in arg.split(",") if n]
    else:
        names = bench_models.DEFAULT_MODELS

    unknown = [n for n in names if n not in bench_models.REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown model(s): {', '.join(unknown)}\n"
            f"available: {', '.join(bench_models.REGISTRY)}"
        )

    os.makedirs(args.out, exist_ok=True)
    print(f"Writing .pt benchmarks to {args.out}")
    for name in names:
        build(name, args.out)


if __name__ == "__main__":
    main()

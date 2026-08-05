#!/usr/bin/env python3
"""Convert a PyTorch ``.pt`` checkpoint into an ExecuTorch ``.pte`` program.

This is the whole `.pt` -> `.pte` story in one place:

    torch.load  ->  torch.export.export  ->  to_edge  ->  to_executorch

``torch.export`` traces the module into an ATen graph with static shapes, which
is why an example input is required; ``to_edge`` lowers that graph to the Edge
dialect (the op set the ExecuTorch runtime knows); ``to_executorch`` serializes
it into the flatbuffer that ``Executorch::Model.new`` mmaps.

Usage:
    # shapes read from the sidecar written by make_pt_models.py
    python3 bench/pt_to_pte.py bench/models/tiny_mlp.pt

    # or supply them yourself for any .pt you have lying around
    python3 bench/pt_to_pte.py my_model.pt --input-shape 1,3,224,224 -o my_model.pte

Note: ``torch.load`` unpickles arbitrary code. Only point this at checkpoints
you produced or otherwise trust.
"""

import argparse
import json
import os
import sys
import time

import torch
from executorch.exir import to_edge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so pickled bench models resolve


def parse_shape(text):
    return tuple(int(part) for part in text.replace("x", ",").split(",") if part.strip())


def load_shapes(pt_path, explicit):
    if explicit:
        return [parse_shape(s) for s in explicit]

    meta_path = pt_path.replace(".pt", ".meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            return [tuple(json.load(fh)["input_shape"])]

    raise SystemExit(
        f"no input shape given and no sidecar at {meta_path}; pass --input-shape 1,3,224,224"
    )


def convert(pt_path, pte_path, shapes, verbose=True):
    model = torch.load(pt_path, weights_only=False)
    model.eval()

    example = tuple(torch.randn(*shape) for shape in shapes)

    t0 = time.perf_counter()
    exported = torch.export.export(model, example)
    t_export = time.perf_counter() - t0

    t0 = time.perf_counter()
    program = to_edge(exported).to_executorch()
    t_lower = time.perf_counter() - t0

    with open(pte_path, "wb") as fh:
        program.write_to_file(fh)

    size_kb = os.path.getsize(pte_path) / 1024.0
    if verbose:
        print(
            f"{os.path.basename(pt_path)} -> {os.path.basename(pte_path)}  "
            f"({size_kb:,.1f} KiB, export {t_export:.2f}s, lower {t_lower:.2f}s)"
        )
    return pte_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="+", help="one or more .pt files")
    ap.add_argument("-o", "--output", help="output .pte path (single input only)")
    ap.add_argument(
        "--input-shape",
        action="append",
        help="example input shape, e.g. 1,3,224,224. Repeat for multi-input models. "
             "Defaults to the shape in the .meta.json sidecar.",
    )
    args = ap.parse_args()

    if args.output and len(args.checkpoint) > 1:
        raise SystemExit("-o only makes sense with a single checkpoint")

    for pt_path in args.checkpoint:
        pte_path = args.output or pt_path.replace(".pt", ".pte")
        convert(pt_path, pte_path, load_shapes(pt_path, args.input_shape))


if __name__ == "__main__":
    main()

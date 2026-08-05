# executorch-ruby benchmarks

An eval + profiling harness for the gem. It answers two questions:

1. **Are we right?** Does Ruby produce the same numbers PyTorch does?
2. **Where does the time go?** How much of an inference call is actual
   inference, and how much is the Ruby ↔ C++ boundary?

## The pipeline

```
bench_models.py     model definitions (shared by both scripts below)
      │
      ▼
make_pt_models.py   → models/<name>.pt         eager module, torch.save
                    → models/<name>.meta.json  shapes, param count, eager latency
                    → models/<name>.io.bin     golden input + output, raw float32
      │
      ▼
pt_to_pte.py        → models/<name>.pte        torch.export → to_edge → to_executorch
      │
      ▼
run_bench.rb        → results/<label>.json     eval + per-phase timings
      │
      ▼
compare.rb          baseline vs current, phase by phase
```

## Running it

```bash
# 1. Build the .pt benchmarks (needs torch; torchvision for resnet/mobilenet)
python3 bench/make_pt_models.py --all

# 2. Convert every checkpoint to .pte
python3 bench/pt_to_pte.py bench/models/*.pt

# 3. Eval + profile
bundle exec ruby bench/run_bench.rb --label mine

# 4. Compare against a stored run
bundle exec ruby bench/compare.rb bench/results/baseline.json bench/results/mine.json
```

To benchmark an XNNPACK-delegated build (see `FINDINGS.md` -- this is worth
~658x on `forward`), export the lowered variant and point the harness at it:

```bash
python3 bench/pt_to_pte.py bench/models/*.pt --xnnpack   # writes <name>.xnnpack.pte
bundle exec ruby bench/run_bench.rb --label xnnpack --variant xnnpack
```

`pt_to_pte.py` works on any `.pt` you have, not just these:

```bash
python3 bench/pt_to_pte.py my_model.pt --input-shape 1,3,224,224 -o my_model.pte
```

(It calls `torch.load`, which unpickles arbitrary code — only point it at
checkpoints you trust.)

## The model set

Chosen to span three regimes, because binding overhead only shows up in one of
them:

| model | input | regime | why it's here |
|---|---|---|---|
| `add_mul` | `[1,3]` | overhead | `x*2+1`, the README example. Almost no math — nearly all of the wall time is the boundary. |
| `tiny_mlp` | `[1,8]` | overhead | Smallest real layer. Same idea. |
| `mlp_512x2` | `[1,512]` | mixed | Enough math to matter, small enough that per-call cost still shows. |
| `mnist_cnn` | `[1,1,28,28]` | mixed | Convolutions, small input. |
| `mlp_1024x4` | `[1,1024]` | compute | 4.2M params — kernels dominate. |
| `resnet18` | `[1,3,224,224]` | compute | Real vision model, 150k-element input. |
| `mobilenet_v2` | `[1,3,224,224]` | compute | Same, depthwise-separable. |

## Reading the output

Each model reports an eval line and one row per phase:

```
tiny_mlp  [overhead]  in=[1, 8] out=[1, 8] params=72
  eval: PASS  max_abs_error=5.96e-08  (pytorch eager p50 0.010 ms)
  phase                  p50 ms     p90 ms    allocs/call
  tensor_new_flat        0.0150     0.0170            1.0
  ...
  binding overhead: 73.9% of e2e is not forward()
```

The phases:

| phase | what it measures |
|---|---|
| `tensor_new_flat` | `Tensor.new(flat_array, shape:)` |
| `tensor_new_nested` | `Tensor.new(nested_array)` — shape inferred, the README's style |
| `forward` | `model.predict([t])` with the tensor already built |
| `to_a_flat` | `Tensor#flat_to_a` |
| `to_a_nested` | `Tensor#to_a` — reshaped to match `shape` |
| `e2e` | build → predict → read, the way an app calls it |
| `tensor_from_bytes` | `Tensor.from_bytes(packed, shape:)` — the memcpy path |
| `to_binary` | `Tensor#to_binary` — the memcpy path out |
| `e2e_binary` | round trip using packed bytes at both ends |

`allocs/call` comes from `GC.stat`. It is often the more useful number: object
churn on this boundary is usually what the latency is made of.

Timings are batched — each sample runs the block enough times to span ~500 µs —
so a per-call cost of a few microseconds isn't swamped by the cost of reading
the clock. Phases stop early once they've spent `--budget-s` (default 3 s), so
a slow model doesn't stall the suite.

## Caveats

- `eager_p50_ms` in the metadata is PyTorch's own latency for the same
  checkpoint, measured on the machine that generated it. It's a sanity anchor,
  not a like-for-like comparison — a different machine, or one busy compiling,
  will skew it. Regenerate the models to refresh it.
- Absolute numbers are machine-specific. The *ratios between phases* are the
  durable finding.
- `forward` speed depends heavily on which kernels the runtime was built with;
  see the "Kernels" section in `bench/FINDINGS.md`.

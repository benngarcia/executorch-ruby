# What the benchmarks found

Measured on the machine that produced `bench/results/*.json` (4-core x86_64,
Ruby 3.3.6, Rice 4.12, ExecuTorch 0.7.0). Absolute numbers are
machine-specific; the ratios are the durable part.

There are two separate performance stories here, and it's worth keeping them
apart:

1. **The bindings** — the cost of getting data across the Ruby ↔ C++ boundary.
   This is what the gem controls, and it was expensive.
2. **The kernels** — the cost of the math itself. The gem doesn't control this,
   but it does control which kernels get linked, and the default is the slow
   one.

Short version: the kernels are worth **658×** and the bindings are worth up to
**20×**, and you need both — on a fast runtime, 96.6% of a mobilenet_v2 call
was binding overhead. Part 3 has the cross-product.

---

## Part 1: the bindings

### The finding

Profiling a call by phase showed that on a small model, most of an "inference"
call wasn't inference:

| model | e2e | of which `forward()` | binding overhead |
|---|---|---|---|
| `tiny_mlp` | 0.082 ms | 0.020 ms | **76%** |
| `mlp_512x2` | 4.22 ms | 3.24 ms | 23% |
| `resnet18` | 9313 ms | 8261 ms | 11% |

Broken down per element, the boundary cost about **0.6 µs per element in** and
**1.1 µs per element out**. For a resnet18 input that's 92 ms to build a tensor
from an Array of 150,528 floats — before any math happens.

For scale: 1.1 µs per element is roughly 3,000 CPU cycles to move one float
from a C array into a Ruby Array.

### The cause

Rice is a lovely API, but its default conversion path is built for safety, not
throughput. Every `Array` element access and every scalar conversion goes
through `Rice::detail::protect()`:

```cpp
// rice.hpp
inline VALUE Array::Proxy::value() const {
  return detail::protect(rb_ary_entry, array_.value(), index_);
}

static T convert(VALUE value) {
  return (T)protect(RubyType<T>::fromRuby, value);   // rb_num2dbl
}
```

and `protect()` is:

```cpp
int state = (int)JumpException::RUBY_TAG_NONE;
rb_protect(trampoline, (VALUE)(&invoker), &state);
```

`rb_protect` pushes a VM tag and does a `setjmp` — it exists so a Ruby
exception can't longjmp past C++ destructors. Entirely correct for arbitrary
Ruby calls. But the original code read tensor data like this:

```cpp
for (size_t i = 0; i < data.size(); i++) {
  float_data.push_back(static_cast<float>(
    detail::From_Ruby<double>().convert(data[i].value())));
}
```

That's **two `rb_protect` calls per element** — one for `data[i].value()`, one
for the numeric conversion — to read a `Float` that's sitting right there in
the array. On the way out, `result.push(data[i])` cost another one each.

The setjmp was the entire cost. The conversion itself is a pointer dereference.

### The fixes

**1. Inline the common conversions** (`ext/executorch/utils.h`). `Float` and
`Fixnum` cover essentially all real tensor data and can't raise, so they take a
direct path; anything else (Bignum, Rational, an object with `to_f`) still goes
through Rice's protected call, because those genuinely can run Ruby code:

```cpp
inline double to_double_fast(VALUE v) {
  if (RB_FLOAT_TYPE_P(v)) return RFLOAT_VALUE(v);
  if (FIXNUM_P(v))        return static_cast<double>(FIX2LONG(v));
  return Rice::detail::protect(rb_num2dbl, v);   // rare, and really can raise
}
```

Building the output array uses `rb_ary_new_capa` + `rb_ary_push` — sized once,
and `rb_ary_push` on an array we just created can't raise, so no `protect` is
needed. Doubles mostly become flonums, so nothing is allocated per element.

**2. Stop deep-copying every input.** `forward()` cloned each input tensor to
"own the data during forward". But the caller's Array holds a live reference to
every input for the whole call, so nothing can be collected underneath us — the
copy bought nothing and cost a full pass over the input. Outputs *are* still
cloned, and must be: they point into the method's planned memory arena, which
the next call overwrites.

**3. Dispatch on type, not on exceptions.** Input type detection was:

```cpp
try   { RubyTensor& t = detail::From_Ruby<RubyTensor&>().convert(...); }
catch (...) { try { /* RubyEValue */ } catch (...) { rb_raise(...); } }
```

Throwing a C++ exception to discover a type costs more than a small model's
entire inference. Replaced with `Data_Type<RubyTensor>::is_descendant(v)`.

**4. Flatten nested input level-by-level in Ruby.** `Tensor.new([[1.0, 2.0]])`
went through a recursive `flat_map` that allocated an intermediate Array *per
leaf* — 153,238 objects for a resnet18 input, before a single number reached
C++. Walking one level at a time with `Array#concat` does the gathering in C.
Jagged-array detection is preserved (the level walk catches size mismatches
directly; a final flatten-size check catches uneven nesting depth).

**5. A raw-binary escape hatch.** `Tensor.from_bytes` / `Tensor#to_binary` skip
per-element conversion entirely and memcpy the buffer. This is the right path
whenever the data is already bytes — an image decoded to a string, a file, a
socket — or when you can pack once and reuse.

### The results

`bundle exec ruby bench/compare.rb bench/results/baseline.json bench/results/optimized.json`

| phase | model | before | after | speedup |
|---|---|---|---|---|
| `tensor_new_flat` | resnet18 | 92.20 ms | 7.00 ms | **13.2×** |
| `tensor_new_nested` | resnet18 | 135.22 ms | 13.27 ms | **10.2×** |
| `tensor_from_bytes` | resnet18 | — | 0.53 ms | **174×** vs baseline |
| `to_a_flat` | mobilenet_v2 | 1.105 ms | 0.017 ms | **64.6×** |
| `to_a_nested` | mlp_1024x4 | 1.174 ms | 0.038 ms | **31.3×** |
| `to_binary` | mobilenet_v2 | — | 0.004 ms | **291×** vs baseline |
| `forward` | add_mul | 0.019 ms | 0.014 ms | 1.33× |

Allocations per call, resnet18 nested input: **153,238 → 11**.

Binding overhead as a share of an end-to-end call:

| model | before | after |
|---|---|---|
| `mnist_cnn` | 10.7% | 0.2% |
| `mlp_512x2` | 23.3% | 1.1% |
| `mlp_1024x4` | 17.3% | 2.5% |

All 59 existing tests pass, and every model's output still matches PyTorch to
within 1e-4 (most to 1e-7).

`tiny_mlp` still shows ~74% overhead, and that's honest: when `forward()` is
13 µs, the fixed per-call cost of crossing the boundary at all — allocating the
result Tensor object, wrapping outputs — is the floor. It's just a much lower
floor than before (0.082 ms → 0.050 ms e2e).

### What's left on the table

- **A zero-copy input view.** `from_bytes` still copies once into a tensor-owned
  buffer. A tensor that borrows a frozen Ruby String's memory would remove even
  that, at the cost of a lifetime rule users have to respect.
- **Reusing output tensors.** Every `forward()` allocates a fresh `Tensor` per
  output and clones the data. An opt-in "write into this tensor I already have"
  API would suit a hot inference loop.
- **`Tensor#to_a` reshaping** still happens in Ruby. It's now a small share of
  the cost, but for a many-dimensional output it could move into C.
- **`shape()`** still uses `Rice::Array::push` (one `rb_protect` per dimension).
  Irrelevant at 4 dimensions, noted for consistency.

---

## Part 2: the kernels

This one dwarfs everything above, and it isn't a binding problem at all.

With the runtime built the way the README described, `forward()` on resnet18
takes **8.26 seconds**. PyTorch eager, same checkpoint, same machine: **25 ms**.
That's ~330× slower, and no amount of binding work touches it.

The reason is that a default ExecuTorch build links `portable_ops_lib` — the
reference kernel implementations. They're written for correctness and
portability, not speed: no vectorization, no blocking, no threading. They're the
right default for a runtime that has to build anywhere, and the wrong choice for
anything latency-sensitive.

Two candidate fixes, and it's worth being precise about what each one bought,
because they were not equal:

### Optimized CPU kernels — no measurable help

`EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` swaps in vectorized implementations and
works on existing `.pte` files with no re-export, which makes it sound like the
easy win. Measured (`results/optimized_kernels.json`):

| model | portable | optimized kernels |
|---|---|---|
| resnet18 `forward` | 8262 ms | 8204 ms |
| mobilenet_v2 `forward` | 1618 ms | 1629 ms |
| mlp_512x2 `forward` | 2.95 ms | 3.26 ms |

Noise. The optimized set covers elementwise and a few BLAS-backed ops; these
models spend all their time in convolution, which still falls back to the
portable implementation. Worth knowing before reaching for it as a fix.

### The XNNPACK delegate — 658×

`EXECUTORCH_BUILD_XNNPACK=ON` plus a re-export through `XnnpackPartitioner`
hands whole subgraphs to XNNPACK. This needs both a rebuild *and* a re-export —
the backend has to exist at build time and be targeted at export time, and
missing either half silently leaves you on portable kernels.

| model | portable `forward` | XNNPACK `forward` | speedup | PyTorch eager |
|---|---|---|---|---|
| resnet18 | 8262 ms | **12.56 ms** | 658× | 25.3 ms |
| mobilenet_v2 | 1618 ms | **3.39 ms** | 477× | 16.7 ms |
| mlp_1024x4 | 48.6 ms | **0.27 ms** | 178× | 0.30 ms |
| mnist_cnn | 4.91 ms | **0.26 ms** | 18.6× | 0.29 ms |
| mlp_512x2 | 3.24 ms | **0.12 ms** | 26.6× | 0.055 ms |

Delegated ExecuTorch is *faster than PyTorch eager* on both vision models —
2.0× on resnet18, 4.9× on mobilenet_v2 — which is the whole point of an
ahead-of-time-compiled edge runtime.

`bench/pt_to_pte.py --xnnpack` produces the lowered variant; `run_bench.rb
--variant xnnpack` benchmarks it.

### A linker trap worth knowing about

Kernel libraries and backend delegates register themselves from global
constructors. A plain `-lxnnpack_backend` only pulls in object files that
resolve some undefined symbol — and a self-registering object resolves nothing,
so the linker discards it and the registration silently never happens. You find
out much later, when a model fails to load with a missing-operator or
missing-backend error.

They have to be whole-archived: `-Wl,--whole-archive` on Linux,
`-Wl,-force_load` on macOS. `extconf.rb` already did this for
`portable_ops_lib`; it's now generalized to the optimized set and the XNNPACK
backend.

One constraint that follows: **exactly one operator library may be
whole-archived.** Each registers the full op set into the same global table, so
linking two aborts the runtime at init on duplicate registration. `extconf.rb`
picks `optimized_native_cpu_ops_lib` when present and `portable_ops_lib`
otherwise; `EXECUTORCH_OPS_LIB` overrides.

---

## Part 3: why the two halves need each other

Neither piece of work looks impressive alone. Together they're the whole story.

On portable kernels, the binding fixes barely move end-to-end time — `forward()`
is so slow that nothing else is visible. That's exactly why the original
bindings could carry a 0.6 µs/element conversion cost without anyone noticing.

So the real test is the cross-product: **original bindings, XNNPACK kernels**
(`results/baseline_xnnpack.json`). Once the math is fast, the boundary is all
that's left:

| model | e2e | `forward()` | overhead |
|---|---|---|---|
| mobilenet_v2 | 107.9 ms | 3.7 ms | **96.6%** |
| resnet18 | 116.4 ms | 11.7 ms | **90.0%** |
| mlp_1024x4 | 2.61 ms | 0.30 ms | 88.4% |
| mnist_cnn | 1.21 ms | 0.25 ms | 79.6% |

96.6% of a mobilenet_v2 call spent not doing inference. Building the input
tensor alone (102.9 ms) cost **28× more than running the model** (3.7 ms).

With the optimized bindings on the same XNNPACK runtime:

| model | e2e before | e2e after | with `from_bytes` | best speedup |
|---|---|---|---|---|
| mobilenet_v2 | 107.9 ms | 12.97 ms | **5.26 ms** | **20.5×** |
| resnet18 | 116.4 ms | 19.67 ms | **13.69 ms** | **8.5×** |
| mlp_1024x4 | 2.61 ms | 0.47 ms | **0.38 ms** | **6.9×** |
| mnist_cnn | 1.21 ms | 0.47 ms | **0.29 ms** | **4.2×** |

The takeaway: **fix the kernels first, because that's the 658×** — but the
moment you do, the bindings become the bottleneck, and on mobilenet_v2 they're
worth another 20×.

It also changes which API matters. On a fast runtime, `Tensor.from_bytes` isn't
a micro-optimization: for mobilenet_v2 it's the difference between spending
7.6 ms or 0.85 ms getting the input across, against a 3.4 ms inference.

---

## Notes for reproducing

Four result files, the full cross-product:

| file | bindings | kernels |
|---|---|---|
| `baseline.json` | original | portable |
| `optimized.json` | optimized | portable |
| `optimized_kernels.json` | optimized | optimized CPU kernels |
| `baseline_xnnpack.json` | original | XNNPACK |
| `xnnpack.json` | optimized | XNNPACK |

```bash
bundle exec ruby bench/compare.rb bench/results/baseline_xnnpack.json bench/results/xnnpack.json
```

The eager PyTorch timings recorded in each `.meta.json` were measured on the
same machine but not under controlled conditions (some runs overlapped a
compile). Treat them as an anchor, not a benchmark; regenerate the models on an
idle machine if you want to lean on them.

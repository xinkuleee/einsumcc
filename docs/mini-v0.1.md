# Mini v0.1 milestone

Mini v0.1 closes the schedule-to-machine-code loop for the native CPU Direct
backend. It builds on the Nano v1 frontend, planner, semantic backends, custom
`tc` dialect, and AOT runtime.

## Implemented

- `DirectSchedule.block_m/n/k` deterministically changes the generated MLIR
  loop structure for arbitrary supported Einstein contractions.
- Collapsed M/N/K tile budgets are projected onto original Einstein dimensions
  in `output + reduction` loop order. Within a group, inner dimensions consume
  the budget first. Batch dimensions receive a fixed unit tile and are not part
  of the M/N/K search space.
- The project-owned `einsumcc-schedule-direct` pass tiles only the marked
  contraction `linalg.generic`; output zero initialization remains outside the
  tiled reduction loops.
- Partial tiles lower through `affine.min` and `memref.subview` to LLVM. The
  pinned LLVM 23 `clang` consumes the emitted LLVM IR.
- Raw schedules, expanded tile sizes, the complete pipeline, workload, and
  toolchain/host identity participate in artifact cache identity and manifests.
- `NativeInvocation` validates and binds the ranked-memref ABI once, then
  supports repeated calls without repeating shape/stride/alias checks.
- `NativeDirectTuner` deduplicates equivalent expanded schedules, compiles each
  remaining dylib, checks it against NumPy, measures bound calls, selects by
  median, and writes a reusable hardware/toolchain-scoped tuning record.
- `emit-mlir --stage scheduled`, `run-native`, and `tune-native` expose the
  pipeline, explicit schedules, cache reuse, and machine-readable reports.

## Tile projection

For one logical group with extents $(d_0,ldots,d_r)$ and collapsed budget $T$,
projection walks from $d_r$ toward $d_0$. At each dimension it chooses

$$
t_i = min(d_i, R),qquad R leftarrow max(1,lfloor R/t_ifloor),
$$

starting with $R=T$. The product never exceeds $T$. For M extents $(2,3,4)$
and `block_m=8`, the result is $(1,2,4)`. This is a deterministic structured-loop
projection; it does not claim that MLIR first reassociates the dimensions into
one flat loop.

The complete tile vector is aligned with the `linalg.generic` iterator order.
That shared definition is used by code generation, artifact identity, tuning
deduplication, manifests, CLI diagnostics, and tests.

## Native pipeline

```text
Einstein equation + static shape/stride
  -> tc.contract
  -> initialized linalg.generic
  -> one-shot bufferization
  -> result-to-out-parameter conversion
  -> einsumcc-schedule-direct
  -> linalg loops
  -> affine and subview metadata lowering
  -> SCF/CF/LLVM dialect
  -> LLVM IR
  -> host dylib
  -> ranked-memref C wrapper
```

Reduction tiling is correct because `linalg.fill` initializes the output once,
outside all schedule loops. Each K tile then reads and updates the same output
subview, accumulating across tiles. Boundary tiles use dynamic subview sizes.

## Native tuning methodology

Static scoring orders candidates but does not decide the winner. The native
tuner performs the following steps:

1. Generate legal CPU candidates (`threads=1`, `vector_width=1`).
2. Deduplicate candidates with identical expanded tile vectors.
3. Compile or load each schedule-specific artifact.
4. Bind input/output descriptors once and execute once for correctness.
5. Compare with `numpy.einsum` under recorded tolerances.
6. Run warmups and collect repeated bound `ctypes` host-call latency samples.
7. Select the lowest median and persist its schedule under a host/toolchain
   target identity.

The report's baseline is the first statically ranked distinct schedule, not an
untiled kernel or an external library. A measured speedup of 1.0 is valid. Mini
v0.1 makes no claim that tiling always improves latency.

## Commands

Inspect schedule-driven IR:

```bash
PYTHONPATH=src python3 -m einsumcc emit-mlir \
  'mk,kn->mn' --lhs-shape 32,24 --rhs-shape 24,32 \
  --stage scheduled --block-m 8 --block-n 16 --block-k 4
```

Tune real dylibs and save a JSON report plus reusable schedule cache:

```bash
PYTHONPATH=src python3 -m einsumcc tune-native \
  'mk,kn->mn' --lhs-shape 32,24 --rhs-shape 24,32 \
  --warmups 2 --repeats 20 --max-schedules 8 \
  --cache .einsumcc-cache/tuning-native-mini-v0.1.json \
  -o native-tuning.json
```

Reuse the selected schedule:

```bash
PYTHONPATH=src python3 -m einsumcc run-native \
  'mk,kn->mn' --lhs-shape 32,24 --rhs-shape 24,32 \
  --tuning-cache .einsumcc-cache/tuning-native-mini-v0.1.json
```

## Verification matrix

The native differential suite covers ordinary matmul, batched contraction,
permuted output, scalar output, multiple reduction dimensions, the course's
high-rank contraction, and positive-stride inputs. Separate tests cover:

- schedule-to-SCF step changes and different artifact keys;
- raw/expanded schedule manifests;
- rejection of unimplemented vector width and thread count;
- malformed pass options and tile-rank mismatch;
- reusable bound invocations;
- artifact corruption, dangling symlinks, and concurrent population;
- native candidate deduplication, correctness, samples, and cache records.

Run the complete local gate with:

```bash
make test
make test-mlir
make test-cpu-codegen
git diff --check
```

## Honest boundary

Mini v0.1 is a real small compiler, not a production CPU or GPU kernel system.
It does not implement native GEMM-view/Packed-GEMM calls, vectorization, loop
interchange, multithreading, dynamic shapes, CUDA, GPU/NVVM, Tensor Cores, or
mixed precision. `threads` and `vector_width` are rejected unless both are one;
they are not silently ignored.

The next useful milestones are stable Vector-to-LLVM lowering, loop
interchange, a versioned native benchmark corpus with BLAS baselines, native
GEMM plan lowering, and then GPU/NVVM code generation on NVIDIA hardware.

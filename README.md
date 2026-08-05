# EinsumCC

EinsumCC is a small tensor-contraction compiler. It accepts an explicit
two-input Einstein summation, analyses its indices and layouts, and chooses
between a direct contraction and GEMM-based execution plans.

The project is intentionally CPU-first: parsing, legality, planning, tuning,
and numerical semantics are testable on a development laptop. The same IR and
plan interfaces are designed to feed an MLIR/CUDA backend on an A100 machine.

## Current milestone: Nano v1 CPU compiler

- explicit `einsum` syntax such as `bij,bjk->bik`;
- binary contractions with batch, free, and multiple reduction indices;
- static shapes, positive input strides, and contiguous output;
- direct, zero-copy GEMM, and packed-GEMM execution plans;
- a target-aware analytical cost model and explainable plan selection;
- schedule-space generation, empirical tuning hooks, and a JSON tuning cache;
- CPU reference implementations for differential correctness tests;
- a TableGen-defined `tc` dialect and verified `tc.contract` operation;
- `tc.contract` to `linalg.generic` conversion in the project-owned
  `einsumcc-opt` driver;
- a tested macOS path from Einstein notation through loops and LLVM IR to a
  native arm64 executable.

FP32 is the executable Nano v1 datatype. FP16/BF16, Tensor Cores, arbitrary
GPU layouts, and fused epilogues are later milestones.

## Quick start

No installation is required for development. NumPy is the only runtime
dependency.

```bash
make test
make test-mlir
make test-cpu-codegen
make demo
make check
make benchmark
```

Inspect the course contraction with small dimensions:

```bash
PYTHONPATH=src python3 -m einsumcc explain \
  'aijd,bckd->abcijk' \
  --lhs-shape 2,3,4,5 \
  --rhs-shape 6,7,8,5
```

Run every legal CPU plan and compare it with NumPy:

```bash
PYTHONPATH=src python3 -m einsumcc verify \
  'bij,bjk->bik' --lhs-shape 2,3,4 --rhs-shape 2,4,5
```

Emit the semantic `tc.contract` module (use `--stage linalg` to inspect its
frontend-only expansion):

```bash
PYTHONPATH=src python3 -m einsumcc emit-mlir \
  'mk,kn->mn' --lhs-shape 32,16 --rhs-shape 16,64
```

`make test-cpu-codegen` performs the real compiler smoke test: Python Einstein
frontend → `tc.contract` → `linalg` → bufferization → loops → LLVM dialect →
LLVM IR → native arm64 binary, then checks its numeric matrix result.

## Documentation

- [Architecture](docs/architecture.md) explains the compiler pipeline and
  component boundaries.
- [Nano v1 scope](docs/nano-v1.md) records supported semantics and completion
  gates.
- [CPU-first decision](docs/adr/0001-cpu-first.md) explains why CPU execution
  is a semantic backend rather than throwaway scaffolding.
- [Planning decision](docs/adr/0002-plan-model.md) explains the three execution
  plans.
- [Development guide](docs/development.md) covers the Mac workflow and A100
  continuation checklist.
- [Benchmarks](docs/benchmarks.md) defines reproducible workload and result
  formats.

## Repository status

This repository is developed directly on `main`. Every milestone should keep
the test suite green and document changes to public IR or plan semantics.

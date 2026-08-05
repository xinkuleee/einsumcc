# Development guide

## Local macOS workflow

The compiler core needs Python 3.9+ and NumPy. No CUDA installation is expected
on macOS.

```bash
make test
make demo
make check
```

The test suite uses `unittest`, so no additional test framework is required.
One optional test invokes `mlir-opt`; it is skipped when MLIR is unavailable.

macOS system Python may place bytecode under `~/Library/Caches`. In restricted
environments, either disable bytecode or redirect its cache:

```bash
PYTHONDONTWRITEBYTECODE=1 make test
PYTHONPYCACHEPREFIX=/tmp/einsumcc-pycache make test
```

## Before committing

Run:

```bash
make check
git diff --check
git status --short
```

When changing parser or IR semantics, update `docs/nano-v1.md`, architecture,
and tests in the same commit. When changing a plan decision, add a test for both
legality and explanation text.

## CPU benchmark discipline

The `tune` command is useful for testing the tuner and cache. Laptop results are
not transferable to A100. Each cache entry contains the target name, and future
device targets must use a distinct target identity.

Use small inputs for Direct CPU tests: the implementation is intentionally
literal rather than a high-performance CPU kernel. Its purpose is transparent
semantics and differential correctness.

## A100 continuation checklist

1. Pin an LLVM/MLIR revision in build documentation.
2. Parse and verify emitted MLIR with that toolchain.
3. Introduce an explicit GPU runtime interface; do not pass Python arrays into
   target-independent IR.
4. Implement Direct lowering first for contiguous FP32 inputs.
5. Add a cuBLAS wrapper for GEMM-view and Packed-GEMM.
6. Differential-test every generated kernel against the CPU backend.
7. Calibrate target model constants from reproducible A100 measurements.
8. Store GPU tuning records under a hardware/toolchain-specific target name.
9. Add cuTENSOR and course starter kernels as external baselines.
10. Report latency, GFLOP/s, workspace, selected plan, and selector regret.


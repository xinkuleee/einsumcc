# Development guide

## Local macOS workflow

The compiler core needs Python 3.9+ and NumPy. No CUDA installation is expected
on macOS.

```bash
make test
make test-mlir
make test-cpu-codegen
make demo
make check
```

The Python suite uses `unittest`. The MLIR suite uses the pinned `FileCheck`
and the project's own optimizer. The native smoke test also needs the Apple
Command Line Tools C compiler.
Install the pinned Apple Silicon toolchain used by this project with:

```bash
./scripts/bootstrap-mlir.sh
export PATH="$PWD/.deps/llvm-macos-arm64/bin:$PATH"
```

The 359 MiB archive and extracted toolchain live under `.deps/` and are not
tracked by Git. The bootstrap script verifies the published SHA-256 digest.
This Triton archive is a development toolchain: it supplies MLIR headers,
libraries, CMake configuration, `mlir-tblgen`, and `mlir-translate`. EinsumCC
builds its own optimizer driver instead of assuming a bundled `mlir-opt`.
If `cmake` and `ninja` are not installed globally, they may be installed into
`.deps/build-tools`; `scripts/build-mlir.sh` detects that isolated location.
For example: `python3 -m pip install --target .deps/build-tools cmake ninja`.

The complete local compiler checks are:

```bash
./scripts/build-mlir.sh
./scripts/test-mlir.sh
./scripts/test-cpu-codegen.sh
```

The CPU codegen test starts from the Einstein frontend and crosses the custom
dialect boundary before compiling LLVM IR to a native arm64 dynamic library.
The project runtime passes ranked-memref descriptors and checks matrix,
batched, permuted, scalar-output, multi-reduction, course, and positive-stride
input contractions against NumPy. Input descriptors carry runtime strides; the
Nano output ABI remains C-contiguous. This harness is an end-to-end compiler
test. `NativeCpuCompiler` also exposes this narrow static AOT path through the
Python API and `run-native`; dynamic shapes, arbitrary device buffers, and a
long-lived JIT execution engine remain deferred to Mini.

Native artifacts default to `.einsumcc-cache/native-v1`. The `run-native` CLI
accepts `--native-cache`, and embedding callers can pass `cache_dir` to
`NativeCpuCompiler`.
Tool discovery can be overridden with `EINSUMCC_ROOT`, `EINSUMCC_OPT`,
`EINSUMCC_MLIR_TRANSLATE`, and `EINSUMCC_CLANG`.

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
make test-mlir
make test-cpu-codegen
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

1. Introduce an explicit GPU runtime interface; do not pass Python arrays into
   target-independent IR.
2. Implement Direct GPU lowering first for contiguous FP32 inputs.
3. Add a cuBLAS wrapper for GEMM-view and Packed-GEMM.
4. Differential-test every generated kernel against the CPU backend.
5. Calibrate target model constants from reproducible A100 measurements.
6. Store GPU tuning records under a hardware/toolchain-specific target name.
7. Add cuTENSOR and course starter kernels as external baselines.
8. Report latency, GFLOP/s, workspace, selected plan, and selector regret.

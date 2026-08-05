# Architecture

EinsumCC is split into target-independent compiler logic, executable semantic
backends, and target models. Nano v1 keeps these boundaries even though all
locally executable code runs on the CPU.

## Pipeline

```text
explicit einsum + static shapes/strides
                 |
                 v
       Equation parser/verifier
                 |
                 v
      ContractionProblem semantic IR
                 |
       +---------+----------+
       |                    |
       v                    v
 layout legality       tc.contract emission
       |                    |
       |                    v
       |               tc verifier/pass
       |                    |
       |                    v
       |               linalg.generic
       v
 cost all legal plans
       |
       v
 Direct / GEMM-view / Packed-GEMM
       |
       +------ Direct schedule search
       |
       v
 analytical choice or target-specific tuning record
       |
       v
 executable backend
```

## Semantic IR

`ContractionProblem` is the stable boundary between the frontend and all
backends. It owns no arrays or device buffers. It records:

- a verified `Equation`;
- input and output `TensorSpec` values;
- the B/M/N/K `IndexGroups`;
- label extents and collapsed B/M/N/K sizes;
- a target-independent workload key.

The workload key includes equation, shape, strides, dtype, and output metadata.
The target is deliberately not part of this hash; the cache combines the hash
with a target name so one problem can have different CPU and A100 records.

## Execution plans

`Planner.enumerate()` always constructs the same three plan kinds. Legality and
cost are separate: an illegal plan remains visible in diagnostics.

### Direct

Direct preserves the original tensor layouts and reconstructs operand indices
inside the contraction. On the CPU, the implementation is a literal tiled
B/M/N/K loop and never calls `einsum` or `matmul`. Future GPU lowering will map
the same schedule parameters to generated loops, blocks, and shared memory.

### GEMM-view

GEMM-view is legal only if each contiguous operand can be interpreted as a
batched matrix using at most whole-matrix transpose flags and the output already
has canonical B+M+N order. The CPU backend asserts that NumPy reshaping shares
storage before invoking `matmul`; this guards against accidentally calling a
copy a zero-copy view.

### Packed-GEMM

Packed-GEMM transposes and copies both inputs into canonical B/M/K and B/K/N
orders, invokes batched matrix multiplication, and materializes the requested
output order. It is always legal for a verified Nano v1 problem and explicitly
reports workspace and layout traffic.

## Planning versus tuning

The analytical model estimates compute, memory, launch, indexing, and materialized
layout costs. It is intentionally inspectable and must not be reported as device
performance. Its job is to provide a deterministic default and prune candidates.

`EmpiricalTuner` measures legal plans on an executable backend. Direct gets a
small statically pruned schedule set; GEMM plans have no Nano v1 schedule knobs.
Measurements are cached under `(target, workload_key)`. A CPU record therefore
cannot override an A100 decision.

## MLIR boundary

`TcEmitter` generates the project-owned `tc.contract` operation. Its C++
verifier independently checks ranked static FP32 tensors, map ranks, exact
B/M/N/K memberships, iterator classes, loop extents, and consistency between
the optional Einstein equation and maps. Nano v1 rejects encoded tensor types
at this boundary because its dense lowering cannot preserve an encoding. The
`tc-contract-to-linalg` pass materializes zero initialization and a
`linalg.generic` multiply-add body.

The tested macOS Direct path is:

```text
Einstein frontend -> tc.contract -> linalg.generic -> bufferization
  -> scf/cf loops -> LLVM dialect -> LLVM IR -> native arm64
```

`NativeCpuCompiler` drives that path and stores content-addressed AOT artifacts
under `.einsumcc-cache/native-v1`. The cache identity covers the workload
(including layout), semantic IR, pipeline revision, executable metadata, and
host identity; the manifest also checksums the shared library. A sidecar file
lock serializes concurrent population. `NativeKernel` calls the generated C
wrapper with validated ranked-memref descriptors. Python owns the input and
output arrays, so generated code never transfers heap ownership across the ABI;
output/input overlap is rejected because the kernel zero-initializes output.
This native pipeline implements Direct only. The three-plan selector and
`DirectSchedule` are fully exercised by the Python semantic backend, but the
selected schedule does not yet rewrite the native `linalg`/loop pipeline.

The intended A100 pipeline shares the frontend and verified dialect:

```text
tc/einsum frontend
  -> linalg.generic
  -> plan selection
     -> Direct: linalg/scf/vector -> gpu -> nvvm
     -> GEMM-view: collapse/reinterpret -> cuBLAS runtime call
     -> Packed-GEMM: generated pack kernels -> cuBLAS -> unpack
```

Nano v1 owns its optimizer driver instead of depending on a separately bundled
`mlir-opt`. FileCheck covers round-trip, rejected IR, and `tc`-to-`linalg`; the
Python native runtime loads generated arm64 code through `ctypes` and compares
it with NumPy.

## Modules

| Module | Responsibility |
|---|---|
| `equation.py` | parse and verify explicit binary einsum |
| `problem.py` | immutable semantic IR and workload identity |
| `layout.py` | zero-copy GEMM-view legality proof |
| `plans.py` | plan construction, cost model, diagnostics |
| `schedule.py` | Direct schedule generation and target pruning |
| `compiler.py` | public compile/explain/execute façade |
| `cpu_backend.py` | independent plan semantics |
| `native_backend.py` | native Direct lowering, AOT cache, and memref ABI |
| `tuner.py`, `cache.py` | measurement and persistent target-specific choices |
| `mlir_emitter.py` | portable semantic MLIR |
| `tc_emitter.py` | frontend emission of verified `tc.contract` IR |
| `include/`, `lib/` | `tc` dialect, verifier, and lowering pass |
| `tools/einsumcc-opt` | project optimizer and upstream pass driver |
| `target.py` | transparent target models and hardware constraints |
| `cli.py` | explain, verify, tune, emit-mlir, and run-native workflows |

## Extension rules

1. Backend-specific objects must not enter `ContractionProblem`.
2. Every new plan must expose legality reasons and workspace cost.
3. Analytical estimates and measured timings remain distinguishable.
4. A cache entry is valid only for its exact target and workload key.
5. New GPU lowering is differential-tested against CPU semantics before it is
   benchmarked.
6. Performance claims require the real target; CPU timings cannot stand in for
   A100 measurements.

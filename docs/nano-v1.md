# Nano v1 scope

Nano v1 is a complete but deliberately narrow compiler loop. It is not a
production replacement for cuTENSOR.

## Supported input

- exactly two input tensors and one explicit output;
- single-letter Einstein indices without ellipsis or diagonals;
- rank 1-6 operands;
- batch, left-free, right-free, and one or more reduction indices;
- static positive dimensions, explicit positive input strides, and contiguous output;
- FP32 execution.

An index must belong to exactly one supported class:

- **B**: present in both inputs and the output;
- **M**: present in the left input and output;
- **N**: present in the right input and output;
- **K**: present in both inputs and absent from the output.

This excludes broadcasting, diagonal extraction, unary reductions, ellipsis,
and contractions with more than two operands. Each exclusion is diagnosed by
the verifier rather than silently miscompiled.

## Required compiler loop

1. Parse and verify an equation.
2. Bind static shapes and strides into `ContractionProblem`.
3. Classify indices and analyse GEMM-view legality.
4. Cost all legal execution plans and explain the decision.
5. Generate/prune Direct schedules and expose an empirical tuning hook.
6. Cache a measured plan/schedule using a stable workload key.
7. Execute CPU semantics and compare all legal plans against a reference.
8. Emit a verified `tc.contract`, lower it to initialized `linalg.generic`,
   then lower static CPU cases (including positive-stride inputs) through the
   native Direct path.

The Python semantic backend executes Direct, GEMM-view, and Packed-GEMM. The
Nano v1 MLIR backend implements Direct only; its current generic loop lowering
does not consume the planner's `DirectSchedule`.

## Completion gates

- parser and verifier tests cover valid and invalid classes;
- every supported index pattern has differential CPU correctness tests;
- plan-selection tests cover Direct, zero-copy GEMM, and packed GEMM;
- tuning cache round-trips without losing workload identity;
- the project-owned `einsumcc-opt` parses/verifies `tc.contract` and has
  positive, negative, and lowering tests;
- frontend-generated matrix, batch, permuted, multi-reduction, scalar, and
  course contractions reach LLVM IR and pass native arm64 differential tests;
- the course contraction appears in both tests and the CLI documentation;
- architecture and extension points are documented.

## Deferred to Mini

- a long-lived JIT execution engine, dynamically shaped tensors, and arbitrary
  device-buffer ownership (Nano has a small static AOT/memref runtime);
- FP16/BF16 and mixed-precision accumulation;
- Tensor Core-specific Direct lowering;
- native GEMM-view/Packed-GEMM lowering and schedule-driven native loops;
- contraction/epilogue fusion;
- learned cost models and multi-input contraction-order search.

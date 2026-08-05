# ADR 0001: CPU-first semantic backend

## Status

Accepted.

## Context

Development starts on macOS without an NVIDIA GPU or CUDA toolkit. Parsing,
index classification, layout legality, plan construction, and numerical
semantics are independent of CUDA. Deferring all tests until an A100 is
available would couple compiler correctness to device debugging.

## Decision

EinsumCC has a CPU semantic backend with independent implementations of Direct
and GEMM-based plans. NumPy `einsum` is used only as a differential oracle. GPU
backends consume the same immutable `ContractionProblem`, `ExecutionPlan`, and
`Schedule` values.

MLIR text emission is testable on macOS. If `mlir-opt` is installed, tests may
add parser verification; its absence does not disable compiler-domain tests.

## Consequences

- most correctness bugs are reproducible without a GPU;
- GPU benchmarks remain necessary for performance claims;
- CPU timings must never be presented as evidence for A100 plan quality;
- backend-specific details cannot leak into the semantic IR.


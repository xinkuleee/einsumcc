# ADR 0002: Direct and two GEMM plan classes

## Status

Accepted.

## Context

Calling every GEMM-based contraction "TTGT" hides an important distinction.
Some layouts can be collapsed into matrices and consumed through transpose
flags without copying data. Other layouts require materialized pack/unpack
kernels and workspace. Their costs and optimization opportunities differ.

## Decision

The planner exposes three plan kinds:

1. `direct`: generate a contraction kernel over the original layouts;
2. `gemm-view`: collapse compatible contiguous groups and invoke GEMM without
   materialized layout conversion;
3. `packed-gemm`: materialize canonical B/M/K and B/K/N layouts, invoke GEMM,
   and reorder the output when necessary.

`gemm-view` is legal only when batch and matrix groups form supported contiguous
groups. `packed-gemm` is the general GEMM fallback for Nano v1.

## Consequences

- the cost model accounts for workspace and data movement explicitly;
- explanations can say which operand or output forced packing;
- future GPU code generation can optimize pack kernels independently of GEMM;
- Direct versus GEMM is an execution-plan decision, not a claim about distinct
  hardware backends.


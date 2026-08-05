"""Independent CPU semantics for every Nano v1 execution plan."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import BackendError
from .layout import MatrixView, analyze_gemm_view
from .plans import ExecutionPlan, PlanKind
from .problem import ContractionProblem
from .schedule import DirectSchedule, ScheduleSpace
from .target import CPU_MODEL


def _actual_element_strides(array: np.ndarray) -> Tuple[int, ...]:
    return tuple(stride // array.dtype.itemsize for stride in array.strides)


def _validate_operand(
    name: str, array: np.ndarray, shape: Sequence[int], strides: Sequence[int]
) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.float32:
        raise BackendError("{} operand must have dtype float32".format(name))
    if value.shape != tuple(shape):
        raise BackendError(
            "{} operand shape {} does not match compiled shape {}".format(
                name, value.shape, tuple(shape)
            )
        )
    if _actual_element_strides(value) != tuple(strides):
        raise BackendError(
            "{} operand element strides {} do not match compiled strides {}".format(
                name, _actual_element_strides(value), tuple(strides)
            )
        )
    return value


def _unravel(flat: int, labels: Sequence[str], extents: Mapping[str, int]) -> Tuple[int, ...]:
    if not labels:
        return ()
    shape = tuple(extents[label] for label in labels)
    result = [0] * len(shape)
    for index in range(len(shape) - 1, -1, -1):
        flat, result[index] = divmod(flat, shape[index])
    return tuple(result)


def _bind(indices: Dict[str, int], labels: Sequence[str], values: Sequence[int]) -> None:
    for label, value in zip(labels, values):
        indices[label] = value


def execute_direct(
    problem: ContractionProblem,
    lhs: np.ndarray,
    rhs: np.ndarray,
    schedule: Optional[DirectSchedule] = None,
) -> np.ndarray:
    """Execute a tiled Direct contraction without calling einsum or matmul.

    This intentionally literal implementation is the semantic oracle for future
    generated GPU kernels. B/M/N/K collapsing affects iteration order only; all
    accesses are reconstructed from original labels and therefore cover arbitrary
    positive input strides.
    """

    schedule = schedule or ScheduleSpace().default(problem, CPU_MODEL)
    output = np.empty(problem.output.shape, dtype=np.float32)
    groups = problem.groups

    for batch_flat in range(problem.batch_size):
        batch_values = _unravel(batch_flat, groups.batch, problem.extents)
        for m_base in range(0, problem.m, schedule.block_m):
            for n_base in range(0, problem.n, schedule.block_n):
                for m_flat in range(m_base, min(problem.m, m_base + schedule.block_m)):
                    m_values = _unravel(m_flat, groups.left_free, problem.extents)
                    for n_flat in range(n_base, min(problem.n, n_base + schedule.block_n)):
                        n_values = _unravel(n_flat, groups.right_free, problem.extents)
                        index: Dict[str, int] = {}
                        _bind(index, groups.batch, batch_values)
                        _bind(index, groups.left_free, m_values)
                        _bind(index, groups.right_free, n_values)
                        accumulator = np.float32(0.0)
                        for k_base in range(0, problem.k, schedule.block_k):
                            for k_flat in range(k_base, min(problem.k, k_base + schedule.block_k)):
                                k_values = _unravel(k_flat, groups.reduction, problem.extents)
                                _bind(index, groups.reduction, k_values)
                                lhs_index = tuple(index[label] for label in problem.equation.lhs)
                                rhs_index = tuple(index[label] for label in problem.equation.rhs)
                                accumulator = np.float32(
                                    accumulator + np.float32(lhs[lhs_index] * rhs[rhs_index])
                                )
                        output_index = tuple(index[label] for label in problem.equation.output)
                        output[output_index] = accumulator
    return output


def _canonical_operand(
    array: np.ndarray, physical: Sequence[str], logical: Sequence[str], copy: bool
) -> np.ndarray:
    axes = tuple(physical.index(label) for label in logical)
    value = np.transpose(array, axes) if axes != tuple(range(array.ndim)) else array
    # `np.ascontiguousarray` may return the input unchanged. A packed plan must
    # really materialize its workspace so CPU measurements reflect the plan.
    return np.array(value, dtype=np.float32, order="C", copy=True) if copy else value


def _gemm_shapes(problem: ContractionProblem) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    groups = problem.groups
    batch_shape = tuple(problem.extents[label] for label in groups.batch)
    m_shape = tuple(problem.extents[label] for label in groups.left_free)
    n_shape = tuple(problem.extents[label] for label in groups.right_free)
    return batch_shape, m_shape, n_shape


def _zero_copy_matrix(
    array: np.ndarray,
    view: MatrixView,
    batch_size: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Collapse physical contiguous groups and apply a matrix transpose view.

    Collapsing must happen before transposing. For example, physical labels
    `k0,k1,m0,m1` form one contiguous K group followed by one contiguous M
    group. Transposing every source axis and then reshaping can materialize;
    reshaping to `[B,K,M]` first and swapping the matrix axes is always a view.
    """

    if view.transpose:
        matrix = array.reshape(batch_size, columns, rows).swapaxes(-1, -2)
    else:
        matrix = array.reshape(batch_size, rows, columns)
    if not np.shares_memory(matrix, array):
        raise BackendError("group collapse unexpectedly requires materialization")
    return matrix


def _gemm_result_to_output(problem: ContractionProblem, matrix: np.ndarray, copy: bool) -> np.ndarray:
    batch_shape, m_shape, n_shape = _gemm_shapes(problem)
    canonical_labels = problem.groups.canonical_output
    canonical_shape = batch_shape + m_shape + n_shape
    canonical = matrix.reshape(canonical_shape)
    axes = tuple(canonical_labels.index(label) for label in problem.equation.output)
    result = np.transpose(canonical, axes) if axes != tuple(range(len(axes))) else canonical
    result = result.reshape(problem.output.shape)
    return np.array(result, dtype=np.float32, order="C", copy=True) if copy else result


def _batched_gemm(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Run a batch of 2-D BLAS GEMMs with deterministic FP32 outputs.

    NumPy 2.0 linked against macOS Accelerate may emit spurious floating-point
    warnings from `matmul` for finite strided operands. `dot` reaches the same
    BLAS operation without that platform bug, so batching is explicit here.
    """

    output = np.empty((lhs.shape[0], lhs.shape[1], rhs.shape[2]), dtype=np.float32)
    for batch in range(lhs.shape[0]):
        output[batch] = np.dot(lhs[batch], rhs[batch])
    return output


def execute_gemm_view(
    problem: ContractionProblem, lhs: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    analysis = analyze_gemm_view(problem)
    if not analysis.legal:
        raise BackendError(
            "zero-copy GEMM is illegal: {}".format("; ".join(analysis.reasons))
        )
    lhs_matrix = _zero_copy_matrix(
        lhs, analysis.lhs, problem.batch_size, problem.m, problem.k
    )
    rhs_matrix = _zero_copy_matrix(
        rhs, analysis.rhs, problem.batch_size, problem.k, problem.n
    )
    matrix = _batched_gemm(lhs_matrix, rhs_matrix)
    return _gemm_result_to_output(problem, matrix, copy=False)


def execute_packed_gemm(
    problem: ContractionProblem, lhs: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    groups = problem.groups
    lhs_logical = groups.batch + groups.left_free + groups.reduction
    rhs_logical = groups.batch + groups.reduction + groups.right_free
    lhs_pack = _canonical_operand(lhs, problem.equation.lhs, lhs_logical, copy=True)
    rhs_pack = _canonical_operand(rhs, problem.equation.rhs, rhs_logical, copy=True)
    lhs_matrix = lhs_pack.reshape(problem.batch_size, problem.m, problem.k)
    rhs_matrix = rhs_pack.reshape(problem.batch_size, problem.k, problem.n)
    matrix = _batched_gemm(lhs_matrix, rhs_matrix)
    needs_unpack = problem.equation.output != problem.groups.canonical_output
    return _gemm_result_to_output(problem, matrix, copy=needs_unpack)


def execute(
    problem: ContractionProblem,
    lhs: np.ndarray,
    rhs: np.ndarray,
    plan: ExecutionPlan,
    schedule: Optional[DirectSchedule] = None,
) -> np.ndarray:
    lhs = _validate_operand("left", lhs, problem.lhs.shape, problem.lhs.strides)
    rhs = _validate_operand("right", rhs, problem.rhs.shape, problem.rhs.strides)
    if not plan.legal:
        raise BackendError("cannot execute illegal plan '{}'".format(plan.kind.value))
    if plan.kind == PlanKind.DIRECT:
        return execute_direct(problem, lhs, rhs, schedule)
    if plan.kind == PlanKind.GEMM_VIEW:
        return execute_gemm_view(problem, lhs, rhs)
    if plan.kind == PlanKind.PACKED_GEMM:
        return execute_packed_gemm(problem, lhs, rhs)
    raise BackendError("unsupported CPU plan '{}'".format(plan.kind.value))

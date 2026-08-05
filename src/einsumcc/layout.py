"""Layout legality analysis for zero-copy GEMM views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .problem import ContractionProblem, TensorSpec


@dataclass(frozen=True)
class MatrixView:
    """How an operand is interpreted as a batched matrix."""

    transpose: bool
    physical_labels: Tuple[str, ...]
    logical_labels: Tuple[str, ...]


@dataclass(frozen=True)
class GemmViewAnalysis:
    legal: bool
    reasons: Tuple[str, ...]
    lhs: MatrixView
    rhs: MatrixView


def _matches(labels: Sequence[str], groups: Sequence[Sequence[str]]) -> bool:
    expected = tuple(label for group in groups for label in group)
    return tuple(labels) == expected


def _operand_view(
    name: str,
    labels: Sequence[str],
    spec: TensorSpec,
    batch: Sequence[str],
    rows: Sequence[str],
    columns: Sequence[str],
) -> Tuple[MatrixView, Tuple[str, ...]]:
    logical = tuple(batch) + tuple(rows) + tuple(columns)
    normal = _matches(labels, (batch, rows, columns))
    transposed = _matches(labels, (batch, columns, rows))
    reasons = []
    if not spec.is_c_contiguous:
        reasons.append("{} operand is not C-contiguous".format(name))
    if not (normal or transposed):
        reasons.append(
            "{} indices {} do not form contiguous B/row/column groups".format(
                name, "".join(labels)
            )
        )
    return (
        MatrixView(
            transpose=bool(transposed and not normal),
            physical_labels=tuple(labels),
            logical_labels=logical,
        ),
        tuple(reasons),
    )


def analyze_gemm_view(problem: ContractionProblem) -> GemmViewAnalysis:
    """Prove whether contraction operands can feed GEMM without packing.

    Nano v1 accepts whole-matrix transpose flags, but not strided batched GEMM
    or output scatter. Consequently batch dimensions must prefix each operand
    and the requested output must already be canonical B+M+N C-order storage.
    """

    groups = problem.groups
    lhs, lhs_reasons = _operand_view(
        "left", problem.equation.lhs, problem.lhs, groups.batch, groups.left_free, groups.reduction
    )
    rhs, rhs_reasons = _operand_view(
        "right", problem.equation.rhs, problem.rhs, groups.batch, groups.reduction, groups.right_free
    )
    reasons = list(lhs_reasons) + list(rhs_reasons)
    if problem.equation.output != groups.canonical_output:
        reasons.append(
            "output order {} is not canonical B+M+N ({})".format(
                "".join(problem.equation.output), "".join(groups.canonical_output)
            )
        )
    if not problem.output.is_c_contiguous:
        reasons.append("output is not C-contiguous")
    return GemmViewAnalysis(not reasons, tuple(reasons), lhs, rhs)


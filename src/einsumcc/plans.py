"""Execution-plan construction and explainable cost-based selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .layout import analyze_gemm_view
from .problem import ContractionProblem
from .target import CPU_MODEL, Target


class PlanKind(str, Enum):
    DIRECT = "direct"
    GEMM_VIEW = "gemm-view"
    PACKED_GEMM = "packed-gemm"


@dataclass(frozen=True)
class ExecutionPlan:
    kind: PlanKind
    estimated_us: float
    legal: bool
    rationale: Tuple[str, ...]
    workspace_bytes: int = 0


@dataclass(frozen=True)
class PlanDecision:
    target: Target
    selected: ExecutionPlan
    candidates: Tuple[ExecutionPlan, ...]
    measured: bool = False

    def explain(self) -> str:
        lines = [
            "Target: {}".format(self.target.name),
            "Selected: {} ({:.3f} us{})".format(
                self.selected.kind.value,
                self.selected.estimated_us,
                ", measured" if self.measured else ", analytical estimate",
            ),
            "Candidates:",
        ]
        for plan in self.candidates:
            status = "legal" if plan.legal else "illegal"
            lines.append(
                "  - {}: {} | {:.3f} us | workspace={} bytes".format(
                    plan.kind.value, status, plan.estimated_us, plan.workspace_bytes
                )
            )
            for reason in plan.rationale:
                lines.append("      {}".format(reason))
        return "\n".join(lines)


class Planner:
    """Construct all plan candidates and choose the cheapest legal one."""

    def __init__(self, target: Target = CPU_MODEL) -> None:
        self.target = target

    def enumerate(self, problem: ContractionProblem) -> Tuple[ExecutionPlan, ...]:
        target = self.target
        launch = target.launch_overhead_us
        base_bytes = problem.lhs.nbytes + problem.rhs.nbytes + problem.output.nbytes

        # Direct code touches original layouts. The reuse factor is deliberately
        # conservative: the model is for candidate ordering, not a benchmark.
        direct_compute = problem.flops / target.direct_flops_per_second * 1.0e6
        direct_memory = base_bytes / target.memory_bytes_per_second * 1.0e6
        indexing_penalty = (problem.output.numel * (len(problem.equation.output) + 1)) / 3.0e10 * 1.0e6
        direct_us = launch + max(direct_compute, direct_memory) + indexing_penalty
        direct = ExecutionPlan(
            PlanKind.DIRECT,
            direct_us,
            True,
            (
                "operates on original tensor layouts",
                "estimate includes high-rank index reconstruction",
            ),
        )

        view = analyze_gemm_view(problem)
        gemm_compute = problem.flops / target.gemm_flops_per_second * 1.0e6
        gemm_memory = base_bytes / target.memory_bytes_per_second * 1.0e6
        view_us = launch + max(gemm_compute, gemm_memory)
        view_reasons = ("all B/M/N/K groups collapse without materialization",) if view.legal else view.reasons
        gemm_view = ExecutionPlan(
            PlanKind.GEMM_VIEW,
            view_us if view.legal else float("inf"),
            view.legal,
            tuple(view_reasons),
        )

        # Packing reads and writes both operands. Unpacking reads a canonical
        # GEMM output and writes the requested output. If output order is already
        # canonical, only the final GEMM write is necessary.
        input_pack_bytes = 2 * (problem.lhs.nbytes + problem.rhs.nbytes)
        needs_unpack = problem.equation.output != problem.groups.canonical_output or not problem.output.is_c_contiguous
        output_pack_bytes = 2 * problem.output.nbytes if needs_unpack else 0
        layout_bytes = input_pack_bytes + output_pack_bytes
        pack_us = layout_bytes / target.memory_bytes_per_second * 1.0e6
        packed_launches = 3 + int(needs_unpack)
        packed_us = packed_launches * launch + pack_us + max(gemm_compute, gemm_memory)
        workspace = problem.lhs.nbytes + problem.rhs.nbytes + (problem.output.nbytes if needs_unpack else 0)
        packed = ExecutionPlan(
            PlanKind.PACKED_GEMM,
            packed_us,
            True,
            (
                "materializes canonical B/M/K and B/K/N operands",
                "{} output unpack".format("requires" if needs_unpack else "avoids"),
            ),
            workspace,
        )
        return direct, gemm_view, packed

    def choose(
        self, problem: ContractionProblem, force: Optional[PlanKind] = None
    ) -> PlanDecision:
        candidates = self.enumerate(problem)
        legal = [candidate for candidate in candidates if candidate.legal]
        if force is not None:
            matching = [candidate for candidate in candidates if candidate.kind == force]
            if not matching or not matching[0].legal:
                reasons = matching[0].rationale if matching else ("plan was not generated",)
                raise ValueError(
                    "forced plan '{}' is illegal: {}".format(force.value, "; ".join(reasons))
                )
            selected = matching[0]
        else:
            selected = min(legal, key=lambda candidate: candidate.estimated_us)
        return PlanDecision(self.target, selected, candidates)

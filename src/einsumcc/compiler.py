"""Compiler façade joining semantic IR, planning, tuning, and execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from .cache import TuningCache, TuningRecord
from .cpu_backend import execute
from .plans import PlanDecision, PlanKind, Planner
from .problem import ContractionProblem
from .schedule import DirectSchedule, ScheduleSpace
from .target import CPU_MODEL, Target
from .tuner import EmpiricalTuner, TuningResult


@dataclass(frozen=True)
class CompiledContraction:
    """An immutable plan ready for execution on a named target."""

    problem: ContractionProblem
    decision: PlanDecision
    schedule: Optional[DirectSchedule]
    cache_record: Optional[TuningRecord] = None

    def run_cpu(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if self.decision.target.name != CPU_MODEL.name:
            raise ValueError(
                "compiled target '{}' cannot execute on the CPU backend".format(
                    self.decision.target.name
                )
            )
        return execute(self.problem, lhs, rhs, self.decision.selected, self.schedule)

    def explain(self) -> str:
        problem = self.problem
        lines = [
            "Equation: {}".format(problem.equation.text),
            "Index groups:",
            "  B (batch): {}".format("".join(problem.groups.batch) or "-"),
            "  M (left free): {}".format("".join(problem.groups.left_free) or "-"),
            "  N (right free): {}".format("".join(problem.groups.right_free) or "-"),
            "  K (reduction): {}".format("".join(problem.groups.reduction)),
            "Collapsed sizes: B={} M={} N={} K={}".format(
                problem.batch_size, problem.m, problem.n, problem.k
            ),
            "Work: {} FLOPs".format(problem.flops),
            self.decision.explain(),
        ]
        if self.schedule is not None:
            lines.append("Direct schedule: {}".format(dict(self.schedule.as_dict())))
        lines.append("Decision source: {}".format("tuning cache" if self.cache_record else "cost model"))
        lines.append("Workload key: {}".format(problem.workload_key))
        return "\n".join(lines)


class Compiler:
    """Target-aware compiler with optional persistent empirical choices."""

    def __init__(
        self,
        target: Target = CPU_MODEL,
        *,
        cache: Optional[TuningCache] = None,
        schedule_space: Optional[ScheduleSpace] = None,
    ) -> None:
        self.target = target
        self.planner = Planner(target)
        self.cache = cache
        self.schedule_space = schedule_space or ScheduleSpace()

    def make_problem(
        self,
        equation: str,
        lhs_shape: Sequence[int],
        rhs_shape: Sequence[int],
        **metadata: object
    ) -> ContractionProblem:
        return ContractionProblem.create(equation, lhs_shape, rhs_shape, **metadata)

    def compile(
        self,
        problem: ContractionProblem,
        *,
        force: Optional[PlanKind] = None,
        use_cache: bool = True,
    ) -> CompiledContraction:
        analytical = self.planner.choose(problem, force=force)
        record = (
            self.cache.lookup(problem.workload_key, self.target.name)
            if self.cache is not None and use_cache and force is None
            else None
        )
        if record is not None:
            matches = [
                plan
                for plan in analytical.candidates
                if plan.kind == record.plan_kind and plan.legal
            ]
            schedule_is_valid = True
            if record.plan_kind == PlanKind.DIRECT:
                schedule_is_valid = (
                    record.schedule is not None
                    and self.schedule_space.assess(
                        problem, self.target, record.schedule
                    ).legal
                )
            elif record.schedule is not None:
                schedule_is_valid = False
            if matches and schedule_is_valid:
                measured_plan = replace(matches[0], estimated_us=record.median_us)
                decision = PlanDecision(
                    self.target, measured_plan, analytical.candidates, measured=True
                )
                return CompiledContraction(problem, decision, record.schedule, record)

        schedule = None
        if analytical.selected.kind == PlanKind.DIRECT:
            ranked = self.schedule_space.ranked(problem, self.target, limit=1)
            if ranked:
                schedule = ranked[0]
            elif force == PlanKind.DIRECT:
                raise ValueError(
                    "forced Direct plan has no legal schedule for target '{}'".format(
                        self.target.name
                    )
                )
            else:
                fallback = min(
                    (
                        candidate
                        for candidate in analytical.candidates
                        if candidate.legal and candidate.kind != PlanKind.DIRECT
                    ),
                    key=lambda candidate: candidate.estimated_us,
                )
                analytical = PlanDecision(
                    self.target, fallback, analytical.candidates, measured=False
                )
        return CompiledContraction(problem, analytical, schedule)

    def tune_cpu(
        self,
        problem: ContractionProblem,
        lhs: np.ndarray,
        rhs: np.ndarray,
        *,
        warmups: int = 1,
        repeats: int = 5,
        max_direct_schedules: int = 12,
    ) -> TuningResult:
        if self.target.name != CPU_MODEL.name:
            raise ValueError("CPU tuner cannot create records for target '{}'".format(self.target.name))
        tuner = EmpiricalTuner(
            self.planner,
            self.schedule_space,
            warmups=warmups,
            repeats=repeats,
            max_direct_schedules=max_direct_schedules,
        )
        result = tuner.tune_cpu(problem, lhs, rhs)
        if self.cache is not None:
            self.cache.store(result.to_record(problem, self.target.name))
        return result

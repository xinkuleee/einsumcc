"""Empirical plan and Direct-schedule tuning on executable backends."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .cache import TuningRecord
from .cpu_backend import execute
from .plans import ExecutionPlan, PlanKind, Planner
from .problem import ContractionProblem
from .schedule import DirectSchedule, ScheduleSpace


@dataclass(frozen=True)
class TuningMeasurement:
    plan: ExecutionPlan
    schedule: Optional[DirectSchedule]
    samples_us: Tuple[float, ...]

    @property
    def median_us(self) -> float:
        return float(median(self.samples_us))


@dataclass(frozen=True)
class TuningResult:
    best: TuningMeasurement
    measurements: Tuple[TuningMeasurement, ...]

    def to_record(self, problem: ContractionProblem, target: str) -> TuningRecord:
        return TuningRecord.create(
            problem.workload_key,
            target,
            self.best.plan.kind,
            self.best.median_us,
            self.best.samples_us,
            self.best.schedule,
        )


class EmpiricalTuner:
    """Measure statically pruned candidates and return the fastest one."""

    def __init__(
        self,
        planner: Planner,
        schedule_space: Optional[ScheduleSpace] = None,
        *,
        warmups: int = 1,
        repeats: int = 5,
        max_direct_schedules: int = 12,
    ) -> None:
        if warmups < 0 or repeats <= 0 or max_direct_schedules <= 0:
            raise ValueError("invalid tuner iteration counts")
        self.planner = planner
        self.schedule_space = schedule_space or ScheduleSpace()
        self.warmups = warmups
        self.repeats = repeats
        self.max_direct_schedules = max_direct_schedules

    def _measure(self, operation: Callable[[], np.ndarray]) -> Tuple[float, ...]:
        for _ in range(self.warmups):
            operation()
        samples: List[float] = []
        for _ in range(self.repeats):
            started = perf_counter_ns()
            operation()
            samples.append((perf_counter_ns() - started) / 1000.0)
        return tuple(samples)

    def tune_cpu(
        self, problem: ContractionProblem, lhs: np.ndarray, rhs: np.ndarray
    ) -> TuningResult:
        """Tune all legal plans using the independent CPU implementations.

        This selects a CPU record only. It must never be reused for the A100
        target because the target name participates in the cache key.
        """

        reference = np.einsum(problem.equation.text, lhs, rhs, dtype=np.float32)
        measurements: List[TuningMeasurement] = []
        for plan in self.planner.enumerate(problem):
            if not plan.legal:
                continue
            schedules: Sequence[Optional[DirectSchedule]]
            if plan.kind == PlanKind.DIRECT:
                schedules = self.schedule_space.ranked(
                    problem, self.planner.target, self.max_direct_schedules
                )
            else:
                schedules = (None,)
            for schedule in schedules:
                operation = lambda plan=plan, schedule=schedule: execute(
                    problem, lhs, rhs, plan, schedule
                )
                candidate = operation()
                np.testing.assert_allclose(candidate, reference, rtol=1.0e-4, atol=1.0e-5)
                measurements.append(
                    TuningMeasurement(plan, schedule, self._measure(operation))
                )
        if not measurements:
            raise RuntimeError("no executable tuning candidates")
        best = min(measurements, key=lambda value: value.median_us)
        return TuningResult(best, tuple(measurements))


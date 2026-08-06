"""Empirical plan and Direct-schedule tuning on executable backends."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from statistics import median
from time import perf_counter_ns
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .cache import TuningRecord
from .cpu_backend import execute
from .plans import ExecutionPlan, PlanKind, Planner
from .problem import ContractionProblem
from .schedule import DirectSchedule, ScheduleSpace
from .target import CPU_MODEL

if TYPE_CHECKING:
    from .native_backend import NativeCpuCompiler


NATIVE_CPU_TUNING_TARGET = "native-cpu-mini-v0.1"


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


@dataclass(frozen=True)
class NativeTuningMeasurement:
    """One schedule measured by a compiled native shared library."""

    schedule: DirectSchedule
    expanded_tile_sizes: Tuple[int, ...]
    samples_us: Tuple[float, ...]
    max_abs_error: float
    artifact_key: str
    cache_hit: bool

    @property
    def median_us(self) -> float:
        return float(median(self.samples_us))

    def to_json(self) -> Mapping[str, object]:
        return {
            "schedule": dict(self.schedule.as_dict()),
            "expanded_tile_sizes": list(self.expanded_tile_sizes),
            "samples_us": list(self.samples_us),
            "median_us": self.median_us,
            "min_us": min(self.samples_us),
            "max_us": max(self.samples_us),
            "max_abs_error": self.max_abs_error,
            "artifact_key": self.artifact_key,
            "artifact_cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class NativeTuningResult:
    target: str
    environment: Mapping[str, object]
    best: NativeTuningMeasurement
    baseline: NativeTuningMeasurement
    measurements: Tuple[NativeTuningMeasurement, ...]

    @property
    def speedup_vs_baseline(self) -> Optional[float]:
        if self.best.median_us == 0.0:
            return 1.0 if self.baseline.median_us == 0.0 else None
        return self.baseline.median_us / self.best.median_us

    def to_record(self, problem: ContractionProblem) -> TuningRecord:
        return TuningRecord.create(
            problem.workload_key,
            self.target,
            PlanKind.DIRECT,
            self.best.median_us,
            self.best.samples_us,
            self.best.schedule,
        )


class NativeDirectTuner:
    """Compile, validate, and time real native Direct schedules.

    The cost model only orders the search. Candidates that project to the same
    Einstein-loop tiles are deduplicated; every surviving candidate is compiled
    to a dylib, checked against NumPy once, bound once, and measured by repeated
    calls to that binding.
    """

    def __init__(
        self,
        compiler: "NativeCpuCompiler",
        schedule_space: Optional[ScheduleSpace] = None,
        *,
        warmups: int = 1,
        repeats: int = 5,
        max_schedules: int = 12,
        rtol: float = 1.0e-4,
        atol: float = 1.0e-5,
    ) -> None:
        if warmups < 0 or repeats <= 0 or max_schedules <= 0:
            raise ValueError("invalid native tuner iteration counts")
        if rtol < 0 or atol < 0:
            raise ValueError("native tuner tolerances must be non-negative")
        self.compiler = compiler
        self.schedule_space = schedule_space or ScheduleSpace()
        self.warmups = warmups
        self.repeats = repeats
        self.max_schedules = max_schedules
        self.rtol = float(rtol)
        self.atol = float(atol)

    def candidates(
        self, problem: ContractionProblem
    ) -> Tuple[DirectSchedule, ...]:
        ranked = self.schedule_space.ranked(problem, CPU_MODEL, limit=4096)
        unique = []
        seen = set()
        for schedule in ranked:
            expanded = schedule.expanded_tile_sizes(problem)
            if expanded in seen:
                continue
            seen.add(expanded)
            unique.append(schedule)
            if len(unique) == self.max_schedules:
                break
        if not unique:
            raise RuntimeError("no native Direct schedules survived pruning")
        return tuple(unique)

    def _measure(self, operation: Callable[[], np.ndarray]) -> Tuple[float, ...]:
        for _ in range(self.warmups):
            operation()
        samples = []
        for _ in range(self.repeats):
            started = perf_counter_ns()
            operation()
            samples.append((perf_counter_ns() - started) / 1000.0)
        return tuple(samples)

    def tune(
        self, problem: ContractionProblem, lhs: np.ndarray, rhs: np.ndarray
    ) -> NativeTuningResult:
        reference = np.einsum(
            problem.equation.text, lhs, rhs, dtype=np.float32
        )
        measurements = []
        for schedule in self.candidates(problem):
            kernel = self.compiler.compile(problem, schedule)
            output = np.empty(problem.output.shape, dtype=np.float32, order="C")
            invocation = kernel.bind(lhs, rhs, output)
            result = invocation.run()
            if not np.isfinite(result).all():
                raise AssertionError(
                    "native schedule {} produced non-finite values".format(
                        dict(schedule.as_dict())
                    )
                )
            np.testing.assert_allclose(
                result, reference, rtol=self.rtol, atol=self.atol
            )
            difference = np.abs(result - reference)
            max_abs_error = (
                float(np.max(difference))
                if difference.size
                else float(difference)
            )
            samples = self._measure(invocation.run)
            measurements.append(
                NativeTuningMeasurement(
                    schedule,
                    kernel.expanded_tile_sizes,
                    samples,
                    max_abs_error,
                    kernel.cache_key,
                    kernel.cache_hit,
                )
            )
        baseline = measurements[0]
        best = min(measurements, key=lambda value: value.median_us)
        return NativeTuningResult(
            self.compiler.tuning_target,
            {
                **self.compiler.toolchain.identity(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "timer": "perf_counter_ns",
                "timing_scope": "bound ctypes host-call latency",
            },
            best,
            baseline,
            tuple(measurements),
        )

"""Reproducible semantic and native CPU benchmark corpora."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .compiler import Compiler
from .cpu_backend import execute
from .cache import TuningCache
from .native_backend import NativeCpuCompiler
from .problem import ContractionProblem
from .schedule import ScheduleSpace
from .target import CPU_MODEL
from .tuner import NativeDirectTuner

if TYPE_CHECKING:
    from .native_backend import NativeToolchain

CORPUS_SCHEMA = 1
RESULT_SCHEMA = 1


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    equation: str
    lhs_shape: Tuple[int, ...]
    rhs_shape: Tuple[int, ...]
    lhs_strides: Optional[Tuple[int, ...]] = None
    rhs_strides: Optional[Tuple[int, ...]] = None

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "BenchmarkCase":
        try:
            return cls(
                str(value["name"]),
                str(value["equation"]),
                tuple(int(item) for item in value["lhs_shape"]),  # type: ignore[union-attr]
                tuple(int(item) for item in value["rhs_shape"]),  # type: ignore[union-attr]
                (
                    None
                    if value.get("lhs_strides") is None
                    else tuple(int(item) for item in value["lhs_strides"])  # type: ignore[union-attr]
                ),
                (
                    None
                    if value.get("rhs_strides") is None
                    else tuple(int(item) for item in value["rhs_strides"])  # type: ignore[union-attr]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid benchmark case: {}".format(error)) from error

    def problem(self) -> ContractionProblem:
        return ContractionProblem.create(
            self.equation,
            self.lhs_shape,
            self.rhs_shape,
            lhs_strides=self.lhs_strides,
            rhs_strides=self.rhs_strides,
        )


@dataclass(frozen=True)
class BenchmarkCorpus:
    name: str
    description: str
    cases: Tuple[BenchmarkCase, ...]

    @classmethod
    def load(cls, path: Path) -> "BenchmarkCorpus":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("cannot read benchmark corpus '{}': {}".format(path, error)) from error
        if not isinstance(payload, dict) or payload.get("schema") != CORPUS_SCHEMA:
            raise ValueError("unsupported benchmark corpus schema")
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("benchmark corpus must contain at least one case")
        cases = tuple(BenchmarkCase.from_json(value) for value in raw_cases)
        names = [case.name for case in cases]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("benchmark case names must be non-empty and unique")
        return cls(str(payload.get("name", path.stem)), str(payload.get("description", "")), cases)


def _random_array(
    shape: Sequence[int], strides: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    required = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    storage = rng.standard_normal(required).astype(np.float32)
    return np.lib.stride_tricks.as_strided(
        storage,
        shape=tuple(shape),
        strides=tuple(stride * storage.itemsize for stride in strides),
    )


class CpuBenchmarkRunner:
    """Benchmark default schedules for all plans; never used for A100 claims."""

    def __init__(self, warmups: int = 1, repeats: int = 5, seed: int = 0) -> None:
        if warmups < 0 or repeats <= 0:
            raise ValueError("invalid benchmark iteration counts")
        self.warmups = warmups
        self.repeats = repeats
        self.seed = seed
        self.compiler = Compiler(CPU_MODEL)

    def _measure(self, operation) -> Tuple[float, ...]:
        for _ in range(self.warmups):
            operation()
        samples = []
        for _ in range(self.repeats):
            started = perf_counter_ns()
            operation()
            samples.append((perf_counter_ns() - started) / 1000.0)
        return tuple(samples)

    def run_case(self, case: BenchmarkCase, case_index: int) -> Mapping[str, object]:
        problem = case.problem()
        rng = np.random.default_rng(self.seed + case_index)
        lhs = _random_array(problem.lhs.shape, problem.lhs.strides, rng)
        rhs = _random_array(problem.rhs.shape, problem.rhs.strides, rng)
        reference = np.einsum(problem.equation.text, lhs, rhs, dtype=np.float32)
        selected = self.compiler.compile(problem, use_cache=False)
        plans: List[Mapping[str, object]] = []
        for plan in self.compiler.planner.enumerate(problem):
            entry: Dict[str, object] = {
                "kind": plan.kind.value,
                "legal": plan.legal,
                "workspace_bytes": plan.workspace_bytes,
                "analytical_us": plan.estimated_us if plan.legal else None,
                "rationale": list(plan.rationale),
            }
            if plan.legal:
                compiled = self.compiler.compile(
                    problem, force=plan.kind, use_cache=False
                )
                operation = lambda compiled=compiled: compiled.run_cpu(lhs, rhs)
                result = operation()
                if not np.isfinite(result).all():
                    raise AssertionError("plan '{}' produced non-finite values".format(plan.kind.value))
                np.testing.assert_allclose(result, reference, rtol=1.0e-4, atol=1.0e-5)
                samples = self._measure(operation)
                entry.update(
                    {
                        "samples_us": list(samples),
                        "median_us": float(median(samples)),
                        "min_us": min(samples),
                        "max_abs_error": float(np.max(np.abs(result - reference))),
                        "schedule": (
                            None
                            if compiled.schedule is None
                            else dict(compiled.schedule.as_dict())
                        ),
                    }
                )
            plans.append(entry)
        return {
            "name": case.name,
            "equation": problem.equation.text,
            "lhs_shape": list(problem.lhs.shape),
            "rhs_shape": list(problem.rhs.shape),
            "lhs_strides": list(problem.lhs.strides),
            "rhs_strides": list(problem.rhs.strides),
            "workload_key": problem.workload_key,
            "flops": problem.flops,
            "analytical_selection": selected.decision.selected.kind.value,
            "plans": plans,
        }

    def run(self, corpus: BenchmarkCorpus) -> Mapping[str, object]:
        return {
            "schema": RESULT_SCHEMA,
            "corpus": corpus.name,
            "description": corpus.description,
            "target": CPU_MODEL.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "configuration": {
                "warmups": self.warmups,
                "repeats": self.repeats,
                "seed": self.seed,
            },
            "cases": [
                self.run_case(case, index) for index, case in enumerate(corpus.cases)
            ],
        }


class NativeBenchmarkRunner:
    """Tune generated Direct dylibs for every case in a versioned corpus.

    One resolved native compiler is shared by the whole run. Consequently all
    cases have the same hardware/toolchain target identity, artifact cache, and
    timing scope. Each winner is persisted independently under the workload
    and target-scoped tuning-cache key.
    """

    def __init__(
        self,
        *,
        warmups: int = 1,
        repeats: int = 5,
        max_schedules: int = 12,
        seed: int = 0,
        rtol: float = 1.0e-4,
        atol: float = 1.0e-5,
        native_cache: Optional[Path] = None,
        tuning_cache: Optional[TuningCache] = None,
        toolchain: Optional["NativeToolchain"] = None,
        timeout_seconds: int = 180,
        schedule_space: Optional[ScheduleSpace] = None,
        native_compiler: Optional[NativeCpuCompiler] = None,
    ) -> None:
        if warmups < 0 or repeats <= 0 or max_schedules <= 0:
            raise ValueError("invalid native benchmark iteration counts")
        if rtol < 0 or atol < 0:
            raise ValueError("native benchmark tolerances must be non-negative")
        if timeout_seconds <= 0:
            raise ValueError("native benchmark timeout must be positive")
        if native_compiler is not None and (
            native_cache is not None or toolchain is not None
        ):
            raise ValueError(
                "native_compiler cannot be combined with native_cache or toolchain"
            )
        self.warmups = warmups
        self.repeats = repeats
        self.max_schedules = max_schedules
        self.seed = seed
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.schedule_space = schedule_space or ScheduleSpace()
        self.native_compiler = native_compiler or NativeCpuCompiler(
            toolchain=toolchain,
            cache_dir=native_cache,
            timeout_seconds=timeout_seconds,
        )
        self.tuning_cache = tuning_cache

    def _run_case(
        self, case: BenchmarkCase, case_index: int
    ) -> Tuple[Mapping[str, object], Mapping[str, object], str]:
        problem = case.problem()
        rng = np.random.default_rng(self.seed + case_index)
        lhs = _random_array(problem.lhs.shape, problem.lhs.strides, rng)
        rhs = _random_array(problem.rhs.shape, problem.rhs.strides, rng)
        result = NativeDirectTuner(
            self.native_compiler,
            self.schedule_space,
            warmups=self.warmups,
            repeats=self.repeats,
            max_schedules=self.max_schedules,
            rtol=self.rtol,
            atol=self.atol,
        ).tune(problem, lhs, rhs)
        if self.tuning_cache is not None:
            self.tuning_cache.store(result.to_record(problem))
        best_gflops = (
            None
            if result.best.median_us == 0.0
            else problem.flops / (result.best.median_us * 1000.0)
        )
        baseline_gflops = (
            None
            if result.baseline.median_us == 0.0
            else problem.flops / (result.baseline.median_us * 1000.0)
        )
        entry = {
            "name": case.name,
            "equation": problem.equation.text,
            "lhs_shape": list(problem.lhs.shape),
            "rhs_shape": list(problem.rhs.shape),
            "lhs_strides": list(problem.lhs.strides),
            "rhs_strides": list(problem.rhs.strides),
            "workload_key": problem.workload_key,
            "flops": problem.flops,
            "best": result.best.to_json(),
            "best_gflops": best_gflops,
            "baseline": result.baseline.to_json(),
            "baseline_gflops": baseline_gflops,
            "speedup_vs_baseline": result.speedup_vs_baseline,
            "measurements": [item.to_json() for item in result.measurements],
        }
        return entry, result.environment, result.target

    def run(self, corpus: BenchmarkCorpus) -> Mapping[str, object]:
        cases = []
        environment: Optional[Mapping[str, object]] = None
        target: Optional[str] = None
        for index, case in enumerate(corpus.cases):
            entry, case_environment, case_target = self._run_case(case, index)
            if environment is None:
                environment = case_environment
                target = case_target
            elif case_environment != environment or case_target != target:
                raise RuntimeError(
                    "native benchmark target changed during a corpus run"
                )
            cases.append(entry)
        return {
            "schema": RESULT_SCHEMA,
            "kind": "native-direct-corpus",
            "corpus": corpus.name,
            "description": corpus.description,
            "target": target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": environment,
            "configuration": {
                "warmups": self.warmups,
                "repeats": self.repeats,
                "max_schedules": self.max_schedules,
                "seed": self.seed,
                "rtol": self.rtol,
                "atol": self.atol,
                "baseline_definition": (
                    "first statically ranked distinct schedule"
                ),
                "timing_unit": "microseconds",
            },
            "native_artifact_cache": str(self.native_compiler.cache_dir),
            "tuning_cache": (
                None if self.tuning_cache is None else str(self.tuning_cache.path)
            ),
            "cases": cases,
        }

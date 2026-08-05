"""Reproducible CPU benchmark corpus and machine-readable result schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .compiler import Compiler
from .cpu_backend import execute
from .problem import ContractionProblem
from .target import CPU_MODEL

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

"""Template-based Direct schedule generation and legality pruning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping, Sequence, Tuple

from .problem import ContractionProblem
from .target import Target


@dataclass(frozen=True)
class DirectSchedule:
    """Portable schedule parameters consumed by CPU and future GPU codegen.

    The CPU backend uses B/M/N/K tiles to test semantics and empirical tuning.
    Threads and vector width are backend hints; their target legality is still
    checked here so cached schedules cannot be reused on incompatible targets.
    """

    block_m: int
    block_n: int
    block_k: int
    threads: int
    vector_width: int

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.block_m,
                self.block_n,
                self.block_k,
                self.threads,
                self.vector_width,
            )
        ):
            raise ValueError("Direct schedule values must be positive")

    def as_dict(self) -> Mapping[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, int]) -> "DirectSchedule":
        return cls(
            int(values["block_m"]),
            int(values["block_n"]),
            int(values["block_k"]),
            int(values["threads"]),
            int(values["vector_width"]),
        )


@dataclass(frozen=True)
class ScheduleAssessment:
    schedule: DirectSchedule
    legal: bool
    score: float
    reasons: Tuple[str, ...]


class ScheduleSpace:
    """Small, explicit search space suitable for Nano v1."""

    def __init__(
        self,
        block_m: Sequence[int] = (8, 16, 32, 64),
        block_n: Sequence[int] = (8, 16, 32, 64),
        block_k: Sequence[int] = (4, 8, 16, 32),
        gpu_threads: Sequence[int] = (128, 256),
        vector_widths: Sequence[int] = (1, 2, 4),
    ) -> None:
        self.block_m = tuple(block_m)
        self.block_n = tuple(block_n)
        self.block_k = tuple(block_k)
        self.gpu_threads = tuple(gpu_threads)
        self.vector_widths = tuple(vector_widths)
        if any(
            value <= 0
            for values in (
                self.block_m,
                self.block_n,
                self.block_k,
                self.gpu_threads,
                self.vector_widths,
            )
            for value in values
        ):
            raise ValueError("schedule-space values must be positive")

    def assess(
        self, problem: ContractionProblem, target: Target, schedule: DirectSchedule
    ) -> ScheduleAssessment:
        reasons = []
        if schedule.threads > target.max_threads_per_block:
            reasons.append("thread count exceeds target limit")
        if schedule.vector_width > 1 and problem.k % schedule.vector_width != 0:
            reasons.append("K is not divisible by vector width")
        shared_bytes = (
            schedule.block_m * schedule.block_k
            + schedule.block_n * schedule.block_k
        ) * problem.lhs.itemsize
        if shared_bytes > target.shared_memory_bytes:
            reasons.append("tile exceeds target shared-memory budget")

        # Prefer useful tiles, coalesced vector loads, and limited boundary
        # waste. The score only ranks candidates before empirical measurement.
        used_m = min(problem.m, schedule.block_m)
        used_n = min(problem.n, schedule.block_n)
        used_k = min(problem.k, schedule.block_k)
        utilization = (used_m * used_n * used_k) / (
            schedule.block_m * schedule.block_n * schedule.block_k
        )
        reuse = (used_m * used_n * used_k) / max(1, used_m * used_k + used_n * used_k)
        score = reuse * utilization * schedule.vector_width
        return ScheduleAssessment(schedule, not reasons, score, tuple(reasons))

    def candidates(
        self, problem: ContractionProblem, target: Target
    ) -> Tuple[ScheduleAssessment, ...]:
        cpu_like = target.max_threads_per_block == 1
        threads = (1,) if cpu_like else self.gpu_threads
        vector_widths = (1,) if cpu_like else self.vector_widths
        assessments = [
            self.assess(problem, target, DirectSchedule(bm, bn, bk, th, vw))
            for bm, bn, bk, th, vw in product(
                self.block_m, self.block_n, self.block_k, threads, vector_widths
            )
        ]
        return tuple(assessments)

    def ranked(
        self, problem: ContractionProblem, target: Target, limit: int = 20
    ) -> Tuple[DirectSchedule, ...]:
        legal = [candidate for candidate in self.candidates(problem, target) if candidate.legal]
        legal.sort(key=lambda candidate: candidate.score, reverse=True)
        return tuple(candidate.schedule for candidate in legal[:limit])

    def default(self, problem: ContractionProblem, target: Target) -> DirectSchedule:
        ranked = self.ranked(problem, target, limit=1)
        if not ranked:
            raise ValueError("no legal Direct schedule for target '{}'".format(target.name))
        return ranked[0]

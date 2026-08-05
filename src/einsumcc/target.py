"""Target descriptions used by legality pruning and analytical costing.

These are intentionally transparent first-order models, not benchmark claims.
Measured tuning results are stored separately and always take precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Target:
    name: str
    memory_bandwidth_gbps: float
    direct_compute_tflops: float
    gemm_compute_tflops: float
    launch_overhead_us: float
    max_threads_per_block: int = 1024
    shared_memory_bytes: int = 48 * 1024

    @property
    def memory_bytes_per_second(self) -> float:
        return self.memory_bandwidth_gbps * 1.0e9

    @property
    def direct_flops_per_second(self) -> float:
        return self.direct_compute_tflops * 1.0e12

    @property
    def gemm_flops_per_second(self) -> float:
        return self.gemm_compute_tflops * 1.0e12


CPU_MODEL = Target(
    name="cpu-model",
    memory_bandwidth_gbps=45.0,
    direct_compute_tflops=0.12,
    gemm_compute_tflops=0.8,
    launch_overhead_us=1.0,
    max_threads_per_block=1,
    shared_memory_bytes=256 * 1024,
)

# Conservative effective throughputs for relative planning, not A100 peak
# specifications. They should eventually be calibrated from the course server.
A100_MODEL = Target(
    name="a100-model",
    memory_bandwidth_gbps=1200.0,
    direct_compute_tflops=8.0,
    gemm_compute_tflops=45.0,
    launch_overhead_us=5.0,
    max_threads_per_block=1024,
    shared_memory_bytes=48 * 1024,
)

TARGETS: Dict[str, Target] = {target.name: target for target in (CPU_MODEL, A100_MODEL)}


def get_target(name: str) -> Target:
    try:
        return TARGETS[name]
    except KeyError as error:
        raise ValueError(
            "unknown target '{}'; choose one of {}".format(name, ", ".join(sorted(TARGETS)))
        ) from error


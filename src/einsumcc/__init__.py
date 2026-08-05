"""Public API for the EinsumCC Nano v1 compiler core."""

from .compiler import CompiledContraction, Compiler
from .equation import Equation
from .errors import EinsumCCError, ParseError, VerificationError
from .native_backend import NativeCpuCompiler, NativeKernel, NativeToolchain
from .plans import ExecutionPlan, PlanDecision, PlanKind, Planner
from .problem import ContractionProblem, IndexGroups, TensorSpec

__all__ = [
    "CompiledContraction",
    "Compiler",
    "ContractionProblem",
    "EinsumCCError",
    "Equation",
    "ExecutionPlan",
    "IndexGroups",
    "NativeCpuCompiler",
    "NativeKernel",
    "NativeToolchain",
    "ParseError",
    "PlanDecision",
    "PlanKind",
    "Planner",
    "TensorSpec",
    "VerificationError",
]

# Keep the version available without importing packaging metadata.
__version__ = "0.1.0"

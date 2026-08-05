"""End-to-end differential tests for the native macOS CPU pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "build" / "tools" / "einsumcc-opt" / "einsumcc-opt"
TRANSLATE = ROOT / ".deps" / "llvm-macos-arm64" / "bin" / "mlir-translate"
WORK = ROOT / "build" / "cpu-codegen-tests"

sys.path.insert(0, str(ROOT / "src"))

from einsumcc.native_backend import NativeCpuCompiler, NativeToolchain  # noqa: E402
from einsumcc.problem import ContractionProblem  # noqa: E402


Case = Tuple[str, Tuple[int, ...], Tuple[int, ...]]
CASES: Tuple[Case, ...] = (
    ("mk,kn->mn", (3, 4), (4, 5)),
    ("bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
    ("aij,jib->ab", (2, 3, 4), (4, 3, 5)),
    ("ij,ij->", (3, 4), (3, 4)),
    ("mkl,kln->mn", (2, 3, 4), (3, 4, 5)),
    ("aijd,bckd->abcijk", (2, 2, 2, 3), (2, 2, 2, 3)),
)
STRIDED_CASE: Case = ("mk,kn->mn", (3, 4), (4, 5))


def _strided_random(
    shape: Tuple[int, ...], strides: Tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Create a positive-stride view whose descriptor matches Nano metadata."""

    storage_size = 1 + sum(
        (extent - 1) * stride for extent, stride in zip(shape, strides)
    )
    storage = rng.standard_normal(storage_size).astype(np.float32)
    return np.lib.stride_tricks.as_strided(
        storage,
        shape=shape,
        strides=tuple(stride * storage.itemsize for stride in strides),
    )


def run_case(compiler: NativeCpuCompiler, case: Case, index: int) -> None:
    equation, lhs_shape, rhs_shape = case
    problem = ContractionProblem.create(equation, lhs_shape, rhs_shape)
    rng = np.random.default_rng(100 + index)
    lhs = rng.standard_normal(lhs_shape).astype(np.float32)
    rhs = rng.standard_normal(rhs_shape).astype(np.float32)
    reference = np.einsum(equation, lhs, rhs, dtype=np.float32)
    output = compiler.compile(problem).run(lhs, rhs)

    np.testing.assert_allclose(output, reference, rtol=1.0e-4, atol=1.0e-5)
    print("PASS {}".format(equation))


def run_strided_case(compiler: NativeCpuCompiler) -> None:
    """Exercise the native memref ABI with legal non-contiguous inputs."""

    equation, lhs_shape, rhs_shape = STRIDED_CASE
    lhs_strides = (12, 2)
    rhs_strides = (10, 1)
    problem = ContractionProblem.create(
        equation,
        lhs_shape,
        rhs_shape,
        lhs_strides=lhs_strides,
        rhs_strides=rhs_strides,
    )
    rng = np.random.default_rng(200)
    lhs = _strided_random(lhs_shape, lhs_strides, rng)
    rhs = _strided_random(rhs_shape, rhs_strides, rng)
    reference = np.einsum(equation, lhs, rhs, dtype=np.float32)
    output = compiler.compile(problem).run(lhs, rhs)

    np.testing.assert_allclose(output, reference, rtol=1.0e-4, atol=1.0e-5)
    print("PASS {} (positive-stride inputs)".format(equation))


def main() -> int:
    if not OPT.is_file() or not TRANSLATE.is_file():
        raise SystemExit("build einsumcc-opt and bootstrap MLIR before CPU codegen tests")
    compiler = NativeCpuCompiler(
        toolchain=NativeToolchain.discover(ROOT), cache_dir=WORK / "native-cache"
    )
    for index, case in enumerate(CASES):
        run_case(compiler, case, index)
    run_strided_case(compiler)
    print("native CPU differential tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

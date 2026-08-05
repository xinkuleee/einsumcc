"""End-to-end differential tests for the native macOS CPU pipeline."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "build" / "tools" / "einsumcc-opt" / "einsumcc-opt"
TRANSLATE = ROOT / ".deps" / "llvm-macos-arm64" / "bin" / "mlir-translate"
WORK = ROOT / "build" / "cpu-codegen-tests"

sys.path.insert(0, str(ROOT / "src"))

from einsumcc.problem import ContractionProblem  # noqa: E402
from einsumcc.tc_emitter import TcEmitter  # noqa: E402


Case = Tuple[str, Tuple[int, ...], Tuple[int, ...]]
CASES: Tuple[Case, ...] = (
    ("mk,kn->mn", (3, 4), (4, 5)),
    ("bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
    ("aij,jib->ab", (2, 3, 4), (4, 3, 5)),
    ("ij,ij->", (3, 4), (3, 4)),
    ("mkl,kln->mn", (2, 3, 4), (3, 4, 5)),
    ("aijd,bckd->abcijk", (2, 2, 2, 3), (2, 2, 2, 3)),
)


PIPELINE = (
    "--tc-contract-to-linalg",
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--buffer-results-to-out-params=hoist-static-allocs modify-public-functions",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--convert-arith-to-llvm",
    "--convert-index-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
)


def _memref_arguments(array: np.ndarray) -> Tuple[object, ...]:
    pointer = ctypes.c_void_p(int(array.ctypes.data))
    strides = tuple(value // array.itemsize for value in array.strides)
    return (
        pointer,
        pointer,
        ctypes.c_int64(0),
        *(ctypes.c_int64(value) for value in array.shape),
        *(ctypes.c_int64(value) for value in strides),
    )


def _argument_types(arrays: Iterable[np.ndarray]) -> Sequence[type]:
    result = []
    for array in arrays:
        result.extend((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64))
        result.extend(ctypes.c_int64 for _ in array.shape)
        result.extend(ctypes.c_int64 for _ in array.strides)
    return result


def compile_case(problem: ContractionProblem, name: str) -> Path:
    case_dir = WORK / name
    case_dir.mkdir(parents=True, exist_ok=True)
    tc_path = case_dir / "input.mlir"
    llvm_mlir_path = case_dir / "llvm.mlir"
    llvm_ir_path = case_dir / "module.ll"
    library_path = case_dir / "module.dylib"
    tc_path.write_text(TcEmitter().emit(problem).text, encoding="utf-8")

    subprocess.run(
        [str(OPT), str(tc_path), *PIPELINE, "-o", str(llvm_mlir_path)],
        check=True,
    )
    subprocess.run(
        [str(TRANSLATE), "--mlir-to-llvmir", str(llvm_mlir_path), "-o", str(llvm_ir_path)],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/clang",
            "-O2",
            "-Wno-override-module",
            "-dynamiclib",
            str(llvm_ir_path),
            "-o",
            str(library_path),
        ],
        check=True,
    )
    return library_path


def run_case(case: Case, index: int) -> None:
    equation, lhs_shape, rhs_shape = case
    problem = ContractionProblem.create(equation, lhs_shape, rhs_shape)
    rng = np.random.default_rng(100 + index)
    lhs = rng.standard_normal(lhs_shape).astype(np.float32)
    rhs = rng.standard_normal(rhs_shape).astype(np.float32)
    output = np.empty(problem.output.shape, dtype=np.float32)
    reference = np.einsum(equation, lhs, rhs, dtype=np.float32)

    library = ctypes.CDLL(str(compile_case(problem, "case-{}".format(index))))
    function = library.contract
    arrays = (lhs, rhs, output)
    function.argtypes = _argument_types(arrays)
    function.restype = None
    arguments = tuple(value for array in arrays for value in _memref_arguments(array))
    function(*arguments)

    np.testing.assert_allclose(output, reference, rtol=1.0e-4, atol=1.0e-5)
    print("PASS {}".format(equation))


def main() -> int:
    if not OPT.is_file() or not TRANSLATE.is_file():
        raise SystemExit("build einsumcc-opt and bootstrap MLIR before CPU codegen tests")
    for index, case in enumerate(CASES):
        run_case(case, index)
    print("native CPU differential tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

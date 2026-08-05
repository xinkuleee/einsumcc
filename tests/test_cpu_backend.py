import unittest

import numpy as np

from einsumcc.compiler import Compiler
from einsumcc.cpu_backend import execute
from einsumcc.plans import PlanKind, Planner
from einsumcc.problem import ContractionProblem
from einsumcc.target import CPU_MODEL


def strided_random(shape, strides, seed):
    rng = np.random.default_rng(seed)
    size = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    storage = rng.standard_normal(size).astype(np.float32)
    return np.lib.stride_tricks.as_strided(
        storage,
        shape=shape,
        strides=tuple(value * storage.itemsize for value in strides),
    )


class CpuBackendTest(unittest.TestCase):
    CASES = (
        ("mk,kn->mn", (3, 4), (4, 5)),
        ("bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
        ("aij,jib->ab", (2, 3, 4), (4, 3, 5)),
        ("ij,ij->", (3, 4), (3, 4)),
        ("ki,kj->ij", (4, 3), (4, 5)),
        ("klij,mnkl->ijmn", (2, 3, 4, 5), (6, 7, 2, 3)),
        ("aijd,bckd->abcijk", (2, 2, 2, 3), (2, 2, 2, 3)),
    )

    def test_every_legal_plan_matches_numpy(self):
        rng = np.random.default_rng(42)
        for equation, lhs_shape, rhs_shape in self.CASES:
            with self.subTest(equation=equation):
                problem = ContractionProblem.create(equation, lhs_shape, rhs_shape)
                lhs = rng.standard_normal(lhs_shape).astype(np.float32)
                rhs = rng.standard_normal(rhs_shape).astype(np.float32)
                reference = np.einsum(equation, lhs, rhs, dtype=np.float32)
                for plan in Planner(CPU_MODEL).enumerate(problem):
                    if not plan.legal:
                        continue
                    result = execute(problem, lhs, rhs, plan)
                    self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_allclose(result, reference, rtol=1.0e-4, atol=1.0e-5)

    def test_strided_inputs_match_numpy_for_direct_and_packed(self):
        problem = ContractionProblem.create(
            "mk,kn->mn",
            (3, 4),
            (4, 5),
            lhs_strides=(12, 2),
            rhs_strides=(10, 1),
        )
        lhs = strided_random(problem.lhs.shape, problem.lhs.strides, 1)
        rhs = strided_random(problem.rhs.shape, problem.rhs.strides, 2)
        reference = np.einsum(problem.equation.text, lhs, rhs, dtype=np.float32)
        plans = {plan.kind: plan for plan in Planner().enumerate(problem)}
        for kind in (PlanKind.DIRECT, PlanKind.PACKED_GEMM):
            result = execute(problem, lhs, rhs, plans[kind])
            np.testing.assert_allclose(result, reference, rtol=1.0e-4, atol=1.0e-5)

    def test_compiler_facade_runs_selected_cpu_plan(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        lhs = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs = np.arange(20, dtype=np.float32).reshape(4, 5)
        compiled = Compiler(CPU_MODEL).compile(problem)
        np.testing.assert_allclose(
            compiled.run_cpu(lhs, rhs), np.einsum("mk,kn->mn", lhs, rhs)
        )
        self.assertIn("Index groups", compiled.explain())


if __name__ == "__main__":
    unittest.main()

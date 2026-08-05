import unittest

from einsumcc.layout import analyze_gemm_view
from einsumcc.plans import PlanKind, Planner
from einsumcc.problem import ContractionProblem
from einsumcc.target import A100_MODEL, CPU_MODEL


class PlannerTest(unittest.TestCase):
    def test_zero_copy_gemm_for_canonical_matmul(self):
        problem = ContractionProblem.create("mk,kn->mn", (32, 16), (16, 64))
        analysis = analyze_gemm_view(problem)
        self.assertTrue(analysis.legal, analysis.reasons)
        plans = {plan.kind: plan for plan in Planner(A100_MODEL).enumerate(problem)}
        self.assertTrue(plans[PlanKind.GEMM_VIEW].legal)
        self.assertEqual(plans[PlanKind.GEMM_VIEW].workspace_bytes, 0)

    def test_whole_matrix_transpose_can_be_a_view(self):
        problem = ContractionProblem.create("km,nk->mn", (16, 32), (64, 16))
        analysis = analyze_gemm_view(problem)
        self.assertTrue(analysis.legal, analysis.reasons)
        self.assertTrue(analysis.lhs.transpose)
        self.assertTrue(analysis.rhs.transpose)

    def test_multidimensional_matrix_groups_can_use_transpose_views(self):
        problem = ContractionProblem.create(
            "klij,mnkl->ijmn", (2, 3, 4, 5), (6, 7, 2, 3)
        )
        analysis = analyze_gemm_view(problem)
        self.assertTrue(analysis.legal, analysis.reasons)
        self.assertTrue(analysis.lhs.transpose)
        self.assertTrue(analysis.rhs.transpose)

    def test_course_output_permutation_requires_materialization(self):
        problem = ContractionProblem.create(
            "aijd,bckd->abcijk", (2, 3, 4, 5), (6, 7, 8, 5)
        )
        analysis = analyze_gemm_view(problem)
        self.assertFalse(analysis.legal)
        self.assertTrue(any("output order" in reason for reason in analysis.reasons))

    def test_non_contiguous_input_disables_view_but_not_other_plans(self):
        problem = ContractionProblem.create(
            "mk,kn->mn", (3, 4), (4, 5), lhs_strides=(8, 2)
        )
        plans = {plan.kind: plan for plan in Planner().enumerate(problem)}
        self.assertFalse(plans[PlanKind.GEMM_VIEW].legal)
        self.assertTrue(plans[PlanKind.DIRECT].legal)
        self.assertTrue(plans[PlanKind.PACKED_GEMM].legal)

    def test_force_each_legal_plan(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        planner = Planner(CPU_MODEL)
        for kind in PlanKind:
            self.assertEqual(planner.choose(problem, force=kind).selected.kind, kind)

    def test_cannot_force_illegal_view(self):
        problem = ContractionProblem.create(
            "aijd,bckd->abcijk", (2, 2, 2, 3), (2, 2, 2, 3)
        )
        with self.assertRaisesRegex(ValueError, "illegal"):
            Planner().choose(problem, force=PlanKind.GEMM_VIEW)


if __name__ == "__main__":
    unittest.main()

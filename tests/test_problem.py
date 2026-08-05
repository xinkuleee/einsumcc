import unittest

from einsumcc.errors import VerificationError
from einsumcc.problem import ContractionProblem, TensorSpec, contiguous_strides


class ProblemTest(unittest.TestCase):
    def test_course_problem_metadata(self):
        problem = ContractionProblem.create(
            "aijd,bckd->abcijk", (2, 3, 4, 5), (6, 7, 8, 5)
        )
        self.assertEqual(problem.groups.left_free, ("a", "i", "j"))
        self.assertEqual(problem.groups.right_free, ("b", "c", "k"))
        self.assertEqual(problem.groups.reduction, ("d",))
        self.assertEqual((problem.m, problem.n, problem.k), (24, 336, 5))
        self.assertEqual(problem.output.shape, (2, 6, 7, 3, 4, 8))

    def test_rejects_extent_mismatch(self):
        with self.assertRaisesRegex(VerificationError, "inconsistent extents"):
            ContractionProblem.create("mk,kn->mn", (3, 4), (5, 6))

    def test_rejects_rank_mismatch(self):
        with self.assertRaisesRegex(VerificationError, "rank"):
            ContractionProblem.create("mk,kn->mn", (12,), (4, 6))

    def test_rejects_non_contiguous_output(self):
        with self.assertRaisesRegex(VerificationError, "C-contiguous output"):
            ContractionProblem.create(
                "mk,kn->mn", (3, 4), (4, 5), output_strides=(1, 3)
            )

    def test_workload_key_is_stable_and_layout_sensitive(self):
        first = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        same = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        strided = ContractionProblem.create(
            "mk,kn->mn", (3, 4), (4, 5), lhs_strides=(8, 2)
        )
        self.assertEqual(first.workload_key, same.workload_key)
        self.assertNotEqual(first.workload_key, strided.workload_key)

    def test_contiguous_strides(self):
        self.assertEqual(contiguous_strides((2, 3, 4)), (12, 4, 1))
        self.assertTrue(TensorSpec.create((2, 3, 4)).is_c_contiguous)


if __name__ == "__main__":
    unittest.main()


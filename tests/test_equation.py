import unittest

from einsumcc.equation import Equation
from einsumcc.errors import ParseError, VerificationError


class EquationTest(unittest.TestCase):
    def test_classifies_batched_contraction(self):
        equation = Equation.parse(" bij, bjkl -> bikl ")
        self.assertEqual(equation.text, "bij,bjkl->bikl")
        self.assertEqual(
            equation.classify(),
            {
                "batch": ("b",),
                "left_free": ("i",),
                "right_free": ("k", "l"),
                "reduction": ("j",),
            },
        )

    def test_preserves_multiple_reduction_order_from_left(self):
        equation = Equation.parse("aij,jib->ab")
        self.assertEqual(equation.reduction_labels, ("i", "j"))

    def test_allows_scalar_output(self):
        equation = Equation.parse("ij,ij->")
        self.assertEqual(equation.output, ())

    def test_rejects_implicit_output(self):
        with self.assertRaisesRegex(ParseError, "explicit '->'"):
            Equation.parse("ij,jk")

    def test_rejects_more_than_two_operands(self):
        with self.assertRaisesRegex(ParseError, "exactly two"):
            Equation.parse("ij,jk,kl->il")

    def test_rejects_ellipsis(self):
        with self.assertRaisesRegex(ParseError, "ellipsis"):
            Equation.parse("...ij,jk->...ik")

    def test_rejects_diagonal_semantics(self):
        with self.assertRaisesRegex(VerificationError, "diagonal"):
            Equation.parse("ii,ij->j")

    def test_rejects_missing_free_index(self):
        with self.assertRaisesRegex(VerificationError, "free indices"):
            Equation.parse("ij,jk->i")

    def test_rejects_unknown_output_index(self):
        with self.assertRaisesRegex(VerificationError, "absent"):
            Equation.parse("ij,jk->ix")

    def test_rejects_outer_product(self):
        with self.assertRaisesRegex(VerificationError, "reduction"):
            Equation.parse("i,j->ij")


if __name__ == "__main__":
    unittest.main()


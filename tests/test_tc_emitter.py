import unittest

from einsumcc.problem import ContractionProblem
from einsumcc.tc_emitter import TcEmitter


class TcEmitterTest(unittest.TestCase):
    def test_emits_matmul_contract(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        module = TcEmitter().emit(problem)
        self.assertEqual(module.loop_labels, ("m", "n", "k"))
        self.assertIn("tc.contract %lhs, %rhs", module.text)
        self.assertIn('equation = "mk,kn->mn"', module.text)
        self.assertIn(
            'iterator_types = ["parallel", "parallel", "reduction"]',
            module.text,
        )

    def test_emits_scalar_result(self):
        problem = ContractionProblem.create("ij,ij->", (3, 4), (3, 4))
        module = TcEmitter().emit(problem)
        self.assertIn("-> tensor<f32>", module.text)
        self.assertEqual(module.loop_labels, ("i", "j"))


if __name__ == "__main__":
    unittest.main()

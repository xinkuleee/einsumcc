from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from einsumcc.mlir_emitter import MlirEmitter
from einsumcc.problem import ContractionProblem


class MlirEmitterTest(unittest.TestCase):
    def test_matmul_indexing_maps_and_iterators(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        module = MlirEmitter().emit(problem)
        self.assertEqual(module.loop_labels, ("m", "n", "k"))
        self.assertIn(
            "affine_map<(d0, d1, d2)->(d0, d2)>", module.text
        )
        self.assertIn(
            "affine_map<(d0, d1, d2)->(d2, d1)>", module.text
        )
        self.assertIn(
            'iterator_types = ["parallel", "parallel", "reduction"]',
            module.text,
        )
        self.assertIn("%zero = arith.constant 0.0 : f32", module.text)
        self.assertIn("%init = linalg.fill", module.text)
        self.assertNotIn("%init: tensor", module.text)

    def test_course_contraction_has_six_parallel_loops(self):
        problem = ContractionProblem.create(
            "aijd,bckd->abcijk", (2, 2, 2, 3), (2, 2, 2, 3)
        )
        module = MlirEmitter().emit(problem)
        self.assertEqual(module.loop_labels, ("a", "b", "c", "i", "j", "k", "d"))
        self.assertEqual(module.text.count('"parallel"'), 6)
        self.assertEqual(module.text.count('"reduction"'), 1)

    @unittest.skipUnless(shutil.which("mlir-opt"), "mlir-opt is not installed")
    def test_mlir_opt_parses_emitted_module_when_available(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "module.mlir"
            source.write_text(MlirEmitter().emit(problem).text, encoding="utf-8")
            subprocess.run(
                ["mlir-opt", str(source), "-o", "/dev/null"],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from einsumcc.cli import main


class CliTest(unittest.TestCase):
    def invoke(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(arguments)
        return status, output.getvalue()

    def test_explain_course_contraction(self):
        status, output = self.invoke(
            [
                "explain",
                "aijd,bckd->abcijk",
                "--lhs-shape",
                "2,2,2,3",
                "--rhs-shape",
                "2,2,2,3",
            ]
        )
        self.assertEqual(status, 0)
        self.assertIn("M (left free): aij", output)
        self.assertIn("gemm-view: illegal", output)
        self.assertIn("Selected:", output)

    def test_verify_all_legal_plans(self):
        status, output = self.invoke(
            [
                "verify",
                "mk,kn->mn",
                "--lhs-shape",
                "3,4",
                "--rhs-shape",
                "4,5",
                "--seed",
                "9",
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(output.count("PASS"), 3)

    def test_emit_mlir_to_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "contract.mlir"
            status, _ = self.invoke(
                [
                    "emit-mlir",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "3,4",
                    "--rhs-shape",
                    "4,5",
                    "-o",
                    str(destination),
                ]
            )
            self.assertEqual(status, 0)
            self.assertIn("linalg.generic", destination.read_text(encoding="utf-8"))

    def test_tune_then_explain_uses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = str(Path(directory) / "tuning.json")
            status, tune_output = self.invoke(
                [
                    "tune",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "2,3",
                    "--rhs-shape",
                    "3,2",
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "--max-direct-schedules",
                    "1",
                    "--cache",
                    cache,
                ]
            )
            self.assertEqual(status, 0)
            self.assertIn("Measured candidates: 3", tune_output)
            status, explain_output = self.invoke(
                [
                    "explain",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "2,3",
                    "--rhs-shape",
                    "3,2",
                    "--cache",
                    cache,
                ]
            )
            self.assertEqual(status, 0)
            self.assertIn("Decision source: tuning cache", explain_output)


if __name__ == "__main__":
    unittest.main()


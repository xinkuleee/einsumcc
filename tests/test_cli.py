import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

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
            self.assertIn("tc.contract", destination.read_text(encoding="utf-8"))

    def test_emit_lowered_linalg_stage(self):
        compiler = mock.Mock()
        compiler.lower.return_value = "module { linalg.generic }\n"
        with mock.patch("einsumcc.cli.NativeCpuCompiler", return_value=compiler):
            status, output = self.invoke(
                [
                    "emit-mlir",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "3,4",
                    "--rhs-shape",
                    "4,5",
                    "--stage",
                    "linalg",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("linalg.generic", output)
        compiler.lower.assert_called_once()

    def test_emit_tc_rejects_schedule_options_that_cannot_affect_it(self):
        with self.assertRaises(SystemExit) as raised:
            self.invoke(
                [
                    "emit-mlir",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "3,4",
                    "--rhs-shape",
                    "4,5",
                    "--block-m",
                    "2",
                    "--block-n",
                    "3",
                    "--block-k",
                    "4",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_run_native_reports_artifact_and_cache_status(self):
        class Kernel:
            library_path = Path("/tmp/einsumcc-test/module.dylib")
            cache_hit = True
            from einsumcc.schedule import DirectSchedule
            schedule = DirectSchedule(3, 5, 4, 1, 1)
            expanded_tile_sizes = (3, 5, 4)

            def run(self, lhs, rhs):
                return np.einsum("mk,kn->mn", lhs, rhs, dtype=np.float32)

        compiler = mock.Mock()
        compiler.compile_native_direct.return_value = Kernel()
        with mock.patch("einsumcc.cli.Compiler", return_value=compiler):
            status, output = self.invoke(
                [
                    "run-native",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "3,4",
                    "--rhs-shape",
                    "4,5",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("PASS native-direct", output)
        self.assertIn("Cache: hit", output)
        compiler.compile_native_direct.assert_called_once()

    def test_run_native_accepts_explicit_schedule(self):
        class Kernel:
            library_path = Path("/tmp/einsumcc-test/module.dylib")
            cache_hit = False
            schedule = None
            expanded_tile_sizes = (2, 3, 4)

            def run(self, lhs, rhs):
                return np.einsum("mk,kn->mn", lhs, rhs, dtype=np.float32)

        compiler = mock.Mock()
        kernel = Kernel()
        from einsumcc.schedule import DirectSchedule

        kernel.schedule = DirectSchedule(2, 3, 4, 1, 1)
        compiler.compile_native_direct.return_value = kernel
        with mock.patch("einsumcc.cli.Compiler", return_value=compiler):
            status, output = self.invoke(
                [
                    "run-native",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "3,4",
                    "--rhs-shape",
                    "4,5",
                    "--block-m",
                    "2",
                    "--block-n",
                    "3",
                    "--block-k",
                    "4",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("Expanded tiles: (2, 3, 4)", output)
        supplied = compiler.compile_native_direct.call_args.kwargs["schedule"]
        self.assertEqual(supplied, kernel.schedule)

    def test_tune_native_writes_machine_readable_report(self):
        from einsumcc.schedule import DirectSchedule
        from einsumcc.tuner import NativeTuningMeasurement, NativeTuningResult

        measurement = NativeTuningMeasurement(
            DirectSchedule(2, 3, 4, 1, 1),
            (2, 3, 4),
            (2.0, 3.0),
            0.0,
            "artifact",
            False,
        )
        compiler = mock.Mock()
        compiler.tune_native_direct.return_value = NativeTuningResult(
            "native-cpu-mini-v0.1-test",
            {"cpu_model": "test-cpu"},
            measurement,
            measurement,
            (measurement,),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "einsumcc.cli.Compiler", return_value=compiler
        ):
            destination = Path(directory) / "native.json"
            status, _ = self.invoke(
                [
                    "tune-native",
                    "mk,kn->mn",
                    "--lhs-shape",
                    "3,4",
                    "--rhs-shape",
                    "4,5",
                    "--warmups",
                    "0",
                    "--repeats",
                    "2",
                    "--max-schedules",
                    "1",
                    "-o",
                    str(destination),
                    "--cache",
                    str(Path(directory) / "tuning.json"),
                ]
            )
            payload = __import__("json").loads(destination.read_text())
        self.assertEqual(status, 0)
        self.assertEqual(payload["kind"], "native-direct-tuning")
        self.assertEqual(
            payload["baseline_definition"],
            "first statically ranked distinct schedule",
        )
        self.assertEqual(payload["environment"]["cpu_model"], "test-cpu")
        self.assertEqual(payload["best"]["expanded_tile_sizes"], [2, 3, 4])

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

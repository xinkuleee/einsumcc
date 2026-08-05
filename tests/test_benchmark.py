import json
from pathlib import Path
import tempfile
import unittest

from einsumcc.benchmark import BenchmarkCorpus, CpuBenchmarkRunner
from einsumcc.cli import main


class BenchmarkTest(unittest.TestCase):
    def test_versioned_corpus_runs_all_plans(self):
        corpus = BenchmarkCorpus.load(Path("benchmarks/cpu-smoke.json"))
        report = CpuBenchmarkRunner(warmups=0, repeats=1, seed=4).run(corpus)
        self.assertEqual(report["schema"], 1)
        self.assertEqual(report["target"], "cpu-model")
        self.assertEqual(len(report["cases"]), 5)
        course = next(case for case in report["cases"] if case["name"] == "course-layout")
        view = next(plan for plan in course["plans"] if plan["kind"] == "gemm-view")
        self.assertFalse(view["legal"])
        for case in report["cases"]:
            for plan in case["plans"]:
                if plan["legal"]:
                    self.assertGreaterEqual(plan["median_us"], 0.0)
                    self.assertLessEqual(plan["max_abs_error"], 1.0e-4)

    def test_cli_writes_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            status = main(
                [
                    "benchmark",
                    "benchmarks/cpu-smoke.json",
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "-o",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["corpus"], "cpu-smoke")

    def test_rejects_duplicate_case_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            case = {
                "name": "same",
                "equation": "mk,kn->mn",
                "lhs_shape": [2, 3],
                "rhs_shape": [3, 2],
            }
            path.write_text(
                json.dumps({"schema": 1, "cases": [case, case]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                BenchmarkCorpus.load(path)


if __name__ == "__main__":
    unittest.main()


import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from einsumcc.benchmark import (
    BenchmarkCorpus,
    CpuBenchmarkRunner,
    NativeBenchmarkRunner,
)
from einsumcc.cache import TuningCache
from einsumcc.cli import main
from einsumcc.schedule import DirectSchedule
from einsumcc.tuner import NativeTuningMeasurement, NativeTuningResult


class BenchmarkTest(unittest.TestCase):
    @staticmethod
    def native_result(problem):
        schedule = DirectSchedule(2, 3, 4, 1, 1)
        measurement = NativeTuningMeasurement(
            schedule,
            schedule.expanded_tile_sizes(problem),
            (2.0, 3.0),
            0.0,
            "artifact-{}".format(problem.workload_key),
            False,
        )
        return NativeTuningResult(
            "native-cpu-mini-v0.1-test",
            {"cpu_model": "test-cpu", "timing_scope": "bound call"},
            measurement,
            measurement,
            (measurement,),
        )

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

    def test_native_corpus_runner_records_cases_and_cache(self):
        corpus = BenchmarkCorpus.load(Path("benchmarks/native-mini-v0.1.json"))
        native = mock.Mock()
        native.cache_dir = Path("/tmp/einsumcc-native-test")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "einsumcc.benchmark.NativeDirectTuner"
        ) as tuner_type:
            tuner_type.return_value.tune.side_effect = (
                lambda problem, lhs, rhs: self.native_result(problem)
            )
            cache = TuningCache(Path(directory) / "tuning.json")
            report = NativeBenchmarkRunner(
                warmups=0,
                repeats=2,
                max_schedules=1,
                seed=7,
                tuning_cache=cache,
                native_compiler=native,
            ).run(corpus)
            records = cache.records()

        self.assertEqual(report["kind"], "native-direct-corpus")
        self.assertEqual(report["target"], "native-cpu-mini-v0.1-test")
        self.assertEqual(len(report["cases"]), 5)
        self.assertEqual(tuner_type.call_count, 5)
        self.assertEqual(len(records), 5)
        first = report["cases"][0]
        self.assertGreater(first["best_gflops"], 0.0)
        self.assertEqual(first["speedup_vs_baseline"], 1.0)
        self.assertEqual(
            report["configuration"]["baseline_definition"],
            "first statically ranked distinct schedule",
        )

    def test_native_benchmark_cli_writes_json_report(self):
        fake_report = {
            "schema": 1,
            "kind": "native-direct-corpus",
            "target": "native-cpu-mini-v0.1-test",
            "cases": [],
        }
        runner = mock.Mock()
        runner.run.return_value = fake_report
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "einsumcc.cli.NativeBenchmarkRunner", return_value=runner
        ):
            output = Path(directory) / "native-results.json"
            status = main(
                [
                    "benchmark-native",
                    "benchmarks/native-mini-v0.1.json",
                    "--warmups",
                    "0",
                    "--repeats",
                    "2",
                    "--max-schedules",
                    "3",
                    "--cache",
                    str(Path(directory) / "tuning.json"),
                    "-o",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertEqual(payload["kind"], "native-direct-corpus")
        runner.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()

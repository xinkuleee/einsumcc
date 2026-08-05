import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from einsumcc.cache import TuningCache, TuningRecord
from einsumcc.compiler import Compiler
from einsumcc.errors import CacheError
from einsumcc.plans import PlanKind
from einsumcc.problem import ContractionProblem
from einsumcc.schedule import DirectSchedule, ScheduleSpace
from einsumcc.target import A100_MODEL, CPU_MODEL


class ScheduleAndCacheTest(unittest.TestCase):
    def test_schedule_values_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            DirectSchedule(-1, 8, 4, 1, 1)

    def test_schedule_space_prunes_vector_misalignment(self):
        problem = ContractionProblem.create("mk,kn->mn", (8, 7), (7, 8))
        space = ScheduleSpace(
            block_m=(8,), block_n=(8,), block_k=(4,), gpu_threads=(128,), vector_widths=(4,)
        )
        candidate = space.candidates(problem, A100_MODEL)[0]
        self.assertFalse(candidate.legal)
        self.assertTrue(any("vector width" in reason for reason in candidate.reasons))

    def test_cache_round_trip_and_compiler_override(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        schedule = DirectSchedule(8, 8, 4, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            cache = TuningCache(Path(directory) / "tuning.json")
            record = TuningRecord.create(
                problem.workload_key, CPU_MODEL.name, PlanKind.DIRECT, 12.5, (12.0, 13.0), schedule
            )
            cache.store(record)
            loaded = cache.lookup(problem.workload_key, CPU_MODEL.name)
            self.assertEqual(loaded, record)
            compiled = Compiler(CPU_MODEL, cache=cache).compile(problem)
            self.assertEqual(compiled.decision.selected.kind, PlanKind.DIRECT)
            self.assertTrue(compiled.decision.measured)
            self.assertEqual(compiled.schedule, schedule)

    def test_target_is_part_of_cache_identity(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        with tempfile.TemporaryDirectory() as directory:
            cache = TuningCache(Path(directory) / "tuning.json")
            cache.store(
                TuningRecord.create(
                    problem.workload_key, CPU_MODEL.name, PlanKind.PACKED_GEMM, 1.0, (1.0,)
                )
            )
            self.assertIsNone(cache.lookup(problem.workload_key, A100_MODEL.name))

    def test_cache_rejects_mismatched_record_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tuning.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "records": {
                            "wrong-key": {
                                "workload_key": "abc",
                                "target": CPU_MODEL.name,
                                "plan_kind": PlanKind.PACKED_GEMM.value,
                                "median_us": 1.0,
                                "samples_us": [1.0],
                                "schedule": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CacheError, "does not match"):
                TuningCache(path).records()

    def test_cache_rejects_non_finite_measurement(self):
        with self.assertRaisesRegex(CacheError, "finite"):
            TuningRecord.create(
                "abc", CPU_MODEL.name, PlanKind.DIRECT, float("nan"), (1.0,)
            )

    def test_cache_rejects_non_object_json_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tuning.json"
            for payload in (None, [], {"schema": 1, "records": []}):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(CacheError):
                    TuningCache(path).records()

    def test_compiler_falls_back_when_direct_has_no_legal_schedule(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        impossible = ScheduleSpace(block_m=(), block_n=(), block_k=())
        compiled = Compiler(CPU_MODEL, schedule_space=impossible).compile(problem)
        self.assertNotEqual(compiled.decision.selected.kind, PlanKind.DIRECT)
        self.assertIsNone(compiled.schedule)
        with self.assertRaisesRegex(ValueError, "no legal schedule"):
            Compiler(CPU_MODEL, schedule_space=impossible).compile(
                problem, force=PlanKind.DIRECT
            )

    def test_compiler_ignores_schedule_attached_to_gemm_record(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        with tempfile.TemporaryDirectory() as directory:
            cache = TuningCache(Path(directory) / "tuning.json")
            cache.store(
                TuningRecord.create(
                    problem.workload_key,
                    CPU_MODEL.name,
                    PlanKind.GEMM_VIEW,
                    0.1,
                    (0.1,),
                    DirectSchedule(8, 8, 4, 1, 1),
                )
            )
            compiled = Compiler(CPU_MODEL, cache=cache).compile(problem)
            self.assertIsNone(compiled.cache_record)

    def test_cpu_tuner_writes_reusable_record(self):
        problem = ContractionProblem.create("mk,kn->mn", (2, 3), (3, 2))
        lhs = np.arange(6, dtype=np.float32).reshape(2, 3)
        rhs = np.arange(6, dtype=np.float32).reshape(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            cache = TuningCache(Path(directory) / "tuning.json")
            compiler = Compiler(CPU_MODEL, cache=cache)
            result = compiler.tune_cpu(
                problem, lhs, rhs, warmups=0, repeats=1, max_direct_schedules=1
            )
            self.assertGreaterEqual(len(result.measurements), 3)
            self.assertIsNotNone(cache.lookup(problem.workload_key, CPU_MODEL.name))


if __name__ == "__main__":
    unittest.main()

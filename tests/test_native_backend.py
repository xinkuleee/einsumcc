import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from einsumcc.errors import BackendError
from einsumcc.compiler import Compiler
from einsumcc.native_backend import NativeCpuCompiler, NativeToolchain
from einsumcc.problem import ContractionProblem
from einsumcc.schedule import DirectSchedule, ScheduleSpace
from einsumcc.tuner import NATIVE_CPU_TUNING_TARGET, NativeDirectTuner


ROOT = Path(__file__).resolve().parents[1]
NATIVE_AVAILABLE = NativeToolchain.available(ROOT)


def strided_random(shape, strides, seed):
    rng = np.random.default_rng(seed)
    size = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    storage = rng.standard_normal(size).astype(np.float32)
    return np.lib.stride_tricks.as_strided(
        storage,
        shape=shape,
        strides=tuple(stride * storage.itemsize for stride in strides),
    )


@unittest.skipUnless(NATIVE_AVAILABLE, "native MLIR toolchain has not been built")
class NativeBackendTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.compiler = NativeCpuCompiler(
            toolchain=NativeToolchain.discover(ROOT),
            cache_dir=Path(self.temporary.name),
        )

    def test_compiles_executes_and_reuses_content_addressed_artifact(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        lhs = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs = np.arange(20, dtype=np.float32).reshape(4, 5)

        first = self.compiler.compile(problem)
        self.assertFalse(first.cache_hit)
        np.testing.assert_allclose(
            first.run(lhs, rhs), np.einsum(problem.equation.text, lhs, rhs)
        )

        second = self.compiler.compile(problem)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.cache_key, second.cache_key)
        manifest = json.loads(
            (second.library_path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["key"], second.cache_key)
        self.assertEqual(manifest["problem"]["equation"], "mk,kn->mn")
        self.assertEqual(manifest["schedule"], dict(second.schedule.as_dict()))
        self.assertEqual(
            manifest["expanded_tile_sizes"], list(second.expanded_tile_sizes)
        )

    def test_schedule_changes_ir_artifact_identity_and_manifest(self):
        problem = ContractionProblem.create("mk,kn->mn", (7, 9), (9, 5))
        first_schedule = DirectSchedule(2, 3, 4, 1, 1)
        second_schedule = DirectSchedule(4, 5, 8, 1, 1)
        first = self.compiler.compile(problem, first_schedule)
        second = self.compiler.compile(problem, second_schedule)
        self.assertNotEqual(first.cache_key, second.cache_key)
        self.assertEqual(first.expanded_tile_sizes, (2, 3, 4))
        self.assertEqual(second.expanded_tile_sizes, (4, 5, 8))

        scheduled = self.compiler.lower(problem, "scheduled", first_schedule)
        self.assertIn("scf.for", scheduled)
        self.assertIn("step %c2", scheduled)
        self.assertIn("step %c3", scheduled)
        self.assertIn("step %c4", scheduled)

    def test_native_backend_rejects_unimplemented_schedule_fields(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        with self.assertRaisesRegex(BackendError, "threads=1"):
            self.compiler.compile(problem, DirectSchedule(2, 2, 2, 2, 1))
        with self.assertRaisesRegex(BackendError, "vector_width=1"):
            self.compiler.compile(problem, DirectSchedule(2, 2, 2, 1, 2))

    def test_bound_invocation_is_reusable(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        lhs = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs = np.arange(20, dtype=np.float32).reshape(4, 5)
        output = np.empty((3, 5), dtype=np.float32)
        invocation = self.compiler.compile(problem).bind(lhs, rhs, output)
        self.assertIs(invocation.run(), output)
        output.fill(np.nan)
        self.assertIs(invocation.run(), output)
        np.testing.assert_allclose(output, np.einsum("mk,kn->mn", lhs, rhs))

    def test_native_tuner_compiles_deduplicates_and_measures_dylibs(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 3))
        lhs = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs = np.arange(12, dtype=np.float32).reshape(4, 3)
        # Several raw choices saturate to the same loop tiles on this small
        # problem; only distinct generated loop structures are measured.
        space = ScheduleSpace(
            block_m=(2, 4, 8), block_n=(2, 4), block_k=(2, 4, 8)
        )
        result = NativeDirectTuner(
            self.compiler, space, warmups=0, repeats=2, max_schedules=8
        ).tune(problem, lhs, rhs)
        expanded = [item.expanded_tile_sizes for item in result.measurements]
        self.assertEqual(len(expanded), len(set(expanded)))
        self.assertGreater(len(expanded), 1)
        self.assertIn(result.best, result.measurements)
        self.assertEqual(len(result.best.samples_us), 2)
        self.assertLessEqual(result.best.max_abs_error, 1.0e-4)
        self.assertIsNotNone(result.speedup_vs_baseline)
        self.assertGreater(result.speedup_vs_baseline or 0.0, 0.0)
        record = result.to_record(problem)
        self.assertTrue(record.target.startswith(NATIVE_CPU_TUNING_TARGET))
        self.assertEqual(record.schedule, result.best.schedule)

    def test_invalid_cache_entry_is_rebuilt(self):
        problem = ContractionProblem.create("mk,kn->mn", (2, 3), (3, 2))
        first = self.compiler.compile(problem)
        first.library_path.write_bytes(b"corrupt")

        rebuilt = self.compiler.compile(problem)
        self.assertFalse(rebuilt.cache_hit)
        lhs = np.arange(6, dtype=np.float32).reshape(2, 3)
        rhs = np.arange(6, dtype=np.float32).reshape(3, 2)
        np.testing.assert_allclose(
            rebuilt.run(lhs, rhs), np.einsum("mk,kn->mn", lhs, rhs)
        )

    def test_dangling_symlink_cache_entry_is_rebuilt(self):
        problem = ContractionProblem.create("mk,kn->mn", (2, 3), (3, 2))
        first = self.compiler.compile(problem)
        entry = first.library_path.parent
        shutil.rmtree(entry)
        entry.symlink_to(entry.with_name("missing-entry"), target_is_directory=True)

        rebuilt = self.compiler.compile(problem)
        self.assertFalse(rebuilt.cache_hit)
        self.assertTrue(rebuilt.library_path.is_file())

    def test_layout_is_part_of_native_artifact_identity(self):
        contiguous = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        strided = ContractionProblem.create(
            "mk,kn->mn", (3, 4), (4, 5), lhs_strides=(12, 2)
        )
        contiguous_kernel = self.compiler.compile(contiguous)
        strided_kernel = self.compiler.compile(strided)
        self.assertNotEqual(contiguous_kernel.cache_key, strided_kernel.cache_key)

    def test_positive_stride_inputs_cross_ranked_memref_abi(self):
        problem = ContractionProblem.create(
            "mk,kn->mn",
            (3, 4),
            (4, 5),
            lhs_strides=(12, 2),
            rhs_strides=(10, 1),
        )
        lhs = strided_random(problem.lhs.shape, problem.lhs.strides, 11)
        rhs = strided_random(problem.rhs.shape, problem.rhs.strides, 12)
        result = self.compiler.compile(problem).run(lhs, rhs)
        np.testing.assert_allclose(
            result,
            np.einsum(problem.equation.text, lhs, rhs, dtype=np.float32),
            rtol=1.0e-4,
            atol=1.0e-5,
        )

    def test_scalar_output_and_run_into(self):
        problem = ContractionProblem.create("ij,ij->", (3, 4), (3, 4))
        lhs = np.arange(12, dtype=np.float32).reshape(3, 4)
        rhs = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
        output = np.empty((), dtype=np.float32)
        returned = self.compiler.compile(problem).run_into(lhs, rhs, output)
        self.assertIs(returned, output)
        np.testing.assert_allclose(
            output, np.einsum(problem.equation.text, lhs, rhs, dtype=np.float32)
        )

    def test_rejects_arrays_that_do_not_match_static_abi(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        kernel = self.compiler.compile(problem)
        lhs = np.zeros((3, 4), dtype=np.float32)
        rhs = np.zeros((4, 5), dtype=np.float32)
        output = np.empty((3, 5), dtype=np.float32)

        with self.assertRaisesRegex(BackendError, "dtype float32"):
            kernel.run(lhs.astype(np.float64), rhs)
        with self.assertRaisesRegex(BackendError, "shape"):
            kernel.run(lhs[:, :3], rhs)
        with self.assertRaisesRegex(BackendError, "element strides"):
            kernel.run(lhs[:, ::-1], rhs)
        output.flags.writeable = False
        with self.assertRaisesRegex(BackendError, "output must be writable"):
            kernel.run_into(lhs, rhs, output)

    def test_rejects_non_element_byte_stride(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 4), (4, 5))
        kernel = self.compiler.compile(problem)
        storage = bytearray(64)
        lhs = np.ndarray(
            (3, 4), dtype=np.float32, buffer=storage, strides=(13, 4)
        )
        rhs = np.zeros((4, 5), dtype=np.float32)
        with self.assertRaisesRegex(BackendError, "byte strides"):
            kernel.run(lhs, rhs)

    def test_run_into_rejects_output_input_alias(self):
        problem = ContractionProblem.create("mk,kn->mn", (3, 3), (3, 3))
        kernel = self.compiler.compile(problem)
        lhs = np.arange(9, dtype=np.float32).reshape(3, 3)
        rhs = np.eye(3, dtype=np.float32)
        with self.assertRaisesRegex(BackendError, "must not overlap"):
            kernel.run_into(lhs, rhs, lhs)

    def test_exposes_real_linalg_and_llvm_stages(self):
        problem = ContractionProblem.create("mk,kn->mn", (2, 3), (3, 2))
        self.assertIn("linalg.generic", self.compiler.lower(problem, "linalg"))
        self.assertIn("llvm.func @contract", self.compiler.lower(problem, "llvm"))
        self.assertIn("define void @contract", self.compiler.lower(problem, "llvm-ir"))

    def test_compiler_facade_exposes_native_direct_without_hiding_scope(self):
        problem = ContractionProblem.create("mk,kn->mn", (2, 3), (3, 2))
        kernel = Compiler().compile_native_direct(
            problem,
            cache_dir=Path(self.temporary.name) / "facade",
            toolchain=self.compiler.toolchain,
        )
        lhs = np.arange(6, dtype=np.float32).reshape(2, 3)
        rhs = np.arange(6, dtype=np.float32).reshape(3, 2)
        np.testing.assert_allclose(
            kernel.run(lhs, rhs), np.einsum("mk,kn->mn", lhs, rhs)
        )

    def test_concurrent_processes_populate_one_valid_cache_entry(self):
        cache = Path(self.temporary.name) / "concurrent"
        source = "; ".join(
            (
                "import sys",
                "from pathlib import Path",
                "from einsumcc.native_backend import NativeCpuCompiler, NativeToolchain",
                "from einsumcc.problem import ContractionProblem",
                "root = Path(sys.argv[1])",
                "cache = Path(sys.argv[2])",
                "toolchain = NativeToolchain.discover(root)",
                "problem = ContractionProblem.create('mk,kn->mn', (7, 9), (9, 5))",
                "kernel = NativeCpuCompiler(toolchain=toolchain, cache_dir=cache).compile(problem)",
                "print(int(kernel.cache_hit))",
            )
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        arguments = (
            sys.executable,
            "-B",
            "-c",
            source,
            str(ROOT),
            str(cache),
        )
        processes = [
            subprocess.Popen(
                arguments,
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        results = [process.communicate(timeout=30) for process in processes]
        for process, (_, error) in zip(processes, results):
            self.assertEqual(process.returncode, 0, error)
        self.assertEqual(sorted(output.strip() for output, _ in results), ["0", "1"])
        entries = [
            path
            for path in cache.iterdir()
            if path.is_dir() and ".tmp." not in path.name and ".stale." not in path.name
        ]
        self.assertEqual(len(entries), 1)
        self.assertTrue((entries[0] / "manifest.json").is_file())
        self.assertTrue(
            (entries[0] / ("module" + self.compiler.shared_library_suffix)).is_file()
        )


if __name__ == "__main__":
    unittest.main()

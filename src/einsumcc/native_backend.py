"""Schedule-driven native CPU compilation for the Mini v0.1 Direct path.

This module is intentionally a small AOT runtime, not a general JIT.  It joins
the Python Einstein frontend to the project-owned MLIR driver, translates the
fully lowered LLVM dialect to LLVM IR, builds a local shared library, and calls
the generated C wrapper with ranked-memref descriptors.

Compiled artifacts are content addressed by semantic IR, pipeline revision,
and toolchain identity.  A stable sidecar lock makes cache population safe
across concurrent processes; the manifest is written before the temporary
directory is atomically installed as the final cache entry.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows is not a Nano v1 native target.
    fcntl = None  # type: ignore[assignment]
from functools import lru_cache
from typing import Mapping, Optional, Sequence, Tuple, Type

import numpy as np

from .errors import BackendError
from .problem import ContractionProblem
from .schedule import DirectSchedule, ScheduleSpace
from .target import CPU_MODEL
from .tc_emitter import TcEmitter


NATIVE_CACHE_SCHEMA = 2
NATIVE_PIPELINE_REVISION = 2
FUNCTION_NAME = "contract"

# The output-parameter conversion gives execution a simple ownership model:
# Python allocates all three buffers and generated code never returns or owns a
# heap allocation.  Input function-boundary layouts remain dynamic strided
# memrefs, preserving Nano v1's positive-stride input semantics.
NATIVE_PIPELINE_PREFIX: Tuple[str, ...] = (
    "--tc-contract-to-linalg",
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--buffer-results-to-out-params=hoist-static-allocs modify-public-functions",
)
NATIVE_PIPELINE_SUFFIX: Tuple[str, ...] = (
    "--convert-linalg-to-loops",
    # Tiling introduces affine.min for partial boundary tiles. Lower it before
    # the SCF/LLVM conversions so mlir-translate receives pure LLVM dialect.
    "--lower-affine",
    # linalg tiling also creates memref.subview. Expose its offset/size/stride
    # arithmetic before finalizing all memref operations to LLVM.
    "--expand-strided-metadata",
    # Expanding strided metadata may materialize affine.apply address
    # expressions, so run affine lowering once more after the expansion.
    "--lower-affine",
    "--convert-scf-to-cf",
    "--convert-arith-to-llvm",
    "--convert-index-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
)


def _native_schedule(
    problem: ContractionProblem, schedule: Optional[DirectSchedule]
) -> DirectSchedule:
    """Return a schedule whose every field is implemented by this backend."""

    selected = schedule or ScheduleSpace().default(problem, CPU_MODEL)
    if selected.threads != 1:
        raise BackendError(
            "Mini v0.1 native CPU codegen supports only threads=1"
        )
    if selected.vector_width != 1:
        raise BackendError(
            "Mini v0.1 native CPU codegen supports only vector_width=1; "
            "native vector lowering is not implemented"
        )
    return selected


def _schedule_pass_argument(
    problem: ContractionProblem, schedule: DirectSchedule
) -> str:
    sizes = ",".join(str(value) for value in schedule.expanded_tile_sizes(problem))
    return "--einsumcc-schedule-direct=tile-sizes={}".format(sizes)


def _native_pipeline(
    problem: ContractionProblem, schedule: DirectSchedule
) -> Tuple[str, ...]:
    return (
        *NATIVE_PIPELINE_PREFIX,
        _schedule_pass_argument(problem, schedule),
        *NATIVE_PIPELINE_SUFFIX,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_path(path: Path) -> None:
    """Remove one cache object without following a directory symlink."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _repository_root() -> Path:
    """Locate an editable/source checkout without depending on cwd."""

    override = os.environ.get("EINSUMCC_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / "CMakeLists.txt").is_file() and (
            parent / "src" / "einsumcc"
        ).is_dir():
            return parent
    return Path.cwd().resolve()


def _resolve_executable(
    environment_name: str, candidates: Sequence[Optional[Path]], display_name: str
) -> Path:
    override = os.environ.get(environment_name)
    search = []
    if override:
        search.append(Path(override).expanduser())
    search.extend(candidate for candidate in candidates if candidate is not None)
    located = shutil.which(display_name)
    if located:
        search.append(Path(located))
    for candidate in search:
        path = candidate.expanduser().resolve()
        if path.is_file() and os.access(str(path), os.X_OK):
            return path
    raise BackendError(
        "cannot find executable '{}'; build the MLIR tools with "
        "'make build-mlir' or set {}".format(display_name, environment_name)
    )


@lru_cache(maxsize=1)
def _host_cpu_model() -> str:
    """Best-effort stable CPU model for native tuning/cache identity."""

    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                completed = subprocess.run(
                    ("/usr/sbin/sysctl", "-n", key),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            value = completed.stdout.strip()
            if completed.returncode == 0 and value:
                return value
        try:
            completed = subprocess.run(
                ("/usr/sbin/system_profiler", "SPHardwareDataType", "-detailLevel", "mini"),
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            fields = {}
            for line in completed.stdout.splitlines():
                if ":" not in line:
                    continue
                name, value = (part.strip() for part in line.split(":", 1))
                if name in ("Chip", "Model Identifier") and value:
                    fields[name] = value
            if fields:
                return "{} ({})".format(
                    fields.get("Chip", "Apple Silicon"),
                    fields.get("Model Identifier", "unknown model"),
                )
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    _, value = line.split(":", 1)
                    if value.strip():
                        return value.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


@dataclass(frozen=True)
class NativeToolchain:
    """Resolved executables used by the native CPU pipeline."""

    optimizer: Path
    translator: Path
    clang: Path
    repository_root: Path

    @classmethod
    def discover(cls, repository_root: Optional[Path] = None) -> "NativeToolchain":
        root = Path(repository_root or _repository_root()).resolve()
        optimizer = _resolve_executable(
            "EINSUMCC_OPT",
            (root / "build" / "tools" / "einsumcc-opt" / "einsumcc-opt",),
            "einsumcc-opt",
        )
        translator = _resolve_executable(
            "EINSUMCC_MLIR_TRANSLATE",
            (root / ".deps" / "llvm-macos-arm64" / "bin" / "mlir-translate",),
            "mlir-translate",
        )
        # Match the LLVM IR producer. Newer MLIR/LLVM may emit syntax (for
        # example memory effects on attribute groups) that an older Apple
        # clang cannot parse. Fall back to the system compiler only when the
        # pinned toolchain has no clang binary.
        clang_candidates = [
            root / ".deps" / "llvm-macos-arm64" / "bin" / "clang-23"
        ]
        if sys.platform == "darwin":
            clang_candidates.append(Path("/usr/bin/clang"))
        clang = _resolve_executable(
            "EINSUMCC_CLANG", tuple(clang_candidates), "clang"
        )
        return cls(optimizer, translator, clang, root)

    @classmethod
    def available(cls, repository_root: Optional[Path] = None) -> bool:
        try:
            cls.discover(repository_root)
        except BackendError:
            return False
        return True

    def identity(self) -> Mapping[str, object]:
        def executable_identity(path: Path) -> Mapping[str, object]:
            metadata = path.stat()
            return {
                "path": str(path),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }

        return {
            "optimizer": executable_identity(self.optimizer),
            "translator": executable_identity(self.translator),
            "clang": executable_identity(self.clang),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": _host_cpu_model(),
        }


def _run(
    arguments: Sequence[str],
    *,
    input_text: Optional[str] = None,
    timeout_seconds: int = 180,
) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BackendError(
            "cannot execute native compiler stage '{}': {}".format(
                Path(arguments[0]).name, error
            )
        ) from error
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()
        if len(diagnostic) > 4000:
            diagnostic = diagnostic[-4000:]
        raise BackendError(
            "native compiler stage '{}' failed with status {}: {}".format(
                Path(arguments[0]).name, completed.returncode, diagnostic
            )
        )
    return completed.stdout


@lru_cache(maxsize=None)
def _descriptor_type(rank: int) -> Type[ctypes.Structure]:
    if rank < 0:
        raise ValueError("memref rank cannot be negative")
    fields = [
        ("base_ptr", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]
    if rank:
        fields.extend(
            (
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            )
        )

    class RankedMemRefDescriptor(ctypes.Structure):
        _fields_ = fields

    RankedMemRefDescriptor.__name__ = "RankedMemRefDescriptor{}D".format(rank)
    return RankedMemRefDescriptor


def _make_descriptor(array: np.ndarray) -> ctypes.Structure:
    rank = array.ndim
    descriptor = _descriptor_type(rank)()
    address = int(array.ctypes.data)
    descriptor.base_ptr = address
    descriptor.data = address
    descriptor.offset = 0
    if rank:
        descriptor.sizes = (ctypes.c_int64 * rank)(*array.shape)
        descriptor.strides = (ctypes.c_int64 * rank)(
            *(stride // array.itemsize for stride in array.strides)
        )
    return descriptor


def _validate_array(
    role: str, array: np.ndarray, shape: Sequence[int], strides: Sequence[int]
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise BackendError("{} must be a NumPy ndarray".format(role))
    value = array
    if value.dtype != np.float32:
        raise BackendError("{} must have dtype float32".format(role))
    if int(value.ctypes.data) % value.itemsize != 0:
        raise BackendError("{} data pointer must be float32-aligned".format(role))
    if any(stride % value.itemsize != 0 for stride in value.strides):
        raise BackendError(
            "{} byte strides must be multiples of the float32 item size".format(
                role
            )
        )
    if value.shape != tuple(shape):
        raise BackendError(
            "{} shape {} does not match compiled shape {}".format(
                role, value.shape, tuple(shape)
            )
        )
    actual_strides = tuple(stride // value.itemsize for stride in value.strides)
    if actual_strides != tuple(strides):
        raise BackendError(
            "{} element strides {} do not match compiled strides {}".format(
                role, actual_strides, tuple(strides)
            )
        )
    return value


class NativeKernel:
    """Loaded native Direct kernel for one static contraction problem."""

    def __init__(
        self,
        problem: ContractionProblem,
        library_path: Path,
        cache_key: str,
        schedule: DirectSchedule,
        expanded_tile_sizes: Tuple[int, ...],
        *,
        cache_hit: bool,
    ) -> None:
        self.problem = problem
        self.library_path = Path(library_path)
        self.cache_key = cache_key
        self.schedule = schedule
        self.expanded_tile_sizes = expanded_tile_sizes
        self.cache_hit = cache_hit
        try:
            self._library = ctypes.CDLL(str(self.library_path))
            self._function = getattr(
                self._library, "_mlir_ciface_{}".format(FUNCTION_NAME)
            )
        except (OSError, AttributeError) as error:
            raise BackendError(
                "cannot load native kernel '{}': {}".format(
                    self.library_path, error
                )
            ) from error

        descriptor_types = (
            _descriptor_type(len(problem.lhs.shape)),
            _descriptor_type(len(problem.rhs.shape)),
            _descriptor_type(len(problem.output.shape)),
        )
        self._function.argtypes = [
            ctypes.POINTER(descriptor_type)
            for descriptor_type in descriptor_types
        ]
        self._function.restype = None

    def bind(
        self, lhs: np.ndarray, rhs: np.ndarray, output: np.ndarray
    ) -> "NativeInvocation":
        """Validate one ABI binding and retain its memref descriptors.

        Tuners call :meth:`NativeInvocation.run` repeatedly so measured samples
        contain generated-kernel execution, not Python validation and
        descriptor construction. Arrays and descriptors remain strongly
        referenced for the lifetime of the binding.
        """

        lhs_value = _validate_array(
            "left operand", lhs, self.problem.lhs.shape, self.problem.lhs.strides
        )
        rhs_value = _validate_array(
            "right operand", rhs, self.problem.rhs.shape, self.problem.rhs.strides
        )
        output_value = _validate_array(
            "output", output, self.problem.output.shape, self.problem.output.strides
        )
        if not output_value.flags.writeable:
            raise BackendError("output must be writable")
        if np.shares_memory(output_value, lhs_value) or np.shares_memory(
            output_value, rhs_value
        ):
            raise BackendError(
                "output must not overlap an input; native Direct initializes it in place"
            )

        descriptors = tuple(
            _make_descriptor(value) for value in (lhs_value, rhs_value, output_value)
        )
        return NativeInvocation(
            self._function, lhs_value, rhs_value, output_value, descriptors
        )

    def run_into(
        self, lhs: np.ndarray, rhs: np.ndarray, output: np.ndarray
    ) -> np.ndarray:
        return self.bind(lhs, rhs, output).run()

    def run(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        output = np.empty(self.problem.output.shape, dtype=np.float32, order="C")
        return self.run_into(lhs, rhs, output)


class NativeInvocation:
    """A validated, reusable call binding for one loaded native kernel."""

    def __init__(
        self,
        function: object,
        lhs: np.ndarray,
        rhs: np.ndarray,
        output: np.ndarray,
        descriptors: Tuple[ctypes.Structure, ...],
    ) -> None:
        self._function = function
        self._arrays = (lhs, rhs, output)
        self._descriptors = descriptors
        self.output = output

    def run(self) -> np.ndarray:
        self._function(  # type: ignore[operator]
            *(ctypes.byref(descriptor) for descriptor in self._descriptors)
        )
        return self.output


class NativeCpuCompiler:
    """Compile and cache schedule-driven Direct contractions for the CPU."""

    def __init__(
        self,
        *,
        toolchain: Optional[NativeToolchain] = None,
        cache_dir: Optional[Path] = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.toolchain = toolchain or NativeToolchain.discover()
        self.cache_dir = Path(
            cache_dir
            or self.toolchain.repository_root
            / ".einsumcc-cache"
            / "native-mini-v0.1"
        )
        self.timeout_seconds = int(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("native compiler timeout must be positive")

    @property
    def shared_library_suffix(self) -> str:
        return ".dylib" if sys.platform == "darwin" else ".so"

    @property
    def tuning_target(self) -> str:
        """Hardware/toolchain-scoped identity for native measurements."""

        payload = {
            "kind": "native-cpu-mini-v0.1",
            "pipeline_revision": NATIVE_PIPELINE_REVISION,
            "toolchain": self.toolchain.identity(),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "native-cpu-mini-v0.1-{}".format(
            hashlib.sha256(encoded).hexdigest()[:16]
        )

    def _source(self, problem: ContractionProblem, *, c_interface: bool) -> str:
        return TcEmitter().emit(
            problem, FUNCTION_NAME, emit_c_interface=c_interface
        ).text

    def lower(
        self,
        problem: ContractionProblem,
        stage: str,
        schedule: Optional[DirectSchedule] = None,
    ) -> str:
        """Return real output from a selected native lowering stage."""

        if stage == "tc":
            return self._source(problem, c_interface=False)
        source = self._source(problem, c_interface=False)
        if stage == "linalg":
            return _run(
                (
                    str(self.toolchain.optimizer),
                    "-",
                    "--tc-contract-to-linalg",
                ),
                input_text=source,
                timeout_seconds=self.timeout_seconds,
            )
        selected = _native_schedule(problem, schedule)
        if stage == "scheduled":
            return _run(
                (
                    str(self.toolchain.optimizer),
                    "-",
                    *NATIVE_PIPELINE_PREFIX,
                    _schedule_pass_argument(problem, selected),
                ),
                input_text=source,
                timeout_seconds=self.timeout_seconds,
            )
        if stage not in ("llvm", "llvm-ir"):
            raise ValueError(
                "unknown native lowering stage '{}'; expected tc, linalg, scheduled, llvm, or llvm-ir".format(
                    stage
                )
            )
        pipeline = _native_pipeline(problem, selected)
        llvm_dialect = _run(
            (str(self.toolchain.optimizer), "-", *pipeline),
            input_text=source,
            timeout_seconds=self.timeout_seconds,
        )
        if stage == "llvm":
            return llvm_dialect
        return _run(
            (str(self.toolchain.translator), "--mlir-to-llvmir", "-"),
            input_text=llvm_dialect,
            timeout_seconds=self.timeout_seconds,
        )

    def _cache_identity(
        self,
        problem: ContractionProblem,
        source: str,
        schedule: DirectSchedule,
        pipeline: Tuple[str, ...],
    ) -> Tuple[str, Mapping[str, object]]:
        identity = {
            "schema": NATIVE_CACHE_SCHEMA,
            "pipeline_revision": NATIVE_PIPELINE_REVISION,
            "pipeline": list(pipeline),
            "schedule": dict(schedule.as_dict()),
            "expanded_tile_sizes": list(schedule.expanded_tile_sizes(problem)),
            # Layout is intentionally part of workload identity even though
            # the generated function accepts dynamic input strides. This keeps
            # one manifest faithful to the exact static ABI validated by the
            # NativeKernel returned for it.
            "workload_key": problem.workload_key,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "toolchain": self.toolchain.identity(),
        }
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), identity

    def _valid_entry(
        self, entry: Path, key: str, identity: Mapping[str, object]
    ) -> bool:
        manifest_path = entry / "manifest.json"
        library_path = entry / ("module" + self.shared_library_suffix)
        if not manifest_path.is_file() or not library_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        structurally_valid = (
            isinstance(manifest, dict)
            and manifest.get("key") == key
            and manifest.get("identity") == identity
            and isinstance(manifest.get("library_sha256"), str)
        )
        if not structurally_valid:
            return False
        try:
            return _sha256_file(library_path) == manifest["library_sha256"]
        except OSError:
            return False

    def _compile_entry(
        self,
        problem: ContractionProblem,
        source: str,
        temporary: Path,
        key: str,
        identity: Mapping[str, object],
        schedule: DirectSchedule,
        pipeline: Tuple[str, ...],
    ) -> None:
        tc_path = temporary / "module.tc.mlir"
        llvm_dialect_path = temporary / "module.llvm.mlir"
        llvm_ir_path = temporary / "module.ll"
        library_path = temporary / ("module" + self.shared_library_suffix)
        tc_path.write_text(source, encoding="utf-8")

        _run(
            (
                str(self.toolchain.optimizer),
                str(tc_path),
                *pipeline,
                "-o",
                str(llvm_dialect_path),
            ),
            timeout_seconds=self.timeout_seconds,
        )
        _run(
            (
                str(self.toolchain.translator),
                "--mlir-to-llvmir",
                str(llvm_dialect_path),
                "-o",
                str(llvm_ir_path),
            ),
            timeout_seconds=self.timeout_seconds,
        )
        clang_arguments = [str(self.toolchain.clang), "-O2"]
        if sys.platform == "darwin":
            clang_arguments.extend(("-Wno-override-module", "-dynamiclib"))
        else:
            clang_arguments.extend(("-shared", "-fPIC"))
        clang_arguments.extend((str(llvm_ir_path), "-o", str(library_path)))
        _run(clang_arguments, timeout_seconds=self.timeout_seconds)

        manifest = {
            "schema": NATIVE_CACHE_SCHEMA,
            "key": key,
            "identity": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "problem": dict(problem.describe()),
            "schedule": dict(schedule.as_dict()),
            "expanded_tile_sizes": list(schedule.expanded_tile_sizes(problem)),
            "library_sha256": _sha256_file(library_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def compile(
        self,
        problem: ContractionProblem,
        schedule: Optional[DirectSchedule] = None,
    ) -> NativeKernel:
        if fcntl is None:
            raise BackendError(
                "native artifact locking requires a POSIX host (macOS or Linux)"
            )
        selected = _native_schedule(problem, schedule)
        expanded = selected.expanded_tile_sizes(problem)
        pipeline = _native_pipeline(problem, selected)
        source = self._source(problem, c_interface=True)
        key, identity = self._cache_identity(
            problem, source, selected, pipeline
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = self.cache_dir / key
        lock_path = self.cache_dir / (key + ".lock")

        try:
            with lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                if self._valid_entry(entry, key, identity):
                    return NativeKernel(
                        problem,
                        entry / ("module" + self.shared_library_suffix),
                        key,
                        selected,
                        expanded,
                        cache_hit=True,
                    )

                temporary = Path(
                    tempfile.mkdtemp(prefix=key + ".tmp.", dir=str(self.cache_dir))
                )
                try:
                    self._compile_entry(
                        problem,
                        source,
                        temporary,
                        key,
                        identity,
                        selected,
                        pipeline,
                    )
                    if entry.exists() or entry.is_symlink():
                        # Invalid entries are quarantined before installing a
                        # replacement.  Renaming a directory over another
                        # non-empty directory is not portable across POSIX
                        # hosts, while deleting first would leave a crash
                        # window with no recoverable prior artifact.
                        stale = self.cache_dir / (key + ".stale.{}".format(os.getpid()))
                        if stale.exists() or stale.is_symlink():
                            _remove_path(stale)
                        os.replace(str(entry), str(stale))
                        try:
                            os.replace(str(temporary), str(entry))
                        except BaseException:
                            os.replace(str(stale), str(entry))
                            raise
                        _remove_path(stale)
                    else:
                        os.replace(str(temporary), str(entry))
                finally:
                    if temporary.exists():
                        _remove_path(temporary)
                return NativeKernel(
                    problem,
                    entry / ("module" + self.shared_library_suffix),
                    key,
                    selected,
                    expanded,
                    cache_hit=False,
                )
        except OSError as error:
            raise BackendError(
                "cannot populate native cache '{}': {}".format(
                    self.cache_dir, error
                )
            ) from error

"""Command-line interface for inspecting and testing the Nano v1 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .benchmark import BenchmarkCorpus, CpuBenchmarkRunner
from .cache import TuningCache
from .compiler import Compiler
from .errors import EinsumCCError
from .mlir_emitter import MlirEmitter
from .problem import ContractionProblem
from .target import CPU_MODEL, TARGETS, get_target
from .tc_emitter import TcEmitter


def _integers(text: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("equation", help="explicit two-input einsum, e.g. bij,bjk->bik")
    parser.add_argument("--lhs-shape", required=True, type=_integers)
    parser.add_argument("--rhs-shape", required=True, type=_integers)
    parser.add_argument("--lhs-strides", type=_integers, default=None)
    parser.add_argument("--rhs-strides", type=_integers, default=None)


def _problem(arguments: argparse.Namespace) -> ContractionProblem:
    return ContractionProblem.create(
        arguments.equation,
        arguments.lhs_shape,
        arguments.rhs_shape,
        lhs_strides=arguments.lhs_strides,
        rhs_strides=arguments.rhs_strides,
    )


def _random_array(
    shape: Sequence[int], strides: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    required = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    storage = rng.standard_normal(required).astype(np.float32)
    byte_strides = tuple(stride * storage.dtype.itemsize for stride in strides)
    return np.lib.stride_tricks.as_strided(storage, shape=tuple(shape), strides=byte_strides)


def _arrays(problem: ContractionProblem, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        _random_array(problem.lhs.shape, problem.lhs.strides, rng),
        _random_array(problem.rhs.shape, problem.rhs.strides, rng),
    )


def _command_explain(arguments: argparse.Namespace) -> int:
    problem = _problem(arguments)
    target = get_target(arguments.target)
    cache = TuningCache(Path(arguments.cache)) if arguments.cache else None
    compiled = Compiler(target, cache=cache).compile(problem, use_cache=not arguments.no_cache)
    print(compiled.explain())
    return 0


def _command_verify(arguments: argparse.Namespace) -> int:
    problem = _problem(arguments)
    lhs, rhs = _arrays(problem, arguments.seed)
    reference = np.einsum(problem.equation.text, lhs, rhs, dtype=np.float32)
    compiler = Compiler(CPU_MODEL)
    failures = 0
    for plan in compiler.planner.enumerate(problem):
        if not plan.legal:
            print("SKIP {:12s} {}".format(plan.kind.value, "; ".join(plan.rationale)))
            continue
        compiled = compiler.compile(problem, force=plan.kind, use_cache=False)
        result = compiled.run_cpu(lhs, rhs)
        try:
            np.testing.assert_allclose(
                result, reference, rtol=arguments.rtol, atol=arguments.atol
            )
        except AssertionError as error:
            failures += 1
            print("FAIL {:12s} {}".format(plan.kind.value, str(error).splitlines()[0]))
        else:
            maximum = float(np.max(np.abs(result - reference))) if result.size else 0.0
            print("PASS {:12s} max_abs_error={:.6g}".format(plan.kind.value, maximum))
    return 1 if failures else 0


def _command_tune(arguments: argparse.Namespace) -> int:
    problem = _problem(arguments)
    lhs, rhs = _arrays(problem, arguments.seed)
    cache = TuningCache(Path(arguments.cache))
    compiler = Compiler(CPU_MODEL, cache=cache)
    result = compiler.tune_cpu(
        problem,
        lhs,
        rhs,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        max_direct_schedules=arguments.max_direct_schedules,
    )
    print("Best plan: {}".format(result.best.plan.kind.value))
    print("Median: {:.3f} us".format(result.best.median_us))
    if result.best.schedule is not None:
        print("Schedule: {}".format(dict(result.best.schedule.as_dict())))
    print("Measured candidates: {}".format(len(result.measurements)))
    print("Cache: {}".format(cache.path))
    return 0


def _command_emit_mlir(arguments: argparse.Namespace) -> int:
    emitter = TcEmitter() if arguments.stage == "tc" else MlirEmitter()
    module = emitter.emit(_problem(arguments), arguments.function)
    if arguments.output:
        Path(arguments.output).write_text(module.text, encoding="utf-8")
    else:
        print(module.text, end="")
    return 0


def _command_benchmark(arguments: argparse.Namespace) -> int:
    corpus = BenchmarkCorpus.load(Path(arguments.corpus))
    report = CpuBenchmarkRunner(
        warmups=arguments.warmups, repeats=arguments.repeats, seed=arguments.seed
    ).run(corpus)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="einsumcc", description="Compile and inspect binary tensor contractions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    explain = subparsers.add_parser("explain", help="show index and plan analysis")
    _add_problem_arguments(explain)
    explain.add_argument("--target", choices=sorted(TARGETS), default=CPU_MODEL.name)
    explain.add_argument("--cache", help="optional tuning-cache path")
    explain.add_argument("--no-cache", action="store_true")
    explain.set_defaults(handler=_command_explain)

    verify = subparsers.add_parser("verify", help="differentially test all CPU plans")
    _add_problem_arguments(verify)
    verify.add_argument("--seed", type=int, default=0)
    verify.add_argument("--rtol", type=float, default=1.0e-4)
    verify.add_argument("--atol", type=float, default=1.0e-5)
    verify.set_defaults(handler=_command_verify)

    tune = subparsers.add_parser("tune", help="measure CPU plans and Direct schedules")
    _add_problem_arguments(tune)
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--warmups", type=int, default=1)
    tune.add_argument("--repeats", type=int, default=5)
    tune.add_argument("--max-direct-schedules", type=int, default=12)
    tune.add_argument("--cache", default=".einsumcc-cache/tuning-v1.json")
    tune.set_defaults(handler=_command_tune)

    emit = subparsers.add_parser(
        "emit-mlir", help="emit tc.contract or lowered semantic MLIR"
    )
    _add_problem_arguments(emit)
    emit.add_argument("--function", default="contract")
    emit.add_argument("--stage", choices=("tc", "linalg"), default="tc")
    emit.add_argument("-o", "--output")
    emit.set_defaults(handler=_command_emit_mlir)

    benchmark = subparsers.add_parser(
        "benchmark", help="run a versioned CPU workload corpus"
    )
    benchmark.add_argument("corpus")
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("-o", "--output")
    benchmark.set_defaults(handler=_command_benchmark)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (EinsumCCError, ValueError, OSError) as error:
        parser.exit(2, "einsumcc: error: {}\n".format(error))
    return 2

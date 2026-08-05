"""Immutable semantic IR for a statically shaped contraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import prod
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from .equation import Equation
from .errors import VerificationError


def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    """Return C-order element strides (not byte strides)."""

    running = 1
    result = []
    for extent in reversed(tuple(shape)):
        result.append(running)
        running *= extent
    return tuple(reversed(result))


@dataclass(frozen=True)
class TensorSpec:
    """Static tensor metadata consumed by target-independent passes."""

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str = "f32"

    @classmethod
    def create(
        cls,
        shape: Sequence[int],
        strides: Optional[Sequence[int]] = None,
        dtype: str = "f32",
    ) -> "TensorSpec":
        normalized_shape = tuple(int(value) for value in shape)
        if not normalized_shape or any(value <= 0 for value in normalized_shape):
            raise VerificationError("tensor dimensions must be positive")
        if dtype != "f32":
            raise VerificationError("Nano v1 executable semantics support only f32")
        normalized_strides = (
            contiguous_strides(normalized_shape)
            if strides is None
            else tuple(int(value) for value in strides)
        )
        if len(normalized_strides) != len(normalized_shape):
            raise VerificationError("shape and stride ranks differ")
        if any(value <= 0 for value in normalized_strides):
            raise VerificationError("Nano v1 requires positive element strides")
        return cls(normalized_shape, normalized_strides, dtype)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def itemsize(self) -> int:
        return 4

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def is_c_contiguous(self) -> bool:
        return self.strides == contiguous_strides(self.shape)


@dataclass(frozen=True)
class IndexGroups:
    """Stable B/M/N/K classification for GEMM-like planning."""

    batch: Tuple[str, ...]
    left_free: Tuple[str, ...]
    right_free: Tuple[str, ...]
    reduction: Tuple[str, ...]

    @property
    def canonical_output(self) -> Tuple[str, ...]:
        return self.batch + self.left_free + self.right_free


@dataclass(frozen=True)
class ContractionProblem:
    """The target-independent IR shared by every backend."""

    equation: Equation
    lhs: TensorSpec
    rhs: TensorSpec
    output: TensorSpec
    groups: IndexGroups
    extents: Mapping[str, int]

    @classmethod
    def create(
        cls,
        equation: Union[str, Equation],
        lhs_shape: Sequence[int],
        rhs_shape: Sequence[int],
        *,
        lhs_strides: Optional[Sequence[int]] = None,
        rhs_strides: Optional[Sequence[int]] = None,
        output_strides: Optional[Sequence[int]] = None,
        dtype: str = "f32",
    ) -> "ContractionProblem":
        parsed = Equation.parse(equation) if isinstance(equation, str) else equation
        # Be defensive for Equation-like objects created by older callers or
        # deserializers: semantic construction always rechecks its boundary.
        parsed.verify_structure()
        lhs = TensorSpec.create(lhs_shape, lhs_strides, dtype)
        rhs = TensorSpec.create(rhs_shape, rhs_strides, dtype)
        if len(parsed.lhs) != len(lhs.shape):
            raise VerificationError("left subscript rank does not match left shape")
        if len(parsed.rhs) != len(rhs.shape):
            raise VerificationError("right subscript rank does not match right shape")

        extents: Dict[str, int] = {}
        for labels, shape, operand in (
            (parsed.lhs, lhs.shape, "left"),
            (parsed.rhs, rhs.shape, "right"),
        ):
            for label, extent in zip(labels, shape):
                previous = extents.setdefault(label, extent)
                if previous != extent:
                    raise VerificationError(
                        "index '{}' has inconsistent extents ({} versus {} in {} operand)".format(
                            label, previous, extent, operand
                        )
                    )

        output_shape = tuple(extents[label] for label in parsed.output)
        # TensorSpec intentionally rejects rank zero; construct the scalar case
        # directly because C-order scalar strides are also empty.
        if output_shape:
            output = TensorSpec.create(output_shape, output_strides, dtype)
        else:
            if output_strides not in (None, (), []):
                raise VerificationError("scalar output cannot have strides")
            output = TensorSpec((), (), dtype)
        if not output.is_c_contiguous:
            raise VerificationError(
                "Nano v1 requires a C-contiguous output layout; non-contiguous output lowering is deferred"
            )

        classified = parsed.classify()
        groups = IndexGroups(
            classified["batch"],
            classified["left_free"],
            classified["right_free"],
            classified["reduction"],
        )
        return cls(parsed, lhs, rhs, output, groups, MappingProxyType(dict(extents)))

    def extent_product(self, labels: Sequence[str]) -> int:
        return prod(self.extents[label] for label in labels)

    @property
    def batch_size(self) -> int:
        return self.extent_product(self.groups.batch)

    @property
    def m(self) -> int:
        return self.extent_product(self.groups.left_free)

    @property
    def n(self) -> int:
        return self.extent_product(self.groups.right_free)

    @property
    def k(self) -> int:
        return self.extent_product(self.groups.reduction)

    @property
    def flops(self) -> int:
        return 2 * self.batch_size * self.m * self.n * self.k

    @property
    def workload_key(self) -> str:
        """Stable hash used by tuning caches across processes."""

        payload = {
            "schema": 1,
            "equation": self.equation.text,
            "lhs": asdict(self.lhs),
            "rhs": asdict(self.rhs),
            "output": asdict(self.output),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def describe(self) -> Mapping[str, object]:
        return {
            "equation": self.equation.text,
            "lhs_shape": self.lhs.shape,
            "rhs_shape": self.rhs.shape,
            "output_shape": self.output.shape,
            "batch": self.groups.batch,
            "left_free": self.groups.left_free,
            "right_free": self.groups.right_free,
            "reduction": self.groups.reduction,
            "B": self.batch_size,
            "M": self.m,
            "N": self.n,
            "K": self.k,
            "flops": self.flops,
            "workload_key": self.workload_key,
        }

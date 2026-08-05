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

    def __post_init__(self) -> None:
        """Normalize and verify metadata on every construction path.

        ``TensorSpec`` is part of the public semantic IR, so callers must not
        be able to bypass Nano v1 invariants by invoking the dataclass
        constructor instead of :meth:`create`. Rank zero is valid here for a
        scalar result; ``ContractionProblem`` separately requires both inputs
        to have the ranks declared by their non-empty subscripts.
        """

        try:
            shape = tuple(self.shape)
            strides = tuple(self.strides)
        except TypeError as error:
            raise VerificationError("tensor shape and strides must be integers") from error
        if any(type(value) is not int for value in shape + strides):
            raise VerificationError("tensor shape and strides must be integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        if any(value <= 0 for value in shape):
            raise VerificationError("tensor dimensions must be positive")
        if self.dtype != "f32":
            raise VerificationError("Nano v1 executable semantics support only f32")
        if len(strides) != len(shape):
            raise VerificationError("shape and stride ranks differ")
        if any(value <= 0 for value in strides):
            raise VerificationError("Nano v1 requires positive element strides")

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
        normalized_strides = (
            contiguous_strides(normalized_shape)
            if strides is None
            else tuple(int(value) for value in strides)
        )
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

    def __post_init__(self) -> None:
        names = ("batch", "left_free", "right_free", "reduction")
        groups = []
        for name in names:
            labels = tuple(getattr(self, name))
            object.__setattr__(self, name, labels)
            if any(
                not isinstance(label, str)
                or len(label) != 1
                or not ("A" <= label <= "Z" or "a" <= label <= "z")
                for label in labels
            ):
                raise VerificationError("index groups require single ASCII-letter labels")
            if len(labels) != len(set(labels)):
                raise VerificationError("an index group cannot repeat a label")
            groups.append(labels)
        flattened = tuple(label for group in groups for label in group)
        if len(flattened) != len(set(flattened)):
            raise VerificationError("B/M/N/K index groups must be disjoint")
        if not self.reduction:
            raise VerificationError("Nano v1 requires at least one reduction index")

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

    def __post_init__(self) -> None:
        """Re-establish the complete semantic-IR invariant.

        Backends may safely trust a ``ContractionProblem`` regardless of
        whether it came from :meth:`create`, direct public construction, or a
        future deserializer. Supplied derived metadata is checked rather than
        silently repaired so contradictory IR is diagnosed at its boundary.
        """

        if not isinstance(self.equation, Equation):
            raise VerificationError("contraction equation must be a verified Equation")
        self.equation.verify_structure()
        if not all(
            isinstance(spec, TensorSpec) for spec in (self.lhs, self.rhs, self.output)
        ):
            raise VerificationError("contraction operands must be TensorSpec values")
        if not isinstance(self.groups, IndexGroups):
            raise VerificationError("contraction groups must be an IndexGroups value")

        if len(self.equation.lhs) != len(self.lhs.shape):
            raise VerificationError("left subscript rank does not match left shape")
        if len(self.equation.rhs) != len(self.rhs.shape):
            raise VerificationError("right subscript rank does not match right shape")

        expected_extents: Dict[str, int] = {}
        for labels, shape, operand in (
            (self.equation.lhs, self.lhs.shape, "left"),
            (self.equation.rhs, self.rhs.shape, "right"),
        ):
            for label, extent in zip(labels, shape):
                previous = expected_extents.setdefault(label, extent)
                if previous != extent:
                    raise VerificationError(
                        "index '{}' has inconsistent extents ({} versus {} in {} operand)".format(
                            label, previous, extent, operand
                        )
                    )

        expected_output_shape = tuple(
            expected_extents[label] for label in self.equation.output
        )
        if self.output.shape != expected_output_shape:
            raise VerificationError(
                "output shape {} does not match equation-derived shape {}".format(
                    self.output.shape, expected_output_shape
                )
            )
        if not self.output.is_c_contiguous:
            raise VerificationError(
                "Nano v1 requires a C-contiguous output layout; non-contiguous output lowering is deferred"
            )

        classified = self.equation.classify()
        expected_groups = IndexGroups(
            classified["batch"],
            classified["left_free"],
            classified["right_free"],
            classified["reduction"],
        )
        if self.groups != expected_groups:
            raise VerificationError("B/M/N/K groups do not match the equation")

        try:
            supplied_items = tuple(self.extents.items())
        except (AttributeError, TypeError) as error:
            raise VerificationError("contraction extents must be an integer mapping") from error
        if any(
            not isinstance(label, str)
            or type(extent) is not int
            for label, extent in supplied_items
        ):
            raise VerificationError("contraction extents must be an integer mapping")
        supplied_extents = dict(supplied_items)
        if supplied_extents != expected_extents:
            raise VerificationError("extent mapping does not match operand shapes")
        object.__setattr__(
            self, "extents", MappingProxyType(dict(expected_extents))
        )

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

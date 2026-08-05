"""Parser and structural verifier for the Nano v1 einsum language.

The accepted syntax intentionally follows the explicit-output subset of
NumPy/PyTorch einsum. Keeping parsing independent from NumPy prevents the
reference implementation from defining compiler semantics accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Tuple

from .errors import ParseError, VerificationError

_LABELS = re.compile(r"^[A-Za-z]*$")


@dataclass(frozen=True)
class Equation:
    """A verified binary Einstein summation equation."""

    lhs: Tuple[str, ...]
    rhs: Tuple[str, ...]
    output: Tuple[str, ...]

    @classmethod
    def parse(cls, text: str) -> "Equation":
        """Parse and structurally verify an explicit binary equation.

        Whitespace is insignificant. Output indices are mandatory syntactically
        (`->` must be present), although the output may be empty for a scalar.
        """

        compact = "".join(text.split())
        if compact.count("->") != 1:
            raise ParseError("equation must contain exactly one explicit '->'")
        inputs, output = compact.split("->")
        operands = inputs.split(",")
        if len(operands) != 2:
            raise ParseError("Nano v1 requires exactly two input operands")
        lhs, rhs = operands
        if not lhs or not rhs:
            raise ParseError("input subscripts cannot be empty")
        for name, labels in (("left input", lhs), ("right input", rhs), ("output", output)):
            if "..." in labels:
                raise ParseError("ellipsis is not supported in Nano v1")
            if not _LABELS.fullmatch(labels):
                raise ParseError("{} contains a non-letter index".format(name))
            if len(labels) != len(set(labels)):
                raise VerificationError(
                    "{} repeats an index; diagonal semantics are not supported".format(name)
                )

        equation = cls(tuple(lhs), tuple(rhs), tuple(output))
        equation.verify_structure()
        return equation

    def verify_structure(self) -> None:
        """Verify that every label belongs to the supported B/M/N/K model."""

        if not (1 <= len(self.lhs) <= 6 and 1 <= len(self.rhs) <= 6):
            raise VerificationError("each input rank must be between 1 and 6")

        lhs = set(self.lhs)
        rhs = set(self.rhs)
        output = set(self.output)
        inputs = lhs | rhs

        unknown_output = output - inputs
        if unknown_output:
            raise VerificationError(
                "output indices are absent from both inputs: {}".format(
                    ", ".join(sorted(unknown_output))
                )
            )

        missing_output = (lhs ^ rhs) - output
        if missing_output:
            raise VerificationError(
                "free indices must appear in the output: {}".format(
                    ", ".join(sorted(missing_output))
                )
            )

        reductions = (lhs & rhs) - output
        if not reductions:
            raise VerificationError("Nano v1 requires at least one reduction index")

        # Shared output indices are batches. Shared non-output indices are K.
        # All other possible memberships have been ruled out above.
        classified = output | reductions
        if classified != inputs:
            raise VerificationError("equation contains an unsupported index pattern")

    @property
    def text(self) -> str:
        return "{},{}->{}".format(
            "".join(self.lhs), "".join(self.rhs), "".join(self.output)
        )

    @property
    def reduction_labels(self) -> Tuple[str, ...]:
        output = set(self.output)
        rhs = set(self.rhs)
        return tuple(label for label in self.lhs if label in rhs and label not in output)

    def classify(self) -> Dict[str, Tuple[str, ...]]:
        """Return stable B/M/N/K orders used by planning and codegen.

        B/M/N follow output order, making the desired output layout canonical
        whenever it is B+M+N. K follows the left operand; the right operand may
        require a permutation or pack when its reduction order differs.
        """

        lhs = set(self.lhs)
        rhs = set(self.rhs)
        output = set(self.output)
        batch = tuple(label for label in self.output if label in lhs and label in rhs)
        left_free = tuple(label for label in self.output if label in lhs and label not in rhs)
        right_free = tuple(label for label in self.output if label in rhs and label not in lhs)
        reduction = tuple(label for label in self.lhs if label in rhs and label not in output)
        return {
            "batch": batch,
            "left_free": left_free,
            "right_free": right_free,
            "reduction": reduction,
        }


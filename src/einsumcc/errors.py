"""Domain-specific exceptions with user-facing messages."""


class EinsumCCError(Exception):
    """Base class for expected compiler errors."""


class ParseError(EinsumCCError):
    """Raised when an Einstein equation is syntactically invalid."""


class VerificationError(EinsumCCError):
    """Raised when parsed input is outside Nano v1 semantics."""


class BackendError(EinsumCCError):
    """Raised when an execution backend cannot honor a compiled plan."""


class CacheError(EinsumCCError):
    """Raised when a tuning-cache record is malformed or incompatible."""


"""Immutable contracts for Nautical's Taskwarrior integration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Generic, TypeAlias, TypeVar


class IntegrationContractError(ValueError):
    """Raised when an integration model violates a boundary invariant."""


class CommandFailureKind(str, Enum):
    SUCCESS = "success"
    ABSENT = "absent"
    TIMEOUT = "timeout"
    BUSY = "busy"
    MISSING_BINARY = "missing_binary"
    INVALID_RESPONSE = "invalid_response"
    REJECTED = "rejected"
    EXECUTION_FAILURE = "execution_failure"


_RETRYABLE_FAILURES = frozenset(
    {
        CommandFailureKind.TIMEOUT,
        CommandFailureKind.BUSY,
        CommandFailureKind.EXECUTION_FAILURE,
    }
)


def _required_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise IntegrationContractError(f"{field} is required")
    return text


def _non_negative_float(value: object, field: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise IntegrationContractError(f"{field} must be a finite {qualifier} number")
    return number


@dataclass(frozen=True, slots=True)
class TaskCommand:
    """One bounded Taskwarrior invocation with an observable purpose."""

    argv: tuple[str, ...]
    purpose: str
    timeout: float
    input_text: str | None = None

    def __post_init__(self) -> None:
        argv = tuple(str(arg) for arg in self.argv)
        if not argv or not argv[0].strip():
            raise IntegrationContractError("task command requires an executable")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "purpose", _required_text(self.purpose, "command purpose"))
        object.__setattr__(self, "timeout", _non_negative_float(self.timeout, "command timeout", positive=True))
        if self.input_text is not None and not isinstance(self.input_text, str):
            raise IntegrationContractError("command input_text must be text or None")


@dataclass(frozen=True, slots=True)
class TaskCommandResult:
    """Lossless evidence from one final Taskwarrior command attempt."""

    command: TaskCommand
    returncode: int
    stdout: str
    stderr: str
    kind: CommandFailureKind
    attempt: int
    duration: float

    def __post_init__(self) -> None:
        if not isinstance(self.command, TaskCommand):
            raise IntegrationContractError("command result requires a TaskCommand")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise IntegrationContractError("command returncode must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise IntegrationContractError("command output must be text")
        try:
            kind = CommandFailureKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid command failure kind") from exc
        if kind is CommandFailureKind.SUCCESS and self.returncode != 0:
            raise IntegrationContractError("successful command result requires returncode 0")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise IntegrationContractError("command attempt must be a positive integer")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "duration", _non_negative_float(self.duration, "command duration"))

    @property
    def ok(self) -> bool:
        return self.kind is CommandFailureKind.SUCCESS


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    """Structured evidence explaining why an authoritative read is unavailable."""

    command: TaskCommand
    kind: CommandFailureKind
    returncode: int
    attempt: int
    duration: float
    retryable: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.command, TaskCommand):
            raise IntegrationContractError("failure evidence requires a TaskCommand")
        try:
            kind = CommandFailureKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise IntegrationContractError("invalid failure evidence kind") from exc
        if kind in {CommandFailureKind.SUCCESS, CommandFailureKind.ABSENT}:
            raise IntegrationContractError("failure evidence requires an unavailable failure kind")
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise IntegrationContractError("failure returncode must be an integer")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise IntegrationContractError("failure attempt must be a positive integer")
        if not isinstance(self.retryable, bool):
            raise IntegrationContractError("failure retryable flag must be boolean")
        if self.retryable and kind not in _RETRYABLE_FAILURES:
            raise IntegrationContractError(f"{kind.value} failures cannot be marked retryable")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "duration", _non_negative_float(self.duration, "failure duration"))
        object.__setattr__(self, "detail", str(self.detail or "").strip())


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Found(Generic[T]):
    """An authoritative read that contains one non-null domain value."""

    value: T
    query: str

    def __post_init__(self) -> None:
        if self.value is None:
            raise IntegrationContractError("found read cannot contain None")
        object.__setattr__(self, "query", _required_text(self.query, "read query"))


@dataclass(frozen=True, slots=True)
class Absent:
    """An authoritative read proving that its requested value is absent."""

    query: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, "read query"))
        object.__setattr__(self, "reason", _required_text(self.reason, "absence reason"))


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A read that cannot authoritatively prove either presence or absence."""

    query: str
    evidence: FailureEvidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _required_text(self.query, "read query"))
        if not isinstance(self.evidence, FailureEvidence):
            raise IntegrationContractError("unavailable read requires failure evidence")

    @property
    def retryable(self) -> bool:
        return self.evidence.retryable


TaskRead: TypeAlias = Found[T] | Absent | Unavailable


__all__ = (
    "Absent",
    "CommandFailureKind",
    "FailureEvidence",
    "Found",
    "IntegrationContractError",
    "TaskCommand",
    "TaskCommandResult",
    "TaskRead",
    "Unavailable",
)

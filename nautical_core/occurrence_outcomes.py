"""Typed outcomes for scheduler occurrence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

from .occurrence_provider import Occurrence
from .scheduler_models import OccurrenceSearchExhausted
from .scheduler_cursor import OccurrenceCursor


@dataclass(frozen=True, slots=True)
class FoundOccurrence:
    occurrence: Occurrence
    source: str
    local_datetime: datetime
    utc_datetime: datetime
    projection: Any = None
    selected_term: Any = None

    status: Literal["found"] = "found"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "local": self.local_datetime.isoformat(),
            "utc": self.utc_datetime.isoformat(),
            "projection": self.projection,
            "selected_term": self.selected_term,
        }


@dataclass(frozen=True, slots=True)
class AbsentOccurrence:
    reason: str = "no matching occurrence"
    status: Literal["absent"] = "absent"

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ExhaustedOccurrence:
    error: OccurrenceSearchExhausted
    status: Literal["exhausted"] = "exhausted"

    @property
    def terminal_evidence(self) -> dict[str, Any]:
        return {
            "scope": self.error.scope,
            "kind": self.error.kind,
            "reference": self.error.reference,
            "limit": self.error.limit,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "error": str(self.error), **self.terminal_evidence}


@dataclass(frozen=True, slots=True)
class UnavailableOccurrence:
    reason: str
    error_type: str = ""
    status: Literal["unavailable"] = "unavailable"

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason, "error_type": self.error_type}


@dataclass(frozen=True, slots=True)
class InvalidOccurrence:
    reason: str
    error_type: str = ""
    status: Literal["invalid"] = "invalid"

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason, "error_type": self.error_type}


OccurrenceOutcome = (
    FoundOccurrence
    | AbsentOccurrence
    | ExhaustedOccurrence
    | UnavailableOccurrence
    | InvalidOccurrence
)


@dataclass(frozen=True, slots=True)
class OccurrenceCollectionResult:
    """Immutable bounded collection with explicit empty/terminal evidence."""

    occurrences: tuple[Occurrence, ...]
    cursor: OccurrenceCursor
    source: str = "scheduler"
    terminal: OccurrenceSearchExhausted | None = None
    empty_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.occurrences, tuple):
            raise TypeError("Occurrence collection must be immutable.")
        if not isinstance(self.cursor, OccurrenceCursor):
            raise TypeError("Occurrence collection requires its source cursor.")
        if not self.occurrences and not self.empty_reason:
            object.__setattr__(self, "empty_reason", "no matching occurrence")

    @property
    def status(self) -> Literal["found", "empty", "exhausted"]:
        if self.occurrences:
            return "found"
        if self.terminal is not None:
            return "exhausted"
        return "empty"

    def __len__(self) -> int:
        return len(self.occurrences)

    def __iter__(self) -> Iterator[Occurrence]:
        return iter(self.occurrences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "cursor": self.cursor.local_datetime.isoformat(),
            "inclusive": self.cursor.inclusive,
            "occurrences": [
                {
                    "local": item.local_datetime.isoformat() if item.local_datetime else None,
                    "source": item.source,
                    "description": item.description,
                    "omitted": item.omitted,
                }
                for item in self.occurrences
            ],
            "empty_reason": self.empty_reason,
            "terminal": str(self.terminal) if self.terminal else None,
        }


def mutation_candidate(outcome: OccurrenceOutcome) -> FoundOccurrence:
    """Return a found occurrence or fail closed for mutation callers."""
    if isinstance(outcome, FoundOccurrence):
        return outcome
    raise RuntimeError(
        f"recurrence mutation requires a found occurrence; received {outcome.status}"
    )


def presentation_summary(outcome: OccurrenceOutcome) -> str:
    """Return intentionally compact text for UI-only consumers."""
    if isinstance(outcome, FoundOccurrence):
        return outcome.local_datetime.isoformat()
    if isinstance(outcome, ExhaustedOccurrence):
        return f"exhausted: {outcome.error}"
    return f"{outcome.status}: {getattr(outcome, 'reason', '')}".rstrip(": ")


def outcome_from_occurrence(occurrence: Occurrence | None) -> FoundOccurrence | AbsentOccurrence:
    if occurrence is None:
        return AbsentOccurrence()
    if occurrence.local_datetime is None:
        raise ValueError("Cannot build a found outcome without a local datetime.")
    local = occurrence.local_datetime
    utc = local.astimezone(timezone.utc) if local.tzinfo else local
    return FoundOccurrence(
        occurrence=occurrence,
        source=occurrence.source,
        local_datetime=local,
        utc_datetime=utc,
    )


__all__ = (
    "AbsentOccurrence",
    "ExhaustedOccurrence",
    "FoundOccurrence",
    "InvalidOccurrence",
    "OccurrenceOutcome",
    "OccurrenceCollectionResult",
    "UnavailableOccurrence",
    "outcome_from_occurrence",
    "mutation_candidate",
    "presentation_summary",
)

"""Typed outcomes for scheduler occurrence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .occurrence_provider import Occurrence
from .scheduler_models import OccurrenceSearchExhausted


@dataclass(frozen=True, slots=True)
class FoundOccurrence:
    occurrence: Occurrence
    source: str
    local_datetime: datetime
    utc_datetime: datetime
    projection: Any = None
    selected_term: Any = None

    status: Literal["found"] = "found"


@dataclass(frozen=True, slots=True)
class AbsentOccurrence:
    reason: str = "no matching occurrence"
    status: Literal["absent"] = "absent"


@dataclass(frozen=True, slots=True)
class ExhaustedOccurrence:
    error: OccurrenceSearchExhausted
    status: Literal["exhausted"] = "exhausted"


@dataclass(frozen=True, slots=True)
class UnavailableOccurrence:
    reason: str
    error_type: str = ""
    status: Literal["unavailable"] = "unavailable"


@dataclass(frozen=True, slots=True)
class InvalidOccurrence:
    reason: str
    error_type: str = ""
    status: Literal["invalid"] = "invalid"


OccurrenceOutcome = (
    FoundOccurrence
    | AbsentOccurrence
    | ExhaustedOccurrence
    | UnavailableOccurrence
    | InvalidOccurrence
)


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
    "UnavailableOccurrence",
    "outcome_from_occurrence",
)

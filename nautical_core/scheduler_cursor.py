"""Explicit cursor semantics for occurrence lookups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class OccurrenceCursor:
    """A local lookup instant with explicit inclusivity and bounds.

    Ordinary scheduler lookups are strict-after by default.  Callers that
    need a range start can opt into inclusivity without subtracting an
    arbitrary microsecond or day in their own code.
    """

    local_datetime: datetime
    inclusive: bool = False
    timezone: Any | None = None
    date_limit: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.local_datetime, datetime):
            raise TypeError("Occurrence cursor requires a datetime instant.")
        if not isinstance(self.inclusive, bool):
            raise TypeError("Occurrence cursor inclusivity must be boolean.")
        if self.date_limit is not None and (
            not isinstance(self.date_limit, date) or isinstance(self.date_limit, datetime)
        ):
            raise TypeError("Occurrence cursor date_limit must be a calendar date.")

    @classmethod
    def strict_after(cls, local_datetime: datetime, *, timezone: Any | None = None, date_limit: date | None = None) -> "OccurrenceCursor":
        return cls(local_datetime, inclusive=False, timezone=timezone, date_limit=date_limit)

    @classmethod
    def inclusive_at(cls, local_datetime: datetime, *, timezone: Any | None = None, date_limit: date | None = None) -> "OccurrenceCursor":
        return cls(local_datetime, inclusive=True, timezone=timezone, date_limit=date_limit)


__all__ = ("OccurrenceCursor",)

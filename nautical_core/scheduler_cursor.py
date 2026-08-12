"""Explicit cursor semantics for occurrence lookups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal


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


@dataclass(frozen=True, slots=True)
class OccurrenceRangeRequest:
    """Validated bounded collection request for scheduler consumers."""

    cursor: OccurrenceCursor
    end_local: datetime | None = None
    limit: int = 1
    omission_policy: Literal["exclude", "include", "report"] = "exclude"
    max_iterations: int = 512
    max_file_skips: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.cursor, OccurrenceCursor):
            raise TypeError("Occurrence range requires an OccurrenceCursor.")
        if self.end_local is not None and not isinstance(self.end_local, datetime):
            raise TypeError("Occurrence range end must be a datetime.")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 0:
            raise ValueError("Occurrence range limit must be a non-negative integer.")
        if self.omission_policy not in {"exclude", "include", "report"}:
            raise ValueError("Occurrence omission policy must be exclude, include, or report.")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("Occurrence range iteration limit must be positive.")
        if isinstance(self.max_file_skips, bool) or not isinstance(self.max_file_skips, int) or self.max_file_skips <= 0:
            raise ValueError("Occurrence range file-skip limit must be positive.")
        if self.end_local is not None:
            cursor_value = self.cursor.local_datetime
            if (self.end_local.tzinfo is None) != (cursor_value.tzinfo is None):
                raise ValueError("Occurrence range boundaries must use compatible timezone awareness.")
            if self.end_local < cursor_value:
                raise ValueError("Occurrence range end must not precede its cursor.")

    @property
    def timezone(self) -> Any | None:
        return self.cursor.timezone


__all__ = ("OccurrenceCursor", "OccurrenceRangeRequest")

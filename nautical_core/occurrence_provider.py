"""Typed occurrence values and provider contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Protocol


@dataclass(frozen=True, slots=True)
class Occurrence:
    """A local calendar occurrence independent of Taskwarrior task shape."""

    day: date
    hour: int
    minute: int
    source: str = "anchor"
    description: str = ""
    local_datetime: datetime | None = field(default=None, compare=False, repr=False)
    omitted: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise TypeError("Occurrence day must be a calendar date.")
        if isinstance(self.hour, bool) or not isinstance(self.hour, int) or not 0 <= self.hour <= 23:
            raise ValueError("Occurrence hour must be between 0 and 23.")
        if isinstance(self.minute, bool) or not isinstance(self.minute, int) or not 0 <= self.minute <= 59:
            raise ValueError("Occurrence minute must be between 0 and 59.")
        if self.local_datetime is not None:
            if not isinstance(self.local_datetime, datetime):
                raise TypeError("Occurrence local_datetime must be a datetime.")
            if (self.local_datetime.date(), self.local_datetime.hour, self.local_datetime.minute) != (self.day, self.hour, self.minute):
                raise ValueError("Occurrence local_datetime does not match its date and clock fields.")
        if not isinstance(self.omitted, bool):
            raise TypeError("Occurrence omitted flag must be boolean.")

    @property
    def hhmm(self) -> tuple[int, int]:
        return self.hour, self.minute


class LazyOccurrenceProvider(Protocol):
    def next_after(
        self,
        after_local: datetime,
        *,
        build_local_datetime: Callable[[date, tuple[int, int]], datetime],
        to_local: Callable[[datetime], datetime],
    ) -> Occurrence | None:
        """Return the first occurrence strictly after a local datetime."""


class OccurrenceProvider(LazyOccurrenceProvider, Protocol):
    def occurrences(self) -> list[Occurrence]:
        """Return sorted, deduplicated local occurrences."""


def collect_after(
    provider: LazyOccurrenceProvider,
    after_local: datetime,
    *,
    limit: int,
    inclusive: bool = False,
    max_iterations: int = 512,
    build_local_datetime: Callable[[date, tuple[int, int]], datetime],
    to_local: Callable[[datetime], datetime],
) -> list[Occurrence]:
    """Collect a bounded stream while counting only non-omitted occurrences."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Occurrence collection limit must be a non-negative integer.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("Occurrence collection iteration limit must be a positive integer.")
    if limit == 0:
        return []
    cursor = after_local - timedelta(microseconds=1) if inclusive else after_local
    out: list[Occurrence] = []
    included_count = 0
    iterations = 0
    while included_count < limit and iterations < max_iterations:
        iterations += 1
        occurrence = provider.next_after(
            cursor,
            build_local_datetime=build_local_datetime,
            to_local=to_local,
        )
        if occurrence is None:
            break
        if not isinstance(occurrence, Occurrence):
            raise TypeError("Occurrence provider returned an invalid value.")
        if occurrence.local_datetime is None:
            raise ValueError("Lazy occurrence provider returned no local datetime.")
        cursor = occurrence.local_datetime
        out.append(occurrence)
        if not occurrence.omitted:
            included_count += 1
    if included_count < limit and iterations >= max_iterations:
        raise ValueError("Occurrence provider exceeded its collection iteration limit.")
    return out


def _require_forward_progress(after_local: datetime, value: datetime) -> None:
    if not isinstance(after_local, datetime) or not isinstance(value, datetime):
        raise TypeError("Occurrence provider must return datetime values.")
    try:
        advanced = value > after_local
    except TypeError as exc:
        raise ValueError("Occurrence provider returned an incomparable datetime.") from exc
    if not advanced:
        raise ValueError("Occurrence provider returned a non-advancing occurrence.")


class AnchorOccurrenceProvider:
    """Typed adapter for ordinary anchor occurrence projection.

    The scheduling engine remains injected so hooks and Navigator can adopt
    this boundary without duplicating recurrence semantics during migration.
    """

    def __init__(
        self,
        next_occurrence_after: Callable[[datetime], datetime | None],
    ) -> None:
        self._next_occurrence_after = next_occurrence_after

    def next_after(
        self,
        after_local: datetime,
        *,
        build_local_datetime: Callable[[date, tuple[int, int]], datetime],
        to_local: Callable[[datetime], datetime],
    ) -> Occurrence | None:
        del build_local_datetime
        value = self._next_occurrence_after(after_local)
        if value is None:
            return None
        local = to_local(value)
        if not isinstance(local, datetime):
            raise TypeError("Occurrence provider returned a non-datetime local value.")
        _require_forward_progress(after_local, local)
        return Occurrence(day=local.date(), hour=local.hour, minute=local.minute, local_datetime=local)


class AnchorEventOccurrenceProvider:
    """Typed adapter for anchor streams that retain omitted-event markers."""

    def __init__(self, next_event_after: Callable[[datetime], tuple[datetime, bool] | None]) -> None:
        self._next_event_after = next_event_after

    def next_after(
        self,
        after_local: datetime,
        *,
        build_local_datetime: Callable[[date, tuple[int, int]], datetime],
        to_local: Callable[[datetime], datetime],
    ) -> Occurrence | None:
        del build_local_datetime
        event = self._next_event_after(after_local)
        if event is None:
            return None
        value, omitted = event
        local = to_local(value)
        if not isinstance(local, datetime):
            raise TypeError("Occurrence event provider returned a non-datetime local value.")
        if not isinstance(omitted, bool):
            raise TypeError("Occurrence event provider returned a non-boolean omitted flag.")
        _require_forward_progress(after_local, local)
        return Occurrence(
            day=local.date(),
            hour=local.hour,
            minute=local.minute,
            local_datetime=local,
            omitted=omitted,
        )


__all__ = ("AnchorEventOccurrenceProvider", "AnchorOccurrenceProvider", "LazyOccurrenceProvider", "Occurrence", "OccurrenceProvider", "collect_after")

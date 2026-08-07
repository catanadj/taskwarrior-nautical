"""Typed occurrence values and provider contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Protocol

from .timeutil import compare_datetimes


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
        if not isinstance(self.source, str):
            raise TypeError("Occurrence source must be text.")
        if not isinstance(self.description, str):
            raise TypeError("Occurrence description must be text.")
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


def _datetime_is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _compare_datetimes(left: datetime, right: datetime) -> int:
    """Compatibility wrapper for the shared datetime comparator."""
    try:
        return compare_datetimes(left, right)
    except TypeError as exc:
        raise TypeError("Occurrence provider must compare datetime values.") from exc
    except ValueError as exc:
        raise ValueError("Occurrence provider returned an incomparable datetime.") from exc


def _sort_datetimes(values: list[datetime]) -> list[datetime]:
    """Sort a homogeneous datetime list without losing DST fold ordering."""
    if not values:
        return []
    aware = _datetime_is_aware(values[0])
    if any(_datetime_is_aware(value) != aware for value in values):
        raise ValueError("Occurrence provider returned incomparable datetime values.")
    if aware:
        return sorted(values, key=lambda value: value.astimezone(timezone.utc))
    return sorted(values)


def _cursor_before(value: datetime) -> datetime:
    """Return the instant immediately before an inclusive cursor."""
    if _datetime_is_aware(value):
        return (value.astimezone(timezone.utc) - timedelta(microseconds=1)).astimezone(value.tzinfo)
    return value - timedelta(microseconds=1)


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
    if not isinstance(after_local, datetime):
        raise TypeError("Occurrence collection requires a datetime cursor.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Occurrence collection limit must be a non-negative integer.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("Occurrence collection iteration limit must be a positive integer.")
    if limit == 0:
        return []
    cursor = _cursor_before(after_local) if inclusive else after_local
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
        _require_forward_progress(cursor, occurrence.local_datetime)
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
        advanced = _compare_datetimes(value, after_local) > 0
    except (TypeError, ValueError) as exc:
        raise ValueError("Occurrence provider returned an incomparable datetime.") from exc
    if not advanced:
        raise ValueError("Occurrence provider returned a non-advancing occurrence.")


def _occurrence_from_datetime(
    value: datetime,
    after_local: datetime,
    *,
    to_local: Callable[[datetime], datetime],
    source: str,
    description: str,
    omitted: bool = False,
) -> Occurrence:
    """Normalize a legacy datetime callback result into a typed occurrence."""
    if not isinstance(value, datetime):
        raise TypeError("Occurrence provider returned a non-datetime value.")
    local = to_local(value)
    if not isinstance(local, datetime):
        raise TypeError("Occurrence provider returned a non-datetime local value.")
    _require_forward_progress(after_local, local)
    return Occurrence(
        day=local.date(),
        hour=local.hour,
        minute=local.minute,
        source=source,
        description=description,
        local_datetime=local,
        omitted=omitted,
    )


class AnchorOccurrenceProvider:
    """Typed adapter for ordinary anchor occurrence projection.

    The scheduling engine remains injected so hooks and Navigator can adopt
    this boundary without duplicating recurrence semantics during migration.
    """

    def __init__(
        self,
        next_occurrence_after: Callable[[datetime], Occurrence | datetime | None],
        *,
        source: str = "anchor",
        description: str = "",
    ) -> None:
        self._next_occurrence_after = next_occurrence_after
        self._source = source
        self._description = description

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
        if isinstance(value, Occurrence):
            if value.local_datetime is None:
                raise ValueError("Occurrence provider returned an event without local datetime.")
            _require_forward_progress(after_local, value.local_datetime)
            return value
        return _occurrence_from_datetime(
            value,
            after_local,
            to_local=to_local,
            source=self._source,
            description=self._description,
        )


class AnchorEventOccurrenceProvider:
    """Typed adapter for anchor streams that retain omitted-event markers."""

    def __init__(
        self,
        next_event_after: Callable[[datetime], Occurrence | tuple[datetime, bool] | None],
        *,
        source: str = "anchor",
        description: str = "",
    ) -> None:
        self._next_event_after = next_event_after
        self._source = source
        self._description = description

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
        if isinstance(event, Occurrence):
            if event.local_datetime is None:
                raise ValueError("Occurrence event provider returned an event without local datetime.")
            _require_forward_progress(after_local, event.local_datetime)
            return event
        if not isinstance(event, tuple) or len(event) != 2:
            raise TypeError("Occurrence event provider must return a (datetime, omitted) tuple.")
        value, omitted = event
        if not isinstance(omitted, bool):
            raise TypeError("Occurrence event provider returned a non-boolean omitted flag.")
        return _occurrence_from_datetime(
            value,
            after_local,
            to_local=to_local,
            source=self._source,
            description=self._description,
            omitted=omitted,
        )


__all__ = ("AnchorEventOccurrenceProvider", "AnchorOccurrenceProvider", "LazyOccurrenceProvider", "Occurrence", "OccurrenceProvider", "collect_after")

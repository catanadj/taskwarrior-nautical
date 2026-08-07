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
        if occurrence is None or occurrence.local_datetime is None:
            break
        cursor = occurrence.local_datetime
        out.append(occurrence)
        if not occurrence.omitted:
            included_count += 1
    return out


def _require_forward_progress(after_local: datetime, value: datetime) -> None:
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
        _require_forward_progress(after_local, local)
        return Occurrence(
            day=local.date(),
            hour=local.hour,
            minute=local.minute,
            local_datetime=local,
            omitted=bool(omitted),
        )


__all__ = ("AnchorEventOccurrenceProvider", "AnchorOccurrenceProvider", "LazyOccurrenceProvider", "Occurrence", "OccurrenceProvider", "collect_after")

"""Test-only parity helpers for the scheduler migration.

These helpers deliberately accept the legacy callback from the test. They are
not imported by Nautical runtime modules and must be removed after cutover.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timezone
from typing import Any

from nautical_core.occurrence_outcomes import FoundOccurrence, OccurrenceCollectionResult
from nautical_core.occurrence_provider import Occurrence
from nautical_core.scheduler_cursor import OccurrenceCursor
from nautical_core.scheduler_service import SchedulerService


class SchedulerParityMismatch(AssertionError):
    """Raised when old and new scheduling paths disagree."""


def _occurrence_signature(item: Occurrence) -> tuple[Any, bool, str, str]:
    if not isinstance(item, Occurrence):
        raise SchedulerParityMismatch(f"scheduler returned a non-occurrence value: {item!r}")
    if item.local_datetime is None:
        raise SchedulerParityMismatch("scheduler returned an occurrence without a local datetime")
    instant = item.local_datetime.astimezone(timezone.utc) if item.local_datetime.tzinfo else item.local_datetime
    return instant, item.omitted, item.source, item.description


def assert_monotonic(items: Iterable[Occurrence]) -> tuple[Occurrence, ...]:
    """Assert one strict instant-ordered occurrence stream and return it."""
    values = tuple(items)
    previous = None
    for item in values:
        signature = _occurrence_signature(item)
        instant = signature[0]
        if previous is not None and instant <= previous:
            raise SchedulerParityMismatch(
                f"occurrence stream is not strictly monotonic: {previous!r} then {instant!r}"
            )
        previous = instant
    return values


def compare_next(
    service: SchedulerService,
    cursor: OccurrenceCursor,
    legacy_next: Callable[[], Occurrence | None],
    **kwargs: Any,
) -> FoundOccurrence | None:
    outcome = service.next(cursor, **kwargs)
    legacy = legacy_next()
    if isinstance(outcome, FoundOccurrence):
        current = outcome.occurrence.local_datetime
    else:
        current = None
    expected = legacy.local_datetime if legacy is not None else None
    if current != expected:
        raise SchedulerParityMismatch(
            f"next occurrence mismatch: service={current!r}, legacy={expected!r}"
        )
    return outcome if isinstance(outcome, FoundOccurrence) else None


def compare_collection(
    service: SchedulerService,
    cursor: OccurrenceCursor,
    legacy_collect: Callable[[], Iterable[Occurrence]],
    *,
    limit: int,
    **kwargs: Any,
) -> OccurrenceCollectionResult:
    result = service.collect(cursor, limit=limit, **kwargs)
    legacy = tuple(legacy_collect())
    current_values = assert_monotonic(result.occurrences)
    expected_values = assert_monotonic(legacy)
    current = tuple(_occurrence_signature(item) for item in current_values)
    expected = tuple(_occurrence_signature(item) for item in expected_values)
    if current != expected:
        raise SchedulerParityMismatch(
            f"collection mismatch: service={current!r}, legacy={expected!r}"
        )
    return result


__all__ = ("SchedulerParityMismatch", "assert_monotonic", "compare_collection", "compare_next")

"""Test-only parity helpers for the scheduler migration.

These helpers deliberately accept the legacy callback from the test. They are
not imported by Nautical runtime modules and must be removed after cutover.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from nautical_core.occurrence_outcomes import FoundOccurrence, OccurrenceCollectionResult
from nautical_core.occurrence_provider import Occurrence
from nautical_core.scheduler_cursor import OccurrenceCursor
from nautical_core.scheduler_service import SchedulerService


class SchedulerParityMismatch(AssertionError):
    """Raised when old and new scheduling paths disagree."""


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
    current = tuple(item.local_datetime for item in result.occurrences)
    expected = tuple(item.local_datetime for item in legacy)
    if current != expected:
        raise SchedulerParityMismatch(
            f"collection mismatch: service={current!r}, legacy={expected!r}"
        )
    return result


__all__ = ("SchedulerParityMismatch", "compare_collection", "compare_next")

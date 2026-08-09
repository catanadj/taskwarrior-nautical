"""Narrow callback contracts shared by recurrence orchestration."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol


class RecurrenceCacheLoader(Protocol):
    """Load one evaluator-owned cached value."""

    def __call__(self) -> Any:
        ...


class NextOccurrenceCallback(Protocol):
    """Resolve the next local occurrence after a local cursor."""

    def __call__(
        self,
        dnf: Any,
        after_local_dt: datetime,
        *,
        default_seed_date: date | None,
        seed_base: str,
        omit_dnf: Any = None,
        fallback_hhmm: tuple[int, int] | None = None,
    ) -> datetime | None:
        ...


class PickOccurrenceCallback(Protocol):
    """Resolve an occurrence on or after a cursor for inclusive lookups."""

    def __call__(
        self,
        dnf: Any,
        ref_dt_local: datetime,
        inclusive: bool,
        fallback_hhmm: tuple[int, int],
        interval_seed: date | None,
        seed_base: str,
        omit_dnf: Any = None,
    ) -> datetime | None:
        ...


__all__ = (
    "NextOccurrenceCallback",
    "PickOccurrenceCallback",
    "RecurrenceCacheLoader",
)

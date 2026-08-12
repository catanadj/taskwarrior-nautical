"""Typed boundary between calendar-date selection and time projection.

The scheduler chooses a calendar date.  This module projects a parsed ``@t``
modifier onto that date without ever advancing or replacing the date itself.
Consumers can therefore distinguish an unavailable event on that date from an
invalid modifier or a terminal scheduler failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping

from .scheduler_models import OccurrenceSearchExhausted
from .time_slots import resolve_time_slots_with_offsets


@dataclass(frozen=True, slots=True)
class ProjectedTime:
    """Successful projection owned by one selected calendar date."""

    selected_date: date
    slots: tuple[tuple[int, int, int], ...]
    source: str = "clock"

    def __post_init__(self) -> None:
        if not isinstance(self.selected_date, date):
            raise TypeError("A time projection requires its selected calendar date.")
        if not isinstance(self.slots, tuple):
            raise TypeError("Projected time slots must be immutable.")
        for slot in self.slots:
            if not isinstance(slot, tuple) or len(slot) != 3:
                raise ValueError("Projected time slots must be (day offset, hour, minute) tuples.")
            day_offset, hour, minute = slot
            if not isinstance(day_offset, int) or not 0 <= day_offset <= 1:
                raise ValueError("Projected day offsets must be 0 or 1.")
            if not isinstance(hour, int) or not 0 <= hour <= 23:
                raise ValueError("Projected hours must be between 0 and 23.")
            if not isinstance(minute, int) or not 0 <= minute <= 59:
                raise ValueError("Projected minutes must be between 0 and 59.")


@dataclass(frozen=True, slots=True)
class ProjectionUnavailable:
    """The selected date is valid, but its requested event is unavailable."""

    selected_date: date
    reason: str
    error_type: str = ""


@dataclass(frozen=True, slots=True)
class ProjectionInvalid:
    """The time modifier or projection context is invalid."""

    selected_date: date
    reason: str
    error_type: str = ""


@dataclass(frozen=True, slots=True)
class ProjectionTerminal:
    """Projection stopped because the authoritative scheduler was exhausted."""

    selected_date: date
    error: OccurrenceSearchExhausted


ProjectionResult = ProjectedTime | ProjectionUnavailable | ProjectionInvalid | ProjectionTerminal
ProjectionResolver = Callable[..., list[tuple[int, int, int]]]


def _source_name(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("time_random"):
            return "random"
        if value.get("time_window"):
            return "window"
        raw = value.get("t")
    else:
        raw = value
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"sunrise", "sunset", "dawn", "dusk", "moonrise", "moonset"}:
            return "astronomy"
        if "," in token:
            return "clock-list"
    return "clock"


@dataclass(frozen=True, slots=True)
class TimeProjectionService:
    """Project time modifiers while preserving the selected calendar date."""

    resolver: ProjectionResolver = resolve_time_slots_with_offsets

    def project(
        self,
        value: Any,
        selected_date: date,
        *,
        config: dict[str, Any] | None = None,
        to_local: Callable[[Any], Any] | None = None,
        seed_base: str = "",
        context: Any | None = None,
    ) -> ProjectionResult:
        if not isinstance(selected_date, date):
            raise TypeError("Time projection requires a selected calendar date.")
        try:
            raw_slots = self.resolver(
                value,
                selected_date,
                config=config,
                to_local=to_local,
                seed_base=seed_base,
                context=context,
            )
            slots = tuple(tuple(slot) for slot in raw_slots)
            if not slots:
                return ProjectionInvalid(
                    selected_date,
                    "time modifier produced no slots",
                    "EmptyProjection",
                )
            return ProjectedTime(selected_date, slots, _source_name(value))
        except OccurrenceSearchExhausted as exc:
            return ProjectionTerminal(selected_date, exc)
        except (LookupError, OSError) as exc:
            return ProjectionUnavailable(selected_date, str(exc) or type(exc).__name__, type(exc).__name__)
        except (TypeError, ValueError) as exc:
            return ProjectionInvalid(selected_date, str(exc) or type(exc).__name__, type(exc).__name__)


__all__ = (
    "ProjectedTime",
    "ProjectionInvalid",
    "ProjectionResult",
    "ProjectionTerminal",
    "ProjectionUnavailable",
    "TimeProjectionService",
)

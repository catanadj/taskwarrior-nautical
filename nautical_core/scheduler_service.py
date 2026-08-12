"""Authoritative occurrence scheduling service boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .evaluation_session import EvaluationSession
from .occurrence_outcomes import OccurrenceOutcome
from .occurrence_provider import Occurrence
from .recurrence_context import RecurrenceContext
from .recurrence_spec import RecurrenceSpec
from .scheduler_cursor import OccurrenceCursor


@dataclass(slots=True)
class SchedulerService:
    """Resolve occurrences through one task-scoped evaluation session."""

    session: EvaluationSession

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, Any],
        *,
        context: RecurrenceContext | None = None,
    ) -> "SchedulerService":
        return cls(EvaluationSession.from_task(task, context=context))

    @property
    def fingerprint(self) -> str:
        return self.session.fingerprint

    def refresh(self, spec: RecurrenceSpec) -> bool:
        return self.session.refresh(spec)

    def next(self, cursor: OccurrenceCursor, **kwargs: Any) -> OccurrenceOutcome:
        return self.session.next_outcome(cursor, **kwargs)

    def collect(
        self,
        cursor: OccurrenceCursor,
        *,
        limit: int,
        **kwargs: Any,
    ) -> list[Occurrence]:
        return self.session.collect_after_cursor(cursor, limit=limit, **kwargs)

    def preview(
        self,
        start: datetime,
        *,
        limit: int = 5,
        inclusive: bool = True,
        timezone: Any | None = None,
        **kwargs: Any,
    ) -> list[Occurrence]:
        cursor = OccurrenceCursor(
            start,
            inclusive=inclusive,
            timezone=timezone,
        )
        return self.collect(cursor, limit=limit, **kwargs)


__all__ = ("SchedulerService",)

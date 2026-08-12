"""Authoritative occurrence scheduling service boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .evaluation_session import EvaluationSession
from .occurrence_outcomes import OccurrenceCollectionResult, OccurrenceOutcome
from .occurrence_provider import OccurrenceBatch
from .occurrence_provider import Occurrence
from .recurrence_context import RecurrenceContext
from .recurrence_spec import RecurrenceSpec
from .scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest


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
    ) -> OccurrenceCollectionResult:
        batch = self.session.collect_after_cursor(cursor, limit=limit, **kwargs)
        if not isinstance(batch, OccurrenceBatch):
            batch = OccurrenceBatch(batch)
        return OccurrenceCollectionResult(
            occurrences=tuple(batch),
            cursor=cursor,
            source=self.session.evaluator.kind or "scheduler",
            terminal=batch.terminal,
        )

    def collect_request(self, request: OccurrenceRangeRequest) -> OccurrenceCollectionResult:
        """Collect one validated range request through this service."""
        if not isinstance(request, OccurrenceRangeRequest):
            raise TypeError("Scheduler collection requires an OccurrenceRangeRequest.")
        context_timezone = self.session.evaluator.context.timezone
        if request.timezone is not None and context_timezone is not None:
            expected = getattr(context_timezone, "key", context_timezone)
            actual = getattr(request.timezone, "key", request.timezone)
            if str(expected) != str(actual):
                raise ValueError("Occurrence range timezone does not match scheduler context.")
        if request.omission_policy != "exclude":
            raise NotImplementedError(
                "The current scheduler collection path supports omission_policy='exclude' only."
            )
        result = self.collect(
            request.cursor,
            limit=request.limit,
            max_iterations=request.max_iterations,
            max_file_skips=request.max_file_skips,
        )
        if request.end_local is not None:
            result = OccurrenceCollectionResult(
                occurrences=tuple(
                    item for item in result.occurrences
                    if item.local_datetime is not None and item.local_datetime <= request.end_local
                ),
                cursor=request.cursor,
                source=result.source,
                terminal=result.terminal,
                request=request,
            )
        else:
            result = OccurrenceCollectionResult(
                occurrences=result.occurrences,
                cursor=request.cursor,
                source=result.source,
                terminal=result.terminal,
                request=request,
            )
        return result

    def preview(
        self,
        start: datetime,
        *,
        limit: int = 5,
        inclusive: bool = True,
        timezone: Any | None = None,
        **kwargs: Any,
    ) -> OccurrenceCollectionResult:
        cursor = OccurrenceCursor(
            start,
            inclusive=inclusive,
            timezone=timezone,
        )
        return self.collect(cursor, limit=limit, **kwargs)


__all__ = ("SchedulerService",)

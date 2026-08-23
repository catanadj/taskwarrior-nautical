"""Authoritative occurrence scheduling service boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .evaluation_session import EvaluationSession
from .occurrence_outcomes import (
    InvalidOccurrence,
    OccurrenceCollectionResult,
    OccurrenceOutcome,
    UnavailableOccurrence,
)
from .occurrence_provider import OccurrenceBatch
from .occurrence_provider import Occurrence
from .recurrence_context import RecurrenceContext
from .recurrence_spec import RecurrenceSpec
from .scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from .scheduler_models import OccurrenceSearchExhausted
from .time_projection import ProjectionResult
from .scheduler_trace import SchedulerTrace, activate
from .task_models import TaskObservation


@dataclass(slots=True)
class SchedulerService:
    """Resolve occurrences through one task-scoped evaluation session."""

    session: EvaluationSession
    trace: SchedulerTrace | None = None

    @classmethod
    def from_observation(
        cls,
        observation: TaskObservation,
        *,
        context: RecurrenceContext | None = None,
        trace: SchedulerTrace | None = None,
    ) -> "SchedulerService":
        trace = trace or SchedulerTrace.from_env()
        return cls(EvaluationSession.from_observation(observation, context=context), trace)

    def _flush_trace(self) -> None:
        if self.trace is not None and self.trace.enabled:
            self.trace.emit()
            self.trace.clear()

    def _trace_scope(self):
        return activate(self.trace)

    def _record_outcome(self, phase: str, outcome: Any, *, cursor: OccurrenceCursor | None = None) -> None:
        if self.trace is None or not self.trace.enabled:
            return
        terminal = getattr(outcome, "terminal_evidence", None)
        occurrence = getattr(outcome, "occurrence", None)
        candidate = getattr(outcome, "local_datetime", None)
        if candidate is None and occurrence is not None:
            candidate = getattr(occurrence, "local_datetime", None)
        self.trace.record(
            phase,
            provider=self.session.evaluator.kind or "scheduler",
            status=getattr(outcome, "status", type(outcome).__name__),
            cursor=cursor.local_datetime if cursor is not None else None,
            candidate=candidate,
            reason=getattr(outcome, "reason", ""),
            terminal=terminal() if callable(terminal) else terminal,
        )

    @property
    def fingerprint(self) -> str:
        return self.session.fingerprint

    def refresh(self, spec: RecurrenceSpec) -> bool:
        return self.session.refresh(spec)

    def next(self, cursor: OccurrenceCursor, **kwargs: Any) -> OccurrenceOutcome:
        try:
            with self._trace_scope():
                outcome = self.session.next_outcome(cursor, **kwargs)
            self._record_outcome("next", outcome, cursor=cursor)
            return outcome
        finally:
            self._flush_trace()

    def select_mode(self, mode: str, **kwargs: Any) -> Any:
        """Select a recurrence-mode successor through the shared session."""
        return self.session.evaluator.select_mode(mode, **kwargs)

    def project_time(self, value: Any, selected_date: Any, **kwargs: Any) -> ProjectionResult:
        """Project ``@t`` on an already selected calendar date."""
        try:
            with self._trace_scope():
                result = self.session.project_time(value, selected_date, **kwargs)
            if self.trace is not None and self.trace.enabled:
                self.trace.record(
                    "projection",
                    provider=self.session.evaluator.kind or "scheduler",
                    status=type(result).__name__,
                    candidate=selected_date,
                )
            return result
        finally:
            self._flush_trace()

    def collect(
        self,
        cursor: OccurrenceCursor,
        *,
        limit: int,
        **kwargs: Any,
    ) -> OccurrenceCollectionResult:
        try:
            # A caller requesting omission-aware evidence needs the event stream;
            # ordinary collection remains on the included-only path.
            with self._trace_scope():
                if "count_omitted" in kwargs:
                    batch = self.session.collect_events_after_cursor(cursor, limit=limit, **kwargs)
                else:
                    batch = self.session.collect_after_cursor(cursor, limit=limit, **kwargs)
            if not isinstance(batch, OccurrenceBatch):
                batch = OccurrenceBatch(batch)
            result = OccurrenceCollectionResult(
                occurrences=tuple(batch),
                cursor=cursor,
                source=self.session.evaluator.kind or "scheduler",
                terminal=batch.terminal,
            )
            if self.trace is not None and self.trace.enabled:
                self.trace.record(
                    "collect",
                    provider=result.source,
                    status=result.status,
                    cursor=cursor.local_datetime,
                    candidate=result.occurrences[0].local_datetime if result.occurrences else None,
                    terminal=(
                        {
                            "kind": result.terminal.kind,
                            "scope": result.terminal.scope,
                            "limit": result.terminal.limit,
                        }
                        if result.terminal is not None else None
                    ),
                )
            return result
        finally:
            self._flush_trace()

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
        include_omitted = request.omission_policy in {"include", "report"}
        count_omitted = request.omission_policy in {"include", "report"}
        try:
            if request.omission_policy == "exclude":
                if request.end_local is not None:
                    batch = self.session.evaluator.events_between(
                        request.cursor.local_datetime,
                        request.end_local,
                        limit=request.limit,
                        inclusive=request.cursor.inclusive,
                        include_omitted=False,
                        max_iterations=request.max_iterations,
                        max_file_skips=request.max_file_skips,
                    )
                    result = OccurrenceCollectionResult(
                        occurrences=tuple(batch),
                        cursor=request.cursor,
                        source=self.session.evaluator.kind or "scheduler",
                        terminal=batch.terminal,
                    )
                else:
                    result = self.collect(
                        request.cursor,
                        limit=request.limit,
                        max_iterations=request.max_iterations,
                        max_file_skips=request.max_file_skips,
                    )
                return OccurrenceCollectionResult(
                    occurrences=result.occurrences,
                    cursor=request.cursor,
                    source=result.source,
                    terminal=result.terminal,
                    request=request,
                )

            if request.end_local is not None:
                batch = self.session.evaluator.events_between(
                    request.cursor.local_datetime,
                    request.end_local,
                    limit=request.limit,
                    inclusive=request.cursor.inclusive,
                    include_omitted=include_omitted,
                    count_omitted=count_omitted,
                    max_iterations=request.max_iterations,
                    max_file_skips=request.max_file_skips,
                )
            else:
                batch = self.session.collect_events_after_cursor(
                    request.cursor,
                    limit=request.limit,
                    count_omitted=count_omitted,
                    max_iterations=request.max_iterations,
                    max_file_skips=request.max_file_skips,
                )
        except OccurrenceSearchExhausted as exc:
            return OccurrenceCollectionResult(
                occurrences=(),
                cursor=request.cursor,
                source=self.session.evaluator.kind or "scheduler",
                terminal=exc,
                request=request,
            )
        except (LookupError, OSError) as exc:
            failure: UnavailableOccurrence | InvalidOccurrence = UnavailableOccurrence(
                str(exc) or "scheduler dependency unavailable",
                type(exc).__name__,
            )
            return OccurrenceCollectionResult(
                occurrences=(),
                cursor=request.cursor,
                source=self.session.evaluator.kind or "scheduler",
                request=request,
                failure=failure,
            )
        except (TypeError, ValueError) as exc:
            failure = InvalidOccurrence(
                str(exc) or "scheduler returned an invalid result",
                type(exc).__name__,
            )
            return OccurrenceCollectionResult(
                occurrences=(),
                cursor=request.cursor,
                source=self.session.evaluator.kind or "scheduler",
                request=request,
                failure=failure,
            )
        events = tuple(batch)
        omitted = tuple(event for event in events if event.omitted)
        occurrences = events if request.omission_policy == "include" else tuple(
            event for event in events if not event.omitted
        )
        return OccurrenceCollectionResult(
            occurrences=occurrences,
            cursor=request.cursor,
            source=self.session.evaluator.kind or "scheduler",
            terminal=batch.terminal,
            request=request,
            omitted_occurrences=omitted if request.omission_policy == "report" else (),
        )

    def included_occurrences_after(
        self,
        after_local_dt: datetime,
        *,
        inclusive: bool,
        limit: int,
    ) -> list[datetime]:
        """Return included local occurrences after one cursor with strict errors."""
        timezone = self.session.evaluator.context.timezone
        result = self.collect_request(
            OccurrenceRangeRequest(
                OccurrenceCursor(
                    after_local_dt,
                    inclusive=inclusive,
                    timezone=timezone,
                ),
                limit=limit,
                omission_policy="exclude",
            )
        )
        if result.failure is not None:
            raise ValueError(f"Anchor occurrence collection {result.status}: {result.failure.reason}")
        if result.terminal is not None and not result.occurrences:
            raise result.terminal
        return [
            occurrence.local_datetime
            for occurrence in result.occurrences
            if occurrence.local_datetime is not None
        ]

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

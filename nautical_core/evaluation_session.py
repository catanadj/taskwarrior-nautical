"""Task-scoped recurrence evaluation session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .compiled_schedule import CompiledSchedule
from .recurrence_context import RecurrenceContext
from .recurrence_evaluator import RecurrenceEvaluator
from .recurrence_spec import RecurrenceSpec
from .occurrence_outcomes import OccurrenceOutcome
from .occurrence_provider import OccurrenceBatch
from .scheduler_cursor import OccurrenceCursor
from .time_projection import ProjectionResult, TimeProjectionService


@dataclass(slots=True)
class EvaluationSession:
    """Own one compiled schedule, evaluator, and bounded task-local state."""

    compiled: CompiledSchedule
    max_cache_entries: int = 32
    _evaluator: RecurrenceEvaluator = field(init=False, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.compiled, CompiledSchedule):
            raise TypeError("evaluation session requires a CompiledSchedule")
        if self.max_cache_entries <= 0:
            raise ValueError("evaluation session cache capacity must be positive")
        self._evaluator = RecurrenceEvaluator.from_compiled(self.compiled)

    @classmethod
    def from_spec(cls, spec: RecurrenceSpec, *, max_cache_entries: int = 32) -> "EvaluationSession":
        return cls(CompiledSchedule.from_spec(spec), max_cache_entries=max_cache_entries)

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, Any],
        *,
        context: RecurrenceContext | None = None,
        max_cache_entries: int = 32,
    ) -> "EvaluationSession":
        spec = RecurrenceSpec.from_task(task, context=context)
        return cls.from_spec(spec, max_cache_entries=max_cache_entries)

    @property
    def evaluator(self) -> RecurrenceEvaluator:
        return self._evaluator

    def next_outcome(self, cursor: OccurrenceCursor, **kwargs: Any) -> OccurrenceOutcome:
        return self._evaluator.next_outcome(cursor, **kwargs)

    def collect_after_cursor(self, cursor: OccurrenceCursor, *, limit: int, **kwargs: Any) -> OccurrenceBatch:
        return self._evaluator.collect_after_cursor(cursor, limit=limit, **kwargs)

    def collect_events_after_cursor(self, cursor: OccurrenceCursor, *, limit: int, **kwargs: Any) -> OccurrenceBatch:
        return self._evaluator.collect_events_after_cursor(cursor, limit=limit, **kwargs)

    def project_time(self, value: Any, selected_date: Any, **kwargs: Any) -> ProjectionResult:
        """Project a time modifier without allowing it to change the date."""
        if kwargs.get("config") is None and self._evaluator.context.astronomy_config is not None:
            kwargs = dict(kwargs)
            kwargs["config"] = dict(self._evaluator.context.astronomy_config)
        service = self.get_or_create("time_projection_service", TimeProjectionService)
        return service.project(value, selected_date, context=self._evaluator.context, **kwargs)

    @property
    def fingerprint(self) -> str:
        return self.compiled.fingerprint

    def get_or_create(self, key: str, factory: Any) -> Any:
        if key not in self._cache:
            if len(self._cache) >= self.max_cache_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = factory()
        return self._cache[key]

    def matches(self, spec: RecurrenceSpec) -> bool:
        return CompiledSchedule.from_spec(spec).fingerprint == self.fingerprint

    def invalidate(self) -> None:
        self._cache.clear()
        self._evaluator = RecurrenceEvaluator.from_compiled(self.compiled)

    def refresh(self, spec: RecurrenceSpec) -> bool:
        """Replace the session when scheduling-affecting state changes."""
        replacement = CompiledSchedule.from_spec(spec)
        if replacement.fingerprint == self.fingerprint:
            return False
        self.compiled = replacement
        self._cache.clear()
        self._evaluator = RecurrenceEvaluator.from_compiled(replacement)
        return True


__all__ = ("EvaluationSession",)

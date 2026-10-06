"""Task-scoped recurrence evaluation session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .compiled_schedule import CompiledSchedule
from .recurrence_context import RecurrenceContext
from .recurrence_evaluator import RecurrenceEvaluator
from .recurrence_spec import RecurrenceSpec
from .occurrence_outcomes import OccurrenceOutcome
from .occurrence_provider import OccurrenceBatch
from .scheduler_cursor import OccurrenceCursor
from .time_projection import ProjectionResult, TimeProjectionService
from .task_models import NauticalTask, TaskObservation
from .recurrence_protocols import PickOccurrenceCallback


@dataclass(slots=True)
class EvaluationSession:
    """Own one compiled schedule, evaluator, and bounded task-local state."""

    compiled: CompiledSchedule
    max_cache_entries: int = 32
    task: NauticalTask | None = None
    _evaluator: RecurrenceEvaluator = field(init=False, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.compiled, CompiledSchedule):
            raise TypeError("evaluation session requires a CompiledSchedule")
        if self.max_cache_entries <= 0:
            raise ValueError("evaluation session cache capacity must be positive")
        self._evaluator = RecurrenceEvaluator.from_compiled(self.compiled)
        if self.task is not None and not isinstance(self.task, NauticalTask):
            raise TypeError("evaluation session task must be a validated NauticalTask")

    @classmethod
    def from_spec(cls, spec: RecurrenceSpec, *, max_cache_entries: int = 32) -> "EvaluationSession":
        return cls(CompiledSchedule.from_spec(spec), max_cache_entries=max_cache_entries)

    @classmethod
    def from_task(
        cls,
        task: NauticalTask,
        *,
        context: RecurrenceContext | None = None,
        max_cache_entries: int = 32,
    ) -> "EvaluationSession":
        if not isinstance(task, NauticalTask):
            raise TypeError("evaluation session requires a validated NauticalTask")
        return cls(
            CompiledSchedule.from_spec(RecurrenceSpec.from_task(task, context=context)),
            task=task,
            max_cache_entries=max_cache_entries,
        )

    @classmethod
    def from_observation(
        cls,
        observation: TaskObservation,
        *,
        context: RecurrenceContext | None = None,
        max_cache_entries: int = 32,
    ) -> "EvaluationSession":
        task = NauticalTask.from_observation(observation)
        return cls.from_task(task, context=context, max_cache_entries=max_cache_entries)

    @property
    def evaluator(self) -> RecurrenceEvaluator:
        return self._evaluator

    def next_outcome(
        self,
        cursor: OccurrenceCursor,
        *,
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        pick_occurrence_local: PickOccurrenceCallback | None = None,
        anchor_file_provider: Any | None = None,
        max_file_skips: int = 512,
    ) -> OccurrenceOutcome:
        return self._evaluator.next_outcome(
            cursor,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_provider=anchor_file_provider,
            max_file_skips=max_file_skips,
        )

    def collect_after_cursor(
        self,
        cursor: OccurrenceCursor,
        *,
        limit: int,
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        pick_occurrence_local: PickOccurrenceCallback | None = None,
        anchor_file_provider: Any | None = None,
        max_iterations: int = 512,
        max_file_skips: int = 512,
    ) -> OccurrenceBatch:
        return self._evaluator.collect_after_cursor(
            cursor,
            limit=limit,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_provider=anchor_file_provider,
            max_iterations=max_iterations,
            max_file_skips=max_file_skips,
        )

    def collect_events_after_cursor(
        self,
        cursor: OccurrenceCursor,
        *,
        limit: int,
        count_omitted: bool = False,
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        pick_occurrence_local: PickOccurrenceCallback | None = None,
        anchor_file_provider: Any | None = None,
        max_iterations: int = 512,
        max_file_skips: int = 512,
    ) -> OccurrenceBatch:
        return self._evaluator.collect_events_after_cursor(
            cursor,
            limit=limit,
            count_omitted=count_omitted,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_provider=anchor_file_provider,
            max_iterations=max_iterations,
            max_file_skips=max_file_skips,
        )

    def project_time(
        self,
        value: Any,
        selected_date: date,
        *,
        config: dict[str, Any] | None = None,
        to_local: Any | None = None,
        seed_base: str = "",
    ) -> ProjectionResult:
        """Project a time modifier without allowing it to change the date."""
        if config is None and self._evaluator.context.astronomy_config is not None:
            config = dict(self._evaluator.context.astronomy_config)
        service = self.get_or_create("time_projection_service", TimeProjectionService)
        return service.project(
            value,
            selected_date,
            config=config,
            to_local=to_local,
            seed_base=seed_base,
            context=self._evaluator.context,
        )

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

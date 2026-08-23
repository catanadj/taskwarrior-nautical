"""Normalized recurrence specification shared by orchestration layers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .recurrence_context import RecurrenceContext
from .task_models import NauticalTask, TaskObservation


@dataclass(frozen=True, slots=True)
class RecurrenceSpec:
    """Immutable recurrence fields paired with their evaluation context."""

    context: RecurrenceContext
    anchor: str = ""
    anchor_file: str = ""
    omit: str = ""
    omit_file: str = ""
    cp: str = ""
    anchor_mode: str = "skip"
    chain_max: int | None = None
    chain_until: str = ""

    @classmethod
    def from_task(
        cls,
        task: NauticalTask,
        *,
        context: RecurrenceContext | None = None,
    ) -> "RecurrenceSpec":
        """Build a specification from an already validated domain task."""
        if not isinstance(task, NauticalTask):
            raise TypeError("recurrence specification requires a validated NauticalTask")
        spec = task.recurrence.spec
        if not isinstance(spec, cls):
            raise TypeError("NauticalTask recurrence does not contain a RecurrenceSpec")
        recurrence_context = context or spec.context
        if recurrence_context.chain_id != task.identity.chain_id.value:
            raise ValueError("Conflicting recurrence identities: context.chain_id does not match task chainID.")
        return replace(
            spec,
            context=recurrence_context,
            anchor_mode=str(spec.anchor_mode or "skip").strip().lower() or "skip",
        )

    @classmethod
    def from_observation(
        cls,
        observation: TaskObservation,
        *,
        context: RecurrenceContext | None = None,
    ) -> "RecurrenceSpec":
        """Build a recurrence specification without thawing a task mapping."""
        if not isinstance(observation, TaskObservation):
            raise TypeError("recurrence specification requires a TaskObservation")
        return cls.from_task(NauticalTask.from_observation(observation), context=context)

    @property
    def kind(self) -> str | None:
        if self.cp:
            return "cp"
        if self.anchor or self.anchor_file:
            return "anchor"
        return None

    @property
    def enabled(self) -> bool:
        return self.kind is not None


__all__ = ("RecurrenceSpec",)

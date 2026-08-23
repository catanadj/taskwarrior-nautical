"""Normalized recurrence specification shared by orchestration layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .recurrence_context import RecurrenceContext
from .task_codec import TaskCodec
from .task_models import FieldPresence, NauticalTask, TaskObservation


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
        task: Mapping[str, Any],
        *,
        context: RecurrenceContext | None = None,
    ) -> "RecurrenceSpec":
        task_chain_id = str(task.get("chainID") or "").strip()
        if context is not None and task_chain_id and context.chain_id != task_chain_id:
            raise ValueError(
                "Conflicting recurrence identities: context.chain_id does not match task.chainID."
            )
        recurrence_context = context or RecurrenceContext.from_task(task)
        chain_max = task.get("chainMax")
        if chain_max in (None, ""):
            normalized_max = None
        else:
            try:
                normalized_max = int(chain_max)
            except (TypeError, ValueError) as exc:
                raise ValueError("chainMax must be an integer in a recurrence specification.") from exc
        return cls(
            context=recurrence_context,
            anchor=TaskCodec.normalize_text(task.get("anchor")),
            anchor_file=TaskCodec.normalize_text(task.get("anchor_file")),
            omit=TaskCodec.normalize_text(task.get("omit")),
            omit_file=TaskCodec.normalize_text(task.get("omit_file")),
            cp=TaskCodec.normalize_text(task.get("cp")),
            anchor_mode=TaskCodec.normalize_text(task.get("anchor_mode") or "skip").lower() or "skip",
            chain_max=normalized_max,
            chain_until=TaskCodec.normalize_text(task.get("chainUntil")),
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
        # Scheduling is an operational path: reject incomplete identity,
        # status, temporal, or recurrence observations before compilation.
        NauticalTask.from_observation(observation)

        def value(name: str) -> object:
            state = observation.field(name)
            if state.presence is FieldPresence.ABSENT:
                return None
            return getattr(state.value, "value", state.value)

        task_chain_id = str(value("chainID") or "").strip()
        if context is not None and task_chain_id and context.chain_id != task_chain_id:
            raise ValueError("Conflicting recurrence identities: context.chain_id does not match task.chainID.")
        recurrence_context = context or RecurrenceContext.from_observation(observation)
        raw_max = value("chainMax")
        if raw_max in (None, ""):
            normalized_max = None
        else:
            try:
                normalized_max = int(raw_max)
            except (TypeError, ValueError) as exc:
                raise ValueError("chainMax must be an integer in a recurrence specification.") from exc
        return cls(
            context=recurrence_context,
            anchor=TaskCodec.normalize_text(value("anchor")),
            anchor_file=TaskCodec.normalize_text(value("anchor_file")),
            omit=TaskCodec.normalize_text(value("omit")),
            omit_file=TaskCodec.normalize_text(value("omit_file")),
            cp=TaskCodec.normalize_text(value("cp")),
            anchor_mode=TaskCodec.normalize_text(value("anchor_mode") or "skip").lower() or "skip",
            chain_max=normalized_max,
            chain_until=TaskCodec.normalize_text(value("chainUntil")),
        )

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

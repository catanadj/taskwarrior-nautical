"""Normalized recurrence specification shared by orchestration layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .recurrence_context import RecurrenceContext


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
            anchor=str(task.get("anchor") or "").strip(),
            anchor_file=str(task.get("anchor_file") or "").strip(),
            omit=str(task.get("omit") or "").strip(),
            omit_file=str(task.get("omit_file") or "").strip(),
            cp=str(task.get("cp") or "").strip(),
            anchor_mode=str(task.get("anchor_mode") or "skip").strip().lower() or "skip",
            chain_max=normalized_max,
            chain_until=str(task.get("chainUntil") or "").strip(),
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

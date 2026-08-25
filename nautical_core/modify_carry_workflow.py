"""Typed decisions for recurring non-completion temporal carry."""

from __future__ import annotations

from dataclasses import dataclass

from .task_models import TaskTimestamp


@dataclass(frozen=True, slots=True)
class TemporalCarryAdjustment:
    field: str
    old_value: TaskTimestamp
    new_value: TaskTimestamp
    offset_seconds: float

    def __post_init__(self) -> None:
        if str(self.field).strip() not in {"due", "scheduled", "wait", "until"}:
            raise ValueError("unsupported temporal carry field")
        if not isinstance(self.old_value, TaskTimestamp) or not isinstance(self.new_value, TaskTimestamp):
            raise TypeError("temporal carry values must be TaskTimestamp values")
        object.__setattr__(self, "field", str(self.field).strip())
        object.__setattr__(self, "offset_seconds", float(self.offset_seconds))


@dataclass(frozen=True, slots=True)
class TemporalCarryDecision:
    status: str
    adjustments: tuple[TemporalCarryAdjustment, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in {"unchanged", "adjusted", "rejected"}:
            raise ValueError("invalid temporal carry status")
        adjustments = tuple(self.adjustments)
        if any(not isinstance(item, TemporalCarryAdjustment) for item in adjustments):
            raise TypeError("temporal carry adjustments must be typed")
        if status == "adjusted" and not adjustments:
            raise ValueError("adjusted carry requires at least one field")
        if status == "rejected" and not str(self.reason).strip():
            raise ValueError("rejected carry requires a reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "adjustments", adjustments)
        object.__setattr__(self, "reason", str(self.reason).strip())


__all__ = ("TemporalCarryAdjustment", "TemporalCarryDecision")

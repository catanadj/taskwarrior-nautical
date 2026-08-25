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

    def __bool__(self) -> bool:
        return self.status == "adjusted"

    @property
    def serialized_changes(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.field, item.new_value.value.isoformat().replace("+00:00", "Z"))
            for item in self.adjustments
        )


def decision_from_cp_adjustments(result, *, timestamp_factory=TaskTimestamp) -> TemporalCarryDecision:
    """Normalize the established CP carry result into the typed decision."""
    if result is None:
        return TemporalCarryDecision("unchanged")
    _old_due, _new_due, adjustments = result
    typed: list[TemporalCarryAdjustment] = []
    for field, old_value, new_value, offset in adjustments:
        typed.append(
            TemporalCarryAdjustment(
                field,
                timestamp_factory(old_value),
                timestamp_factory(new_value),
                offset,
            )
        )
    return TemporalCarryDecision("adjusted", tuple(typed)) if typed else TemporalCarryDecision("unchanged")


__all__ = ("TemporalCarryAdjustment", "TemporalCarryDecision", "decision_from_cp_adjustments")

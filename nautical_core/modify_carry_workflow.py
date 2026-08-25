"""Typed decisions for recurring non-completion temporal carry."""

from __future__ import annotations

from dataclasses import dataclass

from .task_changes import TaskPatch
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
class NativeUntilDecision:
    status: str
    value: TaskTimestamp | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in {"unchanged", "carried", "rejected"}:
            raise ValueError("invalid native-until decision status")
        if status == "carried" and self.value is None:
            raise ValueError("carried native-until decision requires a value")
        if status == "rejected" and not str(self.reason).strip():
            raise ValueError("rejected native-until decision requires a reason")
        if self.value is not None and not isinstance(self.value, TaskTimestamp):
            raise TypeError("native-until decision value must be a TaskTimestamp")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", str(self.reason).strip())


@dataclass(frozen=True, slots=True)
class TemporalCarryDecision:
    status: str
    adjustments: tuple[TemporalCarryAdjustment, ...] = ()
    reason: str = ""
    target_old: TaskTimestamp | None = None
    target_new: TaskTimestamp | None = None

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in {"unchanged", "adjusted", "rejected"}:
            raise ValueError("invalid temporal carry status")
        adjustments = tuple(self.adjustments)
        if any(not isinstance(item, TemporalCarryAdjustment) for item in adjustments):
            raise TypeError("temporal carry adjustments must be typed")
        if (self.target_old is None) != (self.target_new is None):
            raise ValueError("temporal carry target must include both values")
        if self.target_old is not None and not isinstance(self.target_old, TaskTimestamp):
            raise TypeError("temporal carry target must be a TaskTimestamp")
        if self.target_new is not None and not isinstance(self.target_new, TaskTimestamp):
            raise TypeError("temporal carry target must be a TaskTimestamp")
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
                offset.total_seconds() if hasattr(offset, "total_seconds") else offset,
            )
        )
    old_target = timestamp_factory(_old_due)
    new_target = timestamp_factory(_new_due)
    return (
        TemporalCarryDecision("adjusted", tuple(typed), target_old=old_target, target_new=new_target)
        if typed
        else TemporalCarryDecision("unchanged", target_old=old_target, target_new=new_target)
    )


def apply_temporal_carry_patch(task: dict, decision: TemporalCarryDecision) -> None:
    """Apply a validated carry decision as one ordinary-carry patch."""
    if decision.status != "adjusted":
        return
    uuid = str(task.get("uuid") or "").strip()
    from .task_models import TaskUUID

    patch = TaskPatch.ordinary_carry(
        TaskUUID(uuid),
        **dict(decision.serialized_changes),
    )
    for field, value in patch.set_values().items():
        task[field] = value


def apply_native_until_patch(task: dict, decision: NativeUntilDecision) -> None:
    """Apply a carried native-until value through the same typed patch path."""
    if decision.status != "carried" or decision.value is None:
        return
    from .task_models import TaskUUID

    patch = TaskPatch.ordinary_carry(TaskUUID(str(task.get("uuid") or "")), until=decision.value.value.isoformat().replace("+00:00", "Z"))
    for field, value in patch.set_values().items():
        task[field] = value


__all__ = (
    "NativeUntilDecision",
    "TemporalCarryAdjustment",
    "TemporalCarryDecision",
    "apply_temporal_carry_patch",
    "decision_from_cp_adjustments",
)

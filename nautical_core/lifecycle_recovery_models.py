"""Typed results produced by lifecycle recovery planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .lifecycle_models import LifecyclePlan
from .task_models import TaskObservation


class RecoveryStatus(str, Enum):
    """Non-plan outcomes which must not be treated as safe mutations."""

    STALE = "stale"
    RETRYABLE = "retryable"
    PARTIAL = "partial"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RecoveryPlanResult:
    """A validated lifecycle plan and its recovery execution facts."""

    parent: TaskObservation
    plan: LifecyclePlan
    reason: str = ""
    child_short: str = ""
    child_due: Any = None
    child_observation: TaskObservation | None = None
    terminal_kind: str | None = None
    applied: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.parent, TaskObservation):
            raise TypeError("recovery plan result requires a TaskObservation parent")
        if not isinstance(self.plan, LifecyclePlan):
            raise TypeError("recovery plan result requires a LifecyclePlan")
        if self.child_observation is not None and not isinstance(self.child_observation, TaskObservation):
            raise TypeError("recovery plan child evidence requires a TaskObservation")
        reason = str(self.reason or "").strip()
        child_short = str(self.child_short or "").strip()
        if len(child_short) > 64:
            raise ValueError("recovery child identity is too long")
        if not isinstance(self.applied, bool):
            raise TypeError("recovery plan result applied must be boolean")
        terminal_kind = None if self.terminal_kind in (None, "") else str(self.terminal_kind).strip()
        if terminal_kind and terminal_kind not in {"date_limit", "search_limit", "chain_max", "chain_until"}:
            raise ValueError("invalid recovery terminal kind")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "child_short", child_short)
        object.__setattr__(self, "terminal_kind", terminal_kind)


@dataclass(frozen=True, slots=True)
class RecoveryRefusal:
    """A typed refusal or unavailable result with no mutation-capable plan."""

    parent: TaskObservation
    status: RecoveryStatus
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parent, TaskObservation):
            raise TypeError("recovery refusal requires a TaskObservation parent")
        try:
            status = RecoveryStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid recovery refusal status") from exc
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("recovery refusal reason is required")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("recovery refusal evidence must be a mapping")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


RecoveryResult = RecoveryPlanResult | RecoveryRefusal


__all__ = ["RecoveryPlanResult", "RecoveryRefusal", "RecoveryResult", "RecoveryStatus"]

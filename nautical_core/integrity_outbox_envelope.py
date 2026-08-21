"""Typed integrity payload variant for Nautical's shared outbox."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from .chain_integrity_models import IntegrityRepairPlan
from .lifecycle_models import ExecutionStage
from .lifecycle_outbox import OutboxProcessingState


class OutboxWorkKind(str, Enum):
    LIFECYCLE = "lifecycle"
    INTEGRITY = "integrity"


@dataclass(frozen=True, slots=True)
class IntegrityOutboxEnvelope:
    """One validated integrity plan in the same durable work vocabulary."""

    plan: IntegrityRepairPlan
    configuration_fingerprint: str
    schedule_fingerprint: str
    stage: ExecutionStage = ExecutionStage.PLANNED
    state: OutboxProcessingState = OutboxProcessingState.READY

    def __post_init__(self) -> None:
        if not isinstance(self.plan, IntegrityRepairPlan):
            raise TypeError("integrity envelope requires an IntegrityRepairPlan")
        config = str(self.configuration_fingerprint or "").strip()
        schedule = str(self.schedule_fingerprint or "").strip()
        if not config or not schedule:
            raise ValueError("integrity envelope requires configuration and schedule fingerprints")
        stage = ExecutionStage(self.stage)
        state = OutboxProcessingState(self.state)
        if stage is not ExecutionStage.PLANNED:
            raise ValueError("new integrity envelope must begin at planned stage")
        if state not in {OutboxProcessingState.READY, OutboxProcessingState.RETRY}:
            raise ValueError("new integrity envelope must be ready or retryable")
        object.__setattr__(self, "configuration_fingerprint", config)
        object.__setattr__(self, "schedule_fingerprint", schedule)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "state", state)

    @property
    def work_kind(self) -> OutboxWorkKind:
        return OutboxWorkKind.INTEGRITY

    @property
    def intent_id(self) -> str:
        return f"integrity:{self.plan.plan_id}"

    def to_dict(self) -> dict[str, object]:
        return {
            "work_kind": self.work_kind.value,
            "intent_id": self.intent_id,
            "plan": self.plan.to_dict(),
            "configuration_fingerprint": self.configuration_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "stage": self.stage.value,
            "state": self.state.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "IntegrityOutboxEnvelope":
        if not isinstance(value, dict) or value.get("work_kind") != OutboxWorkKind.INTEGRITY.value:
            raise ValueError("not an integrity outbox envelope")
        plan_value = value.get("plan")
        if not isinstance(plan_value, dict):
            raise ValueError("integrity envelope requires a plan object")
        envelope = cls(
            IntegrityRepairPlan.from_dict(plan_value),
            str(value.get("configuration_fingerprint") or ""),
            str(value.get("schedule_fingerprint") or ""),
            value.get("stage", ExecutionStage.PLANNED.value),
            value.get("state", OutboxProcessingState.READY.value),
        )
        if str(value.get("intent_id") or "") != envelope.intent_id:
            raise ValueError("integrity envelope intent ID does not match plan")
        return envelope


__all__ = ["IntegrityOutboxEnvelope", "OutboxWorkKind"]

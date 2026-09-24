"""Typed, redacted evidence for guided manual-review workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


def _short(value: object) -> str:
    return str(value or "").strip()[:8]


class ManualReviewAction(str, Enum):
    ACCEPT_CONNECTED = "accept-connected"
    RETRY = "retry"
    REPLAN = "replan"
    RESOLVE_APPLIED = "resolve-applied"
    QUARANTINE = "quarantine"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class ManualReviewEvidence:
    chain_id: str
    source_link: int | None
    target_link: int | None
    parent_uuid: str
    expected_child_uuid: str
    occupants: tuple[str, ...] = ()
    reason: str = ""
    snapshot_status: str = "complete"

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", str(self.chain_id or "").strip())
        object.__setattr__(self, "parent_uuid", str(self.parent_uuid or "").strip())
        object.__setattr__(self, "expected_child_uuid", str(self.expected_child_uuid or "").strip())
        object.__setattr__(self, "occupants", tuple(str(item).strip() for item in self.occupants if str(item).strip()))
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        object.__setattr__(self, "snapshot_status", str(self.snapshot_status or "complete").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "source_link": self.source_link,
            "target_link": self.target_link,
            "parent_uuid": _short(self.parent_uuid),
            "expected_child_uuid": _short(self.expected_child_uuid),
            "occupants": [_short(item) for item in self.occupants],
            "reason": self.reason,
            "snapshot_status": self.snapshot_status,
        }


@dataclass(frozen=True, slots=True)
class ManualReviewItem:
    intent_id: str
    state: str
    evidence: ManualReviewEvidence
    actions: tuple[ManualReviewAction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", str(self.intent_id or "").strip())
        object.__setattr__(self, "state", str(self.state or "").strip())
        object.__setattr__(self, "actions", tuple(ManualReviewAction(item) for item in self.actions))

    @classmethod
    def from_evidence(cls, intent_id: str, state: str, evidence: ManualReviewEvidence) -> "ManualReviewItem":
        # Only expose actions currently backed by guarded implementations.
        # Replan/retry/branch acceptance remain intentionally hidden until
        # their mutation preconditions are wired through the review command.
        actions = [ManualReviewAction.RESOLVE_APPLIED, ManualReviewAction.SKIP]
        return cls(intent_id, state, evidence, tuple(actions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "state": self.state,
            "evidence": self.evidence.to_dict(),
            "actions": [action.value for action in self.actions],
        }


@dataclass(frozen=True, slots=True)
class ManualReviewUnavailable:
    intent_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"status": "unavailable", "intent_id": self.intent_id, "reason": self.reason}


__all__ = ["ManualReviewAction", "ManualReviewEvidence", "ManualReviewItem", "ManualReviewUnavailable"]

"""Typed, immutable contracts for Nautical lifecycle transitions.

This module deliberately contains no Taskwarrior, SQLite, or presentation
logic.  It describes a transition and its durable execution state so planners,
hooks, and reconcile can share one vocabulary without sharing mutation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias


class LifecycleContractError(ValueError):
    """Raised when a lifecycle model violates a transition invariant."""


class LifecycleEvent(str, Enum):
    ACTIVATE = "activate"
    RESUME = "resume"
    DISABLE = "disable"
    COMPLETE = "complete"
    EXPIRE = "expire"
    MANUAL_DELETE = "manual_delete"
    CHAIN_MAX = "chain_max"
    CHAIN_UNTIL = "chain_until"


class LifecycleAction(str, Enum):
    NOOP = "noop"
    UPDATE_PARENT = "update_parent"
    SPAWN_CHILD = "spawn_child"
    DISABLE_CHAIN = "disable_chain"
    FINALIZE_CHAIN = "finalize_chain"
    RETRY = "retry"
    MANUAL_REVIEW = "manual_review"


class ExecutionStage(str, Enum):
    PLANNED = "planned"
    PERSISTED = "persisted"
    CHILD_PRESENT = "child_present"
    PARENT_LINKED = "parent_linked"
    VERIFIED = "verified"
    FINALIZED = "finalized"
    RETRYABLE = "retryable"
    MANUAL_REVIEW = "manual_review"


class LifecycleOutcomeKind(str, Enum):
    APPLIED = "applied"
    NOOP = "noop"
    TERMINAL = "terminal"
    RETRYABLE = "retryable"
    MANUAL_REVIEW = "manual_review"


class TaskLifecycleState(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    TERMINAL = "terminal"


class QueueProcessingState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    FINALIZED = "finalized"
    DEAD_LETTERED = "dead_lettered"


FrozenValue: TypeAlias = Any
FrozenPairs: TypeAlias = tuple[tuple[str, FrozenValue], ...]


def _freeze(value: Any) -> FrozenValue:
    """Convert supported payload containers to immutable deterministic values."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def _freeze_pairs(value: Mapping[str, Any] | None) -> FrozenPairs:
    if not value:
        return ()
    return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))


def _thaw(value: FrozenValue) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ParentGuard:
    """The parent facts that must still hold before a transition mutates it."""

    status: str
    chain: str
    chain_id: str
    link: int
    recurrence_fingerprint: str = ""

    def __post_init__(self) -> None:
        for name in ("status", "chain", "chain_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise LifecycleContractError(f"parent guard requires {name}")
            object.__setattr__(self, name, value)
        if isinstance(self.link, bool) or not isinstance(self.link, int) or self.link < 0:
            raise LifecycleContractError("parent guard link must be a non-negative integer")
        object.__setattr__(self, "recurrence_fingerprint", str(self.recurrence_fingerprint or "").strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParentGuard":
        if not isinstance(value, Mapping):
            raise LifecycleContractError("parent guard must be an object")
        try:
            link = int(value["link"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LifecycleContractError("parent guard requires an integer link") from exc
        return cls(
            status=str(value.get("status", "")),
            chain=str(value.get("chain", "")),
            chain_id=str(value.get("chainID", value.get("chain_id", ""))),
            link=link,
            recurrence_fingerprint=str(value.get("recurrence_fingerprint", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "chain": self.chain,
            "chainID": self.chain_id,
            "link": self.link,
        }
        if self.recurrence_fingerprint:
            result["recurrence_fingerprint"] = self.recurrence_fingerprint
        return result


@dataclass(frozen=True, slots=True)
class LifecycleIdentity:
    """Stable identity for one parent-to-slot lifecycle transition."""

    chain_id: str
    parent_uuid: str
    source_link: int
    target_link: int | None
    event: LifecycleEvent

    def __post_init__(self) -> None:
        chain_id = str(self.chain_id).strip()
        parent_uuid = str(self.parent_uuid).strip()
        if not chain_id:
            raise LifecycleContractError("chainID is mandatory for lifecycle transitions")
        if not parent_uuid:
            raise LifecycleContractError("parent UUID is mandatory for lifecycle transitions")
        if isinstance(self.source_link, bool) or not isinstance(self.source_link, int) or self.source_link < 0:
            raise LifecycleContractError("source link must be a non-negative integer")
        target = self.target_link
        if target is not None and (
            isinstance(target, bool) or not isinstance(target, int) or target <= self.source_link
        ):
            raise LifecycleContractError("target link must be greater than source link")
        try:
            event = LifecycleEvent(self.event)
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError(f"unsupported lifecycle event: {self.event!r}") from exc
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "parent_uuid", parent_uuid)
        object.__setattr__(self, "event", event)

    @property
    def key(self) -> str:
        target = "-" if self.target_link is None else str(self.target_link)
        return f"{self.chain_id}:{self.parent_uuid}:{self.source_link}:{target}:{self.event.value}"


_EVENT_ACTIONS: dict[LifecycleEvent, frozenset[LifecycleAction]] = {
    LifecycleEvent.ACTIVATE: frozenset({LifecycleAction.UPDATE_PARENT, LifecycleAction.NOOP}),
    LifecycleEvent.RESUME: frozenset({LifecycleAction.UPDATE_PARENT, LifecycleAction.NOOP}),
    LifecycleEvent.DISABLE: frozenset({LifecycleAction.DISABLE_CHAIN, LifecycleAction.NOOP}),
    LifecycleEvent.COMPLETE: frozenset({LifecycleAction.SPAWN_CHILD, LifecycleAction.FINALIZE_CHAIN, LifecycleAction.NOOP}),
    LifecycleEvent.EXPIRE: frozenset({LifecycleAction.SPAWN_CHILD, LifecycleAction.FINALIZE_CHAIN, LifecycleAction.NOOP}),
    LifecycleEvent.MANUAL_DELETE: frozenset({LifecycleAction.DISABLE_CHAIN, LifecycleAction.FINALIZE_CHAIN, LifecycleAction.NOOP}),
    LifecycleEvent.CHAIN_MAX: frozenset({LifecycleAction.FINALIZE_CHAIN, LifecycleAction.NOOP}),
    LifecycleEvent.CHAIN_UNTIL: frozenset({LifecycleAction.FINALIZE_CHAIN, LifecycleAction.NOOP}),
}


@dataclass(frozen=True, slots=True)
class LifecyclePlan:
    """Complete side-effect-free description of one lifecycle transition."""

    identity: LifecycleIdentity
    action: LifecycleAction
    parent_guard: ParentGuard
    stage: ExecutionStage = ExecutionStage.PLANNED
    child_payload: FrozenPairs = ()
    parent_patch: FrozenPairs = ()
    expected_postconditions: tuple[str, ...] = ()
    max_attempts: int = 3

    def __post_init__(self) -> None:
        try:
            action = LifecycleAction(self.action)
            stage = ExecutionStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("invalid lifecycle plan action or stage") from exc
        allowed = _EVENT_ACTIONS[self.identity.event]
        if action not in allowed and action not in {LifecycleAction.RETRY, LifecycleAction.MANUAL_REVIEW}:
            raise LifecycleContractError(
                f"action {action.value!r} is invalid for event {self.identity.event.value!r}"
            )
        if self.identity.chain_id != self.parent_guard.chain_id:
            raise LifecycleContractError("plan identity and parent guard chainID differ")
        if self.identity.source_link != self.parent_guard.link:
            raise LifecycleContractError("plan identity and parent guard link differ")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise LifecycleContractError("max_attempts must be a positive integer")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "child_payload", tuple(self.child_payload))
        object.__setattr__(self, "parent_patch", tuple(self.parent_patch))
        object.__setattr__(self, "expected_postconditions", tuple(str(item) for item in self.expected_postconditions))

    @classmethod
    def from_mappings(
        cls,
        *,
        identity: LifecycleIdentity,
        action: LifecycleAction,
        parent_guard: ParentGuard,
        child_payload: Mapping[str, Any] | None = None,
        parent_patch: Mapping[str, Any] | None = None,
        expected_postconditions: tuple[str, ...] = (),
        max_attempts: int = 3,
    ) -> "LifecyclePlan":
        return cls(
            identity=identity,
            action=action,
            parent_guard=parent_guard,
            child_payload=_freeze_pairs(child_payload),
            parent_patch=_freeze_pairs(parent_patch),
            expected_postconditions=expected_postconditions,
            max_attempts=max_attempts,
        )

    def child_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.child_payload}

    def parent_patch_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.parent_patch}


@dataclass(frozen=True, slots=True)
class LifecycleOutcome:
    """Tagged result returned by a lifecycle executor or recovery pass."""

    kind: LifecycleOutcomeKind
    stage: ExecutionStage
    identity: LifecycleIdentity
    reason: str = ""

    def __post_init__(self) -> None:
        try:
            kind = LifecycleOutcomeKind(self.kind)
            stage = ExecutionStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("invalid lifecycle outcome kind or stage") from exc
        required_stage = {
            LifecycleOutcomeKind.RETRYABLE: ExecutionStage.RETRYABLE,
            LifecycleOutcomeKind.MANUAL_REVIEW: ExecutionStage.MANUAL_REVIEW,
            LifecycleOutcomeKind.APPLIED: ExecutionStage.FINALIZED,
            LifecycleOutcomeKind.TERMINAL: ExecutionStage.FINALIZED,
        }.get(kind)
        if required_stage is not None and stage is not required_stage:
            raise LifecycleContractError(f"{kind.value} outcome requires {required_stage.value} stage")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "reason", str(self.reason or "").strip())


__all__ = (
    "ExecutionStage",
    "LifecycleAction",
    "LifecycleContractError",
    "LifecycleEvent",
    "LifecycleIdentity",
    "LifecycleOutcome",
    "LifecycleOutcomeKind",
    "LifecyclePlan",
    "ParentGuard",
    "QueueProcessingState",
    "TaskLifecycleState",
)

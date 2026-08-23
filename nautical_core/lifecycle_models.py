"""Typed, immutable contracts for Nautical lifecycle transitions.

This module deliberately contains no Taskwarrior, SQLite, or presentation
logic.  It describes a transition and its durable execution state so planners,
hooks, and reconcile can share one vocabulary without sharing mutation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping, TypeAlias

from .task_models import FieldPresence, TaskObservation


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


@dataclass(frozen=True, slots=True)
class LifecycleRecoveryDecision:
    """Typed successor/expiration decision consumed by reconcile and UI."""

    action: str
    parent: TaskObservation
    next_link: int
    reason: str
    child: dict[str, Any] | None = None
    child_short: str = ""
    child_due: Any = None
    terminal_kind: str | None = None
    lifecycle_plan: "LifecyclePlan | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent, TaskObservation):
            raise TypeError("lifecycle recovery decision requires a TaskObservation parent")


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


class DeletionDisposition(str, Enum):
    """Evidence-based classification for a deleted Nautical occurrence."""

    NOT_APPLICABLE = "not_applicable"
    EXPIRATION = "expiration"
    MANUAL = "manual"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class DeletionEvidence:
    disposition: DeletionDisposition
    reason: str = ""

    def __post_init__(self) -> None:
        try:
            disposition = DeletionDisposition(self.disposition)
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("invalid deletion disposition") from exc
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason", str(self.reason or "").strip())


FrozenValue: TypeAlias = Any
FrozenPairs: TypeAlias = tuple[tuple[str, FrozenValue], ...]
LIFECYCLE_PLAN_SCHEMA_VERSION = 1


# These are the inputs that can change the occurrence or carried timing of a
# lifecycle successor.  Presentation-only edits (description, project, value,
# and similar fields) intentionally do not invalidate a queued transition.
_RECURRENCE_FINGERPRINT_FIELDS = (
    "anchor",
    "anchor_file",
    "omit",
    "omit_file",
    "cp",
    "anchor_mode",
    "chainMax",
    "chainUntil",
    "bc",
    "due",
    "scheduled",
    "until",
    "wait",
)
_RECURRENCE_DATETIME_FIELDS = frozenset({"chainUntil", "due", "scheduled", "until", "wait"})
_TASKWARRIOR_DATETIME_RE = re.compile(r"^(\d{8})T(\d{6})(Z|[+-]\d{4})$")
_PLAN_DATETIME_FIELDS = frozenset({"due", "scheduled", "until", "wait", "chainUntil"})
_PLAN_VOLATILE_CHILD_FIELDS = frozenset(
    {"id", "entry", "modified", "urgency", "status", "end", "start", "nextLink", "mask", "imask", "parent", "recur", "rc"}
)


def _canonical_datetime_text(value: Any, parse_datetime: Callable[[Any], Any] | None) -> str:
    """Return one stable representation without making parsing mandatory."""
    if parse_datetime is not None:
        try:
            parsed = parse_datetime(value)
        except Exception:
            parsed = None
        if isinstance(parsed, datetime):
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    text = str(value).strip()
    match = _TASKWARRIOR_DATETIME_RE.fullmatch(text)
    if match:
        try:
            raw_date, raw_time, zone = match.groups()
            parsed = datetime.strptime(raw_date + raw_time, "%Y%m%d%H%M%S")
            if zone == "Z":
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                sign = 1 if zone[0] == "+" else -1
                offset_minutes = sign * (int(zone[1:3]) * 60 + int(zone[3:5]))
                parsed = parsed.replace(tzinfo=timezone(timedelta(minutes=offset_minutes)))
            return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        except (OverflowError, ValueError):
            pass
    return re.sub(r"\s+", " ", text)


def _canonical_recurrence_value(
    field: str,
    value: Any,
    parse_datetime: Callable[[Any], Any] | None,
) -> Any:
    if value is None or value == "":
        return None
    if field in _RECURRENCE_DATETIME_FIELDS:
        return _canonical_datetime_text(value, parse_datetime)
    if field == "anchor_mode":
        return str(value).strip().lower()
    if field == "chainMax":
        try:
            return int(value)
        except (TypeError, ValueError):
            return re.sub(r"\s+", " ", str(value).strip())
    if isinstance(value, (list, tuple)):
        return [re.sub(r"\s+", " ", str(item).strip()) for item in value]
    return re.sub(r"\s+", " ", str(value).strip())


def _canonical_plan_datetime(value: Any) -> Any:
    """Canonicalize Taskwarrior and ISO timestamps for intent comparison."""
    normalized = _canonical_datetime_text(value, None)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return normalized
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_plan_value(value: Any) -> Any:
    """Normalize JSON-compatible values before comparing immutable plans."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_plan_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_plan_value(item) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Taskwarrior commonly serializes numeric UDAs as floats while plans
        # built in Python naturally contain integers.
        return float(value)
    return value


def recurrence_fingerprint(
    task: Mapping[str, Any],
    *,
    parse_datetime: Callable[[Any], Any] | None = None,
    extra_fields: Iterable[str] = (),
) -> str:
    """Hash canonical recurrence inputs for stale-transition protection.

    The version prefix permits future field-set changes to invalidate old
    guards while keeping formatting-only datetime and mode differences stable.
    """
    fields = tuple(dict.fromkeys((*_RECURRENCE_FINGERPRINT_FIELDS, *(str(item) for item in extra_fields))))
    canonical: dict[str, Any] = {}
    for field in fields:
        if field not in task:
            continue
        value = _canonical_recurrence_value(field, task.get(field), parse_datetime)
        if value is not None:
            canonical[field] = value
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"rf1-{digest}"


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
    modified: str = ""

    def __post_init__(self) -> None:
        for name in ("status", "chain", "chain_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise LifecycleContractError(f"parent guard requires {name}")
            object.__setattr__(self, name, value)
        if isinstance(self.link, bool) or not isinstance(self.link, int) or self.link < 0:
            raise LifecycleContractError("parent guard link must be a non-negative integer")
        object.__setattr__(self, "recurrence_fingerprint", str(self.recurrence_fingerprint or "").strip())
        object.__setattr__(self, "modified", str(self.modified or "").strip())

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
            modified=str(value.get("modified", "")),
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
        if self.modified:
            result["modified"] = self.modified
        return result


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """Immutable Taskwarrior row supplied to lifecycle planning."""

    observation: TaskObservation

    @classmethod
    def from_observation(cls, value: TaskObservation) -> "TaskSnapshot":
        if not isinstance(value, TaskObservation):
            raise LifecycleContractError("task snapshot requires a TaskObservation")
        return cls(value)

    def get(self, key: str, default: Any = None) -> Any:
        state = self.observation.field(key)
        if state.presence is FieldPresence.ABSENT:
            return default
        return state.raw_value()

    def to_dict(self) -> dict[str, Any]:
        return self.observation.to_mapping()


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LifecycleIdentity":
        if not isinstance(value, Mapping):
            raise LifecycleContractError("lifecycle identity must be an object")
        target = value.get("target_link", value.get("targetLink"))
        return cls(
            chain_id=str(value.get("chainID", value.get("chain_id", ""))),
            parent_uuid=str(value.get("parent_uuid", value.get("parentUUID", ""))),
            source_link=int(value.get("source_link", value.get("sourceLink"))),
            target_link=None if target in (None, "") else int(target),
            event=LifecycleEvent(value.get("event")),
        )

    @property
    def key(self) -> str:
        target = "-" if self.target_link is None else str(self.target_link)
        return f"{self.chain_id}:{self.parent_uuid}:{self.source_link}:{target}:{self.event.value}"

    @property
    def idempotency_key(self) -> str:
        """Compact durable key for one transition, independent of a retry."""
        digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:24]
        return f"li1-{digest}"


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
    terminal_kind: str | None = None

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
        terminal_kind = None if self.terminal_kind in (None, "") else str(self.terminal_kind).strip()
        if terminal_kind is not None and terminal_kind not in {
            "date_limit", "search_limit", "chain_max", "chain_until",
        }:
            raise LifecycleContractError("invalid lifecycle terminal kind")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "child_payload", tuple(self.child_payload))
        object.__setattr__(self, "parent_patch", tuple(self.parent_patch))
        object.__setattr__(self, "expected_postconditions", tuple(str(item) for item in self.expected_postconditions))
        object.__setattr__(self, "terminal_kind", terminal_kind)

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
        stage: ExecutionStage = ExecutionStage.PLANNED,
        terminal_kind: str | None = None,
    ) -> "LifecyclePlan":
        return cls(
            identity=identity,
            action=action,
            parent_guard=parent_guard,
            stage=stage,
            child_payload=_freeze_pairs(child_payload),
            parent_patch=_freeze_pairs(parent_patch),
            expected_postconditions=expected_postconditions,
            max_attempts=max_attempts,
            terminal_kind=terminal_kind,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize a complete plan for the durable lifecycle outbox."""
        return {
            "schema_version": LIFECYCLE_PLAN_SCHEMA_VERSION,
            "identity": {
                "chainID": self.identity.chain_id,
                "parent_uuid": self.identity.parent_uuid,
                "source_link": self.identity.source_link,
                "target_link": self.identity.target_link,
                "event": self.identity.event.value,
            },
            "action": self.action.value,
            "parent_guard": self.parent_guard.to_dict(),
            "stage": self.stage.value,
            "child_payload": self.child_dict(),
            "parent_patch": self.parent_patch_dict(),
            "expected_postconditions": list(self.expected_postconditions),
            "max_attempts": self.max_attempts,
            "terminal_kind": self.terminal_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LifecyclePlan":
        """Load one supported plan version; future versions fail closed."""
        if not isinstance(value, Mapping):
            raise LifecycleContractError("lifecycle plan must be an object")
        try:
            version = int(value.get("schema_version"))
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("lifecycle plan schema_version is required") from exc
        if version != LIFECYCLE_PLAN_SCHEMA_VERSION:
            raise LifecycleContractError(
                f"unsupported lifecycle plan schema version: {version}"
            )
        child_payload = value.get("child_payload") or {}
        parent_patch = value.get("parent_patch") or {}
        expected = value.get("expected_postconditions") or ()
        if not isinstance(child_payload, Mapping) or not isinstance(parent_patch, Mapping):
            raise LifecycleContractError("lifecycle plan payloads must be objects")
        if not isinstance(expected, (list, tuple)):
            raise LifecycleContractError("lifecycle plan postconditions must be a list")
        try:
            stage = ExecutionStage(value.get("stage", ExecutionStage.PLANNED.value))
            action = LifecycleAction(value.get("action"))
            max_attempts = int(value.get("max_attempts", 3))
            identity = LifecycleIdentity.from_mapping(value.get("identity") or {})
            parent_guard = ParentGuard.from_mapping(value.get("parent_guard") or {})
        except (TypeError, ValueError, KeyError) as exc:
            raise LifecycleContractError("invalid lifecycle plan fields") from exc
        return cls.from_mappings(
            identity=identity,
            action=action,
            parent_guard=parent_guard,
            stage=stage,
            child_payload=child_payload,
            parent_patch=parent_patch,
            expected_postconditions=tuple(str(item) for item in expected),
            max_attempts=max_attempts,
            terminal_kind=value.get("terminal_kind"),
        )

    def child_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.child_payload}

    def parent_patch_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.parent_patch}

    def semantic_key(self) -> str:
        """Return a stable comparison key excluding durable execution stage."""
        payload = self.to_dict()
        payload.pop("stage", None)
        payload.pop("max_attempts", None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))

    def compatibility_key(self) -> str:
        """Compare plans while tolerating legacy literal-null recurrence UDAs."""
        payload = self.compatibility_payload()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))

    def compatibility_payload(self) -> dict[str, Any]:
        """Return the immutable intent fields used for safe replay comparison."""
        payload = _canonical_plan_value(self.to_dict())
        payload.pop("stage", None)
        payload.pop("max_attempts", None)
        child = payload.get("child_payload")
        if isinstance(child, dict):
            for field in _PLAN_VOLATILE_CHILD_FIELDS:
                child.pop(field, None)
            for field in _PLAN_DATETIME_FIELDS:
                if field in child:
                    child[field] = _canonical_plan_datetime(child[field])
            optional_fields = ("anchor", "anchor_file", "omit", "omit_file", "cp", "chainMax", "chainUntil", "bc")
            for field in optional_fields:
                value = child.get(field)
                if value is None or (isinstance(value, str) and value.strip().casefold() in {"", "null"}):
                    child.pop(field, None)
            if child.get("anchor_mode") is None or (
                isinstance(child.get("anchor_mode"), str)
                and child["anchor_mode"].strip().casefold() == "null"
            ):
                child["anchor_mode"] = "skip"
        guard = payload.get("parent_guard")
        if isinstance(guard, dict):
            guard.pop("modified", None)
        return payload

    def with_stage(self, stage: ExecutionStage) -> "LifecyclePlan":
        """Return the same immutable plan at a new durable execution stage."""
        return LifecyclePlan(
            identity=self.identity,
            action=self.action,
            parent_guard=self.parent_guard,
            stage=stage,
            child_payload=self.child_payload,
            parent_patch=self.parent_patch,
            expected_postconditions=self.expected_postconditions,
            max_attempts=self.max_attempts,
        )


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
    "DeletionDisposition",
    "DeletionEvidence",
    "ExecutionStage",
    "LIFECYCLE_PLAN_SCHEMA_VERSION",
    "LifecycleAction",
    "LifecycleContractError",
    "LifecycleEvent",
    "LifecycleIdentity",
    "LifecycleOutcome",
    "LifecycleOutcomeKind",
    "LifecyclePlan",
    "LifecycleRecoveryDecision",
    "ParentGuard",
    "TaskLifecycleState",
    "TaskSnapshot",
    "recurrence_fingerprint",
)

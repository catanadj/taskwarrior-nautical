"""Pure lifecycle transition planning.

The planner creates a complete contract object but performs no Taskwarrior,
SQLite, queue, panel, or input mutation.  External recurrence/child work is
provided as a callback so this module can be tested independently and later
shared by on-modify and reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .lifecycle_models import (
    ExecutionStage,
    LifecycleAction,
    LifecycleContractError,
    LifecycleEvent,
    LifecycleIdentity,
    LifecyclePlan,
    ParentGuard,
    TaskSnapshot,
)


class LifecyclePlanningError(RuntimeError):
    """Raised when a lifecycle plan cannot be constructed safely."""


class ChildPlanBuilder(Protocol):
    def __call__(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class RecurrenceCandidate:
    """Pure recurrence result consumed by completion/expiration planning."""

    child_due: Any
    metadata: tuple[tuple[str, Any], ...] = ()
    dnf: Any = None
    until: Any = None
    terminal_reason: str = ""


class RecurrencePlanningService(Protocol):
    """Narrow recurrence boundary required by the lifecycle planner."""

    def next_candidate(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        kind: str,
        next_link: int,
    ) -> RecurrenceCandidate | None: ...

    def build_child(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        candidate: RecurrenceCandidate,
        next_link: int,
    ) -> Mapping[str, Any] | None: ...


def _link(value: Any, *, default: int | None = None) -> int:
    if value is None or value == "":
        if default is not None:
            return default
        raise LifecyclePlanningError("lifecycle task has no numeric link")
    if isinstance(value, bool):
        raise LifecyclePlanningError("lifecycle link cannot be boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LifecyclePlanningError(f"invalid lifecycle link: {value!r}") from exc
    if parsed < 0:
        raise LifecyclePlanningError("lifecycle link must be non-negative")
    return parsed


def _parent_guard(snapshot: TaskSnapshot) -> ParentGuard:
    try:
        return ParentGuard(
            status=str(snapshot.get("status") or "pending"),
            chain=str(snapshot.get("chain") or "on"),
            chain_id=str(snapshot.get("chainID") or ""),
            link=_link(snapshot.get("link"), default=0),
            recurrence_fingerprint=str(snapshot.get("recurrence_fingerprint") or ""),
        )
    except LifecycleContractError as exc:
        raise LifecyclePlanningError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class LifecyclePlanner:
    """Construct side-effect-free lifecycle plans from immutable snapshots."""

    validated_configuration: Any
    child_builder: ChildPlanBuilder | None = None
    recurrence_service: RecurrencePlanningService | None = None

    def __post_init__(self) -> None:
        if self.validated_configuration is None:
            raise LifecyclePlanningError("validated scheduling configuration is required")

    def plan(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
    ) -> LifecyclePlan:
        try:
            event = LifecycleEvent(event)
        except (TypeError, ValueError) as exc:
            raise LifecyclePlanningError(f"unsupported lifecycle event: {event!r}") from exc

        guard = _parent_guard(snapshot)
        target_link = guard.link + 1 if event in {LifecycleEvent.COMPLETE, LifecycleEvent.EXPIRE} else None
        identity = LifecycleIdentity(
            chain_id=guard.chain_id,
            parent_uuid=str(snapshot.get("uuid") or ""),
            source_link=guard.link,
            target_link=target_link,
            event=event,
        )

        if event in {LifecycleEvent.ACTIVATE, LifecycleEvent.RESUME}:
            return LifecyclePlan.from_mappings(
                identity=identity,
                action=LifecycleAction.UPDATE_PARENT,
                parent_guard=guard,
                parent_patch={"chain": "on"},
                expected_postconditions=("parent_chain_on",),
            )

        if event in {LifecycleEvent.DISABLE, LifecycleEvent.MANUAL_DELETE}:
            return LifecyclePlan.from_mappings(
                identity=identity,
                action=LifecycleAction.DISABLE_CHAIN,
                parent_guard=guard,
                parent_patch={"chain": "off"},
                expected_postconditions=("parent_chain_off",),
            )

        if event in {LifecycleEvent.CHAIN_MAX, LifecycleEvent.CHAIN_UNTIL}:
            return LifecyclePlan.from_mappings(
                identity=identity,
                action=LifecycleAction.FINALIZE_CHAIN,
                parent_guard=guard,
                parent_patch={"chain": "off"},
                expected_postconditions=("terminal_chain", "no_successor"),
            )

        if event not in {LifecycleEvent.COMPLETE, LifecycleEvent.EXPIRE}:
            raise LifecyclePlanningError(f"event {event.value!r} has no planning policy")

        child = None
        if self.recurrence_service is not None:
            kind = (
                "cp"
                if str(snapshot.get("cp") or "").strip()
                else "anchor_file"
                if str(snapshot.get("anchor_file") or "").strip()
                else "anchor"
            )
            if not any(str(snapshot.get(field) or "").strip() for field in ("cp", "anchor", "anchor_file")):
                return LifecyclePlan.from_mappings(
                    identity=identity,
                    action=LifecycleAction.FINALIZE_CHAIN,
                    parent_guard=guard,
                    parent_patch={"chain": "off"},
                    expected_postconditions=("terminal_chain", "no_successor"),
                )
            try:
                candidate = self.recurrence_service.next_candidate(snapshot, event, kind, target_link or 0)
                if candidate is None or candidate.terminal_reason:
                    return LifecyclePlan.from_mappings(
                        identity=identity,
                        action=LifecycleAction.FINALIZE_CHAIN,
                        parent_guard=guard,
                        parent_patch={"chain": "off"},
                        expected_postconditions=("terminal_chain", "no_successor"),
                    )
                child = self.recurrence_service.build_child(
                    snapshot,
                    event,
                    candidate,
                    target_link or 0,
                )
            except Exception as exc:
                raise LifecyclePlanningError(
                    f"could not build {event.value} successor: {type(exc).__name__}: {exc}"
                ) from exc
        elif self.child_builder is not None:
            try:
                child = self.child_builder(snapshot, event)
            except Exception as exc:
                raise LifecyclePlanningError(
                    f"could not build {event.value} successor: {type(exc).__name__}: {exc}"
                ) from exc
        if child is None:
            return LifecyclePlan.from_mappings(
                identity=identity,
                action=LifecycleAction.FINALIZE_CHAIN,
                parent_guard=guard,
                parent_patch={"chain": "off"},
                expected_postconditions=("terminal_chain", "no_successor"),
            )
        if not isinstance(child, Mapping) or not child.get("uuid"):
            raise LifecyclePlanningError("child builder returned an incomplete successor")
        return LifecyclePlan.from_mappings(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            child_payload=child,
            parent_patch={"nextLink": str(child.get("uuid"))[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )


__all__ = (
    "ChildPlanBuilder",
    "LifecyclePlanner",
    "LifecyclePlanningError",
    "RecurrenceCandidate",
    "RecurrencePlanningService",
)

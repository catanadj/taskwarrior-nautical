"""Pure lifecycle transition planning.

The planner creates a complete contract object but performs no Taskwarrior,
SQLite, queue, panel, or input mutation.  External recurrence/child work is
provided as a callback so this module can be tested independently and later
shared by on-modify and reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .lifecycle_models import (
    ExecutionStage,
    LifecycleAction,
    LifecycleContractError,
    LifecycleEvent,
    LifecycleIdentity,
    LifecyclePlan,
    ParentGuard,
    TaskSnapshot,
    recurrence_fingerprint,
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


class SuccessorLimitPolicy(Protocol):
    """Return a terminal reason when a candidate exceeds a chain limit."""

    def __call__(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        candidate: RecurrenceCandidate,
        next_link: int,
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ChainGenerationPlanningService:
    """Adapt ``ChainGenerationService`` to the planner's narrow protocol."""

    generation: Any

    def next_candidate(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        kind: str,
        next_link: int,
    ) -> RecurrenceCandidate | None:
        del event, next_link
        parent = snapshot.to_dict()
        if kind == "cp":
            child_due, metadata = self.generation.compute_cp_child_due(parent)
            dnf = None
        else:
            child_due, metadata, dnf = self.generation.compute_anchor_child_due(parent)
        if child_due is None:
            return None
        meta = dict(metadata or {})
        until = None
        raw_until = parent.get("chainUntil")
        if raw_until:
            until, error = self.generation.safe_parse_datetime(raw_until)
            if error or until is None:
                raise LifecyclePlanningError(f"invalid chainUntil: {error or raw_until}")
        return RecurrenceCandidate(
            child_due=child_due,
            metadata=tuple(sorted(meta.items())),
            dnf=dnf,
            until=until,
        )

    def build_child(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        candidate: RecurrenceCandidate,
        next_link: int,
    ) -> Mapping[str, Any] | None:
        del event
        parent = snapshot.to_dict()
        metadata = dict(candidate.metadata)
        child_field = str(metadata.get("target_field") or "due")
        kind = (
            "cp"
            if str(parent.get("cp") or "").strip()
            else "anchor_file"
            if str(parent.get("anchor_file") or "").strip()
            else "anchor"
        )
        cpmax = self.generation.core.coerce_int(parent.get("chainMax"), 0)
        parent_short = str(parent.get("uuid") or "").strip()[:8]
        return self.generation.build_child_from_parent(
            parent,
            candidate.child_due,
            child_field,
            next_link,
            parent_short,
            kind,
            cpmax,
            candidate.until,
        )


@dataclass(frozen=True, slots=True)
class PrecomputedRecurrencePlanningService:
    """Reuse a completion candidate while delegating child construction."""

    candidate: RecurrenceCandidate
    child_service: ChainGenerationPlanningService

    def next_candidate(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        kind: str,
        next_link: int,
    ) -> RecurrenceCandidate:
        del snapshot, event, kind, next_link
        return self.candidate

    def build_child(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        candidate: RecurrenceCandidate,
        next_link: int,
    ) -> Mapping[str, Any] | None:
        return self.child_service.build_child(snapshot, event, candidate, next_link)


@dataclass(frozen=True, slots=True)
class ChainGenerationLimitPolicy:
    """Default numeric and datetime limit policy for generation candidates."""

    compare_datetimes: Callable[[Any, Any], int]

    def __call__(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        candidate: RecurrenceCandidate,
        next_link: int,
    ) -> str | None:
        del event
        raw_max = snapshot.get("chainMax")
        if raw_max not in (None, ""):
            try:
                max_link = int(raw_max)
            except (TypeError, ValueError) as exc:
                raise LifecyclePlanningError(f"invalid chainMax: {raw_max!r}") from exc
            if max_link > 0 and next_link > max_link:
                return f"chainMax reached at link {max_link}"
        if candidate.until is not None and self.compare_datetimes(candidate.child_due, candidate.until) > 0:
            return "chainUntil reached"
        return None


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
            recurrence_fingerprint=recurrence_fingerprint(snapshot.to_dict()),
        )
    except LifecycleContractError as exc:
        raise LifecyclePlanningError(str(exc)) from exc


def terminal_plan_for_snapshot(
    snapshot: TaskSnapshot,
    event: LifecycleEvent,
) -> LifecyclePlan:
    """Build the shared terminal contract used by hooks and reconcile."""
    try:
        event = LifecycleEvent(event)
    except (TypeError, ValueError) as exc:
        raise LifecyclePlanningError(f"unsupported terminal event: {event!r}") from exc
    terminal_events = {
        LifecycleEvent.DISABLE,
        LifecycleEvent.MANUAL_DELETE,
        LifecycleEvent.CHAIN_MAX,
        LifecycleEvent.CHAIN_UNTIL,
        LifecycleEvent.COMPLETE,
        LifecycleEvent.EXPIRE,
    }
    if event not in terminal_events:
        raise LifecyclePlanningError(f"event {event.value!r} is not a terminal transition")
    guard = _parent_guard(snapshot)
    target_link = guard.link + 1 if event in {LifecycleEvent.COMPLETE, LifecycleEvent.EXPIRE} else None
    identity = LifecycleIdentity(
        chain_id=guard.chain_id,
        parent_uuid=str(snapshot.get("uuid") or ""),
        source_link=guard.link,
        target_link=target_link,
        event=event,
    )
    action = (
        LifecycleAction.DISABLE_CHAIN
        if event in {LifecycleEvent.DISABLE, LifecycleEvent.MANUAL_DELETE}
        else LifecycleAction.FINALIZE_CHAIN
    )
    if action is LifecycleAction.FINALIZE_CHAIN and str(snapshot.get("nextLink") or "").strip():
        raise LifecyclePlanningError(
            "terminal finalization has a persisted successor; retain it and review the chain"
        )
    postconditions = (
        ("parent_chain_off",)
        if action is LifecycleAction.DISABLE_CHAIN
        else ("terminal_chain", "no_successor")
    )
    return LifecyclePlan.from_mappings(
        identity=identity,
        action=action,
        parent_guard=guard,
        parent_patch={"chain": "off"},
        expected_postconditions=postconditions,
    )


@dataclass(frozen=True, slots=True)
class LifecyclePlanner:
    """Construct side-effect-free lifecycle plans from immutable snapshots."""

    validated_configuration: Any
    child_builder: ChildPlanBuilder | None = None
    recurrence_service: RecurrencePlanningService | None = None
    successor_limit_policy: SuccessorLimitPolicy | None = None

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

        if event in {
            LifecycleEvent.DISABLE,
            LifecycleEvent.MANUAL_DELETE,
            LifecycleEvent.CHAIN_MAX,
            LifecycleEvent.CHAIN_UNTIL,
        }:
            return terminal_plan_for_snapshot(snapshot, event)

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
                return terminal_plan_for_snapshot(snapshot, event)
            try:
                candidate = self.recurrence_service.next_candidate(snapshot, event, kind, target_link or 0)
                if candidate is None or candidate.terminal_reason:
                    return terminal_plan_for_snapshot(snapshot, event)
                if self.successor_limit_policy is not None:
                    try:
                        limit_reason = self.successor_limit_policy(
                            snapshot,
                            event,
                            candidate,
                            target_link or 0,
                        )
                    except Exception as exc:
                        raise LifecyclePlanningError(
                            f"successor limit evaluation failed: {type(exc).__name__}: {exc}"
                        ) from exc
                    if limit_reason:
                        return terminal_plan_for_snapshot(snapshot, event)
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
            return terminal_plan_for_snapshot(snapshot, event)
        if not isinstance(child, Mapping) or child.get("link") in (None, ""):
            raise LifecyclePlanningError("child builder returned an incomplete successor")
        child_uuid = str(child.get("uuid") or "").strip()
        parent_patch = {"nextLink": child_uuid[:8]} if child_uuid else {}
        return LifecyclePlan.from_mappings(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            child_payload=child,
            parent_patch=parent_patch,
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )


__all__ = (
    "ChildPlanBuilder",
    "ChainGenerationLimitPolicy",
    "ChainGenerationPlanningService",
    "LifecyclePlanner",
    "LifecyclePlanningError",
    "RecurrenceCandidate",
    "RecurrencePlanningService",
    "SuccessorLimitPolicy",
    "PrecomputedRecurrencePlanningService",
    "terminal_plan_for_snapshot",
)

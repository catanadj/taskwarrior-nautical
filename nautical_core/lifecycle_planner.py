"""Pure lifecycle transition planning.

The planner creates a complete contract object but performs no Taskwarrior,
SQLite, queue, panel, or input mutation.  External recurrence/child work is
provided as a callback so this module can be tested independently and later
shared by on-modify and reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
from .task_codec import TaskCodec
from .task_codec import DEFAULT_TASK_CODEC
from .task_models import NauticalTask, TaskDraft, TaskPayload


class LifecyclePlanningError(RuntimeError):
    """Raised when a lifecycle plan cannot be constructed safely."""


def _terminal_kind_for(event: LifecycleEvent, reason: str = "") -> str | None:
    """Normalize terminal provenance into the durable plan vocabulary."""
    if event is LifecycleEvent.CHAIN_MAX:
        return "chain_max"
    if event is LifecycleEvent.CHAIN_UNTIL:
        return "chain_until"
    normalized = str(reason or "").strip().lower().replace("-", "_")
    if "chain_max" in normalized or "chainmax" in normalized:
        return "chain_max"
    if "chain_until" in normalized or "chainuntil" in normalized:
        return "chain_until"
    if "search_limit" in normalized or "scheduler_exhaust" in normalized:
        return "search_limit"
    if "date_limit" in normalized or "until" in normalized:
        return "date_limit"
    return None


def _recurrence_kind(task: TaskSnapshot | NauticalTask) -> str:
    """Return the active recurrence kind, treating Taskwarrior null as unset."""
    def raw_value(key: str) -> Any:
        if isinstance(task, TaskSnapshot):
            return task.get(key)
        return task.observation.field(key).raw_value()

    if TaskCodec.normalize_text(raw_value("cp")):
        return "cp"
    if TaskCodec.normalize_text(raw_value("anchor_file")):
        return "anchor_file"
    return "anchor"


@dataclass(frozen=True, slots=True)
class RecurrenceCandidate:
    """Pure recurrence result consumed by completion/expiration planning."""

    child_due: Any
    metadata: tuple[tuple[str, Any], ...] = ()
    dnf: Any = None
    until: Any = None
    terminal_reason: str = ""


@dataclass(frozen=True, slots=True)
class LifecyclePreflight:
    """Validated, side-effect-free inputs gathered before planning."""

    base_link: int
    next_link: int
    kind: str
    chain_id: str

    @classmethod
    def from_context(
        cls,
        *,
        base_link: Any,
        next_link: Any,
        kind: str,
        chain_id: Any,
    ) -> "LifecyclePreflight":
        try:
            base = _link(base_link)
            target = _link(next_link)
        except LifecyclePlanningError:
            raise
        if target != base + 1:
            raise LifecyclePlanningError("lifecycle preflight next link is not adjacent to the parent")
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind not in {"cp", "anchor", "anchor_file"}:
            raise LifecyclePlanningError(f"unsupported lifecycle recurrence kind: {kind!r}")
        normalized_chain = str(chain_id or "").strip()
        if not normalized_chain:
            raise LifecyclePlanningError("lifecycle preflight has no chainID")
        return cls(base, target, normalized_kind, normalized_chain)


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
    ) -> TaskDraft | None: ...


class SuccessorLimitPolicy(Protocol):
    """Return a terminal reason when a candidate exceeds a chain limit."""

    def __call__(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        candidate: RecurrenceCandidate,
        next_link: int,
    ) -> str | None: ...


class CarryValidator(Protocol):
    def __call__(
        self,
        snapshot: TaskSnapshot,
        child: TaskDraft,
        candidate: RecurrenceCandidate,
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
        parent = NauticalTask.from_observation(snapshot.observation)
        if kind == "cp":
            child_due, metadata = self.generation.compute_cp_child_due(parent)
            dnf = None
        else:
            child_due, metadata, dnf = self.generation.compute_anchor_child_due(parent)
        if child_due is None:
            return None
        meta = dict(metadata or {})
        until = None
        raw_until = parent.observation.field("chainUntil").raw_value()
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
    ) -> TaskDraft | None:
        del event
        parent = NauticalTask.from_observation(snapshot.observation)
        metadata = dict(candidate.metadata)
        child_field = str(metadata.get("target_field") or "due")
        kind = _recurrence_kind(parent)
        cpmax = self.generation.core.coerce_int(parent.observation.field("chainMax").raw_value(), 0)
        parent_short = str(parent.observation.field("uuid").raw_value() or "").strip()[:8]
        return self.generation.build_child_draft(
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
    ) -> TaskDraft | None:
        return self.child_service.build_child(snapshot, event, candidate, next_link)


def plan_candidate_successor(
    snapshot: TaskSnapshot,
    event: LifecycleEvent,
    candidate: RecurrenceCandidate,
    *,
    generation: Any,
    validated_configuration: Any,
    compare_datetimes: Callable[[Any, Any], int],
    preflight: LifecyclePreflight | None = None,
    carry_validator: CarryValidator | None = None,
) -> LifecyclePlan:
    """Build one candidate-backed plan for hooks, expiration, and reconcile.

    Candidate calculation remains owned by the caller, but the conversion
    into a lifecycle plan is deliberately centralized so those entry points
    cannot drift in child construction or successor-limit handling.
    """
    recurrence = PrecomputedRecurrencePlanningService(
        candidate=candidate,
        child_service=ChainGenerationPlanningService(generation),
    )
    planner = LifecyclePlanner(
        validated_configuration,
        recurrence_service=recurrence,
        successor_limit_policy=ChainGenerationLimitPolicy(compare_datetimes),
    )
    return planner.plan(snapshot, event, preflight=preflight, carry_validator=carry_validator)


def expiration_candidate(snapshot: TaskSnapshot, *, generation: Any) -> RecurrenceCandidate:
    """Calculate an expiration successor from the prior recurrence target."""
    parent = snapshot.to_dict()
    target_field = "due" if parent.get("due") else "scheduled"
    target = parent.get(target_field)
    if not str(target or "").strip():
        raise LifecyclePlanningError("expired recurrence has no due or scheduled timestamp")
    calculation_parent = dict(parent)
    calculation_parent["end"] = target
    kind = (
        "cp" if TaskCodec.normalize_text(parent.get("cp"))
        else "anchor_file" if TaskCodec.normalize_text(parent.get("anchor_file"))
        else "anchor"
    )
    calculation_observation = DEFAULT_TASK_CODEC.decode_row(
        calculation_parent,
        source_query="lifecycle expiration calculation",
    )
    calculation_task = NauticalTask.from_observation(calculation_observation)
    if kind in {"anchor", "anchor_file"}:
        child_due, metadata, dnf = generation.compute_anchor_child_due(calculation_task)
    else:
        child_due, metadata = generation.compute_cp_child_due(calculation_task)
        dnf = None
    result_metadata = dict(metadata or {})
    result_metadata["basis"] = f"{target_field} recurrence target (expired)"
    result_metadata["target_field"] = target_field
    return RecurrenceCandidate(
        child_due=child_due,
        metadata=tuple(sorted(result_metadata.items())),
        dnf=dnf,
    )


def plan_expiration_successor(
    snapshot: TaskSnapshot,
    *,
    generation: Any,
    validated_configuration: Any,
    compare_datetimes: Callable[[Any, Any], int],
    preflight: LifecyclePreflight | None = None,
    carry_validator: CarryValidator | None = None,
) -> LifecyclePlan:
    """Build the shared expiration plan from the prior recurrence target."""
    candidate = expiration_candidate(snapshot, generation=generation)
    return plan_candidate_successor(
        snapshot,
        LifecycleEvent.EXPIRE,
        candidate,
        generation=generation,
        validated_configuration=validated_configuration,
        compare_datetimes=compare_datetimes,
        preflight=preflight,
        carry_validator=carry_validator,
    )


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
        # Taskwarrior exports numeric UDAs as fixed-point text (for example
        # ``256.000000``). Accept only an exactly integral representation;
        # fractional, NaN, and infinite values remain invalid links.
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError) as float_exc:
            raise LifecyclePlanningError(f"invalid lifecycle link: {value!r}") from float_exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise LifecyclePlanningError(f"invalid lifecycle link: {value!r}") from exc
        parsed = int(numeric)
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
            modified=str(snapshot.get("modified") or ""),
        )
    except LifecycleContractError as exc:
        raise LifecyclePlanningError(str(exc)) from exc


def terminal_plan_for_snapshot(
    snapshot: TaskSnapshot,
    event: LifecycleEvent,
    *,
    terminal_kind: str | None = None,
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
    return LifecyclePlan(
        identity=identity,
        action=action,
        parent_guard=guard,
        parent_patch=(("chain", "off"),),
        expected_postconditions=postconditions,
        terminal_kind=terminal_kind or _terminal_kind_for(event),
    )


@dataclass(frozen=True, slots=True)
class LifecyclePlanner:
    """Construct side-effect-free lifecycle plans from immutable snapshots."""

    validated_configuration: Any
    recurrence_service: RecurrencePlanningService | None = None
    successor_limit_policy: SuccessorLimitPolicy | None = None

    def __post_init__(self) -> None:
        if self.validated_configuration is None:
            raise LifecyclePlanningError("validated scheduling configuration is required")

    def plan(
        self,
        snapshot: TaskSnapshot,
        event: LifecycleEvent,
        *,
        preflight: LifecyclePreflight | None = None,
        carry_validator: CarryValidator | None = None,
    ) -> LifecyclePlan:
        try:
            event = LifecycleEvent(event)
        except (TypeError, ValueError) as exc:
            raise LifecyclePlanningError(f"unsupported lifecycle event: {event!r}") from exc

        guard = _parent_guard(snapshot)
        target_link = guard.link + 1 if event in {LifecycleEvent.COMPLETE, LifecycleEvent.EXPIRE} else None
        if preflight is not None:
            if preflight.chain_id != guard.chain_id:
                raise LifecyclePlanningError("lifecycle preflight chainID differs from the task snapshot")
            if preflight.base_link != guard.link:
                raise LifecyclePlanningError("lifecycle preflight parent link differs from the task snapshot")
            if target_link is not None and preflight.next_link != target_link:
                raise LifecyclePlanningError("lifecycle preflight next link differs from the task snapshot")
            snapshot_kind = _recurrence_kind(snapshot)
            if preflight.kind != snapshot_kind:
                raise LifecyclePlanningError("lifecycle preflight recurrence kind differs from the task snapshot")
        identity = LifecycleIdentity(
            chain_id=guard.chain_id,
            parent_uuid=str(snapshot.get("uuid") or ""),
            source_link=guard.link,
            target_link=target_link,
            event=event,
        )

        if event in {LifecycleEvent.ACTIVATE, LifecycleEvent.RESUME}:
            return LifecyclePlan(
                identity=identity,
                action=LifecycleAction.UPDATE_PARENT,
                parent_guard=guard,
                parent_patch=(("chain", "on"),),
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
        candidate: RecurrenceCandidate | None = None
        if self.recurrence_service is not None:
            kind = _recurrence_kind(snapshot)
            if not any(TaskCodec.normalize_text(snapshot.get(field)) for field in ("cp", "anchor", "anchor_file")):
                return terminal_plan_for_snapshot(
                    snapshot,
                    event,
                    terminal_kind=_terminal_kind_for(event),
                )
            try:
                candidate = self.recurrence_service.next_candidate(snapshot, event, kind, target_link or 0)
                if candidate is None or candidate.terminal_reason:
                    return terminal_plan_for_snapshot(
                        snapshot,
                        event,
                        terminal_kind=_terminal_kind_for(
                            event,
                            candidate.terminal_reason if candidate is not None else "",
                        ),
                    )
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
                        return terminal_plan_for_snapshot(
                            snapshot,
                            event,
                            terminal_kind=_terminal_kind_for(event, limit_reason),
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
        if child is None:
            return terminal_plan_for_snapshot(
                snapshot,
                event,
                terminal_kind=_terminal_kind_for(event),
            )
        if not isinstance(child, TaskDraft):
            raise LifecyclePlanningError("recurrence builder returned a non-TaskDraft successor")
        if child.identity.link.value <= 0:
            raise LifecyclePlanningError("child draft is missing its link")
        if carry_validator is not None:
            try:
                carry_error = carry_validator(
                    snapshot,
                    child,
                    candidate or RecurrenceCandidate(child_due=None),
                )
            except Exception as exc:
                raise LifecyclePlanningError(
                    f"carry validation failed: {type(exc).__name__}: {exc}"
                ) from exc
            if carry_error:
                raise LifecyclePlanningError(str(carry_error))
        child_uuid = child.identity.task_uuid.value
        parent_patch = {"nextLink": child_uuid[:8]} if child_uuid else {}
        return LifecyclePlan.from_draft(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            draft=child,
            parent_patch=parent_patch,
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )


__all__ = (
    "CarryValidator",
    "ChainGenerationLimitPolicy",
    "ChainGenerationPlanningService",
    "LifecyclePlanner",
    "LifecyclePlanningError",
    "LifecyclePreflight",
    "RecurrenceCandidate",
    "RecurrencePlanningService",
    "SuccessorLimitPolicy",
    "PrecomputedRecurrencePlanningService",
    "plan_candidate_successor",
    "expiration_candidate",
    "plan_expiration_successor",
    "terminal_plan_for_snapshot",
)

"""Pure typed route classification for modify workflow transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .task_changes import TaskTransition
from .task_field_policy import VOLATILE_TASK_FIELDS


class ModifyRouteKind(str, Enum):
    ORDINARY = "ordinary"
    RECURRING_EDIT = "recurring_edit"
    ACTIVATION = "activation"
    COMPLETION = "completion"
    IDEMPOTENT_COMPLETION = "idempotent_completion"
    DELETION = "deletion"
    DISABLE = "disable"
    RECURRENCE_REMOVAL = "recurrence_removal"
    RESUME = "resume"
    MANUAL_CHAIN_OFF = "manual_chain_off"
    INVALID_IDENTITY_EDIT = "invalid_identity_edit"


_RECURRENCE_FIELDS = frozenset({"anchor", "anchor_file", "cp", "omit", "omit_file"})
_CHAIN_FIELDS = frozenset({"chainID", "link", "prevLink", "nextLink", "chain"})
_IDENTITY_FIELDS = frozenset({"chainID", "link", "prevLink", "nextLink"})


@dataclass(frozen=True, slots=True)
class ModifyWorkflowRoute:
    """One mutually exclusive route and its local evidence summary."""

    kind: ModifyRouteKind
    has_nautical_fields: bool
    changed_fields: tuple[str, ...]
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = ModifyRouteKind(self.kind)
        changed = tuple(sorted(set(self.changed_fields)))
        evidence = tuple(sorted(set(str(item) for item in self.evidence)))
        if kind is ModifyRouteKind.ORDINARY and self.has_nautical_fields:
            raise ValueError("ordinary modify route cannot contain Nautical fields")
        if kind is ModifyRouteKind.INVALID_IDENTITY_EDIT and "identity_mutation" not in evidence:
            raise ValueError("invalid identity route requires identity mutation evidence")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "changed_fields", changed)
        object.__setattr__(self, "evidence", evidence)

    @property
    def volatile_fields(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.changed_fields).intersection(VOLATILE_TASK_FIELDS)))

    @property
    def user_changed_fields(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.changed_fields).difference(VOLATILE_TASK_FIELDS)))

    @property
    def identity_mutation(self) -> bool:
        return "identity_mutation" in self.evidence

    @property
    def requires_spawn_evidence(self) -> bool:
        return self.kind is ModifyRouteKind.COMPLETION and "already_linked" not in self.evidence

    @property
    def required_evidence(self) -> tuple[str, ...]:
        if self.kind in {ModifyRouteKind.ORDINARY, ModifyRouteKind.INVALID_IDENTITY_EDIT}:
            return ()
        if self.kind in {ModifyRouteKind.COMPLETION, ModifyRouteKind.IDEMPOTENT_COMPLETION}:
            return ("chain_slot", "parent_snapshot")
        return ("task_snapshot",)


def _value(transition: TaskTransition, side: str, field: str) -> str:
    observation = transition.old if side == "old" else transition.new
    return str(observation.field(field).raw_value() or "").strip().lower()


def _has_recurrence(transition: TaskTransition, side: str) -> bool:
    observation = transition.old if side == "old" else transition.new
    return any(str(observation.field(field).raw_value() or "").strip() for field in _RECURRENCE_FIELDS)


def classify_modify_transition(transition: TaskTransition) -> ModifyWorkflowRoute:
    """Classify a typed old/new transition without callbacks or side effects."""
    if not isinstance(transition, TaskTransition):
        raise TypeError("modify route classification requires a TaskTransition")
    old_status = _value(transition, "old", "status")
    new_status = _value(transition, "new", "status")
    old_recurrence = _has_recurrence(transition, "old")
    new_recurrence = _has_recurrence(transition, "new")
    old_chain = _value(transition, "old", "chain")
    new_chain = _value(transition, "new", "chain")
    has_nautical = old_recurrence or new_recurrence or any(
        transition.old.field(field).raw_value() or transition.new.field(field).raw_value()
        for field in _CHAIN_FIELDS
    )
    evidence: list[str] = []

    linking_successor = (
        transition.changed("nextLink")
        and not _value(transition, "old", "nextLink")
        and bool(_value(transition, "new", "nextLink"))
        and new_status == "completed"
    )
    identity_edit = bool(_IDENTITY_FIELDS.intersection(transition.changed_fields)) and not linking_successor
    if identity_edit and old_recurrence and new_recurrence:
        kind = ModifyRouteKind.INVALID_IDENTITY_EDIT
        evidence.append("identity_mutation")
    elif not has_nautical:
        kind = ModifyRouteKind.ORDINARY
    elif new_status == "deleted":
        kind = ModifyRouteKind.DELETION
    elif old_status != "completed" and new_status == "completed" and not _value(transition, "new", "nextLink"):
        kind = ModifyRouteKind.COMPLETION
    elif new_status == "completed":
        kind = ModifyRouteKind.IDEMPOTENT_COMPLETION
        evidence.append("already_completed")
    elif not old_recurrence and new_recurrence:
        kind = ModifyRouteKind.ACTIVATION
    elif old_recurrence and not new_recurrence:
        kind = ModifyRouteKind.RECURRENCE_REMOVAL
    elif old_chain == "off" and new_chain == "on":
        kind = ModifyRouteKind.RESUME
    elif old_chain != "off" and new_chain == "off" and new_recurrence:
        kind = ModifyRouteKind.MANUAL_CHAIN_OFF
    else:
        kind = ModifyRouteKind.RECURRING_EDIT

    if old_status == new_status:
        evidence.append("status_unchanged")
    if transition.changed_fields and set(transition.changed_fields).issubset(VOLATILE_TASK_FIELDS):
        evidence.append("volatile_only")
    if transition.changed_fields and set(transition.changed_fields).issubset(_CHAIN_FIELDS):
        evidence.append("chain_identity_edit")
    if _value(transition, "new", "nextLink"):
        evidence.append("already_linked")
    return ModifyWorkflowRoute(kind, has_nautical, transition.changed_fields, tuple(evidence))


__all__ = ("ModifyRouteKind", "ModifyWorkflowRoute", "classify_modify_transition")

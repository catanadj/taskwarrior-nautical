"""Shared field policies for task patches and lifecycle payload comparison."""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class DraftFieldClass(str, Enum):
    """Role of a field at the child-draft boundary."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    COPIED = "copied"
    CLEARED = "cleared"
    GENERATED = "taskwarrior_generated"
    FORBIDDEN = "forbidden"


# Draft fields are classified once here so child generation and mutation
# verification do not maintain separate ad-hoc exclusion lists.
DRAFT_REQUIRED_FIELDS: FrozenSet[str] = frozenset(
    {"description", "chain", "chainID", "link", "status"}
)
DRAFT_COPIED_FIELDS: FrozenSet[str] = frozenset(
    {"project", "tags", "priority", "wait", "scheduled", "duration", "value"}
)
DRAFT_OPTIONAL_FIELDS: FrozenSet[str] = frozenset(
    {"anchor", "anchor_file", "anchor_mode", "cp", "omit", "omit_file", "bc", "chainMax", "chainUntil"}
)
DRAFT_CLEARED_FIELDS: FrozenSet[str] = frozenset({"nextLink", "end", "start"})
DRAFT_GENERATED_FIELDS: FrozenSet[str] = frozenset(
    {"id", "uuid", "modified", "end", "urgency", "nextLink"}
)
DRAFT_FORBIDDEN_FIELDS: FrozenSet[str] = frozenset(
    {"uuid", "chainID", "link", "prevLink", "id", "status", "modified", "end"}
)


VOLATILE_TASK_FIELDS: FrozenSet[str] = frozenset({"id", "urgency", "modified", "end"})
IMMUTABLE_TASK_FIELDS: FrozenSet[str] = frozenset({"uuid", "chainID", "link", "prevLink"})
TASK_TIMESTAMP_FIELDS: FrozenSet[str] = frozenset(
    {"due", "scheduled", "wait", "until", "entry", "modified", "end"}
)

# Fields emitted by Taskwarrior or structural linking that must not make a
# deterministic lifecycle child payload look semantically different.
LIFECYCLE_VOLATILE_CHILD_FIELDS: FrozenSet[str] = frozenset(
    {
        "id", "entry", "modified", "urgency", "status", "end", "start",
        "nextLink", "mask", "imask", "parent", "recur", "rc",
    }
)


def classify_draft_field(field: str) -> DraftFieldClass:
    """Return the single authoritative role for a draft field."""
    name = str(field)
    if name in DRAFT_REQUIRED_FIELDS:
        return DraftFieldClass.REQUIRED
    if name in DRAFT_COPIED_FIELDS:
        return DraftFieldClass.COPIED
    if name in DRAFT_OPTIONAL_FIELDS:
        return DraftFieldClass.OPTIONAL
    if name in DRAFT_CLEARED_FIELDS:
        return DraftFieldClass.CLEARED
    if name in DRAFT_GENERATED_FIELDS:
        return DraftFieldClass.GENERATED
    if name in DRAFT_FORBIDDEN_FIELDS:
        return DraftFieldClass.FORBIDDEN
    # Unknown fields are arbitrary user fields and are preserved as copied data.
    return DraftFieldClass.COPIED


def draft_field_may_be_supplied(field: str) -> bool:
    """Whether a field may be carried in ``TaskDraft.fields``.

    Required, generated, cleared, and identity fields are emitted by the draft
    owner and must not be smuggled in through the arbitrary carried mapping.
    """
    return classify_draft_field(field) is DraftFieldClass.COPIED


__all__ = (
    "DraftFieldClass",
    "DRAFT_REQUIRED_FIELDS",
    "DRAFT_COPIED_FIELDS",
    "DRAFT_OPTIONAL_FIELDS",
    "DRAFT_CLEARED_FIELDS",
    "DRAFT_GENERATED_FIELDS",
    "DRAFT_FORBIDDEN_FIELDS",
    "VOLATILE_TASK_FIELDS",
    "IMMUTABLE_TASK_FIELDS",
    "TASK_TIMESTAMP_FIELDS",
    "LIFECYCLE_VOLATILE_CHILD_FIELDS",
    "classify_draft_field",
    "draft_field_may_be_supplied",
)

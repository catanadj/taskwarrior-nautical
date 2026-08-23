"""Shared field policies for task patches and lifecycle payload comparison."""

from __future__ import annotations

from typing import FrozenSet


# Draft fields are classified once here so child generation and mutation
# verification do not maintain separate ad-hoc exclusion lists.
DRAFT_REQUIRED_FIELDS: FrozenSet[str] = frozenset(
    {"description", "chain", "chainID", "link", "status"}
)
DRAFT_COPIED_FIELDS: FrozenSet[str] = frozenset(
    {"project", "tags", "priority", "wait", "scheduled", "duration", "value"}
)
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


__all__ = (
    "DRAFT_REQUIRED_FIELDS",
    "DRAFT_COPIED_FIELDS",
    "DRAFT_GENERATED_FIELDS",
    "DRAFT_FORBIDDEN_FIELDS",
    "VOLATILE_TASK_FIELDS",
    "IMMUTABLE_TASK_FIELDS",
    "TASK_TIMESTAMP_FIELDS",
    "LIFECYCLE_VOLATILE_CHILD_FIELDS",
)

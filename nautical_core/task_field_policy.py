"""Shared field policies for task patches and lifecycle payload comparison."""

from __future__ import annotations

from typing import FrozenSet


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
    "VOLATILE_TASK_FIELDS",
    "IMMUTABLE_TASK_FIELDS",
    "TASK_TIMESTAMP_FIELDS",
    "LIFECYCLE_VOLATILE_CHILD_FIELDS",
)

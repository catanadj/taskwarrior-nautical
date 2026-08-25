"""Small task-field decisions used by typed modify effects."""

from __future__ import annotations

from typing import Any


def recurrence_anchor_field(task: dict[str, Any] | None) -> str:
    if isinstance(task, dict):
        if task.get("due"):
            return "due"
        if task.get("scheduled"):
            return "scheduled"
    return "due"


__all__ = ("recurrence_anchor_field",)

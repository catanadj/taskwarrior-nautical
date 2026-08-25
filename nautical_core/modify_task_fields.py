"""Small task-field decisions used by typed modify effects."""

from __future__ import annotations

from typing import Any


def recurrence_anchor_field(payload: dict[str, Any] | None) -> str:
    if isinstance(payload, dict):
        if payload.get("due"):
            return "due"
        if payload.get("scheduled"):
            return "scheduled"
    return "due"


def root_uuid(payload: dict[str, Any]) -> str:
    return str(payload.get("chainID") or "").strip()


__all__ = ("recurrence_anchor_field", "root_uuid")

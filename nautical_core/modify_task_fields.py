"""Small task-field decisions used by typed modify effects."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
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


def _canonical(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return Decimal(text)
        except (InvalidOperation, ValueError):
            return text
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(value)


def field_changed(old: dict[str, Any], new: dict[str, Any], key: str) -> bool:
    return _canonical(old.get(key)) != _canonical(new.get(key))


__all__ = ("recurrence_anchor_field", "root_uuid", "field_changed")

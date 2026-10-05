"""Typed, lossless representations for Taskwarrior JSON payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypeGuard


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _is_json_value(value: object, ancestors: set[int]) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if not isinstance(value, (list, dict)):
        return False

    identity = id(value)
    if identity in ancestors:
        return False
    ancestors.add(identity)
    try:
        if isinstance(value, list):
            return all(_is_json_value(item, ancestors) for item in value)
        return all(
            isinstance(key, str) and _is_json_value(item, ancestors)
            for key, item in value.items()
        )
    finally:
        ancestors.remove(identity)


def is_json_object(value: object) -> TypeGuard[JsonObject]:
    """Return whether a value is a complete, acyclic JSON task object."""
    if not isinstance(value, dict):
        return False
    ancestors = {id(value)}
    return all(
        isinstance(key, str) and _is_json_value(item, ancestors)
        for key, item in value.items()
    )


@dataclass(slots=True)
class TaskDocument:
    """Lossless task wrapper used at hook boundaries.

    Taskwarrior permits arbitrary UDAs and extensions.  The wrapper therefore
    keeps the original dictionary instead of projecting it into a closed
    schema; typed accessors are opt-in conveniences for known scalar fields.
    """

    data: JsonObject

    @classmethod
    def from_object(cls, value: object) -> "TaskDocument | None":
        if not is_json_object(value):
            return None
        return cls(value)

    def as_dict(self) -> JsonObject:
        """Return the live payload so hook mutations remain lossless."""
        return self.data

    def get(self, field: str, default: JsonValue = None) -> JsonValue:
        return self.data.get(field, default)

    def has_value(self, field: str) -> bool:
        value = self.data.get(field)
        return value is not None and bool(str(value).strip())

    def text(self, field: str, default: str = "") -> str:
        value = self.data.get(field)
        if value is None:
            return default
        return value if isinstance(value, str) else str(value)

    def number(self, field: str, default: int | float | None = None) -> int | float | None:
        value = self.data.get(field)
        if isinstance(value, bool) or value is None:
            return default
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                parsed = float(value)
            except ValueError:
                return default
            return int(parsed) if parsed.is_integer() else parsed
        return default

    def integer(self, field: str, default: int | None = None) -> int | None:
        value = self.number(field)
        if value is None:
            return default
        return int(value)

    def boolean(self, field: str, default: bool | None = None) -> bool | None:
        value = self.data.get(field)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "on", "yes"}:
                return True
            if lowered in {"0", "false", "off", "no"}:
                return False
        return default


__all__ = ("JsonObject", "JsonScalar", "JsonValue", "TaskDocument", "is_json_object")

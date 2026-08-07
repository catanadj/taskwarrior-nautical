"""Structured diagnostic events with a stable, human-readable rendering."""

from __future__ import annotations

from typing import Mapping


class DiagnosticEvent:
    __slots__ = ("code", "level", "message", "hook", "context")

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hook: str = "nautical",
        level: str = "info",
        context: Mapping[str, object] | None = None,
    ) -> None:
        self.code = str(code or "diagnostic")
        self.level = str(level or "info")
        self.message = str(message or "")
        self.hook = str(hook or "nautical")
        self.context = dict(context or {})

    @classmethod
    def from_message(cls, message: object, *, hook: str) -> "DiagnosticEvent":
        text = str(message or "")
        return cls("legacy.message", text, hook=hook)

    def render(self) -> str:
        return self.message

    def to_log_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "code": self.code,
            "level": self.level,
            "hook": self.hook,
            "message": self.message,
        }
        if self.context:
            record["context"] = dict(self.context)
        return record


__all__ = ("DiagnosticEvent",)

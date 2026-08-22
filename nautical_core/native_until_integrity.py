"""Typed native-until audit results shared by reconcile and integrity tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeUntilAudit:
    """Fail-closed result of one native-until evidence pass."""

    status: str
    repairs: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        status = str(self.status or "").strip().lower()
        if status not in {"valid", "invalid", "unavailable"}:
            raise ValueError(f"invalid native-until audit status: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "repairs", tuple(dict(item) for item in self.repairs))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))
        object.__setattr__(self, "reason", str(self.reason or "").strip())


def audit_result(repairs: list[dict[str, Any]], errors: list[str]) -> NativeUntilAudit:
    """Classify an audit without allowing an empty result to hide errors."""
    return NativeUntilAudit("invalid" if repairs or errors else "valid", tuple(repairs), tuple(errors))


__all__ = ["NativeUntilAudit", "audit_result"]

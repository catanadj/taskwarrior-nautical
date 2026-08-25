"""Typed result models for the on-exit lifecycle boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExitDrainStats:
    entries_total: int = 0
    processed: int = 0
    errors: int = 0
    retry_released: int = 0
    manual_reviewed: int = 0
    quarantined: int = 0
    conflicted: int = 0
    outbox_lock_failures: int = 0
    diagnostics_suppressed: int = 0
    drain_ms: float = 0.0

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_mapping().get(key, default)


__all__ = ("ExitDrainStats",)

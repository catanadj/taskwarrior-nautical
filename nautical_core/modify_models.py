from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompletionPreflightContext:
    parent_short: str
    base_no: int
    next_no: int
    kind: str
    chain_id: str


@dataclass(slots=True)
class CompletionComputeResult:
    child_due: Any
    meta: Any
    dnf: Any
    until_dt: Any
    cpmax: int
    cap_no: int | None
    finals: list[tuple[str, Any]]
    until_cap_no: int | None


@dataclass(slots=True)
class CpCompletionFeedbackModel:
    new: dict[str, Any]
    child: dict[str, Any]
    child_due: Any
    child_short: str
    next_no: int
    parent_short: str
    cap_no: int | None
    finals: list[tuple[str, Any]]
    now_utc: Any
    until_dt: Any
    until_cap_no: int | None
    meta: dict[str, Any]
    deferred_spawn: bool
    spawn_intent_id: str | None
    chain_by_short: dict[str, Any] | None
    analytics_advice: str | None
    integrity_warnings: list[str] | None
    base_no: int

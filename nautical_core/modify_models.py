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

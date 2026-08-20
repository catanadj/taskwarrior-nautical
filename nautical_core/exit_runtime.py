"""Minimal runtime-owned state for on-exit diagnostics and startup timing.

Mutation, verification, and queue-drain orchestration now live in
``lifecycle_application.LifecycleApplicationService``; this module only
keeps the small bit of invocation-scoped bookkeeping the thin hook adapter
(``hooks/exit_impl.py``) uses for diagnostics (``NAUTICAL_DIAG=1``) and
startup timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExitRuntimeState:
    unit_of_work: Any | None = None
    repository: Any | None = None
    diag_stats: dict[str, Any] = field(default_factory=dict)
    startup_stats: dict[str, float | int] = field(default_factory=dict)


def new_runtime_state() -> ExitRuntimeState:
    return ExitRuntimeState()


__all__ = (
    "ExitRuntimeState",
    "new_runtime_state",
)

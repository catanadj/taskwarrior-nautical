from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExitEntryContext:
    entry: dict[str, Any]
    idx: int
    state: Any
    parent_uuid: str
    child_short: str
    expected_parent_nextlink: str | None
    child: dict[str, Any]
    child_uuid: str
    spawn_intent_id: str


@dataclass(slots=True)
class ExitPrecheckServices:
    parent_nextlink_state: Any
    requeue_or_dead_letter_for_lock: Any


@dataclass(slots=True)
class ExitEnsureChildServices:
    export_uuid: Any
    import_child: Any
    is_lock_error: Any
    diag: Any
    requeue_or_dead_letter_for_lock: Any


@dataclass(slots=True)
class ExitApplyParentUpdateServices:
    update_parent_nextlink: Any
    is_lock_error: Any
    cleanup_orphan_child: Any
    diag: Any
    requeue_or_dead_letter_for_lock: Any

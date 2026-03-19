from __future__ import annotations

from typing import Any, Callable


def precheck_parent_link_state(
    entry: dict[str, Any],
    idx: int,
    state: Any,
    *,
    parent_uuid: str,
    child_short: str,
    expected_parent_nextlink: str,
    parent_nextlink_state: Callable[[str, str, str | None], tuple[str, str]],
    requeue_or_dead_letter_for_lock: Callable[[dict[str, Any], int, Any], bool],
) -> tuple[str, bool]:
    if not (parent_uuid and child_short):
        return "ok", False

    link_state, link_err = parent_nextlink_state(parent_uuid, child_short, expected_parent_nextlink)
    if link_state == "locked":
        return ("break", False) if requeue_or_dead_letter_for_lock(entry, idx, state) else ("continue", False)
    if link_state in {"conflict", "missing", "invalid"}:
        state.dead_letter(entry, f"parent update failed: {link_err}")
        state.reset_lock_streak()
        return "continue", False
    if link_state == "already":
        return "ok", True
    return "ok", False


def ensure_child_exists_for_entry(
    entry: dict[str, Any],
    idx: int,
    state: Any,
    *,
    child: dict[str, Any],
    child_uuid: str,
    spawn_intent_id: str,
    export_uuid: Callable[[str], dict[str, Any]],
    import_child: Callable[[dict[str, Any]], tuple[bool, str]],
    is_lock_error: Callable[[str], bool],
    diag: Callable[[str], None],
    requeue_or_dead_letter_for_lock: Callable[[dict[str, Any], int, Any], bool],
) -> tuple[str, bool]:
    export_res = export_uuid(child_uuid)
    imported = False
    if not export_res.get("exists"):
        if export_res.get("retryable"):
            if spawn_intent_id:
                diag(f"task lock active; requeue (intent={spawn_intent_id})")
            return ("break", False) if requeue_or_dead_letter_for_lock(entry, idx, state) else ("continue", False)
        ok, err = import_child(child)
        if not ok:
            if is_lock_error(err):
                return ("break", False) if requeue_or_dead_letter_for_lock(entry, idx, state) else ("continue", False)
            if not export_uuid(child_uuid).get("exists"):
                if spawn_intent_id:
                    diag(f"child import failed (intent={spawn_intent_id}): {err}")
                else:
                    diag(f"child import failed: {err}")
                state.dead_letter(entry, f"child import failed: {err}")
                state.reset_lock_streak()
                return "continue", False
        imported = True

    if imported:
        confirm_res = export_uuid(child_uuid)
        if not confirm_res.get("exists"):
            if confirm_res.get("retryable"):
                return ("break", False) if requeue_or_dead_letter_for_lock(entry, idx, state) else ("continue", False)
            if spawn_intent_id:
                diag(f"child missing after import (intent={spawn_intent_id})")
            state.dead_letter(entry, "child missing after import")
            state.reset_lock_streak()
            return "continue", False

    return "ok", imported


def apply_parent_update_for_entry(
    entry: dict[str, Any],
    idx: int,
    state: Any,
    *,
    parent_uuid: str,
    child_short: str,
    expected_parent_nextlink: str,
    child_uuid: str,
    spawn_intent_id: str,
    parent_linked_already: bool,
    imported: bool,
    update_parent_nextlink: Callable[[str, str, str | None], tuple[bool, str]],
    is_lock_error: Callable[[str], bool],
    cleanup_orphan_child: Callable[[str, str], None],
    diag: Callable[[str], None],
    requeue_or_dead_letter_for_lock: Callable[[dict[str, Any], int, Any], bool],
) -> str:
    if not (parent_uuid and child_short) or parent_linked_already:
        return "ok"

    ok, err = update_parent_nextlink(parent_uuid, child_short, expected_parent_nextlink)
    if ok:
        return "ok"

    if spawn_intent_id:
        diag(f"parent update failed (intent={spawn_intent_id}): {parent_uuid}")
    else:
        diag(f"parent update failed: {parent_uuid}")
    if err in {"parent export locked", "parent lock busy"} or is_lock_error(err):
        return "break" if requeue_or_dead_letter_for_lock(entry, idx, state) else "continue"
    if imported:
        cleanup_orphan_child(child_uuid, spawn_intent_id)
    state.dead_letter(entry, f"parent update failed: {err}")
    state.reset_lock_streak()
    return "continue"

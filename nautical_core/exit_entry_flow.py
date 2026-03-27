from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nautical_core.exit_models import (
        ExitApplyParentUpdateServices,
        ExitEnsureChildServices,
        ExitEntryContext,
        ExitPrecheckServices,
    )


def precheck_parent_link_state(
    ctx: ExitEntryContext,
    *,
    services: ExitPrecheckServices,
) -> tuple[str, bool]:
    if not (ctx.parent_uuid and ctx.child_short):
        return "ok", False

    link_state, link_err = services.parent_nextlink_state(
        ctx.parent_uuid,
        ctx.child_short,
        ctx.expected_parent_nextlink,
    )
    if link_state == "locked":
        return ("break", False) if services.requeue_or_dead_letter_for_lock(ctx.entry, ctx.idx, ctx.state) else ("continue", False)
    if link_state in {"conflict", "missing", "invalid"}:
        ctx.state.dead_letter(ctx.entry, f"parent update failed: {link_err}")
        ctx.state.reset_lock_streak()
        return "continue", False
    if link_state == "already":
        return "ok", True
    return "ok", False


def ensure_child_exists_for_entry(
    ctx: ExitEntryContext,
    *,
    services: ExitEnsureChildServices,
) -> tuple[str, bool]:
    export_res = services.export_uuid(ctx.child_uuid)
    imported = False
    if not export_res.get("exists"):
        if export_res.get("retryable"):
            if ctx.spawn_intent_id:
                services.diag(f"task lock active; requeue (intent={ctx.spawn_intent_id})")
            return ("break", False) if services.requeue_or_dead_letter_for_lock(ctx.entry, ctx.idx, ctx.state) else ("continue", False)
        ok, err = services.import_child(ctx.child)
        if not ok:
            if services.is_lock_error(err):
                return ("break", False) if services.requeue_or_dead_letter_for_lock(ctx.entry, ctx.idx, ctx.state) else ("continue", False)
            if not services.export_uuid(ctx.child_uuid).get("exists"):
                if ctx.spawn_intent_id:
                    services.diag(f"child import failed (intent={ctx.spawn_intent_id}): {err}")
                else:
                    services.diag(f"child import failed: {err}")
                ctx.state.dead_letter(ctx.entry, f"child import failed: {err}")
                ctx.state.reset_lock_streak()
                return "continue", False
        imported = True

    if imported:
        confirm_res = services.export_uuid(ctx.child_uuid)
        if not confirm_res.get("exists"):
            if confirm_res.get("retryable"):
                return ("break", False) if services.requeue_or_dead_letter_for_lock(ctx.entry, ctx.idx, ctx.state) else ("continue", False)
            if ctx.spawn_intent_id:
                services.diag(f"child missing after import (intent={ctx.spawn_intent_id})")
            ctx.state.dead_letter(ctx.entry, "child missing after import")
            ctx.state.reset_lock_streak()
            return "continue", False

    return "ok", imported


def apply_parent_update_for_entry(
    ctx: ExitEntryContext,
    *,
    parent_linked_already: bool,
    imported: bool,
    services: ExitApplyParentUpdateServices,
) -> str:
    if not (ctx.parent_uuid and ctx.child_short) or parent_linked_already:
        return "ok"

    ok, err = services.update_parent_nextlink(ctx.parent_uuid, ctx.child_short, ctx.expected_parent_nextlink)
    if ok:
        return "ok"

    if ctx.spawn_intent_id:
        services.diag(f"parent update failed (intent={ctx.spawn_intent_id}): {ctx.parent_uuid}")
    else:
        services.diag(f"parent update failed: {ctx.parent_uuid}")
    if err in {"parent export locked", "parent lock busy"} or services.is_lock_error(err):
        return "break" if services.requeue_or_dead_letter_for_lock(ctx.entry, ctx.idx, ctx.state) else "continue"
    if imported:
        services.cleanup_orphan_child(ctx.child_uuid, ctx.spawn_intent_id)
    ctx.state.dead_letter(ctx.entry, f"parent update failed: {err}")
    ctx.state.reset_lock_streak()
    return "continue"

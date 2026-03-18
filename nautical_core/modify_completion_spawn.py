from __future__ import annotations


def completion_build_and_spawn_child(
    new: dict,
    *,
    child_due,
    next_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt,
    build_child_from_parent,
    spawn_child_atomic,
    panel,
    print_task,
    diag=None,
) -> dict | None:
    try:
        child = build_child_from_parent(new, child_due, next_no, parent_short, kind, cpmax, until_dt)
    except Exception as exc:
        if callable(diag):
            diag(f"build child failed: {exc}")
        panel(
            "⛓ Chain error",
            [("Reason", "Failed to build next link")],
            kind="error",
        )
        print_task(new)
        return None

    deferred_spawn = False
    spawn_intent_id = None
    try:
        (
            child_short,
            stripped_attrs,
            verified,
            deferred_spawn,
            defer_reason,
            spawn_intent_id,
        ) = spawn_child_atomic(child, new)
        if not verified and not deferred_spawn:
            panel(
                "⛓ Chain warning",
                [("Reason", defer_reason or "Child spawn could not be verified; parent not updated")],
                kind="warning",
            )
            print_task(new)
            return None
    except Exception as exc:
        if callable(diag):
            diag(f"spawn child failed: {exc}")
        panel(
            "⛓ Chain error",
            [("Reason", "Failed to spawn next link")],
            kind="error",
        )
        print_task(new)
        return None

    if verified:
        new["nextLink"] = child_short

    return {
        "child": child,
        "child_short": child_short,
        "stripped_attrs": stripped_attrs,
        "verified": verified,
        "deferred_spawn": deferred_spawn,
        "spawn_intent_id": spawn_intent_id,
    }

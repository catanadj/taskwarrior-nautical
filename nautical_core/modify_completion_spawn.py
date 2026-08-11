from __future__ import annotations

from typing import Any

from nautical_core.chain_generation import CarryFieldError
from nautical_core.modify_models import CompletionSpawnResult, CompletionSpawnServices


def completion_build_and_spawn_child(
    new: dict[str, Any],
    *,
    child_due: Any,
    child_field: str = "due",
    next_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt: Any,
    planned_child: dict[str, Any] | None = None,
    services: CompletionSpawnServices,
) -> CompletionSpawnResult | None:
    build_child_from_parent = services.build_child_from_parent
    spawn_child_atomic = services.spawn_child_atomic
    panel = services.panel
    print_task = services.print_task
    diag = services.diag
    try:
        child = planned_child or build_child_from_parent(
            new,
            child_due,
            child_field,
            next_no,
            parent_short,
            kind,
            cpmax,
            until_dt,
        )
    except Exception as exc:
        if callable(diag):
            diag(f"build child failed: {exc}")
        reason = str(exc) if isinstance(exc, CarryFieldError) else "Failed to build next link"
        panel(
            "⛓ Chain error",
            [("Reason", reason)],
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
            review_reason = defer_reason or "Child spawn could not be verified; parent not updated"
            panel(
                "⛓ Chain warning",
                [("Reason", review_reason)],
                kind="warning",
            )
            print_task(new)
            return CompletionSpawnResult(
                child=child,
                child_short=child_short,
                stripped_attrs=stripped_attrs,
                verified=False,
                deferred_spawn=False,
                spawn_intent_id=spawn_intent_id,
                outcome_state="manual_review",
                reason=review_reason,
            )
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

    return CompletionSpawnResult(
        child=child,
        child_short=child_short,
        stripped_attrs=stripped_attrs,
        verified=verified,
        deferred_spawn=deferred_spawn,
        spawn_intent_id=spawn_intent_id,
    )

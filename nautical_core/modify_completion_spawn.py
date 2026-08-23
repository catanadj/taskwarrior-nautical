from __future__ import annotations

from typing import Any

from nautical_core.chain_generation import CarryFieldError
from nautical_core.modify_models import CompletionSpawnResult, CompletionSpawnServices
from nautical_core.task_models import TaskDraft, TaskPayload


def completion_build_and_spawn_child(
    new: TaskPayload,
    *,
    child_due: Any,
    child_field: str = "due",
    next_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt: Any,
    services: CompletionSpawnServices,
) -> CompletionSpawnResult | None:
    build_child_draft = services.build_child_draft
    spawn_child_atomic = services.spawn_child_atomic
    diag = services.diag
    try:
        child_draft = build_child_draft(
            new, child_due, child_field, next_no, parent_short, kind, cpmax, until_dt,
        )
        if not isinstance(child_draft, TaskDraft):
            raise TypeError("child builder returned a non-TaskDraft value")
        child = child_draft.to_mapping()
    except Exception as exc:
        if callable(diag):
            diag(f"build child failed: {exc}")
        reason = str(exc) if isinstance(exc, CarryFieldError) else "Failed to build next link"
        return CompletionSpawnResult(
            child={},
            child_short="",
            stripped_attrs=[],
            verified=False,
            deferred_spawn=False,
            spawn_intent_id=None,
            outcome_state="retryable",
            reason=reason,
        )

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
        ) = spawn_child_atomic(child_draft or child, new)
        if not verified and not deferred_spawn:
            review_reason = defer_reason or "Child spawn could not be verified; parent not updated"
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
        reason = str(exc).strip() or "Failed to spawn next link"
        return CompletionSpawnResult(
            child=child,
            child_short="",
            stripped_attrs=[],
            verified=False,
            deferred_spawn=False,
            spawn_intent_id=None,
            outcome_state="retryable",
            reason=reason,
        )

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

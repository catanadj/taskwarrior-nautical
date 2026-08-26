from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from nautical_core.modify_models import (
    CompletionLifecycleResult,
    CompletionLifecycleDiagnostic,
    CompletionComputeResult,
    CompletionFinalizeServices,
    CompletionPreflightContext,
    TaskView,
)
from nautical_core.task_changes import TaskTransition
from nautical_core.task_models import TaskObservation, TaskPayload


@dataclass(slots=True)
class CompletionFlowServices:
    """Typed collaborators for the complete-on-modify lifecycle boundary."""

    runtime_state: Callable[[], Any]
    prepare_recurrence: Callable[[TaskPayload, TaskPayload], tuple[str, str, str]]
    preserve_cp_relative_offsets: Callable[[TaskPayload, TaskPayload, str], None]
    preserve_native_until: Callable[[TaskPayload, TaskPayload, str], None]
    validate_native_until: Callable[[TaskPayload], None]
    validate_native_until_slots: Callable[[TaskPayload], None]
    now_utc: Callable[[], Any]
    preflight_context: Callable[[TaskPayload, Any, Any], CompletionPreflightContext | None]
    compute_next_and_limits: Callable[..., CompletionComputeResult | CompletionLifecycleResult | None]
    lifecycle_read_service: Any
    diag_count: Callable[[str, int], None]
    diag_lifecycle_result: Callable[[CompletionLifecycleResult], None]
    finalize_completion: Callable[..., CompletionLifecycleResult]
    finalize_services: CompletionFinalizeServices
    transition: TaskTransition | None = None


def handle_completion_modify(
    old: TaskPayload,
    new: TaskPayload,
    unit_of_work: Any,
    *,
    services: CompletionFlowServices,
) -> CompletionLifecycleResult | None:
    """Prepare, compute, and finalize one completed recurring task."""
    services.runtime_state().task_repository = unit_of_work.repository
    prepared = dict(new)
    new_cp, new_anchor, new_anchor_file = services.prepare_recurrence(old, prepared)
    services.preserve_cp_relative_offsets(old, prepared, new_cp)
    recurrence_fields = ("cp", "anchor", "anchor_file")
    has_previous_recurrence = (
        any(services.transition.old.field(field).raw_value() for field in recurrence_fields)
        if services.transition is not None
        else any(str(old.get(field) or "").strip() for field in recurrence_fields)
    )
    if has_previous_recurrence:
        recurrence_kind = "cp" if new_cp else "anchor_file" if new_anchor_file else "anchor"
        services.preserve_native_until(old, prepared, recurrence_kind)
    services.validate_native_until(prepared)
    services.validate_native_until_slots(prepared)
    new.clear()
    new.update(prepared)

    now_utc = services.now_utc()
    ctx = services.preflight_context(new, now_utc, unit_of_work.repository)
    if ctx is None:
        return None

    computed = services.compute_next_and_limits(
        new,
        ctx.kind,
        ctx.next_no,
        now_utc,
        preflight=ctx,
    )
    if computed is None:
        return None
    if isinstance(computed, CompletionLifecycleResult):
        services.diag_lifecycle_result(computed)
        return computed

    snapshot = ctx.chain_snapshot
    preloaded_chain = list(snapshot.rows)
    indexes = services.lifecycle_read_service.build_indexes(preloaded_chain)
    preloaded_chain_by_link, preloaded_chain_by_short = indexes.by_link, indexes.by_short
    if snapshot.mode == "full" and snapshot.loaded:
        services.lifecycle_read_service.replace_chain_cache(ctx.chain_id, preloaded_chain)
        services.diag_count("chain_cache_seeded", 1)

    result = services.finalize_completion(
        new=new,
        ctx=ctx,
        computed=computed,
        now_utc=now_utc,
        need_chain=snapshot.mode == "full",
        chain_snapshot_loaded=snapshot.loaded,
        preloaded_chain=preloaded_chain,
        preloaded_chain_by_link=preloaded_chain_by_link,
        preloaded_chain_by_short=preloaded_chain_by_short,
        chain_id=ctx.chain_id,
        services=services.finalize_services,
    )
    services.diag_lifecycle_result(result)
    return result


def _completion_diagnostic(
    ctx: CompletionPreflightContext,
    chain_id: str,
    *,
    stage: str,
    failure_kind: str = "",
    transition_id: str = "",
) -> CompletionLifecycleDiagnostic:
    transition = transition_id or f"{chain_id}:{ctx.base_no}->{ctx.next_no}"
    return CompletionLifecycleDiagnostic(
        transition_id=transition,
        chain_id=chain_id,
        parent_link=ctx.base_no,
        child_link=ctx.next_no,
        stage=stage,
        failure_kind=failure_kind,
    )


def _render_lifecycle_result(services: CompletionFinalizeServices, result: CompletionLifecycleResult, task: TaskView) -> None:
    """Keep presentation failures from suppressing the task response."""
    try:
        services.render_lifecycle_result(result, task)
    except Exception as exc:
        if services.diagnostic is not None:
            services.diagnostic(f"completion lifecycle presentation failed: {type(exc).__name__}: {exc}")


def finalize_completion_modify(
    *,
    new: TaskPayload,
    ctx: CompletionPreflightContext,
    computed: CompletionComputeResult,
    now_utc: Any,
    need_chain: bool,
    chain_snapshot_loaded: bool,
    preloaded_chain: list[TaskObservation],
    preloaded_chain_by_link: dict[int, list[TaskObservation]],
    preloaded_chain_by_short: dict[str, TaskObservation],
    chain_id: str,
    services: CompletionFinalizeServices,
) -> CompletionLifecycleResult:
    parent_short = ctx.parent_short
    base_no = ctx.base_no
    next_no = ctx.next_no
    kind = ctx.kind
    spawn_args = {
        "child_due": computed.child_due,
        "child_field": "scheduled" if isinstance(computed.meta, dict) and computed.meta.get("target_field") == "scheduled" else "due",
        "next_no": next_no,
        "parent_short": parent_short,
        "kind": kind,
        "cpmax": computed.cpmax,
        "until_dt": computed.until_dt,
        "lifecycle_plan": getattr(computed, "lifecycle_plan", None),
    }
    spawned = services.build_and_spawn_child(new, **spawn_args)
    if spawned is None:
        lifecycle_result = CompletionLifecycleResult(
            state="retryable",
            reason="completion child operation returned no result",
            diagnostic=_completion_diagnostic(
                ctx, chain_id, stage="spawn", failure_kind="missing_result"
            ),
        )
        _render_lifecycle_result(services, lifecycle_result, TaskView.from_mapping(new))
        services.print_task(new)
        return lifecycle_result
    spawn_state = str(getattr(spawned, "outcome_state", "applied") or "applied").strip().lower()
    if spawn_state != "applied":
        result_state = {
            "already_applied": "already_applied",
            "stale": "stale",
            "manual_review": "manual_review",
        }.get(spawn_state, "retryable")
        lifecycle_result = CompletionLifecycleResult(
            state=result_state,
            child_short=getattr(spawned, "child_short", ""),
            spawn_intent_id=getattr(spawned, "spawn_intent_id", None),
            reason=getattr(spawned, "reason", "") or "child spawn could not be completed",
            diagnostic=_completion_diagnostic(
                ctx,
                chain_id,
                stage="spawn",
                failure_kind=spawn_state,
                transition_id=str(getattr(spawned, "spawn_intent_id", "") or ""),
            ),
        )
        _render_lifecycle_result(services, lifecycle_result, TaskView.from_mapping(new))
        services.print_task(new)
        return lifecycle_result

    child = spawned.child
    child_view = TaskView.from_mapping(child)
    services.seed_runtime_lookup_tasks(new, child)
    child_short = spawned.child_short
    stripped_attrs = spawned.stripped_attrs
    deferred_spawn = spawned.deferred_spawn
    spawn_intent_id = spawned.spawn_intent_id
    lifecycle_result = CompletionLifecycleResult(
        state="queued" if deferred_spawn else "applied",
        child_short=child_short,
        deferred_spawn=deferred_spawn,
        spawn_intent_id=spawn_intent_id,
        diagnostic=_completion_diagnostic(
            ctx,
            chain_id,
            stage="queued" if deferred_spawn else "finalize",
            transition_id=spawn_intent_id or "",
        ),
    )

    chain = list(preloaded_chain)
    chain_by_link = preloaded_chain_by_link
    chain_by_short = preloaded_chain_by_short
    read_service = services.lifecycle_read_service
    if chain_id:
        try:
            if chain and spawned.verified and not deferred_spawn:
                chain = read_service.merge_spawned_child(
                    chain,
                    parent_task=new,
                    child_task=child,
                    child_short=child_short,
                    short_uuid=lambda value: str(value or "")[:8],
                )
                indexes = read_service.build_indexes(chain)
                chain_by_link, chain_by_short = indexes.by_link, indexes.by_short
                read_service.replace_chain_cache(chain_id, chain)
            elif need_chain and not chain_snapshot_loaded:
                chain = read_service.get_chain_export(chain_id)
                if chain:
                    indexes = read_service.build_indexes(chain)
                    chain_by_link, chain_by_short = indexes.by_link, indexes.by_short
                    read_service.replace_chain_cache(chain_id, chain)
        except Exception as exc:
            if services.diagnostic is not None:
                services.diagnostic(f"completion chain refresh failed: {type(exc).__name__}: {exc}")

    state = services.modify_chain_state()
    state.panel_chain_by_link = chain_by_link
    state.panel_chain_by_short = chain_by_short
    state.panel_chain_snapshot_loaded = True

    # The lifecycle read cache remains in its operational row form.  Panels
    # receive immutable views so presentation cannot mutate chain history.
    presentation_chain_by_short = {
        short: TaskView.from_observation(row)
        for short, row in (chain_by_short or {}).items()
    }

    analytics_advice = None
    integrity_warnings = None
    if chain and services.show_analytics:
        try:
            analytics_advice = services.chain_health_advice(chain, kind, new, style=services.analytics_style)
        except Exception as exc:
            analytics_advice = None
            if services.diagnostic is not None:
                services.diagnostic(
                    f"completion analytics failed: {type(exc).__name__}: {exc}"
                )
    if chain and services.check_integrity:
        try:
            integrity_warnings = services.chain_integrity_warnings(chain, expected_chain_id=chain_id)
        except Exception as exc:
            integrity_warnings = None
            if services.diagnostic is not None:
                services.diagnostic(
                    f"completion integrity presentation failed: {type(exc).__name__}: {exc}"
                )

    if kind in {"anchor", "anchor_file"}:
        services.render_anchor_completion_feedback(
            new=new,
            child=child_view,
            child_due=computed.child_due,
            child_short=child_short,
            next_no=next_no,
            parent_short=parent_short,
            cap_no=computed.cap_no,
            finals=computed.finals,
            now_utc=now_utc,
            until_dt=computed.until_dt,
            until_cap_no=computed.until_cap_no,
            dnf=computed.dnf,
            meta=computed.meta,
            stripped_attrs=stripped_attrs,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
            lifecycle_result=lifecycle_result,
            chain_by_short=presentation_chain_by_short,
            analytics_advice=analytics_advice,
            integrity_warnings=integrity_warnings,
            base_no=base_no,
        )
    else:
        services.render_cp_completion_feedback(
            new=new,
            child=child_view,
            child_due=computed.child_due,
            child_short=child_short,
            next_no=next_no,
            parent_short=parent_short,
            cap_no=computed.cap_no,
            finals=computed.finals,
            now_utc=now_utc,
            until_dt=computed.until_dt,
            until_cap_no=computed.until_cap_no,
            meta=computed.meta,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
            lifecycle_result=lifecycle_result,
            chain_by_short=presentation_chain_by_short,
            analytics_advice=analytics_advice,
            integrity_warnings=integrity_warnings,
            base_no=base_no,
        )

    services.print_task(new)
    services.diag_summary()
    return lifecycle_result


__all__ = (
    "CompletionFinalizeServices",
    "finalize_completion_modify",
)

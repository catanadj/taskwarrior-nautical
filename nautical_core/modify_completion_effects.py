"""Completion preflight and occurrence-limit effects for on-modify."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .task_models import TaskPayload


def link_numbers_or_fail(host: Any, new: TaskPayload):
    return host._module("modify_completion_preflight").completion_link_numbers_or_fail(
        new,
        coerce_int=host.core.coerce_int,
        max_link_number=host.core.MAX_LINK_NUMBER,
        panel=host._panel,
        print_task=host._print_task,
    )


def kind_or_stop(host: Any, new: TaskPayload, now_utc: datetime):
    return host._module("modify_completion_preflight").completion_kind_or_stop(
        new,
        now_utc,
        panel=host._panel,
        print_task=host._print_task,
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
    )


def chain_id_or_fail(host: Any, new: TaskPayload) -> str | None:
    preflight = host._module("modify_completion_preflight")
    return preflight.completion_chain_id_or_fail(
        new,
        panel=host._panel,
        print_task=host._print_task,
    )


def existing_next_or_fail(host: Any, new: TaskPayload, next_no: int, chain_snapshot, repository) -> bool:
    preflight = host._module("modify_completion_preflight")
    return preflight.completion_existing_next_or_fail(
        new,
        next_no,
        existing_next_lookup=lambda task, link: repository.exact_child_slot(
            str(task.get("chainID") or ""), link
        ),
        short=host._short,
        panel=host._panel,
        print_task=host._print_task,
    )


def _snapshot_mode(host: Any) -> str:
    if host._SHOW_ANALYTICS or host._CHECK_CHAIN_INTEGRITY:
        return "full"
    mode = str(getattr(host.core, "PANEL_MODE", "rich") or "rich").strip().lower()
    if mode in {"line", "minimal", "quiet", "text"}:
        return "next"
    return "recent"


def chain_snapshot(host: Any, chain_id: str, base_no: int, next_no: int, repository):
    del base_no, next_no
    models = host._module("modify_models")
    from .integration_models import Absent, Found, Unavailable

    snapshot = repository.chain_snapshot(chain_id)
    if isinstance(snapshot, Found):
        value = getattr(snapshot.value, "rows", snapshot.value)
        task_models = host._module("task_models")
        rows = [
            row if hasattr(row, "to_mapping") else task_models.TaskObservation.from_mapping(
                row, source_query=f"chain:{chain_id}:completion"
            )
            for row in value
        ]
        loaded, error = True, ""
    elif isinstance(snapshot, Absent):
        rows, loaded, error = [], True, ""
    elif isinstance(snapshot, Unavailable):
        rows, loaded = [], False
        error = snapshot.evidence.detail or snapshot.evidence.kind.value
    else:
        rows, loaded, error = [], False, "typed chain read returned an unsupported result"
    return models.CompletionChainSnapshot(
        mode=_snapshot_mode(host), rows=rows, loaded=loaded, chain_id=chain_id, error=error
    )


def preflight_context(host: Any, new: TaskPayload, now_utc: datetime, repository):
    preflight = host._module("modify_completion_preflight")
    runtime = host._module("modify_runtime")
    services = runtime.build_preflight_services(
        short=host._short,
        completion_link_numbers_or_fail=lambda task: link_numbers_or_fail(host, task),
        completion_kind_or_stop=lambda task, clock: kind_or_stop(host, task, clock),
        completion_chain_id_or_fail=lambda task: chain_id_or_fail(host, task),
        completion_chain_snapshot=lambda chain_id, base_no, next_no: chain_snapshot(host, chain_id, base_no, next_no, repository),
        completion_existing_next_or_fail=lambda task, next_no, snapshot: existing_next_or_fail(host, task, next_no, snapshot, repository),
    )
    return preflight.completion_preflight_context(new, now_utc, services=services)


def compute_child_due(host: Any, new: TaskPayload, kind: str):
    compute = host._module("modify_completion_compute")
    generation = host._chain_generation_service()
    codec = host._module("task_codec")
    models = host._module("task_models")

    def typed_task(task):
        return models.NauticalTask.from_observation(
            codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify completion")
        )

    def handle_terminal(exc) -> None:
        message = host.core._import_sibling("scheduler_models").occurrence_exhaustion_message(exc)
        if exc.is_date_limit:
            host._module("modify_presentation_effects").ensure_terminal_chain_off(host, new, "complete")
            try:
                host._module("modify_diagnostics_effects").end_chain_summary(host, new, message, host._workflow_now_utc(), current_task=new)
            except Exception as summary_exc:
                host._diag(f"terminal chain summary failed: {summary_exc}")
                host._panel("⛔ Nautical chain stopped", [("Reason", message), ("Task", host._short(new.get("uuid")) or "–")], kind="summary")
            host._print_task(new)
            return
        host._panel("⛔ Chain error", [("Scheduler", message), ("Fix", "Use a less sparse rule or adjust its search limits.")], kind="error")
        host._print_task(new)

    return compute.completion_compute_child_due(
        new,
        kind,
        compute_anchor_child_due=lambda task: generation.compute_anchor_child_due(typed_task(task)),
        compute_cp_child_due=lambda task: generation.compute_cp_child_due(typed_task(task)),
        panel=host._panel,
        print_task=host._print_task,
        diag=host._diag,
        on_terminal=handle_terminal,
    )


def until_or_fail(host: Any, new: TaskPayload, now_utc: datetime):
    compute = host._module("modify_completion_compute")
    return compute.completion_until_or_fail(
        new, now_utc,
        safe_parse_datetime=host._safe_parse_datetime,
        validate_until_not_past=host._validate_until_not_past,
        panel=host._panel,
        print_task=host._print_task,
    )


def until_guard_or_stop(host: Any, new: TaskPayload, child_due, until_dt, now_utc: datetime) -> bool:
    return host._module("modify_completion_compute").completion_until_guard_or_stop(
        new, child_due, until_dt, now_utc,
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
        print_task=host._print_task,
    )


def require_child_due_or_fail(host: Any, new: TaskPayload, child_due) -> bool:
    return host._module("modify_completion_compute").completion_require_child_due_or_fail(
        new, child_due, panel=host._panel, print_task=host._print_task
    )


def warn_unreasonable_duration(host: Any, new: TaskPayload, child_due, until_dt, now_utc: datetime) -> None:
    host._module("modify_completion_compute").completion_warn_unreasonable_duration(
        new, child_due, until_dt, now_utc,
        validate_chain_duration_reasonable=host._validate_chain_duration_reasonable,
        panel=host._panel,
    )


def caps(host: Any, kind: str, new: TaskPayload, child_due, dnf):
    return host._module("modify_completion_compute").completion_caps(
        kind, new, child_due, dnf,
        coerce_int=host.core.coerce_int,
        dtparse=host._dtparse,
        estimate_cp_final_by_max=host._estimate_cp_final_by_max,
        estimate_anchor_final_by_max=host._estimate_anchor_final_by_max,
        cap_from_until_cp=host._cap_from_until_cp,
        cap_from_until_anchor=host._cap_from_until_anchor,
    )


def cap_guard_or_stop(host: Any, new: TaskPayload, next_no: int, cap_no: int | None, now_utc: datetime) -> bool:
    return host._module("modify_completion_compute").completion_cap_guard_or_stop(
        new, next_no, cap_no, now_utc,
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
        print_task=host._print_task
    )


def compute_next_and_limits(host: Any, new: TaskPayload, kind: str, next_no: int, now_utc: datetime, *, preflight=None):
    compute = host._module("modify_completion_compute")
    runtime = host._module("modify_runtime")
    services = runtime.build_compute_services(
        completion_compute_child_due=lambda value, value_kind: compute_child_due(host, value, value_kind),
        completion_until_or_fail=lambda value, clock: until_or_fail(host, value, clock),
        completion_until_guard_or_stop=lambda value, due, until, clock: until_guard_or_stop(host, value, due, until, clock),
        completion_require_child_due_or_fail=lambda value, due: require_child_due_or_fail(host, value, due),
        completion_warn_unreasonable_duration=lambda value, due, until, clock: warn_unreasonable_duration(host, value, due, until, clock),
        completion_caps=lambda value_kind, value, due, dnf: caps(host, value_kind, value, due, dnf),
        completion_cap_guard_or_stop=lambda value, number, cap, clock: cap_guard_or_stop(host, value, number, cap, clock),
    )
    computed = compute.completion_compute_next_and_limits(new, kind, next_no, now_utc, services=services)
    if computed is None:
        return None
    if isinstance(computed, host._module("modify_models").CompletionLifecycleResult):
        return computed
    if not str(new.get("uuid") or "").strip() or not str(new.get("chainID") or "").strip():
        return computed
    fingerprint_fn = getattr(host.core, "scheduler_config_fingerprint", None)
    return compute.attach_lifecycle_plan(
        new, computed, next_no, now_utc,
        preflight=preflight,
        generation=host._chain_generation_service(),
        scheduler_fingerprint=fingerprint_fn() if callable(fingerprint_fn) else "",
        compare_datetimes=host._compare_datetimes,
        invalid_relative_carry_reason=host._module("chain_integrity_lifecycle").invalid_relative_carry_reason,
        lifecycle_planner=host._module("lifecycle_planner"),
        lifecycle_models=host._module("lifecycle_models"),
        modify_models=host._module("modify_models"),
        end_chain_summary=lambda task, reason, now, current_task=None: host._module("modify_diagnostics_effects").end_chain_summary(host, task, reason, now, current_task),
        ensure_terminal_chain_off=lambda task, event=None: host._module("modify_presentation_effects").ensure_terminal_chain_off(host, task, event),
        panel=host._panel,
        print_task=host._print_task,
        diag=host._diag,
    )


def build_and_spawn_child(host: Any, new: TaskPayload, **kwargs):
    spawn = host._module("modify_completion_spawn")
    runtime = host._module("modify_runtime")
    generation = host._chain_generation_service()
    codec = host._module("task_codec")
    models = host._module("task_models")

    def build_child_draft(task, *args, **inner_kwargs):
        typed_task = models.NauticalTask.from_observation(
            codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify completion")
        )
        return generation.build_child_draft(typed_task, *args, **inner_kwargs)

    services = runtime.build_spawn_services(
        build_child_draft=build_child_draft,
        spawn_child_atomic=host._spawn_child_atomic,
        panel=host._panel,
        print_task=host._print_task,
        diag=host._diag,
    )
    return spawn.completion_build_and_spawn_child(new, services=services, **kwargs)


__all__ = (
    "link_numbers_or_fail", "kind_or_stop", "chain_id_or_fail", "existing_next_or_fail", "chain_snapshot", "preflight_context",
    "compute_child_due", "until_or_fail", "until_guard_or_stop", "require_child_due_or_fail",
    "warn_unreasonable_duration", "caps", "cap_guard_or_stop",
    "compute_next_and_limits", "build_and_spawn_child",
)

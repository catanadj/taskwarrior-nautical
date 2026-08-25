"""Presentation and terminal effects used by the typed on-modify routes."""

from __future__ import annotations

from typing import Any

from .task_models import TaskPayload


def _task_view(models: Any, value: Any):
    """Use typed views for real tasks while keeping panel-only fixtures permissive."""
    if isinstance(value, models.TaskView):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("status", "pending")
        payload.setdefault("link", 1)
        return models.TaskView.from_mapping(payload)
    return value


def chain_colour_for_task(host: Any, task: TaskPayload, kind: str) -> str:
    """Resolve the configured presentation colour for a chain root."""
    return host.core.chain_colour_root(kind, host._root_uuid_from(task))


def future_style_for_chain(host: Any, task: TaskPayload, kind: str) -> str:
    """Return static or per-chain future styling for timeline presentation."""
    if not host._CHAIN_COLOR_PER_CHAIN:
        return "dark_orange" if kind == "cp" else "cyan"
    return chain_colour_for_task(host, task, kind)


def render_recurrence_updated_panel(host: Any, changes: list[tuple[str, str, str]], new: TaskPayload) -> None:
    feedback = host._module("modify_feedback")
    models = host._module("modify_models")
    add_validation = host.core._import_sibling("add_validation")
    feedback.render_recurrence_updated_panel(
        changes,
        models.TaskView.from_mapping(new),
        parse_datetime=host.core.parse_dt_any,
        format_local=host._fmtlocal,
        describe_native_until_carry=add_validation.describe_native_until_carry,
        to_local=host.core.to_local,
        coerce_int=host.core.coerce_int,
        describe_anchor=host.core.describe_anchor_expr,
        resolve_omit_presets=host.core.resolve_omit_presets,
        first_recurrence_target=lambda task, source: first_recurrence_target(host, task, source),
        panel_mode=getattr(host.core, "PANEL_MODE", "rich"),
        strip_markup=host.core.strip_rich_markup,
        panel=host._panel,
    )


def first_recurrence_target(host: Any, new: TaskPayload, source: str):
    task_view = host._module("modify_models").TaskView.from_mapping(new)
    return host._module("modify_completion_compute").first_recurrence_target(
        task_view,
        source,
        parse_datetime=host.core.parse_dt_any,
        format_datetime=host.core.fmt_isoz,
        generation_service=host._chain_generation_service,
    )


def recurrence_enabled_rows(host: Any, new: TaskPayload, source: str) -> list[tuple[str, str]]:
    task_view = host._module("modify_models").TaskView.from_mapping(new)
    return host._module("modify_feedback").recurrence_enabled_rows(
        task_view,
        source,
        describe_anchor=host.core.describe_anchor_expr,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        first_recurrence_target=lambda task, value: first_recurrence_target(host, task, value),
        format_local=host._fmtlocal,
    )


def render_cp_schedule_adjusted_panel(host: Any, adjustment) -> None:
    host._module("modify_feedback").render_cp_schedule_adjusted_panel(
        adjustment,
        format_local=host._fmtlocal,
        semantic_diff_value=host._module("modify_validation_effects").semantic_diff_value,
        format_offset=host._fmt_td_dd_hhmm,
        panel=host._panel,
    )


def render_explicit_timing_order_warning(host: Any, new: TaskPayload, changed_fields: tuple[str, ...]) -> None:
    task = host._module("modify_models").TaskView.from_mapping(new)
    host._module("modify_feedback").render_explicit_timing_order_warning(
        task,
        changed_fields,
        format_offset=host._fmt_td_dd_hhmm,
        panel=host._panel,
    )


def render_disabled_chain_summary(host: Any, old: TaskPayload, new: TaskPayload, decision) -> None:
    reason = str(getattr(decision, "reason", decision))
    if not (old.get("chainID") or new.get("chainID")):
        return
    models = host._module("modify_models")
    old_view = models.TaskView.from_mapping(old)
    new_view = models.TaskView.from_mapping(new)
    now_utc = host._workflow_now_utc()
    try:
        host._module("modify_diagnostics_effects").end_chain_summary(
            host, old_view, reason, now_utc, current_task=new_view
        )
    except Exception as exc:
        host._diag(f"removed recurrence chain summary failed: {exc}")
        host._panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", reason),
                ("Root", host._module("modify_queries").cached_format_root_and_age(host, old_view, now_utc)),
                ("Task", host._short(old_view.get("uuid")) or "–"),
            ],
            kind="summary",
        )


def ensure_terminal_chain_off(host: Any, task: TaskPayload, event: str | None = None) -> bool:
    """Validate and apply one idempotent terminal patch for hook-side stops."""
    if event:
        lifecycle_models = host._module("lifecycle_models")
        lifecycle_planner = host._module("lifecycle_planner")
        task_codec = host._module("task_codec")
        lifecycle_planner.terminal_plan_for_snapshot(
            lifecycle_models.TaskSnapshot.from_observation(
                task_codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify terminal")
            ),
            lifecycle_models.LifecycleEvent(event),
        )
    return host._module("modify_lifecycle").ensure_terminal_chain_off(task)


def render_anchor_completion_feedback(host: Any, **kwargs) -> None:
    calendar_feedback = host.importlib.import_module("nautical_core.calendar_feedback")
    feedback = host._module("modify_feedback")
    models = host._module("modify_models")
    feedback.orchestrate_anchor_completion_feedback(
        **{
            **kwargs,
            "new": _task_view(models, kwargs["new"]),
            "child": _task_view(models, kwargs["child"]),
            "core": host.core,
            "panel": host._panel,
            "calendar_feedback": calendar_feedback,
            "panel_diagnostics": host._module("panel_diagnostics"),
            "modify_models": models,
            "modify_runtime": host._module("modify_runtime"),
            "build_runtime_services": lambda: build_runtime_services(host),
        }
    )


def render_cp_completion_feedback(host: Any, **kwargs) -> None:
    feedback = host._module("modify_feedback")
    models = host._module("modify_models")
    feedback.orchestrate_cp_completion_feedback(
        **{
            **kwargs,
            "new": _task_view(models, kwargs["new"]),
            "child": _task_view(models, kwargs["child"]),
            "core": host.core,
            "panel_diagnostics": host._module("panel_diagnostics"),
            "modify_models": models,
            "modify_runtime": host._module("modify_runtime"),
            "build_runtime_services": lambda: build_runtime_services(host),
        }
    )


def render_lifecycle_result(host: Any, result, task) -> None:
    """Render one finalized non-success outcome without deciding its state."""
    state = str(getattr(result, "state", "retryable") or "retryable").strip().lower()
    title = "⛓ Chain warning" if state == "manual_review" else "⛓ Chain error"
    rows = [("Result", state.replace("_", " ").title())]
    reason = str(getattr(result, "reason", "") or "").strip()
    if reason:
        rows.append(("Reason", reason))
    child_short = str(getattr(result, "child_short", "") or "").strip()
    if child_short:
        rows.append(("Child", child_short))
    intent_id = str(getattr(result, "spawn_intent_id", "") or "").strip()
    if intent_id:
        rows.append(("Intent", intent_id))
    host._panel(title, rows, kind="warning" if state == "manual_review" else "error")


def timeline_lines(host: Any, kind: str, task, child_due_utc, child_short: str, dnf, **kwargs) -> list[str]:
    if not host._require_core():
        return []
    return host._module("modify_timeline").timeline_lines_for_task(
        kind, task, child_due_utc, child_short, dnf, **kwargs,
        core=host.core, max_iterations=host._MAX_ITERATIONS,
        future_style_for_chain=lambda task, kind: future_style_for_chain(host, task, kind),
        collect_prev_two=host._collect_prev_two, dtparse=host._dtparse,
        fmt_on_time_delta=host._fmt_on_time_delta, fmtlocal=host._fmtlocal,
        short=host._short, tolocal=host._tolocal,
        next_occurrence_after_local_dt=host._next_occurrence_after_local_dt,
        to_local_cached=host._to_local_cached, safe_parse_datetime=host._safe_parse_datetime,
        format_gap=host._module("modify_timeline").format_gap,
        module_loader=host._module, omit_dnf_from_parent=host._omit_dnf_from_parent,
        recurrence_evaluator_for_task=host._recurrence_evaluator_for_task,
        scheduler_service_for_task=host._scheduler_service_for_task,
    )


def build_runtime_services(host: Any):
    runtime = host._module("modify_runtime")
    return runtime.ModifyRuntimeServices(
        state=host._modify_runtime_state(), core=host.core,
        debug_wait_sched=host._DEBUG_WAIT_SCHED,
        last_wait_sched_debug=host._LAST_WAIT_SCHED_DEBUG,
        diag_enabled=host.os.environ.get("NAUTICAL_DIAG") == "1",
        format_root_and_age=lambda task, now: host._module("modify_queries").cached_format_root_and_age(host, task, now),
        append_next_wait_sched_rows=host._append_next_wait_sched_rows,
        timeline_lines=getattr(host, "_timeline_lines", lambda *args, **kwargs: timeline_lines(host, *args, **kwargs)),
        show_timeline_gaps=host._SHOW_TIMELINE_GAPS,
        root_uuid_from=host._root_uuid_from, short=host._short,
        format_next_anchor_rows=host._format_next_anchor_rows,
        format_next_cp_rows=host._format_next_cp_rows,
        format_line_preview=host._format_line_preview,
        panel_line=host._panel_line, text_line=host._text_line,
        panel=host._panel, print_task=host._print_task, diag=host._diag,
        chain_color_per_chain=host._CHAIN_COLOR_PER_CHAIN,
        chain_colour_for_task=lambda task, kind: chain_colour_for_task(host, task, kind),
        strip_quotes=host._strip_quotes, human_delta=host._human_delta,
    )


__all__ = (
    "render_recurrence_updated_panel", "first_recurrence_target", "recurrence_enabled_rows",
    "render_cp_schedule_adjusted_panel", "render_explicit_timing_order_warning",
    "render_disabled_chain_summary", "ensure_terminal_chain_off",
    "render_anchor_completion_feedback", "render_cp_completion_feedback", "render_lifecycle_result",
    "timeline_lines", "build_runtime_services",
)

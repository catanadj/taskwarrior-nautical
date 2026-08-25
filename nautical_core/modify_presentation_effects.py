"""Presentation and terminal effects used by the typed on-modify routes."""

from __future__ import annotations

from typing import Any

from .task_models import TaskPayload


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
        semantic_diff_value=host._semantic_diff_value,
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
        host._end_chain_summary(old_view, reason, now_utc, current_task=new_view)
    except Exception as exc:
        host._diag(f"removed recurrence chain summary failed: {exc}")
        host._panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", reason),
                ("Root", host._format_root_and_age(old_view, now_utc)),
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


__all__ = (
    "render_recurrence_updated_panel", "first_recurrence_target", "recurrence_enabled_rows",
    "render_cp_schedule_adjusted_panel", "render_explicit_timing_order_warning",
    "render_disabled_chain_summary", "ensure_terminal_chain_off",
)

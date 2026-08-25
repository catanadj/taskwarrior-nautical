"""Presentation formatting helpers for typed on-modify effects."""

from __future__ import annotations

from typing import Any


def human_delta(host: Any, start, end, prefer_months: bool = True):
    try:
        return host.core.humanize_delta(start, end, use_months_days=bool(prefer_months))
    except TypeError:
        return host.core.humanize_delta(start, end)


def on_time_delta(host: Any, due_dt, end_dt, tol_secs: int = 60):
    if not (due_dt and end_dt):
        return ""
    diff = (end_dt - due_dt).total_seconds()
    if diff > tol_secs:
        text = human_delta(host, due_dt, end_dt, False)
        return f"[yellow](+{text.replace('overdue by ', '').replace('in ', '')} late)[/]"
    if diff < -tol_secs:
        text = human_delta(host, end_dt, due_dt, False)
        return f"[cyan](-{text.replace('in ', '')} early)[/]"
    return "[green](on time)[/]"


def line_preview(host: Any, link_no: int, task: dict, child_due_utc, child_short: str, now_utc, **kwargs) -> str:
    task_view = host._module("modify_models").TaskView.from_mapping(task)
    return host._module("modify_feedback").format_line_preview(
        link_no, task_view, child_due_utc, child_short, now_utc,
        core=host.core,
        format_local=host._fmtlocal,
        on_time_delta=lambda due, end, tol=60: on_time_delta(host, due, end, tol),
        human_delta=lambda start, end, prefer=True: human_delta(host, start, end, prefer),
        **kwargs,
    )


__all__ = ("human_delta", "on_time_delta", "line_preview")

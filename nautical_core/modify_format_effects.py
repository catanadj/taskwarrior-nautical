"""Presentation formatting helpers for typed on-modify effects."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from .task_models import TaskPayload


@dataclass(frozen=True, slots=True)
class HumanDeltaPort:
    humanize: Any


def human_delta(port: HumanDeltaPort, start, end, prefer_months: bool = True):
    return port.humanize(start, end, use_months_days=bool(prefer_months))


def on_time_delta(port: HumanDeltaPort, due_dt, end_dt, tol_secs: int = 60):
    if not (due_dt and end_dt):
        return ""
    diff = (end_dt - due_dt).total_seconds()
    if diff > tol_secs:
        text = human_delta(port, due_dt, end_dt, False)
        return f"[yellow](+{text.replace('overdue by ', '').replace('in ', '')} late)[/]"
    if diff < -tol_secs:
        text = human_delta(port, end_dt, due_dt, False)
        return f"[cyan](-{text.replace('in ', '')} early)[/]"
    return "[green](on time)[/]"


def line_preview(host: Any, link_no: int, task: TaskPayload, child_due_utc, child_short: str, now_utc, **kwargs) -> str:
    delta_port = HumanDeltaPort(host.core.humanize_delta)
    task_view = host._module("modify_models").TaskView.from_mapping(task)
    return host._module("modify_feedback").format_line_preview(
        link_no, task_view, child_due_utc, child_short, now_utc,
        core=host.core,
        format_local=host._fmtlocal,
        on_time_delta=lambda due, end, tol=60: on_time_delta(delta_port, due, end, tol),
        human_delta=lambda start, end, prefer=True: human_delta(delta_port, start, end, prefer),
        **kwargs,
    )


__all__ = ("human_delta", "on_time_delta", "line_preview")

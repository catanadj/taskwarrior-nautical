"""Presentation formatting helpers for typed on-modify effects."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from .task_models import TaskPayload


@dataclass(frozen=True, slots=True)
class HumanDeltaPort:
    humanize: Any


@dataclass(frozen=True, slots=True)
class LinePreviewPorts:
    task_view: Any
    format_line_preview: Any
    core: Any
    format_local: Any
    delta: HumanDeltaPort


def line_preview_ports_for(host: Any) -> LinePreviewPorts:
    return LinePreviewPorts(
        task_view=host._module("modify_models").TaskView,
        format_line_preview=host._module("modify_feedback").format_line_preview,
        core=host.core,
        format_local=host._fmtlocal,
        delta=HumanDeltaPort(host.core.humanize_delta),
    )


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


def line_preview(ports: LinePreviewPorts, link_no: int, task: TaskPayload, child_due_utc, child_short: str, now_utc, **kwargs) -> str:
    task_view = ports.task_view.from_mapping(task)
    return ports.format_line_preview(
        link_no, task_view, child_due_utc, child_short, now_utc,
        core=ports.core,
        format_local=ports.format_local,
        on_time_delta=lambda due, end, tol=60: on_time_delta(ports.delta, due, end, tol),
        human_delta=lambda start, end, prefer=True: human_delta(ports.delta, start, end, prefer),
        **kwargs,
    )


__all__ = ("HumanDeltaPort", "LinePreviewPorts", "line_preview_ports_for", "human_delta", "on_time_delta", "line_preview")

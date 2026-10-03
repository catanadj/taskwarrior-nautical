"""Presentation formatting helpers for typed on-modify effects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol

from .modify_models import PreviewLineFormatter
from .task_models import TaskPayload


class MarkupStripper(Protocol):
    def strip_rich_markup(self, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class HumanDeltaPort:
    humanize: Callable[[datetime, datetime, bool], str]


@dataclass(frozen=True, slots=True)
class LinePreviewPorts:
    format_line_preview: PreviewLineFormatter
    core: MarkupStripper
    format_local: Callable[[Any], str]
    delta: HumanDeltaPort


def line_preview_ports_for(host: Any) -> LinePreviewPorts:
    return LinePreviewPorts(
        format_line_preview=host._module("modify_feedback").format_line_preview,
        core=host.core,
        format_local=host._fmtlocal,
        delta=HumanDeltaPort(host.core.humanize_delta),
    )


def human_delta(
    port: HumanDeltaPort,
    start: datetime,
    end: datetime,
    prefer_months: bool = True,
) -> str:
    return port.humanize(start, end, bool(prefer_months))


def on_time_delta(port: HumanDeltaPort, due_dt: Any, end_dt: Any, tol_secs: int = 60) -> Any:
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


def line_preview(
    ports: LinePreviewPorts,
    link_no: int,
    task: TaskPayload,
    child_due_utc: Any,
    child_short: str,
    now_utc: Any,
    **kwargs: Any,
) -> str:
    def format_on_time_delta(due: Any, end: Any) -> str:
        return on_time_delta(ports.delta, due, end)

    def format_human_delta(start: Any, end: Any, prefer: bool = True) -> str:
        return human_delta(ports.delta, start, end, prefer)

    return ports.format_line_preview(
        link_no, task, child_due_utc, child_short, now_utc,
        core=ports.core,
        format_local=ports.format_local,
        on_time_delta=format_on_time_delta,
        human_delta=format_human_delta,
        **kwargs,
    )


__all__ = ("HumanDeltaPort", "LinePreviewPorts", "line_preview_ports_for", "human_delta", "on_time_delta", "line_preview")

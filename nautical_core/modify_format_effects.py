"""Presentation formatting helpers for typed on-modify effects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal, Protocol

from .modify_models import MarkupStripper, PreviewLineFormatter
from .task_models import TaskPayload


@dataclass(frozen=True, slots=True)
class HumanDeltaPort:
    humanize: Callable[[datetime, datetime, bool], str]


@dataclass(frozen=True, slots=True)
class LinePreviewPorts:
    format_line_preview: PreviewLineFormatter
    core: MarkupStripper
    format_local: Callable[[datetime], str]
    delta: HumanDeltaPort


class _ModifyFeedback(Protocol):
    format_line_preview: PreviewLineFormatter


class _LinePreviewCore(MarkupStripper, Protocol):
    humanize_delta: Callable[[datetime, datetime, bool], str]


class LinePreviewHost(Protocol):
    core: _LinePreviewCore
    _fmtlocal: Callable[[datetime], str]

    def _module(self, name: Literal["modify_feedback"]) -> _ModifyFeedback: ...


def line_preview_ports_for(host: LinePreviewHost) -> LinePreviewPorts:
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


def on_time_delta(
    port: HumanDeltaPort,
    due_dt: datetime | None,
    end_dt: datetime | None,
    tol_secs: int = 60,
) -> str:
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
    child_due_utc: datetime | None,
    child_short: str,
    now_utc: datetime,
    *,
    child_field: str = "due",
    cap_no: int | None = None,
    until_dt: datetime | None = None,
    until_no: int | None = None,
    child_until_dt: datetime | None = None,
    kind: str = "cp",
    minimal: bool = False,
) -> str:
    def format_on_time_delta(due: Any, end: Any) -> str:
        return on_time_delta(ports.delta, due, end)

    def format_human_delta(start: Any, end: Any, prefer: bool = True) -> str:
        return human_delta(ports.delta, start, end, prefer)

    return ports.format_line_preview(
        link_no, task, child_due_utc, child_short, now_utc,
        child_field=child_field,
        cap_no=cap_no,
        until_dt=until_dt,
        until_no=until_no,
        child_until_dt=child_until_dt,
        kind=kind,
        minimal=minimal,
        core=ports.core,
        format_local=ports.format_local,
        on_time_delta=format_on_time_delta,
        human_delta=format_human_delta,
    )


__all__ = ("HumanDeltaPort", "LinePreviewPorts", "line_preview_ports_for", "human_delta", "on_time_delta", "line_preview")

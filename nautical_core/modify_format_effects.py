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


__all__ = ("human_delta", "on_time_delta")

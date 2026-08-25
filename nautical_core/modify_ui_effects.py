"""UI effects for the typed on-modify workflow."""

from __future__ import annotations

import sys
from typing import Any


def print_task(host: Any, task) -> None:
    hook_results = host._module("hook_results")
    if host.core is None:
        try:
            host._load_core()
        except Exception:
            hook_results.emit_passthrough_json(task)
            return
    hook_results.emit_task_json(task, sanitize=True, core=host.core)


def panel(
    host: Any,
    title,
    rows,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    label_style: str | None = None,
):
    if host.core is None:
        try:
            host._load_core()
        except Exception:
            try:
                sys.stderr.write(f"[nautical] {title}\n")
            except Exception:
                pass
            return
    themes = host.core.panel_themes()
    theme = dict(themes.get(kind, themes.get("info", {})))
    if border_style:
        theme["border"] = border_style
    if title_style:
        theme["title"] = title_style
    if label_style:
        theme["label"] = label_style
    themes[kind] = theme
    host.core.render_panel(
        title,
        rows,
        kind=kind,
        panel_mode=host.core.PANEL_MODE,
        live_duration_ms=getattr(host.core, "LIVE_PANEL_DURATION_MS", 160),
        live_footer=getattr(host.core, "LIVE_PANEL_FOOTER", "NAUTICAL"),
        fast_color=host.core.FAST_COLOR,
        themes=themes,
        allow_line=True,
        line_force_rich_kinds={"summary"},
        label_width_min=6,
        label_width_max=14,
    )


def panel_line(host: Any, title: str, line: str, *, kind: str = "info", border_style=None, title_style=None, markup_body=False) -> None:
    host.core.panel_line(
        title,
        line,
        kind=kind,
        themes=host.core.panel_themes(),
        border_style=border_style,
        title_style=title_style,
        markup_body=markup_body,
    )


def text_line(host: Any, line: str, *, kind: str = "info", markup_body: bool = False) -> None:
    host.core.text_line(line, kind=kind, markup_body=markup_body)


__all__ = ("print_task", "panel", "panel_line", "text_line")

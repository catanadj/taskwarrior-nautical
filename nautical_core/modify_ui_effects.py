"""UI effects for the typed on-modify workflow."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Callable, Literal, Protocol

from .task_models import TaskPayload


class EmitTaskJson(Protocol):
    def __call__(
        self,
        task: TaskPayload,
        *,
        sanitize: bool = False,
        core: Any = None,
        prof: Any = None,
    ) -> None: ...


class _HookResults(Protocol):
    def emit_passthrough_json(self, task: TaskPayload) -> None: ...

    def emit_task_json(
        self,
        task: TaskPayload,
        *,
        sanitize: bool = False,
        core: Any = None,
        prof: Any = None,
    ) -> None: ...


class UIEffectsHost(Protocol):
    @property
    def core(self) -> object | None: ...

    def _load_core(self) -> None: ...

    def _module(self, name: Literal["hook_results"]) -> _HookResults: ...


@dataclass(frozen=True)
class UIEffectsPorts:
    """Process-boundary capabilities required by modify UI effects."""

    core: Callable[[], Any]
    load_core: Callable[[], None]
    override: Callable[[str], Callable[..., Any] | None]
    emit_passthrough_json: Callable[[TaskPayload], None]
    emit_task_json: EmitTaskJson
    stderr_write: Callable[[str], int]


def ui_ports_for(host: UIEffectsHost) -> UIEffectsPorts:
    """Adapt the hook host once at the composition boundary."""
    values = getattr(host, "_values", None)
    if values is None and hasattr(host, "__dict__"):
        values = vars(host)
    values_map = values if isinstance(values, dict) else {}

    def test_override(name: str) -> Callable[..., Any] | None:
        override = values_map.get(name)
        is_root_delegate = callable(override) and getattr(override, "__name__", "") == name and (
            getattr(getattr(override, "__code__", None), "co_filename", "") == values_map.get("__file__")
        )
        return override if callable(override) and not is_root_delegate else None

    return UIEffectsPorts(
        core=lambda: host.core,
        load_core=host._load_core,
        override=test_override,
        emit_passthrough_json=lambda task: host._module("hook_results").emit_passthrough_json(
            task
        ),
        emit_task_json=(
            lambda task, *, sanitize=False, core=None, prof=None: host._module(
                "hook_results"
            ).emit_task_json(task, sanitize=sanitize, core=core, prof=prof)
        ),
        stderr_write=sys.stderr.write,
    )


def print_task(ports: UIEffectsPorts, task: Any) -> None:
    override = ports.override("_print_task")
    if override is not None:
        return override(task)
    core = ports.core()
    if core is None:
        try:
            ports.load_core()
        except Exception:
            ports.emit_passthrough_json(task)
            return
        core = ports.core()
    ports.emit_task_json(task, sanitize=True, core=core)


def panel(
    ports: UIEffectsPorts,
    title: Any,
    rows: Any,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    label_style: str | None = None,
) -> Any:
    override = ports.override("_panel")
    if override is not None:
        return override(title, rows, kind=kind)
    core = ports.core()
    if core is None:
        try:
            ports.load_core()
        except Exception:
            try:
                ports.stderr_write(f"[nautical] {title}\n")
            except (OSError, UnicodeError, ValueError):
                pass
            return
        core = ports.core()
    ui = core._import_sibling("ui")
    themes = ui.panel_themes()
    theme = dict(themes.get(kind, themes.get("info", {})))
    if border_style:
        theme["border"] = border_style
    if title_style:
        theme["title"] = title_style
    if label_style:
        theme["label"] = label_style
    themes[kind] = theme
    core.render_panel(
        title,
        rows,
        kind=kind,
        panel_mode=core.PANEL_MODE,
        live_duration_ms=getattr(core, "LIVE_PANEL_DURATION_MS", 160),
        live_footer=getattr(core, "LIVE_PANEL_FOOTER", "NAUTICAL"),
        fast_color=core.FAST_COLOR,
        themes=themes,
        allow_line=True,
        line_force_rich_kinds={"summary"},
        label_width_min=6,
        label_width_max=14,
    )


def panel_line(ports: UIEffectsPorts, title: str, line: str, *, kind: str = "info", border_style: Any = None, title_style: Any = None, markup_body: bool = False) -> None:
    override = ports.override("_panel_line")
    if override is not None:
        return override(title, line, kind=kind, border_style=border_style, title_style=title_style, markup_body=markup_body)
    core = ports.core()
    core._import_sibling("ui").panel_line(
        title,
        line,
        kind=kind,
        themes=core.panel_themes(),
        border_style=border_style,
        title_style=title_style,
        markup_body=markup_body,
    )


def text_line(ports: UIEffectsPorts, line: str, *, kind: str = "info", markup_body: bool = False) -> None:
    override = ports.override("_text_line")
    if override is not None:
        return override(line, kind=kind, markup_body=markup_body)
    ports.core()._import_sibling("ui").text_line(line, kind=kind, markup_body=markup_body)


__all__ = ("UIEffectsPorts", "ui_ports_for", "print_task", "panel", "panel_line", "text_line")

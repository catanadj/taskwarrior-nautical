from __future__ import annotations

import os
import re
import sys
import time


_RICH_TAG_RE = re.compile(r"\[/\]|\[/?[A-Za-z0-9_ ]+\]")
_DEFAULT_LIVE_PANEL_DURATION_MS = 160
_MAX_LIVE_PANEL_DURATION_MS = 1000
_LIVE_PANEL_MAX_HEIGHT_RATIO = 0.75
_LIVE_PANEL_MIN_SPARE_LINES = 3
_LIVE_ANIMATION_USED = False


def strip_rich_markup(s: str) -> str:
    # Strip simple Rich tags; preserve bracketed literals with non-word chars.
    if not s:
        return s
    return _RICH_TAG_RE.sub("", s)


def term_width_stderr(default: int = 80) -> int:
    try:
        w = os.get_terminal_size(sys.stderr.fileno()).columns
    except Exception:
        w = default
    return max(40, min(70, int(w)))


def fast_color_enabled(force: bool | None = None, fast_color: bool = True) -> bool:
    if not sys.stderr.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if force is not None:
        return bool(force)
    return bool(fast_color)


def ansi(code: str) -> str:
    return f"\x1b[{code}m"


def emit_wrapped(prefix: str, text: str, width: int, style: str | None = None) -> None:
    if text is None:
        text = ""
    text = str(text)
    text = strip_rich_markup(text)

    avail = max(10, width - len(prefix))
    parts = text.splitlines() if "\n" in text else [text]

    for pi, raw_line in enumerate(parts):
        line = raw_line.rstrip("\n")
        if not line:
            sys.stderr.write((prefix.rstrip() if pi == 0 else (" " * len(prefix))) + "\n")
            continue

        cur = ""
        for token in line.split(" "):
            if not cur:
                cur = token
            elif len(cur) + 1 + len(token) <= avail:
                cur += " " + token
            else:
                out = cur
                if style:
                    sys.stderr.write(prefix + style + out + ansi("0") + "\n")
                else:
                    sys.stderr.write(prefix + out + "\n")
                prefix = " " * len(prefix)
                avail = max(10, width - len(prefix))
                cur = token

        if cur:
            if style:
                sys.stderr.write(prefix + style + cur + ansi("0") + "\n")
            else:
                sys.stderr.write(prefix + cur + "\n")
        prefix = " " * len(prefix)
        avail = max(10, width - len(prefix))


def emit_line(msg: str) -> None:
    if not msg:
        return
    try:
        sys.stderr.write(msg + "\n")
    except Exception:
        pass


def _plain_text_report_lines(text: str) -> list[str]:
    parts = str(text or "").splitlines() if "\n" in str(text or "") else [str(text or "")]
    out: list[str] = []
    for part in parts:
        line = strip_rich_markup(part)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out


def text_line(line: str, *, kind: str = "info", markup_body: bool = False) -> None:
    raw_lines = str(line or "").splitlines() if "\n" in str(line or "") else [str(line or "")]
    plain_lines = _plain_text_report_lines(line)
    if not plain_lines:
        return
    if not fast_color_enabled():
        for text in plain_lines:
            emit_line(text)
        return
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console(file=sys.stderr, force_terminal=True, soft_wrap=True)
        for raw in raw_lines:
            if not str(raw).strip():
                continue
            text_obj = Text.from_markup(raw) if markup_body else Text(str(raw))
            console.print(text_obj)
        return
    except Exception:
        pass
    color_code = {
        "preview_anchor": "36",
        "preview_cp": "33",
        "warning": "33",
        "error": "31",
        "summary": "35",
    }.get(str(kind or "info"), "36")
    try:
        for text in plain_lines:
            sys.stderr.write(f"{ansi(color_code)}{text}{ansi('0')}\n")
    except Exception:
        for text in plain_lines:
            emit_line(text)


def _wrap_prefixed_lines(prefix: str, text: str, width: int) -> list[str]:
    text = strip_rich_markup("" if text is None else str(text))
    avail = max(10, width - len(prefix))
    parts = text.splitlines() if "\n" in text else [text]
    out: list[str] = []
    current_prefix = prefix
    for raw_line in parts:
        line = raw_line.rstrip("\n")
        if not line:
            out.append(current_prefix.rstrip())
            current_prefix = " " * len(prefix)
            avail = max(10, width - len(current_prefix))
            continue
        cur = ""
        for token in line.split(" "):
            if not cur:
                cur = token
            elif len(cur) + 1 + len(token) <= avail:
                cur += " " + token
            else:
                out.append(current_prefix + cur)
                current_prefix = " " * len(prefix)
                avail = max(10, width - len(current_prefix))
                cur = token
        if cur:
            out.append(current_prefix + cur)
        current_prefix = " " * len(prefix)
        avail = max(10, width - len(current_prefix))
    return out or [prefix.rstrip()]


def _box_write_line(text: str, inner_width: int, style: str | None = None) -> None:
    raw = strip_rich_markup(text)
    pad = max(0, inner_width - len(raw))
    if style:
        sys.stderr.write(f"│ {style}{raw}{ansi('0')}{' ' * pad} │\n")
    else:
        sys.stderr.write(f"│ {raw}{' ' * pad} │\n")


def panel_line_from_rows(title, rows) -> str:
    title_txt = strip_rich_markup(str(title))
    if not rows:
        return title_txt
    for k, v in rows:
        if k is None:
            continue
        ktxt = strip_rich_markup(str(k))
        vtxt = strip_rich_markup(str(v)) if v is not None else ""
        if not vtxt:
            continue
        return f"{title_txt} — {ktxt}: {vtxt}"
    return title_txt


def panel_line(
    title: str,
    line: str,
    *,
    kind: str = "info",
    themes: dict | None = None,
    border_style: str | None = None,
    title_style: str | None = None,
    markup_body: bool = False,
) -> None:
    try:
        if not sys.stderr.isatty():
            raise RuntimeError("no tty")
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        emit_line(strip_rich_markup(line) if markup_body else line)
        return

    theme = (themes or {}).get(kind) or (themes or {}).get("info") or {}
    border = border_style or theme.get("border", "blue")
    tstyle = title_style or theme.get("title", "cyan")

    console = Console(file=sys.stderr, force_terminal=True)
    body = Text.from_markup(line) if markup_body else Text(line)
    console.print(
        Panel(
            body,
            title=Text(str(title), style=f"bold {tstyle}"),
            border_style=border,
            expand=False,
            padding=(0, 1),
        )
    )


def _normalize_panel_mode(
    panel_mode: str,
    *,
    kind: str,
    allow_line: bool,
    line_force_rich_kinds: set[str] | None,
) -> str:
    mode = str(panel_mode or "").strip().lower()
    if mode in {"plain"}:
        mode = "fast"
    if mode == "quiet":
        mode = "text"
    if mode == "minimal":
        mode = "line"
    if mode == "line" and not allow_line:
        mode = "rich"
    if mode == "line" and line_force_rich_kinds and kind in line_force_rich_kinds:
        mode = "rich"
    return mode


def _panel_label_width(rows, label_width_min: int, label_width_max: int) -> int:
    label_w = 0
    for k, _v in rows:
        if k is None:
            continue
        klen = len(str(k))
        if klen > label_w:
            label_w = klen
    return min(label_width_max, max(label_width_min, label_w))


def _panel_style_for_row(k: str, v: str, *, palette: dict[str, str]) -> str | None:
    lk = k.lower()
    lsv = (v or "").lower()
    if k.strip().lower() == "pattern":
        return palette["CYAN"]
    if "natural" in lk:
        return palette["DIM"]
    if k.strip().lower() in {"basis", "root"}:
        return palette["DIM"]
    if k.strip().lower() in {"first due", "next due"}:
        if "overdue" in lsv or "late" in lsv:
            return palette["RED"]
        return palette["GREEN"]
    if "warning" in lk:
        return palette["YELLOW"]
    if "error" in lk:
        return palette["RED"]
    if lk.startswith("chain"):
        return palette["DIM"] + palette["GREEN"]
    return None


def _panel_emit_timeline_row(label: str, value: str, inner_width: int, label_w: int) -> None:
    prefix0 = f"{label:<{label_w}} "
    lines = [ln for ln in value.splitlines() if ln.strip()] if "\n" in value else ([value] if value else [])
    if not lines:
        for wrapped in _wrap_prefixed_lines(prefix0, "", inner_width):
            _box_write_line(wrapped, inner_width)
        return
    for wrapped in _wrap_prefixed_lines(prefix0, lines[0], inner_width):
        _box_write_line(wrapped, inner_width)
    for ln in lines[1:]:
        for wrapped in _wrap_prefixed_lines(" " * len(prefix0), ln, inner_width):
            _box_write_line(wrapped, inner_width)


def _render_panel_fast(
    title,
    rows,
    *,
    fast_color: bool,
    label_width_min: int,
    label_width_max: int,
    force_color: bool | None = None,
) -> None:
    width = term_width_stderr()
    use_color = fast_color_enabled(force=force_color, fast_color=fast_color)
    palette = {
        "RESET": ansi("0"),
        "BOLD": ansi("1") if use_color else "",
        "DIM": ansi("2") if use_color else "",
        "CYAN": ansi("36") if use_color else "",
        "GREEN": ansi("32") if use_color else "",
        "RED": ansi("31") if use_color else "",
        "YELLOW": ansi("33") if use_color else "",
    }

    inner_width = max(20, width - 4)
    title_txt = strip_rich_markup(str(title))
    top_inner = f"─ {title_txt} "
    if len(top_inner) < inner_width:
        top_inner += "─" * (inner_width - len(top_inner))
    else:
        top_inner = top_inner[:inner_width]
    sys.stderr.write(f"┌{top_inner}┐\n")

    label_w = _panel_label_width(rows, label_width_min, label_width_max)
    for k, v in rows:
        if k is None:
            _box_write_line("", inner_width)
            continue

        label = strip_rich_markup(str(k))
        text = "" if v is None else strip_rich_markup(str(v))
        if label.lower().startswith("timeline"):
            _panel_emit_timeline_row(label, text, inner_width, label_w)
            continue

        prefix = f"{label:<{label_w}} "
        style = _panel_style_for_row(label, text, palette=palette)
        for wrapped in _wrap_prefixed_lines(prefix, text, inner_width):
            _box_write_line(wrapped, inner_width, style=style)

    sys.stderr.write(f"└{'─' * inner_width}┘\n")


def _build_rich_panel(
    title,
    rows,
    *,
    kind: str,
    themes: dict | None,
    live: bool = False,
    active_row: int | None = None,
) -> object:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    theme = (themes or {}).get(kind) or (themes or {}).get("info") or {}
    border = theme.get("border", "blue")
    tstyle = theme.get("title", "cyan")
    lstyle = theme.get("label", "cyan")
    palette = {
        "DIM": "dim",
        "CYAN": "cyan",
        "GREEN": "green",
        "RED": "red",
        "YELLOW": "yellow",
    }

    t = Table.grid(padding=(0, 1), expand=False)
    if live:
        t.min_width = len("NAUTICAL") + 2
    t.add_column(style=f"bold {lstyle}", no_wrap=True, justify="right")
    t.add_column(style="white")

    for row_index, (k, v) in enumerate(rows):
        row_active = live and row_index == active_row
        marker = "▸ " if row_active else ("  " if live else "")
        if k is None:
            if not live:
                t.add_row("", v or "")
                continue
            value_raw = "" if v is None else str(v)
            try:
                value_text = Text.from_markup(value_raw)
            except Exception:
                value_text = Text(value_raw)
            marker_text = Text(marker)
            if row_active:
                marker_text.stylize(f"bold {tstyle}")
                value_text.stylize("bold")
            t.add_row(marker_text, value_text)
            continue

        label_text = Text(marker)
        label_start = len(label_text)
        label_text.append(str(k))
        value_raw = "" if v is None else str(v)
        try:
            value_text = Text.from_markup(value_raw)
        except Exception:
            value_text = Text(value_raw)
        lk = str(k).lower()
        if "warning" in lk:
            label_text.stylize("bold yellow", label_start)
        elif "error" in lk:
            label_text.stylize("bold red", label_start)
        elif "note" in lk:
            label_text.stylize("italic cyan", label_start)

        row_style = _panel_style_for_row(str(k), value_raw, palette=palette)
        if row_style:
            value_text.stylize(row_style)
            if lk in {"basis", "root"}:
                label_text.stylize(row_style, label_start)
        if row_active:
            if label_start:
                label_text.stylize(f"bold {tstyle}", 0, label_start)
            label_text.stylize("bold", label_start)
            value_text.stylize("bold")

        t.add_row(label_text, value_text)

    panel_kwargs = {}
    if live:
        panel_kwargs = {
            "subtitle": Text("NAUTICAL", style=f"dim {tstyle}"),
            "subtitle_align": "right",
        }

    return Panel(
        t,
        title=Text(title, style=f"bold {tstyle}"),
        border_style=f"bold {border}" if live and active_row is not None else border,
        expand=False,
        padding=(0, 1),
        **panel_kwargs,
    )


def _render_panel_rich(
    title,
    rows,
    *,
    kind: str,
    themes: dict | None,
) -> bool:
    try:
        if not sys.stderr.isatty():
            raise RuntimeError("no tty")
        from rich.console import Console
    except Exception:
        return False

    try:
        panel = _build_rich_panel(title, rows, kind=kind, themes=themes)
    except ImportError:
        return False
    console = Console(file=sys.stderr, force_terminal=True)
    console.print(panel)
    return True


def _render_panel_live(
    title,
    rows,
    *,
    kind: str,
    themes: dict | None,
    duration_ms: int | float = _DEFAULT_LIVE_PANEL_DURATION_MS,
) -> bool:
    global _LIVE_ANIMATION_USED

    try:
        if not sys.stderr.isatty() or os.environ.get("TERM", "").strip().lower() == "dumb":
            raise RuntimeError("no tty")
        from rich.console import Console
        from rich.live import Live

        row_list = list(rows)
        settled_panel = _build_rich_panel(
            title,
            row_list,
            kind=kind,
            themes=themes,
            live=True,
        )
        console = Console(file=sys.stderr, force_terminal=True)
    except Exception:
        return False

    effective_duration_ms = _live_animation_duration_ms(kind, duration_ms)
    if effective_duration_ms > 0 and _live_panel_too_tall(console, settled_panel):
        effective_duration_ms = 0.0
    if effective_duration_ms <= 0:
        try:
            console.print(settled_panel)
            return True
        except Exception:
            return False

    try:
        reveal_frames = _live_reveal_frames(row_list)
        if len(reveal_frames) <= 1:
            console.print(settled_panel)
            return True
        first_rows, first_active_row = reveal_frames[0]
        panel = _build_rich_panel(
            title,
            first_rows,
            kind=kind,
            themes=themes,
            live=True,
            active_row=first_active_row,
        )
        reveal_delays = _live_reveal_delays(len(reveal_frames), effective_duration_ms)
    except Exception:
        try:
            console.print(settled_panel)
            return True
        except Exception:
            return False

    started = False
    try:
        with Live(
            panel,
            console=console,
            auto_refresh=False,
            transient=False,
        ) as live:
            started = True
            _LIVE_ANIMATION_USED = True
            try:
                for (frame_rows, active_row), reveal_delay in zip(
                    reveal_frames[1:],
                    reveal_delays[:-1],
                ):
                    panel = _build_rich_panel(
                        title,
                        frame_rows,
                        kind=kind,
                        themes=themes,
                        live=True,
                        active_row=active_row,
                    )
                    time.sleep(reveal_delay)
                    live.update(panel, refresh=True)
                time.sleep(reveal_delays[-1])
                live.update(settled_panel, refresh=True)
            except Exception:
                try:
                    live.update(settled_panel, refresh=True)
                except Exception:
                    pass
        return True
    except Exception:
        return started


def _normalized_live_duration_ms(duration_ms: int | float) -> float:
    try:
        duration = float(duration_ms)
    except Exception:
        duration = float(_DEFAULT_LIVE_PANEL_DURATION_MS)
    return max(0.0, min(float(_MAX_LIVE_PANEL_DURATION_MS), duration))


def _live_panel_too_tall(console, panel) -> bool:
    try:
        terminal_height = max(1, int(console.height))
        rendered_height = len(console.render_lines(panel, pad=False))
        max_animated_height = max(
            1,
            min(
                terminal_height - _LIVE_PANEL_MIN_SPARE_LINES,
                int(terminal_height * _LIVE_PANEL_MAX_HEIGHT_RATIO),
            ),
        )
        return rendered_height > max_animated_height
    except Exception:
        return False


def _live_animation_duration_ms(kind: str, duration_ms: int | float) -> float:
    duration = _normalized_live_duration_ms(duration_ms)
    if duration <= 0 or str(kind or "").strip().lower() == "error" or _LIVE_ANIMATION_USED:
        return 0.0
    if str(kind or "").strip().lower() == "warning":
        return duration / 2.0
    return duration


def _reset_live_animation_state() -> None:
    global _LIVE_ANIMATION_USED
    _LIVE_ANIMATION_USED = False


def _live_reveal_frames(rows) -> list[tuple[list[tuple[object, object]], int]]:
    frames: list[tuple[list[tuple[object, object]], int]] = []
    completed: list[tuple[object, object]] = []
    for row_index, (key, value) in enumerate(rows):
        raw_value = "" if value is None else str(value)
        value_lines = raw_value.splitlines()
        if len(value_lines) <= 1:
            frames.append(([*completed, (key, value)], row_index))
        else:
            for visible_lines in range(1, len(value_lines) + 1):
                partial_value = "\n".join(value_lines[:visible_lines])
                frames.append(([*completed, (key, partial_value)], row_index))
        completed.append((key, value))
    return frames


def _live_reveal_delays(
    frame_count: int,
    duration_ms: int | float = _DEFAULT_LIVE_PANEL_DURATION_MS,
) -> list[float]:
    frame_count = max(0, int(frame_count))
    total_seconds = _normalized_live_duration_ms(duration_ms) / 1000.0
    if frame_count <= 1 or total_seconds <= 0:
        return []
    weights = [
        0.75 + (0.5 * index / (frame_count - 1))
        for index in range(frame_count)
    ]
    weight_total = sum(weights)
    return [total_seconds * weight / weight_total for weight in weights]


def render_panel(
    title,
    rows,
    *,
    kind: str = "info",
    panel_mode: str = "rich",
    live_duration_ms: int | float = _DEFAULT_LIVE_PANEL_DURATION_MS,
    fast_color: bool = True,
    themes: dict | None = None,
    allow_line: bool = True,
    line_force_rich_kinds: set[str] | None = None,
    label_width_min: int = 6,
    label_width_max: int = 14,
):
    """
    Render a panel using Rich or a fast fallback.
    """
    try:
        mode = _normalize_panel_mode(
            panel_mode,
            kind=kind,
            allow_line=allow_line,
            line_force_rich_kinds=line_force_rich_kinds,
        )

        if mode == "line":
            line = panel_line_from_rows(title, rows)
            if line:
                panel_line(title, line, kind=kind, themes=themes)
            return

        if mode == "text":
            line = panel_line_from_rows(title, rows)
            if line:
                text_line(line, kind=kind, markup_body=False)
            return

        if mode == "fast":
            _render_panel_fast(
                title,
                rows,
                fast_color=fast_color,
                label_width_min=label_width_min,
                label_width_max=label_width_max,
                force_color=None,
            )
            return

        if mode == "live":
            live_rows = list(rows)
            if _render_panel_live(
                title,
                live_rows,
                kind=kind,
                themes=themes,
                duration_ms=live_duration_ms,
            ):
                return
            rows = live_rows

        # Rich mode (default)
        if _render_panel_rich(title, rows, kind=kind, themes=themes):
            return
        _render_panel_fast(
            title,
            rows,
            fast_color=fast_color,
            label_width_min=label_width_min,
            label_width_max=label_width_max,
            force_color=False,
        )
    except Exception as e:
        try:
            sys.stderr.write(f"[{strip_rich_markup(str(title))}]\n")
            for k, v in rows or []:
                if k is None:
                    continue
                sys.stderr.write(f"  {strip_rich_markup(str(k))}: {strip_rich_markup(str(v))}\n")
            if os.environ.get("NAUTICAL_DIAG") == "1":
                sys.stderr.write(f"[nautical] panel error: {e}\n")
        except Exception:
            pass

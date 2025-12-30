#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chained next-link spawner for Taskwarrior.

- Works for classic cp (cp/chainMax/chainUntil) and anchors (anchor/anchor_mode).
- Cap logic unified (chainMax, chainUntil -> numeric cap_no).
- Spawns child with `task import -` (preserves annotation timestamps).
- Timeline is capped and marks (last link).
"""

import sys, json, os, uuid, subprocess, importlib
from pathlib import Path
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
import re
from decimal import Decimal, InvalidOperation
import shlex
import time as _time
try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None

from typing import Optional


# Optional: DST-aware local TZ helpers (used by some carry-forward variants)
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None



 # Ensure hook IO supports Unicode (emoji, symbols) in JSON output.
 # Python's json.dumps() defaults to ensure_ascii=True, which escapes non-ASCII
 # as "\\uXXXX". Prefer human-readable UTF-8 JSON for hook passthrough.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

_MAX_CHAIN_WALK = 500  # max tasks to walk backwards in chain
_MAX_UUID_LOOKUPS = 50  # max individual UUID exports before giving up
_MAX_ITERATIONS = 2000  # prevent infinite loops in stepping functions
_MIN_FUTURE_WARN = 365 * 2  # warn if chain extends >2 years


_MAX_SPAWN_ATTEMPTS = 3
_SPAWN_RETRY_DELAY = 0.1  # seconds between retries

# ------------------------------------------------------------------------------
# Colour per chain toggle - performance in termux has a significative reduction. 
# ------------------------------------------------------------------------------

_CHAIN_COLOR_PER_CHAIN = os.environ.get("NAUTICAL_CHAIN_COLOR", "").strip().lower() in {
    "chain",
    "per-chain",
    "per",
    "1",
    "yes",
    "true",
    "on",
}

# ------------------------------------------------------------------------------
# Show timeline gaps 
# ------------------------------------------------------------------------------
_SHOW_TIMELINE_GAPS = os.environ.get("NAUTICAL_TIMELINE_GAPS", "").strip().lower() not in {
    "0", "no", "false", "off", "none"
}

# ------------------------------------------------------------------------------
# Debug: wait/scheduled carry-forward
# Set NAUTICAL_DEBUG_WAIT_SCHED=1 to include carry computations in the feedback panel.
# ------------------------------------------------------------------------------
_DEBUG_WAIT_SCHED = os.environ.get("NAUTICAL_DEBUG_WAIT_SCHED", "").strip().lower() in {
    "1", "yes", "true", "on"
}
_LAST_WAIT_SCHED_DEBUG: dict[str, dict] = {}


def _fmt_td_dd_hhmm(delta: timedelta) -> str:
    """Format a timedelta as ±Dd HHh:MMm (UTC-seconds based; seconds omitted)."""
    try:
        total = int(delta.total_seconds())
    except Exception:
        return str(delta)
    sign = "-" if total < 0 else "+"
    total = abs(total)
    # truncate seconds
    total_minutes = total // 60
    dd, rem_m = divmod(total_minutes, 1440)  # 24*60
    hh, mm = divmod(rem_m, 60)
    return f"{sign}{dd}d {hh:02}h:{mm:02}m"


def _append_next_wait_sched_rows(
    fb: list[tuple[str, str]],
    nxt: dict,
    nxt_due_utc: datetime,
) -> None:
    """Append Scheduled/Wait lines for the next link showing local time and Δ to due."""
    if not (isinstance(nxt_due_utc, datetime) and nxt_due_utc):
        return

    dt_s = _dtparse(nxt.get("scheduled"))
    dt_w = _dtparse(nxt.get("wait"))

    for fld, label, dt in (
        ("scheduled", "Scheduled", dt_s),
        ("wait", "Wait", dt_w),
    ):
        if not isinstance(dt, datetime):
            continue
        delta_s = _fmt_td_dd_hhmm(dt - nxt_due_utc)
        fb.append((label, f"{core.fmt_dt_local(dt)}  (Δ {delta_s})"))

    # Informative order validation: due > scheduled > wait
    # This can be violated when due is auto-assigned but scheduled/wait are user-specified.
    issues: list[str] = []
    if isinstance(dt_s, datetime) and dt_s > nxt_due_utc:
        issues.append(f"scheduled is after due by {_fmt_td_dd_hhmm(dt_s - nxt_due_utc)}")
    if isinstance(dt_w, datetime) and dt_w > nxt_due_utc:
        issues.append(f"wait is after due by {_fmt_td_dd_hhmm(dt_w - nxt_due_utc)}")
    if isinstance(dt_s, datetime) and isinstance(dt_w, datetime) and dt_w > dt_s:
        issues.append(f"wait is after scheduled by {_fmt_td_dd_hhmm(dt_w - dt_s)}")

    if issues:
        fb.append((
            "⚠ Wait/Sched",
            "Expected order: due > scheduled > wait. " + "; ".join(issues),
        ))
        fb.append((
            "⚠ Wait/Sched",
            "This can happen when due is auto-assigned; adjust scheduled/wait if undesired.",
        ))

# ------------------------------------------------------------------------------
# Locate nautical_core
# ------------------------------------------------------------------------------
HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

TW_DATA_DIR = Path(os.environ.get("TASKDATA") or str(TW_DIR)).expanduser()

# ------------------------------------------------------------------------------
# Deferred next-link spawn queue (used when nested `task import` times out due to TW lock)
# ------------------------------------------------------------------------------
_SPAWN_QUEUE_PATH = TW_DATA_DIR / ".nautical_spawn_queue.jsonl"
_SPAWN_QUEUE_LOCK = TW_DATA_DIR / ".nautical_spawn_queue.lock"
_SPAWN_QUEUE_KICK = TW_DATA_DIR / ".nautical_spawn_queue.kick"
_SPAWN_DRAIN_SLEEP_SECS = 0.25
_SPAWN_DRAIN_THROTTLE_SECS = 1.0


_candidates = []


def _add(p):
    if not p:
        return
    p = Path(p).expanduser().resolve()
    if p not in _candidates:
        _candidates.append(p)


_add(HOOK_DIR)
_add(TW_DIR)
if os.environ.get("TASKDATA"):
    _add(os.environ["TASKDATA"])
if os.environ.get("TASKRC"):
    _add(Path(os.environ["TASKRC"]).parent)
_add(Path.home() / ".task")

core = None
for base in _candidates:
    pyfile = base / "nautical_core.py"
    pkgini = base / "nautical_core" / "__init__.py"
    if pyfile.is_file():
        sys.path.insert(0, str(base))
        core = importlib.import_module("nautical_core")
        break
    if pkgini.is_file():
        sys.path.insert(0, str(base))
        core = importlib.import_module("nautical_core")
        break

if core is None:
    msg = "nautical_core.py not found. Looked in:\n  " + "\n  ".join(
        str(p) for p in _candidates
    )
    raise ModuleNotFoundError(msg)


# ------------------------------------------------------------------------------
# Small cached helpers for speed + consistency
# ------------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _parse_dt_any_cached(s: str):
    return core.parse_dt_any(s)


@lru_cache(maxsize=512)
def _fmt_dt_local_cached(dt):
    return core.fmt_dt_local(dt)


@lru_cache(maxsize=512)
def _to_local_cached(dt):
    # Accept either datetime or (datetime, meta) tuples from helper parsers.
    if isinstance(dt, (tuple, list)) and dt:
        dt0 = dt[0]
        if isinstance(dt0, datetime):
            dt = dt0
    return core.to_local(dt)


@lru_cache(maxsize=256)
def _validate_anchor_expr_cached(expr: str):
    return core.validate_anchor_expr_strict(expr)


@lru_cache(maxsize=512)
def _export_uuid_short_cached(u_short: str):
    return _export_uuid_short(u_short, env=None)


def _dtparse(s):
    return _parse_dt_any_cached(s)


def _fmtlocal(dt):
    return _fmt_dt_local_cached(dt)


def _tolocal(dt):
    return _to_local_cached(dt)


# ------------------------------------------------------------------------------
# Basic IO and panel
# ------------------------------------------------------------------------------
def _read_two():
    raw = sys.stdin.read().strip()
    if not raw:
        print("", end="")
        sys.exit(0)
    parts = [p for p in raw.split("\n") if p.strip()]
    if len(parts) < 2:
        obj = json.loads(parts[0])
        return obj, obj
    old = json.loads(parts[0])
    new = json.loads(parts[-1])
    return old, new


def _print_task(task):
    print(json.dumps(task, ensure_ascii=False), end="")


_RICH_TAG_RE = re.compile(r"\[/\]|\[/?[A-Za-z0-9_ ]+\]")


def _strip_rich_markup(s: str) -> str:
    # Strip simple Rich tags like [cyan], [/cyan], [bold], [dim], and the bare [/].
    # Preserve bracketed literals like [auto-due] (contains '-').
    if not s:
        return s
    return _RICH_TAG_RE.sub("", s)


def _term_width_stderr(default: int = 80) -> int:
    try:
        w = os.get_terminal_size(sys.stderr.fileno()).columns
    except Exception:
        w = default
    # Termux can have narrow+wrapped terminals; keep delimiters within a safe band.
    return max(40, min(70, int(w)))


def _fast_color_enabled() -> bool:
    if not sys.stderr.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    v = os.environ.get("NAUTICAL_FAST_COLOR", "1").strip().lower()
    return v not in {"0", "no", "false", "off"}


def _ansi(code: str) -> str:
    return f"\x1b[{code}m"


def _emit_wrapped(prefix: str, text: str, width: int, style: str | None = None) -> None:
    if text is None:
        text = ""
    text = str(text)
    text = _strip_rich_markup(text)

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
                    sys.stderr.write(prefix + style + out + _ansi("0") + "\n")
                else:
                    sys.stderr.write(prefix + out + "\n")
                prefix = " " * len(prefix)
                avail = max(10, width - len(prefix))
                cur = token

        if cur:
            if style:
                sys.stderr.write(prefix + style + cur + _ansi("0") + "\n")
            else:
                sys.stderr.write(prefix + cur + "\n")
        prefix = " " * len(prefix)
        avail = max(10, width - len(prefix))


def _panel(
    title,
    rows,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    label_style: str | None = None,
):
    """
    Render a panel.

    Modes:
      - default / NAUTICAL_PANEL=rich : Rich panel (colors + borders)
      - NAUTICAL_PANEL=fast           : plain/ANSI panel (fast, Termux-friendly)

    The fast panel is designed to be stable, minimal, and easy to scan.
    """
    mode = os.environ.get("NAUTICAL_PANEL", "").strip().lower()
    if mode in {"plain"}:
        mode = "fast"

    if mode == "fast":
        width = _term_width_stderr()

        use_color = _fast_color_enabled()
        RESET = _ansi("0")
        BOLD = _ansi("1") if use_color else ""
        DIM = _ansi("2") if use_color else ""
        CYAN = _ansi("36") if use_color else ""
        GREEN = _ansi("32") if use_color else ""
        RED = _ansi("31") if use_color else ""
        YELLOW = _ansi("33") if use_color else ""

        delim = "─" * width
        sys.stderr.write(delim + "\n")
        sys.stderr.write((BOLD + CYAN + _strip_rich_markup(str(title)) + RESET) + "\n")

        keys = [str(k) for (k, _v) in rows if k is not None]
        label_w = 0
        for k in keys:
            if len(k) > label_w:
                label_w = len(k)
        label_w = min(14, max(6, label_w))

        def _style_for_row(k: str, v: str) -> str | None:
            lk = k.lower()
            sv = (v or "")
            lsv = sv.lower()
            if k.strip().lower() == "pattern":
                return CYAN
            if "natural" in lk:
                return DIM
            if k.strip().lower() in {"basis", "root"}:
                return DIM
            if k.strip().lower() in {"first due", "next due"}:
                if "overdue" in lsv or "late" in lsv:
                    return RED
                return GREEN
            if "warning" in lk:
                return YELLOW
            if "error" in lk:
                return RED
            if lk.startswith("chain"):
                return DIM + GREEN
            return None

        for k, v in rows:
            if k is None:
                sys.stderr.write("\n")
                continue

            k = _strip_rich_markup(str(k))
            v = "" if v is None else _strip_rich_markup(str(v))

            if k.lower().startswith("timeline"):
                prefix0 = f"{k:<{label_w}} "
                lines = [ln for ln in v.splitlines() if ln.strip() != ""] if "\n" in v else ([v] if v else [])
                if lines:
                    _emit_wrapped(prefix0, lines[0], width, style=None)
                    for ln in lines[1:]:
                        _emit_wrapped(" " * len(prefix0), ln, width, style=None)
                else:
                    _emit_wrapped(prefix0, "", width, style=None)
                continue

            prefix = f"{k:<{label_w}} "
            style = _style_for_row(k, v)
            _emit_wrapped(prefix, v, width, style=style)

        sys.stderr.write(delim + "\n")
        return

    # Rich mode (default)
    try:
        if not sys.stderr.isatty():
            raise RuntimeError("no tty")
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception:
        # If Rich is unavailable, fall back to fast mode without ANSI.
        os.environ.setdefault("NAUTICAL_PANEL", "fast")
        os.environ.setdefault("NAUTICAL_FAST_COLOR", "0")
        return _panel(title, rows, kind=kind, border_style=border_style, title_style=title_style, label_style=label_style)

    THEMES = {
        "preview_anchor": {"border": "turquoise2", "title": "bright_cyan", "label": "light_sea_green"},
        "preview_cp": {"border": "deep_pink1", "title": "deep_pink1", "label": "deep_pink3"},
        "summary": {"border": "indian_red", "title": "indian_red", "label": "red"},
        "disabled": {"border": "yellow", "title": "yellow", "label": "yellow"},
        "error": {"border": "red", "title": "red", "label": "red"},
        "warning": {"border": "yellow", "title": "yellow", "label": "yellow"},
        "info": {"border": "blue", "title": "cyan", "label": "cyan"},
    }
    theme = THEMES.get(kind, THEMES["info"])

    border = border_style or theme["border"]
    tstyle = title_style or theme["title"]
    lstyle = label_style or theme["label"]

    console = Console(file=sys.stderr, force_terminal=True)
    t = Table.grid(padding=(0, 1), expand=False)
    t.add_column(style=f"bold {lstyle}", no_wrap=True, justify="right")
    t.add_column(style="white")

    for k, v in rows:
        if k is None:
            t.add_row("", v or "")
            continue

        label_text = Text(str(k))
        lk = str(k).lower()
        if "warning" in lk:
            label_text.stylize("bold yellow")
        elif "error" in lk:
            label_text.stylize("bold red")
        elif "note" in lk:
            label_text.stylize("italic cyan")

        t.add_row(label_text, "" if v is None else str(v))

    console.print(
        Panel(
            t,
            title=Text(title, style=f"bold {tstyle}"),
            border_style=border,
            expand=False,
            padding=(0, 1),
        )
    )

def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') else s


def _format_chain_summary_rows(
    rows: list[tuple[str, str]]
) -> list[tuple[str | None, str]]:
    """
    Compact layout for chain-finished summary, without section headers.

    Order:
      Reason / Chain / Pattern-Natural-Period / (other)
      <blank>
      First due / Last end / Span
      <blank>
      Performance rows
      <blank>
      Limits
      <blank>
      History
    """
    chain_keys = {"Reason", "Chain", "Pattern", "Natural", "Period"}
    schedule_keys = {"First due", "Last end", "Span"}
    perf_keys = {
        "Performance",
        "Avg lateness",
        "Median lateness",
        "Best early",
        "Worst late",
    }
    limits_keys = {"Limits"}
    history_keys = {"History"}

    chain: list[tuple[str, str]] = []
    schedule: list[tuple[str, str]] = []
    perf: list[tuple[str, str]] = []
    limits: list[tuple[str, str]] = []
    history: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        if k in chain_keys:
            chain.append((k, v))
        elif k in schedule_keys:
            schedule.append((k, v))
        elif k in perf_keys:
            perf.append((k, v))
        elif k in limits_keys:
            limits.append((k, v))
        elif k in history_keys:
            history.append((k, v))
        else:
            others.append((k, v))

    # Put unknowns next to the chain meta so nothing disappears
    chain.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(chain)
    _add(schedule)
    _add(perf)
    _add(limits)
    _add(history)

    return out or rows



def _format_next_anchor_rows(
    rows: list[tuple[str, str]]
) -> list[tuple[str | None, str]]:
    """
    Compact layout for anchor next-link feedback, without section headers.

    Order:
      Pattern / Natural / Basis / Sanitised / (other)
      <blank>
      Next Due / Link status / Links left / Limits
      <blank>
      Final(...) rows
      <blank>
      Timeline
      <blank>
      Rand
    """
    chain_keys = {"Pattern", "Natural", "Basis", "Sanitised"}
    next_keys = {"Next Due", "Scheduled", "Wait", "Link status", "Links left", "Limits"}
    timeline_keys = {"Timeline"}
    footer_keys = {"Rand"}

    chain: list[tuple[str, str]] = []
    next_sec: list[tuple[str, str]] = []
    finals: list[tuple[str, str]] = []
    timeline: list[tuple[str, str]] = []
    footer: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        if k in chain_keys:
            chain.append((k, v))
        elif k in next_keys:
            next_sec.append((k, v))
        elif isinstance(k, str) and k.startswith("Final ("):
            finals.append((k, v))
        elif k in timeline_keys:
            timeline.append((k, v))
        elif k in footer_keys:
            footer.append((k, v))
        else:
            others.append((k, v))

    chain.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(chain)
    _add(next_sec)
    _add(finals)
    _add(timeline)
    _add(footer)

    return out or rows

def _format_gap(prev_dt: datetime, next_dt: datetime, kind: str = "cp", round_hours: bool = True) -> str:
    """
    Format the time gap between two timeline items as a compact inline annotation.
    Returns a string like " └─ +3d →" or empty string.
    """
    if not (prev_dt and next_dt):
        return ""
    
    gap_seconds = (next_dt - prev_dt).total_seconds()
    
    # Skip very small gaps
    if abs(gap_seconds) < 60:
        return ""
    
    # Format based on chain type
    if kind == "cp":
        # For CP chains, show days/hours
        days = gap_seconds / 86400
        if abs(days) >= 1:
            if days.is_integer():
                gap_str = f"{int(days)}d"
            else:
                # Show fractional days for non-24h multiples
                gap_str = f"{days:.1f}d"
        else:
            hours = gap_seconds / 3600
            if abs(hours) >= 1:
                gap_str = f"{hours:.1f}h"
            else:
                minutes = gap_seconds / 60
                gap_str = f"{int(minutes)}m"
    else:
        # For anchor chains, optionally round to nearest day
        total_hours = gap_seconds / 3600
        days = total_hours / 24
        
        if round_hours and abs(days) >= 0.5:  # Only round if > 12h
            # Round to nearest day
            rounded_days = round(days)
            gap_str = f"{rounded_days}d"
        else:
            # Show days with fractional part
            if abs(days) >= 1:
                gap_str = f"{days:.1f}d"
            else:
                # Show hours for sub-day gaps
                gap_str = f"{total_hours:.0f}h"
    
    return f" ➔ {gap_str}"




def _format_next_cp_rows(
    rows: list[tuple[str, str]]
) -> list[tuple[str | None, str]]:
    """
    Compact layout for cp next-link feedback, without section headers.

    Order:
      Period / Basis / (other)
      <blank>
      Next Due / Link status / Links left / Limits
      <blank>
      Final(...) rows
      <blank>
      Timeline
    """
    chain_keys = {"Period", "Basis"}
    next_keys = {"Next Due", "Scheduled", "Wait", "Link status", "Links left", "Limits"}
    timeline_keys = {"Timeline"}

    chain: list[tuple[str, str]] = []
    next_sec: list[tuple[str, str]] = []
    finals: list[tuple[str, str]] = []
    timeline: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        if k in chain_keys:
            chain.append((k, v))
        elif k in next_keys:
            next_sec.append((k, v))
        elif isinstance(k, str) and k.startswith("Final ("):
            finals.append((k, v))
        elif k in timeline_keys:
            timeline.append((k, v))
        else:
            others.append((k, v))

    chain.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(chain)
    _add(next_sec)
    _add(finals)
    _add(timeline)

    return out or rows




# ------------------------------------------------------------------------------
# Taskwarrior integration
# ------------------------------------------------------------------------------
_TW_JISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_UNREC_ATTR_RE = re.compile(r"Unrecognized attribute '([^']+)'", re.I)



class _SpawnDeferred(RuntimeError):
    """Raised when inline spawn is deferred due to Taskwarrior lock contention."""

    def __init__(self, *, child_short: str, stripped_attrs: set[str] | None = None, reason: str = ""):
        super().__init__(reason or "Deferred spawn")
        self.child_short = child_short
        self.stripped_attrs = stripped_attrs or set()
        self.reason = reason or "Deferred spawn"


def _queue_locked(fn):
    """Run `fn()` under an advisory file lock (best-effort)."""
    if fcntl is None:
        return fn()
    try:
        _SPAWN_QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    lf = None
    try:
        lf = open(_SPAWN_QUEUE_LOCK, "a", encoding="utf-8")
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        return fn()
    finally:
        try:
            if lf is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            if lf is not None:
                lf.close()
        except Exception:
            pass


def _requeue_deferred_spawn_payload(payload: str) -> None:
    if not payload:
        return

    def _put():
        try:
            _SPAWN_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            with open(_SPAWN_QUEUE_PATH, "a", encoding="utf-8") as f:
                f.write(payload)
        except Exception:
            # Best-effort; if we cannot persist, we simply drop (but keep hook alive).
            pass

    _queue_locked(_put)


def _enqueue_deferred_spawn(task_obj: dict) -> None:
    """Append one task JSON object to the deferred spawn queue (JSONL, UTF-8)."""
    try:
        line = json.dumps(task_obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    except Exception:
        # As a last resort, fall back to default encoding.
        line = json.dumps(task_obj) + "\n"
    _requeue_deferred_spawn_payload(line)


def _iter_json_objects_from_mixed_text(raw: str):
    """Yield JSON objects from a text stream that may contain whitespace and literal '\\n' separators.

    This is intentionally tolerant: earlier Nautical builds mistakenly wrote the literal two
    characters '\\n' between JSON objects. This parser treats those sequences as separators
    *only when they occur between objects* (i.e., outside of decoded JSON strings).
    """
    dec = json.JSONDecoder()
    i = 0
    n = len(raw)
    while i < n:
        # Skip whitespace and literal "\n" separators between objects.
        while i < n:
            ch = raw[i]
            if ch.isspace():
                i += 1
                continue
            if raw.startswith("\\n", i):
                i += 2
                continue
            break
        if i >= n:
            break

        try:
            obj, j = dec.raw_decode(raw, i)
        except Exception:
            # Resync: jump to next object start if possible.
            nxt = raw.find("{", i + 1)
            if nxt == -1:
                break
            i = nxt
            continue

        if isinstance(obj, dict):
            yield obj
        i = j


def _take_deferred_spawn_payload() -> str:
    """Take (and truncate) the queue, returning normalized JSONL payload (one object per line)."""

    def _take():
        try:
            if not _SPAWN_QUEUE_PATH.exists():
                return ""
        except Exception:
            return ""

        try:
            raw = _SPAWN_QUEUE_PATH.read_text(encoding="utf-8")
        except Exception:
            raw = ""

        if not (raw or "").strip():
            try:
                _SPAWN_QUEUE_PATH.unlink()
            except Exception:
                pass
            return ""

        objs = list(_iter_json_objects_from_mixed_text(raw))
        if not objs:
            # Do not truncate if we couldn't parse anything (avoid data loss).
            return ""

        # Truncate while under lock; import happens after lock is released.
        try:
            _SPAWN_QUEUE_PATH.write_text("", encoding="utf-8")
        except Exception:
            pass

        return "\n".join(json.dumps(o, ensure_ascii=False, separators=(",", ":")) for o in objs) + "\n"

    return _queue_locked(_take) or ""



def _drain_deferred_spawn_queue() -> int:
    """Drain queued spawns (best-effort). Returns 0 on success/empty, non-zero on failure."""
    payload = _take_deferred_spawn_payload()
    if not (payload or "").strip():
        return 0

    env = os.environ.copy()
    last_err = ""

    for attempt in range(1, 8):
        try:
            r = subprocess.run(
                ["task", f"rc.data.location={TW_DATA_DIR}", "rc.hooks=off", "rc.verbose=nothing", "import", "-"],
                input=payload,
                text=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
            if r.returncode == 0:
                return 0

            last_err = (r.stderr or "").strip()

            # If the only issue is schema drift (unknown attrs), sanitise and retry.
            if "Unrecognized attribute" in last_err:
                objs = []
                for ln in (payload or "").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    _sanitize_unknown_attrs(last_err, obj)
                    objs.append(obj)

                if objs:
                    payload = "\n".join(json.dumps(o, ensure_ascii=False, separators=(",", ":")) for o in objs) + "\n"
                    continue

        except subprocess.TimeoutExpired:
            last_err = "task import timed out"
        except Exception as e:
            last_err = str(e)

        # Backoff: the parent command should have released the lock by now, but be defensive.
        _time.sleep(0.15 * attempt)

    # Put the payload back so it can be retried later (keep the most recent sanitised payload).
    _requeue_deferred_spawn_payload(payload)
    return 1


def _kick_deferred_spawn_drain_async() -> None:
    """Start a detached drain run (throttled, best-effort).

    This must *not* block the hook. It schedules a short delayed drain in a
    detached process so the parent Taskwarrior command can release its lock first.
    """
    try:
        if not _SPAWN_QUEUE_PATH.exists() or _SPAWN_QUEUE_PATH.stat().st_size <= 0:
            return
    except Exception:
        # If we cannot stat, still attempt a kick.
        pass

    # Throttle kicks (avoid spawning many drainers in bursts).
    try:
        now = _time.time()
        last = _SPAWN_QUEUE_KICK.stat().st_mtime if _SPAWN_QUEUE_KICK.exists() else 0.0
        if (now - last) < float(_SPAWN_DRAIN_THROTTLE_SECS):
            return
        _SPAWN_QUEUE_KICK.write_text("", encoding="utf-8")
    except Exception:
        pass

    script = str(Path(__file__).resolve())
    py = sys.executable or "python3"

    # Avoid shell dependencies. Use Python to sleep, then drain.
    cmd_list = [py, script, "--drain-spawn-queue"]
    code = (
        "import time,subprocess; "
        f"time.sleep({_SPAWN_DRAIN_SLEEP_SECS}); "
        f"subprocess.run({repr(cmd_list)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
    )
    try:
        subprocess.Popen(
            [py, "-c", code],
            start_new_session=True,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # If cannot kick, the queue remains and can be drained manually.
        pass

def _short(u):
    return (u or "")[:8]


def _export_uuid_short(u_short: str, env=None):
    env = env or os.environ.copy()
    r = subprocess.run(
        ["task", "rc.hooks=off", "rc.json.array=off", f"uuid:{u_short}", "export"],
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        obj = json.loads(r.stdout.strip() or "{}")
        return obj if obj.get("uuid") else None
    except Exception:
        return None


def _task_exists_by_uuid(u: str, env: dict) -> bool:
    q = ["task", "rc.hooks=off", "rc.json.array=off", f"uuid:{u}", "export"]
    vr = subprocess.run(
        q, text=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        data = json.loads(vr.stdout.strip() or "{}")
    except Exception:
        data = {}
    return bool(data.get("uuid"))


def _sanitize_unknown_attrs(stderr: str, payload: dict) -> set[str]:
    removed = set()
    for m in _UNREC_ATTR_RE.finditer(stderr or ""):
        bad = m.group(1)
        if bad in payload:
            payload.pop(bad, None)
            removed.add(bad)
    return removed


def _normalise_datetime_fields(obj: dict) -> None:
    def _to_tw_compact_isoz(s: str) -> str:
        if isinstance(s, str) and _TW_JISO.fullmatch(s):
            return s.replace("-", "").replace(":", "")
        return s

    for k in ("entry", "modified", "due", "end", "wait", "until", "scheduled"):
        if k in obj and obj[k]:
            obj[k] = _to_tw_compact_isoz(obj[k])
    if "annotations" in obj and isinstance(obj["annotations"], list):
        for ann in obj["annotations"]:
            if isinstance(ann, dict) and ann.get("entry"):
                ann["entry"] = _to_tw_compact_isoz(ann["entry"])


def _strip_none_and_cast(obj: dict):
    out = {}
    for k, v in obj.items():
        if v is None:
            continue
        if k in ("link", "chainMax"):
            try:
                v = int(v)
            except Exception:
                pass
        out[k] = v
    return out


def _spawn_child(child_task: dict) -> tuple[str, set[str]]:
    """
    Create child via `task import -`, preserving annotation entries.
    Returns (short_uuid, stripped_attrs).
    Raises RuntimeError with detailed context on failure.
    """
    env = os.environ.copy()
    child_uuid = str(uuid.uuid4())
    obj = dict(child_task)
    obj["uuid"] = child_uuid
    if "entry" not in obj:
        obj["entry"] = core.fmt_isoz(core.now_utc())
    if "modified" not in obj:
        obj["modified"] = obj["entry"]


    obj = _strip_none_and_cast(obj)
    _normalise_datetime_fields(obj)

    attempts = 0
    stripped_accum = set()
    last_stderr = ""
    last_category = ""

    while attempts < _MAX_SPAWN_ATTEMPTS:
        attempts += 1
        payload = json.dumps(obj) + "\n"

        try:
            r = subprocess.run(
                ["task", f"rc.data.location={TW_DATA_DIR}", "rc.hooks=off", "import", "-"],
                input=payload,
                text=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,  # prevent hanging
            )
        except subprocess.TimeoutExpired:
            last_stderr = "Task import timed out (>10s)"
            last_category = "taskwarrior"
            continue
        except Exception as e:
            last_stderr = f"Subprocess error: {str(e)}"
            last_category = "taskwarrior"
            continue

        if r.returncode == 0 and _task_exists_by_uuid(child_uuid, env):
            return child_uuid[:8], stripped_accum

        last_stderr = r.stderr or ""
        category, is_retryable = _categorize_spawn_error(r.returncode, last_stderr)
        last_category = category

        if category == "attribute":
            # Strip unknown attributes and retry
            removed = _sanitize_unknown_attrs(last_stderr, obj)
            if removed:
                stripped_accum |= removed
                continue

        # For non-attribute errors, only retry once more
        if is_retryable and attempts < 2:
            continue

        # Otherwise, bail out with context
        break

    # Surface failure with categorized error message
    error_msg = ""
    if last_category == "attribute":
        error_msg = f"Unknown task attributes even after stripping: {', '.join(sorted(stripped_accum))}"
    elif last_category == "parse":
        error_msg = (
            f"Failed to serialize task payload (JSON error). Check task field types."
        )
    elif last_category == "validation":
        error_msg = f"Task validation failed. Common causes: invalid due date, bad field format, or unsupported attribute value."
    elif last_category == "taskwarrior":
        error_msg = f"Taskwarrior import failed (after {attempts} attempts)"
    else:
        error_msg = "Child import failed for unknown reason"

    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console(file=sys.stderr, force_terminal=True)
        panel_text = f"[red]{error_msg}[/red]\n\n"
        panel_text += f"[bold]Category:[/bold] {last_category}\n"
        panel_text += f"[bold]Attempts:[/bold] {attempts}/{_MAX_SPAWN_ATTEMPTS}\n"
        panel_text += f"[bold]stderr:[/bold]\n{last_stderr}"
        console.print(
            Panel(
                panel_text, border_style="red", title="[bold]Child import failed[/bold]"
            )
        )
    except Exception:
        sys.stderr.write(f"{error_msg}\n")
        sys.stderr.write(f"Category: {last_category}\n")
        sys.stderr.write(f"Attempts: {attempts}/{_MAX_SPAWN_ATTEMPTS}\n")
        sys.stderr.write(f"Error details:\n{last_stderr}\n")

    raise RuntimeError(f"Failed to import child task: {error_msg}")


# Helper to categorize subprocess failures
def _categorize_spawn_error(returncode: int, stderr: str) -> tuple[str, bool]:
    """
    Categorize spawn errors and return (category, is_retryable).
    category: "parse", "attribute", "validation", "taskwarrior", "unknown"
    is_retryable: whether we should retry this attempt
    """
    stderr_lower = (stderr or "").lower()

    if returncode == 0:
        return ("success", False)

    # Unrecognized attribute - NOT retryable, just strip and retry
    if "unrecognized attribute" in stderr_lower:
        return ("attribute", True)

    # JSON parsing errors - likely malformed task, NOT retryable
    if "json" in stderr_lower or "parse" in stderr_lower:
        return ("parse", False)

    # Validation errors (e.g., bad due date format) - NOT retryable
    if "invalid" in stderr_lower or "bad date" in stderr_lower:
        return ("validation", False)

    # Taskwarrior internal errors - possibly retryable
    if "error" in stderr_lower or "failed" in stderr_lower:
        return ("taskwarrior", True)

    return ("unknown", True)


def _spawn_child_atomic(child_task: dict, parent_task_with_nextlink: dict) -> tuple[str, set[str]]:
    """
    Spawn a child via `task import -`.

    Important: The parent update is applied by Taskwarrior using this hook's stdout.
    We intentionally avoid importing the parent from inside the hook to reduce the
    risk of re-entering Taskwarrior while it is holding the datastore lock.

    If the nested import times out (typically lock contention during `task done`),
    we enqueue the child for a deferred drain after the main command returns and
    raise `_SpawnDeferred` (callers should keep the returned nextLink value).
    """
    env = os.environ.copy()

    # Prepare child with a stable UUID (so retries / deferred drain cannot fork).
    child_uuid = str(uuid.uuid4())
    child_obj = dict(child_task)
    child_obj["uuid"] = child_uuid
    if "entry" not in child_obj:
        child_obj["entry"] = core.fmt_isoz(core.now_utc())
    if "modified" not in child_obj:
        child_obj["modified"] = child_obj["entry"]

    child_short = child_uuid[:8]

    # Normalise
    child_obj = _strip_none_and_cast(child_obj)
    _normalise_datetime_fields(child_obj)

    stripped_attrs: set[str] = set()
    last_stderr = ""
    last_category = "unknown"

    for attempt in range(1, _MAX_SPAWN_ATTEMPTS + 1):
        payload = json.dumps(child_obj, ensure_ascii=False) + "\n"

        try:
            r = subprocess.run(
                ["task", f"rc.data.location={TW_DATA_DIR}", "rc.hooks=off", "import", "-"],
                input=payload,
                text=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
            )
        except subprocess.TimeoutExpired:
            # Likely Taskwarrior datastore lock contention (re-entrance from hook).
            _enqueue_deferred_spawn(child_obj)
            raise _SpawnDeferred(
                child_short=child_short,
                stripped_attrs=stripped_attrs,
                reason="Task import timed out (likely lock contention); queued for deferred drain",
            )

        last_stderr = r.stderr or ""
        if r.returncode == 0:
            return (child_short, stripped_attrs)

        # Strip unrecognized attributes and retry (best-effort resilience).
        if "Unrecognized attribute" in last_stderr:
            stripped_attrs |= _sanitize_unknown_attrs(last_stderr, child_obj)
            last_category = "attribute"
            continue

        category, is_retryable = _categorize_spawn_error(r.returncode, last_stderr)
        last_category = category
        if not is_retryable:
            break

    raise RuntimeError(
        f"Child import failed after {_MAX_SPAWN_ATTEMPTS} attempts "
        f"(category={last_category}): {last_stderr.strip()}"
    )



def _root_uuid_from(task: dict) -> str:
    """Return the stable chain seed.

    New releases are chainID-first: we do not walk legacy prevLink chains.
    If chainID is missing we fall back to this task's UUID.
    """

    cid = (task.get("chainID") or "").strip()
    if cid:
        return cid
    return (task.get("uuid") or "").strip()

# --- Chain export: prefer chainID, else walk the legacy links -----------------
def _task(args, env=None) -> str:
    """
    Thin wrapper around 'task' returning stdout as text.
    Always disables hooks; caller should provide rc.json.array flag when needed.
    """
    env = env or os.environ.copy()
    r = subprocess.run(
        ["task", "rc.hooks=off"] + list(args),
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return r.stdout or ""

def _export_uuid_full(u: str, env=None) -> dict | None:
    """Export a single task by full UUID."""
    try:
        out = _task(["rc.json.array=1", f"export uuid:{u}"], env=env)  # uses existing _task()
        arr = json.loads(out) if out and out.strip().startswith("[") else []
        return arr[0] if arr else None
    except Exception:
        return None

def tw_export_chain_or_fallback(seed_task, env=None):
    """Return full chain export for a task.

    Policy: chainID is mandatory (legacy link-walk fallback removed).
    """
    chain_id = seed_task.get('chainID') or seed_task.get('chainid')
    if not chain_id:
        raise RuntimeError('ChainID is required (legacy chain traversal removed). Run your chainID backfill tool, then retry.')
    return tw_export_chain(chain_id, env=env)
def _tw_get_cached(ref: str) -> str:
    """Return `task _get <ref>` stdout stripped. Cached within one hook run."""
    try:
        out = _task(["rc.verbose=nothing", "_get", ref], env=None)
        return (out or "").strip()
    except Exception:
        return ""

def _chain_root_and_age(task: dict, now_utc: datetime) -> tuple[str, int | None]:
    """Get chain root (short chainID or UUID) and age in days.
    Returns (root_short, age_days). age_days is None if unavailable."""
    try:
        # Get root short ID (chainID or short UUID)
        root_short = (task.get("chainID") or "").strip()
        
        if not root_short:
            # Fallback to root UUID from chain
            root_uuid = _root_uuid_from(task)
            if root_uuid:
                root_short = root_uuid[:8]
            else:
                root_short = _short(task.get("uuid"))
        
        # Get age if we have a root
        age_days = None
        if root_short:
            root_entry = _tw_get_cached(f"{root_short}.entry")
            entry_dt = _dtparse(root_entry)
            
            if entry_dt:
                entry_local = _tolocal(entry_dt).date()
                today_local = _tolocal(now_utc).date()
                age_days = (today_local - entry_local).days
                
                if age_days < 0:
                    age_days = 0
        
        return root_short, age_days
    except Exception:
        return "—", None

def _format_root_and_age(task: dict, now_utc: datetime) -> str:
    """Format root and age as a single string.
    Returns root (age) or just root if age is 0 or unavailable."""
    root_short, age_days = _chain_root_and_age(task, now_utc)
    
    if not root_short or root_short == "—":
        return "—"
    
    # Only show age if it's > 0
    if age_days is not None and age_days > 0:
        return f"{root_short} ▻ {age_days}d"
    
    return root_short

# ------------------------------------------------------------------------------
# On modify-without-completion helpers
# ------------------------------------------------------------------------------


def _canon_for_compare(v):
    """Canonicalize values so 5 == 5.0, strings are trimmed, and
    dict/list comparisons are stable."""
    if v is None:
        return None
    # Booleans/numbers
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))  # 5 and 5.0 normalize equal
    # Strings that might be numeric
    if isinstance(v, str):
        s = v.strip()
        try:
            return Decimal(s)  # if numeric string, compare numerically
        except (InvalidOperation, ValueError):
            return s  # non-numeric string
    # Collections → stable JSON
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(v)

def _field_changed(old: dict, new: dict, key: str) -> bool:
    ov = old.get(key)
    nv = new.get(key)
    return _canon_for_compare(ov) != _canon_for_compare(nv)





def _validate_anchor_on_modify(expr: str):
    """Mirror on-add strict checks for anchor; raise ValueError on problems."""
    if not expr or not expr.strip():
        raise ValueError("anchor is required if chaining by anchor")

    # Syntax first (for friendlier messages)
    try:
        dnf_raw = core.parse_anchor_expr_to_dnf(expr)
    except Exception as e:
        raise ValueError(f"anchor syntax error: {str(e)}")


    # NOTE: legacy weekday ':' syntax is accepted for backward compatibility.

    # Strict validation
    try:
        _validate_anchor_expr_cached(expr)  # calls core.validate_anchor_expr_strict
    except Exception as e:
        raise ValueError(f"anchor validation failed: {str(e)}")


def _validate_cp_on_modify(cp_str: str, chain_max_val, chain_until_val):
    """
    Mirror on-add CP checks for a plain modify:
      - cp must parse as a duration
      - optional chainMax must be a non-negative int
      - optional chainUntil must parse as a datetime
    """
    if not cp_str or not cp_str.strip():
        return  # nothing to validate

    td = core.parse_cp_duration(cp_str)
    if td is None:
        raise ValueError(f"Invalid duration format '{cp_str}' (expected: 3d, 2w, 1h, etc.)")

    # chainMax
    if chain_max_val not in (None, ""):
        try:
            cpmax = core.coerce_int(chain_max_val, 0)
        except Exception:
            # fallback if coerce_int isn't available
            try:
                cpmax = int(str(chain_max_val).strip())
            except Exception:
                raise ValueError("chainMax must be an integer")
        if cpmax < 0:
            raise ValueError("chainMax must be a non-negative integer")

    # chainUntil
    cu = (chain_until_val or "").strip()
    if cu:
        dt = core.parse_dt_any(cu)
        if dt is None:
            raise ValueError(f"Invalid chainUntil '{cu}'")


# ------------------------------------------------------------------------------
# Pretty helpers
# ------------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _chain_colour_root(kind: str, root_uuid: str) -> str:
    """
    Deterministic but cheap 'random' colour per chain root.

    `kind` is "anchor" or "cp".
    """
    # Palettes chosen to match / complement the panel edge colours
    anchor_palette = [
            "bright_cyan",
            "cyan",
            "turquoise2",
            "deep_sky_blue1",
            "light_sky_blue1",
            "medium_turquoise",
            "dark_turquoise",
            "cyan3",
            "sky_blue1",
            "dodger_blue1",
            "steel_blue1",
            "cornflower_blue",
        ]
    cp_palette = [
        "indian_red1",
        "deep_pink3",
        "medium_orchid",
        "orchid1",
        "orange_red1",
        "hot_pink",
        "pink3",
        "light_coral",
        "salmon1",
        "light_pink3",
        "pale_violet_red1",
        "medium_violet_red",
    ]

    palette = cp_palette if kind == "cp" else anchor_palette
    if not palette:
        return "bright_magenta" if kind == "cp" else "bright_cyan"

    s = (root_uuid or "").replace("-", "")
    if not s:
        return palette[0]

    # Very cheap "hash": last 4 hex chars; fall back to sum of ords.
    try:
        h = int(s[-4:], 16)
    except ValueError:
        h = 0
        for ch in s:
            h += ord(ch)

    return palette[h % len(palette)]


def _chain_colour_for_task(task: dict, kind: str) -> str:
    """
    Get the chain colour for this task (uses root uuid, cached).
    """
    root = (_root_uuid_from(task) or task.get("uuid") or "").strip()
    return _chain_colour_root(kind, root)


def _future_style_for_chain(task: dict, kind: str) -> str:
    """
    Style for FUTURE links in the timeline.
    - When per-chain mode is OFF → static colour per kind (fast path)
    - When per-chain mode is ON  → cached colour per chain root
    """
    if not _CHAIN_COLOR_PER_CHAIN:
        # Static behaviour
        return "medium_violet_red" if kind == "cp" else "cyan"

    return _chain_colour_for_task(task, kind)




def _fmt_on_time_delta(due_dt, end_dt, tol_secs: int = 60):
    if not (due_dt and end_dt):
        return ""
    diff = (end_dt - due_dt).total_seconds()
    if diff > tol_secs:
        human = core.humanize_delta(due_dt, end_dt, use_months_days=False)
        return f"[yellow](+{human.replace('overdue by ','').replace('in ','')} late)[/]"
    if diff < -tol_secs:
        human = core.humanize_delta(end_dt, due_dt, use_months_days=False)
        return f"[cyan](-{human.replace('in ','')} early)[/]"
    return "[green](on time)[/]"


def _collect_prev_two(current_task: dict) -> list[dict]:
    """Return up to two previous tasks (older first) using chainID export only."""

    chain_id = (current_task.get("chainID") or "").strip()
    if not chain_id:
        return []

    cur_no = core.coerce_int(current_task.get("link"), None)
    if not cur_no or cur_no <= 1:
        return []

    try:
        chain = tw_export_chain(chain_id)
    except Exception:
        return []

    # Choose tasks by link index. In the unlikely event of duplicates, prefer non-deleted tasks.
    def _pick_best(candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        for st in ("pending", "completed", "deleted"):
            for t in candidates:
                if (t.get("status") or "").strip().lower() == st:
                    return t
        return candidates[0]

    by_link: dict[int, list[dict]] = {}
    for t in chain:
        ln = core.coerce_int(t.get("link"), None)
        if ln is None:
            continue
        by_link.setdefault(ln, []).append(t)

    prevs: list[dict] = []
    for want in (cur_no - 2, cur_no - 1):
        if want < 1:
            continue
        obj = _pick_best(by_link.get(want, []))
        if obj:
            prevs.append(obj)
    return prevs


def _pretty_basis_cp(task: dict, meta: dict) -> str:
    td = core.parse_cp_duration(task.get("cp") or "")
    if not td:
        return "end + cp"
    secs = int(td.total_seconds())
    rem = secs % 86400
    if rem != 0:
        hrs, rems = divmod(rem, 3600)
        mins, _ = divmod(rems, 60)
        hint = []
        if hrs:
            hint.append(f"{hrs}h")
        if mins:
            hint.append(f"{mins}m")
        rem_s = " ".join(hint) if hint else f"{rem}s"
        return f"Exact end + cp (remainder {rem_s} vs 24h)"
    return "Preserve wall clock (period is multiple of 24h)"


def _pretty_basis_anchor(meta: dict, task: dict) -> str:
    mode = (meta.get("mode") or "skip").lower()
    basis = meta.get("basis")
    missed = int(meta.get("missed_count") or 0)
    due0 = core.parse_dt_any(task.get("due"))
    due_s = core.fmt_dt_local(due0) if due0 else "(no due)"
    if mode == "skip":
        return "SKIP — Next anchor after completion (multi-time: between slots counts as previous slot)"
    if mode == "flex":
        return f"FLEX — Skip missed up to now; next after completion ({missed} missed since {due_s})"
    if basis == "missed":
        return f"ALL — Backfilling first of {missed} missed anchor(s) since {due_s}"
    if basis == "after_due":
        return "ALL (no missed) — Next anchor after original due"
    return "ALL — Next anchor after completion"


# ------------------------------------------------------------------------------
# Multi-time occurrence helpers (hook-level)
# ------------------------------------------------------------------------------

_HHMM_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _parse_hhmm_token(tok: str) -> tuple[int, int] | None:
    tok = (tok or "").strip()
    if not tok or not _HHMM_RE.match(tok):
        return None
    hh, mm = tok.split(":", 1)
    return (int(hh), int(mm))


def _norm_hhmm_list(v) -> list[tuple[int, int]]:
    """Normalize various core representations of @t into a sorted list of (hh, mm)."""
    if v is None:
        return []
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int) for x in v):
        hh, mm = v
        return [(hh, mm)]
    if isinstance(v, str):
        out: list[tuple[int, int]] = []
        for part in v.split(","):
            t = _parse_hhmm_token(part)
            if t is not None:
                out.append(t)
        return out
    if isinstance(v, list):
        out: list[tuple[int, int]] = []
        for it in v:
            out.extend(_norm_hhmm_list(it))
        return out
    return []


def _extract_time_slots_from_dnf(dnf) -> list[tuple[int, int]]:
    """Extract a unique, sorted list of time slots from a parsed anchor DNF."""
    out: set[tuple[int, int]] = set()
    try:
        for term in dnf:
            for atom in term:
                mods = atom.get("mods") or {}
                for hhmm in _norm_hhmm_list(mods.get("t")):
                    out.add(hhmm)
    except Exception:
        return []
    return sorted(out)

def _skip_reference_dt_local(
    dnf,
    end_local: "datetime",
    due_local: Optional["datetime"],
) -> "datetime":
    """Choose the reference datetime for SKIP mode.

    For multi-time anchors, completing *between* scheduled slots should advance to the *next* slot
    (e.g. 09→12) rather than skipping it (09→18) due to a future due timestamp.

    Rules:
      - If there is no due: advance from completion time.
      - If completion is on/after due: advance from completion time.
      - If completion is before due:
          * Single-slot anchors: treat completion as fulfilling the due slot (advance from due)
            to avoid respawning the same slot.
          * Multi-slot anchors (same day): if completion time is after an earlier slot on that day,
            treat completion as fulfilling the latest earlier slot; otherwise, treat as fulfilling
            the due slot.
    """
    if due_local is None:
        return end_local

    if end_local >= due_local:
        return end_local

    slots = _extract_time_slots_from_dnf(dnf)
    if len(slots) <= 1:
        return due_local

    if end_local.date() != due_local.date():
        return due_local

    end_hhmm = (end_local.hour, end_local.minute)
    prev_slots = [s for s in slots if s <= end_hhmm]
    if not prev_slots:
        return due_local

    hh, mm = prev_slots[-1]
    tz = end_local.tzinfo or _local_tz()
    return datetime.combine(end_local.date(), time(hh, mm), tzinfo=tz)

def _as_local_dt(d: datetime | None) -> datetime | None:
    if d is None:
        return None
    if d.tzinfo is None:
        return d.replace(tzinfo=timezone.utc).astimezone(_local_tz())
    return d.astimezone(_local_tz())


def _next_occurrence_after_local_dt(
    dnf,
    after_local_dt: datetime,
    default_seed_date,
    seed_base: str,
    fallback_hhmm: tuple[int, int] | None = None,
):
    """Return the next occurrence strictly after `after_local_dt`.

    This is hook-level logic because core recurrence is date-based.
    """
    tz = after_local_dt.tzinfo or _local_tz()
    slots = _extract_time_slots_from_dnf(dnf)
    if not slots:
        slots = [fallback_hhmm] if fallback_hhmm else [(0, 0)]

    # same-day: only if the expression hits on that date
    adate = after_local_dt.date()
    try:
        prev = adate - timedelta(days=1)
        nxt_date, _ = core.next_after_expr(dnf, prev, default_seed_date, seed_base=seed_base)
    except Exception:
        nxt_date = None

    if nxt_date == adate:
        for hh, mm in slots:
            cand = datetime.combine(adate, time(hh, mm), tzinfo=tz)
            if cand > after_local_dt:
                return cand

    # otherwise, find the next matching date strictly after adate
    nxt_date, _ = core.next_after_expr(dnf, adate, default_seed_date, seed_base=seed_base)
    hh, mm = slots[0]
    return datetime.combine(nxt_date, time(hh, mm), tzinfo=tz)


def _missed_occurrences_between_local(
    dnf,
    due_local_dt: datetime,
    end_local_dt: datetime,
    default_seed_date,
    seed_base: str,
    fallback_hhmm: tuple[int, int] | None = None,
    guard: int = 512,
):
    """Return occurrences strictly after due and <= end."""
    if end_local_dt <= due_local_dt:
        return []
    missed: list[datetime] = []
    probe = due_local_dt
    for _ in range(guard):
        nxt = _next_occurrence_after_local_dt(
            dnf,
            probe,
            default_seed_date,
            seed_base,
            fallback_hhmm=fallback_hhmm,
        )
        if nxt is None or nxt > end_local_dt:
            break
        missed.append(nxt)
        probe = nxt
    return missed


def _collect_missed_occurrences(
    dnf,
    after_local_dt: datetime,
    until_local_dt: datetime,
    default_seed_date,
    seed_base: str,
    fallback_hhmm: tuple[int, int] | None = None,
    limit: int = 25,
) -> list[datetime]:
    """Collect missed *datetime* occurrences in (after_local_dt, until_local_dt].

    This is a hook-level helper: core recurrence operates on dates, while Nautical
    multi-time anchors (@t=...) need occurrence-level stepping.
    """
    if limit is None or limit <= 0:
        limit = 25
    return _missed_occurrences_between_local(
        dnf,
        due_local_dt=after_local_dt,
        end_local_dt=until_local_dt,
        default_seed_date=default_seed_date,
        seed_base=seed_base,
        fallback_hhmm=fallback_hhmm,
        guard=int(limit),
    )

def _human_delta(a, b, prefer_months=True):
    try:
        return core.humanize_delta(a, b, use_months_days=bool(prefer_months))
    except TypeError:
        return core.humanize_delta(a, b)


# ------------------------------------------------------------------------------
# Due calculators
# ------------------------------------------------------------------------------


def _compute_cp_child_due(parent: dict):
    dur = (parent.get("cp") or "").strip()
    if not dur:
        return (None, None)

    td, err = _safe_parse_cp_duration(dur)
    if err:
        raise ValueError(f"cp field: {err}")
    if not td:
        return (None, None)

    end_dt, err = _safe_parse_datetime(parent.get("end"))
    if err:
        raise ValueError(f"end field: {err}")
    if not end_dt:
        return (None, None)

    due_dt0, err = _safe_parse_datetime(parent.get("due"))
    if err:
        raise ValueError(f"due field: {err}")

    td_secs = int(td.total_seconds())
    rem = td_secs % 86400

    cand = (end_dt + td).replace(microsecond=0)
    if rem != 0:
        return cand, {"period": dur, "basis": "end+cp (exact)"}

    # preserve wall clock
    if due_dt0:
        dl = _tolocal(due_dt0)
        hh, mm = dl.hour, dl.minute
    else:
        el = _tolocal(end_dt)
        hh, mm = el.hour, el.minute
    cand_local = _tolocal(cand)
    due_local = cand_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return due_local.astimezone(timezone.utc), {
        "period": dur,
        "basis": "end+cp (preserve clock)",
    }


def _safe_parse_datetime(dt_str: str) -> tuple[datetime | None, str | None]:
    """
    Parse datetime safely.
    Returns (datetime, error_msg).
    error_msg is None on success, or a user-friendly explanation on failure.
    """
    if not (dt_str or "").strip():
        return (None, None)

    try:
        dt = core.parse_dt_any(dt_str)
        if dt is None:
            return (None, f"Unrecognized datetime format '{dt_str}'")
        return (dt, None)
    except ValueError as e:
        return (None, f"DateTime parsing error: {str(e)}")
    except TypeError as e:
        return (None, f"DateTime type error: {str(e)}")
    except Exception as e:
        return (None, f"Unexpected error parsing datetime '{dt_str}': {str(e)}")


def _safe_parse_cp_duration(duration_str: str) -> tuple[timedelta | None, str | None]:
    """
    Parse cp duration safely.
    Returns (timedelta, error_msg).
    error_msg is None on success, or a user-friendly explanation on failure.
    """
    if not (duration_str or "").strip():
        return (None, None)

    try:
        td = core.parse_cp_duration(duration_str)
        if td is None:
            return (
                None,
                f"Invalid duration format '{duration_str}' (expected: 3d, 2w, 1h, etc.)",
            )
        return (td, None)
    except ValueError as e:
        return (None, f"Duration parsing error: {str(e)}")
    except TypeError as e:
        return (None, f"Duration type error: {str(e)}")
    except Exception as e:
        return (None, f"Unexpected error parsing duration '{duration_str}': {str(e)}")


def _compute_anchor_child_due(parent: dict):
    """Return (next_due_utc, meta, dnf).

    The core recurrence engine computes *dates*; the hook expands into *datetimes* to
    respect multi-time lists: @t=HH:MM[,HH:MM...].
    """

    expr_str = (parent.get("anchor") or "").strip()
    if not expr_str:
        return (None, None, None)

    mode = (parent.get("anchor_mode") or "skip").strip().lower()
    if mode not in ("skip", "all", "flex"):
        raise ValueError(
            f"anchor_mode must be 'skip', 'all', or 'flex', got '{mode}'"
        )

    try:
        dnf = _validate_anchor_expr_cached(expr_str)
    except Exception as e:
        raise ValueError(f"Invalid anchor expression '{expr_str}': {str(e)}")

    end_dt_utc, err = _safe_parse_datetime(parent.get("end"))
    if err:
        raise ValueError(f"end field: {err}")
    if not end_dt_utc:
        return (None, None, None)

    due_dt_utc, err = _safe_parse_datetime(parent.get("due"))
    if err:
        raise ValueError(f"due field: {err}")

    end_local = _tolocal(end_dt_utc)
    due_local = _tolocal(due_dt_utc) if due_dt_utc else end_local

    default_seed = due_local.date()
    seed_base = (parent.get("chainID") or "").strip() or (_root_uuid_from(parent) or "preview")

    # Fallback time if the pattern carries no explicit @t
    fallback_hhmm = (due_local.hour, due_local.minute)

    info = {"mode": mode, "basis": None, "missed_count": 0, "missed_preview": []}

    if mode == "all":
        missed_dts = _collect_missed_occurrences(
            dnf,
            after_local_dt=due_local,
            until_local_dt=end_local,
            default_seed_date=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
            limit=25,
        )
        if missed_dts:
            nxt_local = missed_dts[0]
            info.update(
                basis="missed",
                missed_count=len(missed_dts),
                missed_preview=[x.isoformat() for x in missed_dts[:5]],
            )
        else:
            ref_local = _skip_reference_dt_local(
                dnf,
                end_local=end_local,
                due_local=(due_local if due_dt_utc else None),
            )
            nxt_local = _next_occurrence_after_local_dt(
                dnf,
                after_local_dt=ref_local,
                default_seed_date=default_seed,
                seed_base=seed_base,
                fallback_hhmm=fallback_hhmm,
            )
            info["basis"] = "after_due"
    elif mode == "flex":
        missed_dts = []
        if due_dt_utc and end_local > due_local:
            missed_dts = _collect_missed_occurrences(
                dnf,
                after_local_dt=due_local,
                until_local_dt=end_local,
                default_seed_date=default_seed,
                seed_base=seed_base,
                fallback_hhmm=fallback_hhmm,
                limit=25,
            )
        nxt_local = _next_occurrence_after_local_dt(
            dnf,
            after_local_dt=end_local,
            default_seed_date=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
        )
        info.update(
            basis="flex",
            missed_count=len(missed_dts),
            missed_preview=[x.isoformat() for x in missed_dts[:5]],
        )
    else:
        nxt_local = _next_occurrence_after_local_dt(
            dnf,
            after_local_dt=(max(end_local, due_local) if due_local else end_local),
            default_seed_date=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
        )
        info["basis"] = "after_end"

    if not nxt_local:
        raise ValueError("Could not compute next anchor occurrence")

    return nxt_local.astimezone(timezone.utc), info, dnf


def _estimate_cp_final_by_max(task: dict, next_due_utc):
    """
    Estimate the final due date when chainMax cap is reached.
    Returns the due datetime of link #chainMax.
    """
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if not cpmax:
        return None

    cur_no = core.coerce_int(task.get("link"), 1)
    if cur_no >= cpmax:
        return None

    td = core.parse_cp_duration(task.get("cp") or "")
    if not td:
        return None

    secs = int(td.total_seconds())
    fut_dt = next_due_utc
    fut_no = cur_no + 1

    # Step forward from next due until we reach cap_no
    while fut_no < cpmax:
        fut_no += 1
        if secs % 86400 == 0:
            dl = _tolocal(fut_dt)
            fut_dt = core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
            ).astimezone(timezone.utc)
        else:
            fut_dt = fut_dt + td

    return fut_dt


def _estimate_anchor_final_by_max(task: dict, next_due_utc, dnf):
    """
    Estimate the final due date when chainMax cap is reached for anchors.
    Returns the due datetime of link #chainMax.
    """
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if not cpmax:
        return None

    cur_no = core.coerce_int(task.get("link"), 1)
    if cur_no >= cpmax:
        return None

    seed_base = (task.get("chainID") or "").strip() or (_root_uuid_from(task) or "preview")
    nxt_local = _to_local_cached(next_due_utc)

    # Use a stable default seed (prefer the original due date).
    due0, _ = _safe_parse_datetime(task.get("due"))
    default_seed = _to_local_cached(due0 or next_due_utc).date()

    fallback_hhmm = (nxt_local.hour, nxt_local.minute)

    fut_no = cur_no + 1
    fut_local = nxt_local
    while fut_no < cpmax:
        fut_local = _next_occurrence_after_local_dt(
            dnf,
            fut_local,
            default_seed_date=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
        )
        fut_no += 1

    return fut_local.astimezone(timezone.utc)


# Helper to validate chainUntil is in the future
def _validate_until_not_past(
    until_dt: datetime, now_utc: datetime
) -> tuple[bool, str | None]:
    """
    Check if chainUntil is in the past.
    Returns (is_valid, error_msg).
    """
    if not until_dt:
        return (True, None)

    # Allow small grace period (1 minute) for race conditions
    grace = timedelta(minutes=1)
    if until_dt < (now_utc - grace):
        past_by = now_utc - until_dt
        past_s = core.humanize_delta(until_dt, now_utc, use_months_days=False)
        return (False, f"chainUntil is in the past (was {past_s} ago)")

    return (True, None)


# Helper to warn if chain extends too far into future
def _validate_chain_duration_reasonable(
    child_due: datetime, until_dt: datetime, now_utc: datetime
) -> tuple[bool, str | None]:
    """
    Warn if chain will extend unreasonably far into the future.
    Returns (is_reasonable, warning_msg).
    """
    if not until_dt:
        return (True, None)

    span = until_dt - now_utc
    days = span.days

    if days > _MIN_FUTURE_WARN:
        years = days / 365.25
        return (
            True,
            f"Chain extends {years:.1f} years into future (until {core.fmt_dt_local(until_dt)})",
        )

    return (True, None)


# ------------------------------------------------------------------------------
# Child build (copy almost everything; override minimal set)
# ------------------------------------------------------------------------------
_RESERVED_DROP = {
    "id",
    "uuid",
    "urgency",
    "status",
    "modified",
    "start",
    "end",
    "mask",
    "imask",
    "parent",
    "recur",
    "rc",
    "nextLink",  # set on parent, not copied
}

_RESERVED_OVERRIDE = {"due", "entry", "status", "chain", "prevLink", "link"}




# ------------------------------------------------------------------------------
# wait/scheduled carry-forward (relative to due)
# ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _nautical_local_tz():
    """Return ZoneInfo for configured local TZ (or None if unavailable)."""
    if ZoneInfo is None:
        return None
    try:
        name = getattr(core, "LOCAL_TZ_NAME", "") or "UTC"
        return ZoneInfo(name)
    except Exception:
        return None


def _utc_to_local_naive(dt_utc: datetime) -> datetime:
    """UTC -> local naive (wall-clock)."""
    if not isinstance(dt_utc, datetime):
        raise TypeError("dt_utc must be datetime")
    # Prefer core.to_local to honor any future core logic.
    try:
        dloc = core.to_local(dt_utc)
    except Exception:
        tz = _nautical_local_tz()
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dloc = dt_utc.astimezone(tz) if tz else dt_utc
    return dloc.replace(tzinfo=None)


def _local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    """Local naive (wall-clock) -> UTC, best-effort DST aware."""
    if not isinstance(dt_local_naive, datetime):
        raise TypeError("dt_local_naive must be datetime")
    tz = _nautical_local_tz()
    if tz:
        aware = dt_local_naive.replace(tzinfo=tz)
        return aware.astimezone(timezone.utc).replace(microsecond=0)
    return dt_local_naive.replace(tzinfo=timezone.utc).replace(microsecond=0)


def _carry_relative_datetime(parent: dict, child: dict, child_due_utc: datetime, field: str) -> None:
    """Carry wait/scheduled forward relative to due, preserving local wall-clock offset.

    offset := (parent[field] - parent[due]) computed in local wall-clock space
    child[field] := child_due + offset (also in local wall-clock space)
    """
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return
    if not (parent.get(field) and parent.get("due")):
        return

    # Do not carry absolute timestamps forward.
    child.pop(field, None)

    p_due = core.parse_dt_any(parent.get("due"))
    p_val = core.parse_dt_any(parent.get(field))
    if not (p_due and p_val and isinstance(child_due_utc, datetime)):
        return

    try:
        delta = _utc_to_local_naive(p_val) - _utc_to_local_naive(p_due)
        c_due_local = _utc_to_local_naive(child_due_utc)
        c_val_local = c_due_local + delta
        c_val_utc = _local_naive_to_utc(c_val_local)
        child[field] = core.fmt_isoz(c_val_utc)
    except Exception:
        # If anything goes wrong, do not mutate the child's field (leave inherited value).
        return

def _build_child_from_parent(
    parent: dict,
    child_due_utc,
    next_link_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt,
):
    child = {k: v for k, v in parent.items() if k not in _RESERVED_DROP}
    if _DEBUG_WAIT_SCHED:
        _LAST_WAIT_SCHED_DEBUG.clear()
    for k in _RESERVED_OVERRIDE:
        child.pop(k, None)
    child.update(
        {
            "status": "pending",
            "due": core.fmt_isoz(child_due_utc),
            "entry": core.fmt_isoz(core.now_utc()),
            "chain": "on",
            "prevLink": parent_short,
            "link": next_link_no,
        }
    )
    if kind == "anchor":
        child["anchor"] = parent.get("anchor")
        child["anchor_mode"] = parent.get("anchor_mode") or "skip"
        child.pop("cp", None)
    else:
        child["cp"] = parent.get("cp")
        child.pop("anchor", None)
        child.pop("anchor_mode", None)


    # Carry wait/scheduled forward relative to due (if present on parent)
    _carry_rel_dt_utc(parent, child, child_due_utc, "wait")
    _carry_rel_dt_utc(parent, child, child_due_utc, "scheduled")

    if cpmax:
        child["chainMax"] = int(cpmax)
    if until_dt:
        child["chainUntil"] = core.fmt_isoz(until_dt)

    # [CHAINID] Inherit parent chainID (fallback to parent's short uuid)
    try:
        if core.ENABLE_CHAIN_ID:
            parent_chain = (parent.get("chainID") or "").strip()
            if not parent_chain:
                parent_chain = core.short_uuid(parent.get("uuid"))
            child["chainID"] = parent_chain
    except Exception:
        pass

    return child

def _carry_rel_dt_utc(parent: dict, child: dict, child_due_utc: datetime, field: str) -> None:
    """Carry a datetime field forward relative to due using a UTC timedelta.

    offset := parent[field] - parent[due]
    child[field] := child_due + offset

    Notes:
      - This is intentionally UTC-timedelta based (seconds-accurate).
      - We always remove any inherited absolute value first to avoid 'sticky' wait/scheduled.
      - When NAUTICAL_DEBUG_WAIT_SCHED=1, we stash a short debug payload for the feedback panel.
    """
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return
    if not (parent.get("due") and parent.get(field)):
        return

    # Never carry absolute timestamps forward.
    child.pop(field, None)

    p_due = _dtparse(parent.get("due"))
    p_val = _dtparse(parent.get(field))
    if not (isinstance(p_due, datetime) and isinstance(p_val, datetime) and isinstance(child_due_utc, datetime)):
        if _DEBUG_WAIT_SCHED:
            _LAST_WAIT_SCHED_DEBUG[field] = {
                "ok": False,
                "reason": "parse-failed",
                "parent_due": parent.get("due"),
                "parent_val": parent.get(field),
                "child_due": child.get("due") or (core.fmt_isoz(child_due_utc) if isinstance(child_due_utc, datetime) else None),
            }
        return

    delta = (p_val - p_due)
    c_val = (child_due_utc + delta).replace(microsecond=0)
    child[field] = core.fmt_isoz(c_val)

    if _DEBUG_WAIT_SCHED:
        delta_s = _fmt_td_dd_hhmm(delta)

        _LAST_WAIT_SCHED_DEBUG[field] = {
            "ok": True,
            "parent_due": parent.get("due"),
            "parent_val": parent.get(field),
            "delta": delta_s,
            "child_due": child.get("due"),
            "child_val": child.get(field),
        }

# ------------------------------------------------------------------------------
# End-of-chain summary + stats
# ------------------------------------------------------------------------------
def _walk_chain_all(cur: dict, cap: int = None) -> list[dict]:
    cap = cap or _MAX_CHAIN_WALK
    seq = []
    node = cur
    steps = 0
    while node and steps < cap:
        seq.append(node)
        prev = node.get("prevLink")
        if not prev:
            break
        node = _export_uuid_short_cached(prev)
        if not node:
            break
        steps += 1
    return list(reversed(seq))


def _median(nums: list[float]) -> float | None:
    if not nums:
        return None
    s = sorted(nums)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _lateness_stats(chain: list[dict], tol_secs: int = 60) -> dict:
    early = on = late = 0
    deltas = []
    best = None
    worst = None
    for obj in chain:
        due = _dtparse(obj.get("due"))
        end = _dtparse(obj.get("end"))
        if not (due and end):
            continue
        diff = (end - due).total_seconds()
        deltas.append(diff)
        if diff > tol_secs:
            late += 1
            worst = diff if (worst is None or diff > worst) else worst
        elif diff < -tol_secs:
            early += 1
            best = diff if (best is None or diff < best) else best
        else:
            on += 1
    avg = (sum(deltas) / len(deltas)) if deltas else None
    med = _median(deltas) if deltas else None
    return {
        "early": early,
        "on_time": on,
        "late": late,
        "avg": avg,
        "median": med,
        "best_early": best,
        "worst_late": worst,
        "count": len(deltas),
    }


def _fmt_secs_delta(now_ref, secs: float | None) -> str:
    if secs is None:
        return "—"
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tgt = base + timedelta(seconds=secs)
    s = (
        core.humanize_delta(base, tgt, use_months_days=False)
        .replace("in ", "")
        .replace("overdue by ", "")
    )
    if secs > 0:
        return f"[yellow]+{s}[/]"
    if secs < 0:
        return f"[cyan]-{s}[/]"
    return "[green]±0[/]"


def _last_n_timeline(chain: list[dict], n: int = 6) -> list[str]:
    if not chain:
        return []
    
    # Get link number for sorting - handle tasks without link numbers
    def get_link(obj):
        link = obj.get("link")
        if link is None or link == "":
            return -1  # Put tasks without links at the beginning
        return core.coerce_int(link, 999999)
    
    # Sort by link number descending (most recent first)
    # But tasks without links (link=-1) should go at the end (oldest)
    chain_sorted = sorted(chain, key=get_link, reverse=True)
    
    # Filter out tasks without link numbers for display (they're usually root tasks)
    chain_with_links = [t for t in chain_sorted if get_link(t) > 0]
    
    # Determine max link number for formatting (only from tasks with links)
    if chain_with_links:
        max_link = max(get_link(obj) for obj in chain_with_links)
        label_width = len(str(max_link)) + 1  # +1 for the # symbol
    else:
        label_width = 4  # default width
    
    # If chain has more than 10 tasks, show top 3 (most recent) and bottom 3 (oldest)
    if len(chain_with_links) > 10:
        # Top 3: most recent tasks (highest link numbers)
        top_tasks = chain_with_links[:3]
        
        # Bottom 3: oldest tasks (lowest link numbers)
        bottom_tasks = chain_with_links[-3:]  # Already in descending order (e.g., [3, 2, 1])
        
        # Create lines for top tasks (most recent)
        top_lines = []
        for obj in top_tasks:
            no = get_link(obj)
            end = _dtparse(obj.get("end"))
            due = _dtparse(obj.get("due"))
            end_s = _fmtlocal(end) if end else "(no end)"
            delta = _fmt_on_time_delta(due, end)
            short = _short(obj.get("uuid"))
            lab = f"[bold]#{no:<{label_width}}[/]"
            line = f"{lab} {end_s} {delta} [dim]{short}[/]"
            # Highlight the most recent task
            if no == get_link(chain_with_links[0]):
                line = f"[green]{line}[/]"
            top_lines.append(line)
        
        # Add ellipsis
        ellipsis_line = f"[dim]{' ' * label_width}... ({len(chain_with_links) - 6} more tasks) ...[/dim]"
        
        # Create lines for bottom tasks (oldest) - also in descending order
        bottom_lines = []
        for obj in bottom_tasks:  # Already in descending order (e.g., 3, 2, 1)
            no = get_link(obj)
            end = _dtparse(obj.get("end"))
            due = _dtparse(obj.get("due"))
            end_s = _fmtlocal(end) if end else "(no end)"
            delta = _fmt_on_time_delta(due, end)
            short = _short(obj.get("uuid"))
            lab = f"[bold]#{no:<{label_width}}[/]"
            line = f"{lab} {end_s} {delta} [dim]{short}[/]"
            bottom_lines.append(line)
        
        return top_lines + [ellipsis_line] + bottom_lines
    
    # For chains with <= 10 tasks, show all in reverse order (most recent at top)
    lines = []
    for obj in chain_with_links[:n]:
        no = get_link(obj)
        end = _dtparse(obj.get("end"))
        due = _dtparse(obj.get("due"))
        end_s = _fmtlocal(end) if end else "(no end)"
        delta = _fmt_on_time_delta(due, end)
        short = _short(obj.get("uuid"))
        lab = f"[bold]#{no:<{label_width}}[/]"
        line = f"{lab} {end_s} {delta} [dim]{short}[/]"
        # Highlight the most recent task
        if no == get_link(chain_with_links[0]):
            line = f"[green]{line}[/]"
        lines.append(line)
    
    return lines

def _create_timeline_segment(tasks: list[dict], last_link_num) -> list[str]:
    """Helper to create timeline lines for a segment of tasks."""
    if not tasks:
        return []
    
    base_no = core.coerce_int(last_link_num, len(tasks))
    labelw = max(4, len(f"#{base_no}"))
    lines = []
    start_no = base_no - (len(tasks) - 1)
    
    for i, obj in enumerate(tasks):
        no = start_no + i
        end = _dtparse(obj.get("end"))
        due = _dtparse(obj.get("due"))
        end_s = _fmtlocal(end) if end else "(no end)"
        delta = _fmt_on_time_delta(due, end)
        short = _short(obj.get("uuid"))
        lab = f"[bold]#{no:<{labelw-1}}[/]"
        line = f"{lab} {end_s} {delta} [dim]{short}[/]"
        if i == len(tasks) - 1:
            line = f"[green]{line}[/]"
        lines.append(line)
    
    return lines


@lru_cache(maxsize=256)
def _chain_length_cached(u_short: str):
    """
    Estimate chain length by walking backwards with a limit.
    Useful for display without full chain traversal.
    """
    obj = _export_uuid_short_cached(u_short)
    if not obj:
        return 1

    count = 1
    prev = obj.get("prevLink")
    steps = 0

    while prev and steps < _MAX_CHAIN_WALK:
        prev_obj = _export_uuid_short_cached(prev)
        if not prev_obj:
            break
        count += 1
        prev = prev_obj.get("prevLink")
        steps += 1

    return count


def _end_chain_summary(current: dict, reason: str, now_utc, current_task: dict = None) -> None:
    # Use the passed current_task if provided, otherwise use current
    actual_current = current_task if current_task else current
    
    kind_anchor = bool((actual_current.get("anchor") or "").strip())
    kind = "anchor" if kind_anchor else "cp"
    
    # Prefer chainID export; fallback to legacy prev/next traversal
    chain = tw_export_chain_or_fallback(actual_current)
    
    # Replace the last task in chain with the actual_current if it's more up-to-date
    if actual_current and chain:
        last_idx = -1
        for i, task in enumerate(chain):
            if task.get("uuid") == actual_current.get("uuid"):
                last_idx = i
                break
        if last_idx >= 0:
            chain[last_idx] = actual_current

    # Sort chronologically by link number (falls back to due date when link missing)
    def _link_sort_key(obj):
        ln = core.coerce_int(obj.get("link"), None)
        if ln is not None:
            return (0, ln)
        due = _dtparse(obj.get("due")) or datetime.max.replace(tzinfo=timezone.utc)
        return (1, due)

    try:
        chain.sort(key=_link_sort_key)
    except Exception:
        pass

    L = core.coerce_int(current.get("link"), len(chain))
    root = _short(chain[0].get("uuid") if chain else _root_uuid_from(current))

    cur_s = _short(current.get("uuid"))
    first = _dtparse(chain[0].get("due")) if chain else None
    last = _dtparse(chain[-1].get("end")) if chain else None
    span = "–"
    if first and last:
        span = (
            _human_delta(first, last, prefer_months=True)
            .replace("in ", "")
            .replace("overdue by ", "")
        )

    rows = []
    rows.append(("Reason", reason))

    # Get root and age
    root_short, age_str = _chain_root_and_age(current, now_utc)
    
    # Show chain info with root and age
    chain_display = f"{root_short}"
    if age_str != "—":
        chain_display += f" ({age_str})"
    
    rows.append(("Root", chain_display))

    # Show if chain was truncated
    chain_display = f"{root} … {cur_s}  [dim](#{L}, {len(chain)} tasks"
    if len(chain) >= _MAX_CHAIN_WALK:
        chain_display += f", truncated at {_MAX_CHAIN_WALK})"
    else:
        chain_display += ")"
    rows.append(("Chain", chain_display))

    if kind == "anchor":
        expr = (current.get("anchor") or "").strip()
        mode = (current.get("anchor_mode") or "skip").lower()
        expr_key = core.cache_key_for_task(expr, mode)
        tag = {
            "skip": "[cyan]SKIP[/]",
            "all": "[yellow]ALL[/]",
            "flex": "[magenta]FLEX[/]",
        }.get(mode, "[cyan]SKIP[/]")
        rows.append(("Pattern", f"{expr}  {tag}"))
        try:
            dnf = _validate_anchor_expr_cached(expr)
            rows.append(("Natural", core.describe_anchor_dnf(dnf, current)))
        except Exception:
            pass
    else:
        rows.append(("Period", current.get("cp") or "–"))

    if first:
        rows.append(("First due", core.fmt_dt_local(first)))
    if last:
        rows.append(("Last end", core.fmt_dt_local(last)))
    rows.append(("Span", span))

    stats = _lateness_stats(chain)
    rows.append(
        (
            "Performance",
            f"early {stats['early']}, on-time {stats['on_time']}, late {stats['late']}",
        )
    )
    rows.append(("Avg lateness", _fmt_secs_delta(now_utc, stats["avg"])))
    rows.append(("Median lateness", _fmt_secs_delta(now_utc, stats["median"])))
    rows.append(("Best early", _fmt_secs_delta(now_utc, stats["best_early"])))
    rows.append(("Worst late", _fmt_secs_delta(now_utc, stats["worst_late"])))

    cpmax = core.coerce_int(current.get("chainMax"), 0)
    until = _dtparse(current.get("chainUntil"))
    lims = []
    if cpmax:
        lims.append(f"max {cpmax}")
    if until:
        lims.append(f"until {core.fmt_dt_local(until)}")
    rows.append(("Limits", " | ".join(lims) if lims else "–"))

    tail = _last_n_timeline(chain, n=6)
    if tail:
        rows.append(("History", "\n".join(tail)))

    rows = _format_chain_summary_rows(rows)
    _panel("⛔ Chain finished – summary", rows, kind="summary")



# ------------------------------------------------------------------------------
# Timeline (capped) — no dependency on core.next_anchor_after
# ------------------------------------------------------------------------------

def _timeline_lines(
    kind: str,
    task: dict,
    child_due_utc,
    child_short: str,
    dnf,
    next_count: int = 3,
    cap_no: int | None = None,
    cur_no: int | None = None,
    show_gaps: bool = True,
    round_anchor_gaps: bool = True,  # Round anchor gaps to nearest day
) -> list[str]:
    """
    Compact timeline with inline gaps.
    """
    cur_no = core.coerce_int(task.get("link") if cur_no is None else cur_no, 1)
    nxt_no = cur_no + 1
    allowed_future = (
        next_count if cap_no is None else max(0, min(next_count, cap_no - nxt_no))
    )
    
    if kind == "cp":
        prev_style = "dim green"        
        cur_style = "spring_green1"
        next_style = "bold yellow"
    else:
        prev_style = "sky_blue3"
        cur_style = "spring_green1"
        next_style = "bold yellow"
    
    future_style = _future_style_for_chain(task, kind)
    
    # Collect items and their dates
    items = []
    
    # Previous tasks
    prevs = _collect_prev_two(task)
    for obj in prevs:
        no = core.coerce_int(obj.get("link"), None) or (cur_no - (len(prevs) - prevs.index(obj)))
        end_dt = _dtparse(obj.get("end"))
        due_dt = _dtparse(obj.get("due"))
        delta = _fmt_on_time_delta(due_dt, end_dt)
        end_s = _fmtlocal(end_dt) if end_dt else "(no end)"
        short = _short(obj.get("uuid"))
        items.append((no, end_dt, obj, "prev"))
    
    # Current task
    cur_end = _dtparse(task.get("end"))
    items.append((cur_no, cur_end, task, "current"))
    
    # Next task
    items.append((nxt_no, child_due_utc, {"uuid": child_short}, "next"))
    
    # Future tasks
    fut_dt = child_due_utc
    fut_no = nxt_no
    iterations = 0
    
    if allowed_future > 0:
        if kind == "cp":
            td = core.parse_cp_duration(task.get("cp") or "")
            if td:
                secs = int(td.total_seconds())
                for _ in range(allowed_future):
                    if iterations >= _MAX_ITERATIONS:
                        break
                    iterations += 1
                    
                    fut_no += 1
                    if secs % 86400 == 0:
                        dl = _tolocal(fut_dt)
                        fut_dt = core.build_local_datetime(
                            (dl + timedelta(days=int(secs // 86400))).date(),
                            (dl.hour, dl.minute),
                        ).astimezone(timezone.utc)
                    else:
                        fut_dt = fut_dt + td
                    
                    if cap_no is not None and fut_no > cap_no:
                        break
                    
                    items.append((fut_no, fut_dt, {"is_future": True}, "future"))
        else:
            # Anchor patterns are date-based in the core; multi-time @t=HH:MM,HH:MM,...
            # expansion is performed here at the hook level.
            seed_base = (task.get("chainID") or _root_uuid_from(task) or "preview")

            nxt_local = _to_local_cached(child_due_utc)
            fallback_hhmm = (nxt_local.hour, nxt_local.minute)

            due0, _ = _safe_parse_datetime(task.get("due"))
            default_seed = _to_local_cached(due0 or child_due_utc).date()

            after_local = nxt_local
            for _ in range(allowed_future):
                if iterations >= _MAX_ITERATIONS:
                    break
                iterations += 1

                fut_no += 1
                try:
                    next_local = _next_occurrence_after_local_dt(
                        dnf,
                        after_local,
                        default_seed_date=default_seed,
                        seed_base=seed_base,
                        fallback_hhmm=fallback_hhmm,
                    )
                except Exception:
                    break

                if not next_local:
                    break

                fut_dt = next_local.astimezone(timezone.utc)
                after_local = next_local

                if cap_no is not None and fut_no > cap_no:
                    break

                items.append((fut_no, fut_dt, {"is_future": True}, "future"))
    
    # Build lines with inline gaps
    lines = []
    for i, (no, dt, obj, item_type) in enumerate(items):
        # Build the base line
        if item_type == "prev":
            end_dt = _dtparse(obj.get("end"))
            due_dt = _dtparse(obj.get("due"))
            delta = _fmt_on_time_delta(due_dt, end_dt)
            end_s = _fmtlocal(end_dt) if end_dt else "(no end)"
            short = _short(obj.get("uuid"))
            base_line = f"[{prev_style}]# {no:>2} {end_s} {delta} {short}[/]"
            
        elif item_type == "current":
            cur_end = _dtparse(task.get("end"))
            cur_due = _dtparse(task.get("due"))
            cur_delta = _fmt_on_time_delta(cur_due, cur_end)
            cur_end_s = _fmtlocal(cur_end) if cur_end else "(no end)"
            base_line = f"[{cur_style}]# {no:>2} {cur_end_s} {cur_delta} {_short(task.get('uuid'))}[/]"
            
        elif item_type == "next":
            is_last = cap_no is not None and no == cap_no
            next_text = f"# {no:>2} → {core.fmt_dt_local(dt)} {_short(obj.get('uuid'))}"
            if is_last:
                base_line = f"[{next_style}]{next_text} [bold red](last link)[/][/]"
            else:
                base_line = f"[{next_style}]{next_text}[/]"
                
        elif item_type == "future":
            is_last = cap_no is not None and no == cap_no
            future_text = f"# {no:>2} {core.fmt_dt_local(dt)}"
            if is_last:
                base_line = f"[{future_style}]{future_text} [bold red](last link)[/][/]"
            else:
                base_line = f"[{future_style}]{future_text}[/]"
        
        # Add gap annotation if applicable
        full_line = base_line
        if show_gaps and i < len(items) - 1:
            next_item = items[i + 1]
            next_dt = next_item[1]
            
            if dt and next_dt:
                gap_text = _format_gap(dt, next_dt, kind, round_anchor_gaps)
                if gap_text:
                    full_line = f"{base_line}{gap_text}"
        
        lines.append(full_line)
    
    return lines

def _got_anchor_invalid(msg: str) -> None:
    _panel("❌ Invalid anchor", [("Validation", msg)], kind="error")
    print(json.dumps({"error": "Invalid anchor", "message": msg}, ensure_ascii=False))
    sys.exit(1)


# chainUntil -> numeric cap and "Final (until)"
def _cap_from_until_cp(task, next_due_utc):
    until = _dtparse(task.get("chainUntil"))
    if not until:
        return (None, None)
    td = core.parse_cp_duration(task.get("cp") or "")
    if not td:
        return (None, None)
    secs = int(td.total_seconds())
    cur = core.coerce_int(task.get("link"), 1)
    nno = cur + 1
    ndt = next_due_utc
    last_no = None
    last_dt = None
    iterations = 0

    while ndt and ndt <= until and iterations < _MAX_ITERATIONS:
        iterations += 1
        last_no, last_dt = nno, ndt
        if secs % 86400 == 0:
            dl = _tolocal(ndt)
            ndt = core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
            ).astimezone(timezone.utc)
        else:
            ndt = ndt + td
        nno += 1

    return (last_no, last_dt)


def _cap_from_until_anchor(task, next_due_utc, dnf):
    """
    Return (final_no, final_dt) for anchors limited by chainUntil.
    WITH iteration guard to prevent infinite loops.
    """
    until_utc = _dtparse(task.get("chainUntil"))
    if not until_utc:
        return (None, None)

    cur_no = core.coerce_int(task.get("link"), 1)
    seed_base = task.get("chainID") or _root_uuid_from(task) or "preview"

    nxt_local = _to_local_cached(next_due_utc)
    until_local = _to_local_cached(until_utc)
    fallback_hhmm = (nxt_local.hour, nxt_local.minute)
    due0, _ = _safe_parse_datetime(task.get("due"))
    default_seed = _to_local_cached(due0 or next_due_utc).date()

    count = 0
    last_hit = None
    cursor = nxt_local
    iterations = 0

    # Count occurrences starting with the already-computed next due.
    while iterations < _MAX_ITERATIONS and cursor <= until_local:
        iterations += 1
        count += 1
        last_hit = cursor
        cursor = _next_occurrence_after_local_dt(
            dnf,
            cursor,
            default_seed=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
        )
        if cursor is None:
            break

    if count == 0 or last_hit is None:
        return (None, None)

    final_no = cur_no + count
    final_dt = last_hit.astimezone(timezone.utc)
    return (final_no, final_dt)

def _ensure_acf(task: dict) -> None:
    anch = (task.get("anchor") or "").strip()
    try:
        task["acf"] = core.build_acf(anch) if anch else ""
    except Exception:
        task["acf"] = ""

def _safe_dt(v):
    try:
        return _dtparse(v) if isinstance(v, str) else v
    except Exception:
        return None

def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None, env=None) -> list[dict]:
    if not chain_id:
        return []
    args = ["task", "rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
    if since:
        args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
    if extra:
        args += shlex.split(extra)
    args.append("export")
    try:
        out = subprocess.check_output(args, text=True, env=env)
        data = json.loads(out.strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception:
        return []

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    old, new = _read_two()

    # Skip all Nautical logic when task is being deleted
    if (new.get("status") or "").lower() == "deleted":
        print(json.dumps(new, ensure_ascii=False))
        return


    # --- pre-flight: validate on simple modify (not completion) ---
    if (old.get("status") == new.get("status")) or (new.get("status") != "completed"):
        # This is a non-completion modification
        anchor_raw = (new.get("anchor") or "").strip()
        new_anchor = _strip_quotes(anchor_raw) 
        
        if new_anchor:
            # Anchor validation on modification (catch errors early)
            try:
                # Lint only for non-blocking hints
                _, warns = core.lint_anchor_expr(new_anchor)
                if warns:
                    _panel("ℹ️  Lint", [("Hint", w) for w in warns], kind="note")

                anchor_mode = ((new.get("anchor_mode") or old.get("anchor_mode") or "").strip().upper() or "ALL")
                due_dt = _safe_dt(new.get("due") or old.get("due"))

                if core.ENABLE_ANCHOR_CACHE:
                    # precompute; if core trips over timedelta formatting, fall back
                    _ = core.build_and_cache_hints(new_anchor, anchor_mode, default_due_dt=due_dt)
                else:
                    _ = core.validate_anchor_expr_strict(new_anchor)
            except TypeError:
                _ = core.validate_anchor_expr_strict(new_anchor)
            except Exception as e:
                emsg = str(e)
                has_type_colon = bool(
                    re.search(r"(?:^|[^A-Za-z])(w|m|y)(?:/\d+)?:", new_anchor, re.IGNORECASE)
                )
                if not has_type_colon:
                    # Common pitfall: user typed a weekday pattern without the leading `w:`
                    if re.match(r"^(mon|tue|wed|thu|fri|sat|sun)\b", new_anchor, re.IGNORECASE):
                        emsg = (
                            "Weekly anchors must start with 'w:'. "
                            "Examples: 'w:mon..fri' or 'w:mon,tue,wed,thu,fri'."
                        )
                    else:
                        emsg = (
                            "Anchors must start with 'w:', 'm:' or 'y:'. "
                            "Examples: 'w:mon', 'm:-1', 'y:06-01'."
                        )
                _got_anchor_invalid(emsg)

            # Deep checks only if anchor field changed
            if _field_changed(old, new, "anchor") or _field_changed(old, new, "anchor_mode"):
                _validate_anchor_on_modify(new_anchor)
                # _ensure_acf(new)  # keep in-memory ACF consistent (no UDA writes)

        # CP validation only happens on completion, NOT on modification
        # because taskwarrior already validates the duration format
        
        # [CHAINID] stamp only when task just became nautical and has no chainID/links
        try:
            became_anchor = (not (old.get("anchor") or "").strip()) and ((new.get("anchor") or "").strip())
            became_cp     = (not (old.get("cp")     or "").strip()) and ((new.get("cp")     or "").strip())
            already_chain = bool((new.get("chainID") or "").strip())
            linked_already = bool((new.get("prevLink") or new.get("nextLink") or "").strip())
            if core.ENABLE_CHAIN_ID and (became_anchor or became_cp) and not already_chain and not linked_already:
                new["chainID"] = core.short_uuid(new.get("uuid"))
        except Exception:
            pass

        _print_task(new)
        return

    # If we reach here, the task is being completed
    # Now we should validate CP (in addition to anchor which was already validated on modify)
    cp_raw = (new.get("cp") or "").strip()
    new_cp = _strip_quotes(cp_raw)

    if new_cp:
        # Validate CP on completion
        try:
            td = core.parse_cp_duration(new_cp)
            if td is None:
                raise ValueError(f"Invalid duration format '{new_cp}'")
        except ValueError as e:
            _panel("❌ Invalid CP", [("Validation", str(e))], kind="error")
            print(json.dumps({"error": "Invalid CP", "message": str(e)}, ensure_ascii=False))
            sys.exit(1)
        except Exception as e:
            _panel("❌ CP Error", [("Validation", f"Unexpected error: {str(e)}")], kind="error")
            print(json.dumps({"error": "CP parsing error", "message": str(e)}, ensure_ascii=False))
            sys.exit(1)

            # Deep checks only if fields changed
            if _field_changed(old, new, "anchor") or _field_changed(old, new, "anchor_mode"):
                if new_anchor:
                    _validate_anchor_on_modify(new_anchor)
                # _ensure_acf(new)  # keep in-memory ACF consistent (no UDA writes)

            if (_field_changed(old, new, "cp")
                or _field_changed(old, new, "chainMax")
                or _field_changed(old, new, "chainUntil")) and new_cp:
                _validate_cp_on_modify(new_cp, new.get("chainMax"), new.get("chainUntil"))

            # [CHAINID] stamp only when task just became nautical and has no chainID/links
            try:
                became_anchor = (not (old.get("anchor") or "").strip()) and ((new.get("anchor") or "").strip())
                became_cp     = (not (old.get("cp")     or "").strip()) and ((new.get("cp")     or "").strip())
                already_chain = bool((new.get("chainID") or "").strip())
                linked_already = bool((new.get("prevLink") or new.get("nextLink") or "").strip())
                if core.ENABLE_CHAIN_ID and (became_anchor or became_cp) and not already_chain and not linked_already:
                    new["chainID"] = core.short_uuid(new.get("uuid"))
            except Exception:
                pass

            _print_task(new)
            return



    now_utc = core.now_utc()
    parent_short = _short(new.get("uuid"))
    base_no = core.coerce_int(new.get("link"), 1)
    next_no = base_no + 1

    # Determine whether chaining is effectively enabled
    raw_ch = (new.get("chain") or "").strip().lower()
    has_anchor = bool((new.get("anchor") or "").strip())
    has_cp = bool((new.get("cp") or "").strip())

    # Back-compat default: if chain is unset AND it's a chainable task, treat as on.
    effective_on = (raw_ch == "on") or (raw_ch == "" and (has_anchor or has_cp))

    if not effective_on:
        if has_anchor or has_cp:
            _panel(
                "Chain disabled (chain:off) — no next link will be spawned.",
                [],
                kind="disabled",
            )
            _print_task(new)
            _end_chain_summary(new, "Manual stop.", now_utc)
        else:
            # Not a chain task at all → just pass it through quietly.
            _print_task(new)
        return


    has_anchor = bool((new.get("anchor") or "").strip())
    has_cp = bool((new.get("cp") or "").strip())
    kind = "anchor" if has_anchor else ("cp" if has_cp else None)
    if not kind:
        _print_task(new)
        return

    # Compute next due
    try:
        if kind == "anchor":
            child_due, meta, dnf = _compute_anchor_child_due(new)
        else:
            child_due, meta = _compute_cp_child_due(new)
            dnf = None
    except ValueError as e:
        _panel(
            "⛔ Chain error",
            [("Reason", f"Invalid task field: {str(e)}")],
            kind="error",
        )

        _print_task(new)
        return
    except Exception as e:
        _panel(
            "⛔ Chain error",
            [("Reason", f"Could not compute next due: {str(e)}")],
            kind="error",
        )
        _print_task(new)
        return

    # Parse chainUntil once (you already do this later; do it early too or reuse)
    until_dt, err = _safe_parse_datetime(new.get("chainUntil"))
    if err:
        _panel(
            "⛔ Chain error", [("Reason", f"Invalid chainUntil: {err}")], style="red"
        )
        _print_task(new)
        return

    # GUARD: chainUntil must be in future
    if until_dt:
        is_valid, err_msg = _validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            _panel(
                "⛔ Chain error",
                [("Reason", f"Invalid chainUntil: {err_msg}")],
                kind="error",
            )
            _print_task(new)
            return

    # GUARD: if the computed next due would exceed chainUntil, stop here.
    if until_dt and child_due > until_dt:
        _end_chain_summary(new, "Reached 'until' limit", now_utc)
        new["chain"] = "off"
        _print_task(new)
        return

    if not child_due:
        _panel(
            "⛔ Chain error",
            [("Reason", "Could not compute next due (no end date on parent)")],
            kind="error",
        )
        _print_task(new)
        return

    # GUARD: Warn if chain extends unreasonably far
    if until_dt:
        is_reasonable, warn_msg = _validate_chain_duration_reasonable(
            child_due, until_dt, now_utc
        )
        if warn_msg and not is_reasonable:
            _panel("⚠ Chain duration warning", [("Warning", warn_msg)], kind="warning")


    # Effective cap (max/until) -> numeric cap_no and finals for panel
    cpmax = core.coerce_int(new.get("chainMax"), 0)
    until_dt = _dtparse(new.get("chainUntil"))
    cap_no = cpmax if cpmax else None
    finals = []

    # Final (max) for cp
    if kind == "cp" and cpmax:
        try:
            fmax = _estimate_cp_final_by_max(new, child_due)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass
    # Final (max) for anchor
    if kind == "anchor" and cpmax:
        try:
            fmax = _estimate_anchor_final_by_max(new, child_due, dnf)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass

    if until_dt:
        if kind == "cp":
            u_no, u_dt = _cap_from_until_cp(new, child_due)
        else:
            u_no, u_dt = _cap_from_until_anchor(new, child_due, dnf)
        if u_no:
            cap_no = min(cap_no, u_no) if cap_no else u_no
        if u_dt:
            finals.append(("until", u_dt))

    # Stop if next would exceed cap
    if cap_no and next_no > cap_no:
        _end_chain_summary(new, f"Reached cap #{cap_no}", now_utc, current_task=new)
        new["chain"] = "off"
        _print_task(new)
        return

    # Build child payload & spawn
    try:
        child = _build_child_from_parent(
            new, child_due, next_no, parent_short, kind, cpmax, until_dt
        )
    except Exception as e:
        _panel(
            "⛓ Chain error",
            [("Reason", f"Failed to build next link ({e})")],
            kind="error",
        )
        _print_task(new)
        return

    deferred_spawn = False
    spawn_note = None
    try:
        child_short, stripped_attrs = _spawn_child_atomic(child, new)
    except _SpawnDeferred as sd:
        # Parent update (nextLink) is still applied by the outer TW command.
        # We keep the chosen UUID and defer the child import until after the lock is released.
        child_short = sd.child_short
        stripped_attrs = sd.stripped_attrs
        deferred_spawn = True
        spawn_note = "[yellow]Queued[/] (Taskwarrior lock active; will import after this command finishes)"
        _kick_deferred_spawn_drain_async()
    except Exception as e:
        _panel(
            "⛓ Chain error",
            [("Reason", f"Failed to spawn next link ({e})")],
            kind="error",
        )
        _print_task(new)
        return

    # Reflect link on parent for nice UX (even if the child is queued)
    new["nextLink"] = child_short

    # Feedback panel
    fb = []
    if spawn_note:
        fb.append(("Spawn", spawn_note))
    if kind == "anchor":
        anchor_raw = (new.get("anchor") or "").strip()
        expr_str = _strip_quotes(anchor_raw) 
        mode_tag = {
            "skip": "[cyan]SKIP[/]",
            "all": "[yellow]ALL[/]",
            "flex": "[magenta]FLEX[/]",
        }.get((new.get("anchor_mode") or "skip").lower(), "[cyan]SKIP[/]")
        fb.append(("Pattern", f"{expr_str}  {mode_tag}"))
        fb.append(("Natural", core.describe_anchor_dnf(dnf, new)))
        fb.append(("Basis", _pretty_basis_anchor(meta, new)))
        fb.append(("Root", _format_root_and_age(new, now_utc)))

        if _DEBUG_WAIT_SCHED and _LAST_WAIT_SCHED_DEBUG:
            for _fld in ("scheduled", "wait"):
                d = _LAST_WAIT_SCHED_DEBUG.get(_fld)
                if not d:
                    continue
                if d.get("ok"):
                    fb.append((
                        f"{_fld} carry",
                        f"Δ {d.get('delta')}  parent {d.get('parent_val')} vs {d.get('parent_due')}  →  child {d.get('child_val')}"
                    ))
                else:
                    fb.append((
                        f"{_fld} carry",
                        f"[yellow]skip[/] ({d.get('reason')})  parent {d.get('parent_val')} vs {d.get('parent_due')}"
                    ))
        if stripped_attrs:
            fb.append(
                (
                    "Sanitised",
                    f"Removed unknown fields: {', '.join(sorted(stripped_attrs))}",
                )
            )

        delta = core.humanize_delta(
            now_utc, child_due, use_months_days=core.expr_has_m_or_y(dnf)
        )
        fb.append(("Next Due", f"{core.fmt_dt_local(child_due)}  ({delta})"))
        _append_next_wait_sched_rows(fb, child, child_due)

        if cap_no:
            if base_no >= cap_no:
                fb.append(("Link status", "[bold red]This was the last link[/]"))
            elif base_no == cap_no - 1:
                fb.append(
                    ("Link status", "[yellow]This was the second-to-last link[/]")
                )
            fb.append(
                ("Links left", f"{max(0, cap_no - base_no)} left (cap #{cap_no})")
            )

        for label, when in finals:
            fb.append(
                (
                    f"Final ({label})",
                    f"{core.fmt_dt_local(when)}  ({_human_delta(now_utc, when, True)})",
                )
            )


        child_id = ""
        if not deferred_spawn:
            child_obj = _export_uuid_short_cached(child_short)
            child_id = child_obj.get("id", "") if child_obj else ""

        if deferred_spawn:
            title = f"⚓︎ Next anchor  #{next_no}  {parent_short} → {child_short} [queued]"
        else:
            title = f"⚓︎ Next anchor  #{next_no}  {parent_short} → {child_short} [{child_id}]"
        tl = _timeline_lines(
            "anchor",
            new,
            child_due,
            child_short,
            dnf,
            next_count=3,
            cap_no=cap_no,
            cur_no=base_no,
            show_gaps=_SHOW_TIMELINE_GAPS,
            round_anchor_gaps=True,  # Round to nearest day

        )
        if tl:
            fb.append(("Timeline", "\n".join(tl)))
        if "rand" in expr_str.lower():
            fb.append(
                (
                    "Rand",
                    f"[dim]Deterministic picks seeded by root {_short(_root_uuid_from(new))}[/]",
                )
            )

        fb = _format_next_anchor_rows(fb)

        if _CHAIN_COLOR_PER_CHAIN:
            chain_colour = _chain_colour_for_task(new, "anchor")
            _panel(
                title,
                fb,
                kind="preview_anchor",
                border_style=chain_colour,
                title_style=chain_colour,
            )
        else:
            # Fast path: use the static theme colours
            _panel(title, fb, kind="preview_anchor")


    else:
        delta = core.humanize_delta(now_utc, child_due, use_months_days=False)
        fb.append(("Period", new.get("cp")))
        fb.append(("Basis", _pretty_basis_cp(new, meta)))
        fb.append(("Root", _format_root_and_age(new, now_utc)))
        fb.append(("Next Due", f"{core.fmt_dt_local(child_due)}  ({delta})"))
        _append_next_wait_sched_rows(fb, child, child_due)

        if cap_no:
            if base_no >= cap_no:
                fb.append(("Link status", "[bold red]This was the last link[/]"))
            elif base_no == cap_no - 1:
                fb.append(
                    ("Link status", "[yellow]Next link is the last in the chain.[/]")
                )
            fb.append(
                ("Links left", f"{max(0, cap_no - base_no)} left (cap #{cap_no})")
            )
        else:
            fb.append(("Limits", "—"))

        for label, when in finals:
            fb.append(
                (
                    f"Final ({label})",
                    f"{core.fmt_dt_local(when)}  ({_human_delta(now_utc, when, True)})",
                )
            )

        child_id = ""
        if not deferred_spawn:
            child_obj = _export_uuid_short_cached(child_short)
            child_id = child_obj.get("id", "") if child_obj else ""

        if deferred_spawn:
            title = f"⛓ Next link  #{next_no}  {parent_short} → {child_short} [queued]"
        else:
            title = f"⛓ Next link  #{next_no}  {parent_short} → {child_short} [{child_id}]"
        tl = _timeline_lines(
            "cp",
            new,
            child_due,
            child_short,
            None,
            next_count=3,
            cap_no=cap_no,
            cur_no=base_no,
            show_gaps=_SHOW_TIMELINE_GAPS,
            round_anchor_gaps=False,  # CP gaps are exact

        )
        if tl:
            fb.append(("Timeline", "\n".join(tl)))

        fb = _format_next_cp_rows(fb)

        if _CHAIN_COLOR_PER_CHAIN:
            chain_colour = _chain_colour_for_task(new, "cp")
            _panel(
                title,
                fb,
                kind="preview_cp",
                border_style=chain_colour,
                title_style=chain_colour,
            )
        else:
            _panel(title, fb, kind="preview_cp")




    _print_task(new)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if "--drain-spawn-queue" in sys.argv[1:]:
        raise SystemExit(_drain_deferred_spawn_queue())
    main()
